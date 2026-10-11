"""Disk-backed normalization of pinned Neelagiri rating histories."""

import gzip
import hashlib
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .dataset_metadata import MetadataResult

SCHEMA = "arb-interactions-v1"
HEADER = b"dataset_user_key\tmal_id\trating\twatch_status\n"
STATUSES = ("watching", "completed", "on_hold", "dropped", "plan_to_watch", "unknown")
NAMESPACE = "neelagiri-2025-v1"
CSV_COLUMNS = ("username", "anime_id", "status", "score", "is_rewatching", "num_watched_episodes")


@dataclass(frozen=True)
class InteractionResult:
    qa: dict
    rows: int
    uncompressed_bytes: int
    canonical_sha256: str
    compressed_bytes: int
    compressed_sha256: str
    rated_ids: frozenset[int]


def user_key(username: str, namespace: str = NAMESPACE) -> str:
    return "u1_" + hashlib.sha256(
        b"arb-user-v1\0" + namespace.encode("utf-8") + b"\0" + username.encode("utf-8")
    ).hexdigest()


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _scalar(db, sql: str) -> int:
    return int(db.execute(sql).fetchone()[0] or 0)


def _hist(db, sql: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in db.execute(sql).fetchall()}


def _bin(value: int) -> str:
    return str(min((value // 5) * 5, 100))


def _check_csv_header(path: Path) -> None:
    expected = b",".join(column.encode("ascii") for column in CSV_COLUMNS)
    try:
        with Path(path).open("rb") as stream:
            header = stream.readline(len(expected) + 3)
    except OSError:
        raise ValueError("ratings_csv_invalid") from None
    if header not in (expected + b"\n", expected + b"\r\n"):
        raise ValueError("ratings_csv_invalid")


def build_interactions(ratings_path: Path, metadata: MetadataResult, output_path: Path,
                       work_dir: Path, limits: dict, namespace: str = NAMESPACE) -> InteractionResult:
    """Group all score states without loading rating rows into Python memory."""
    if namespace != NAMESPACE:
        raise ValueError("dataset_integrity_error")
    _check_csv_header(ratings_path)
    try:
        import duckdb
    except ImportError as exc:
        raise ValueError("resource_limit") from exc
    previous_umask = os.umask(0o077)
    work_dir = Path(work_dir)
    db = None
    try:
        work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(work_dir, 0o700)
        db = duckdb.connect(str(work_dir / "ratings.duckdb"))
        db.execute(f"SET memory_limit='{int(limits['memory_limit_mib'])}MB'")
        db.execute(f"SET threads={int(limits['threads'])}")
        db.execute(f"SET temp_directory={_sql_string(str(work_dir / 'spill'))}")
        db.execute(f"SET max_temp_directory_size='{int(limits['spill_limit_mib'])}MB'")
        db.execute("SET preserve_insertion_order=false")
        db.execute("PRAGMA disable_progress_bar")
        columns = ",".join(f"'{field}':'VARCHAR'" for field in CSV_COLUMNS)
        csv_sql = ("read_csv(?, auto_detect=false, header=true, delim=',', quote='\"', "
                   "escape='\"', columns={" + columns + "}, strict_mode=true, "
                   "ignore_errors=false, null_padding=false, encoding='utf-8')")
        db.execute(f"""
            CREATE TABLE raw AS
            WITH parsed AS (
              SELECT username, anime_id, status,
                TRY_CAST(TRIM(score) AS DOUBLE) AS numeric_score,
                score
              FROM {csv_sql}
            )
            SELECT username,
              CASE WHEN regexp_full_match(TRIM(anime_id), '[0-9]+')
                   THEN TRY_CAST(TRIM(anime_id) AS BIGINT) END AS mal_id,
              CASE lower(replace(replace(trim(coalesce(status,'')), ' ', '_'), '-', '_'))
                WHEN 'watching' THEN 'watching'
                WHEN 'completed' THEN 'completed'
                WHEN 'on_hold' THEN 'on_hold'
                WHEN 'dropped' THEN 'dropped'
                WHEN 'plan_to_watch' THEN 'plan_to_watch'
                ELSE 'unknown' END AS watch_status,
              CASE WHEN score IS NULL OR TRIM(score)='' OR numeric_score=0 THEN 'unset'
                   WHEN regexp_full_match(TRIM(score), '[+]?0*(10|[1-9])(\\.0+)?')
                        THEN 'explicit'
                   ELSE 'invalid' END AS score_kind,
              CASE WHEN regexp_full_match(TRIM(score), '[+]?0*(10|[1-9])(\\.0+)?')
                   THEN TRY_CAST(numeric_score AS INTEGER) END AS rating
            FROM parsed
        """, [str(ratings_path)])
        base = dict(db.execute("""
            SELECT 'raw_rows',COUNT(*) FROM raw
            UNION ALL SELECT 'blank_user_rows',COUNT(*) FROM raw
              WHERE username IS NULL OR TRIM(username)=''
            UNION ALL SELECT 'zero_or_missing_rows',COUNT(*) FROM raw WHERE score_kind='unset'
            UNION ALL SELECT 'valid_explicit_rows',COUNT(*) FROM raw
              WHERE score_kind='explicit' AND username IS NOT NULL AND TRIM(username)<>''
            UNION ALL SELECT 'invalid_score_rows',COUNT(*) FROM raw WHERE score_kind='invalid'
            UNION ALL SELECT 'invalid_mal_id_rows',COUNT(*) FROM raw
              WHERE mal_id IS NULL OR mal_id<1 OR mal_id>2147483647
        """).fetchall())
        base = {key: int(value) for key, value in base.items()}
        base["raw_nonblank_users"] = _scalar(db, """
            SELECT COUNT(DISTINCT username) FROM raw
            WHERE username IS NOT NULL AND TRIM(username)<>''
        """)
        raw_status = _hist(db, "SELECT watch_status,COUNT(*) FROM raw GROUP BY 1")
        raw_score = _hist(db, """
            SELECT CASE WHEN score_kind='explicit' THEN CAST(rating AS VARCHAR)
                        ELSE score_kind END,COUNT(*) FROM raw GROUP BY 1
        """)
        db.execute("""
            CREATE TABLE explicit_groups AS
            SELECT username,mal_id,MIN(rating) AS min_rating,MAX(rating) AS max_rating,
                   COUNT(*) AS explicit_rows,MIN(watch_status) AS min_status,
                   MAX(watch_status) AS max_status
            FROM raw WHERE score_kind='explicit' AND username IS NOT NULL
              AND TRIM(username)<>'' AND mal_id BETWEEN 1 AND 2147483647
            GROUP BY username,mal_id
        """)
        db.execute("""
            CREATE TABLE other_groups AS
            SELECT username,mal_id,
                   COUNT(*) FILTER (WHERE score_kind='unset') AS unset_rows,
                   COUNT(*) FILTER (WHERE score_kind='invalid') AS invalid_rows
            FROM raw WHERE score_kind<>'explicit' AND username IS NOT NULL
              AND TRIM(username)<>'' AND mal_id BETWEEN 1 AND 2147483647
            GROUP BY username,mal_id
        """)
        db.execute("CREATE TABLE catalog_ids(mal_id BIGINT PRIMARY KEY)")
        db.executemany("INSERT INTO catalog_ids VALUES (?)", [(item,) for item in sorted(metadata.ids)])
        db.execute("""
            CREATE TABLE pairs AS
            SELECT e.*,COALESCE(o.unset_rows,0) AS unset_rows,
                   COALESCE(o.invalid_rows,0) AS invalid_rows,
                   c.mal_id IS NOT NULL AS mapped
            FROM explicit_groups e LEFT JOIN other_groups o
              ON e.username=o.username AND e.mal_id=o.mal_id
            LEFT JOIN catalog_ids c ON e.mal_id=c.mal_id
        """)
        pair_counts = {key: int(value) for key, value in db.execute("""
            SELECT 'raw_explicit_pairs',COUNT(*) FROM pairs
            UNION ALL SELECT 'raw_mapped_pairs',COUNT(*) FROM pairs WHERE mapped
            UNION ALL SELECT 'unmapped_pairs',COUNT(*) FROM pairs WHERE NOT mapped
            UNION ALL SELECT 'mixed_unset_pairs',COUNT(*) FROM pairs WHERE unset_rows>0
            UNION ALL SELECT 'invalid_state_pairs',COUNT(*) FROM pairs WHERE invalid_rows>0
            UNION ALL SELECT 'conflicting_score_pairs',COUNT(*) FROM pairs
              WHERE min_rating<>max_rating
            UNION ALL SELECT 'status_conflict_pairs',COUNT(*) FROM pairs
              WHERE min_status<>max_status AND min_rating=max_rating
            UNION ALL SELECT 'explicit_pairs_with_other_states',COUNT(*) FROM pairs
              WHERE unset_rows>0 OR invalid_rows>0
            UNION ALL SELECT 'accepted_pairs',COUNT(*) FROM pairs
              WHERE mapped AND min_rating=max_rating AND unset_rows=0 AND invalid_rows=0
        """).fetchall()}
        pair_counts["raw_explicit_mal_ids"] = _scalar(
            db, "SELECT COUNT(DISTINCT mal_id) FROM explicit_groups")
        pair_counts["unrated_or_invalid_only_pairs"] = (
            _scalar(db, "SELECT COUNT(*) FROM other_groups") -
            pair_counts["explicit_pairs_with_other_states"])
        repeated = _scalar(db, """
            SELECT COALESCE(SUM(n-1),0) FROM (
              SELECT COUNT(*) AS n FROM raw WHERE score_kind='explicit'
                AND username IS NOT NULL AND TRIM(username)<>''
                AND mal_id BETWEEN 1 AND 2147483647
              GROUP BY username,mal_id,rating HAVING COUNT(*)>1
            )
        """)
        primary_loss = _hist(db, """
            SELECT CASE WHEN NOT mapped THEN 'missing_metadata'
                        WHEN min_rating<>max_rating THEN 'explicit_score_conflict'
                        WHEN unset_rows>0 THEN 'mixed_unset'
                        WHEN invalid_rows>0 THEN 'invalid_state'
                        ELSE 'accepted' END AS reason,COUNT(*)
            FROM pairs GROUP BY 1
        """)
        # SHA256 receives DuckDB UTF-8 strings exactly as Python user_key does.
        prefix = "'arb-user-v1'||chr(0)||" + _sql_string(namespace) + "||chr(0)"
        db.execute(f"""
            CREATE TABLE accepted AS
            SELECT 'u1_'||sha256({prefix}||username) AS dataset_user_key,
                   username,mal_id,min_rating AS rating,
                   CASE WHEN min_status=max_status THEN min_status ELSE 'unknown' END AS watch_status
            FROM pairs WHERE mapped AND min_rating=max_rating
              AND unset_rows=0 AND invalid_rows=0
        """)
        collisions = _scalar(db, """
            SELECT COUNT(*) FROM (
              SELECT dataset_user_key FROM accepted GROUP BY 1
              HAVING COUNT(DISTINCT username)>1
            )
        """)
        if collisions:
            raise ValueError("user_key_collision")
        score_hist = _hist(db, "SELECT CAST(rating AS VARCHAR),COUNT(*) FROM accepted GROUP BY 1")
        accepted_status = _hist(db, "SELECT watch_status,COUNT(*) FROM accepted GROUP BY 1")
        users = _scalar(db, "SELECT COUNT(DISTINCT dataset_user_key) FROM accepted")
        rated_ids = frozenset(int(row[0]) for row in db.execute(
            "SELECT DISTINCT mal_id FROM accepted").fetchall())
        per_user = db.execute("""
            SELECT COUNT(*) AS rated,COUNT(*) FILTER(WHERE rating>=8) AS positive,
                   COUNT(*) FILTER(WHERE watch_status='completed') AS completed,
                   COUNT(*) FILTER(WHERE rating>=8 AND watch_status='completed') AS completed_positive
            FROM accepted GROUP BY dataset_user_key
        """)
        rated_bins: Counter[str] = Counter()
        positive_bins: Counter[str] = Counter()
        positive_cohort = completed_users = completed_positive_cohort = 0
        while batch := per_user.fetchmany(65536):
            for rated, positive, completed, completed_positive in batch:
                rated_bins[_bin(rated)] += 1
                positive_bins[_bin(positive)] += 1
                positive_cohort += positive >= 15
                completed_users += completed > 0
                completed_positive_cohort += completed_positive >= 15
        completed_pairs = int(accepted_status.get("completed", 0))
        output_path = Path(output_path)
        digest = hashlib.sha256()
        count_bytes = 0
        rows = 0
        with output_path.open("xb") as raw_out:
            os.chmod(output_path, 0o600)
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out,
                               compresslevel=6, mtime=0) as zipped:
                zipped.write(HEADER)
                digest.update(HEADER)
                count_bytes += len(HEADER)
                cursor = db.execute("""
                    SELECT dataset_user_key,mal_id,rating,watch_status FROM accepted
                    ORDER BY dataset_user_key,mal_id
                """)
                while batch := cursor.fetchmany(65536):
                    chunk = "".join(f"{key}\t{mal}\t{rating}\t{status}\n"
                                    for key, mal, rating, status in batch).encode("ascii")
                    zipped.write(chunk)
                    digest.update(chunk)
                    count_bytes += len(chunk)
                    rows += len(batch)
            raw_out.flush()
            os.fsync(raw_out.fileno())
        compressed_bytes = output_path.stat().st_size
        with output_path.open("rb") as compressed:
            compressed_sha = hashlib.file_digest(compressed, "sha256").hexdigest()
        qa = {
            "counts": {**base, **pair_counts, "repeated_explicit_triples": repeated,
                       "accepted_users": users, "accepted_rated_mal_ids": len(rated_ids),
                       "positive_users_ge_15": positive_cohort,
                       "completed_only_pairs": completed_pairs,
                       "completed_only_users": completed_users,
                       "completed_only_users_ge_15_positive": completed_positive_cohort},
            "distributions": {
                "raw_status_histogram": raw_status,
                "raw_score_histogram": raw_score,
                "accepted_status_histogram": accepted_status,
                "accepted_score_histogram": score_hist,
                "rated_items_per_user_bins": dict(sorted(rated_bins.items())),
                "positive_items_per_user_bins": dict(sorted(positive_bins.items())),
            },
            "loss_reasons": {
                "primary": primary_loss,
                "overlap_flags": {
                    "missing_metadata": pair_counts["unmapped_pairs"],
                    "explicit_score_conflict": pair_counts["conflicting_score_pairs"],
                    "mixed_unset": pair_counts["mixed_unset_pairs"],
                    "invalid_state": pair_counts["invalid_state_pairs"],
                },
            },
        }
        return InteractionResult(qa, rows, count_bytes, digest.hexdigest(), compressed_bytes,
                                 compressed_sha, rated_ids)
    except ValueError:
        raise
    except duckdb.OutOfMemoryException:
        raise ValueError("resource_limit") from None
    except OSError:
        raise ValueError("resource_limit") from None
    except duckdb.Error:
        raise ValueError("ratings_csv_invalid") from None
    finally:
        try:
            if db is not None:
                db.close()
        finally:
            os.umask(previous_umask)
