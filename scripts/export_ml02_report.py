"""Export the public ML-02 validation summary from a completed private campaign directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "src/recsys/resources/protocols/research_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "recsys/reports/ml02-validation-v1"
RANDOM_ORDERS = ("random-liked-1", "random-liked-2", "random-liked-3")
N_RANGE = tuple(range(1, 11))
TABLE_N = (1, 2, 3, 5, 10)
DETAIL_N = 5
TOLERANCE = 1e-9
GROUP_FIELDS = ("coverage", "known_history_repeat_rate")
LABELS = {
    "popularity": "Популярное",
    "content-mean": "GloVe, среднее",
    "content-max": "GloVe, максимум",
    "content-union": "GloVe, объединение",
    "itemknn": "ItemKNN",
    "puresvd": "PureSVD",
    "hybrid": "Гибрид (RRF)",
    "item-seed-mean-bpr-v1": "BPR (PyTorch)",
}


class ReportError(Exception):
    pass


def obj(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReportError(f"{where}: expected object")
    return value


def arr(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise ReportError(f"{where}: expected array")
    return value


def text(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise ReportError(f"{where}: expected string")
    return value


def num(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReportError(f"{where}: expected number")
    return float(value)


def integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(f"{where}: expected integer")
    return value


def numbers(value: object, where: str, length: int) -> list[float]:
    items = arr(value, where)
    if len(items) != length:
        raise ReportError(f"{where}: expected {length} values")
    return [num(item, f"{where}[{index}]") for index, item in enumerate(items)]


def load(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    loaded: object = json.loads(raw)
    return obj(loaded, path.name), raw


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def groups(
    variant: dict[str, object], cohort: str, mask: str, order: str
) -> dict[int, dict[str, object]]:
    found: dict[int, dict[str, object]] = {}
    for index, item in enumerate(arr(variant["checked_groups"], "checked_groups")):
        group = obj(item, f"checked_groups[{index}]")
        if (group["cohort"], group["mask"], group["order"]) == (cohort, mask, order):
            found[integer(group["n"], "n")] = group
    return found


def mean_over_orders(
    variant: dict[str, object], cohort: str, mask: str, field: str
) -> dict[int, tuple[float, int]]:
    by_order = [groups(variant, cohort, mask, order) for order in RANDOM_ORDERS]
    result: dict[int, tuple[float, int]] = {}
    for n in sorted(set.intersection(*(set(order) for order in by_order))):
        values: list[float] = []
        users = {integer(order[n]["queries"], "queries") for order in by_order}
        if len(users) != 1:
            raise ReportError(f"{cohort}/{mask}/n={n}: orders cover different users")
        for order in by_order:
            group = order[n]
            source = group if field in GROUP_FIELDS else obj(group["metrics"], "metrics")
            values.append(num(source[field], field))
        result[n] = (sum(values) / len(values), users.pop())
    return result


def rounded(value: float) -> float:
    return round(value, 6)


def summarize(campaign_dir: Path) -> dict[str, object]:
    manifest, _ = load(campaign_dir / "campaign-report-manifest.json")
    report, report_raw = load(campaign_dir / "campaign-report.json")
    plan, _ = load(campaign_dir / "campaign-plan.json")
    protocol, protocol_raw = load(PROTOCOL)

    descriptor = obj(manifest["report"], "manifest.report")
    if manifest["status"] != "complete" or report["status"] != "complete":
        raise ReportError("campaign is not complete")
    if descriptor["sha256"] != sha256(report_raw) or descriptor["bytes"] != len(report_raw):
        raise ReportError("campaign-report.json does not match its manifest")
    if plan["protocol_identity"] != f"sha256:{sha256(protocol_raw)}":
        raise ReportError("protocol resource differs from the campaign protocol")
    if report["plan_identity"] != manifest["plan_identity"]:
        raise ReportError("report and manifest name different plans")

    primary = obj(report["primary_ndcg5"], "primary_ndcg5")
    paired = obj(report["paired_vs_popularity"], "paired_vs_popularity")
    plan_variants = {
        text(obj(item, "plan.variant")["variant_id"], "variant_id"): obj(item, "plan.variant")
        for item in arr(plan["variants"], "plan.variants")
    }
    comparison = obj(protocol["comparison"], "protocol.comparison")
    storage = obj(plan["storage"], "plan.storage")
    dimensions = obj(storage["prediction_dimensions"], "prediction_dimensions")

    models: list[dict[str, object]] = []
    for index, item in enumerate(arr(report["variants"], "variants")):
        variant = obj(item, f"variants[{index}]")
        vid = text(variant["variant_id"], "variant_id")
        ndcg5 = numbers(obj(primary[vid], vid)["mean_ndcg5_by_n"], f"{vid}.ndcg5", len(N_RANGE))
        controlled = {
            field: mean_over_orders(variant, "main", "controlled", field)
            for field in ("ndcg5", "ndcg10", "recall10", "hit10", "coverage")
        }
        for n, value in zip(N_RANGE, ndcg5, strict=True):
            if abs(controlled["ndcg5"][n][0] - value) > TOLERANCE:
                raise ReportError(f"{vid}: recomputed NDCG@5 differs at n={n}")
        top_rated = groups(variant, "main", "controlled", "top-rated")
        resources = obj(variant["resources"], "resources")
        model: dict[str, object] = {
            "id": vid,
            "framework": variant["framework"],
            "train_device": obj(variant["train_device"], "train_device")["kind"],
            "parameters": plan_variants[vid].get("parameters", {}),
            "ndcg5_by_n": [rounded(value) for value in ndcg5],
            "main_controlled_by_n": {
                str(n): {
                    field: rounded(controlled[field][n][0])
                    for field in ("ndcg10", "recall10", "hit10", "coverage")
                }
                for n in N_RANGE
            },
            "main_top_rated_ndcg5_by_n": [
                rounded(num(obj(top_rated[n]["metrics"], "metrics")["ndcg5"], "ndcg5"))
                for n in N_RANGE
            ],
            "main_product_history_repeat_rate_by_n": [
                rounded(value)
                for _, (value, _) in sorted(
                    mean_over_orders(
                        variant, "main", "product", "known_history_repeat_rate"
                    ).items()
                )
            ],
            "short_controlled_ndcg5_by_n": [
                {"n": n, "users": users, "ndcg5": rounded(value)}
                for n, (value, users) in sorted(
                    mean_over_orders(variant, "short", "controlled", "ndcg5").items()
                )
            ],
            "resources": {
                "fit_seconds": rounded(num(obj(resources["fit"], "fit")["elapsed_seconds"], "s")),
                "peak_rss_bytes": max(
                    integer(obj(phase, "phase")["observed_peak_rss_bytes"], "rss")
                    for phase in resources.values()
                ),
                "scores_stored_bytes": integer(variant["scores_stored_bytes"], "scores"),
            },
        }
        if vid in paired:
            pair = obj(paired[vid], vid)
            model["diff_vs_popularity_by_n"] = [
                rounded(value)
                for value in numbers(pair["mean_difference_by_n"], f"{vid}.diff", len(N_RANGE))
            ]
            model["ci95_vs_popularity_by_n"] = [
                [rounded(bound) for bound in numbers(interval, f"{vid}.ci", 2)]
                for interval in arr(pair["pointwise_ci95_by_n"], f"{vid}.ci")
            ]
        models.append(model)

    return {
        "schema_version": 1,
        "kind": "arb-ml02-validation-summary",
        "split": "validation",
        "campaign": {
            "id": report["campaign_id"],
            "code_revision": plan["code_revision"],
            "created_utc": plan["created_utc"],
            "plan_identity": report["plan_identity"],
            "prepared_identity": report["prepared_identity"],
            "protocol_identity": plan["protocol_identity"],
            "report_sha256": sha256(report_raw),
            "report_bytes": len(report_raw),
            "completed_attempts": report["actual_attempts_completed"],
            "planned_nodes": report["planned_nodes"],
        },
        "data": {
            "catalog_items": dimensions["catalog_count"],
            "validation_queries": dimensions["query_count"],
            "train_interactions": integer(obj(plan["train_interactions"], "train")["rows"], "rows")
            - 1,
            "primary_users": report["primary_users"],
        },
        "protocol": {
            "primary_metric": "NDCG@5",
            "cohort": "main",
            "mask": "controlled",
            "orders": list(RANDOM_ORDERS),
            "n": list(N_RANGE),
            "bootstrap_replicates": comparison["bootstrap_replicates"],
            "bootstrap_seed": comparison["seed"],
            "confidence": comparison["confidence"],
            "epsilon_ndcg5": comparison["epsilon_ndcg5"],
        },
        "models": models,
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def signed(value: float) -> str:
    return f"{value:+.3f}"


def grouped(value: object) -> str:
    return f"{integer(value, 'count'):,}".replace(",", " ")


def mib(value: int) -> str:
    return f"{value / 2**20:,.0f}".replace(",", " ")


def first_significant(model: dict[str, object]) -> int | None:
    intervals = arr(model.get("ci95_vs_popularity_by_n", []), "ci")
    lows = [num(arr(interval, "ci")[0], "ci") for interval in intervals]
    for start in range(len(lows)):
        if all(low > 0 for low in lows[start:]):
            return N_RANGE[start]
    return None


def render(summary: dict[str, object]) -> str:
    campaign = obj(summary["campaign"], "campaign")
    data = obj(summary["data"], "data")
    protocol = obj(summary["protocol"], "protocol")
    models = {
        text(obj(item, "model")["id"], "id"): obj(item, "model")
        for item in arr(summary["models"], "models")
    }

    def series(vid: str, field: str) -> list[float]:
        return numbers(models[vid][field], f"{vid}.{field}", len(N_RANGE))

    ranked = sorted(models, key=lambda vid: -series(vid, "ndcg5_by_n")[-1])
    best = ranked[0]
    best_ndcg = series(best, "ndcg5_by_n")
    popularity = series("popularity", "ndcg5_by_n")[0]
    content_n1 = series("content-mean", "ndcg5_by_n")[0]
    epsilon = num(protocol["epsilon_ndcg5"], "epsilon")
    near_best = next(
        n for n, v in zip(N_RANGE, best_ndcg, strict=True) if v >= max(best_ndcg) - epsilon
    )

    lines = [
        "# ML-02: сравнение моделей на validation",
        "",
        "Офлайн-сравнение восьми моделей рекомендаций по набору любимых тайтлов нового пользователя."
        " Это итог задачи ARB-042. Test-выборка не открывалась; выбор модели и N — следующий этап"
        " (ML-03).",
        "",
        "## Постановка",
        "",
        f"- Данные: Neelagiri Anime Dataset 2025, каталог {grouped(data['catalog_items'])} тайтлов,"
        f" {grouped(data['train_interactions'])} train-взаимодействий. Пользователи разделены на"
        " train/validation/test целиком.",
        f"- Запрос: пользователь называет N понравившихся тайтлов (N = 1…10), модель ранжирует весь"
        f" каталог. Основная когорта — {data['primary_users']} validation-пользователей с не менее"
        " чем 15 понравившимися тайтлами; у каждого 5 скрытых целей, одинаковых для всех N.",
        "- Понравилось = завершено с оценкой ≥ 8. Неоценённое считается неизвестным, а не"
        " отрицательным.",
        f"- Метрика: {protocol['primary_metric']}, среднее по трём случайным порядкам выбора"
        " входных тайтлов. Сравнение с популярным — парный bootstrap по пользователям"
        f" ({protocol['bootstrap_replicates']} повторов, 95% доверительный интервал).",
        "",
        "## NDCG@5 по числу входных тайтлов",
        "",
        "| Модель | "
        + " | ".join(f"N={n}" for n in TABLE_N)
        + f" | Δ к популярному при N={DETAIL_N} |",
        "| --- | " + " | ".join("---:" for _ in TABLE_N) + " | --- |",
    ]
    for vid in ranked:
        values = series(vid, "ndcg5_by_n")
        cells = [fmt(values[n - 1]) for n in TABLE_N]
        if vid == "popularity":
            delta = "—"
        else:
            diff = series(vid, "diff_vs_popularity_by_n")[DETAIL_N - 1]
            low, high = numbers(
                arr(models[vid]["ci95_vs_popularity_by_n"], "ci")[DETAIL_N - 1], "ci", 2
            )
            delta = f"{signed(diff)} [{signed(low)}, {signed(high)}]"
        lines.append(f"| {LABELS[vid]} | " + " | ".join(cells) + f" | {delta} |")

    lines += [
        "",
        f"## Дополнительные метрики при N={DETAIL_N}",
        "",
        "Recall@10, HitRate@10 и покрытие каталога — основная когорта. «Уже в истории» — доля"
        " выдачи, которую пользователь уже оценивал, если, как бот, исключать только введённые"
        " тайтлы. Top-rated — входные тайтлы с самыми высокими оценками вместо случайных."
        " Короткие истории — пользователи с 2–14 понравившимися тайтлами.",
        "",
        "| Модель | Recall@10 | HitRate@10 | Покрытие каталога | Уже в истории"
        " | NDCG@5, top-rated | NDCG@5, короткие истории |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for vid in ranked:
        model = models[vid]
        detail = obj(obj(model["main_controlled_by_n"], "by_n")[str(DETAIL_N)], "detail")
        short = {
            integer(obj(row, "short")["n"], "n"): obj(row, "short")
            for row in arr(model["short_controlled_ndcg5_by_n"], "short")
        }
        short_cell = (
            f"{fmt(num(short[DETAIL_N]['ndcg5'], 'ndcg5'))} ({short[DETAIL_N]['users']} польз.)"
            if DETAIL_N in short
            else "—"
        )
        lines.append(
            f"| {LABELS[vid]} | {fmt(num(detail['recall10'], 'r'))}"
            f" | {fmt(num(detail['hit10'], 'h'))}"
            f" | {num(detail['coverage'], 'c'):.2%}"
            f" | {series(vid, 'main_product_history_repeat_rate_by_n')[DETAIL_N - 1]:.0%}"
            f" | {fmt(series(vid, 'main_top_rated_ndcg5_by_n')[DETAIL_N - 1])}"
            f" | {short_cell} |"
        )

    lines += [
        "",
        "## Ресурсы",
        "",
        "| Модель | Обучение, с | Пик RSS, МиБ | Прогнозы на диске, МиБ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for vid in ranked:
        res = obj(models[vid]["resources"], "resources")
        lines.append(
            f"| {LABELS[vid]} | {num(res['fit_seconds'], 's'):.1f}"
            f" | {mib(integer(res['peak_rss_bytes'], 'rss'))}"
            f" | {mib(integer(res['scores_stored_bytes'], 'b'))} |"
        )

    significant = [
        f"{LABELS[vid]} — начиная с N={n}"
        for vid in ranked
        if (n := first_significant(models[vid])) is not None
    ]
    lines += [
        "",
        "## Выводы",
        "",
        f"- Лучшая модель — {LABELS[best]}: NDCG@5 = {fmt(best_ndcg[-1])} при N=10 против"
        f" {fmt(popularity)} у популярного.",
        "- Значимо лучше популярного (нижняя граница 95% интервала > 0 для этого и всех больших N): "
        + ("; ".join(significant) if significant else "нет")
        + ".",
        f"- При одном входном тайтле {LABELS[best]} не лучше популярного: Δ ="
        f" {signed(series(best, 'diff_vs_popularity_by_n')[0])}. Выигрыш появляется с двух тайтлов.",
        f"- Контентная модель на GloVe, на которой построен текущий бот, при N=1 даёт"
        f" {fmt(content_n1)} — примерно в {popularity / content_n1:.0f} раз хуже популярного.",
        f"- Кривая {LABELS[best]} выходит на плато: точечная оценка отстаёт от максимума меньше чем"
        f" на ε = {epsilon} начиная с N={near_best}. Формальный выбор N по правилу протокола"
        " (верхняя граница сожаления) — задача ML-03.",
        f"- Если, как сейчас в боте, исключать только введённые тайтлы, то при N={DETAIL_N}"
        f" {series(best, 'main_product_history_repeat_rate_by_n')[DETAIL_N - 1]:.0%} выдачи"
        f" {LABELS[best]} — тайтлы, которые пользователь уже оценивал. Для внедрения нужен способ"
        " исключать уже просмотренное.",
        "",
        "## Ограничения",
        "",
        "- Офлайн-оценка восстанавливает известные высокие оценки. Она благоприятна для"
        " коллаборативных моделей и популярного и не измеряет «похожесть», которую ищут контентные"
        " методы.",
        "- Основная когорта — опытные зрители с длинной историей; короткие истории показаны"
        " отдельно и с меньшим числом пользователей.",
        "- Временных меток оценок нет, поэтому это не предсказание следующего просмотра.",
        "- BPR — одна заранее объявленная конфигурация без подбора гиперпараметров.",
        "- Задержка ответа и пропускная способность не измерялись.",
        "",
        "## Происхождение",
        "",
        f"- Кампания `{campaign['id']}`, код `{campaign['code_revision']}`,"
        f" запущена {campaign['created_utc']}; {campaign['completed_attempts']}"
        f" из {campaign['planned_nodes']} этапов завершены.",
        f"- План `{campaign['plan_identity']}`, подготовленные данные `{campaign['prepared_identity']}`,"
        f" протокол `{campaign['protocol_identity']}`.",
        f"- Исходный приватный отчёт: {grouped(campaign['report_bytes'])} байт,"
        f" SHA-256 `{campaign['report_sha256']}`.",
        "- Все числа — в [summary.json](summary.json). Файлы созданы командой"
        " `python3 scripts/export_ml02_report.py <campaign-dir>` из проверенного каталога кампании.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path, help="completed private campaign directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    campaign_dir: Path = args.campaign_dir
    output: Path = args.output
    try:
        summary = summarize(campaign_dir)
    except (OSError, KeyError, ValueError, ReportError) as exc:
        print(f"export_ml02_report: {exc}", file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(render(summary), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
