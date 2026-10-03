"""Blinded catalog evidence packet, strict judgment checks, and fixed-slot summaries."""

import hashlib
import json
from importlib import resources


class AssessmentError(ValueError):
    """Invalid assessment packet or declared judgments."""


PACKET_VERSION = "arb018-assessment-v1"
PROMPT_VERSION = "arb018-catalog-evidence-v1"
FIELDS = ("title", "aliases", "genres", "type", "episodes", "year", "synopsis")


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prompt_sha():
    return sha(resources.files("recsys").joinpath("assessment_prompt_v1.md").read_bytes())


def make_packet(comparison, catalog):
    anime = catalog["anime"]
    pairs = sorted({(query["query_mal_id"], candidate["mal_id"])
                    for method in comparison["methods"] for query in method["queries"]
                    for candidate in query["recommendations"]})
    ids = sorted({i for pair in pairs for i in pair})
    evidence = {str(i): {field: anime[str(i)][field] for field in FIELDS} for i in ids}
    return {"schema_version": 1, "packet_version": PACKET_VERSION,
            "comparison_sha256": sha(canonical(comparison)),
            "catalog_sha256": comparison["catalog_sha256"],
            "query_set_sha256": comparison["query_set_sha256"],
            "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha(),
            "anime": evidence,
            "pairs": [{"query_mal_id": q, "candidate_mal_id": c} for q, c in pairs]}


def make_template(packet):
    return {"schema_version": 1, "assessment_version": PACKET_VERSION,
            "packet_sha256": sha(canonical(packet)),
            "assessor": {"id": None, "kind": None, "method_version": None,
                         "model": None, "prompt_version": None, "prompt_sha256": None},
            "judgments": [{**pair, "relevance": None, "rationale": None, "evidence": [],
                           "same_franchise": None, "franchise_rationale": None}
                          for pair in packet["pairs"]]}


def validate_assessment(value, packet):
    if not isinstance(value, dict) or set(value) != {"schema_version", "assessment_version", "packet_sha256", "assessor", "judgments"}:
        raise AssessmentError("invalid assessment fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["assessment_version"] != PACKET_VERSION:
        raise AssessmentError("invalid assessment version")
    if value["packet_sha256"] != sha(canonical(packet)):
        raise AssessmentError("assessment packet SHA256 mismatch")
    assessor = value["assessor"]
    if not isinstance(assessor, dict) or set(assessor) != {"id", "kind", "method_version", "model", "prompt_version", "prompt_sha256"}:
        raise AssessmentError("invalid assessor declaration")
    if any(not isinstance(assessor[k], str) or not assessor[k].strip() for k in ("id", "method_version")):
        raise AssessmentError("assessor ID and method version are required")
    if assessor["kind"] not in ("human", "llm"):
        raise AssessmentError("assessor kind must be human or llm")
    if assessor["kind"] == "llm":
        if (not isinstance(assessor["model"], str) or not assessor["model"].strip()
                or assessor["prompt_version"] != PROMPT_VERSION
                or assessor["prompt_sha256"] != prompt_sha()):
            raise AssessmentError("LLM model and packaged prompt declaration required")
    elif any(assessor[k] is not None for k in ("model", "prompt_version", "prompt_sha256")):
        raise AssessmentError("human model and prompt fields must be null")
    judgments = value["judgments"]
    expected = [(p["query_mal_id"], p["candidate_mal_id"]) for p in packet["pairs"]]
    if not isinstance(judgments, list) or len(judgments) != len(expected):
        raise AssessmentError("judgments must cover every packet pair")
    seen = set()
    for row in judgments:
        if not isinstance(row, dict) or set(row) != {"query_mal_id", "candidate_mal_id", "relevance", "rationale", "evidence", "same_franchise", "franchise_rationale"}:
            raise AssessmentError("invalid judgment fields")
        q, c = row["query_mal_id"], row["candidate_mal_id"]
        if type(q) is not int or type(c) is not int or (q, c) not in expected or (q, c) in seen:
            raise AssessmentError("unknown or duplicate judgment pair")
        seen.add((q, c))
        grade = row["relevance"]
        if grade is not None and (type(grade) is not int or grade not in (0, 1, 2)):
            raise AssessmentError("relevance must be 0, 1, 2, or null")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise AssessmentError("every judgment needs a rationale")
        evidence = row["evidence"]
        if not isinstance(evidence, list):
            raise AssessmentError("evidence must be a list")
        sides = set()
        for citation in evidence:
            if not isinstance(citation, dict) or set(citation) != {"side", "field", "quote"} or citation["side"] not in ("query", "candidate") or citation["field"] not in FIELDS:
                raise AssessmentError("invalid evidence citation")
            quote = citation["quote"]
            if not isinstance(quote, str) or not quote.strip() or len(quote) > 240:
                raise AssessmentError("evidence quote must be 1..240 characters")
            item = packet["anime"][str(q if citation["side"] == "query" else c)][citation["field"]]
            if isinstance(item, str):
                valid = quote in item
            elif isinstance(item, list):
                valid = quote in item
            elif item is None or isinstance(item, (int, float)):
                valid = quote == str(item)
            else:
                valid = False
            if not valid:
                raise AssessmentError("evidence quote absent from cited catalog field")
            sides.add(citation["side"])
        if grade is not None and sides != {"query", "candidate"}:
            raise AssessmentError("numeric grade needs evidence from both sides")
        franchise = row["same_franchise"]
        if franchise is not None and type(franchise) is not bool:
            raise AssessmentError("same_franchise must be boolean or null")
        if not isinstance(row["franchise_rationale"], str) or not row["franchise_rationale"].strip():
            raise AssessmentError("franchise rationale required")
    return assessor["id"]


def score_method(method, judgments):
    """Compute grade and franchise bounds over five slots for every query."""
    by_pair = {(j["query_mal_id"], j["candidate_mal_id"]): j for j in judgments}
    rows = []
    for query in method["queries"]:
        q = query["query_mal_id"]
        items = [by_pair[(q, c["mal_id"])] for c in query["recommendations"]]
        grades = [item["relevance"] for item in items]
        flags = [item["same_franchise"] for item in items]
        known = [g for g in grades if g is not None]
        unknown = len(grades) - len(known)
        grade_sum, positive = sum(known), sum(g >= 1 for g in known)
        true = sum(flag is True for flag in flags)
        franchise_unknown = sum(flag is None for flag in flags)
        rows.append({"query_mal_id": q, "status": query["status"],
                     "returned": len(items), "missing_slots": 5 - len(items),
                     "assessed": len(known), "unknown_returned": unknown,
                     "fullness": len(items) / 5,
                     "mean_grade_at_5": grade_sum / 5 if unknown == 0 else None,
                     "mean_grade_bounds": [grade_sum / 5, (grade_sum + 2 * unknown) / 5],
                     "precision_at_5": positive / 5 if unknown == 0 else None,
                     "precision_bounds": [positive / 5, (positive + unknown) / 5],
                     "same_franchise_at_5": true / 5 if franchise_unknown == 0 else None,
                     "same_franchise_bounds": [true / 5, (true + franchise_unknown) / 5],
                     "franchise_known": len(flags) - franchise_unknown,
                     "franchise_unknown": franchise_unknown,
                     "grade_histogram": {str(g): grades.count(g) for g in (0, 1, 2)}})
    def macro(key):
        values = [row[key] for row in rows]
        return sum(values) / 20 if all(value is not None for value in values) else None
    def bounds(key):
        return [sum(row[key][i] for row in rows) / 20 for i in (0, 1)]
    returned = sum(row["returned"] for row in rows)
    assessed = sum(row["assessed"] for row in rows)
    franchise_known = sum(row["franchise_known"] for row in rows)
    return {"method_id": method["id"], "queries": rows,
            "macro": {"returned": returned, "missing_slots": 100 - returned,
                      "assessed": assessed, "unknown_returned": returned - assessed,
                      "coverage": assessed / returned if returned else None,
                      "full_five_queries": sum(row["returned"] == 5 for row in rows),
                      "mean_grade_at_5": macro("mean_grade_at_5"),
                      "mean_grade_bounds": bounds("mean_grade_bounds"),
                      "precision_at_5": macro("precision_at_5"),
                      "precision_bounds": bounds("precision_bounds"),
                      "same_franchise_at_5": macro("same_franchise_at_5"),
                      "same_franchise_bounds": bounds("same_franchise_bounds"),
                      "franchise_coverage": franchise_known / returned if returned else None,
                      "grade_histogram": {str(g): sum(row["grade_histogram"][str(g)] for row in rows)
                                          for g in (0, 1, 2)}}}
