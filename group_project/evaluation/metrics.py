"""Deterministic, model-free metrics for the RAG evaluation pipeline."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable


WORD_RE = re.compile(r"\w+", re.UNICODE)
VI_STOPWORDS = {
    "bị",
    "bao",
    "các",
    "cho",
    "có",
    "của",
    "được",
    "giữa",
    "gì",
    "khi",
    "không",
    "là",
    "một",
    "những",
    "này",
    "theo",
    "thì",
    "trong",
    "từ",
    "và",
    "vào",
    "với",
}


def normalize_text(value: Any) -> str:
    """Normalize Unicode and whitespace while keeping Vietnamese diacritics."""

    text = unicodedata.normalize("NFC", str(value or "")).casefold()
    return " ".join(text.split())


def evidence_tokens(value: Any) -> set[str]:
    """Return meaningful tokens used by the deterministic evidence proxy."""

    return {
        token
        for token in WORD_RE.findall(normalize_text(value))
        if len(token) >= 2 and token not in VI_STOPWORDS
    }


def _metadata(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def result_identity(result: dict[str, Any]) -> str:
    """Prefer stable chunk ID, then fall back to normalized content provenance."""

    metadata = _metadata(result)
    chunk_id = normalize_text(metadata.get("chunk_id"))
    if chunk_id:
        return f"chunk:{chunk_id}"
    source = normalize_text(metadata.get("source"))
    content = normalize_text(result.get("content"))
    digest = hashlib.sha256(f"{source}|{content}".encode("utf-8")).hexdigest()
    return f"fallback:{digest}"


def result_matches_aliases(result: dict[str, Any], aliases: Iterable[str], source: str) -> bool:
    """Match retrieval metadata against canonical/runtime IDs or source filename."""

    metadata = _metadata(result)
    result_source = normalize_text(metadata.get("source"))
    result_document_id = normalize_text(metadata.get("document_id"))
    alias_set = {normalize_text(alias) for alias in aliases if normalize_text(alias)}
    return result_source == normalize_text(source) or result_document_id in alias_set


def matched_group_index(result: dict[str, Any], case: dict[str, Any]) -> int | None:
    """Return which expected-document group the result satisfies, if any."""

    aliases = case["expected_document_id_aliases"]
    sources = case["expected_source_files"]
    for index, (group, source) in enumerate(zip(aliases, sources)):
        if result_matches_aliases(result, group, source):
            return index
    return None


def document_ranks(results: list[dict[str, Any]], case: dict[str, Any], k: int) -> list[int | None]:
    """Return the first one-based rank for each expected document."""

    ranks: list[int | None] = [None] * len(case["expected_source_files"])
    for rank, result in enumerate(results[:k], start=1):
        group = matched_group_index(result, case)
        if group is not None and ranks[group] is None:
            ranks[group] = rank
    return ranks


def source_hit_at_k(results: list[dict[str, Any]], case: dict[str, Any], k: int) -> float:
    return float(any(rank is not None for rank in document_ranks(results, case, k)))


def document_recall_at_k(results: list[dict[str, Any]], case: dict[str, Any], k: int) -> float:
    ranks = document_ranks(results, case, k)
    return sum(rank is not None for rank in ranks) / max(1, len(ranks))


def reciprocal_rank(results: list[dict[str, Any]], case: dict[str, Any], k: int) -> float:
    ranks = [rank for rank in document_ranks(results, case, k) if rank is not None]
    return 0.0 if not ranks else 1.0 / min(ranks)


def ndcg_at_k(results: list[dict[str, Any]], case: dict[str, Any], k: int) -> float:
    """Binary document-level nDCG without rewarding duplicate chunks."""

    seen_groups: set[int] = set()
    dcg = 0.0
    for rank, result in enumerate(results[:k], start=1):
        group = matched_group_index(result, case)
        if group is None or group in seen_groups:
            continue
        seen_groups.add(group)
        dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(k, len(case["expected_source_files"]))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def evidence_coverage_at_k(results: list[dict[str, Any]], case: dict[str, Any], k: int) -> float:
    """Lexical evidence coverage proxy; RAGAS context recall remains the judge metric."""

    expected = evidence_tokens(case.get("expected_context", ""))
    if not expected:
        return 1.0
    retrieved = evidence_tokens("\n".join(str(item.get("content", "")) for item in results[:k]))
    return len(expected & retrieved) / len(expected)


def duplicate_context_rate_at_k(results: list[dict[str, Any]], k: int) -> float:
    selected = results[:k]
    if len(selected) <= 1:
        return 0.0
    identities = [result_identity(item) for item in selected]
    return (len(identities) - len(set(identities))) / len(identities)


def evaluate_retrieval_case(
    case: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    top_k: int,
    latency_ms: float,
) -> dict[str, Any]:
    """Calculate all offline metrics and compact diagnostics for one case."""

    metrics: dict[str, Any] = {
        "source_hit_at_1": source_hit_at_k(results, case, 1),
        "source_hit_at_3": source_hit_at_k(results, case, min(3, top_k)),
        "source_hit_at_5": source_hit_at_k(results, case, min(5, top_k)),
        "document_recall_at_5": document_recall_at_k(results, case, min(5, top_k)),
        "mrr_at_5": reciprocal_rank(results, case, min(5, top_k)),
        "ndcg_at_5": ndcg_at_k(results, case, min(5, top_k)),
        "evidence_coverage_at_5": evidence_coverage_at_k(results, case, min(5, top_k)),
        "duplicate_context_rate_at_5": duplicate_context_rate_at_k(results, min(5, top_k)),
        "empty_result": float(not results),
        "latency_ms": float(latency_ms),
    }
    metrics["expected_document_ranks"] = document_ranks(results, case, min(5, top_k))
    metrics["retrieved_sources"] = [str(_metadata(item).get("source", "")) for item in results[:top_k]]
    metrics["retrieved_document_ids"] = [
        str(_metadata(item).get("document_id", "")) for item in results[:top_k]
    ]
    return metrics


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_case_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Macro-average query metrics and expose operational percentiles."""

    if not records:
        return {"case_count": 0}
    metric_names = (
        "source_hit_at_1",
        "source_hit_at_3",
        "source_hit_at_5",
        "document_recall_at_5",
        "mrr_at_5",
        "ndcg_at_5",
        "evidence_coverage_at_5",
        "duplicate_context_rate_at_5",
        "empty_result",
    )
    aggregate: dict[str, float | int] = {"case_count": len(records)}
    for name in metric_names:
        aggregate[name] = mean(float(record["metrics"][name]) for record in records)
    latencies = [float(record["metrics"]["latency_ms"]) for record in records]
    aggregate["latency_mean_ms"] = mean(latencies)
    aggregate["latency_p50_ms"] = median(latencies)
    aggregate["latency_p95_ms"] = percentile(latencies, 0.95)
    return aggregate


def aggregate_by_field(
    records: list[dict[str, Any]], field: str
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(field, "unknown"))].append(record)
    return {key: aggregate_case_metrics(value) for key, value in sorted(grouped.items())}

