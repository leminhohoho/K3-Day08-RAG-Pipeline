"""Fair, reusable retrieval adapters for evaluation configurations."""

from __future__ import annotations

import copy
import time
from typing import Any, Callable


SUPPORTED_CONFIGS = ("dense_only", "bm25_only", "hybrid_rrf")
CONFIG_DESCRIPTIONS = {
    "dense_only": "Task 5 semantic search only",
    "bm25_only": "Task 6 BM25 lexical search only",
    "hybrid_rrf": "Task 5 dense + Task 6 BM25 + weighted RRF",
}
RRF_K = 60
RRF_WEIGHTS = [1.0, 0.9]


SearchFn = Callable[[str, int], list[dict[str, Any]]]
FusionFn = Callable[..., list[dict[str, Any]]]


def _load_default_backends() -> tuple[SearchFn, SearchFn, FusionFn]:
    """Import runtime dependencies only when an actual retrieval run starts."""

    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank_rrf

    return semantic_search, lexical_search, rerank_rrf


def validate_config_names(config_names: list[str]) -> None:
    unknown = sorted(set(config_names) - set(SUPPORTED_CONFIGS))
    if unknown:
        raise ValueError(
            f"Unknown evaluation configs: {', '.join(unknown)}. "
            f"Supported: {', '.join(SUPPORTED_CONFIGS)}"
        )
    if not config_names:
        raise ValueError("At least one evaluation config is required")


def retrieve_configurations(
    query: str,
    config_names: list[str],
    *,
    top_k: int = 5,
    dense_search: SearchFn | None = None,
    bm25_search: SearchFn | None = None,
    rrf_fusion: FusionFn | None = None,
) -> dict[str, dict[str, Any]]:
    """Run shared base retrievers once, then build comparable configurations.

    Candidate generation uses ``top_k * 2`` whenever hybrid fusion is requested.
    This avoids giving hybrid an unfairly small pool and avoids duplicate API calls
    for the same query during one A/B/C evaluation.
    """

    validate_config_names(config_names)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query = str(query or "").strip()
    if not query:
        return {
            name: {
                "results": [],
                "latency_ms": 0.0,
                "component_latency_ms": {},
            }
            for name in config_names
        }

    if dense_search is None or bm25_search is None or rrf_fusion is None:
        defaults = _load_default_backends()
        dense_search = dense_search or defaults[0]
        bm25_search = bm25_search or defaults[1]
        rrf_fusion = rrf_fusion or defaults[2]

    candidate_k = top_k * 2 if "hybrid_rrf" in config_names else top_k
    needs_dense = any(name in config_names for name in ("dense_only", "hybrid_rrf"))
    needs_bm25 = any(name in config_names for name in ("bm25_only", "hybrid_rrf"))

    dense_results: list[dict[str, Any]] = []
    bm25_results: list[dict[str, Any]] = []
    dense_ms = 0.0
    bm25_ms = 0.0

    if needs_dense:
        started = time.perf_counter()
        dense_results = list(dense_search(query, candidate_k))
        dense_ms = (time.perf_counter() - started) * 1000
    if needs_bm25:
        started = time.perf_counter()
        bm25_results = list(bm25_search(query, candidate_k))
        bm25_ms = (time.perf_counter() - started) * 1000

    outputs: dict[str, dict[str, Any]] = {}
    for name in config_names:
        if name == "dense_only":
            outputs[name] = {
                "results": copy.deepcopy(dense_results[:top_k]),
                "latency_ms": dense_ms,
                "component_latency_ms": {"dense": dense_ms},
            }
        elif name == "bm25_only":
            outputs[name] = {
                "results": copy.deepcopy(bm25_results[:top_k]),
                "latency_ms": bm25_ms,
                "component_latency_ms": {"bm25": bm25_ms},
            }
        else:
            started = time.perf_counter()
            fused = rrf_fusion(
                [dense_results, bm25_results],
                top_k=top_k,
                k=RRF_K,
                weights=RRF_WEIGHTS,
            )
            fusion_ms = (time.perf_counter() - started) * 1000
            fused = copy.deepcopy(fused[:top_k])
            for item in fused:
                item["source"] = "hybrid"
            outputs[name] = {
                "results": fused,
                "latency_ms": dense_ms + bm25_ms + fusion_ms,
                "component_latency_ms": {
                    "dense": dense_ms,
                    "bm25": bm25_ms,
                    "fusion": fusion_ms,
                },
            }
    return outputs

