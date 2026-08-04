"""
Task 7 — Reranking Module.

- RRF (Reciprocal Rank Fusion): weighted, deterministic, không cần API key
- Cross-encoder (optional): Jina Reranker v2 multilingual
- MMR (optional): Maximal Marginal Relevance

Lưu ý: RRF score CHỈ dùng xếp hạng (rank-based), KHÔNG dùng làm confidence
hoặc fallback threshold. Confidence lấy từ raw dense cosine score (Task 5).
"""

import hashlib
import re
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

RRF_K = 60
RRF_WEIGHTS = {"dense": 1.0, "bm25": 0.9, "tfidf": 0.7}
FUSION_CANDIDATE_K = 15
FINAL_TOP_K = 5


def _normalize_content(content: str) -> str:
    return re.sub(r"\s+", " ", content).strip()


def candidate_key(item: dict) -> str:
    """
    Dedupe key: ưu tiên chunk_id, fallback sha256(normalized content + source).
    """
    metadata = item.get("metadata") or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"
    content_hash = hashlib.sha256(
        _normalize_content(item.get("content", "")).encode("utf-8")
    ).hexdigest()
    return f"fallback:{content_hash}|{metadata.get('source', '')}"


def _sanitize_float(value) -> Optional[float]:
    """Convert to float, drop NaN/Inf."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    if f in (float("inf"), float("-inf")):
        return None
    return f


def _copy_candidate(item: dict) -> dict:
    """Deep-ish copy của candidate để không mutate input."""
    new_item = dict(item)
    new_item["metadata"] = dict(item.get("metadata") or {})
    new_item["raw_scores"] = dict(item.get("raw_scores") or {})
    return new_item


def _get_native_rank(item: dict) -> int:
    """Best native rank of the item (từ ranks dict nếu có)."""
    ranks = item.get("ranks") or {}
    if not ranks:
        return 10**9
    return min(ranks.values())


def _stable_rrf_sort_key(item: dict):
    """Tie-break: ranker count desc → best dense desc → native rank asc → chunk_id asc."""
    raw = item.get("raw_scores") or {}
    metadata = item.get("metadata") or {}
    return (
        -len(item.get("matched_rankers", [])),
        -float(raw.get("dense") or 0.0),
        _get_native_rank(item),
        metadata.get("chunk_id", ""),
    )


# =============================================================================
# RRF — RECIPROCAL RANK FUSION
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
    weights: Optional[list[float]] = None,
) -> list[dict]:
    """
    Weighted Reciprocal Rank Fusion.

    RRF(d) = Σ weightᵣ / (k + rankᵣ(d)), rank bắt đầu từ 1.

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60)
        weights: Optional list weights cho mỗi ranked list

    Returns:
        List of top_k candidates sorted by RRF score descending.
        Không mutate input objects.
    """
    if not ranked_lists or top_k <= 0:
        return []
    if k <= 0:
        raise ValueError(f"k phải > 0, nhận: {k}")

    # Normalize weights
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) phải bằng ranked_lists length ({len(ranked_lists)})"
        )
    for w in weights:
        if not isinstance(w, (int, float)) or w != w or w in (float("inf"), float("-inf")):
            raise ValueError(f"Invalid weight: {w}")
        if w < 0:
            raise ValueError(f"Weight phải >= 0, nhận: {w}")

    # Merge results
    merged: dict[str, dict] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        if not ranked_list:
            continue
        # Infer ranker name: từ retrieval_method/score_type của items trong list
        first = ranked_list[0]
        method = first.get("retrieval_method") or first.get("score_type") or "unknown"
        if method == "cosine" or method == "cosine_rrf":
            ranker_name = "dense"
        elif method == "bm25":
            ranker_name = "bm25"
        elif method == "tfidf":
            ranker_name = "tfidf"
        else:
            ranker_name = method

        for rank, item in enumerate(ranked_list, 1):
            key = candidate_key(item)
            contribution = weight / (k + rank)
            if key not in merged:
                merged[key] = _copy_candidate(item)
                merged[key]["score"] = 0.0
                merged[key]["score_type"] = "rrf"
                merged[key]["matched_rankers"] = []
                merged[key]["ranks"] = {}
            cur = merged[key]
            cur["score"] += contribution

            # Track matched rankers + native rank
            if ranker_name not in cur["matched_rankers"]:
                cur["matched_rankers"].append(ranker_name)
            cur["ranks"][ranker_name] = rank

            # Merge raw_scores
            raw = item.get("raw_scores") or {}
            for rk, rv in raw.items():
                if rv is not None and rk not in cur["raw_scores"]:
                    cur["raw_scores"][rk] = rv
            # Track dense from score when raw_scores missing
            if "dense" not in cur["raw_scores"] and item.get("score_type") == "cosine":
                cur["raw_scores"]["dense"] = _sanitize_float(item.get("score"))

    if not merged:
        return []

    results = []
    for key, item in merged.items():
        item["score"] = float(item["score"])
        item["raw_scores"]["rrf"] = item["score"]

        # confidence_score = max raw dense cosine (KHÔNG lấy RRF score)
        dense = item["raw_scores"].get("dense")
        if dense is not None:
            item["confidence_score"] = float(dense)
        else:
            item["confidence_score"] = None
        results.append(item)

    # Sort theo RRF score desc, tie-break deterministic
    results.sort(key=lambda it: (-it["score"],) + _stable_rrf_sort_key(it))
    return results[:top_k]


# =============================================================================
# CROSS-ENCODER (OPTIONAL)
# =============================================================================

def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Cross-encoder reranking dùng Jina Reranker API (nếu có JINA_API_KEY).

    Fallback: giữ nguyên order candidates (prior order).
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    if not query or not candidates or top_k <= 0:
        return []

    api_key = os.getenv("JINA_API_KEY", "")
    if not api_key:
        return candidates[:top_k]

    try:
        import requests

        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        reranked = response.json()["results"]
        output = []
        for r in reranked:
            item = _copy_candidate(candidates[r["index"]])
            item["score"] = float(r["relevance_score"])
            item["score_type"] = "reranker"
            item["raw_scores"]["reranker"] = float(r["relevance_score"])
            output.append(item)
        return output[:top_k]
    except Exception as e:
        print(f"⚠ Cross-encoder fallback (prior order): {e}")
        return candidates[:top_k]


# =============================================================================
# MMR (OPTIONAL)
# =============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(
            f"Embedding dimension mismatch: {len(a)} vs {len(b)}"
        )
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — relevant + diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Raises ValueError nếu candidate thiếu embedding hoặc dimension mismatch.
    """
    if not 0 <= lambda_param <= 1:
        raise ValueError(f"lambda_param phải trong [0, 1], nhận: {lambda_param}")
    if not candidates or top_k <= 0:
        return []

    # Validate embeddings
    for c in candidates:
        emb = c.get("embedding")
        if emb is None:
            raise ValueError("Candidate thiếu 'embedding' — không thể chạy MMR")
        if len(emb) != len(query_embedding):
            raise ValueError(
                f"Embedding dimension mismatch: candidate {len(emb)} vs query {len(query_embedding)}"
            )

    selected: list[dict] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")
        for idx in remaining:
            relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(
                    candidates[idx]["embedding"],
                    candidates[sel_idx]["embedding"],
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        item = _copy_candidate(candidates[best_idx])
        item["score"] = float(best_score)
        item["score_type"] = "mmr"
        item["raw_scores"]["mmr"] = float(best_score)
        selected.append(item)
        remaining.remove(best_idx)

    return selected


# =============================================================================
# UNIFIED RERANK INTERFACE
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """
    Unified single-list reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: "rrf" | "cross_encoder" | "mmr" | "none"

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates or top_k <= 0:
        return []

    if method == "rrf":
        # Single-list rank normalization
        return rerank_rrf([candidates], top_k=top_k)
    elif method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query embedding — embed query qua Task 4 helper
        from .task4_chunking_indexing import get_embedding_model, prepare_query_for_embedding

        model = get_embedding_model()
        query_emb = model.encode(prepare_query_for_embedding(query), normalize_embeddings=True)
        try:
            return rerank_mmr(query_emb, candidates, top_k=top_k)
        except ValueError:
            raise
    elif method == "none":
        return candidates[:top_k]
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {"chunk_id": "c1"}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {"chunk_id": "c2"}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {"chunk_id": "c3"}},
    ]
    results = rerank("tuition fee payment", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
