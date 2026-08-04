"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + RRF fusion + PageIndex fallback.

Logic:
    1. Chạy semantic_search + lexical_search
    2. Merge bằng RRF
    3. Rerank (optional)
    4. Nếu best cosine < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ LUÔN dùng raw cosine similarity (dense_results[0]["score"]) cho fallback threshold,
   KHÔNG dùng RRF score (RRF max score luôn ≈ 0.0164 bất kể relevance).
"""

import sys
from pathlib import Path

# Ensure project root is on path for direct script execution
_proj_root = Path(__file__).parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

try:
    from .task5_semantic_search import semantic_search  # type: ignore
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank, rerank_rrf
    from src.task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Calibrate threshold: chạy in-domain vs OOD queries, chọn ngưỡng giữa 2 clusters
SCORE_THRESHOLD = 0.3   # Raw cosine similarity threshold
DEFAULT_TOP_K = 5
RERANK_METHOD = "none"  # "none" | "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với hybrid search + fallback.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ RRF Merge → merged_results
          ├→ Rerank (optional) → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn (tiếng Việt, Anh, hoặc mixed)
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng cosine similarity gốc tối thiểu để skip fallback
        use_reranking: Có áp dụng reranking sau RRF merge hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'score_type': str,       # 'rrf' | 'reranker' | 'mmr' | 'rank_proxy'
            'confidence_score': float | None,
            'metadata': dict,
            'source': str,           # 'hybrid' hoặc 'pageindex'
            'raw_scores': dict
        }
    """
    # Step 1: Parallel retrieval (top_k * 2 để có candidate pool cho rerank)
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: RRF Merge
    ranked_lists = [lst for lst in [dense_results, sparse_results] if lst]
    if not ranked_lists:
        # Try fallback directly
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback
        return []

    merged = rerank_rrf(ranked_lists, top_k=top_k * 2, k=60)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Optional Reranking
    if use_reranking and RERANK_METHOD != "none" and merged:
        try:
            final = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        except (NotImplementedError, Exception):
            # Fallback to plain top_k
            final = merged[:top_k]
    else:
        final = merged[:top_k]

    # Step 4: Fallback check — DÙNG RAW COSINE, KHÔNG PHẢI RRF
    best_cosine = dense_results[0]["score"] if dense_results else 0.0
    if best_cosine < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback
        # If PageIndex also returns empty, return empty list
        return []

    return final[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Điều kiện xin học bổng là gì?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            source = r.get("source", "?")
            score = r.get("score", 0.0)
            content = r.get("content", "")[:80]
            print(f"  {i}. [{score:.4f}] [{source}] {content}...")