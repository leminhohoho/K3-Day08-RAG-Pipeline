"""
Task 5 — Query Expansion (Bonus 5 điểm).

Deterministic query expansion using domain glossary.
Fusion via RRF on chunk_id.
"""

from functools import lru_cache

from .task4_chunking_indexing import get_embedding_model, prepare_query_for_embedding
from .task5_semantic_search import (
    normalize_query,
    semantic_search,
    _deduplicate_by_chunk_id,
    _stable_sort_key,
)


# =============================================================================
# DOMAIN GLOSSARY — bilingual rewrites
# =============================================================================

DOMAIN_GLOSSARY = {
    # Vietnamese -> English
    "học phí": "tuition fee",
    "thanh toán": "payment",
    "học bổng": "scholarship",
    "ký túc xá": "dormitory",
    "đăng ký học phần": "course registration",
    "thư viện": "library",
    "phòng học nhóm": "group study room",
    "sinh viên quốc tế": "international student",
    "hỗ trợ tài chính": "financial aid",
    "chỗ ở": "accommodation",
    "học lại": "retake",
    "tín chỉ": "credit",
    "học kỳ": "semester",
    "năm học": "academic year",
    "điểm": "grade",
    "xét tuyển": "admission",
    # English -> Vietnamese
    "tuition": "học phí",
    "fee": "phí",
    "scholarship": "học bổng",
    "library": "thư viện",
    "dormitory": "ký túc xá",
    "registration": "đăng ký",
    "course": "môn học",
    "international": "quốc tế",
    "payment": "thanh toán",
    "study room": "phòng học",
    "accommodation": "chỗ ở",
    "financial aid": "hỗ trợ tài chính",
    "admission": "tuyển sinh",
    "credit": "tín chỉ",
    "semester": "học kỳ",
    "grade": "điểm số",
}


def _bilingual_rewrite(query: str) -> str:
    """Thay thế glossary terms: query gốc → bilingual variant."""
    result = query
    for src, tgt in DOMAIN_GLOSSARY.items():
        # Case-insensitive replacement
        idx = result.lower().find(src.lower())
        if idx >= 0:
            result = result[:idx] + tgt + result[idx + len(src):]
            break  # chỉ thay 1 term để tránh over-rewrite
    return result if result != query else query


def _canonical_rewrite(query: str) -> str:
    """
    Canonical rewrite: chuyển mixed query về dạng thuần Việt hoặc thuần Anh
    (chọn canonical dạng có nhiều term glossary hơn).
    """
    vi_count = sum(1 for term in DOMAIN_GLOSSARY if term in query.lower() and all(
        ord(c) > 127 or c.isalpha() for c in term
    ))
    en_count = sum(1 for term, _ in DOMAIN_GLOSSARY.items() if term in query.lower() and all(
        c.isascii() and c.isalpha() for c in term
    ))
    if vi_count >= en_count:
        # Try to rewrite English terms to Vietnamese
        result = query
        for src, tgt in DOMAIN_GLOSSARY.items():
            if src.isascii() and src.isalpha():
                idx = result.lower().find(src.lower())
                if idx >= 0:
                    result = result[:idx] + tgt + result[idx + len(src):]
        return result
    return query


def expand_query(query: str, max_variants: int = 3) -> list[str]:
    """
    Deterministic query expansion, không cần API key.

    Args:
        query: Câu truy vấn gốc
        max_variants: Tổng số variants tối đa (bao gồm query gốc)

    Returns:
        List of strings: [original, ...unique variants]
    """
    normalized = normalize_query(query)
    if not normalized:
        return []

    variants = [normalized]
    if max_variants <= 1:
        return variants[:max_variants]

    # Variant 1: bilingual rewrite
    bilingual = _bilingual_rewrite(normalized)
    n_bilingual = normalize_query(bilingual)
    if n_bilingual and n_bilingual != normalized and n_bilingual not in variants:
        variants.append(n_bilingual)

    if len(variants) >= max_variants:
        return variants[:max_variants]

    # Variant 2: canonical rewrite
    canonical = _canonical_rewrite(normalized)
    n_canonical = normalize_query(canonical)
    if n_canonical and n_canonical not in variants:
        variants.append(n_canonical)

    return variants[:max_variants]


def semantic_search_expanded(
    query: str,
    top_k: int = 10,
    max_variants: int = 3,
) -> list[dict]:
    """
    Search original và expanded queries, sau đó RRF fusion by chunk_id.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối
        max_variants: Số variants tối đa

    Returns:
        List of results, mỗi result có thêm 'raw_scores.dense' (best cosine).
    """
    variants = expand_query(query, max_variants=max_variants)
    if not variants:
        return []

    # Search mỗi variant với candidate pool lớn hơn
    per_query_k = max(top_k * 2, 10)
    all_ranked_lists = []
    all_raw = {}  # chunk_id -> best_dense_score

    for variant in variants:
        results = semantic_search(variant, top_k=per_query_k)
        if results:
            all_ranked_lists.append(results)
            for r in results:
                cid = r.get("metadata", {}).get("chunk_id", r["content"])
                score = r.get("raw_scores", {}).get("dense", r["score"])
                if cid not in all_raw or score > all_raw[cid]:
                    all_raw[cid] = score

    if not all_ranked_lists:
        return []

    # RRF fusion
    from .task7_reranking import rerank_rrf

    fused = rerank_rrf(all_ranked_lists, top_k=top_k, k=60)
    for item in fused:
        item["score_type"] = "cosine_rrf"
        cid = item.get("metadata", {}).get("chunk_id", item["content"])
        best_dense = all_raw.get(cid, item["score"])
        if "raw_scores" not in item:
            item["raw_scores"] = {}
        item["raw_scores"]["dense"] = best_dense
        item["confidence_score"] = best_dense
        item["matched_queries"] = [v for v in variants]

    return fused[:top_k]


if __name__ == "__main__":
    q = "payment structure của học phí như thế nào?"
    print(f"Query: {q}")
    print(f"Variants: {expand_query(q)}")
    results = semantic_search_expanded(q, top_k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:80]}...")