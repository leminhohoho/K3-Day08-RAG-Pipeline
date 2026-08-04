"""
Task 5 — Semantic Search Module.

Tìm kiếm ngữ nghĩa (dense retrieval) trên vector store ChromaDB.

- Input: query string + top_k
- Output: danh sách chunks có score cosine, sorted descending
- Dùng chung embedding model + collection từ Task 4
"""

import unicodedata
from functools import lru_cache

from .task4_chunking_indexing import (
    get_embedding_model,
    get_collection,
    prepare_query_for_embedding,
)


# =============================================================================
# QUERY PREPROCESSING
# =============================================================================

def normalize_query(query: str) -> str:
    """
    Normalize query:
        - Unicode NFC
        - Trim đầu/cuối
        - Collapse multiple whitespace thành 1 space
        - Giữ nguyên dấu tiếng Việt
    """
    if not isinstance(query, str):
        raise TypeError(f"Query phải là str, nhận: {type(query)}")
    normalized = unicodedata.normalize("NFC", query).strip()
    return " ".join(normalized.split())


def _deduplicate_by_chunk_id(results: list[dict]) -> list[dict]:
    """Deduplicate theo chunk_id (fallback: content hash + source). Giữ score cao hơn."""
    seen = {}
    for r in results:
        meta = r.get("metadata") or {}
        chunk_id = meta.get("chunk_id")
        if not chunk_id:
            chunk_id = f"{r.get('content', '')}|{meta.get('source', '')}"
        if chunk_id not in seen or r["score"] > seen[chunk_id]["score"]:
            seen[chunk_id] = r
    return list(seen.values())


def _stable_sort_key(item: dict):
    meta = item.get("metadata") or {}
    return (
        item["score"],
        meta.get("document_id", ""),
        meta.get("chunk_index", 0),
        meta.get("chunk_id", ""),
    )


# =============================================================================
# CORE SEARCH
# =============================================================================

@lru_cache(maxsize=256)
def _query_embedding_cached(normalized_query: str, model_name: str) -> tuple:
    """LRU cache cho query embedding (key: normalized query + model id)."""
    prepared = prepare_query_for_embedding(normalized_query)
    emb = get_embedding_model().encode(prepared, normalize_embeddings=True)
    return tuple(float(x) for x in emb)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity trên ChromaDB.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,       # cosine similarity [0, 1]
            'score_type': 'cosine',
            'metadata': dict      # chunk_id, source, document_id, type, ...
            'raw_scores': {'dense': float}
        }
        Sorted by score descending.
    """
    normalized = normalize_query(query)
    if not normalized or top_k <= 0:
        return []

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    query_vector = _query_embedding_cached(normalized, "bge-m3")
    n_results = min(top_k, count)
    response = collection.query(
        query_embeddings=[list(query_vector)],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        if not doc:
            continue
        similarity = 1.0 - float(dist)
        score = max(0.0, min(1.0, similarity))
        score = round(score, 6)
        safe_meta = dict(meta) if meta else {}
        output.append({
            "content": doc,
            "score": float(score),
            "score_type": "cosine",
            "metadata": safe_meta,
            "raw_scores": {"dense": float(score)},
        })

    output = _deduplicate_by_chunk_id(output)
    output.sort(key=_stable_sort_key, reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    results = semantic_search("học phí chương trình Business là bao nhiêu?", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
