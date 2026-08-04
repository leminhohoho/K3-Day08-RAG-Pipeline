"""
Task 5 — Semantic Search.

Luồng xử lý:
    Query của người dùng
        -> tạo query embedding
        -> tìm kiếm trong ChromaDB
        -> đổi cosine distance thành similarity score
        -> sắp xếp score giảm dần
        -> trả về top_k kết quả
"""

from __future__ import annotations

from typing import Any


# Hỗ trợ cả hai cách chạy:
#
#   python -m src.task5_semantic_search
#
# và:
#
#   python src/task5_semantic_search.py
try:
    from src.task4_chunking_indexing import (
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        get_collection,
        get_embedding_model,
    )
except ImportError:
    from task4_chunking_indexing import (
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        get_collection,
        get_embedding_model,
    )


def distance_to_similarity(distance: float) -> float:
    """
    Chuyển cosine distance của ChromaDB thành similarity score.

    Với cosine distance:
        distance càng nhỏ -> nội dung càng giống query.

    Chuyển đổi:
        similarity = 1 - distance

    Sau chuyển đổi:
        similarity càng lớn -> kết quả càng liên quan.
    """
    similarity = 1.0 - float(distance)

    # Giữ score trong khoảng hợp lý [-1, 1].
    return max(-1.0, min(1.0, similarity))


def normalize_metadata(
    metadata: Any,
) -> dict[str, Any]:
    """
    Đảm bảo metadata luôn là dictionary.

    ChromaDB đôi khi có thể trả metadata=None.
    """
    if isinstance(metadata, dict):
        return dict(metadata)

    return {}


def semantic_search(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Tìm kiếm các chunk có ý nghĩa gần nhất với câu hỏi.

    Args:
        query:
            Câu hỏi hoặc nội dung cần tìm kiếm.

        top_k:
            Số kết quả tối đa cần trả về.

    Returns:
        Danh sách kết quả có dạng:

        [
            {
                "content": "...",
                "score": 0.82,
                "metadata": {
                    "source": "...",
                    "source_path": "...",
                    "type": "legal",
                    "chunk_index": 2
                },
                "source": "semantic"
            }
        ]
    """
    if not isinstance(query, str):
        raise TypeError("query phải là chuỗi.")

    clean_query = query.strip()

    if not clean_query:
        return []

    if not isinstance(top_k, int):
        raise TypeError("top_k phải là số nguyên.")

    if top_k <= 0:
        return []

    # Lấy collection đã được tạo ở Task 4.
    collection = get_collection()

    total_records = collection.count()

    # Collection chưa có dữ liệu.
    if total_records == 0:
        return []

    # Không thể yêu cầu nhiều kết quả hơn số record hiện có.
    number_of_results = min(
        top_k,
        total_records,
    )

    # Dùng đúng embedding model của Task 4.
    model = get_embedding_model()

    query_embedding = model.encode(
        clean_query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # ChromaDB nhận embedding ở dạng list[list[float]].
    query_result = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=number_of_results,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    # ChromaDB trả kết quả theo batch.
    # Vì chỉ có 1 query nên lấy phần tử đầu tiên.
    documents = (
        query_result.get("documents")
        or [[]]
    )[0]

    metadatas = (
        query_result.get("metadatas")
        or [[]]
    )[0]

    distances = (
        query_result.get("distances")
        or [[]]
    )[0]

    results: list[dict] = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        if document is None:
            continue

        content = str(document).strip()

        if not content:
            continue

        similarity_score = distance_to_similarity(
            float(distance)
        )

        results.append(
            {
                "content": content,
                "score": similarity_score,
                "metadata": normalize_metadata(
                    metadata
                ),
                "source": "semantic",
            }
        )

    # Score càng cao thì càng liên quan.
    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]


def print_results(
    query: str,
    results: list[dict],
) -> None:
    """
    In kết quả semantic search ra terminal.
    """
    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH")
    print("=" * 70)
    print(f"Query: {query}")
    print(f"Results: {len(results)}")

    if not results:
        print(
            "\nKhông tìm thấy kết quả. "
            "Hãy kiểm tra Task 4 và ChromaDB."
        )
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):
        metadata = result.get(
            "metadata",
            {},
        )

        source_name = metadata.get(
            "source",
            "unknown",
        )

        document_type = metadata.get(
            "type",
            "unknown",
        )

        chunk_index = metadata.get(
            "chunk_index",
            "unknown",
        )

        preview = result["content"][:500]

        print("\n" + "-" * 70)
        print(f"Rank:        {rank}")
        print(f"Score:       {result['score']:.4f}")
        print(f"Source:      {source_name}")
        print(f"Type:        {document_type}")
        print(f"Chunk index: {chunk_index}")
        print("\nContent:")
        print(preview)

        if len(result["content"]) > 500:
            print("...")


def main() -> None:
    """
    Chạy thử Semantic Search.
    """
    print("=" * 70)
    print("TASK 5 — SEMANTIC SEARCH")
    print("=" * 70)
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Collection: {COLLECTION_NAME}")

    example_queries = [
        "Điều kiện nhận học bổng khuyến khích học tập là gì?",
        "Sinh viên không được làm gì trong ký túc xá?",
        "Mức học phí của sinh viên là bao nhiêu?",
    ]

    for query in example_queries:
        results = semantic_search(
            query=query,
            top_k=3,
        )

        print_results(
            query=query,
            results=results,
        )


if __name__ == "__main__":
    main()