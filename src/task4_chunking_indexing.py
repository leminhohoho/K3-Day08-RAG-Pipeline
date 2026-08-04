"""
Task 4 — Chunking & Indexing vào ChromaDB.

Pipeline:
    Markdown files
        -> load_documents()
        -> chunk_documents()
        -> embed_chunks()
        -> index_to_vectorstore()
        -> chroma_db/

Lựa chọn:
    - RecursiveCharacterTextSplitter:
      Phù hợp với tài liệu Markdown và văn bản tiếng Việt.
      Ưu tiên cắt theo đoạn, dòng và câu trước khi cắt cứng.

    - paraphrase-multilingual-MiniLM-L12-v2:
      Hỗ trợ nhiều ngôn ngữ, bao gồm tiếng Việt và tiếng Anh.
      Nhẹ hơn BGE-M3, phù hợp để chạy local trên máy cá nhân.

    - ChromaDB:
      Chạy local, lưu persistent và không cần Docker.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
CHROMA_DIR = PROJECT_DIR / "chroma_db"


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

# 800 ký tự đủ giữ một đoạn thông tin tương đối hoàn chỉnh.
CHUNK_SIZE = 800

# Overlap 100 ký tự giúp giữ ngữ cảnh giữa hai chunk liên tiếp.
CHUNK_OVERLAP = 100

CHUNKING_METHOD = "recursive"


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

# Model multilingual tương đối nhẹ, phù hợp corpus tiếng Việt.
EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

EMBEDDING_DIM = 384


# ============================================================
# VECTOR STORE CONFIGURATION
# ============================================================

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def detect_document_type(file_path: Path) -> str:
    """
    Xác định loại tài liệu dựa trên thư mục chứa file.

    data/standardized/legal/... -> legal
    data/standardized/news/...  -> news
    """
    relative_parts = [
        part.lower()
        for part in file_path.relative_to(STANDARDIZED_DIR).parts
    ]

    if "legal" in relative_parts:
        return "legal"

    if "news" in relative_parts:
        return "news"

    return "unknown"


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ file Markdown trong data/standardized/.

    Returns:
        [
            {
                "content": "...",
                "metadata": {
                    "source": "document.md",
                    "source_path": "legal/document.md",
                    "type": "legal"
                }
            }
        ]
    """
    if not STANDARDIZED_DIR.exists():
        return []

    markdown_files = sorted(
        file_path
        for file_path in STANDARDIZED_DIR.rglob("*.md")
        if file_path.is_file()
    )

    documents: list[dict] = []

    for file_path in markdown_files:
        try:
            content = file_path.read_text(
                encoding="utf-8"
            ).strip()
        except UnicodeDecodeError:
            content = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()

        # Không đưa file rỗng vào pipeline.
        if not content:
            continue

        relative_path = file_path.relative_to(
            STANDARDIZED_DIR
        )

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": file_path.name,
                    "source_path": relative_path.as_posix(),
                    "type": detect_document_type(file_path),
                    "title": file_path.stem,
                },
            }
        )

    return documents


# ============================================================
# CHUNK DOCUMENTS
# ============================================================

def chunk_documents(
    documents: list[dict],
) -> list[dict]:
    """
    Chia tài liệu thành các đoạn nhỏ bằng
    RecursiveCharacterTextSplitter.

    Mỗi chunk giữ lại metadata của tài liệu gốc và được
    bổ sung chunk_index.

    Args:
        documents:
            Danh sách tài liệu từ load_documents().

    Returns:
        Danh sách chunk:
        [
            {
                "content": "...",
                "metadata": {
                    "source": "...",
                    "type": "...",
                    "chunk_index": 0
                }
            }
        ]
    """
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
    )

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    chunks: list[dict] = []

    for document in documents:
        content = str(
            document.get("content", "")
        ).strip()

        if not content:
            continue

        metadata = dict(
            document.get("metadata", {})
        )

        split_texts = splitter.split_text(content)

        for chunk_index, chunk_text in enumerate(
            split_texts
        ):
            clean_text = chunk_text.strip()

            if not clean_text:
                continue

            chunk_metadata = {
                **metadata,
                "chunk_index": chunk_index,
            }

            chunks.append(
                {
                    "content": clean_text,
                    "metadata": chunk_metadata,
                }
            )

    return chunks


# ============================================================
# EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load embedding model một lần và cache trong bộ nhớ.

    Model sẽ được tải từ Hugging Face trong lần chạy đầu tiên.
    """
    from sentence_transformers import (
        SentenceTransformer,
    )

    print(
        f"Loading embedding model: {EMBEDDING_MODEL}"
    )

    return SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )


def embed_chunks(
    chunks: list[dict],
) -> list[dict]:
    """
    Tạo embedding cho tất cả chunk.

    Mỗi chunk sau khi xử lý có thêm:
        "embedding": list[float]
    """
    if not chunks:
        return []

    model = get_embedding_model()

    texts = [
        str(chunk["content"])
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embedded_chunks: list[dict] = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
        strict=True,
    ):
        new_chunk = {
            "content": chunk["content"],
            "metadata": dict(
                chunk.get("metadata", {})
            ),
            "embedding": embedding.tolist(),
        }

        embedded_chunks.append(new_chunk)

    return embedded_chunks


# ============================================================
# CHROMADB
# ============================================================

@lru_cache(maxsize=1)
def get_chroma_client():
    """
    Tạo ChromaDB PersistentClient.

    Dữ liệu được lưu tại thư mục chroma_db/.
    """
    import chromadb

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


def get_collection():
    """
    Lấy hoặc tạo collection dùng chung cho Task 4 và Task 5.
    """
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "description": (
                "University policies and news documents"
            ),
        },
    )


def make_chunk_id(chunk: dict) -> str:
    """
    Tạo ID ổn định và duy nhất cho một chunk.

    ID dựa trên:
        source_path + chunk_index
    """
    metadata = chunk.get("metadata", {})

    source_path = str(
        metadata.get(
            "source_path",
            metadata.get("source", "unknown"),
        )
    )

    chunk_index = int(
        metadata.get("chunk_index", 0)
    )

    source_hash = hashlib.sha1(
        source_path.encode("utf-8")
    ).hexdigest()[:12]

    return f"{source_hash}_chunk_{chunk_index}"


def clean_metadata(
    metadata: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    """
    ChromaDB chỉ chấp nhận metadata dạng primitive:
    str, int, float hoặc bool.
    """
    clean: dict[str, str | int | float | bool] = {}

    for key, value in metadata.items():
        if isinstance(
            value,
            (str, int, float, bool),
        ):
            clean[str(key)] = value
        elif value is None:
            clean[str(key)] = ""
        else:
            clean[str(key)] = str(value)

    return clean


def reset_collection():
    """
    Xóa collection cũ trước khi index lại.

    Việc này tránh tình trạng corpus cũ và corpus mới
    bị trộn lẫn trong cùng vector store.
    """
    client = get_chroma_client()

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
        print(
            f"Removed old collection: "
            f"{COLLECTION_NAME}"
        )
    except Exception:
        # Collection chưa tồn tại thì không cần xóa.
        pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "description": (
                "University policies and news documents"
            ),
        },
    )


def index_to_vectorstore(
    chunks: list[dict],
):
    """
    Lưu chunks, embeddings và metadata vào ChromaDB.

    Returns:
        ChromaDB collection.
    """
    if not chunks:
        raise ValueError(
            "Không có chunk để index."
        )

    missing_embedding = [
        index
        for index, chunk in enumerate(chunks)
        if "embedding" not in chunk
    ]

    if missing_embedding:
        raise ValueError(
            "Một số chunk chưa có embedding. "
            f"Ví dụ index: {missing_embedding[:5]}"
        )

    collection = reset_collection()

    # Chia batch để tránh gửi quá nhiều dữ liệu một lần.
    batch_size = 100

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):
        batch = chunks[
            start:start + batch_size
        ]

        collection.upsert(
            ids=[
                make_chunk_id(chunk)
                for chunk in batch
            ],
            documents=[
                chunk["content"]
                for chunk in batch
            ],
            embeddings=[
                chunk["embedding"]
                for chunk in batch
            ],
            metadatas=[
                clean_metadata(
                    chunk.get("metadata", {})
                )
                for chunk in batch
            ],
        )

        end = min(
            start + batch_size,
            len(chunks),
        )

        print(
            f"  Indexed chunks "
            f"{start + 1}-{end}/{len(chunks)}"
        )

    return collection


# ============================================================
# RUN PIPELINE
# ============================================================

def run_pipeline() -> None:
    """
    Chạy toàn bộ Task 4:
        load -> chunk -> embed -> index
    """
    print("=" * 60)
    print("TASK 4 — CHUNKING AND INDEXING")
    print("=" * 60)

    print(f"Standardized directory: {STANDARDIZED_DIR}")
    print(
        f"Chunking: {CHUNKING_METHOD} "
        f"(size={CHUNK_SIZE}, "
        f"overlap={CHUNK_OVERLAP})"
    )
    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Vector store: {VECTOR_STORE}")
    print(f"Chroma directory: {CHROMA_DIR}")

    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "Không tìm thấy file Markdown trong "
            "data/standardized/. Hãy hoàn thành Task 3 trước."
        )

    print(
        f"\nLoaded documents: {len(documents)}"
    )

    for document in documents:
        metadata = document["metadata"]

        print(
            f"  - [{metadata['type']}] "
            f"{metadata['source']} "
            f"({len(document['content'])} chars)"
        )

    chunks = chunk_documents(documents)

    if not chunks:
        raise RuntimeError(
            "Không tạo được chunk nào."
        )

    print(f"\nCreated chunks: {len(chunks)}")

    chunk_lengths = [
        len(chunk["content"])
        for chunk in chunks
    ]

    print(
        "Chunk length: "
        f"min={min(chunk_lengths)}, "
        f"max={max(chunk_lengths)}, "
        f"avg={sum(chunk_lengths) / len(chunk_lengths):.1f}"
    )

    embedded_chunks = embed_chunks(chunks)

    print(
        f"\nEmbedded chunks: "
        f"{len(embedded_chunks)}"
    )

    collection = index_to_vectorstore(
        embedded_chunks
    )

    print("\n" + "=" * 60)
    print("TASK 4 COMPLETED")
    print("=" * 60)
    print(
        f"Collection: {collection.name}"
    )
    print(
        f"Indexed records: {collection.count()}"
    )
    print(
        f"Saved at: {CHROMA_DIR}"
    )


if __name__ == "__main__":
    run_pipeline()