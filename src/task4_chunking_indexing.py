"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 800        # 800 chars: cân bằng giữa context completeness và retrieval granularity
CHUNK_OVERLAP = 100      # 12.5% overlap: tránh mất thông tin ở chunk boundaries
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

EMBEDDING_MODEL = "BAAI/bge-m3"  # Multilingual 1024-dim, tốt cho cả tiếng Việt và tiếng Anh
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"  # Local persistent, không cần Docker
COLLECTION_NAME = "university_services_docs"

# Prefix cho query/document encoding (dùng cho E5 models, BGE m3 không cần)
EMBEDDING_QUERY_PREFIX = ""
EMBEDDING_DOC_PREFIX = ""

# Embedding qua OpenRouter API (không cần download model 2GB)
EMBEDDING_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
EMBEDDING_API_BASE = "https://openrouter.ai/api/v1"
EMBEDDING_BATCH_SIZE = 20


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of dicts, each with:
            "content": str
            "metadata": dict with source, document_id, type, title, url, section, language
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        document_id = md_file.stem  # filename without extension
        # Derive title from first H1 or filename stem
        title = document_id
        for line in content.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "document_id": document_id,
                "type": doc_type,
                "title": title,
                "url": "",
                "section": "",
                "language": "vi",
            }
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunk_id = f"{doc['metadata']['document_id']}_chunk_{i}"
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "chunk_id": chunk_id,
                }
            })
    return chunks


# =============================================================================
# EMBEDDING — OpenRouter API (fallback: local SentenceTransformer)
# =============================================================================

_openai_client = None
_local_model = None


def _get_openai_client():
    """Lazy-load và cache OpenAI-compatible client (OpenRouter)."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_API_BASE)
    return _openai_client


def _get_local_model():
    """Lazy-load và cache local SentenceTransformer (fallback)."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(EMBEDDING_MODEL)
    return _local_model


def _embed_batch_api(texts: list[str]) -> list[list[float]]:
    """Embed một batch texts qua OpenRouter API (OpenAI-compatible embeddings endpoint)."""
    if not EMBEDDING_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY chưa được set trong .env")
    client = _get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # Sort theo index để giữ thứ tự input
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed texts: ưu tiên OpenRouter API, fallback local SentenceTransformer.
    """
    try:
        return _embed_batch_api(texts)
    except Exception as api_err:
        print(f"⚠ OpenRouter embedding failed ({api_err}), fallback to local model...")
        model = _get_local_model()
        embeddings = model.encode(texts, show_progress_bar=True)
        return [emb.tolist() for emb in embeddings]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng OpenRouter API (fallback local model).

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    texts = [c["content"] for c in chunks]
    # Batch embed theo EMBEDDING_BATCH_SIZE
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        all_embeddings.extend(_embed_texts(batch))
    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c["metadata"]["chunk_id"] for c in chunks]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


# =============================================================================
# SHARED HELPERS (exported for Task 5, 6, 9)
# =============================================================================

_embedding_model = None
_chroma_client = None
_chroma_collection = None


class _APIEmbeddingModel:
    """
    Wrapper mimic SentenceTransformer.encode() interface, dùng OpenRouter API
    với fallback local model. Đảm bảo get_embedding_model() trả instance dùng
    chung cho cả query lẫn document encoding.
    """

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, **kwargs):
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        embeddings = _embed_texts(list(texts))
        if normalize_embeddings:
            import numpy as np
            arr = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            embeddings = arr.tolist()
        if single_input:
            return embeddings[0]
        return embeddings


def get_embedding_model():
    """
    Lazy-loads và caches embedding model instance.
    Returns the same instance on every call within a process.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _APIEmbeddingModel()
    return _embedding_model


def get_collection():
    """
    Opens persistent ChromaDB collection. Lazy-loaded và cached.
    Raises config error if embedding dimension mismatches.
    """
    global _chroma_client, _chroma_collection
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if _chroma_collection is None:
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        # Check dimension if collection has data
        if _chroma_collection.count() > 0:
            sample = _chroma_collection.get(limit=1, include=["embeddings"])
            if sample is not None:
                emb_list = sample.get("embeddings")
                if emb_list is not None and len(emb_list) > 0:
                    actual_dim = len(emb_list[0])
                    if actual_dim != EMBEDDING_DIM:
                        raise ValueError(
                            f"Embedding dimension mismatch: collection has {actual_dim}, "
                            f"config expects {EMBEDDING_DIM}. "
                            f"Delete chroma_db/ and re-run Task 4."
                        )
    return _chroma_collection


def prepare_query_for_embedding(query: str) -> str:
    """
    Applies EMBEDDING_QUERY_PREFIX if configured.
    Must be symmetric with doc-side prefix used during indexing.
    """
    if EMBEDDING_QUERY_PREFIX:
        return f"{EMBEDDING_QUERY_PREFIX}{query}"
    return query


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
