"""
Task 4 — Chunking & Indexing vào Vector Store.

Chunking strategy: fixed_size (size=800, overlap=100)
Embedding: OpenRouter API (BAAI/bge-m3), fallback SentenceTransformer
Vector Store: ChromaDB (persistent, rebuild on each run)
"""

import hashlib
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "fixed_size"  # fixed-size split (size, overlap)

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"

EMBEDDING_QUERY_PREFIX = ""
EMBEDDING_DOC_PREFIX = ""

# Embedding via OpenRouter API
EMBEDDING_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
EMBEDDING_API_BASE = "https://openrouter.ai/api/v1"
EMBEDDING_BATCH_SIZE = 20


# =============================================================================
# PARSE canonical metadata header (Task 3 format)
# =============================================================================

def _parse_markdown_file(filepath: Path) -> dict:
    """
    Parse canonical Task 3 markdown format:
      - Header metadata (## key: value lines) above `---` separator
      - Body content below `---`

    Returns:
        {"content": str, "metadata": dict}
    """
    raw = filepath.read_text(encoding="utf-8-sig")
    lines = raw.split("\n")

    # Find the `---` separator
    sep_index = None
    for i, line in enumerate(lines):
        if line.strip() == "---" and i > 0:  # not the first line
            sep_index = i
            break

    header_lines = lines[:sep_index] if sep_index is not None else []
    body_lines = lines[sep_index + 1:] if sep_index is not None else lines
    body = "\n".join(body_lines).strip()

    # Parse header fields
    header = {}
    _title = filepath.stem
    _document_id = filepath.stem
    _url = ""
    _type = "legal" if "legal" in str(filepath) else "news"
    _language = "vi"
    _content_hash = ""

    for line in header_lines:
        line = line.strip()
        # Match **Key:** value format (bold wraps key AND colon)
        m = re.match(r"\*\*(\w+(?:\s+\w+)*):\*\*\s*(.*)", line)
        if not m:
            # Fallback: **Key:** value
            m = re.match(r"\*\*(\w+(?:\s+\w+)*)\*\*:\s*(.*)", line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            header[key] = val

    # Extract known fields
    if "Document ID" in header:
        _document_id = header["Document ID"]
    if "Title" in header:
        _title = header["Title"]
    if "Source" in header:
        _url = header["Source"]
    if "Type" in header:
        _type = header["Type"].lower()
    if "Language" in header:
        _language = header["Language"].lower()
    if "Content Hash" in header:
        _content_hash = header["Content Hash"]

    # Fallback: extract title from first H1 in header_lines (before ---) or body
    if not _title or _title == filepath.stem:
        for line in header_lines + body_lines:
            if line.startswith("# ") and not line.startswith("##"):
                _title = line[2:].strip()
                break
    if not _title:
        _title = filepath.stem

    # If no separator found, body is the whole file content minus empty leading lines
    if sep_index is None:
        body = raw.strip()

    return {
        "content": body,
        "metadata": {
            "source": filepath.name,
            "source_path": str(filepath.relative_to(filepath.parent.parent.parent)),
            "document_id": _document_id,
            "type": _type,
            "title": _title,
            "url": _url,
            "language": _language,
            "section": "",
            "section_path": "",
            "content_hash": _content_hash,
        },
    }


# =============================================================================
# load_documents
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.
    Parse canonical metadata header, chỉ đưa body vào content.
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        doc = _parse_markdown_file(md_file)
        if not doc["content"]:
            print(f"⚠ Warning: {md_file.name} has empty body after parsing")
            continue
        documents.append(doc)
    return documents


# =============================================================================
# chunk_documents — fixed_size
# =============================================================================

def _fixed_size_split(content: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split content into fixed-size chunks with overlap.
    Cắt cứng theo số ký tự, không tôn trọng ranh giới dòng/đoạn.

    Args:
        content: Nội dung cần split.
        chunk_size: Số ký tự tối đa mỗi chunk.
        chunk_overlap: Số ký tự overlap giữa các chunk kề nhau.

    Returns:
        list[str]: Các chunk text.
    """
    if not content:
        return []

    step = chunk_size - chunk_overlap
    if step <= 0:
        # Nếu overlap >= size, chỉ trả toàn bộ content làm 1 chunk
        return [content]

    chunks = []
    start = 0
    n = len(content)
    while start < n:
        chunks.append(content[start:start + chunk_size])
        start += step

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo fixed_size strategy:
      - Chia nội dung thành các đoạn có độ dài tối đa CHUNK_SIZE
      - Overlap CHUNK_OVERLAP giữa các chunk kề nhau
      - Không giữ cấu trúc heading (section = title fallback)
    """
    chunks = []
    chunk_index = 0

    for doc in documents:
        meta = doc["metadata"]
        content = doc["content"]
        title = meta.get("title", meta.get("document_id", "unknown"))

        splits = _fixed_size_split(content, CHUNK_SIZE, CHUNK_OVERLAP)
        for split_text in splits:
            if not split_text.strip():
                continue
            chunk_id = f"{meta['document_id']}_chunk_{chunk_index}"
            chunk_hash = hashlib.sha256(split_text.encode()).hexdigest()[:12]
            chunks.append({
                "content": split_text,
                "metadata": {
                    **meta,
                    "section": title,
                    "section_path": "",
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                    "chunk_hash": chunk_hash,
                },
            })
            chunk_index += 1

    return chunks


# =============================================================================
# EMBEDDING — OpenRouter API (fallback: local SentenceTransformer)
# =============================================================================

_openai_client = None
_local_model = None
_embedding_model = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_API_BASE)
    return _openai_client


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(EMBEDDING_MODEL)
    return _local_model


def _embed_batch_api(texts: list[str]) -> list[list[float]]:
    if not EMBEDDING_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY chưa được set trong .env")
    client = _get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        return _embed_batch_api(texts)
    except Exception as api_err:
        print(f"⚠ OpenRouter embedding failed ({api_err}), fallback to local model...")
        model = _get_local_model()
        embeddings = model.encode(texts, show_progress_bar=True)
        return [emb.tolist() for emb in embeddings]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks. Chỉ encode chunk content, không encode metadata.
    """
    if not chunks:
        return []

    texts = [c["content"] for c in chunks]
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        all_embeddings.extend(_embed_texts(batch))

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb

    # Verify dimension
    if all_embeddings:
        actual_dim = len(all_embeddings[0])
        if actual_dim != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {actual_dim}. "
                f"Check model: {EMBEDDING_MODEL}"
            )

    return chunks


# =============================================================================
# index_to_vectorstore — rebuild collection
# =============================================================================

def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store.
    Rebuild collection (delete + create) để tránh stale data.
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete old collection if exists, then create fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = [c["metadata"]["chunk_id"] for c in batch]
        documents = [c["content"] for c in batch]
        embeddings = [c["embedding"] for c in batch]
        # Only store scalar metadata (str, int, float, bool)
        metadatas = []
        for c in batch:
            m = dict(c["metadata"])
            # Remove non-scalar fields
            for k in list(m.keys()):
                v = m[k]
                if not isinstance(v, (str, int, float, bool)):
                    m[k] = str(v) if v is not None else ""
            metadatas.append(m)

        collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


# =============================================================================
# SHARED HELPERS (exported for Task 5, 6, 9)
# =============================================================================

_chroma_client = None
_chroma_collection = None


class _APIEmbeddingModel:
    """Wrapper mimic SentenceTransformer.encode() interface, dùng OpenRouter API."""

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
    """Lazy-load và cache embedding model instance."""
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
    """Applies EMBEDDING_QUERY_PREFIX if configured."""
    if EMBEDDING_QUERY_PREFIX:
        return f"{EMBEDDING_QUERY_PREFIX}{query}"
    return query


def prepare_document_for_embedding(document: str) -> str:
    """Applies EMBEDDING_DOC_PREFIX if configured (symmetric with query prefix)."""
    if EMBEDDING_DOC_PREFIX:
        return f"{EMBEDDING_DOC_PREFIX}{document}"
    return document


# =============================================================================
# run_pipeline
# =============================================================================

def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 60)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    # Stats: count by type
    type_counts = {}
    for d in docs:
        t = d["metadata"]["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in type_counts.items():
        print(f"    - {t}: {c}")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    # Stats: min/avg/max chunk length
    if chunks:
        lengths = [len(c["content"]) for c in chunks]
        print(f"    - Length: min={min(lengths)}, avg={sum(lengths)//len(lengths)}, max={max(lengths)}")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")

    # Verify
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(COLLECTION_NAME)
    print(f"\n✓ ChromaDB collection '{COLLECTION_NAME}' has {col.count()} chunks")


if __name__ == "__main__":
    run_pipeline()