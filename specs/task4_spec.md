# Task 4 Spec — Chunking & Indexing into Vector Store

## 1. Overview

Task 4 is the **data indexing layer** of the RAG pipeline. It converts raw markdown documents from `data/standardized/` into searchable vector embeddings stored in a persistent ChromaDB collection.

**Role in the pipeline:**
```
data/standardized/  →  [load_documents]  →  [chunk_documents]  →  [embed_chunks]  →  [index_to_vectorstore]  →  chroma_db/
     (raw .md)                                                                                                        (vector store)
```

**Who implements this:** Role 2 (Data & Dense Search Dev) — Checkpoint 2

**How to run:** `python src/task4_chunking_indexing.py`

---

## 2. Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      run_pipeline()                                 │
│                                                                     │
│   1. load_documents()                                               │
│      └─ Scans data/standardized/ for all .md files                  │
│      └─ Returns list of {content, metadata} dicts                   │
│                                                                     │
│   2. chunk_documents(docs)                                          │
│      └─ Splits each document into smaller chunks                   │
│      └─ Uses RecursiveCharacterTextSplitter                         │
│      └─ Returns list of {content, metadata} dicts (one per chunk)   │
│                                                                     │
│   3. embed_chunks(chunks)                                           │
│      └─ Encodes each chunk text into a vector embedding             │
│      └─ Uses sentence-transformers (BAAI/bge-m3)                    │
│      └─ Adds key "embedding": list[float] to each chunk dict        │
│                                                                     │
│   4. index_to_vectorstore(chunks)                                   │
│      └─ Creates/opens persistent ChromaDB at chroma_db/             │
│      └─ Upserts chunks with embeddings, metadata, unique IDs        │
│                                                                     │
│   5. Prints summary to console                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### ⚠️ Critical: Clean Before Reindex

If you later change the corpus (add/remove documents, change topic), **you MUST delete `chroma_db/`** before re-running. Otherwise old and new chunks co-exist in the same collection, producing garbage retrieval results.

```bash
rm -rf chroma_db/
python src/task4_chunking_indexing.py
```

---

## 3. Interface & API Contract

### 3.1 Configuration Constants (module-level)

| Constant | Type | Default | Description |
|---|---|---|---|
| `STANDARDIZED_DIR` | `Path` | `data/standardized/` | Input directory for markdown files |
| `CHROMA_DIR` | `Path` | `chroma_db/` | Output directory for ChromaDB persistence |
| `CHUNK_SIZE` | `int` | 800 | Max characters per chunk |
| `CHUNK_OVERLAP` | `int` | 100 | Overlap characters between consecutive chunks |
| `CHUNKING_METHOD` | `str` | `"recursive"` | Strategy: `"recursive"` | `"markdown_header"` | `"semantic"` |
| `EMBEDDING_MODEL` | `str` | `"BAAI/bge-m3"` | Sentence-transformer model name |
| `EMBEDDING_DIM` | `int` | 1024 | Output dimension of the embedding model |
| `VECTOR_STORE` | `str` | `"chromadb"` | Vector store choice: `"chromadb"` | `"weaviate"` | `"faiss"` |
| `COLLECTION_NAME` | `str` | `"university_services_docs"` | ChromaDB collection name |

**Constraints (enforced by tests):**
- `CHUNK_SIZE > 0`
- `CHUNK_OVERLAP > 0`
- `CHUNK_OVERLAP < CHUNK_SIZE`

### 3.2 Function: `load_documents()`

```python
def load_documents() -> list[dict]:
    """
    Reads all .md files from data/standardized/ recursively.

    Returns:
        List of dicts, each with:
            "content": str   — full text of the markdown file
            "metadata": dict — containing:
                "source": str  — filename (e.g. "tuition-policy.md")
                "type": str    — "legal" if path contains "legal", else "news"
    """
```

**Behavior:**
- Iterate recursively through `STANDARDIZED_DIR.rglob("*.md")`
- Read each file with `utf-8` encoding
- Infer `type` from parent directory name: `"legal"` if `"legal"` in `str(md_file)`, else `"news"`
- Return empty list if no .md files found (test expects list, not None)

### 3.3 Function: `chunk_documents(documents: list[dict]) -> list[dict]`

```python
def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Splits documents into chunks using the configured strategy.

    Args:
        documents: output from load_documents()

    Returns:
        List of dicts, each with:
            "content": str   — chunk text
            "metadata": dict — inherits parent metadata PLUS:
                "chunk_index": int  — sequential index within the source document
    """
```

**Behavior:**
- Use `RecursiveCharacterTextSplitter` from `langchain_text_splitters` with:
  - `chunk_size=CHUNK_SIZE`
  - `chunk_overlap=CHUNK_OVERLAP`
  - `separators=["\n\n", "\n", ". ", " ", ""]`
- Each chunk content length must not exceed `CHUNK_SIZE * 1.1` (10% tolerance)
- Preserve parent metadata and add `chunk_index` to each chunk

### 3.4 Function: `embed_chunks(chunks: list[dict]) -> list[dict]`

```python
def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Encodes chunk texts into vector embeddings.

    Args:
        chunks: output from chunk_documents()

    Returns:
        Same list of dicts, each augmented with:
            "embedding": list[float]  — vector representation of the chunk text
    """
```

**Behavior:**
- Use `SentenceTransformer(EMBEDDING_MODEL)` from `sentence_transformers`
- Call `model.encode(texts, show_progress_bar=True)` on all chunk texts
- Convert each embedding to `list[float]` via `.tolist()`
- The model `BAAI/bge-m3` produces 1024-dimensional vectors

### 3.5 Function: `index_to_vectorstore(chunks: list[dict])`

```python
def index_to_vectorstore(chunks: list[dict]) -> None:
    """
    Upserts chunks with embeddings into the persistent ChromaDB collection.

    Args:
        chunks: output from embed_chunks() (each dict has content, metadata, embedding)

    Returns:
        None. Side effect: writes to chroma_db/ directory.
    """
```

**Behavior:**
- Create `CHROMA_DIR` if it doesn't exist: `CHROMA_DIR.mkdir(parents=True, exist_ok=True)`
- Initialize `chromadb.PersistentClient(path=str(CHROMA_DIR))`
- Get or create collection with:
  - `name=COLLECTION_NAME`
  - `metadata={"hnsw:space": "cosine"}`
- Generate unique IDs: `f"{metadata['source']}_chunk_{chunk_index}"`
- Call `collection.upsert(ids=ids, documents=[...], embeddings=[...], metadatas=[...])`

### 3.6 Function: `run_pipeline()`

```python
def run_pipeline() -> None:
    """
    Orchestrates the full pipeline: load → chunk → embed → index.
    Prints progress to stdout.
    """
```

---

## 4. Implementation Details

### 4.1 Chunking Strategy: RecursiveCharacterTextSplitter (Recommended)

**Why:** It's the safest, most general-purpose splitter. It tries to break at paragraph boundaries (`\n\n`) first, then sentence boundaries (`. `), then word boundaries, and finally character-by-character if needed. This preserves semantic units as much as possible.

**Alternatives considered:**
- **MarkdownHeaderTextSplitter** — Better for documents with consistent heading structure, but our corpus mixes legal docs and news articles with varying heading patterns.
- **SemanticChunker** — Uses embeddings to find natural breakpoints, which is more accurate but adds latency and is overkill for this lab.

**Configuration:**
- `CHUNK_SIZE = 800` — Balances between context completeness (LLM needs ~500+ chars for meaningful answers) and retrieval granularity (smaller chunks = more precise matches)
- `CHUNK_OVERLAP = 100` — 12.5% overlap prevents information loss at chunk boundaries (e.g., a sentence that spans the split point)

### 4.2 Embedding Model: BAAI/bge-m3 (Recommended)

**Why:** Multilingual model (1024-dim) that handles both Vietnamese and English well. This is critical because the university domain documents may contain both languages.

**Alternatives considered:**
- `all-MiniLM-L6-v2` (384-dim) — Faster and lighter, but English-only. Would lose Vietnamese semantics.
- `text-embedding-3-small` (1536-dim) — High quality via API, but requires internet + API key + adds latency/cost.

### 4.3 Vector Store: ChromaDB (Recommended)

**Why:** Persistent, local, zero-Docker-dependency. The `PersistentClient` mode writes to disk so the index survives restarts.

**Alternatives considered:**
- **Weaviate** — Built-in hybrid search, but requires Docker container.
- **FAISS** — Fast dense search only, no built-in metadata filtering or persistence.

### 4.4 Document Metadata Mapping

| Source Directory | `type` field | Example |
|---|---|---|
| `data/standardized/legal/` | `"legal"` | `{"source": "tuition-policy.md", "type": "legal"}` |
| `data/standardized/news/` | `"news"` | `{"source": "scholarship-announcement.md", "type": "news"}` |

---

## 5. Verification

### 5.1 Run the pipeline

```bash
python src/task4_chunking_indexing.py
```

Expected output:
```
==================================================
Task 4: Chunking & Indexing
  Chunking: recursive (size=800, overlap=100)
  Embedding: BAAI/bge-m3 (dim=1024)
  Vector Store: chromadb
==================================================

✓ Loaded N documents
✓ Created M chunks
✓ Embedded M chunks
✓ Indexed to vector store
```

### 5.2 Run the tests

```bash
pytest tests/test_individual.py::TestTask4 -v
```

Expected: **all 5 tests pass** (7 points total)

| Test | What it checks |
|---|---|
| `test_config_documented` | `CHUNK_SIZE > 0`, `CHUNK_OVERLAP > 0`, `CHUNK_OVERLAP < CHUNK_SIZE` |
| `test_load_documents_returns_list` | `load_documents()` returns a list of dicts with `"content"` key |
| `test_chunk_documents_produces_chunks` | `chunk_documents()` produces chunks with `"content"` key from 1 document |
| `test_chunks_respect_size_limit` | Each chunk content ≤ `CHUNK_SIZE * 1.1` |
| *(implicit)* | `chroma_db/` directory is created and populated |

### 5.3 Manual checks

```bash
# Verify chroma_db/ was created
ls -la chroma_db/

# Quick sanity: query the collection
python -c "
import chromadb
client = chromadb.PersistentClient(path='chroma_db')
col = client.get_collection('university_services_docs')
print(f'Collection contains {col.count()} chunks')
"
```

---

## 6. Edge Cases & Notes

| Scenario | Expected Behavior |
|---|---|
| `data/standardized/` is empty | `load_documents()` returns `[]`; pipeline prints "Loaded 0 documents" |
| File encoding is not UTF-8 | Use `utf-8` encoding; if other encoding is suspected, handle with `errors='replace'` |
| Re-running after data changes | **Must delete `chroma_db/` first** — otherwise old chunks persist |
| Document shorter than CHUNK_SIZE | Single chunk, no splitting needed |
| Very long document without paragraph breaks | `RecursiveCharacterTextSplitter` falls back to character-level splitting |
| Embedding model downloads on first run | First run downloads ~2GB model; subsequent runs use cache |

### Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'langchain_text_splitters'` | Missing dependency | `pip install langchain-text-splitters` |
| `ModuleNotFoundError: No module named 'sentence_transformers'` | Missing dependency | `pip install sentence-transformers` |
| `ModuleNotFoundError: No module named 'chromadb'` | Missing dependency | `pip install chromadb` |
| OSError / HF model download failure | Network issue or model not found | Check internet; verify model name `BAAI/bge-m3` |
| ChromaDB lock error | Another process using chroma_db/ | Close other Python processes, delete chroma_db/, retry |