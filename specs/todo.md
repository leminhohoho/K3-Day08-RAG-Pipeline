# Implementation Todo — RAG Pipeline v2

Tổng hợp các đầu việc cần triển khai từ tất cả spec files. Mỗi task có checklist riêng, pass criteria, và test target.

---

## Task 4 — Chunking & Indexing

**File:** `src/task4_chunking_indexing.py`  
**Role:** Role 2 (Data & Dense Search Dev) — Checkpoint 2  
**Test target:** `pytest tests/test_individual.py::TestTask4 -v` (5 tests, 7 điểm)  
**Spec:** `specs/task4_spec.md`

### Config
- [ ] Set `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, `CHUNKING_METHOD="recursive"`
- [ ] Set `EMBEDDING_MODEL="BAAI/bge-m3"`, `EMBEDDING_DIM=1024`
- [ ] Set `VECTOR_STORE="chromadb"`, `COLLECTION_NAME="university_services_docs"`
- [ ] Set `EMBEDDING_QUERY_PREFIX=""`, `EMBEDDING_DOC_PREFIX=""`

### load_documents
- [ ] Scan `data/standardized/` recursively for `.md` files
- [ ] Derive `document_id` from filename stem, `title` from stem or first H1
- [ ] Infer `type` from parent directory name (`"legal"` / `"news"`)
- [ ] Set `url=""`, `section=""`, `language="vi"` as defaults
- [ ] Return `[]` when no files found

### chunk_documents
- [ ] Use `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)`
- [ ] Each chunk must have `content`, `metadata` (inherited + `chunk_index`, `chunk_id`)
- [ ] `chunk_id` format: `f"{document_id}_chunk_{chunk_index}"`
- [ ] Content length ≤ `CHUNK_SIZE * 1.1` (800 × 1.1 = 880)

### embed_chunks
- [ ] Use OpenRouter API (via OpenAI SDK) for embedding
- [ ] Load `OPENROUTER_API_KEY` from `.env`
- [ ] Embedding model: `BAAI/bge-m3` via OpenRouter, dim=1024
- [ ] Batch embed chunks (max 20 per batch)
- [ ] Add `"embedding": list[float]` to each chunk dict
- [ ] Fallback to `SentenceTransformer` if OpenRouter fails

### index_to_vectorstore
- [ ] Create `chroma_db/` directory if missing
- [ ] Use `chromadb.PersistentClient(path=str(CHROMA_DIR))`
- [ ] Get or create collection with `hnsw:space=cosine`
- [ ] Upsert with IDs from `chunk_id` metadata field
- [ ] Store documents, embeddings, metadatas

### Shared Helpers (exported for Task 5)
- [ ] `get_embedding_model()` — lazy-load, cached instance
- [ ] `get_collection()` — open persistent collection, raise on dimension mismatch
- [ ] `prepare_query_for_embedding(query)` — apply `EMBEDDING_QUERY_PREFIX`

### Verification
- [ ] Run `python src/task4_chunking_indexing.py` — prints summary
- [ ] `pytest tests/test_individual.py::TestTask4 -v` — 5 passed
- [ ] Manual: `chroma_db/` created, collection has chunks

---

## Task 5 — Semantic Search & Query Expansion

**File:** `src/task5_semantic_search.py`, `src/task5_query_expansion.py`  
**Role:** Role 3 (Sparse Search Dev / UI Dev) — Checkpoint 2  
**Test target:** `pytest tests/test_individual.py::TestTask5 -v` (4 tests, 6 điểm + 5 bonus)  
**Spec:** `specs/task5_spec.md`

### Core
- [ ] Reuse Task 4 helpers: `get_embedding_model()`, `get_collection()`, `prepare_query_for_embedding()`
- [ ] `semantic_search(query, top_k=10)` — starter signature
- [ ] Normalize query: NFC, trim, collapse whitespace, preserve diacritics
- [ ] Embed query with same model/config as indexing
- [ ] Query ChromaDB with cosine distance
- [ ] Convert distance to similarity: `1.0 - distance`, clamp `[0, 1]`
- [ ] Sort descending by score, stable tie-break
- [ ] Deduplicate by `chunk_id`
- [ ] Return `[]` for empty query, empty collection, `top_k <= 0`

### Output Contract
- [ ] Each result: `content`, `score` (float), `metadata` (with `chunk_id`, `source`, etc.)
- [ ] Score is Python float, not NumPy scalar
- [ ] Not more than `top_k` results
- [ ] Metadata missing fields → use defaults, don't crash

### Caching & Performance
- [ ] Embedding model: one instance per process
- [ ] Chroma client/collection: cached by path + name
- [ ] Query embedding: LRU cache by model ID + normalized query
- [ ] No permanent exception caching

### Query Expansion (Bonus, 5 điểm)
- [ ] `expand_query(query, max_variants=3)` — deterministic, no API key required
- [ ] Original query always first element
- [ ] Domain glossary for bilingual rewrites (tuition fee ↔ học phí, etc.)
- [ ] `semantic_search_expanded()` — search each variant, RRF fusion by chunk_id
- [ ] Preserve `raw_scores.dense` (best raw cosine) for Task 9 calibration
- [ ] Optional LLM expansion with fallback to deterministic

### Error Handling
- [ ] Empty query → `[]`; `top_k <= 0` → `[]`
- [ ] Collection missing/empty → `[]` with warning
- [ ] Model dimension mismatch → raise clear config error
- [ ] Expansion API fails → use deterministic expansion

### Verification
- [ ] `pytest tests/test_individual.py::TestTask5 -v` — 4 passed
- [ ] In-domain query returns evidence in top 5
- [ ] Vietnamese/English equivalent queries share common document
- [ ] OOD query has low best cosine score

---

## Task 6 — Lexical Search (BM25 & TF-IDF)

**File:** `src/task6_lexical_search.py`  
**Role:** Role 4 (Evaluation & QA Engineer) — Checkpoint 2  
**Test target:** `pytest tests/test_individual.py::TestTask6 -v` (4 tests, 6 điểm + 5 bonus)  
**Spec:** `specs/task6_spec.md`

### Corpus
- [ ] Load chunks from ChromaDB collection (Task 4) or fall back to `load_documents()` + `chunk_documents()`
- [ ] Same `chunk_id` and metadata schema as Task 4/5
- [ ] Corpus managed via lazy index manager (no static empty `CORPUS`)

### Text Normalization
- [ ] `normalize_lexical_text()` — NFC, lowercase, collapse whitespace
- [ ] `tokenize_lexical()` — Unicode-aware regex `\w+(?:[-./]\w+)*`
- [ ] Preserve numbers, codes, Vietnamese diacritics
- [ ] Optional de-accent alias (not replacing original tokens)

### BM25 (Core)
- [ ] `build_bm25_index(corpus)` — lazy, cached by corpus hash
- [ ] `lexical_search(query, top_k=10)` — starter signature
- [ ] Use `rank_bm25.BM25Okapi(k1=1.5, b=0.75)`
- [ ] Only return results with positive BM25 score
- [ ] Sort descending, stable tie-break (score → exact match count → doc_id → chunk_index)
- [ ] Return `[]` for empty query, empty corpus, `top_k <= 0`

### TF-IDF (Bonus, 5 điểm)
- [ ] `build_tfidf_index(corpus)` — `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))`
- [ ] `tfidf_search(query, top_k=10)` — cosine similarity, clamp `[0, 1]`
- [ ] `lexical_search_configured(query, method="bm25"|"tfidf"|"bm25_tfidf")`
- [ ] BM25+TF-IDF fusion via RRF, not direct score addition

### Error Handling
- [ ] Empty/unavailable → `[]`, not `NotImplementedError`
- [ ] Empty chunk after tokenization → skip, fix index mapping
- [ ] Invalid method → `ValueError`

### Verification
- [ ] `pytest tests/test_individual.py::TestTask6 -v` — 4 passed
- [ ] Keyword query returns evidence with positive BM25 score
- [ ] TF-IDF returns `[0,1]` scores
- [ ] BM25+TF-IDF dedupes by `chunk_id`

---

## Task 7 — Reranking (RRF Weighted Fusion)

**File:** `src/task7_reranking.py`  
**Role:** Role 4 (Sparse Search & Advanced Reranking Dev) — Checkpoint 3  
**Test target:** `pytest tests/test_individual.py::TestTask7 -v` (3 tests, 6 điểm)  
**Spec:** `specs/task7_spec.md`

### RRF (Core)
- [ ] `rerank_rrf(ranked_lists, top_k=5, k=60, weights=None)` — weighted RRF
- [ ] Correct formula: `RRF(d) = Σ weightᵣ / (k + rankᵣ(d))`, rank starts at 1
- [ ] Deduplicate by `chunk_id` (fallback: content hash + source)
- [ ] Merge `raw_scores` and `provenance` from all occurrences
- [ ] Set `score_type="rrf"`, `score=rrf_score`, `confidence_score` = max dense cosine
- [ ] Tie-break deterministic: ranker count → best dense → best native rank → chunk_id
- [ ] Do NOT mutate input objects
- [ ] Return `[]` for empty lists, `top_k <= 0`

### Cross-encoder (Optional)
- [ ] `rerank_cross_encoder(query, candidates, top_k=5)` — adapter pattern
- [ ] Timeout (15–30s), retry (1× for 429/5xx), fallback to prior order
- [ ] Cache by query + candidate hashes per process
- [ ] Log `reranker_fallback_reason` on failure

### MMR (Optional)
- [ ] `rerank_mmr(query_embedding, candidates, top_k=5, lambda_param=0.7)`
- [ ] Raise `ValueError` on missing/dimension-mismatched embeddings
- [ ] Do not select same chunk twice

### Unified `rerank()`
- [ ] `rerank(query, candidates, top_k, method="rrf")` — starter signature
- [ ] `"rrf"`: calls `rerank_rrf([candidates], top_k)` (single-list rank normalization)
- [ ] `"cross_encoder"` / `"mmr"` / `"none"`: dispatch accordingly
- [ ] Unknown method → `ValueError`

### Verification
- [ ] `pytest tests/test_individual.py::TestTask7 -v` — 3 passed
- [ ] RRF fusion: dense + BM25 + TF-IDF → single merged list
- [ ] Raw scores preserved after fusion
- [ ] Double-RRF not possible (Task 9 uses `RERANK_METHOD="none"` by default)

---

## Task 8 — PageIndex Vectorless Fallback

**File:** `src/task8_pageindex_vectorless.py`  
**Role:** Role 3 (Frontend & Chatbot Dev) — Checkpoint 3  
**Test target:** `pytest tests/test_individual.py::TestTask8 -v` (2 tests, 4 điểm)  
**Spec:** `specs/task8_spec.md`

### Setup
- [ ] Register account at pageindex.ai, get API key
- [ ] Pin tested SDK version, verify import path
- [ ] Set `PAGEINDEX_API_KEY` in `.env`

### Document Registry
- [ ] Load/update `pageindex_doc_ids.json` with `document_id` → `pageindex_doc_id` mapping
- [ ] Track checksum, status, `retrieval_ready` flag
- [ ] Atomic writes (temp file → replace)
- [ ] Skip upload if checksum unchanged and ready

### Upload
- [ ] `upload_documents(force=False, wait_until_ready=True)` — starter signature
- [ ] Prefer original PDF from `data/landing/legal/`; convert Markdown to Unicode PDF
- [ ] Submit via PageIndex SDK; poll processing status with deadline/backoff
- [ ] Do NOT upload in query path

### Search
- [ ] `pageindex_search(query, top_k=5)` — starter signature
- [ ] Shortlist ready documents (max 3 per query) via file-level metadata
- [ ] Submit retrieval jobs, poll with shared deadline
- [ ] Parse response: flatten `retrieved_nodes` → `relevant_contents`
- [ ] Assign `score_type="rank_proxy"`, `score = 1.0 / global_rank`
- [ ] Return `[]` on missing key, no ready docs, API errors (never crash)

### Output Contract
- [ ] `source="pageindex"`, `score_type="rank_proxy"`, `confidence_score=None`
- [ ] Provenance: `document_id`, `source`, `section`, `node_id`, `page_index`
- [ ] Chunk ID: `f"pageindex:{doc_id}:{node_id}:{content_hash}"`

### Cache & Resilience
- [ ] In-memory TTL cache for query results (TTL: 3600s)
- [ ] Circuit breaker: after N consecutive failures, return `[]` temporarily
- [ ] Cache hit → `metadata.cache_hit=true`
- [ ] Do NOT fabricate results under `source="pageindex"`

### Verification
- [ ] `pytest tests/test_individual.py::TestTask8 -v` — 2 passed
- [ ] At least 3 documents uploaded and retrieval-ready
- [ ] Live query returns non-empty evidence with `source="pageindex"`
- [ ] Missing API key → `[]`, not crash

---

## Task 9 — Retrieval Pipeline (Hybrid + Fallback)

**File:** `src/task9_retrieval_pipeline.py`  
**Role:** Role 2 (Data & Pipeline Specialist) — Checkpoint 4  
**Test target:** `pytest tests/test_individual.py::TestTask9 -v` (4 tests, 7 điểm)  
**Spec:** `specs/task9_spec.md`

### Configuration
- [ ] `SCORE_THRESHOLD` — calibrate by measuring in-domain vs OOD cosine scores
- [ ] `DEFAULT_TOP_K=5`, `RERANK_METHOD="none"` (default, avoid double-RRF)

### Step 1: Parallel Retrieval
- [ ] `dense_results = semantic_search(query, top_k=top_k*2)` — Task 5
- [ ] `sparse_results = lexical_search(query, top_k=top_k*2)` — Task 6

### Step 2: RRF Merge
- [ ] `merged = rerank_rrf([dense_results, sparse_results], top_k=top_k*2)` — Task 7
- [ ] Tag each result: `item["source"] = "hybrid"`

### Step 3: Optional Reranking
- [ ] If `use_reranking=True and RERANK_METHOD != "none"`:
  - [ ] `rerank(query, merged, top_k=top_k, method=RERANK_METHOD)` — Task 7
  - [ ] Fallback to `merged[:top_k]` if `rerank()` raises `NotImplementedError`
- [ ] Else: `final = merged[:top_k]`

### Step 4: Fallback Check
- [ ] `best_cosine = dense_results[0]["score"]` if non-empty, else `0.0`
- [ ] If `best_cosine < score_threshold`:
  - [ ] `fallback = pageindex_search(query, top_k=top_k)` — Task 8
  - [ ] If fallback non-empty: return fallback results
  - [ ] Else: return `[]`
- [ ] ⚠️ Use raw cosine, NOT RRF score for threshold

### Output Contract
- [ ] Each result: `content`, `score`, `score_type`, `confidence_score`, `metadata`, `source`, `raw_scores`
- [ ] `source` is `"hybrid"` or `"pageindex"`
- [ ] `len(results) <= top_k`
- [ ] Never crash on obscure queries

### Verification
- [ ] `pytest tests/test_individual.py::TestTask9 -v` — 4 passed
- [ ] Calibrate threshold: in-domain vs OOD cosine score distribution
- [ ] Fallback triggers correctly for nonsense queries
- [ ] `python src/task9_retrieval_pipeline.py` — runs without error

---

## Task 10 — Generation with Citation

**File:** `src/task10_generation.py`  
**Role:** Role 3 (Frontend & Chatbot Dev) — Checkpoint 4  
**Test target:** `pytest tests/test_individual.py::TestTask10 -v` (4 tests, 4 điểm)  
**Spec:** `specs/task10_spec.md`

### Configuration
- [ ] `TOP_K=5`, `TOP_P=0.9`, `TEMPERATURE=0.3`
- [ ] `LLM_MODEL="openai/gpt-4o-mini"` (or `:free` model)
- [ ] `OPENROUTER_API_KEY` in `.env`

### reorder_for_llm
- [ ] Implement `front + back[::-1]` reordering
- [ ] Best chunk first, worst in middle, second-best last
- [ ] Preserve all values (no duplicates, no missing)
- [ ] Return as-is if `len <= 2`

### format_context
- [ ] Label each chunk: `[Document {i} | Source: {source} | Type: {type}]`
- [ ] Separate chunks with `\n---\n`
- [ ] Fallback: `source="Source {i}"`, `type="unknown"` if metadata missing

### generate_with_citation
- [ ] Call `retrieve(query, top_k=top_k)` — Task 9
- [ ] If no chunks: return no-answer response immediately
- [ ] Reorder → format → build prompt (system + context + question)
- [ ] Call OpenRouter (OpenAI-compatible SDK): `client.chat.completions.create()`
- [ ] Return `{"answer": str, "sources": list, "retrieval_source": str}`

### System Prompt
- [ ] Vietnamese instructions
- [ ] No hallucination, citation required, structured paragraphs
- [ ] No-answer fallback: "Tôi không thể xác minh thông tin này từ nguồn hiện có"

### Error Handling
- [ ] Missing API key → exception propagates (visible to developer)
- [ ] LLM timeout/error → exception propagates
- [ ] Empty response from LLM → return as-is

### Verification
- [ ] `pytest tests/test_individual.py::TestTask10 -v` — 4 passed
- [ ] `python src/task10_generation.py` — prints answer with citations
- [ ] Manual: answer contains source references like `[Source, Section]`

---

## Chat UI — Streamlit Frontend

**File:** `app.py`  
**Role:** Role 3 (Frontend & Chatbot Developer) — Checkpoint 5  
**Run:** `streamlit run app.py`  
**Spec:** `specs/chat_ui_spec.md`

### Page Setup
- [ ] `st.set_page_config(title="University Services RAG Chatbot", icon="🎓", layout="wide")`
- [ ] Initialize session state: `messages=[]`, `pending_query=None`

### Sidebar
- [ ] Title + caption (University Services RAG, trợ lý hỏi đáp)
- [ ] 5 suggested question buttons → set `st.session_state.pending_query`
- [ ] `top_k` slider (3–10, default 5)
- [ ] Architecture info text

### Main Chat Area
- [ ] Header: "🎓 University Services RAG Chatbot"
- [ ] Render chat history: user + assistant messages
- [ ] Assistant messages with sources: expander "📚 Nguồn tham khảo (N chunks)"
- [ ] Source display: `[{i}] {source_name} {doc_type} | score: {score:.4f}` + content preview

### Query Handling
- [ ] `st.chat_input()` with placeholder
- [ ] Process both typed input and `pending_query`
- [ ] Call `generate_with_citation(query, top_k=top_k)` — Task 10
- [ ] Show spinner: "Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."

### Error Handling
- [ ] `NotImplementedError` → "⚠️ Task 10 chưa được implement..."
- [ ] `Exception` → "❌ Lỗi khi chạy RAG Pipeline: {e}"
- [ ] Empty sources → no expander shown

### Verification
- [ ] `streamlit run app.py` — loads without errors
- [ ] Type question → answer + sources appear
- [ ] Click suggested question → triggers query
- [ ] Adjust `top_k` → affects retrieval count
- [ ] Multiple chat turns → history preserved
- [ ] **CP5 Passed:** Chatbot UI phản hồi chính xác kèm danh sách nguồn

---

---

## Summary

| Task | File | Tests | Points | Role | Checkpoint |
|---|---|---|---|---|---|
| **4** | `task4_chunking_indexing.py` | 5 tests | 7 | Role 2 | CP2 |
| **5** | `task5_semantic_search.py` | 4 tests | 6 + 5 bonus | Role 3 | CP2 |
| **6** | `task6_lexical_search.py` | 4 tests | 6 + 5 bonus | Role 4 | CP2 |
| **7** | `task7_reranking.py` | 3 tests | 6 | Role 4 | CP3 |
| **8** | `task8_pageindex_vectorless.py` | 2 tests | 4 | Role 3 | CP3 |
| **9** | `task9_retrieval_pipeline.py` | 4 tests | 7 | Role 2 | CP4 |
| **10** | `task10_generation.py` | 4 tests | 4 | Role 3 | CP4 |
| **UI** | `app.py` | Manual | — | Role 3 | CP5 |
| **Eval** | `group_project/evaluation/` | Manual | — | Role 4/5/6 | CP5 |
| | **Total** | **35 tests** | **50** | | |