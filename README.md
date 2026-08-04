---
title: University Services RAG Chatbot
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# Ngày 8 — RAG Pipeline v2

**Chương 2 | Ngày 8 trong 15**

> Dùng chung chủ đề "University Services" với biến thể K3 của Ngày 7 (`K3_VARIANT.md`), để pipeline Ngày 7 → Ngày 8 nhất quán.

---

## Mục Tiêu

Xây dựng một RAG pipeline thực tế, end-to-end, từ thu thập dữ liệu chính sách và thông tin dịch vụ đại học → xử lý → indexing → retrieval (hybrid + vectorless fallback) → generation có citation.

---

## Chủ Đề Dữ Liệu

**Chính sách/quy định dịch vụ đại học** (học phí, học bổng, ký túc xá, đăng ký học phần) + **Thông tin/thông báo đại học** (sự kiện, dịch vụ thư viện, hỗ trợ sinh viên)

Dữ liệu hiện tại trong repo được thu thập từ các nguồn công khai chính thức của **Trường Đại học Khoa học Xã hội và Nhân văn, ĐHQGHN** và **Đại học Quốc gia Hà Nội** — xem URL, metadata và checksum trong `data/sources_manifest.json`.

---

## Cấu Trúc Thư Mục

```
K3-Day08-RAG-Pipeline-Starter/
├── README.md
├── LAB_GUIDE.md           ← Hướng dẫn chi tiết & Codelab
├── checkpoint_timer.html  ← Dashboard đếm ngược Checkpoint & Phân vai
├── app.py                 ← Streamlit chatbot (bài nhóm)
├── data/
│   ├── landing/           ← Task 1 & 2: raw files (PDF, JSON)
│   └── standardized/      ← Task 3: converted markdown files
├── src/
│   ├── __init__.py
│   ├── task1_collect_legal_docs.py
│   ├── task2_crawl_news.py
│   ├── task3_convert_markdown.py
│   ├── task4_chunking_indexing.py
│   ├── task5_semantic_search.py
│   ├── task6_lexical_search.py
│   ├── task7_reranking.py
│   ├── task8_pageindex_vectorless.py
│   ├── task9_retrieval_pipeline.py
│   ├── task10_generation.py
│   └── supervisor.py      ← Pattern nâng cao: Supervisor + Workers song song
├── chroma_db/             ← Task 4: vector store đã index (sinh ra khi chạy, không tự viết tay)
├── tests/
│   └── test_individual.py ← Chấm điểm phần Task 1-10 (pytest)
├── group_project/
│   ├── README.md          ← Hướng dẫn bài tập nhóm
│   └── evaluation/        ← golden_dataset.json, eval_pipeline.py, results.md
├── requirements.txt
└── .env.example
```

---

## Nhiệm Vụ Chi Tiết

### Task 1 — Thu Thập Văn Bản Chính Sách Đại Học

Tìm và tải về **tối thiểu 3 văn bản chính sách/quy định** dạng PDF/DOCX về dịch vụ đại học (học phí, học bổng, ký túc xá, đăng ký học phần). Lưu vào `data/landing/`.

**Gợi ý nguồn** (ví dụ trang công khai Trường Đại học Khoa học Xã hội và Nhân văn, ĐHQGHN — USSH):
- Học phí & phương thức thanh toán (Tuition Fees)
- Chính sách học bổng (Scholarship eligibility)
- Quy định ký túc xá / hỗ trợ chỗ ở (Accommodation Services)
- Cổng đăng ký học phần (Course Registration Portal)

**Yêu cầu:**
- Lưu file gốc (PDF/DOCX) vào `data/landing/legal/`
- Đặt tên file rõ ràng: `NhanVan_HocPhi.pdf`, `NhanVan_HocBong.pdf`, ...

---

### Task 2 — Crawl Bài Viết/Thông Báo

Crawl **tối thiểu 5 bài viết** về thông tin/thông báo dịch vụ đại học (sự kiện, thư viện, hỗ trợ sinh viên, học bổng).

**Thư viện khuyến nghị:** [Crawl4AI](https://github.com/unclecode/crawl4ai)

**Yêu cầu:**
- Lưu output vào `data/landing/news/`
- Mỗi bài báo lưu thành 1 file (JSON hoặc HTML)
- Ghi rõ metadata: URL gốc, ngày crawl, tiêu đề bài báo

**Code mẫu (Crawl4AI):**
```python
from crawl4ai import AsyncWebCrawler

async def crawl_article(url: str, output_dir: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        # Lưu result.markdown vào file
        ...
```

---

### Task 3 — Convert Sang Markdown

Sử dụng [MarkItDown](https://github.com/microsoft/markitdown) của Microsoft để convert toàn bộ file trong `data/landing/` thành Markdown.

**Cài đặt:**
```bash
pip install markitdown
```

**Code mẫu:**
```python
from markitdown import MarkItDown

md = MarkItDown()

# Convert PDF
result = md.convert("data/landing/legal/NhanVan_HocPhi.pdf")
print(result.text_content)

# Convert DOCX
result = md.convert("data/landing/legal/NhanVan_HocBong.pdf")
```

**Lưu ý:** MarkItDown cần cài thêm extra `pip install "markitdown[pdf]"` để convert được file
PDF — nếu chỉ `pip install markitdown` sẽ báo lỗi `MissingDependencyException` khi convert PDF.

**Yêu cầu:**
- Output lưu vào `data/standardized/`
- Giữ nguyên cấu trúc thư mục con (`legal/`, `news/`)
- Mỗi file output có tên tương ứng: `NhanVan_HocPhi.md`

---

### Task 4 — Chunking & Indexing

Sử dụng **fixed-size chunking** (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`) và **`BAAI/bge-m3`** (dim=1024) để index toàn bộ markdown files vào ChromaDB.

**Cấu hình (bắt buộc):**

| Constant | Giá trị | Lý do |
|---|---:|---|
| `CHUNK_SIZE` | `800` | Đủ ngữ cảnh cho văn bản pháp lý nhưng vẫn giữ retrieval chính xác |
| `CHUNK_OVERLAP` | `100` | Overlap 12.5% để hạn chế mất ý ở biên chunk |
| `CHUNKING_METHOD` | `"fixed_size"` | Chia thành các đoạn có độ dài cố định, không giữ cấu trúc heading |
| `EMBEDDING_MODEL` | `"BAAI/bge-m3"` | Multilingual, phù hợp tiếng Việt và tiếng Anh |
| `EMBEDDING_DIM` | `1024` | Kích thước output của `BAAI/bge-m3` |
| `VECTOR_STORE` | `"chromadb"` | Chạy local, persistent, có metadata filtering |
| `COLLECTION_NAME` | `"university_services_docs"` | Một collection thống nhất cho corpus |

**Chunking — dùng `CharacterTextSplitter` từ [langchain-text-splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/):**
```bash
pip install langchain-text-splitters
```

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separator="\n\n",
)
```

**Embedding — dùng OpenRouter API (OpenAI-compatible SDK) hoặc SentenceTransformer local:**
```python
from openai import OpenAI
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
```

**Vector Store — ChromaDB với cosine distance:**
```bash
pip install chromadb
```

```python
import chromadb
client = chromadb.PersistentClient(path="chroma_db/")
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
```

**Pipeline chính (trong `src/task4_chunking_indexing.py`):**

1. `load_documents()` — đọc `*.md` từ `data/standardized/`, parse metadata header, trả body
2. `chunk_documents(documents)` — cắt fixed-size 800/100, gắn `chunk_id`, `section`, `chunk_index`
3. `embed_chunks(chunks)` — encode từng chunk content bằng `BAAI/bge-m3`
4. `index_to_vectorstore(chunks)` — upsert vector + metadata vào ChromaDB

**Helper export (dùng chung với Task 5/6):**

```python
def get_embedding_model(): ...      # lazy-load, cached instance
def get_collection(): ...           # mở persistent collection
def prepare_query_for_embedding(query: str) -> str: ...
```

**Yêu cầu:**
- Chunk không vượt quá `CHUNK_SIZE` (800) ký tự
- `chunk_id` format: `{document_id}_chunk_{chunk_index}`
- Metadata giữ `source`, `title`, `url`, `type`, `section`, `language`, `chunk_id`, `document_id`
- Index persistent, rebuild idempotent (xóa collection cũ trước khi tạo mới)
- Semantic search (Task 5) có thể query cùng collection ngay sau khi index

---

### Task 5 — Semantic Search & Query Expansion

Module thực hiện **dense semantic retrieval** trên ChromaDB từ Task 4.

**API chính (trong `src/task5_semantic_search.py`):**

```python
def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Score: cosine similarity [0, 1], sorted descending
    """
    ...

def semantic_search_expanded(
    query: str,
    top_k: int = 10,
    max_variants: int = 3,
    history: list[dict] | None = None,
) -> list[dict]:
    """Search original + expanded queries, fuse by RRF, keep best raw cosine."""
```

**Yêu cầu:**
- Tái sử dụng `get_embedding_model()`, `get_collection()`, `prepare_query_for_embedding()` từ Task 4
- Chuẩn hóa query: NFC, trim, collapse whitespace, giữ nguyên dấu tiếng Việt
- Query ChromaDB với cosine distance → `1.0 - distance`, clamp `[0, 1]`
- Deduplicate bằng `chunk_id`
- Cache embedding model (1 instance/process), Chroma client, query embedding (LRU)
- Kết quả có `raw_scores.dense` để Task 9 calibrate fallback threshold

**Query Expansion (Bonus, 5 điểm) — trong `src/task5_query_expansion.py`:**

```python
def expand_query(
    query: str,
    history: list[dict] | None = None,
    max_variants: int = 3,
) -> list[str]:
    """Deterministic bilingual expansion using domain glossary."""
```

- Deterministic, không cần API key
- Domain glossary: `học phí ↔ tuition fee`, `học bổng ↔ scholarship`, ...
- Original query luôn là phần tử đầu tiên
- Fusion các variants bằng RRF trên `chunk_id`
- Giữ `raw_scores.dense` (best raw cosine) cho Task 9
- Optional LLM expansion với fallback về deterministic

---

### Task 6 — Lexical Search (BM25 & TF-IDF)

Module thực hiện **lexical retrieval** trên cùng tập chunk với Task 4/5. Mặc định dùng **BM25**, bonus với **TF-IDF char n-gram**.

**API chính (trong `src/task6_lexical_search.py`):**

```python
def build_bm25_index(corpus: list[dict]):
    """Build BM25 index (k1=1.5, b=0.75) from corpus chunks."""

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """BM25 search, returns chunks with positive BM25 score, sorted descending."""

def build_tfidf_index(corpus: list[dict]):
    """Build TF-IDF index with char_wb n-gram (3,5)."""

def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """TF-IDF cosine search, scores in [0, 1]."""

def lexical_search_configured(
    query: str,
    top_k: int = 10,
    method: str = "bm25",  # "bm25" | "tfidf" | "bm25_tfidf"
) -> list[dict]:
    """Configured lexical search with method selection."""
```

**Cài đặt:**
```bash
pip install rank-bm25 scikit-learn
```

**Text Normalization & Tokenization:**

```python
def normalize_lexical_text(text: str) -> str:
    """NFC, lowercase, collapse whitespace, keep diacritics."""

def tokenize_lexical(text: str) -> list[str]:
    """Unicode-aware regex: \\w+(?:[-./]\\w+)*"""
```

**BM25 (Core):**
- `BM25Okapi(k1=1.5, b=0.75)`
- Corpus load từ ChromaDB collection hoặc fallback `load_documents()` + `chunk_documents()` từ Task 4
- Lazy index manager, rebuild khi corpus hash đổi
- Tie-break: score → exact match count → doc_id → chunk_index

**TF-IDF (Bonus, 5 điểm):**
- `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True)`
- Bền với word segmentation tiếng Việt và typo nhỏ
- Score cosine `[0, 1]`, không cộng trực tiếp với BM25
- Fusion BM25+TF-IDF qua RRF, không cộng raw score khác thang đo

---

### Task 7 — Reranking & Weighted Fusion

Module hợp nhất và sắp hạng lại candidates từ dense (Task 5) và lexical (Task 6) retrieval.

**API chính (trong `src/task7_reranking.py`):**

```python
def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict]:
    """Weighted RRF: RRF(d) = Σ weightᵣ / (k + rankᵣ(d))"""

def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Cross-encoder reranking (optional, requires API/model)."""

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """MMR diversity reranking (optional)."""

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "rrf" | "cross_encoder" | "mmr" | "none"
) -> list[dict]:
    """Unified reranking interface."""
```

**Weighted RRF — Core (bắt buộc):**

| Ranker | Weight |
|--------|-------:|
| Dense (cosine) | `1.0` |
| BM25 | `0.9` |
| TF-IDF | `0.7` |

- Công thức: `RRF(d) = Σ weightᵣ / (k + rankᵣ(d))`
- Deduplicate bằng `chunk_id` (fallback: content hash + source)
- Merge `raw_scores` và `provenance` từ tất cả occurrences
- `score_type="rrf"`, `confidence_score` = max raw dense cosine
- Không mutate input objects

**Cross-encoder (Optional):**
- Adapter pattern, hỗ trợ Jina/Qwen/local model
- Timeout 15-30s, retry 1x cho 429/5xx, fallback về RRF order
- Cache query + candidate hashes trong process

**MMR (Optional):**
- `MMR(d) = λ × relevance(query,d) - (1-λ) × max similarity(d, selected)`
- Chỉ dùng khi top candidates bị trùng content

**Config Profiles:**

| Profile | Pipeline |
|---|---|
| `fusion_only` | Dense + BM25 + TF-IDF → Weighted RRF → top_k (default) |
| `quality_rerank` | Weighted RRF top 15 → cross-encoder → top_k |
| `quality_diverse` | Weighted RRF top 20 → cross-encoder top 10 → MMR → top_k |

---

### Task 8 — PageIndex Vectorless RAG & Resilient Fallback

Tích hợp [PageIndex](https://pageindex.ai/) như một **vectorless retrieval backend** — fallback khi hybrid search confidence thấp.

**API chính (trong `src/task8_pageindex_vectorless.py`):**

```python
def upload_documents(
    force: bool = False,
    wait_until_ready: bool = True,
) -> dict[str, str]:
    """Upload changed documents to PageIndex, return document_id → pageindex_doc_id."""

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval via PageIndex. Returns [] on API error (never crashes)."""

def get_pageindex_status() -> dict:
    """Sanitized availability/readiness diagnostics."""
```

**Cài đặt:**
```bash
pip install "pageindex>=0.2.8"
```

**Document Registry (`pageindex_doc_ids.json`):**
- Lưu mapping `document_id` → `pageindex_doc_id`
- Track checksum, status, `retrieval_ready` flag
- Atomic writes (temp file → replace)
- Skip upload nếu checksum không đổi và ready

**Upload Flow:**
1. Ưu tiên upload original PDF từ `data/landing/legal/`
2. Markdown/news được convert sang Unicode PDF
3. Poll processing status với deadline (600s) + exponential backoff
4. Không upload trong query path

**Query Flow:**
1. Shortlist ready documents (max 3) via file-level metadata
2. Submit retrieval jobs, poll với shared deadline
3. Parse response: flatten `retrieved_nodes` → `relevant_contents`
4. Gán `score_type="rank_proxy"`, `score = 1.0 / global_rank`

**Output Contract:**

```python
{
    "content": "...",
    "score": 1.0,           # rank proxy, KHÔNG phải cosine confidence
    "score_type": "rank_proxy",
    "confidence_score": None,
    "source": "pageindex",
    "metadata": {
        "chunk_id": "pageindex:<doc_id>:<node_id>:<hash>",
        "document_id": "...",
        "source": "NhanVan_HocPhi.pdf",
        "title": "...",
        "section": "...",
        "node_id": "0005",
        "page_index": 10,
    }
}
```

**Cache & Resilience:**
- In-memory TTL cache cho query results (3600s)
- Circuit breaker: sau N lỗi liên tiếp, tạm trả `[]`
- Cache hit → `metadata.cache_hit=true`
- Không giả kết quả dưới `source="pageindex"` khi API unavailable

**Local Structural Search (resilience extension):**

```python
def local_structural_search(query: str, top_k: int = 5) -> list[dict]:
    """Local fallback using Markdown heading tree + TF-IDF."""
```

- `source="hybrid"`, `retrieval_method="local_structural"`
- Không gắn nhãn `pageindex` cho kết quả local

---

### Task 9 — Retrieval Pipeline Hoàn Chỉnh (Hybrid + Fallback)

Tầng **orchestration** kết hợp semantic search (Task 5), lexical search (Task 6), RRF fusion (Task 7) và PageIndex fallback (Task 8) vào một `retrieve()` thống nhất.

**Cấu hình:**

| Constant | Default | Mô tả |
|---|---|---|
| `SCORE_THRESHOLD` | `0.3` | Raw cosine threshold để trigger fallback (cần calibrate) |
| `DEFAULT_TOP_K` | `5` | Số lượng kết quả mặc định |
| `RERANK_METHOD` | `"none"` | Post-fusion reranking method (tránh double-RRF) |

**API chính (trong `src/task9_retrieval_pipeline.py`):**

```python
def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Full retrieval pipeline: hybrid search + RRF fusion + optional reranking + fallback.
    """
```

**Pipeline Flow:**

```
Query
  │
  Step 1 — Parallel retrieval (top_k × 2 candidate pool)
  ├─→ semantic_search(query, top_k=top_k*2)  ──┐   (Task 5)
  └─→ lexical_search(query, top_k=top_k*2)  ───┤   (Task 6)
  │                                              │
  Step 2 — RRF Merge                            │
  └─→ rerank_rrf([dense, sparse], top_k=top_k*2)◄┘   (Task 7)
       │  mỗi result: source="hybrid"
       │
  Step 3 — Optional Reranking
       if use_reranking and RERANK_METHOD != "none":
  └─→ rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
       else: final = merged[:top_k]
       │
  Step 4 — Fallback Check
       best_cosine = dense_results[0]["score"] (raw cosine, NOT RRF)
       if best_cosine < score_threshold:
  └─→ pageindex_search(query, top_k=top_k)     (Task 8)
       if fallback non-empty → return fallback
       else → return []
```

> ⚠️ **Bẫy thường gặp (quan trọng):** RRF score (`≈1/(k+1) ≈ 0.016`) **chỉ phụ thuộc thứ hạng**,
> không phản ánh độ liên quan thực sự. Dùng RRF score làm threshold khiến fallback **không bao giờ
> trigger**. Luôn dùng **raw cosine similarity từ `semantic_search`** (thang `[0,1]`) để so `score_threshold`.

**Output Contract:**

```python
# Hybrid result
{
    "content": "...",
    "score": 0.0321,             # RRF score (rank-based, sắp hạng)
    "score_type": "rrf",
    "confidence_score": 0.82,    # raw cosine similarity (dùng cho threshold)
    "source": "hybrid",
    "metadata": { "chunk_id": "...", "document_id": "...", ... },
    "raw_scores": { "dense": 0.82, "bm25": 6.78, "rrf": 0.0321 }
}

# PageIndex fallback result
{
    "content": "...",
    "score": 0.5,                # rank proxy
    "score_type": "rank_proxy",
    "confidence_score": None,
    "source": "pageindex",
    "metadata": { "chunk_id": "pageindex:...", ... },
    "raw_scores": { "pageindex_rank_proxy": 0.5 }
}
```

**Calibrate `SCORE_THRESHOLD`:**

```bash
python -c "
from src.task5_semantic_search import semantic_search
for q in ['tuition fee', 'scholarship', 'xyzabc123']:
    r = semantic_search(q, top_k=1)
    print(f'{q}: best cosine = {r[0][\"score\"]:.3f}' if r else f'{q}: no results')
"
```

Chọn threshold nằm giữa cluster in-domain (cao) và OOD (thấp).



---

### Task 10 — Generation Có Citation

Tầng **generation cuối cùng**: retrieve → reorder (tránh lost in the middle) → format context → LLM → answer có citation.

**Cấu hình:**

| Constant | Default | Lý do |
|---|---:|---|
| `TOP_K` | `5` | Đủ evidence mà không overflow context window |
| `TOP_P` | `0.9` | Cân bằng diversity và determinism |
| `TEMPERATURE` | `0.3` | Thấp → giảm hallucination, phù hợp factual RAG |
| `LLM_MODEL` | `"openai/gpt-4o-mini"` | OpenRouter model (hỗ trợ `:free` suffix) |

**API chính (trong `src/task10_generation.py`):**

```python
def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder: front + back[::-1]
    Input:  [1, 2, 3, 4, 5]  → Output: [1, 3, 5, 4, 2]
    Best chunk first, worst in middle, second-best last.
    """

def format_context(chunks: list[dict]) -> str:
    """
    Format: [Document {i} | Source: {source} | Type: {type}]
    Separator: \n---\n
    """

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end: retrieve → reorder → format → LLM → answer
    Returns: {"answer": str, "sources": list, "retrieval_source": str}
    """
```

**Pipeline:**

```
1. Retrieve chunks từ Task 9: chunks = retrieve(query, top_k=TOP_K)
   Nếu không có chunks → trả no-answer ngay

2. Reorder: reordered = reorder_for_llm(chunks)
   front = chunks[::2]      # indices 0, 2, 4, ...
   back  = chunks[1::2]     # indices 1, 3, ...
   result = front + back[::-1]

3. Format: context = format_context(reordered)
   [Document 1 | Source: tuition-policy.md | Type: legal]
   Nội dung chunk...
   ---
   [Document 2 | Source: scholarship-news.md | Type: news]
   ...

4. Build prompt:
   system = SYSTEM_PROMPT (tiếng Việt, citation required, no hallucination)
   user = "Context:\n{context}\n\n---\n\nQuestion: {query}"

5. Gọi OpenRouter (OpenAI-compatible SDK):
   client.chat.completions.create(
       model=LLM_MODEL,
       messages=[{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user_message}],
       temperature=TEMPERATURE,
       top_p=TOP_P,
   )
```

**System Prompt (tiếng Việt, yêu cầu citation):**

```
Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context
```

**Yêu cầu:**
- Output phải có citation dạng `[Nguồn, Năm]` hoặc `[Document N, Section]`
- Nếu không đủ evidence → trả về "Tôi không thể xác minh thông tin này từ nguồn hiện có"
- `reorder_for_llm()`: giữ nguyên length, first element không đổi, không duplicate/missing

---

## Bài Tập Nhóm

> **Sau khi cả nhóm hoàn thành Task 1-10**, cùng nhau xây dựng **1 trong 2 sản phẩm** sau:

---

### Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về chính sách và dịch vụ đại học liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

### Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

#### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

#### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

#### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

#### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

#### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="UniversityServices_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

#### Deliverable Evaluation

- [x] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [x] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [x] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

### Yêu Cầu Chung

1. **Tích hợp pipeline** Task 1-10 mà cả nhóm đã xây dựng
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (xem `group_project/README.md`)

---

### Kiến Trúc Hệ Thống

Hệ thống là một **RAG pipeline end-to-end** gồm 4 tầng chính: **Ingestion/Dữ liệu → Indexing → Retrieval (Hybrid + Fallback) → Generation (có citation)**. Toàn bộ pipeline được điều phối bởi tầng Chatbot (Streamlit) và Evaluation (DeepEval/RAGAS).

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        TẦNG QUẢN LÝ PHÁI TRÊN (UI & Eval)                    │
│                                                                              │
│   ┌──────────────┐            ┌────────────────────────────────────────────┐ │
│   │ app.py       │            │ group_project/evaluation/                  │ │
│   │ Streamlit UI │ ─────────▶ │ • golden_dataset.json (≥15 Q&A)             │ │
│   │ Chatbot      │            │ • eval_pipeline.py (DeepEval/RAGAS)         │ │
│   │ + sources    │            │ • results.md (A/B report)                   │ │
│   └──────┬───────┘            └────────────────────────────────────────────┘ │
└──────────┼────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TẦNG GENERATION (Task 10)                            │
│                                                                              │
│   generate_with_citation(query)                                              │
│     reorder_for_llm()  → tránh "lost in the middle" (best đầu, 2nd cuối)     │
│     format_context()   → [Document N | Source | Type]                       │
│     OpenRouter LLM (gpt-4o-mini / gemini :free)  → trả lời có citation      │
│     temperature=0.3, top_p=0.9                                              │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    TẦNG RETRIEVAL (Task 9) — Hybrid + Fallback               │
│                                                                              │
│   retrieve(query, top_k, score_threshold)                                    │
│        │                                                                     │
│        ├─▶ Task 5 semantic_search ──┐ (dense, cosine [0,1])                 │
│        ├─▶ Task 6 lexical_search ───┼ (BM25 + TF-IDF, sparse)               │
│        │        └───────────────────┘                                        │
│        ▼                          ▼                                          │
│   Task 7 rerank_rrf(weighted)  ──▶  top_k*2 candidate pool (dedup by chunk_id)│
│        │                                                                     │
│        │  best_cosine < score_threshold ?                                    │
│        ├── YES ──▶ Task 8 pageindex_search (vectorless fallback)             │
│        │           • rank_proxy score, không phải cosine confidence          │
│        └── NO  ──▶ hybrid results  (source="hybrid")                        │
│                                                                              │
│   ⚠️  Fallback dùng RAW cosine từ semantic_search, KHÔNG dùng RRF score      │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    TẦNG INDEXING (Task 4) — Vector Store                     │
│                                                                              │
│   chunk_documents()  → fixed_size (CHUNK_SIZE=800, OVERLAP=100)             │
│   embed_chunks()     → BAAI/bge-m3 (dim=1024) qua OpenRouter / local         │
│   index_to_vectorstore() → ChromaDB (cosine) tại chroma_db/                 │
│   Export helpers: get_embedding_model(), get_collection()                    │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     TẦNG INGESTION (Task 1-3) — Dữ liệu                     │
│                                                                              │
│   Task 1: Thu thập legal (PDF/DOCX) ──▶  data/landing/legal/                │
│   Task 2: Crawl news (Crawl4AI)       ──▶  data/landing/news/               │
│   Task 3: MarkItDown convert          ──▶  data/standardized/{legal,news}/   │
│   (Task 3-optimize: tối ưu Markdown cho RAG)                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Luồng dữ liệu chính:**

1. **Ingestion** — Thu thập chính sách/đại học (task1) và crawl bài viết (task2), rồi chuẩn hóa thành Markdown (task3).
2. **Indexing** — Chunk cố định 800/100 ký tự, embed bằng `BAAI/bge-m3`, lưu vector + metadata vào ChromaDB (task4).
3. **Retrieval** — Chạy song song **Dense** (task5, cosine) và **Sparse** (task6, BM25/TF-IDF), hợp nhất bằng **Weighted RRF** (task7), rồi kiểm tra ngưỡng `score_threshold` dựa trên **raw cosine**; nếu quá thấp → fallback **PageIndex vectorless** (task8) (task9).
4. **Generation** — Sắp xếp lại chunk để tránh *lost in the middle*, định dạng context có source label, gọi LLM qua OpenRouter để sinh câu trả lời **có citation** (task10).
5. **Hiển thị & Đánh giá** — Chatbot Streamlit hiển thị answer + nguồn tham khảo; evaluation pipeline (DeepEval/RAGAS) đo faithfulness, answer relevance, context recall/precision và so sánh A/B.

**Các điểm thiết kế quan trọng:**

- **Hybrid fusion** bằng RRF (không cộng trực tiếp cosine + BM25 vì khác thang đo).
- **Fallback threshold** dùng raw cosine similarity (thang `[0,1]`) từ Task 5, tách biệt với RRF score dùng để xếp hạng — tránh bẫy fallback không bao giờ trigger.
- **Dedup key thống nhất** là `chunk_id` xuyên suốt Task 4→9.
- **Citation** dựa trên metadata `source/title/url/type/section` giữ nguyên từ tầng Indexing.

---

### Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|---|---|---|---|
| Nguyễn Lê Minh | 2A202601045 | Điều phối và tích hợp dự án; Task 4 — Chunking & Indexing; Task 9 — Retrieval Pipeline; Task 10 — Generation; tích hợp giao diện và chuẩn bị RAG Demo | Hoàn thành |
| Nguyễn Chí Quang | 2A202601932 | Task 5 — Semantic Search; Task 6 — Lexical Search; Task 7 — Reranking; Task 8 — PageIndex/Vectorless RAG; xây dựng Golden Dataset và RAG Evaluation | Hoàn thành |
| Bùi Hoàng Vương | 2A202601553 | Task 1 — thu thập văn bản chính sách đại học; Task 3 — chuẩn hóa và chuyển đổi dữ liệu Legal sang Markdown tối ưu cho RAG; hỗ trợ kiểm chứng Golden Dataset phần Legal | Hoàn thành |
| Đặng Tiến Thành | 2A202601305 | Task 2 — crawl bài viết, tin tức và thông báo; Task 3 — chuẩn hóa và chuyển đổi dữ liệu News sang Markdown tối ưu cho RAG; hỗ trợ kiểm chứng Golden Dataset phần News | Hoàn thành |

---

### Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

### Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.

---

## Cài Đặt Môi Trường

```bash
pip install -r requirements.txt
```

Tạo file `.env` từ `.env.example`:
```bash
cp .env.example .env
# Điền API keys vào .env
```

---

## Chấm Điểm

### Tổng Quan Phân Bổ Điểm

| Thành phần | Tỷ trọng | Mô tả |
|-----------|----------|-------|
| **Pipeline Kỹ Thuật (Task 1-10)** | **50%** | 10 tasks, cả nhóm cùng làm, chấm bằng automated tests + manual review |
| **Bài Nhóm** | **30%** | RAG Chatbot + Evaluation pipeline |
| **Bonus** | **20%** | Các tiêu chí nâng cao (xem bên dưới) |

---

### Pipeline Kỹ Thuật (Task 1-10) — 50 điểm (50%)

Chấm bằng automated test suite (`pytest tests/ -v`). Mỗi task có test riêng.

| Task | Nội dung | Điểm | Test |
|------|----------|------|------|
| 1 | Thu thập văn bản chính sách đại học (≥3 files tồn tại trong `data/landing/legal/`) | 3 | `test_task1_*` |
| 2 | Crawl bài viết/thông báo (≥5 files tồn tại trong `data/landing/news/`) | 3 | `test_task2_*` |
| 3 | Convert markdown (files tồn tại trong `data/standardized/`) | 4 | `test_task3_*` |
| 4 | Chunking + Indexing (vector store có data) | 7 | `test_task4_*` |
| 5 | Semantic search trả về kết quả đúng format, sorted | 6 | `test_task5_*` |
| 6 | Lexical search (BM25) trả về kết quả đúng format | 6 | `test_task6_*` |
| 7 | Reranking hoạt động, output re-sorted | 6 | `test_task7_*` |
| 8 | PageIndex query trả về kết quả | 4 | `test_task8_*` |
| 9 | Retrieval pipeline + fallback logic hoạt động | 7 | `test_task9_*` |
| 10 | Generation có citation + reorder | 4 | `test_task10_*` |
| **Tổng** | | **50** | |

---

### Bài Nhóm — 30 điểm (30%)

| Tiêu chí | Điểm |
|----------|------|
| RAG Chatbot demo hoạt động được | 8 |
| Tích hợp pipeline Task 1-10 đã xây dựng | 4 |
| Kiến trúc rõ ràng + README | 3 |
| Chất lượng câu trả lời (có citation, đúng nội dung) | 3 |
| **Evaluation pipeline** (DeepEval / RAGAS / TruLens) | **12** |
| — Golden dataset ≥15 Q&A pairs | 3 |
| — Chạy eval với ≥4 metrics | 4 |
| — So sánh A/B ≥2 configs + phân tích | 3 |
| — Báo cáo kết quả có phân tích worst performers | 2 |

---

### Bonus — 20 điểm (20%)

| Tiêu chí | Điểm |
|----------|------|
| Giải thích cơ chế lexical search khác BM25 (trong demo) | 5 |
| Implement phương pháp hỗ trợ Semantic Search (HyDE, Query Expansion, ...) | 5 |
| Deploy chatbot online (Hugging Face Spaces / Render / ...) | 4 |
| Conversation memory (multi-turn chat) | 3 |
| UI/UX chất lượng (hiển thị source, score, highlight) | 3 |

---

### Chạy Test Chấm Điểm Pipeline Kỹ Thuật (Task 1-10)

```bash
# Chạy toàn bộ test suite
pytest tests/ -v

# Chạy từng task
pytest tests/test_individual.py::TestTask1 -v
pytest tests/test_individual.py::TestTask5 -v
```

---

## Hướng Dẫn Thời Gian

Theo đúng 7 Checkpoint trong `checkpoint_timer.html` (tổng 180 phút = 3 giờ):

| Checkpoint | Thời gian | Khoảng giờ | Hoạt động |
|------------|-----------|-------------|-----------|
| CP0 | 10 phút | 0:00–0:10 | Setup môi trường & khai báo API keys |
| CP1 | 25 phút | 0:10–0:35 | Task 1–3: Thu thập data + convert markdown |
| CP2 | 25 phút | 0:35–1:00 | Task 4–6: Chunking, indexing, search modules |
| CP3 | 20 phút | 1:00–1:20 | Task 7–8: Reranking + PageIndex fallback |
| CP4 | 25 phút | 1:20–1:45 | Task 9–10: Pipeline hoàn chỉnh + generation (mốc 50đ Task 1-10) |
| CP5 | 30 phút | 1:45–2:15 | Bài nhóm: Chatbot UI & đánh giá RAGAS |
| CP6 | 45 phút | 2:15–3:00 | Thuyết trình demo live & nộp bài |

---

## Tài Liệu Tham Khảo

- [Crawl4AI](https://github.com/unclecode/crawl4ai) — Web crawling library
- [MarkItDown](https://github.com/microsoft/markitdown) — Microsoft document converter
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) — Chunking strategies
- [Weaviate](https://weaviate.io/developers/weaviate) — Vector database with hybrid search
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 implementation
- [PageIndex](https://github.com/VectifyAI/PageIndex) — Vectorless RAG
- [Jina Reranker](https://jina.ai/reranker/) — Cross-encoder reranking API
- Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts*
