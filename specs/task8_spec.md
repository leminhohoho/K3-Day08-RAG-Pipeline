# Task 8 Spec — PageIndex Vectorless Retrieval và Resilient Fallback

**Spec cha:** `../../../../specq.md`  
**File triển khai chính:** `src/task8_pageindex_vectorless.py`  
**File mở rộng đề xuất:** `src/local_structural_search.py`  
**Điểm mục tiêu:** 4 điểm Task 8  
**Vai trò mở rộng:** structural retrieval, fallback an toàn và demo external-service resilience  
**Trạng thái:** Implemented and live-verified (PageIndex 0.2.8, 10/10 documents ready)

---

## 1. Mục tiêu

Task 8 tích hợp PageIndex như một vectorless retrieval backend. Module phải:

1. Upload/process tài liệu thật và lưu stable mapping giữa local document với PageIndex `doc_id`.
2. Chỉ query document đã xử lý xong và `retrieval_ready=true`.
3. Trả relevant passages có provenance rõ ràng.
4. Parse được response hiện tại và chịu được một số biến thể schema cũ.
5. Gán rank-proxy score khi PageIndex không trả similarity score.
6. Không giả kết quả PageIndex khi API/key không khả dụng.
7. Không làm Task 9/UI crash khi external service lỗi.
8. Cache upload/query hợp lý để demo không tốn thời gian/quota không cần thiết.
9. Có local structural retrieval riêng cho resilience, nhưng không gắn nhãn `pageindex`.
10. Tương thích starter tests và contract Task 9.

Task 8 không chịu trách nhiệm:

- Quyết định khi nào fallback: Task 9.
- Sinh answer bằng PageIndex Chat API: Task 10 vẫn là generation layer chính.
- Dùng PageIndex rank-proxy như confidence threshold.
- Giả lập response non-empty để qua test.

---

## 2. Tình trạng API và quyết định tích hợp

Tài liệu PageIndex hiện có ba hướng liên quan:

1. **Python SDK document processing:** upload PDF, nhận `doc_id`, kiểm tra status/tree.
2. **Chat API (beta):** trả lời trên một hoặc nhiều documents, là hướng mới được khuyến nghị.
3. **Retrieval API (legacy):** trả `retrieved_nodes`/evidence passages, vẫn được giữ để backward compatibility.

Task 8 cần **retrieval chunks** để Task 10 tự generation/citation, vì vậy baseline dùng Retrieval API legacy qua một adapter cô lập. Không để endpoint/schema legacy lan sang Task 9/UI.

```text
Task 8 public API
        │
        ▼
PageIndexBackend protocol
        │
        ├── CloudLegacyRetrievalBackend  ← baseline cho lab
        ├── Chat/MCP evidence adapter    ← hướng phát triển
        └── Self-hosted tree backend     ← hướng phát triển
```

Tài liệu chính thức:

- Python SDK: `https://docs.pageindex.ai/sdk`
- API reference: `https://docs.pageindex.ai/api-reference`
- Open-source PageIndex: `https://github.com/VectifyAI/PageIndex`

SDK stable được quan sát khi viết spec là dòng `0.2.x`; sau smoke test nên pin exact version đã xác nhận thay vì để `pageindex>=0.1.0` vô hạn. Không dùng pre-release cho buổi demo.

---

## 3. Kiến trúc

### 3.1 Upload/index flow

```text
data/sources_manifest.json / standardized files
        │
        ▼
select eligible documents
        │
        ├── original PDF có cấu trúc tốt → upload PDF trực tiếp
        └── chỉ có Markdown/news → tạo Unicode PDF hoặc approved adapter
        │
        ▼
PageIndex submit_document
        │
        ▼
poll processing status + retrieval_ready
        │
        ▼
pageindex_doc_ids.json
```

### 3.2 Query flow

```text
query
  → validate
  → load ready document registry
  → shortlist documents
  → submit retrieval jobs
  → poll with shared deadline
  → parse/flatten relevant contents
  → attach local provenance
  → dedupe
  → rank-proxy sort
  → top_k
```

### 3.3 Failure flow

```text
missing key / timeout / quota / no ready docs
        │
        ▼
pageindex_search() returns [] + sanitized status diagnostics
        │
        ▼
Task 9 safe refusal or separately named local structural backend
```

Không chuyển local result thành `source="pageindex"`.

---

## 4. Definition of Done

Task 8 được xem là hoàn thành khi:

- `pageindex_search()` không còn `NotImplementedError`.
- `TestTask8` có `2 passed, 0 skipped` khi chạy contract mode.
- Thiếu API key trả `[]`, không raise ra public boundary.
- Có ít nhất 3 tài liệu thật được upload/process và lưu `doc_id`.
- Registry ghi checksum/status/retrieval readiness.
- Một live query trả non-empty result có `source="pageindex"`.
- Result có content, rank-proxy score, document/section/page/node provenance.
- Polling có timeout/backoff và không loop vô hạn.
- Không upload lại tài liệu chưa đổi.
- API lỗi không làm Task 9/UI crash.
- Có mock tests cho upload, polling và schema parsing.
- Có live integration smoke test được đánh dấu riêng.
- Demo có route `PageIndex direct`, không dùng hidden force/fake response.

Lệnh starter test:

```powershell
python -m pytest tests/test_individual.py::TestTask8 -v -rs
```

Kết quả contract bắt buộc: `2 passed`, không skip. Tuy nhiên Task 8 chỉ được xem là hoàn tất về mặt lab khi có bằng chứng live query thật.

---

## 5. Configuration

```python
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_API_BASE = os.getenv(
    "PAGEINDEX_API_BASE",
    "https://api.pageindex.ai",
)
PAGEINDEX_REGISTRY_PATH = PROJECT_ROOT / "pageindex_doc_ids.json"
PAGEINDEX_PDF_CACHE_DIR = PROJECT_ROOT / "pageindex_pdfs"

PAGEINDEX_REQUEST_TIMEOUT_SECONDS = 30
PAGEINDEX_PROCESSING_TIMEOUT_SECONDS = 600
PAGEINDEX_RETRIEVAL_TIMEOUT_SECONDS = 90
PAGEINDEX_POLL_INITIAL_SECONDS = 1.0
PAGEINDEX_POLL_MAX_SECONDS = 8.0
PAGEINDEX_MAX_DOCUMENTS_PER_QUERY = 3
PAGEINDEX_THINKING = False
PAGEINDEX_CACHE_TTL_SECONDS = 3600
```

Yêu cầu:

- Tất cả timeout hữu hạn.
- API key chỉ lấy từ environment/secrets.
- Không log key/header đầy đủ.
- Base URL override chủ yếu phục vụ mock/test.
- Config values được validate khi module load hoặc client init.
- Registry và PDF cache đã nằm trong `.gitignore` nếu chứa generated/private artifacts.

---

## 6. Document registry

### 6.1 Schema

`pageindex_doc_ids.json`:

```json
{
  "schema_version": 1,
  "sdk_version": "0.2.x",
  "updated_at": "ISO-8601",
  "documents": {
    "ussh-tuition-plan-semester-1-2025-2026": {
      "document_id": "ussh-tuition-plan-semester-1-2025-2026",
      "pageindex_doc_id": "pi-abc123",
      "source": "NhanVan_HocPhi.pdf",
      "source_path": "data/landing/legal/NhanVan_HocPhi.pdf",
      "source_url": "https://...",
      "title": "Tuition Fees",
      "checksum": "sha256:...",
      "pageindex_input_checksum": "sha256:...",
      "status": "completed",
      "retrieval_ready": true,
      "uploaded_at": "ISO-8601",
      "last_checked_at": "ISO-8601",
      "error": null
    }
  }
}
```

### 6.2 Registry rules

- Write atomically: ghi temp file rồi replace.
- Không ghi API key/raw auth response.
- Registry corrupt: backup/report và không upload duplicate hàng loạt tự động.
- Reuse `doc_id` chỉ khi input checksum không đổi.
- Input đổi: upload version mới; không tự xóa remote document cũ trong request path.
- Remote delete là command quản trị riêng, yêu cầu explicit document ID.
- Sort keys để diff/debug ổn định.

---

## 7. Chọn và chuẩn bị tài liệu upload

### 7.1 Ưu tiên input

1. Original PDF hợp lệ trong `data/landing/legal/`.
2. DOCX được convert sang PDF Unicode có cấu trúc.
3. Standardized Markdown/news được convert sang PDF Unicode nếu cần cloud document processing.

Lý do ưu tiên original PDF:

- PageIndex/OCR có cơ hội giữ page/heading/layout tốt hơn.
- Citation có page index.
- Tránh mất cấu trúc do Markdown conversion không hoàn hảo.

### 7.2 Markdown support

PageIndex hiện có endpoint Markdown-to-tree trực tiếp. Tuy nhiên response tree trực tiếp không mặc nhiên thay thế `doc_id` của cloud retrieval flow. Vì vậy:

- Có thể dùng Markdown API để kiểm tra/cải thiện tree hoặc self-hosted structural mode.
- Baseline cloud retrieval vẫn dùng document flow có `doc_id`.
- Không giả định SDK `submit_document()` nhận Markdown nếu version đang dùng chỉ tài liệu hóa PDF.
- Nếu một SDK/API version mới hỗ trợ persisted Markdown document, thêm qua adapter và integration test trước khi đổi baseline.

### 7.3 PDF conversion requirements

- Font Unicode hỗ trợ tiếng Việt, embed font vào PDF.
- Giữ headings, bullets, tables cơ bản.
- Mỗi standardized document tạo một PDF riêng hoặc một dossier có bookmark/source boundary rõ.
- Không mất source title/URL.
- Validate file mở được và lớn hơn ngưỡng tối thiểu.
- Generated filename deterministic theo document ID/checksum.

---

## 8. Public API

### 8.1 `upload_documents`

Giữ starter name, làm rõ return contract:

```python
def upload_documents(
    force: bool = False,
    wait_until_ready: bool = True,
) -> dict[str, str]:
    """Upload changed eligible documents and return document_id -> PageIndex doc_id."""
```

Hành vi:

- Thiếu key: trả `{}` và status `unavailable`, không raise `NotImplementedError`.
- Load manifest/registry.
- Skip document có checksum giống và registry ready nếu `force=False`.
- Submit từng changed document.
- Lưu `doc_id` ngay sau submit để có thể resume polling.
- Poll status nếu `wait_until_ready=True`.
- Partial failure không rollback các upload thành công.
- Return mapping chỉ gồm documents có `doc_id`; readiness xem trong registry/status.

### 8.2 `pageindex_search`

Giữ đúng starter signature:

```python
def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Return PageIndex evidence passages, never fabricated results."""
```

Contract:

- Query trim rỗng trả `[]`.
- `top_k <= 0` trả `[]`.
- Thiếu key/registry/ready docs trả `[]`.
- Known network/auth/quota/timeout/schema errors được sanitize, ghi status và trả `[]`.
- Không trả nhiều hơn `top_k`.
- Mọi non-empty result có `source="pageindex"`.
- Không trả full chat answer như một retrieval chunk nếu không có evidence provenance.

Output:

```python
{
    "content": "...",
    "score": 1.0,
    "score_type": "rank_proxy",
    "confidence_score": None,
    "source": "pageindex",
    "retrieval_method": "pageindex_legacy_retrieval",
    "metadata": {
        "chunk_id": "pageindex:<doc_id>:<node_id>:<content_hash>",
        "document_id": "ussh-tuition-plan-semester-1-2025-2026",
        "pageindex_doc_id": "pi-abc123",
        "source": "NhanVan_HocPhi.pdf",
        "title": "Tuition Fees",
        "url": "https://...",
        "section": "Payment Structure",
        "node_id": "0005",
        "page_index": 10,
        "rank": 1,
        "score_kind": "rank_proxy",
        "cache_hit": false
    },
    "raw_scores": {
        "pageindex_rank_proxy": 1.0
    }
}
```

### 8.3 Service status

```python
def get_pageindex_status() -> dict:
    """Return sanitized availability/readiness diagnostics."""
```

Ví dụ:

```python
{
    "available": True,
    "configured": True,
    "ready_documents": 3,
    "total_registered_documents": 4,
    "last_error_type": None,
    "last_error_message": None,
    "last_success_at": "ISO-8601",
    "backend": "legacy_retrieval",
    "sdk_version": "0.2.x"
}
```

Không trả API key hoặc raw exception chứa headers.

---

## 9. Backend abstraction

```python
class PageIndexBackend(Protocol):
    def submit_document(self, path: Path) -> str: ...
    def get_document_status(self, doc_id: str) -> dict: ...
    def submit_retrieval(self, doc_id: str, query: str, thinking: bool) -> str: ...
    def get_retrieval(self, retrieval_id: str) -> dict: ...
```

`CloudLegacyRetrievalBackend` có thể:

- Dùng `PageIndexClient` cho document processing nếu SDK hỗ trợ.
- Dùng official REST endpoints cho legacy retrieval nếu SDK không expose chúng.
- Tập trung mọi URL/header/schema-specific code trong một file/class.

Import SDK hiện tại theo tài liệu:

```python
from pageindex import PageIndexClient
```

Không khóa spec vào import cũ `from pageindex.client import PageIndexClient` nếu version thực tế không có path đó.

---

## 10. Upload và processing polling

### 10.1 Polling algorithm

```python
deadline = monotonic() + PROCESSING_TIMEOUT
delay = POLL_INITIAL

while monotonic() < deadline:
    state = backend.get_document_status(doc_id)
    if state.status == "completed" and state.retrieval_ready:
        return READY
    if state.status in TERMINAL_FAILURES:
        return FAILED
    sleep(delay)
    delay = min(delay * 2, POLL_MAX)

return TIMEOUT
```

Yêu cầu:

- Dùng `time.monotonic()` cho deadline.
- Không sleep trong UI request quá processing timeout; upload nên là setup/admin action.
- Persist intermediate status để resume.
- Chấp nhận status casing/schema qua adapter normalization.
- `completed` nhưng `retrieval_ready=false` chưa được query legacy retrieval.
- Có cancel/keyboard interrupt an toàn.

### 10.2 Không upload trong query path

`pageindex_search()` không được tự upload toàn corpus. Nếu registry chưa ready:

- Trả `[]`.
- Status hướng dẫn chạy upload/setup.
- UI có admin/setup action riêng nếu cần.

---

## 11. Document shortlisting

Legacy retrieval nhận một `doc_id` mỗi request. Không query mọi remote document vô hạn.

```python
def select_pageindex_documents(
    query: str,
    registry: dict,
    max_documents: int = PAGEINDEX_MAX_DOCUMENTS_PER_QUERY,
) -> list[dict]: ...
```

Baseline shortlist dùng local metadata, không dùng vector DB:

- Title.
- Description.
- Document type.
- Section headings/tree summaries đã cache nếu có.
- BM25/TF-IDF file-level score.

Đây là file-level routing, PageIndex vẫn làm vectorless structural retrieval bên trong document.

Fallback khi không shortlist được:

- Query tối đa một số lượng nhỏ ready docs theo deterministic order/config.
- Không vượt quota/latency budget.

Diagnostics ghi doc IDs đã chọn và lý do/rank file-level.

---

## 12. Retrieval polling

### 12.1 Submit

Trước submit:

- Registry document ready.
- Query non-empty.
- Deadline chưa hết.

Payload legacy baseline:

```json
{
  "doc_id": "pi-abc123",
  "query": "...",
  "thinking": false
}
```

`thinking=true` chỉ bật trong quality profile nếu evaluation chứng minh cải thiện đủ lớn so với latency/cost.

### 12.2 Poll

- Dùng shared overall deadline, không nhân timeout theo số docs.
- Tối đa 3 docs/query mặc định.
- Có thể poll jobs concurrently với bounded worker count.
- Terminal completed/failed/cancelled được normalize.
- Partial success: parse những job completed, không bỏ toàn bộ request.

### 12.3 Retry

- Retry tối đa 1–2 lần cho connect timeout/429/5xx theo `Retry-After`.
- Không retry auth 401/403, invalid doc/query 4xx hoặc schema validation error.
- Circuit breaker nhỏ: sau nhiều lỗi liên tiếp, tạm trả `[]` thay vì tiếp tục spam API.

---

## 13. Response parsing

### 13.1 Current expected shape

```json
{
  "status": "completed",
  "doc_id": "pi-abc123",
  "retrieved_nodes": [
    {
      "title": "Payment Structure",
      "node_id": "0005",
      "relevant_contents": [
        {
          "page_index": 10,
          "relevant_content": "..."
        }
      ]
    }
  ]
}
```

### 13.2 Defensive parser

Starter note đề cập response cũ có thể có nested lists. Parser phải flatten an toàn:

```python
def iter_relevant_items(value):
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        for child in value:
            yield from iter_relevant_items(child)
```

Content keys theo thứ tự:

1. `relevant_content`.
2. `content`.
3. `text`.

Section/title:

1. Item `section_title` nếu có.
2. Node `title`.
3. Registry document title.

Provenance:

- `doc_id` từ response/job context.
- `node_id` từ node.
- `page_index` từ item/node.
- Local document metadata từ registry.

### 13.3 Invalid response

- Missing `retrieved_nodes` trên completed response: schema error, trả partial/empty.
- Content rỗng: bỏ item.
- HTML/error body: không đưa vào evidence.
- Oversized content: cắt theo safe evidence budget nhưng giữ raw response chỉ trong debug artifact bảo vệ.
- Log schema keys/type, không log toàn payload mặc định.

---

## 14. Score, ranking và deduplication

PageIndex legacy retrieval không đảm bảo trả similarity score. Dùng rank proxy:

```python
rank_proxy = 1.0 / global_rank
```

Hoặc một mapping monotonic tương đương, nhưng phải:

- Top rank có score cao nhất.
- Score chỉ dùng sắp hạng trong PageIndex branch.
- Ghi `score_type="rank_proxy"` và `confidence_score=None`.
- Không so rank proxy với dense threshold.
- Không trình bày score này như xác suất liên quan.

Global ordering đề xuất:

1. File-level shortlist rank.
2. Node/relevant-content order do PageIndex trả về.
3. Stable doc/node/content hash.

Dedupe key:

```text
pageindex_doc_id + node_id + normalized_content_hash
```

Nếu cùng content xuất hiện ở nhiều docs, giữ provenance riêng trừ khi xác định rõ là cùng canonical document/version.

---

## 15. Cache và demo resilience

### 15.1 Upload cache

- Registry + checksum là upload cache bền vững.
- Không gọi submit lại nếu input không đổi và document ready.

### 15.2 Query cache

Cache key:

```text
backend_version
+ normalized_query
+ selected_doc_ids/checksums
+ thinking flag
+ parser_version
```

- In-memory TTL cache là mặc định.
- Optional persistent cache chỉ lưu sanitized evidence/provenance từ một response thật.
- Cached result ghi `metadata.cache_hit=true` và timestamp.
- Không dùng hand-written cached answer.
- Cache invalid khi document checksum/backend/parser config đổi.

Nếu API live lỗi nhưng có cache hợp lệ từ cùng query/docs:

- Có thể trả cached PageIndex evidence.
- UI phải hiển thị “cached PageIndex result”.
- Diagnostics ghi live error riêng.

### 15.3 Circuit breaker

Sau N lỗi liên tiếp trong cửa sổ ngắn:

- Đánh dấu degraded.
- Trả cache hợp lệ hoặc `[]` ngay.
- Không tiếp tục chờ timeout cho mọi user request.
- Cho phép reset sau cooldown/admin action.

---

## 16. Local structural retrieval

Tạo API riêng:

```python
def local_structural_search(query: str, top_k: int = 5) -> list[dict]: ...
```

Có thể dùng:

- Markdown heading tree.
- Parent/child section mapping từ Task 4.
- TF-IDF/BM25 trên heading + summary.
- SQLite FTS5.

Output:

```python
{
    "source": "hybrid",
    "retrieval_method": "local_structural",
    ...
}
```

Không dùng:

```python
"source": "pageindex"
```

trừ khi evidence thực sự đến từ PageIndex live hoặc valid PageIndex cache.

Task 9 có thể chọn local structural result như resilience extension, nhưng starter `source` contract chỉ chấp nhận `hybrid/pageindex`; vì vậy `source="hybrid"`, còn method chi tiết nằm ở `retrieval_method`.

---

## 17. Error handling

| Tình huống | Public behavior |
|---|---|
| Query rỗng/top_k không hợp lệ | `[]` |
| Thiếu API key | `[]`, status `not_configured` |
| Registry không tồn tại | `[]`, hướng dẫn upload setup |
| Không có ready document | `[]` |
| Upload một doc lỗi | Tiếp tục docs khác, registry ghi lỗi |
| Processing timeout | Registry `timeout`, có thể resume |
| Retrieval timeout | Partial results hoặc `[]` |
| 401/403 | `[]`, status auth error, không retry |
| 429 | Backoff hữu hạn/cache/`[]` |
| 5xx/network | Retry hữu hạn/cache/`[]` |
| Response schema lạ | Defensive parse; schema error sanitized |
| Cache corrupt | Bỏ cache, không trả dữ liệu không xác minh |
| SDK import/version mismatch | `[]`, status dependency error |

Không dùng `except Exception: pass` không log. Public boundary có thể trả `[]`, nhưng phải lưu error type/diagnostics đã sanitize.

---

## 18. Security và dữ liệu

- Chỉ upload tài liệu public/được phép chia sẻ.
- Manifest ghi rõ source URL và licensing/usage note nếu cần.
- Không upload `.env`, logs, API response chứa key hoặc file ngoài allowlisted data roots.
- Resolve và kiểm tra path nằm trong `data/landing`, `data/standardized` hoặc generated PDF cache.
- Giới hạn file size/page count theo quota.
- Không cho user truyền local file path tùy ý vào upload API.
- Không log request headers.
- Sanitize remote error messages trước UI.
- Remote delete không tự động; cần explicit admin command/document ID.
- Cached evidence không chứa secret và có provenance/timestamp.

---

## 19. Logging và diagnostics

Upload log:

- Local document ID/checksum.
- Reused/uploaded/failed.
- PageIndex doc ID đã truncate nếu cần.
- Processing status/retrieval readiness.
- Poll count/latency.

Query log:

- Query hash.
- Selected docs và shortlist ranks.
- Retrieval IDs đã sanitize/truncate.
- Live/cache path.
- Node/content count trước/sau dedupe.
- Backend/schema/parser version.
- Error/fallback reason.
- Latency submit/poll/parse/total.

UI diagnostics:

- PageIndex configured/available/degraded.
- Ready documents count.
- Route live/cached/unavailable.
- Selected document titles.
- Page/section/node provenance.
- Rank proxy được ghi rõ không phải confidence.

---

## 20. Test specification

### 20.1 Starter tests

1. `pageindex_search` callable.
2. Return list.
3. Non-empty result có `source="pageindex"`.

Để không skip khi thiếu key, public `pageindex_search()` trả `[]` thay vì raise.

### 20.2 Registry/upload unit tests

Dùng mock backend, không gọi live API:

1. Missing key trả `{}`.
2. New checksum gọi submit và lưu doc ID.
3. Same checksum/ready skip upload.
4. `force=True` submit lại.
5. Partial failure không mất successful mappings.
6. Atomic registry write.
7. Corrupt registry được xử lý an toàn.
8. Processing completed nhưng retrieval not ready chưa đánh dấu ready.
9. Timeout không loop vô hạn.
10. Poll backoff không vượt max.

### 20.3 Shortlist tests

1. Query tuition chọn tuition document trước.
2. Không vượt max documents.
3. Chỉ trả registry docs ready.
4. Tie-break deterministic.
5. Empty metadata fallback an toàn.

### 20.4 Retrieval/parser tests

Fixtures:

- Current list-of-dicts `relevant_contents`.
- Legacy nested lists.
- Missing optional page/section.
- Partial completed jobs.
- Empty/invalid content.
- Unknown schema.

Assertions:

1. Content/section/node/page parse đúng.
2. Local registry provenance được attach.
3. Dedupe đúng.
4. Rank proxy sort giảm dần.
5. `score_type=rank_proxy`, `confidence_score=None`.
6. Non-empty result có source marker.
7. Raw chat/error body không bị trả làm evidence.

### 20.5 Failure/cache tests

1. Auth error không retry.
2. 429/5xx retry hữu hạn.
3. Deadline chung được tôn trọng.
4. Valid cache được trả và đánh dấu khi live lỗi.
5. Stale/mismatched cache không được trả.
6. Circuit breaker giảm repeated timeout.

### 20.6 Live integration test

Đánh dấu riêng, chỉ chạy khi có key/quota:

```powershell
python -m pytest tests/test_pageindex_live.py -m live -v
```

Kiểm tra:

- Ít nhất một registry document ready.
- Query trả evidence non-empty.
- Evidence có doc ID, source, section/node và content.
- Latency được ghi.

CI/offline không chạy live test, nhưng demo readiness checklist bắt buộc chạy trước buổi chấm.

---

## 21. Evaluation và tối ưu

Tạo subset fallback/structural questions:

- Câu hỏi gắn với section trong document dài.
- Câu có từ khóa ít xuất hiện trong chunk index.
- Câu multi-section.
- Query OOD phải trả empty evidence.

So sánh:

| Config | Mô tả |
|---|---|
| P1 | Hybrid only |
| P2 | PageIndex direct |
| P3 | Hybrid confidence gate → PageIndex |
| P4 | Hybrid gate → PageIndex → safe refusal |

Metrics:

- Fallback trigger precision/recall.
- Evidence Recall@k/Context recall.
- Context precision.
- OOD false-positive rate.
- Empty result rate.
- Live/cache hit rate.
- Latency p50/p95.
- API requests/query và estimated cost/quota use.

Không đặt threshold theo PageIndex rank proxy. Threshold thuộc Task 9 và được calibrate từ dense evidence confidence.

---

## 22. Kịch bản demo Task 8

### Chuẩn bị

- Upload/process từ trước.
- Registry có 3+ ready docs.
- Chạy live smoke query.
- Cache một response thật cho phương án dự phòng.
- Kiểm tra quota/key nhưng không hiển thị key.

### Demo live

1. Mở diagnostics cho thấy PageIndex ready documents.
2. Chọn route `PageIndex direct` minh bạch.
3. Hỏi câu structural gắn với section document dài.
4. Hiển thị source, section, page/node và `source=pageindex`.
5. Nêu rõ score là rank proxy, không phải cosine confidence.

### Demo resilience

- Cho thấy thiếu key/API lỗi trả empty/safe refusal thay vì crash.
- Nếu dùng cached response thật, UI ghi rõ cached timestamp.
- Không dùng hidden switch tạo fake result.

### Điểm cần giải thích

- PageIndex dùng tree/structure thay vì vector similarity.
- Retrieval API legacy được bọc qua adapter vì cần passages cho Task 10.
- Hướng mới có thể dùng Chat API/MCP, nhưng không trộn full generated answer vào retrieval contract hiện tại.
- Fallback trigger thuộc Task 9, dựa trên raw dense confidence đã calibrate.

---

## 23. Các hướng phát triển thêm

1. **Chat API evidence adapter:** dùng streaming/intermediate tool results để lấy evidence từ hướng API mới.
2. **MCP integration:** cho agent gọi PageIndex tool, vẫn chuẩn hóa provenance về chung result schema.
3. **Self-hosted PageIndex tree:** giảm phụ thuộc cloud/quota, phù hợp Markdown có headings tốt.
4. **Folder/workspace routing:** nhóm documents theo legal/news/topic trước query.
5. **Tree-aware local router:** dùng document descriptions/node summaries để shortlist tốt hơn.
6. **Async multi-document retrieval:** submit/poll bounded concurrency với shared deadline.
7. **Parent-node expansion:** lấy matched node cộng parent/adjacent section để generation đủ context.
8. **Schema-version adapters:** lưu captured fixtures cho mỗi API version.
9. **Cost-aware fallback:** chỉ gọi PageIndex khi expected value cao và quota còn đủ.
10. **Observability dashboard:** success rate, latency, quota, cache và fallback outcome.

Các hướng này chỉ làm sau khi cloud baseline, provenance, safe refusal và evaluation chạy ổn định.

---

## 24. Checklist triển khai

- [x] Dùng SDK/API version đã smoke test và pin stable release.
- [x] Backend adapter cô lập legacy endpoint/schema.
- [x] Registry schema + atomic write + checksum reuse.
- [x] Original PDF được ưu tiên; Markdown conversion giữ Unicode.
- [x] Upload không chạy trong query path.
- [x] Poll processing/retrieval có deadline/backoff.
- [x] Chỉ query docs `retrieval_ready=true`.
- [x] File-level shortlist giới hạn API calls.
- [x] Parser hỗ trợ current và nested legacy relevant contents.
- [x] Provenance source/title/url/section/node/page đầy đủ.
- [x] Rank proxy được label, không dùng làm confidence.
- [x] Public unavailable path trả `[]`, không `NotImplementedError`.
- [x] Không giả/local result dưới source PageIndex.
- [x] Cache chỉ lưu response thật và ghi cache status.
- [x] Unit/mock/live tests pass theo phạm vi.
- [x] Task 9 safe refusal khi PageIndex empty/unavailable.
- [x] Demo direct route và resilience đã diễn tập.

---

## 25. Tiêu chí chấp nhận cuối

Task 8 được nghiệm thu khi:

1. `TestTask8` có `2 passed, 0 skipped`.
2. Có ít nhất 3 ready PageIndex document IDs trace được về nguồn local/public.
3. Một live query trả evidence non-empty với `source="pageindex"`.
4. Polling không vô hạn và partial failures được xử lý.
5. Parser giữ section/node/page/source provenance.
6. Rank proxy không bị dùng như cosine confidence.
7. Thiếu key/quota/API lỗi không làm app crash hoặc tạo fake response.
8. Cache/resilience path minh bạch trong diagnostics.
9. Task 9 chỉ gọi PageIndex sau evidence gate và safe-refuse khi empty.
10. Demo chứng minh vectorless route thật cùng latency/trade-off.

