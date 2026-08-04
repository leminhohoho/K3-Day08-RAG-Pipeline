# Task 4 Spec — Chunking & Indexing into Vector Store

## 1. Mục tiêu và trạng thái

Task 4 biến các tài liệu Markdown chuẩn hóa từ Task 3 thành các chunk có thể truy xuất và lưu chúng trong ChromaDB.

```text
data/standardized/
  -> tách metadata và body
  -> chia thành đoạn cố định kích thước 800 ký tự, overlap 100 ký tự
  -> embedding phần nội dung
  -> ChromaDB (vector + metadata)
```

File `src/task4_chunking_indexing.py` hiện được giữ ở dạng starter/TODO để thành viên phụ trách Task 4 chủ động triển khai. Tài liệu này là hợp đồng triển khai và bàn giao, không có nghĩa là Task 4 đã hoàn thành.

- Người phụ trách: Role 2 — Data & Dense Search Dev
- Lệnh chạy dự kiến: `python src/task4_chunking_indexing.py`
- Đầu vào: toàn bộ `*.md` trong `data/standardized/legal/` và `data/standardized/news/`
- Đầu ra: collection ChromaDB persistent tại `chroma_db/`

---

## 2. Thiết kế đề xuất

### 2.1 Cấu hình

| Constant | Giá trị đề xuất | Lý do |
|---|---:|---|
| `CHUNK_SIZE` | `800` | Đủ ngữ cảnh cho văn bản pháp lý nhưng vẫn giữ retrieval chính xác |
| `CHUNK_OVERLAP` | `100` | Overlap 12.5% để hạn chế mất ý ở biên chunk |
| `CHUNKING_METHOD` | `"fixed_size"` | Chia thành các đoạn có độ dài cố định, không giữ cấu trúc heading |
| `EMBEDDING_MODEL` | `"BAAI/bge-m3"` | Multilingual, phù hợp tiếng Việt và tiếng Anh |
| `EMBEDDING_DIM` | `1024` | Kích thước output của `BAAI/bge-m3` |
| `VECTOR_STORE` | `"chromadb"` | Chạy local, persistent, có metadata filtering |
| `COLLECTION_NAME` | `"university_services_docs"` | Một collection thống nhất cho corpus của lab |

Ràng buộc bắt buộc:

- `CHUNK_SIZE > 0`
- `0 < CHUNK_OVERLAP < CHUNK_SIZE`
- Model dùng để index và model dùng để encode query ở Task 5 phải giống nhau.
- Nếu model cần query/document prefix thì phải cấu hình đối xứng, không tự thêm prefix cho một phía.

### 2.2 Pipeline

```text
run_pipeline()
  1. load_documents()
     - đọc tất cả file Markdown
     - parse canonical metadata header của Task 3
     - chỉ trả body làm content

  2. chunk_documents(documents)
     - chia body thành đoạn cố định kích thước 800 ký tự, overlap 100 ký tự
     - gắn section, section_path, chunk_index và chunk_id

  3. embed_chunks(chunks)
     - chỉ encode chunk content
     - không encode metadata kỹ thuật

  4. index_to_vectorstore(chunks)
     - rebuild collection an toàn
     - upsert vector, content và metadata
```

---

## 3. Hợp đồng dữ liệu

### 3.1 `load_documents()`

```python
def load_documents() -> list[dict]:
    ...
```

Mỗi phần tử trả về có dạng:

```python
{
    "content": "Nội dung thực của tài liệu sau dòng ---",
    "metadata": {
        "source": "article_04.md",
        "source_path": ".../data/standardized/news/article_04.md",
        "document_id": "ussh-undergraduate-admissions-2026",
        "type": "news",
        "title": "Thông tin tuyển sinh đại học chính quy năm 2026",
        "url": "https://ussh.vnu.edu.vn/...",
        "language": "vi",
        "section": "",
        "section_path": "",
        "content_hash": "..."
    }
}
```

Yêu cầu:

- Duyệt đệ quy bằng `STANDARDIZED_DIR.rglob("*.md")` và sắp xếp đường dẫn để kết quả ổn định.
- Đọc bằng `utf-8-sig` để chấp nhận cả UTF-8 có BOM.
- Với định dạng canonical của Task 3, parse header phía trên dòng `---`; chỉ body phía dưới mới được đưa vào `content`.
- Giữ các metadata phục vụ citation và audit: `document_id`, `title`, `url`, `type`, `language`, `organization`, ngày xuất bản, `source_sha256`, `content_hash` nếu có.
- Có fallback an toàn từ tên file/H1 khi một trường không tồn tại.
- Có thể hỗ trợ YAML front matter để tương thích dữ liệu ngoài, nhưng canonical header của Task 3 là đường xử lý chính.
- Trả `[]` khi không có file; không trả `None`.
- Báo lỗi rõ ràng nếu body canonical rỗng hoặc thiếu separator, thay vì embedding metadata header.

Ví dụ từ corpus hiện tại:

| Nhóm | File | `document_id` | URL |
|---|---|---|---|
| legal | `NhanVan_HocPhi.md` | `ussh-tuition-plan-semester-1-2025-2026` | URL thông báo học phí chính thức của USSH |
| news | `article_04.md` | `ussh-undergraduate-admissions-2026` | URL tuyển sinh 2026 chính thức của USSH |

### 3.2 `chunk_documents(documents)`

```python
def chunk_documents(documents: list[dict]) -> list[dict]:
    ...
```

Chiến lược fixed-size:

1. Cắt nội dung thành các đoạn có độ dài tối đa `CHUNK_SIZE` (800) ký tự.
2. Offset giữa hai chunk kề nhau là `CHUNK_SIZE - CHUNK_OVERLAP` (700), tạo overlap 100 ký tự.
3. Không tôn trọng ranh giới dòng/heading — cắt cứng theo số ký tự.

```python
def _fixed_size_split(content, chunk_size, chunk_overlap):
    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    while start < len(content):
        chunks.append(content[start:start + chunk_size])
        start += step
    return chunks
```

Mỗi chunk phải giữ metadata cha và bổ sung:

```python
{
    "section": "{title}",          # fallback từ title của document
    "section_path": "",             # không còn phân cấp heading
    "chunk_index": 0,
    "chunk_id": "{document_id}_chunk_0",
    "chunk_hash": "sha256-của-chunk-content"
}
```

Quy tắc quan trọng cho corpus hiện tại:

- `section` luôn dùng `title` làm fallback; không tạo chunk có section rỗng.
- `section_path` để `""` vì không còn phân cấp heading.
- `chunk_index` bắt đầu từ 0 và tăng liên tục trong từng document.
- `chunk_id` ổn định giữa các lần chạy nếu document và cấu hình chunking không đổi.
- Nội dung chunk không chứa `Source SHA256`, `Content Hash`, URL hay các dòng metadata kỹ thuật ở header.
- Mỗi chunk không vượt quá `CHUNK_SIZE` (800) ký tự.

### 3.3 `embed_chunks(chunks)`

```python
def embed_chunks(chunks: list[dict]) -> list[dict]:
    ...
```

Yêu cầu:

- Lazy-load và cache `SentenceTransformer(EMBEDDING_MODEL)` một lần trong mỗi process.
- Encode theo batch, bật progress bar khi chạy CLI.
- Chỉ đưa `chunk["content"]` vào model; metadata không tham gia embedding.
- Nếu sử dụng cosine similarity, nên normalize embedding và ghi rõ lựa chọn.
- Chuyển từng vector thành `list[float]`.
- Kiểm tra dimension thực tế bằng `EMBEDDING_DIM`; lỗi sai dimension phải có thông báo rõ.
- Với input rỗng, trả `[]` mà không tải model.

### 3.4 `index_to_vectorstore(chunks)`

```python
def index_to_vectorstore(chunks: list[dict]) -> None:
    ...
```

Yêu cầu:

- Dùng `chromadb.PersistentClient(path=str(CHROMA_DIR))`.
- Collection dùng cosine distance: `metadata={"hnsw:space": "cosine"}`.
- ID của Chroma lấy từ `metadata["chunk_id"]`.
- Upsert theo batch để tránh dùng quá nhiều RAM.
- Metadata ghi vào Chroma chỉ gồm scalar hợp lệ: `str`, `int`, `float`, `bool`.
- Không ghi index nếu chunk chưa có embedding.

### 3.5 Re-index an toàn

`upsert()` không tự xóa chunk của tài liệu đã bị loại khỏi corpus. Vì vậy full rebuild phải thay thế collection cũ trước khi index, nếu không retrieval có thể trả về dữ liệu stale.

Cách ưu tiên:

```python
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass  # chỉ bỏ qua trường hợp collection chưa tồn tại

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
```

Không yêu cầu người dùng xóa thủ công toàn bộ thư mục `chroma_db/` trong luồng chạy bình thường. Bản nâng cao có thể lưu `corpus_hash` và `index_config_hash` để chỉ rebuild khi corpus/model/chunk config thay đổi.

### 3.6 Cross-Task Contract (shared schema với Task 5/6/7/9/10)

Task 4 định nghĩa schema chunk mà các task downstream phải dùng chung. Các trường sau là **bắt buộc** trong mọi chunk:

| Field | Type | Ví dụ | Dùng bởi |
|-------|------|-------|----------|
| `chunk_id` | `str` | `"ussh-tuition-plan-semester-1-2025-2026_chunk_0"` | Task 5/6/7/9: dedup key, retrieval |
| `document_id` | `str` | `"ussh-tuition-plan-semester-1-2025-2026"` | Task 5/6/9: filter, tie-break |
| `source` | `str` | `"NhanVan_HocPhi.md"` | Task 10: citation gốc |
| `title` | `str` | `"Thông báo kế hoạch thu học phí..."` | Task 10: citation label |
| `url` | `str` | `"https://ussh.vnu.edu.vn/..."` | Task 10: citation link |
| `type` | `str` | `"legal"` / `"news"` | Task 10: citation doc type |
| `section` | `str` | `"Điều 1. Phạm vi điều chỉnh"` | Task 10: section citation |
| `language` | `str` | `"vi"` | — |

### Dedup key thống nhất (Task 7)

**Toàn bộ pipeline dùng `chunk_id` làm dedup key duy nhất.** Không dùng `content` hash, không dùng `content` string. Task 7 (`rerank_rrf`) phải ưu tiên `chunk_id` từ metadata, fallback content hash chỉ khi `chunk_id` thiếu.

Quy tắc dedup trong `rerank_rrf()`:

```python
def candidate_key(item: dict) -> str:
    chunk_id = item.get("metadata", {}).get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"
    # Fallback an toàn (không nên xảy ra với Task 4 data)
    return "fallback:" + sha256(normalized_content + source)
```

### Phân biệt RRF score với raw cosine confidence (Task 9)

| Score | Ý nghĩa | Nguồn |
|-------|---------|-------|
| `score` | RRF rank-based score (~0.016) — chỉ dùng sắp hạng | `rerank_rrf()` |
| `confidence_score` | Raw cosine similarity [0, 1] — dùng cho fallback threshold | `raw_scores.dense` |

Không dùng RRF score làm fallback threshold.

### Citation-ready metadata (Task 10)

Task 10 dùng các trường sau để tạo citation:

- `title` → hiển thị tên văn bản
- `url` → link nguồn gốc
- `section` → tên section/heading cụ thể
- `source` → tên file gốc
- `type` → legal/news

```python
# format_context() trong Task 10:
label = f"[Document {i} | Source: {title} | Type: {type}]"
# Citation inline:
# "Học phí là 3.300.000đ/tháng [Nguồn: Thông báo thu học phí, Mức thu và phương thức thu]"
```

### Corpus VNU-USSH — không dùng RMIT

Toàn bộ pipeline dùng corpus VNU-USSH, không phải RMIT. Các document_id theo convention:

- `ussh-{topic}-{year}` cho news
- `vnu-{regulation}-{year}` cho legal

Task 5/8/9/10 không được dùng RMIT trong test query hoặc ví dụ code.

## 3.7 Helper dùng chung với Task 5/6

Task 4 nên export các helper sau để Task 5/6 không tạo model hoặc client thứ hai:

```python
def get_embedding_model(): ...
def get_collection(): ...
def prepare_query_for_embedding(query: str) -> str: ...
def prepare_document_for_embedding(document: str) -> str: ...
```

Các import nặng (`sentence_transformers`, `chromadb`, `langchain_text_splitters`) nên đặt trong function. Nhờ đó test config và parse/chunk có thể chạy mà không tải model hoặc mở database.

---

## 4. Tiêu chí chất lượng và bonus

Phần bắt buộc:

- Đọc được toàn bộ legal và news Markdown hiện tại.
- Metadata header không lọt vào embedding.
- Chunk đúng giới hạn kích thước và giữ metadata nguồn.
- Index persistent và Task 5 có thể truy vấn cùng collection.

Các điểm giúp demo nổi bật hơn:

- Section-aware chunking kèm `section_path`, đặc biệt hữu ích cho văn bản legal.
- Citation-ready metadata: mỗi kết quả có title, URL, document ID và section.
- Stable `chunk_id` và `chunk_hash` giúp audit/deduplicate.
- Rebuild idempotent, không còn chunk stale sau khi corpus thay đổi.
- Thống kê sau index: số document, số chunk theo `type`, min/avg/max length, số section fallback, số chunk trùng.
- Lưu manifest cấu hình index gồm corpus hash, embedding model, dimension, chunk size và overlap để tái lập kết quả.
- Kiểm thử retrieval smoke test bằng câu hỏi tiếng Việt cho cả legal và news.

Không nên thêm reranker, hybrid search hoặc RAGAS vào Task 4; các phần đó thuộc Task 5–8 và dễ làm phạm vi Task 4 khó bàn giao.

---

## 5. Kiểm thử

### 5.1 Test chính thức hiện có

```powershell
python -m pytest tests/test_individual.py::TestTask4 -v -rs
```

`TestTask4` hiện có 4 test method:

| Test | Kiểm tra |
|---|---|
| `test_config_documented` | Chunk size/overlap hợp lệ |
| `test_load_documents_returns_list` | Loader trả list và document có `content` |
| `test_chunk_documents_produces_chunks` | Tạo được chunk từ document |
| `test_chunks_respect_size_limit` | Chunk không vượt tolerance 10% |

Khi Task 4 còn là starter, các test implementation được skip là đúng trạng thái. Sau khi triển khai, mục tiêu là cả 4 test pass.

### 5.2 Test bổ sung nên viết cùng Task 4

- Header canonical được tách khỏi body.
- Chuỗi `Source SHA256:` và `Content Hash:` không xuất hiện trong chunk content.
- News không có heading nhận title làm `section` fallback.
- Chunk ID không trùng và ổn định qua hai lần chạy.
- Rebuild lần hai cho cùng corpus không tăng số record.
- Xóa một document rồi rebuild thì các chunk của document đó biến mất.
- Metadata Chroma đủ title, URL, type, document ID và section để Task 5 trả citation.

### 5.3 Kết quả CLI dự kiến

```text
============================================================
Task 4: Chunking & Indexing
  Chunking: fixed_size (size=800, overlap=100)
  Embedding: BAAI/bge-m3 (dim=1024)
  Vector Store: chromadb
============================================================
Loaded 10 documents
Created M chunks
Embedded M chunks
Indexed M chunks to chroma_db/
```

Không cố định `M` trong test vì số chunk có thể thay đổi khi nội dung corpus hoặc splitter được cập nhật.

---

## 6. Edge cases và troubleshooting

| Tình huống | Hành vi mong đợi |
|---|---|
| `data/standardized/` rỗng | Trả `[]`, CLI kết thúc sạch và không tải model |
| Document không có H2/H3 | Split toàn body; dùng title làm section fallback |
| Nội dung trước heading đầu tiên | Giữ nội dung; dùng title làm section fallback |
| Document ngắn hơn chunk size | Tạo một chunk |
| Document dài không có paragraph | Splitter fallback đến word/character boundary |
| Chạy lại sau khi corpus thay đổi | Rebuild collection hoặc xóa chính xác stale IDs |
| ChromaDB bị lock | Đóng process đang dùng collection rồi chạy lại |
| Model tải chậm ở lần đầu | Đây là tải cache ban đầu; các lần sau dùng cache local |
| Sai embedding dimension | Dừng sớm với lỗi model/config rõ ràng |
| Task 5 không tìm thấy helper | Export helper dùng chung từ module Task 4 |

---

## 7. Definition of Done

Task 4 chỉ được coi là hoàn thành khi:

- 4 test chính thức của `TestTask4` pass, không còn skip vì `NotImplementedError`.
- Test bổ sung về header leakage, section fallback, stable ID và stale cleanup pass.
- Index chứa đúng số document nguồn của corpus và không có record cũ.
- Có thể chạy một semantic-search smoke test bằng Task 5 và nhận citation đúng URL/title.
- Chạy pipeline lần hai cho kết quả nhất quán.
- README chỉ cập nhật hướng dẫn chạy sau khi implementation Task 4 thực sự hoàn tất.
