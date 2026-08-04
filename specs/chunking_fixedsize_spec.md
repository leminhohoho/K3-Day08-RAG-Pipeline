# Spec — Đổi chunking strategy từ `markdown_section_recursive` sang `fixed_size`

**Trạng thái:** Planning — chưa triển khai (spec-only)
**File bị ảnh hưởng chính:** `src/task4_chunking_indexing.py`
**File spec liên quan:** `specs/task4_spec.md`, `specs/todo.md`
**Test bị ảnh hưởng:** `tests/test_individual.py::TestTask4` (và downstream Task 5/6/7/9/10)

---

## 1. Mục tiêu

Thay đổi chiến lược chunking hiện tại từ **heading-aware + recursive split**
(`markdown_section_recursive`) sang **fixed-size split** với:

- `CHUNK_SIZE = 800` (giữ nguyên)
- `CHUNK_OVERLAP = 100` (giữ nguyên)
- `CHUNKING_METHOD = "fixed_size"`

Mục tiêu là giảm độ phức tạp của bước chunking, đơn giản hóa phần metadata
`section`/`section_path`, và đảm bảo mọi chunk có độ dài xấp xỉ đồng đều
(multiple-of-size) thay vì theo cấu trúc heading.

---

## 2. Trạng thái hiện tại (baseline)

### 2.1 Chiến lược đang dùng

`src/task4_chunking_indexing.py` hiện có:

- `CHUNKING_METHOD = "markdown_section_recursive"`
- `_split_by_headings(content)` — tách theo H1/H2/H3
- `_build_section_path(heading, title, has_any_heading)` — gán section_path
- Trong `chunk_documents()`:
  - Section ≤ `CHUNK_SIZE * 1.1` → 1 chunk (giữ nguyên nội dung)
  - Section dài → `RecursiveCharacterTextSplitter` với
    `separators=["\n\n", "\n", ". ", " ", ""]`
- Metadata chunk gồm `section`, `section_path`, `chunk_index`, `chunk_id`,
  `chunk_hash`.

### 2.2 Đặc điểm hành vi hiện tại

- Chunk giữ nguyên cấu trúc heading (không bị cắt giữa mục).
- Độ dài chunk **không đồng đều**: section ngắn → chunk ngắn, section dài →
  nhiều chunk.
- `section` lấy từ heading gần nhất (hoặc title fallback).
- Có nhánh code trùng lặp (một cho section ngắn, một cho section dài).

---

## 3. Trạng thái mục tiêu

### 3.1 Cấu hình

| Constant | Hiện tại | Mục tiêu |
|---|---|---|
| `CHUNK_SIZE` | `800` | `800` (giữ nguyên) |
| `CHUNK_OVERLAP` | `100` | `100` (giữ nguyên) |
| `CHUNKING_METHOD` | `"markdown_section_recursive"` | `"fixed_size"` |

`CHUNK_SIZE` và `CHUNK_OVERLAP` **không đổi** — chỉ thay đổi *cách* cắt.
`CHUNKING_METHOD` được đổi thành const string `"fixed_size"` để dùng trong log
và manifest config.

### 3.2 Thuật toán `fixed_size`

Mỗi document được cắt thành các đoạn có độ dài tối đa `CHUNK_SIZE` ký tự,
offset `CHUNK_OVERLAP` ký tự giữa hai chunk liên tiếp.

```
chunk_0: [0               : 800]
chunk_1: [700 (800-100)   : 1500]
chunk_2: [1400 (1500-100) : 2200]
...
```

### 3.3 Công cụ triển khai đề xuất

Ưu tiên dùng `CharacterTextSplitter` từ `langchain_text_splitters`
(đã có dependency, cùng package với `RecursiveCharacterTextSplitter`):

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separator="\n\n",   # hoặc "" để cắt theo ký tự thô
)
```

Lưu ý: `CharacterTextSplitter` theo mặc định cắt theo `separator` và `length_function`
mặc định là `len`. Nếu cần cắt đúng theo số ký tự Unicode, đảm bảo
`length_function=len` (mặc định) và chọn `separator` phù hợp.

> **Quyết định cần chốt (xem mục 7):** separator dùng `"\n\n"` (tôn trọng dấu
> xuống dòng nhưng có thể tạo chunk ngắn hơn 800) hay `""` (cắt cứng theo đúng
> 800 ký tự, không tôn trọng boundary). Lựa chọn này ảnh hưởng trực tiếp tới
> test `test_chunks_respect_size_limit` và độ đồng đều chunk.

---

## 4. Thay đổi cần làm trong `src/task4_chunking_indexing.py`

### 4.1 Cấu hình

```python
# Trước
CHUNKING_METHOD = "markdown_section_recursive"  # heading-aware + recursive split

# Sau
CHUNKING_METHOD = "fixed_size"
```

### 4.2 Xóa các helper không còn dùng

- `_split_by_headings(content)`
- `_build_section_path(heading, title, has_any_heading)`

Nếu không còn nơi nào khác gọi hai hàm này, phải xóa kèm để tránh dead code.
Trước khi xóa, kiểm tra bằng:

```bash
rg -n "_split_by_headings|_build_section_path" src/ tests/
```

(Chỉ xóa khi không còn tham chiếu — theo nguyên tắc "đừng xóa code có chủ đích").

### 4.3 Viết lại `chunk_documents()`

Thay toàn bộ logic heading-aware bằng fixed-size split. Bản nháp:

```python
def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents theo fixed_size strategy (size, overlap)."""
    from langchain_text_splitters import CharacterTextSplitter

    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n\n",  # hoặc "" theo quyết định mục 7
    )

    chunks = []
    chunk_index = 0
    for doc in documents:
        meta = doc["metadata"]
        content = doc["content"]
        title = meta.get("title", meta.get("document_id", "unknown"))

        splits = splitter.split_text(content)
        for split_text in splits:
            if not split_text.strip():
                continue
            chunk_id = f"{meta['document_id']}_chunk_{chunk_index}"
            chunk_hash = hashlib.sha256(split_text.encode()).hexdigest()[:12]
            chunks.append({
                "content": split_text,
                "metadata": {
                    **meta,
                    "section": title,          # fallback: title
                    "section_path": "",        # không còn phân cấp heading
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                    "chunk_hash": chunk_hash,
                },
            })
            chunk_index += 1

    return chunks
```

### 4.4 Cập nhật metadata `section`/`section_path`

- `section_path` → để `""` (không còn phân cấp heading).
- `section` → giữ `title` làm fallback (để Task 10 citation vẫn có section label).
  Giữ `section` không rỗng để không phá contract citation.

> **Quyết định cần chốt (mục 7):** Nếu muốn giữ thông tin heading cho citation,
> cần tách heading trước khi split rồi gán heading làm section cho các chunk
> trong phạm vi đó. Đây là "fixed_size + heading label" — phức tạp hơn fixed-size
> thuần túy. Nếu không cần, dùng `section = title` (đơn giản nhất).

### 4.5 Cập nhật `run_pipeline()` log

Dòng log hiện tại:

```python
print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
```

Tự động in đúng `CHUNKING_METHOD = "fixed_size"` vì đọc từ constant — không cần
sửa. Xác nhận output dự kiến:

```text
Chunking: fixed_size (size=800, overlap=100)
```

---

## 5. Thay đổi cần làm trong `specs/todo.md`

Todo.md hiện khai báo `CHUNKING_METHOD="recursive"` trong mục Task 4. Cần đồng bộ:

```markdown
- [ ] Set `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, `CHUNKING_METHOD="recursive"`
```

→

```markdown
- [ ] Set `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`, `CHUNKING_METHOD="fixed_size"`
```

`specs/task4_spec.md` cũng mô tả chiến lược `markdown_section_recursive` ở mục
2.1 Config và 3.2 chunk_documents. Cần cập nhật toàn bộ mô tả về heading-aware
chunking sang fixed-size nếu muốn spec phản ánh trạng thái mới.

---

## 6. Tác động tới test

### 6.1 `TestTask4` — không phá vỡ

`tests/test_individual.py::TestTask4` có 4 test:

| Test | Tác động với fixed_size |
|---|---|
| `test_config_documented` | Không đổi. Kích thước hợp lệ vẫn giữ. |
| `test_load_documents_returns_list` | Không đổi. |
| `test_chunk_documents_produces_chunks` | Vẫn pass (vẫn tạo ≥1 chunk nếu doc non-empty). |
| `test_chunks_respect_size_limit` | **Cần kiểm tra lại.** Nếu dùng `separator="\n\n"` thì chunk có thể ngắn hơn 800 (dưới limit, vẫn pass). Nếu dùng `separator=""` thì cắt đúng 800, vẫn ≤ 880. Trong cả hai trường hợp đều pass, nhưng phải chạy lại để xác nhận. |

Không có test hiện tại nào *bắt buộc* `section`/`section_path` mang giá trị heading,
nên nhóm test Task 4 không có assert mới cần thêm. Tuy nhiên nên thêm test cho
hành vi fixed-size mới (mục 6.3).

### 6.2 Downstream tests (Task 5/6/7/9/10)

- `TestTask5/6/7/9/10` không assert `section` cụ thể; chúng dùng `chunk_id` làm
  dedup key và `content`/`score`/`source`. Fixed-size giữ nguyên `chunk_id`
  format (`{document_id}_chunk_{i}`) và `content`, nên **không phá vỡ** các test này.
- Lưu ý: nếu `chunk_id` giữ nguyên format thì global `chunk_index` vẫn tăng liên
  tục qua các document — đúng như hiện tại, không cần đổi.

### 6.3 Test mới nên viết (khi triển khai, không phải hiện tại)

Thêm vào `TestTask4` để bảo vệ hành vi fixed-size:

1. Chunk có độ dài ≤ `CHUNK_SIZE` (không cần heading).
2. Chunk liên tiếp overlap đúng `CHUNK_OVERLAP` (đoạn cuối chunk i trùng
   đoạn đầu chunk i+1).
3. Với document dài, số chunk ≈ `ceil(len / (CHUNK_SIZE - CHUNK_OVERLAP))`.
4. Metadata `section` không rỗng (fallback title).
5. `chunk_id` không trùng và ổn định qua hai lần chạy.

---

## 7. Quyết định cần chốt trước khi triển khai

| # | Câu hỏi | Lựa chọn | Ảnh hưởng |
|---|---|---|---|
| 1 | `separator` của `CharacterTextSplitter` | `"\n\n"` (tôn trọng dòng) vs `""` (cắt cứng 800 ký tự) | Độ đồng đều chunk, kết quả size test, độ chính xác retrieval |
| 2 | Giá trị `section` cho chunk | `title` (đơn giản) vs giữ heading thật (fixed-size + heading label) | Citation quality ở Task 10, độ phức tạp code |
| 3 | Có xóa `_split_by_headings`/`_build_section_path` không | Xóa (nếu không còn tham chiếu) vs giữ lại | Dead code, độ sạch codebase |
| 4 | Có cập nhật `specs/task4_spec.md` không | Có (đồng bộ) vs chỉ update todo.md | Tính nhất quán tài liệu |

---

## 8. Migration & verification plan

### 8.1 Trình tự triển khai (khi được duyệt)

1. Cập nhật `CHUNKING_METHOD = "fixed_size"` trong config.
2. Viết lại `chunk_documents()` dùng `CharacterTextSplitter`.
3. Xóa 2 helper heading-aware (sau khi xác nhận không còn tham chiếu).
4. Cập nhật metadata `section`/`section_path` theo quyết định mục 7.
5. Đồng bộ `specs/todo.md` (và `specs/task4_spec.md` nếu chốt).
6. Thêm test mới cho fixed-size (mục 6.3).
7. **Rebuild index** (bắt buộc vì `chunk_id`/content đổi):
   ```bash
   python src/task4_chunking_indexing.py
   ```
   (Pipeline tự xóa + tạo lại collection nên không cần xóa thủ công.)
8. Chạy toàn bộ test:
   ```bash
   python -m pytest tests/test_individual.py -v -rs
   ```

### 8.2 Tiêu chí chấp nhận

- `TestTask4` 4 test pass, không skip.
- Tất cả downstream test (Task 5/6/7/9/10) vẫn pass sau rebuild.
- CLI in đúng `Chunking: fixed_size (size=800, overlap=100)`.
- `chroma_db/` collection chứa chunk mới (content đã đổi theo fixed-size).
- Một smoke test semantic search (Task 5) trả về kết quả hợp lệ với `chunk_id`
  mới.

### 8.3 Rủi ro / lưu ý

- **Thay đổi `chunk_id`/`content` → phải rebuild index.** Dữ liệu Chroma cũ
  (theo heading) sẽ stale nếu không rebuild.
- **Retrieval quality có thể đổi**: fixed-size có thể cắt giữa ý/mục, giảm
  độ mạch lạc ngữ nghĩa của chunk so với section-aware. Cần so sánh recall
  (nếu có golden dataset) trước/sau.
- **`CharacterTextSplitter` cắt theo `separator`**: nếu `separator="\n\n"`,
  chunk có thể ngắn hơn 800; nếu `""`, cắt cứng. Quyết định này ảnh hưởng tới
  test size và chất lượng chunk.

---

## 9. Không nằm trong phạm vi

- Không đổi `CHUNK_SIZE`/`CHUNK_OVERLAP` (giữ 800/100).
- Không đổi embedding model / Chroma / collection name.
- Không đổi logic Task 5–10.
- Không triển khai code — đây là spec để lập kế hoạch.
