# Task 5 Spec — Semantic Search và Query Expansion

**Spec cha:** `specq.md`  
**File triển khai chính:** `src/task5_semantic_search.py`  
**File bonus:** `src/task5_query_expansion.py`  
**Điểm mục tiêu:** 6 điểm Task 5 + 5 điểm Semantic Search bonus  
**Trạng thái:** Ready for implementation

---

## 1. Mục tiêu

Triển khai dense semantic retrieval trên ChromaDB đã được tạo ở Task 4. Module phải:

1. Nhận truy vấn tiếng Việt, tiếng Anh hoặc mixed-language.
2. Embed query bằng đúng model và preprocessing đã dùng khi index document.
3. Trả các chunk gần nhất theo cosine similarity.
4. Có output ổn định, đúng format, sort giảm dần và đúng `top_k`.
5. Không load lại embedding model cho từng request.
6. Không crash khi query rỗng, collection chưa tồn tại hoặc collection không có dữ liệu.
7. Có query expansion chạy được để lấy 5 điểm bonus.
8. Có diagnostics đủ để Task 9 calibrate fallback threshold.

Task 5 không chịu trách nhiệm:

- Lexical search: Task 6.
- Fusion dense/sparse và reranking cuối: Task 7/9.
- PageIndex fallback: Task 8/9.
- Sinh câu trả lời: Task 10.

---

## 2. Definition of Done

Task 5 được xem là hoàn thành khi:

- `semantic_search()` không còn `NotImplementedError`.
- Tất cả test `TestTask5` pass, không skip.
- Query in-domain trả kết quả non-empty nếu collection có dữ liệu.
- Mỗi kết quả có `content`, `score`, `metadata`.
- Score được sort giảm dần và giới hạn đúng `top_k`.
- Query và document dùng cùng embedding model/config.
- Model và Chroma client được cache.
- Có test chứng minh query Việt/Anh cùng chủ đề tìm được cùng document phù hợp.
- Query expansion sinh tối đa 3 variants, không làm mất entity/số quan trọng.
- Có A/B baseline vs query expansion trong evaluation hoặc demo report.

Lệnh kiểm tra starter:

```powershell
python -m pytest tests/test_individual.py::TestTask5 -v -rs
```

Kết quả bắt buộc: `4 passed`, không skip.

---

## 3. Phụ thuộc từ Task 4

Task 5 phải tái sử dụng các thành phần từ `src/task4_chunking_indexing.py`:

```python
EMBEDDING_MODEL: str
EMBEDDING_DIM: int
COLLECTION_NAME: str

def get_embedding_model(): ...
def get_collection(): ...
def prepare_query_for_embedding(query: str) -> str: ...
```

Nếu Task 4 chưa có ba helper trên, phải bổ sung chúng ở Task 4 thay vì tạo model/client riêng trong Task 5.

Yêu cầu đối với helper:

- `get_embedding_model()` lazy-load và trả cùng một instance trong một process.
- `get_collection()` mở đúng persistent directory và collection.
- `prepare_query_for_embedding()` thêm prefix nếu model yêu cầu, ví dụ `query:` với E5.
- Document lúc index phải dùng preprocessing tương ứng, ví dụ `passage:` với E5.

Không được:

- Dùng embedding model khác Task 4.
- Tự tạo collection có cùng tên nhưng khác embedding dimension.
- Gọi model API cho từng document trong request path.
- Re-index corpus khi người dùng chỉ đang search.

---

## 4. Public API bắt buộc

### 4.1 `semantic_search`

Giữ nguyên interface starter để tương thích test:

```python
def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return semantic results sorted by cosine similarity descending."""
```

Output:

```python
[
    {
        "content": "...",
        "score": 0.8123,
        "metadata": {
            "chunk_id": "...",
            "document_id": "...",
            "source": "tuition-fees-rmit.md",
            "title": "Tuition Fees",
            "url": "https://...",
            "type": "legal",
            "section": "Payment Structure",
            "chunk_index": 2,
            "language": "en"
        }
    }
]
```

Contract:

- `query` được trim và normalize Unicode.
- Query rỗng trả `[]`.
- `top_k <= 0` trả `[]`.
- Collection chưa tồn tại/rỗng trả `[]` kèm warning đã sanitize.
- Không trả nhiều hơn `top_k`.
- Không trả document rỗng.
- `score` là `float`, không phải NumPy scalar.
- Score round tối đa 6 chữ số thập phân để log/UI ổn định.
- Metadata thiếu từ dữ liệu cũ được thay bằng default hợp lý, không làm crash.

### 4.2 `semantic_search_expanded`

API dùng cho Task 9 và bonus:

```python
def semantic_search_expanded(
    query: str,
    top_k: int = 10,
    max_variants: int = 3,
    history: list[dict] | None = None,
) -> list[dict]:
    """Search original and expanded queries, then fuse by chunk ID."""
```

Output giữ contract của `semantic_search`, bổ sung trong từng result:

```python
{
    "matched_queries": ["query gốc", "variant 1"],
    "raw_scores": {
        "dense": 0.8123,
        "dense_by_query": {
            "query gốc": 0.78,
            "variant 1": 0.8123
        },
        "expansion_rrf": 0.0325
    }
}
```

`score` của output expanded là score dùng sắp hạng của bước expansion fusion. Raw cosine tốt nhất phải được giữ trong `raw_scores.dense` để Task 9 dùng confidence/fallback.

---

## 5. Query preprocessing

Tạo helper private hoặc dùng chung từ `text_utils.py`:

```python
def normalize_query(query: str) -> str: ...
```

Quy tắc:

1. Chấp nhận `str`; type khác gây `TypeError` rõ ràng.
2. Unicode normalize về NFC.
3. Trim đầu/cuối.
4. Thu gọn nhiều whitespace thành một space.
5. Không tự bỏ dấu tiếng Việt trong semantic query.
6. Không lowercase bắt buộc nếu embedding model là cased.
7. Không dịch query gốc trước khi search; translation chỉ là một expansion variant.
8. Giữ nguyên số tiền, năm, mã chương trình và proper nouns.

Ví dụ:

```text
"  Học   phí Business 2026? "
→ "Học phí Business 2026?"
```

---

## 6. Thuật toán semantic search

### 6.1 Luồng chuẩn

```text
query
  → validate/normalize
  → prepare_query_for_embedding
  → embed một vector
  → query Chroma cosine collection
  → distance-to-similarity
  → clean/deduplicate
  → sort descending
  → top_k
```

### 6.2 Pseudocode

```python
def semantic_search(query, top_k=10):
    normalized = normalize_query(query)
    if not normalized or top_k <= 0:
        return []

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    model_input = prepare_query_for_embedding(normalized)
    query_vector = get_embedding_model().encode(
        model_input,
        normalize_embeddings=True,
    )

    n_results = min(top_k, count)
    response = collection.query(
        query_embeddings=[to_plain_list(query_vector)],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = parse_chroma_response(response)
    output = deduplicate_by_chunk_id(output)
    output.sort(key=stable_dense_sort_key, reverse=True)
    return output[:top_k]
```

Tên argument `normalize_embeddings` có thể khác tùy model wrapper. Điều bắt buộc là index/query phải nhất quán.

### 6.3 Chuyển distance thành score

Collection Task 4 phải dùng cosine distance. Khi đó:

```python
similarity = 1.0 - float(distance)
score = max(0.0, min(1.0, similarity))
```

Lý do clamp `[0, 1]`:

- Starter lab và Task 9 dùng threshold trên thang `[0,1]`.
- Negative cosine similarity không hữu ích như evidence confidence trong bài lab.

Không được áp softmax qua các kết quả vì score sau softmax phụ thuộc số candidates và không còn dùng được để calibrate threshold.

### 6.4 Sort và tie-break

Sort key ưu tiên:

1. `score` giảm dần.
2. `metadata.document_id` tăng dần.
3. `metadata.chunk_index` tăng dần.

Tie-break ổn định giúp evaluation tái lập được.

### 6.5 Deduplication

- Khóa chính: `metadata.chunk_id`.
- Nếu thiếu: hash của normalized content + source.
- Khi trùng, giữ result có cosine score cao hơn.

---

## 7. Query Expansion — bonus 5 điểm

### 7.1 File và API

Triển khai tại `src/task5_query_expansion.py`:

```python
def expand_query(
    query: str,
    history: list[dict] | None = None,
    max_variants: int = 3,
) -> list[str]:
    """Return original query first, followed by unique variants."""
```

Contract:

- Query gốc luôn là phần tử đầu tiên.
- Tổng số query không vượt `max_variants`.
- Query rỗng trả `[]`.
- Mỗi variant non-empty và không trùng sau normalize.
- Không làm mất entity, con số, năm hoặc phủ định.
- Deterministic mode luôn khả dụng, không cần API key.

### 7.2 Domain glossary tối thiểu

Glossary được đặt trong config/JSON riêng, không rải hard-code trong nhiều function:

| Tiếng Việt | Tiếng Anh/biến thể |
|---|---|
| học phí | tuition fee, tuition cost |
| thanh toán | payment, payment method, payment structure |
| học bổng | scholarship, financial aid |
| ký túc xá | dormitory, accommodation, housing |
| đăng ký học phần | course registration, enrolment |
| thư viện | library |
| phòng học nhóm | group study room, study room |
| sinh viên quốc tế | international student |

### 7.3 Deterministic expansion

Ví dụ:

```text
Input: "payment structure của học phí như thế nào?"

Variants:
1. "payment structure của học phí như thế nào?"
2. "payment structure của tuition fee như thế nào?"
3. "phương thức thanh toán học phí"
```

Quy tắc:

- Chỉ thay/mở rộng domain terms đã biết.
- Tối đa một bilingual rewrite và một canonical rewrite.
- Không tạo câu dài hơn query gốc quá 2 lần.
- Giữ question intent: amount, deadline, eligibility, process, location.

### 7.4 Optional LLM expansion

LLM expansion chỉ là enhancement, không phải dependency bắt buộc:

- Temperature `0`.
- Yêu cầu JSON list tối đa 3 query.
- Timeout theo config.
- Cache theo query normalized + corpus/profile.
- Validate output trước khi dùng.
- API lỗi hoặc output sai schema → deterministic fallback.

Không dùng HyDE làm mặc định vì tăng latency và có thể thêm fact không có trong query. Nếu bổ sung HyDE, phải đặt sau feature flag và đánh giá riêng.

### 7.5 Fusion các variants

Search mỗi variant với candidate pool lớn hơn output:

```python
per_query_k = max(top_k * 2, 10)
```

Fusion theo RRF trên `chunk_id`:

```text
expansion_rrf(chunk) = Σ 1 / (60 + rank_variant(chunk))
```

Không cộng trực tiếp cosine giữa các query nếu preprocessing/model khác nhau. Giữ cosine tốt nhất trong `raw_scores.dense`.

---

## 8. Caching và hiệu năng

Yêu cầu:

- Embedding model: một instance/process.
- Chroma client/collection: cache theo path + collection name.
- Query embedding: có LRU cache nhỏ theo model ID + normalized query.
- Expanded variants: cache theo query + glossary version.
- Không cache exception vĩnh viễn.
- Cache bị invalid khi embedding model/corpus collection đổi.

Mục tiêu sau warm-up trên local CPU:

- Single semantic query: ưu tiên dưới 1 giây với fast profile.
- Expanded query: ưu tiên dưới 2 giây, không tính lần tải model đầu tiên.

Latency thực tế phải được đo và ghi trong evaluation, không bịa số.

---

## 9. Error handling và logging

### 9.1 Hành vi lỗi

| Tình huống | Hành vi |
|---|---|
| Query rỗng | Trả `[]` |
| `top_k <= 0` | Trả `[]` |
| Chroma chưa có collection | Trả `[]`, warning hướng dẫn chạy Task 4 |
| Collection rỗng | Trả `[]` |
| Model dimension không khớp | Raise lỗi cấu hình rõ ràng |
| Model load/download lỗi | Raise lỗi service rõ ràng cho Task 9/UI bắt |
| Một metadata item thiếu | Dùng default, không bỏ toàn request |
| Expansion API lỗi | Dùng deterministic expansion |

### 9.2 Logging tối thiểu

- Query hash hoặc normalized query trong debug mode.
- Embedding model/profile.
- Collection count.
- Requested và actual `top_k`.
- Best raw dense score.
- Query variants nếu bật expansion.
- Latency embed/query/parse/total.

Không log API key hoặc toàn bộ `.env`.

---

## 10. Test specification

### 10.1 Starter tests bắt buộc

- Return type là list.
- Result có `content` và `score`.
- Score sort giảm dần.
- Không vượt `top_k`.

### 10.2 Unit tests bổ sung

Tạo test bằng fake model và fake Chroma collection, không cần download model:

1. Empty query trả `[]`.
2. `top_k=0` trả `[]`.
3. Collection rỗng trả `[]`.
4. `n_results = min(top_k, collection.count())`.
5. Distance `0.1, 0.4, 0.2` thành score `0.9, 0.6, 0.8` và sort đúng.
6. Distance ngoài thang hữu ích được clamp.
7. NumPy scalar được convert thành Python float.
8. Duplicate `chunk_id` được loại đúng.
9. Metadata thiếu không crash.
10. Tie-break ổn định.
11. Model được cache, không load lại sau hai query.

### 10.3 Query expansion tests

1. Query gốc luôn đứng đầu.
2. Không quá 3 variants.
3. Không duplicate sau normalize.
4. Giữ `Business`, `2026`, số tiền và phủ định.
5. Mixed query tạo bilingual/canonical variant.
6. API expansion lỗi vẫn trả deterministic variants.
7. Expanded search giữ best raw dense cosine.
8. Fusion dedupe bằng chunk ID.

### 10.4 Integration tests

Sau khi Task 4 đã index corpus thật:

- “Học phí chương trình Business” trả tuition document trong top 5.
- “tuition payment structure” trả tuition/payment section trong top 5.
- “hỗ trợ chỗ ở sinh viên” trả accommodation document trong top 5.
- Query Việt và Anh tương đương có ít nhất một expected document chung.
- Query OOD có best dense score thấp hơn phần lớn in-domain queries.

Không hard-code expected answer vào search function để pass integration test.

---

## 11. Quality evaluation

Task 5 phải có tập retrieval labels trong golden dataset:

```json
{
  "question": "...",
  "expected_document_ids": ["rmit-tuition-fees-2026"],
  "expected_sections": ["Payment Structure"]
}
```

Đo riêng:

- Recall@1/3/5.
- MRR@5.
- Best dense score distribution cho in-domain/OOD.
- Latency p50/p95.

A/B bắt buộc cho bonus:

| Config | Mô tả |
|---|---|
| Dense baseline | Chỉ dùng query gốc |
| Dense + expansion | Original + tối đa 2 variants, RRF fusion |

Bonus chỉ được xem là đạt khi có code chạy thật, toggle/demo và ít nhất một phân tích đo lường; không yêu cầu expansion phải thắng mọi câu.

---

## 12. Tích hợp với Task 9

Task 9 dùng:

```python
if enable_query_expansion:
    dense_results = semantic_search_expanded(query, top_k=candidate_k)
else:
    dense_results = semantic_search(query, top_k=candidate_k)
```

Điểm dùng fallback:

```python
best_dense_score = max(
    (r.get("raw_scores", {}).get("dense", r["score"]) for r in dense_results),
    default=0.0,
)
```

Không dùng `expansion_rrf` hoặc RRF cuối pipeline làm threshold.

Task 5 không được import Task 9 để tránh circular dependency.

---

## 13. Nội dung demo Task 5

### Demo baseline

1. Chọn config `dense_only`.
2. Hỏi một câu paraphrase không chứa exact wording của document.
3. Mở source panel, chỉ ra cosine score và expected section.

Query gợi ý:

```text
Chi phí học chương trình Business mỗi năm khoảng bao nhiêu?
```

### Demo query expansion bonus

1. Tắt query expansion và chạy mixed-language query.
2. Bật query expansion, chạy lại.
3. Mở diagnostics cho thấy original query, variants và document rank thay đổi.

Query gợi ý:

```text
payment structure của học phí như thế nào?
```

Điểm cần giải thích:

- Semantic search tìm theo ý nghĩa, không chỉ exact keyword.
- Query expansion xử lý bilingual/domain synonyms.
- RRF dùng để fusion nhiều query variants.
- Raw cosine vẫn được giữ riêng để calibrate fallback.

---

## 14. Checklist triển khai

- [ ] Task 4 có cached model/collection helpers.
- [ ] Query/document preprocessing nhất quán.
- [ ] `semantic_search()` đúng starter signature.
- [ ] Empty/unavailable state trả list, không `NotImplementedError`.
- [ ] Cosine distance được convert đúng và clamp.
- [ ] Output sort giảm dần, stable, deduplicated.
- [ ] Metadata source/document/section được giữ.
- [ ] Model/client/query embedding được cache.
- [ ] `expand_query()` deterministic chạy không cần API.
- [ ] `semantic_search_expanded()` fusion variants và giữ raw cosine.
- [ ] Unit/integration tests pass.
- [ ] Có Recall/MRR baseline.
- [ ] Có A/B baseline vs expansion.
- [ ] Có demo diagnostics cho query variants.

---

## 15. Tiêu chí chấp nhận cuối

Task 5 được nghiệm thu khi:

1. `TestTask5` có `4 passed, 0 skipped`.
2. Search thật trên corpus trả evidence đúng format và source traceable.
3. Model không bị load lại mỗi query.
4. Query Việt/Anh có retrieval hợp lý.
5. Raw dense cosine dùng được cho Task 9 threshold calibration.
6. Query expansion không phụ thuộc bắt buộc vào LLM.
7. Bonus có code, test, toggle, A/B và demo minh bạch.
