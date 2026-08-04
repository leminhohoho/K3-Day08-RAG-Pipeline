# Task 6 Spec — BM25, TF-IDF và Lexical Retrieval

**Spec cha:** `specq.md`  
**File triển khai chính:** `src/task6_lexical_search.py`  
**Điểm mục tiêu:** 6 điểm Task 6 + 5 điểm lexical search bonus  
**Trạng thái:** Ready for implementation

---

## 1. Mục tiêu

Triển khai lexical retrieval trên cùng tập chunk với semantic retrieval. Module phải:

1. Tìm tốt các từ khóa, tên chính sách, mã chương trình, con số và exact phrase.
2. Dùng BM25 làm backend mặc định để đạt Task 6.
3. Dùng TF-IDF character n-gram làm backend thay thế để đạt bonus.
4. Có thể fusion BM25 + TF-IDF mà không cộng trực tiếp raw score khác thang đo.
5. Trả output đúng format, sort giảm dần và đúng `top_k`.
6. Cache corpus/index; không rebuild trong mỗi query.
7. Dùng cùng `chunk_id` và metadata với Task 4/5 để RRF dedupe đúng.
8. Hoạt động được khi không có LLM/API ngoài.

Task 6 không chịu trách nhiệm:

- Embedding/dense search: Task 5.
- Dense/sparse fusion cuối: Task 7/9.
- Semantic cross-encoder: Task 7.
- PageIndex/generation: Task 8–10.

---

## 2. Definition of Done

Task 6 được xem là hoàn thành khi:

- `build_bm25_index()` và `lexical_search()` không còn `NotImplementedError`.
- `TestTask6` có `4 passed, 0 skipped`.
- BM25 index dùng corpus non-empty từ các chunk đã chuẩn hóa.
- Query keyword in-domain có ít nhất một score dương.
- Kết quả có `content`, `score`, `metadata`, sort giảm dần.
- Corpus/index chỉ build lại khi corpus hash đổi.
- TF-IDF char n-gram backend chạy thật.
- Có config/toggle BM25, TF-IDF và BM25+TF-IDF.
- Có A/B hoặc ví dụ đo lường, không chỉ giải thích lý thuyết.

Lệnh kiểm tra starter:

```powershell
python -m pytest tests/test_individual.py::TestTask6 -v -rs
```

Kết quả bắt buộc: `4 passed`, không skip.

---

## 3. Nguồn corpus

Lexical và semantic retrieval phải dùng cùng đơn vị chunk. Không index toàn document cho BM25 trong khi dense search index chunk.

Thứ tự load corpus:

1. Ưu tiên đọc chunks từ Chroma collection Task 4, gồm IDs, documents và metadatas.
2. Nếu collection chưa được build nhưng standardized files tồn tại, có thể gọi `load_documents()` + `chunk_documents()` từ Task 4 để tạo corpus in-memory.
3. Nếu không có corpus, public search trả `[]`, không raise `NotImplementedError`.

API helper đề xuất:

```python
def load_lexical_corpus() -> list[dict]:
    """Return the same chunks and metadata used by dense retrieval."""
```

Corpus item:

```python
{
    "content": str,
    "metadata": {
        "chunk_id": str,
        "document_id": str,
        "source": str,
        "title": str,
        "url": str,
        "type": str,
        "section": str,
        "chunk_index": int,
        "language": str
    }
}
```

Starter biến `CORPUS` không được giữ là list rỗng cố định. Có thể giữ tên để tương thích nhưng phải được quản lý qua lazy index manager.

---

## 4. Public API bắt buộc

### 4.1 BM25

```python
def build_bm25_index(corpus: list[dict]):
    """Build and return a BM25 index for the supplied corpus."""

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Run BM25 search and return results sorted by raw BM25 score."""
```

Output:

```python
[
    {
        "content": "...",
        "score": 6.7821,
        "metadata": {...},
        "retrieval_method": "bm25",
        "raw_scores": {
            "bm25": 6.7821,
            "tfidf": None,
            "rrf": None
        }
    }
]
```

Contract `lexical_search()`:

- Query được normalize/tokenize bằng cùng pipeline với corpus.
- Query rỗng trả `[]`.
- `top_k <= 0` trả `[]`.
- Corpus rỗng trả `[]`.
- Không trả nhiều hơn `top_k`.
- Chỉ trả result có score BM25 dương.
- Score là Python `float`.
- Sort giảm dần, tie-break ổn định.
- Metadata được giữ nguyên.

### 4.2 TF-IDF bonus

```python
def build_tfidf_index(corpus: list[dict]):
    """Return fitted vectorizer and sparse document matrix."""

def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """Run character n-gram TF-IDF cosine search."""
```

Output giống BM25, nhưng:

```python
"retrieval_method": "tfidf"
"raw_scores": {"bm25": None, "tfidf": 0.7142, "rrf": None}
```

### 4.3 Configured lexical search

Task 9/UI dùng wrapper:

```python
def lexical_search_configured(
    query: str,
    top_k: int = 10,
    method: str = "bm25",
) -> list[dict]:
    """method: bm25 | tfidf | bm25_tfidf"""
```

- `bm25`: gọi `lexical_search()`.
- `tfidf`: gọi `tfidf_search()`.
- `bm25_tfidf`: chạy cả hai và fusion theo rank.
- Method không hợp lệ raise `ValueError` rõ ràng.

Starter tests vẫn gọi `lexical_search()`, do đó BM25 luôn là default bắt buộc.

---

## 5. Text normalization và tokenization

### 5.1 Normalize

Tạo helper dùng chung:

```python
def normalize_lexical_text(text: str) -> str: ...
def tokenize_lexical(text: str) -> list[str]: ...
```

Quy tắc:

1. Unicode NFC.
2. Lowercase.
3. Thu gọn whitespace.
4. Chuẩn hóa Unicode punctuation về dạng đơn giản khi phù hợp.
5. Giữ số, năm, mã chương trình và token chứa chữ-số.
6. Không xóa dấu tiếng Việt khỏi content gốc.
7. Có thể thêm token không dấu như alias, nhưng không thay hoàn toàn token có dấu.
8. Không loại stop words mạnh ở baseline.

### 5.2 Tokenizer baseline

Không bắt buộc dependency NLP nặng. Baseline có thể dùng Unicode-aware regex:

```python
TOKEN_PATTERN = r"\w+(?:[-./]\w+)*"
```

Ví dụ:

```text
"Học phí Business 2026–2027"
→ ["học", "phí", "business", "2026", "2027"]
```

Nếu thêm de-accent alias:

```text
["học", "hoc", "phí", "phi", "business", "2026", "2027"]
```

Alias phải được áp dụng nhất quán cho corpus và query. Cần tránh nhân đôi token quá mức làm sai document length normalization.

### 5.3 Optional Vietnamese tokenizer

Có thể thêm `underthesea`/tokenizer khác sau feature flag. Không được biến nó thành dependency bắt buộc nếu deployment không cài được. Nếu dùng, phải benchmark với tokenizer baseline.

---

## 6. BM25 implementation

### 6.1 Cấu hình

Baseline:

```python
BM25_K1 = 1.5
BM25_B = 0.75
```

Dùng `rank_bm25.BM25Okapi`. Phải lưu:

- Corpus items theo đúng thứ tự index.
- Tokenized corpus.
- BM25 model.
- Corpus hash/config version.

### 6.2 Pseudocode build

```python
def build_bm25_index(corpus):
    if not corpus:
        return None

    tokenized = [tokenize_lexical(item["content"]) for item in corpus]
    validate_no_empty_documents(tokenized)

    return BM25Okapi(
        tokenized,
        k1=BM25_K1,
        b=BM25_B,
    )
```

Document rỗng sau tokenize:

- Bỏ khỏi lexical corpus và ghi warning.
- Đồng thời bỏ item tương ứng khỏi corpus mapping để index không lệch.

### 6.3 Pseudocode search

```python
def lexical_search(query, top_k=10):
    normalized = normalize_lexical_text(query)
    if not normalized or top_k <= 0:
        return []

    state = get_lexical_index_state()
    if not state.corpus or state.bm25 is None:
        return []

    query_tokens = tokenize_lexical(normalized)
    scores = state.bm25.get_scores(query_tokens)

    ranked_indices = stable_rank_indices(scores)
    return build_results(
        ranked_indices,
        scores,
        method="bm25",
        top_k=top_k,
        positive_only=True,
    )
```

### 6.4 Sort/tie-break

Ưu tiên:

1. BM25 score giảm dần.
2. Số exact query tokens xuất hiện trong chunk giảm dần.
3. `document_id` tăng dần.
4. `chunk_index` tăng dần.

Không thêm exact-match count trực tiếp vào raw BM25 score. Nó chỉ là tie-break để score vẫn có ý nghĩa.

---

## 7. TF-IDF character n-gram — bonus 5 điểm

### 7.1 Cấu hình mặc định

```python
from sklearn.feature_extraction.text import TfidfVectorizer

TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
    sublinear_tf=True,
    norm="l2",
    lowercase=True,
)
```

TF-IDF input dùng normalized text, không dùng BM25 token list.

### 7.2 Lý do chọn `char_wb`

- Không phụ thuộc hoàn toàn vào word segmentation tiếng Việt.
- Match được biến thể từ, dấu câu và một phần typo.
- Hữu ích với tên chương trình/mã/phrase.
- Document/query vector đã L2 normalize nên dot product là cosine similarity.
- Thư viện `scikit-learn` đã có trong requirements.

### 7.3 Build

Index state gồm:

```python
{
    "vectorizer": TfidfVectorizer,
    "document_matrix": sparse_matrix,
    "corpus": list[dict],
    "corpus_hash": str
}
```

Pseudocode:

```python
texts = [normalize_lexical_text(x["content"]) for x in corpus]
matrix = vectorizer.fit_transform(texts)
```

Corpus quá nhỏ hoặc toàn empty phải trả state unavailable, public search trả `[]`.

### 7.4 Search

```python
query_vector = vectorizer.transform([normalize_lexical_text(query)])
scores = document_matrix @ query_vector.T
```

- Flatten về array 1 chiều.
- Chỉ trả score `> 0`.
- Sort giảm dần, tie-break giống BM25.
- Score clamp `[0,1]` để tránh sai số floating point nhỏ.

### 7.5 Điểm cần giải thích trong demo

BM25:

- Dựa vào TF, IDF và document length normalization.
- Term frequency có saturation qua `k1`.
- `b` điều chỉnh mức phạt document dài.
- Raw score không bị giới hạn trên.

TF-IDF char n-gram:

- Biểu diễn document bằng các chuỗi ký tự 3–5.
- TF dùng log scaling khi `sublinear_tf=True`.
- IDF làm n-gram hiếm quan trọng hơn.
- L2 normalize và cosine similarity cho score `[0,1]`.
- Bền hơn với word segmentation/typo nhưng có thể match nhiễu ký tự.

---

## 8. Fusion BM25 + TF-IDF

Không cộng trực tiếp raw BM25 và TF-IDF vì khác thang điểm.

Phương án mặc định: RRF theo rank.

```text
rrf(chunk) = 1 / (60 + rank_bm25) + 1 / (60 + rank_tfidf)
```

Pseudocode:

```python
bm25_results = lexical_search(query, top_k=top_k * 2)
tfidf_results = tfidf_search(query, top_k=top_k * 2)
fused = rerank_rrf([bm25_results, tfidf_results], top_k=top_k)
```

Output:

```python
{
    "score": rrf_score,
    "retrieval_method": "bm25_tfidf_rrf",
    "raw_scores": {
        "bm25": float | None,
        "tfidf": float | None,
        "rrf": float
    }
}
```

Dedupe bằng `chunk_id`, không chỉ bằng toàn bộ content string.

Task 9 sau đó vẫn có thể fusion lexical results với dense results. Phải giữ raw scores để diagnostics không nhầm RRF với confidence score.

---

## 9. Index manager và caching

Thay `CORPUS = []` static bằng lazy state manager, ví dụ:

```python
@dataclass
class LexicalIndexState:
    corpus: list[dict]
    corpus_hash: str
    bm25: object | None
    tfidf_vectorizer: object | None
    tfidf_matrix: object | None
```

API đề xuất:

```python
def get_lexical_index_state(force_rebuild: bool = False) -> LexicalIndexState: ...
def clear_lexical_cache() -> None: ...
```

Yêu cầu:

- Build một lần/process sau warm-up.
- Rebuild khi corpus hash/tokenizer config/TF-IDF config đổi.
- Thread-safe nếu dense/sparse chạy song song trong Task 9.
- Không mutate corpus item trong search.
- Không pickle/load object từ nguồn không tin cậy.

Corpus hash tối thiểu dựa trên ordered `chunk_id + content_hash`.

---

## 10. Error handling và logging

### 10.1 Hành vi lỗi

| Tình huống | Hành vi |
|---|---|
| Query rỗng | Trả `[]` |
| `top_k <= 0` | Trả `[]` |
| Không có corpus | Trả `[]`, warning hướng dẫn chạy Task 3/4 |
| BM25 index chưa build | Lazy build; thất bại thì trả `[]` hoặc raise config error rõ ràng |
| TF-IDF vocabulary rỗng | TF-IDF unavailable, không làm BM25 crash |
| Một chunk rỗng | Loại chunk đó và giữ mapping đúng |
| Method không hợp lệ | `ValueError` |
| Metadata thiếu | Dùng default, không crash toàn request |

Không dùng `NotImplementedError` cho trạng thái runtime bình thường.

### 10.2 Logging

- Corpus size/hash.
- Tokenizer/config version.
- Index build latency.
- Query token count.
- Method và requested/actual `top_k`.
- Best BM25/TF-IDF score.
- Search latency.

Không log API key; Task 6 không cần external API.

---

## 11. Test specification

### 11.1 Starter tests bắt buộc

- Return list.
- Result có `content`, `score`.
- Sort giảm dần.
- Ít nhất một keyword match có score dương.

### 11.2 BM25 unit tests

Dùng corpus fixture nhỏ, không phụ thuộc dữ liệu thật:

```python
[
    {"content": "Tuition fee payment schedule", "metadata": {"chunk_id": "c1"}},
    {"content": "Scholarship eligibility requirements", "metadata": {"chunk_id": "c2"}},
    {"content": "Library group study room booking", "metadata": {"chunk_id": "c3"}}
]
```

Test cases:

1. `build_bm25_index([])` xử lý an toàn.
2. Empty query trả `[]`.
3. `top_k=0` trả `[]`.
4. Query `tuition fee` xếp `c1` đầu và score dương.
5. Query `scholarship eligibility` xếp `c2` đầu.
6. Không vượt `top_k`.
7. Score là Python float.
8. Sort/tie-break ổn định.
9. Metadata được giữ.
10. Chunk empty không làm lệch corpus-index mapping.
11. Index cache không rebuild sau hai query cùng corpus.

### 11.3 Tokenization tests

1. Unicode tiếng Việt được giữ.
2. Case normalization nhất quán.
3. Multiple spaces được thu gọn.
4. Giữ `2026`, `BUS1234` và tên chương trình.
5. Alias không dấu hoạt động nhất quán nếu bật.
6. Punctuation-only input trả no tokens.

### 11.4 TF-IDF tests

1. Vectorizer/matrix có đúng số document rows.
2. Exact/near phrase xếp document phù hợp đầu.
3. TF-IDF score thuộc `[0,1]`.
4. Query vocabulary không khớp trả `[]`.
5. Typo nhỏ hoặc biến thể segmentation vẫn tìm được nhờ char n-gram.
6. TF-IDF unavailable không làm BM25 backend lỗi.

### 11.5 Fusion tests

1. Không cộng raw BM25 + TF-IDF.
2. Chunk xuất hiện trong cả hai list có RRF score cao hơn khi rank tương đương.
3. Dedupe bằng chunk ID.
4. Giữ raw BM25/TF-IDF scores.
5. Không vượt `top_k`.

### 11.6 Integration tests

Trên corpus thật:

- Query chứa `tuition fee` trả tuition document top 5.
- Query chứa tên scholarship cụ thể trả scholarship document top 5.
- Query `library study room` trả library/service document top 5 nếu corpus có.
- Query có năm/số/mã giữ được exact evidence.
- BM25 và TF-IDF cùng dùng chunk IDs của Chroma corpus.

---

## 12. Quality evaluation và A/B bonus

Đo riêng retrieval quality trên golden dataset:

- Recall@1/3/5.
- MRR@5.
- nDCG@5 nếu có graded relevance.
- Empty result rate.
- Latency p50/p95.

Configs:

| Config | Backend |
|---|---|
| L1 | BM25 |
| L2 | TF-IDF char n-gram |
| L3 | BM25 + TF-IDF RRF |

Phân tích theo category:

- Exact policy/entity.
- Number/year/code.
- Vietnamese phrase.
- Mixed-language.
- Typo/spacing variation.

Không yêu cầu TF-IDF thắng BM25 mọi metric. Báo cáo phải giải thích loại query nào mỗi phương pháp tốt hơn.

Bonus được xem là đạt khi:

1. TF-IDF chạy thật.
2. Có UI/config toggle.
3. Có test.
4. Có A/B hoặc case study thực tế.
5. Demo giải thích đúng cơ chế và trade-off.

---

## 13. Tích hợp Task 7 và Task 9

Task 9 gọi:

```python
sparse_results = lexical_search_configured(
    query,
    top_k=candidate_k,
    method=config.lexical_method,
)
```

Sau đó Task 7 fusion với dense:

```python
hybrid = rerank_rrf(
    [dense_results, sparse_results],
    top_k=rerank_candidate_k,
)
```

Quy tắc:

- Task 6 giữ `chunk_id` và raw scores.
- Task 6 không quyết định PageIndex fallback.
- BM25/TF-IDF raw score không được dùng trực tiếp chung threshold với cosine.
- RRF score chỉ dùng ranking.
- Task 6 không import Task 9.

Nếu `bm25_tfidf` đã fusion nội bộ, Task 9 coi đó là một lexical ranked list duy nhất.

---

## 14. Nội dung demo Task 6

### Demo BM25

1. Chọn `BM25`.
2. Dùng query có tên chính sách/exact phrase hoặc con số trong corpus.
3. Cho xem raw BM25 score và source section.
4. Giải thích TF, IDF, `k1`, `b`, document length normalization.

### Demo TF-IDF bonus

1. Chuyển sang `TF-IDF char n-gram`.
2. Dùng cùng query hoặc query có spacing/typo nhẹ.
3. Cho xem rank/score thay đổi.
4. Giải thích n-gram 3–5, sublinear TF, IDF, L2/cosine.

### Demo fusion

1. Chọn `BM25 + TF-IDF`.
2. Mở diagnostics hiển thị `bm25`, `tfidf`, `rrf` riêng.
3. Giải thích không cộng raw scores vì khác thang đo.

Query phải được chọn từ corpus đã thu thập thật. Không hard-code query demo vào search logic.

---

## 15. Checklist triển khai

- [ ] Corpus Task 6 cùng chunks/IDs với Task 4/5.
- [ ] `CORPUS` không còn static empty state.
- [ ] Normalizer/tokenizer Unicode-aware.
- [ ] BM25 index lazy/cache theo corpus hash.
- [ ] `lexical_search()` đúng starter signature.
- [ ] Empty/unavailable state trả list, không `NotImplementedError`.
- [ ] Score sort giảm dần, positive-only, stable tie-break.
- [ ] Metadata và raw BM25 score được giữ.
- [ ] TF-IDF char n-gram backend chạy thật.
- [ ] TF-IDF score cosine thuộc `[0,1]`.
- [ ] BM25+TF-IDF dùng RRF, không cộng raw score.
- [ ] UI/config có ba lexical methods.
- [ ] Unit/integration tests pass.
- [ ] Có Recall/MRR và latency.
- [ ] Có A/B/case study cho bonus.
- [ ] Có demo giải thích cơ chế/trade-off.

---

## 16. Tiêu chí chấp nhận cuối

Task 6 được nghiệm thu khi:

1. `TestTask6` có `4 passed, 0 skipped`.
2. BM25 query thật trả keyword evidence đúng format và source traceable.
3. Index không rebuild mỗi query.
4. Lexical corpus khớp dense corpus bằng stable chunk IDs.
5. TF-IDF backend hoạt động độc lập và có score hợp lệ.
6. BM25+TF-IDF fusion giữ được raw scores và không trộn thang điểm sai.
7. Bonus có code, test, toggle, A/B và demo minh bạch.

