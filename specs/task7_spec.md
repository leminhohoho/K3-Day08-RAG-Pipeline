# Task 7 Spec — Weighted Fusion, Reranking và Diversity

**Spec cha:** `../../../../specq.md`  
**File triển khai chính:** `src/task7_reranking.py`  
**Điểm mục tiêu:** 6 điểm Task 7  
**Vai trò mở rộng:** nâng chất lượng hybrid retrieval và tạo cấu hình A/B nổi bật khi demo  
**Trạng thái:** Ready for implementation

---

## 1. Mục tiêu

Task 7 chịu trách nhiệm hợp nhất và sắp hạng lại candidates do các retriever tạo ra. Module phải:

1. Fusion dense, BM25 và TF-IDF mà không cộng trực tiếp các raw score khác thang đo.
2. Dùng RRF/Weighted RRF làm baseline local, deterministic, không cần API key.
3. Dedupe bằng stable `chunk_id` từ Task 4.
4. Giữ toàn bộ provenance và raw score từ Task 5–6.
5. Không dùng fusion score như confidence hoặc fallback threshold.
6. Có optional cross-encoder để tăng precision khi có model/API.
7. Có optional MMR để giảm trùng lặp trong context cuối.
8. Có fallback ổn định khi optional backend không khả dụng.
9. Không mutate candidates do caller truyền vào.
10. Tương thích starter tests và contract Task 9.

Task 7 không chịu trách nhiệm:

- Sinh dense/BM25/TF-IDF candidates: Task 5–6.
- Quyết định PageIndex fallback: Task 9.
- Sinh câu trả lời/citation: Task 10.
- Dùng RRF score để kết luận evidence có đủ hay không.

---

## 2. Quyết định kiến trúc

Pipeline mặc định:

```text
Dense ranked list ──┐
BM25 ranked list ───┼── Weighted RRF một lần ── top candidate pool
TF-IDF ranked list ─┘                            │
                                                ├── baseline: giữ RRF order
                                                ├── quality: cross-encoder
                                                └── diverse: cross-encoder → MMR
```

Nguyên tắc bắt buộc:

- RRF là **rank fusion**, không phải semantic reranker.
- Task 9 chỉ gọi fusion một lần trên các ranked lists gốc.
- Sau fusion, chỉ gọi cross-encoder/MMR nếu bật.
- Không gọi `rerank(..., method="rrf")` lần nữa trên output đã được RRF fusion.
- `score` sau RRF là ranking score; confidence vẫn lấy từ raw dense score của Task 5.

`BM25 + TF-IDF` có thể được fusion riêng trong Task 6 để demo độc lập, nhưng hybrid production path ưu tiên đưa cả ba ranked lists vào một lần Weighted RRF ở Task 7.

---

## 3. Definition of Done

Task 7 được xem là hoàn thành khi:

- `rerank_rrf()` hoạt động thật và đúng công thức.
- `rerank()` mặc định không còn `NotImplementedError`.
- `TestTask7` có `3 passed, 0 skipped`.
- Empty candidates trả `[]`.
- Output không vượt `top_k` và có `score`.
- Dedupe ưu tiên `chunk_id`, không chỉ dùng raw content.
- Raw dense/BM25/TF-IDF scores được giữ sau fusion/rerank.
- Weighted RRF xử lý được 1–3 ranked lists.
- Tie-break deterministic.
- Cross-encoder/MMR lỗi không làm pipeline mặc định crash.
- Có test chứng minh Task 9 không dùng RRF score làm threshold.
- Có A/B `fusion_only` và `fusion + reranker` hoặc phân tích rõ vì sao optional reranker không được bật.

Lệnh starter test:

```powershell
python -m pytest tests/test_individual.py::TestTask7 -v -rs
```

Kết quả bắt buộc: `3 passed`, không skip.

---

## 4. Candidate contract đầu vào

Task 7 chấp nhận candidates theo schema thống nhất:

```python
{
    "content": str,
    "score": float,
    "score_type": "cosine" | "bm25" | "tfidf" | "rrf" | "reranker",
    "confidence_score": float | None,
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
    },
    "raw_scores": {
        "dense": float | None,
        "bm25": float | None,
        "tfidf": float | None,
        "rrf": float | None,
        "reranker": float | None,
        "mmr": float | None
    }
}
```

Backward compatibility:

- Starter candidates chỉ có `content`, `score`, `metadata`; Task 7 phải xử lý được.
- Nếu thiếu `raw_scores`, tạo dict mới mà không sửa object gốc.
- Nếu thiếu `chunk_id`, tạo dedupe key bằng normalized content hash + source.
- Nếu thiếu metadata, dùng `{}`.

Không được làm mất:

- `confidence_score` từ dense retrieval.
- Source URL/section.
- Native raw score của từng retriever.
- Các diagnostics như `matched_queries`.

---

## 5. Public API

### 5.1 `rerank_rrf`

Giữ starter signature và thêm optional weights ở cuối để không phá compatibility:

```python
def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict]:
    """Fuse ranked lists using weighted reciprocal rank fusion."""
```

Contract:

- `ranked_lists=[]` trả `[]`.
- Bỏ các list rỗng nhưng giữ alignment với weights.
- `top_k <= 0` trả `[]`.
- `k > 0`; giá trị khác raise `ValueError`.
- Nếu `weights is None`, mọi list có weight `1.0`.
- `len(weights)` phải bằng `len(ranked_lists)`.
- Weight phải hữu hạn và `>=0`.
- Output sort giảm dần theo RRF score.
- Không mutate input.

### 5.2 `rerank_cross_encoder`

```python
def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Score query-document pairs and return semantic relevance order."""
```

Contract:

- Query rỗng hoặc candidates rỗng trả `[]`.
- Chỉ rerank candidate pool, không search toàn corpus.
- Giữ metadata/raw scores.
- Ghi cross-encoder score vào `raw_scores.reranker`.
- Output `score=reranker_score`, `score_type="reranker"`.
- Không vượt `top_k`.
- Backend/network lỗi raise exception nội bộ rõ ràng; unified `rerank()` quyết định fallback.

### 5.3 `rerank_mmr`

```python
def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Select relevant but non-redundant candidates."""
```

Contract:

- `0 <= lambda_param <= 1`.
- Candidates phải có embedding tương thích.
- Không đủ embedding/dimension mismatch: raise `ValueError` rõ ràng.
- Không chọn một chunk hai lần.
- Ghi selection score vào `raw_scores.mmr`.
- Giữ original relevance/reranker scores.

### 5.4 Unified `rerank`

Giữ đúng starter signature:

```python
def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """Unified single-list reranking interface."""
```

Hành vi:

- `method="rrf"`: gọi `rerank_rrf([candidates], top_k)` để starter test pass; đây chỉ là stable rank normalization cho một list.
- `method="cross_encoder"`: gọi cross-encoder; unavailable thì fallback giữ prior order nếu `strict=False` trong config.
- `method="mmr"`: lấy query embedding/candidate embeddings qua Task 4–5 helper; thiếu dữ liệu thì fallback hoặc raise theo config.
- `method="none"`: giữ prior order và cắt `top_k`.
- Method khác raise `ValueError`.
- Empty candidates trả `[]` cho mọi method.

Task 9 không dùng `rerank(..., method="rrf")` sau khi đã gọi `rerank_rrf()`.

---

## 6. Weighted RRF

### 6.1 Công thức

```text
RRF(d) = Σᵣ weightᵣ / (k + rankᵣ(d))
```

Rank bắt đầu từ 1.

Cấu hình mặc định đề xuất:

```python
RRF_K = 60
RRF_WEIGHTS = {
    "dense": 1.0,
    "bm25": 0.9,
    "tfidf": 0.7,
}
```

Các weights này là baseline, không phải chân lý. Chỉ thay đổi qua A/B evaluation, không chỉnh theo từng câu demo.

### 6.2 Vì sao Weighted RRF

- Không yêu cầu normalize cosine, BM25 và TF-IDF về cùng thang điểm.
- Robust khi một retriever có raw score phân phối khác.
- Có thể ưu tiên dense/BM25 mà vẫn nhận tín hiệu bổ sung từ TF-IDF.
- Dễ giải thích trong demo.

### 6.3 Dedupe key

```python
def candidate_key(item: dict) -> str:
    chunk_id = item.get("metadata", {}).get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"
    return "fallback:" + sha256(
        normalize_content(item.get("content", ""))
        + "|"
        + item.get("metadata", {}).get("source", "")
    )
```

Không chỉ dùng toàn bộ content làm dictionary key vì:

- Content dài tốn memory.
- Whitespace khác nhau gây duplicate giả.
- Hai source có cùng đoạn boilerplate cần provenance riêng.

### 6.4 Merge provenance/raw scores

Khi cùng chunk xuất hiện ở nhiều list:

- Base item lấy từ occurrence có metadata đầy đủ nhất.
- Merge `raw_scores` theo retriever.
- Ghi `matched_rankers`, ví dụ `["dense", "bm25"]`.
- Ghi `ranks`, ví dụ `{"dense": 1, "bm25": 3}`.
- `score=rrf_score`.
- `score_type="rrf"`.
- `raw_scores.rrf=rrf_score`.
- `confidence_score` lấy max raw dense confidence của occurrences, không lấy RRF.

### 6.5 Tie-break deterministic

Nếu hai chunk có RRF score bằng nhau:

1. Số ranker chứa chunk giảm dần.
2. Best dense raw score giảm dần nếu có.
3. Best native rank nhỏ hơn.
4. Stable `chunk_id` tăng dần.

Không dùng list/dict insertion order làm tie-break cuối.

---

## 7. Cross-encoder reranking

### 7.1 Backend strategy

Hỗ trợ qua adapter, không hard-code vào `rerank_cross_encoder()`:

```python
class RerankerBackend(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...
```

Adapters có thể gồm:

- Jina multilingual reranker API.
- Local multilingual cross-encoder.
- Hosted/OpenAI-compatible rerank provider nếu có.

Baseline lab không phụ thuộc backend này. RRF phải chạy được khi không có key/model.

### 7.2 Candidate pool

Không rerank toàn corpus. Đề xuất:

```python
FUSION_CANDIDATE_K = 15
FINAL_TOP_K = 5
```

Luồng:

```text
Weighted RRF top 15 → cross-encoder score 15 pairs → top 5
```

### 7.3 Score handling

- Không giả định mọi reranker trả score `[0,1]`.
- Lưu score gốc vào `raw_scores.reranker`.
- Nếu cần hiển thị confidence-like score, normalize riêng và ghi `score_kind`.
- Không dùng reranker score chưa calibrate làm PageIndex fallback threshold.

### 7.4 Reliability

- Timeout cấu hình được, mặc định khoảng 15–30 giây.
- Retry tối đa 1 lần cho lỗi tạm thời/429/5xx với backoff.
- Không retry lỗi auth/schema.
- API lỗi: unified pipeline giữ RRF order và ghi diagnostics `reranker_fallback_reason`.
- Cache query + ordered candidate hashes trong process để demo/A-B không gọi lại không cần thiết.
- Không log API key hoặc full request headers.

---

## 8. MMR diversity

### 8.1 Công thức

```text
MMR(d) = λ × relevance(query, d)
         - (1 - λ) × max similarity(d, selected)
```

Default:

```python
MMR_LAMBDA = 0.7
```

### 8.2 Khi nào dùng

MMR phù hợp khi top candidates bị trùng:

- Nhiều overlapping chunks cùng document.
- Một section xuất hiện trong cả legal và news snapshot.
- Context top 5 lặp cùng một fact.

Không bật MMR mặc định trước khi đo context recall. Diversity cao quá có thể đẩy mất evidence quan trọng.

### 8.3 Embedding source

Ưu tiên:

1. Candidate đã có embedding từ Task 4/Chroma.
2. Batch embed candidate contents bằng cached model Task 4.

Không load model mới trong Task 7. Query/candidate embedding dimension phải giống nhau.

---

## 9. Config profiles

### `fusion_only` — default/demo-safe

```text
Dense + BM25 + TF-IDF → Weighted RRF → top_k
```

- Không API.
- Latency thấp.
- Deterministic.
- Dùng làm baseline bắt buộc.

### `quality_rerank`

```text
Weighted RRF top 15 → multilingual cross-encoder → top_k
```

- Precision tốt hơn cho câu paraphrase/complex intent.
- Cần model/API.
- Có fallback về `fusion_only`.

### `quality_diverse`

```text
Weighted RRF top 20 → cross-encoder top 10 → MMR top_k
```

- Dùng cho câu cần nhiều evidence.
- Chỉ bật nếu evaluation cho thấy context redundancy giảm mà recall không giảm đáng kể.

---

## 10. Error handling

| Tình huống | Hành vi |
|---|---|
| Query rỗng | RRF vẫn có thể fusion; cross-encoder/MMR unified path trả prior order |
| Candidates rỗng | Trả `[]` |
| `top_k <= 0` | Trả `[]` |
| Một ranked list rỗng | Bỏ list đó, giữ weights alignment |
| Tất cả list rỗng | Trả `[]` |
| Weight count mismatch | `ValueError` |
| `k <= 0` | `ValueError` |
| Candidate thiếu `chunk_id` | Dùng fallback hash |
| Candidate thiếu metadata/raw_scores | Tạo copy/default |
| Cross-encoder missing key/model | Fallback prior RRF order, diagnostics rõ |
| Cross-encoder trả sai số lượng score | Backend error, không ghép sai index |
| MMR thiếu embedding | Fallback hoặc `ValueError` theo strict config |
| NaN/Inf score | Bỏ score lỗi và ghi warning |

Không dùng `NotImplementedError` cho trạng thái backend không khả dụng.

---

## 11. Logging và diagnostics

Mỗi call ghi tối thiểu:

- Method/profile.
- Số ranked lists/candidates.
- Weights và `RRF_K`.
- Số unique chunks sau dedupe.
- Top ranks/raw scores trong debug mode.
- Reranker backend/cache hit/fallback reason.
- MMR lambda và redundancy trước/sau nếu bật.
- Latency fusion/reranker/MMR/total.

Output diagnostics dùng cho UI/evaluation, không bắt buộc nhét toàn bộ vào mỗi result. Có thể trả qua logger/request context hoặc service-level response trong Task 9.

---

## 12. Test specification

### 12.1 Starter tests

1. `rerank()` trả list.
2. Không vượt `top_k`.
3. Result có `score`.

Default `method="rrf"` phải xử lý single candidates list, không raise.

### 12.2 RRF unit tests

1. Empty lists trả `[]`.
2. Single list giữ thứ hạng ban đầu.
3. Công thức RRF đúng với rank bắt đầu từ 1.
4. Weighted RRF áp đúng weights.
5. Chunk xuất hiện ở hai lists được cộng score.
6. Dedupe bằng `chunk_id` dù content whitespace khác.
7. Hai source khác nhau không bị merge chỉ vì boilerplate giống nhau nếu chunk IDs khác.
8. Raw dense/BM25/TF-IDF scores được giữ.
9. `confidence_score` không bị ghi đè bởi RRF.
10. Tie-break deterministic.
11. Input objects không bị mutate.
12. NaN/Inf không làm sort crash.
13. Weight mismatch và invalid `k` raise đúng lỗi.

### 12.3 Cross-encoder tests

Dùng mock backend:

1. Re-score và reorder đúng.
2. Giữ metadata/provenance.
3. Không vượt candidate pool/top_k.
4. Backend score count mismatch bị phát hiện.
5. Timeout/auth/rate-limit được phân loại.
6. Unified `rerank()` fallback prior order khi non-strict.
7. Cache key phụ thuộc query và ordered chunk IDs/content hashes.

### 12.4 MMR tests

1. `lambda=1` ưu tiên relevance.
2. Lambda thấp tăng diversity.
3. Không chọn duplicate.
4. Dimension mismatch bị phát hiện.
5. Thiếu embedding không silently tính sai.
6. Output giữ original relevance scores.

### 12.5 Integration tests

Trên corpus thật:

- Dense/BM25/TF-IDF cùng chunk được fusion thành một result.
- Exact keyword result từ BM25 không mất khi dense rank thấp.
- Paraphrase result từ dense không mất khi lexical không match.
- Cross-encoder optional thay đổi thứ hạng ít nhất một query thực tế.
- MMR giảm duplicate chunks trong ít nhất một multi-evidence query.
- Task 9 fallback vẫn đọc raw dense confidence, không đọc RRF/reranker score.

---

## 13. Evaluation và tối ưu

So sánh ít nhất:

| Config | Mô tả |
|---|---|
| R1 | Dense-only |
| R2 | Dense + BM25 Weighted RRF |
| R3 | Dense + BM25 + TF-IDF Weighted RRF |
| R4 | R3 + cross-encoder |
| R5 | R4 + MMR, nếu cần diversity |

Metrics:

- Recall@1/3/5.
- MRR@5.
- nDCG@5.
- Context precision/recall.
- Duplicate ratio trong top-k.
- Latency p50/p95.
- Reranker API/model cost nếu có.

Không chọn config chỉ vì một query demo đẹp. Config final dựa trên golden dataset và latency budget.

Weighted RRF tuning:

- Dùng grid nhỏ, ví dụ dense `0.8–1.2`, BM25 `0.7–1.1`, TF-IDF `0.4–0.9`.
- Tune trên calibration split.
- Báo cáo trên held-out evaluation split để tránh overfit.

---

## 14. Kịch bản demo Task 7

### Demo fusion

1. Hiển thị top results riêng của dense, BM25, TF-IDF.
2. Chạy Weighted RRF.
3. Mở diagnostics cho thấy ranks và raw scores.
4. Giải thích vì sao không cộng cosine + BM25 trực tiếp.

### Demo reranker

1. Chạy `fusion_only`.
2. Bật cross-encoder và chạy lại cùng query.
3. Chỉ ra section đúng intent được đưa lên cao.
4. Hiển thị latency/cost trade-off.

### Demo diversity

1. Chọn query cần nhiều evidence.
2. Cho thấy top 5 trước MMR có các chunk overlap.
3. Bật MMR và chỉ ra source/section đa dạng hơn.

Điểm phải nói rõ:

- RRF score chỉ dùng xếp hạng.
- Confidence/fallback lấy từ raw dense evidence gate.
- Cross-encoder là optional quality layer.
- MMR giảm redundancy nhưng có thể giảm recall nếu lambda không hợp lý.

---

## 15. Các hướng phát triển thêm

1. **Learned fusion:** học weights từ golden dataset thay vì đặt tay, nhưng cần held-out split.
2. **Query-type routing:** exact code/date ưu tiên lexical; conceptual query ưu tiên dense.
3. **Dynamic candidate budget:** query mơ hồ lấy pool lớn hơn, query exact lấy pool nhỏ.
4. **Reranker calibration:** map raw reranker score sang probability bằng labeled set.
5. **Parent-child selection:** rerank chunks nhưng trả parent section để generation đủ context.
6. **Source diversity constraints:** giới hạn số chunk cùng document trong top-k.
7. **Late interaction retriever:** bổ sung ColBERT-style ranker khi có tài nguyên.
8. **Explainability panel:** hiển thị từng ranker đóng góp bao nhiêu vào final rank.

Các hướng này chỉ làm sau khi baseline RRF, citation và evaluation chạy ổn định.

---

## 16. Checklist triển khai

- [ ] Candidate schema thống nhất với Task 4–6.
- [ ] `rerank_rrf()` đúng công thức, weights optional.
- [ ] Dedupe bằng `chunk_id`, fallback hash an toàn.
- [ ] Merge raw scores/provenance đầy đủ.
- [ ] `score_type` và `confidence_score` tách biệt.
- [ ] Tie-break deterministic.
- [ ] Không mutate input.
- [ ] `rerank()` default xử lý single list và pass starter tests.
- [ ] Task 9 chỉ fusion một lần.
- [ ] Cross-encoder optional có timeout/retry/fallback/cache.
- [ ] MMR optional dùng đúng embedding dimension.
- [ ] Unit/integration tests pass.
- [ ] Có A/B và latency report.
- [ ] Demo hiển thị contributions/raw scores.

---

## 17. Tiêu chí chấp nhận cuối

Task 7 được nghiệm thu khi:

1. `TestTask7` có `3 passed, 0 skipped`.
2. RRF/Weighted RRF fusion đúng dense, BM25 và TF-IDF.
3. Output trace được về source, native ranks và raw scores.
4. RRF/reranker score không bị dùng làm fallback confidence.
5. Optional backend lỗi không làm baseline pipeline crash.
6. Không có double-RRF trong Task 9 production path.
7. A/B cho thấy trade-off quality/latency bằng số liệu thật.
8. Demo giải thích rõ fusion, reranker và diversity.

