# RAG Evaluation Results

## Run

- Timestamp: `2026-08-04T15:52:23+07:00`
- Corpus hash: `160f372ed363c1d362329638085baabd1623b6fd6d7f7b4f02c1f5790680ef3f`
- Golden dataset hash: `27393c4d97569ab72ff095fd76eed4907831303afc10af7b4976b79c79518a90`
- Cases: `28`
- Top-k: `5`
- Local raw artifacts (generated, gitignored): `group_project/evaluation/runs/20260804_155222/`

## Configurations

| Config | Description |
|---|---|
| `dense_only` | Task 5 semantic search only |
| `bm25_only` | Task 6 BM25 lexical search only |
| `hybrid_rrf` | Task 5 dense + Task 6 BM25 + weighted RRF |


## Retrieval Evaluation

| Metric | dense_only | bm25_only | hybrid_rrf |
|---|---:|---:|---:|
| Source Hit@1 | 0.929 | 0.857 | 1.000 |
| Source Hit@3 | 1.000 | 1.000 | 1.000 |
| Source Hit@5 | 1.000 | 1.000 | 1.000 |
| Document Recall@5 | 0.982 | 0.982 | 0.982 |
| MRR@5 | 0.964 | 0.923 | 1.000 |
| nDCG@5 | 0.960 | 0.926 | 0.986 |
| Evidence Coverage@5 | 0.885 | 0.873 | 0.878 |
| Duplicate Context Rate@5 | 0.000 | 0.000 | 0.000 |
| Empty Result Rate | 0.000 | 0.000 | 0.000 |
| Latency p50 (ms) | 487.9 | 8.3 | 496.6 |
| Latency p95 (ms) | 1887.0 | 13.2 | 1906.5 |

Best retrieval configuration by Document Recall@5 then MRR@5: **`hybrid_rrf`**.

Metrics are deterministic source/document retrieval metrics. `Evidence Coverage@5` is a lexical proxy and is not presented as RAGAS Context Recall.

### Retrieval by split

| Split | Config | Cases | Hit@1 | Recall@5 | MRR@5 | nDCG@5 | Evidence@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| challenge | `dense_only` | 8 | 0.875 | 0.938 | 0.938 | 0.906 | 0.901 |
| challenge | `bm25_only` | 8 | 0.875 | 0.938 | 0.938 | 0.895 | 0.850 |
| challenge | `hybrid_rrf` | 8 | 1.000 | 0.938 | 1.000 | 0.952 | 0.894 |
| core | `dense_only` | 18 | 0.944 | 1.000 | 0.972 | 0.979 | 0.915 |
| core | `bm25_only` | 18 | 0.944 | 1.000 | 0.963 | 0.972 | 0.919 |
| core | `hybrid_rrf` | 18 | 1.000 | 1.000 | 1.000 | 1.000 | 0.907 |
| safety | `dense_only` | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.550 |
| safety | `bm25_only` | 2 | 0.000 | 1.000 | 0.500 | 0.631 | 0.550 |
| safety | `hybrid_rrf` | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.550 |

## Required RAGAS Metrics

Answerable cases evaluated: `dense_only`: 3, `bm25_only`: 3, `hybrid_rrf`: 3. Selected case IDs: `GD001`, `GD018`, `GD025`. Treat these as smoke/diagnostic scores, not final full-corpus RAGAS scores.

| Metric | dense_only | bm25_only | hybrid_rrf |
|---|---:|---:|---:|
| Faithfulness | 0.722 | 0.333 | 0.833 |
| Answer Relevance | 0.483 | 0.199 | 0.757 |
| Context Recall | 1.000 | 1.000 | 1.000 |
| Context Precision | 1.000 | 0.983 | 1.000 |

## Safety

Safety cases are excluded from answerable RAGAS averages.

| Metric | dense_only | bm25_only | hybrid_rrf |
|---|---:|---:|---:|
| Refusal Accuracy | 1.000 | 1.000 | 1.000 |
| Unsupported Answer Rate | 0.000 | 0.000 | 0.000 |

## Worst Retrieval Cases

| Case | Config | Composite retrieval score | Question |
|---|---|---:|---|
| GD027 | bm25_only | 0.676 | Học phí ngành Tâm lý học khóa QH-2027-X là bao nhiêu? |
| GD028 | bm25_only | 0.690 | Giá thuê chính xác mỗi tháng của một phòng ký túc xá ĐHQGHN là bao nhiêu? |
| GD026 | bm25_only | 0.746 | Một sinh viên QH-2025-X ngành Tâm lý học muốn biết cả mức thu khi học lại một tín chỉ và chỉ tiêu tuyển sinh năm 2026 của ngành này. Hai con số là bao nhiêu? |
| GD026 | dense_only | 0.746 | Một sinh viên QH-2025-X ngành Tâm lý học muốn biết cả mức thu khi học lại một tín chỉ và chỉ tiêu tuyển sinh năm 2026 của ngành này. Hai con số là bao nhiêu? |
| GD026 | hybrid_rrf | 0.746 | Một sinh viên QH-2025-X ngành Tâm lý học muốn biết cả mức thu khi học lại một tín chỉ và chỉ tiêu tuyển sinh năm 2026 của ngành này. Hai con số là bao nhiêu? |

## Initial Recommendations

1. Use `hybrid_rrf` as the current evidence-backed retrieval baseline; do not select a configuration from one demo query.
2. Inspect the worst cases in `predictions.json` before changing chunk size, RRF weights or fallback threshold.
3. Re-run the same frozen golden dataset after Task 4 canonical metadata/index rebuild; compare by paired case ID.
4. Expand RAGAS from the labelled smoke subset to the complete core split, then challenge split, using `--resume` so retrieval stays frozen.
5. Keep safety refusal metrics separate from answerable quality averages.
