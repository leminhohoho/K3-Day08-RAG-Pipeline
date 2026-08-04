# Task 9 Spec — Retrieval Pipeline (Hybrid + Fallback)

## 1. Overview

Task 9 is the **orchestration layer** of the RAG pipeline. It combines semantic search (Task 5), lexical search (Task 6), RRF reranking (Task 7), and PageIndex vectorless fallback (Task 8) into a single unified `retrieve()` function.

**Role in the pipeline:**
```
Query
  ├→ semantic_search (Task 5)  ─┐
  ├→ lexical_search  (Task 6)  ─┤→ rerank_rrf (Task 7) → hybrid results
  │                              └→ if best cosine < threshold → pageindex_search (Task 8)
  └→ Task 10 (generation) consumes the output
```

**How to run:** `python src/task9_retrieval_pipeline.py`

**Depends on:** Task 5 (`semantic_search`), Task 6 (`lexical_search`), Task 7 (`rerank_rrf`, `rerank`), Task 8 (`pageindex_search`)

---

## 2. Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                         retrieve(query, top_k)                       │
│                                                                      │
│   Step 1 — Parallel retrieval                                       │
│     dense_results  = semantic_search(query, top_k=top_k*2)          │
│     sparse_results = lexical_search(query, top_k=top_k*2)           │
│                                                                      │
│   Step 2 — Merge via RRF                                            │
│     merged = rerank_rrf([dense_results, sparse_results],             │
│                          top_k=top_k*2)                              │
│     for item in merged:                                              │
│         item["source"] = "hybrid"                                    │
│                                                                      │
│   Step 3 — Rerank (optional, method != "none")                       │
│     if use_reranking and RERANK_METHOD != "none":                     │
│         final = rerank(query, merged, top_k=top_k,                   │
│                        method=RERANK_METHOD)                         │
│     else:                                                            │
│         final = merged[:top_k]                                       │
│                                                                      │
│   Step 4 — Fallback check                                            │
│     best_cosine = dense_results[0]["score"] if dense_results else 0  │
│     if best_cosine < score_threshold:                                │
│         fallback = pageindex_search(query, top_k=top_k)              │
│         if fallback: return fallback                                 │
│                                                                      │
│   Return final[:top_k]                                               │
└──────────────────────────────────────────────────────────────────────┘
```

### ⚠️ Critical: Threshold Must Use Raw Cosine, NOT RRF Score

RRF scores are rank-based only — the top RRF result is always ≈ `1/(k+1)` ≈ 0.016 regardless of actual relevance. Using RRF score for the fallback threshold would make the check **meaningless** (no query would ever trigger fallback).

**Always use `dense_results[0]["score"]` (raw cosine similarity from Task 5) for the threshold comparison.**

---

## 3. Interface & API Contract

### 3.1 Configuration Constants

| Constant | Type | Default | Description |
|---|---|---|---|
| `SCORE_THRESHOLD` | `float` | `0.3` | Minimum raw cosine similarity to skip fallback. Must be calibrated per corpus. |
| `DEFAULT_TOP_K` | `int` | `5` | Default number of results to return |
| `RERANK_METHOD` | `str` | `"none"` | Post-fusion reranking: `"none"` | `"cross_encoder"` | `"mmr"` |

**Note:** The default is `"none"` because RRF fusion already produces a ranked list. Use `"cross_encoder"` or `"mmr"` for additional re-scoring after fusion. Do NOT set `"rrf"` — that would apply RRF twice (double-RRF bug).

**Calibration note for `SCORE_THRESHOLD`:**
Run several in-domain queries (known to have answers) and several out-of-domain/garbage queries through `semantic_search()`. Plot the best cosine scores for each group. Choose a threshold that sits **between** the two clusters. Do not blindly copy the default `0.3`.

### 3.2 Function: `retrieve()`

```python
def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Full retrieval pipeline with hybrid search + fallback.

    Args:
        query: User query string (Vietnamese, English, or mixed)
        top_k: Number of final results to return
        score_threshold: Raw cosine similarity threshold for fallback.
                         Uses dense_results[0]["score"], NOT RRF score.
        use_reranking: Whether to apply reranking after RRF merge

    Returns:
        List of dicts, each with:
            "content": str     — chunk text
            "score": float     — final score (RRF score after merge, rank proxy for PageIndex)
            "score_type": str  — "rrf" | "reranker" | "mmr" | "rank_proxy"
            "confidence_score": float | None  — raw cosine similarity from dense search (for fallback calibration)
            "metadata": dict   — chunk metadata (source, document_id, type, chunk_id, etc.)
            "source": str      — "hybrid" or "pageindex"
            "raw_scores": dict — per-retriever scores: {"dense": float|None, "bm25": float|None, "rrf": float|None, ...}
    """
```

**Behavior:**
- Step 1: Run `semantic_search()` and `lexical_search()` in sequence (no threading needed), each with `top_k=top_k*2` to have a larger candidate pool for reranking
- Step 2: Merge via `rerank_rrf([dense_results, sparse_results], top_k=top_k*2)`. Set `source="hybrid"` on each result
- Step 3: If `use_reranking=True`, call `rerank(query, merged, top_k=top_k, method=RERANK_METHOD)`. If `rerank()` raises `NotImplementedError` for the selected method, fall back to `merged[:top_k]`
- Step 4: Extract `best_cosine = dense_results[0]["score"]` if dense_results non-empty, else `0.0`. If `best_cosine < score_threshold`, call `pageindex_search(query, top_k=top_k)`. If PageIndex returns results, **return them directly** (replacing hybrid results). Note: PageIndex results have `score_type="rank_proxy"` — they are NOT cosine scores.
- Return `final[:top_k]`

### 3.3 Output contract

```python
# Hybrid result example
{
    "content": "Tuition fees for international students...",
    "score": 0.0321,           # RRF score (rank-based, not cosine)
    "score_type": "rrf",
    "confidence_score": 0.82,  # raw cosine similarity from dense search
    "metadata": {
        "source": "tuition-policy.md",
        "document_id": "tuition-policy",
        "type": "legal",
        "chunk_index": 2,
        "chunk_id": "tuition-policy_chunk_2"
    },
    "source": "hybrid",
    "raw_scores": {
        "dense": 0.82,
        "bm25": 6.78,
        "rrf": 0.0321
    }
}

# PageIndex fallback result example
{
    "content": "Tuition fees are structured...",
    "score": 0.5,               # rank proxy score (1.0 / global_rank)
    "score_type": "rank_proxy",
    "confidence_score": None,   # PageIndex does not provide cosine confidence
    "metadata": {
        "section": "Payment Structure",
        "source": "tuition-fees-rmit.pdf",
        "document_id": "rmit-tuition-fees-2026",
        "chunk_id": "pageindex:pi-abc123:0005:<hash>"
    },
    "source": "pageindex",
    "raw_scores": {
        "pageindex_rank_proxy": 0.5
    }
}
```

**Test constraints (minimum required by tests):**
- Returns `list[dict]` — never `None` or `NotImplementedError`
- Every result has `"content"`, `"score"`, `"source"`
- `"source"` is always `"hybrid"` or `"pageindex"`
- `len(results) <= top_k`
- Does not crash on obscure queries (fallback may return empty list)

**Additional fields (per Task 7/8 schema):** `score_type`, `confidence_score`, `raw_scores` — these are optional but recommended for provenance and diagnostics. Tasks 7 and 8 define these fields; Task 9 passes them through from downstream tasks.

---

## 4. Implementation Details

### 4.1 Step 1: Candidate Pool Size

Use `top_k * 2` as the candidate pool for both semantic and lexical search. This gives `rerank_rrf` enough items to produce a meaningful reranked top-k. For example, if `top_k=5`, request `top_k=10` from each search method.

### 4.2 Step 2: RRF Merge

Call `rerank_rrf()` from Task 7:

```python
merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
```

After merge, tag each result:
```python
for item in merged:
    item["source"] = "hybrid"
```

**Deduplication note:** `rerank_rrf()` deduplicates by content key. If the same chunk appears in both dense and sparse results, it gets a higher RRF score (appears in both lists). This is the desired behavior.

### 4.3 Step 3: Optional Reranking (Post-Fusion)

If `use_reranking=True` and `RERANK_METHOD != "none"`, call the unified `rerank()` interface from Task 7:

```python
final = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
```

- `"cross_encoder"` — re-scores candidates using a cross-encoder model (requires API key or local model)
- `"mmr"` — applies Maximal Marginal Relevance for diversity (requires embeddings)

**Important:** Do NOT set `RERANK_METHOD="rrf"` — RRF fusion already happened in Step 2. Applying RRF again would be a **double-RRF bug** (Task 7 spec explicitly warns against this). The default for post-fusion reranking is `"none"`.

If `rerank()` raises `NotImplementedError` (e.g., cross-encoder not available), fall back to `merged[:top_k]`.

### 4.4 Step 4: Fallback Logic

```python
best_cosine = dense_results[0]["score"] if dense_results else 0.0

if best_cosine < score_threshold:
    fallback = pageindex_search(query, top_k=top_k)
    if fallback:
        return fallback
    # If PageIndex also returns nothing, return empty list
    return []
```

**Why use raw cosine, not RRF score:**
- Cosine similarity: `[0, 1]` — meaningful absolute measure of relevance
- RRF score: `~0.0164` for top result regardless of relevance — meaningless for thresholds

### 4.5 Edge Cases

| Scenario | Expected Behavior |
|---|---|
| `semantic_search()` returns empty | `dense_results = []`, `best_cosine = 0.0`, falls through to PageIndex |
| `lexical_search()` returns empty | `sparse_results = []`, merge still works with 1 list |
| Both search methods return empty | `merged = []`, `best_cosine = 0.0`, tries PageIndex |
| PageIndex returns empty | Returns `[]` |
| `use_reranking=False` | Skip reranking, return `merged[:top_k]` directly |
| `use_reranking=True, RERANK_METHOD="none"` | Same as `use_reranking=False` (no post-fusion reranking) |
| `use_reranking=True, RERANK_METHOD="cross_encoder"` | Call `rerank()` with cross-encoder; fallback to `merged[:top_k]` if unavailable |
| `score_threshold=0.0` | Never triggers fallback (always use hybrid) |
| `score_threshold=1.0` | Always triggers fallback (always use PageIndex) |

---

## 5. Dependencies on Other Tasks

### 5.1 Task 5 — `semantic_search(query, top_k)`

```python
from src.task5_semantic_search import semantic_search
```

- Returns `list[dict]` with `content`, `score` (cosine similarity in `[0, 1]`), `metadata`
- Score is **raw cosine similarity** — used for fallback threshold
- Candidate pool: `top_k=top_k*2`

### 5.2 Task 6 — `lexical_search(query, top_k)`

```python
from src.task6_lexical_search import lexical_search
```

- Returns `list[dict]` with `content`, `score` (BM25 score, unbounded), `metadata`
- Score is **not comparable** to cosine — used only for RRF ranking
- Candidate pool: `top_k=top_k*2`

### 5.3 Task 7 — `rerank_rrf()`, `rerank()`

```python
from src.task7_reranking import rerank, rerank_rrf
```

- `rerank_rrf(ranked_lists, top_k, k=60, weights=None)` — merges multiple ranked lists by RRF (optional `weights` for weighted RRF, per Task 7 spec)
- `rerank(query, candidates, top_k, method)` — unified reranking interface for post-fusion re-scoring
- `rerank_rrf()` is always used for the initial merge
- `rerank()` is optional (when `use_reranking=True`)

### 5.4 Task 8 — `pageindex_search(query, top_k)`

```python
from src.task8_pageindex_vectorless import pageindex_search
```

- Returns `list[dict]` with `content`, `score`, `score_type="rank_proxy"`, `confidence_score=None`, `metadata`, `source="pageindex"`, `raw_scores`
- Used as **fallback** when hybrid search confidence is low (best cosine < `score_threshold`)
- Returns `[]` if API key is missing, no ready documents, or API error (never crashes the pipeline)
- Vectorless — no embedding or chunking needed

---

## 6. Verification

### 6.1 Run the pipeline

```bash
python src/task9_retrieval_pipeline.py
```

Expected output for each test query:
```
Query: What is the tuition fee at RMIT Vietnam?
------------------------------------------------------------
  1. [0.032] [hybrid] Tuition fees for international students at RMIT Vietnam...
  2. [0.028] [hybrid] Payment schedule for tuition fees...
  ...

Query: xyzabc123nonsense
------------------------------------------------------------
  ⚠ Semantic best score (0.000) < threshold (0.300)
  (PageIndex fallback results or empty list)
```

### 6.2 Run the tests

```bash
pytest tests/test_individual.py::TestTask9 -v
```

Expected: **all 4 tests pass** (7 points total)

| Test | What it checks |
|---|---|
| `test_retrieve_returns_list` | `retrieve()` returns a `list`, not `None` or exception |
| `test_results_have_required_keys` | Each result has `content`, `score`, `source`; source is `"hybrid"` or `"pageindex"` |
| `test_respects_top_k` | `len(results) <= top_k` |
| `test_fallback_logic_exists` | Pipeline doesn't crash when `score_threshold=0.99` (obscure query triggers fallback or returns empty list) |

### 6.3 Manual calibration check

```bash
python -c "
from src.task5_semantic_search import semantic_search

# Test in-domain queries
for q in ['tuition fee', 'scholarship', 'library hours']:
    r = semantic_search(q, top_k=1)
    print(f'In-domain [{q}]: best cosine = {r[0][\"score\"]:.3f}' if r else f'In-domain [{q}]: no results')

# Test out-of-domain queries
for q in ['xyzabc123', 'asdfghjkl', 'nonsense query']:
    r = semantic_search(q, top_k=1)
    print(f'OOD [{q}]: best cosine = {r[0][\"score\"]:.3f}' if r else f'OOD [{q}]: no results')
"
```

This helps calibrate `SCORE_THRESHOLD` between the two clusters.

---

## 7. Edge Cases & Notes

| Scenario | Expected Behavior |
|---|---|
| `top_k <= 0` | Return `[]` (validated by search functions) |
| `query` is empty string | `semantic_search("")` returns `[]` → fallback triggered |
| RRF merge produces fewer than `top_k` results | Return whatever RRF produced (not padded) |
| PageIndex API key missing | `pageindex_search()` raises or returns `[]` — handle gracefully |
| `rerank()` raises `NotImplementedError` | Skip reranking, use `merged[:top_k]` |
| All 3 search + fallback return empty | Return `[]` |

### Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src.task5_semantic_search'` | Circular import or missing `__init__.py` | Ensure `src/__init__.py` exists; use relative imports |
| `retrieve()` always returns hybrid (never falls back) | `SCORE_THRESHOLD` too low | Check raw cosine scores; raise threshold |
| `retrieve()` always falls back | `SCORE_THRESHOLD` too high | Lower threshold or check embedding model |
| RRF scores all look the same | RRF with `k=60` produces similar scores for similar ranks | Normal — RRF is rank-based, not content-based |
| PageIndex takes too long | API call latency | Increase timeout; consider reducing PageIndex calls |

### Calibration Procedure

1. Run 5 in-domain queries through `semantic_search()`, record best cosine scores
2. Run 5 out-of-domain queries, record best cosine scores
3. Choose `SCORE_THRESHOLD` between the two clusters
4. Re-run tests to verify fallback behavior is correct