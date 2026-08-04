# Task 10 Spec — Generation with Citation

## 1. Overview

Task 10 is the **final generation layer** of the RAG pipeline. It takes retrieved chunks from Task 9, reorders them to mitigate the "lost in the middle" effect, formats them into a prompt with source labels, and calls an LLM (via OpenRouter) to produce a cited answer.

**Role in the pipeline:**
```
Query
  → retrieve() (Task 9) → chunks
  → reorder_for_llm() → reordered chunks
  → format_context() → context string
  → LLM (OpenRouter) → answer with citations
```

**How to run:** `python src/task10_generation.py`

**Depends on:** Task 9 (`retrieve`), OpenRouter API key (`OPENROUTER_API_KEY` in `.env`)

---

## 2. Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     generate_with_citation(query)                        │
│                                                                          │
│   Step 1 — Retrieve                                                     │
│     chunks = retrieve(query, top_k=TOP_K)                               │
│     if not chunks: return no-answer response                            │
│                                                                          │
│   Step 2 — Reorder (lost-in-the-middle mitigation)                      │
│     reordered = reorder_for_llm(chunks)                                 │
│     Strategy: best → worst middle → second-best last                    │
│     Input:  [1, 2, 3, 4, 5]  (by score descending)                     │
│     Output: [1, 3, 5, 4, 2]  (front + back[::-1])                      │
│                                                                          │
│   Step 3 — Format context                                               │
│     context = format_context(reordered)                                 │
│     Each chunk gets a [Document N | Source: filename] label             │
│                                                                          │
│   Step 4 — Build prompt                                                 │
│     system = SYSTEM_PROMPT (citations required, no hallucination)       │
│     user   = "Context:\n{context}\n\n---\n\nQuestion: {query}"          │
│                                                                          │
│   Step 5 — Call LLM                                                     │
│     client.chat.completions.create(model=LLM_MODEL, ...)                │
│                                                                          │
│   Step 6 — Return                                                       │
│     { "answer": str, "sources": list, "retrieval_source": str }         │
└──────────────────────────────────────────────────────────────────────────┘
```

### Lost in the Middle

LLMs tend to remember information at the **beginning** and **end** of a long prompt better than information in the middle. The reordering strategy exploits this:

- **Position 1:** Most relevant chunk (highest score)
- **Positions 2..n-1:** Decreasing relevance (middle)
- **Position n:** Second most relevant chunk (lowest score in the ordering, but placed at the end)

This maximizes the chance that the two most important chunks are in high-attention zones.

---

## 3. Interface & API Contract

### 3.1 Configuration Constants

| Constant | Type | Default | Description |
|---|---|---|---|
| `TOP_K` | `int` | `5` | Number of chunks to retrieve and include in context |
| `TOP_P` | `float` | `0.9` | Nucleus sampling parameter for LLM generation |
| `TEMPERATURE` | `float` | `0.3` | Low temperature for factual RAG responses |
| `LLM_MODEL` | `str` | `"openai/gpt-4o-mini"` | OpenRouter model ID (supports `:free` suffix) |

**Rationale for defaults:**
- `TOP_K=5` — Enough evidence for a comprehensive answer without exceeding context window limits
- `TOP_P=0.9` — Balances diversity and determinism for factual responses
- `TEMPERATURE=0.3` — Low temperature reduces hallucination risk; RAG needs factual accuracy, not creativity

### 3.2 Function: `reorder_for_llm(chunks)`

```python
def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to mitigate "lost in the middle" effect.

    Places the most relevant chunk first, the second-most relevant last,
    and the least relevant in the middle.

    Args:
        chunks: List sorted by score descending (from retrieval pipeline)

    Returns:
        Reordered list of the same length.
        Pattern: [1, 3, 5, ..., 4, 2] (front + back[::-1])
    """
```

**Algorithm:**
```
if len(chunks) <= 2:
    return chunks

front = chunks[::2]   # indices 0, 2, 4, ...  (best, 3rd best, 5th best, ...)
back  = chunks[1::2]  # indices 1, 3, ...      (2nd best, 4th best, ...)

result = front + back[::-1]  # best chunks at front, second-best at end
```

**Example with 5 chunks:**
```
Input (by score descending):  [A, B, C, D, E]
front = [A, C, E]
back  = [B, D]
back[::-1] = [D, B]
Output: [A, C, E, D, B]
           ↑  ↑  ↑  ↑  ↑
        best 3rd 5th 4th 2nd
```

**Constraints (enforced by tests):**
- Returns same length as input
- First element stays the same (most relevant chunk remains first)
- No duplicate or missing chunks

### 3.3 Function: `format_context(chunks)`

```python
def format_context(chunks: list[dict]) -> str:
    """
    Format chunks into a context string with source labels for citation.

    Args:
        chunks: List of chunk dicts (after reordering)

    Returns:
        Formatted string with labeled chunks separated by dividers.
    """
```

**Output format:**
```
[Document 1 | Source: tuition-policy.md | Type: legal]
Tuition fees for international students at Trường Đại học Khoa học Xã hội và Nhân văn (USSH)...

---

[Document 2 | Source: scholarship-announcement.md | Type: news]
Scholarship opportunities for international students...

...
```

**Behavior:**
- Number chunks sequentially 1..N
- Extract `source` from `metadata["source"]` (fallback to `"Source {i}"`)
- Extract `type` from `metadata["type"]` (fallback to `"unknown"`)
- Separate chunks with `\n---\n`

**Constraints (enforced by tests):**
- The output string must contain the source filename (e.g., `"tuition-policy.md"`) for citation support

### 3.4 Function: `generate_with_citation(query)`

```python
def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation with citation.

    Pipeline: retrieve → reorder → format → LLM → parse

    Args:
        query: User question (Vietnamese, English, or mixed)
        top_k: Number of chunks to retrieve

    Returns:
        dict with:
            "answer": str           — LLM response with inline citations
            "sources": list[dict]   — the chunks used (from retrieval)
            "retrieval_source": str — "hybrid" or "pageindex" or "none"
    """
```

**Behavior:**
- Step 1: Call `retrieve(query, top_k=top_k)`
- Step 2: If no chunks returned, return `{"answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có", "sources": [], "retrieval_source": "none"}`
- Step 3: Reorder via `reorder_for_llm(chunks)`
- Step 4: Format via `format_context(reordered)`
- Step 5: Build prompt with `SYSTEM_PROMPT` + context + query
- Step 6: Call OpenRouter API (OpenAI-compatible interface)
- Step 7: Return dict with answer, sources, and retrieval source tag

### 3.5 System Prompt (`SYSTEM_PROMPT`)

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

---

## 4. Implementation Details

### 4.1 LLM Call via OpenRouter

OpenRouter provides an OpenAI-compatible API. Use the OpenAI SDK with a custom base URL:

```python
from openai import OpenAI
import os

api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

response = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ],
    temperature=TEMPERATURE,
    top_p=TOP_P,
)
```

**Model selection:**
- `"openai/gpt-4o-mini"` — Good quality, low cost
- `"google/gemini-2.0-flash-exp:free"` — Free tier on OpenRouter
- Browse more free models at: https://openrouter.ai/models?max_price=0

### 4.2 Prompt Construction

```python
user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""
```

The context is placed before the question so the LLM reads the evidence first, reducing hallucination.

### 4.3 Error Handling

| Scenario | Behavior |
|---|---|
| `retrieve()` returns empty list | Return no-answer response immediately |
| API key missing | Let exception propagate (test skips gracefully) |
| LLM API timeout/error | Let exception propagate (visible to developer) |
| LLM returns empty response | Return `{"answer": "(empty response)", ...}` |

### 4.4 Citation Format

The system prompt instructs the LLM to cite sources inline using `[Source, Section]` format. The `format_context()` function provides labeled chunks so the LLM can reference them by document number or source filename.

Example response:
```
Học phí chương trình Business tại Trường Đại học Khoa học Xã hội và Nhân văn (USSH) là 350 triệu đồng/năm [Tuition Fees, 2026].
Sinh viên có thể thanh toán theo kỳ hoặc theo năm [Payment Schedule].
```

---

## 5. Dependencies on Other Tasks

### 5.1 Task 9 — `retrieve(query, top_k)`

```python
from src.task9_retrieval_pipeline import retrieve
```

- Returns `list[dict]` with `content`, `score`, `metadata`, `source`
- Source is `"hybrid"` or `"pageindex"` — used to tag the `retrieval_source` in the output

### 5.2 Environment

```python
# .env file
OPENROUTER_API_KEY=sk-or-v1-...
# or
OPENAI_API_KEY=sk-...
```

---

## 6. Verification

### 6.1 Run the pipeline

```bash
python src/task10_generation.py
```

Expected output:
```
======================================================================
Q: Học phí tại Trường Đại học Khoa học Xã hội và Nhân văn (USSH) là bao nhiêu?
======================================================================

A: Học phí chương trình Business tại Trường Đại học Khoa học Xã hội và Nhân văn (USSH) là khoảng 350 triệu
đồng mỗi năm [Tuition Policy]. Sinh viên có thể thanh toán theo kỳ
hoặc theo năm [Payment Schedule]...

[Sources: 5 chunks | via hybrid]
```

### 6.2 Run the tests

```bash
pytest tests/test_individual.py::TestTask10 -v
```

Expected: **all 4 tests pass** (4 points total)

| Test | What it checks |
|---|---|
| `test_reorder_function_exists` | `reorder_for_llm()` preserves length; first element unchanged |
| `test_format_context_includes_source` | `format_context()` output contains the source filename |
| `test_generate_returns_dict_with_answer` | `generate_with_citation()` returns dict with `"answer"` key (non-empty string) |
| *(implicit)* | Pipeline runs end-to-end without crashing |

### 6.3 Manual checks

```bash
# Test reordering
python -c "
from src.task10_generation import reorder_for_llm
chunks = [{'content': f'Chunk {i}', 'score': 1.0 - i*0.1} for i in range(5)]
reordered = reorder_for_llm(chunks)
for i, c in enumerate(reordered):
    print(f'  Position {i}: {c[\"content\"]}')
"
```

Expected:
```
  Position 0: Chunk 0
  Position 1: Chunk 2
  Position 2: Chunk 4
  Position 3: Chunk 3
  Position 4: Chunk 1
```

---

## 6. Edge Cases & Notes

| Scenario | Expected Behavior |
|---|---|
| `retrieve()` returns empty list | Return no-answer response immediately |
| Only 1 chunk retrieved | `reorder_for_llm()` returns as-is (no reordering needed) |
| 2 chunks retrieved | Return as-is (no reordering needed) |
| LLM returns empty message | Return `{"answer": "(empty response)", ...}` |
| API key not set | Exception raised; test skips gracefully |
| Chunk metadata missing `source` | Fall back to `"Source {i}"` in `format_context()` |
| Chunk metadata missing `type` | Fall back to `"unknown"` |

### Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `openai.AuthenticationError` | API key missing or invalid | Check `OPENROUTER_API_KEY` in `.env` |
| `openai.RateLimitError` | Free tier rate limit hit | Wait or use a different model |
| `openai.NotFoundError` | Model ID doesn't exist | Check model name at openrouter.ai/models |
| No citations in answer | LLM ignores prompt instruction | Strengthen emphasis in SYSTEM_PROMPT |
| Answer is not in Vietnamese | Prompt instructs Vietnamese but LLM may default to English | Add explicit language instruction |
| Context too long for model | TOP_K too large for model's context window | Reduce `TOP_K` or use a model with larger context |

### Key Design Decisions

| Decision | Rationale |
|---|---|
| `reorder_for_llm` front + back[::-1] | Mitigates "lost in the middle" — best chunk first, second-best last |
| `temperature=0.3` | Low randomness for factual RAG |
| `top_p=0.9` | Balances diversity and determinism |
| `TOP_K=5` | Enough evidence without overwhelming context window |
| System prompt in Vietnamese | Target users are Vietnamese-speaking |
| OpenRouter API | Free tier available; OpenAI-compatible SDK |