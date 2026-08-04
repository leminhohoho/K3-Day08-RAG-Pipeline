# Chat UI Spec — Streamlit Frontend (app.py)

## 1. Overview

The Chat UI is the **frontend layer** of the RAG pipeline. It provides a Streamlit-based chat interface for end-users to ask questions about university services and receive cited answers from the RAG pipeline.

**File:** `app.py`  
**Run:** `streamlit run app.py`  
**Role:** Role 3 (Frontend & Chatbot Developer) — Checkpoint 5  
**Depends on:** Task 10 (`generate_with_citation`), Task 9 (`retrieve`), Streamlit

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        app.py (Streamlit)                           │
│                                                                     │
│  ┌─ Sidebar ─────────────────────────────────────────────────┐     │
│  │  🎓 Title & Caption                                       │     │
│  │  💡 Suggested Questions (5 buttons)                       │     │
│  │  ⚙️ Settings: top_k slider (3–10, default 5)             │     │
│  │  📊 Architecture info text                                │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  ┌─ Main Chat Area ───────────────────────────────────────────┐     │
│  │  Header: "🎓 University Services RAG Chatbot"              │     │
│  │                                                             │     │
│  │  Chat History (scrollable)                                  │     │
│  │  ├─ User messages (right-aligned)                           │     │
│  │  └─ Assistant messages (left-aligned)                       │     │
│  │       └─ Answer text with inline citations                  │     │
│  │       └─ 📚 Source references (expandable)                  │     │
│  │            ├─ Chunk #1: source, type, score, preview        │     │
│  │            ├─ Chunk #2: ...                                 │     │
│  │            └─ ...                                           │     │
│  │                                                             │     │
│  │  Chat input box (bottom)                                    │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Pipeline: user input → generate_with_citation() → answer + sources │
│  Error states: NotImplementedError → fallback message              │
│                Exception → error message                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Interface Contract

### 3.1 Page Config

| Property | Value |
|---|---|
| `page_title` | `"University Services RAG Chatbot"` |
| `page_icon` | `"🎓"` |
| `layout` | `"wide"` |
| `initial_sidebar_state` | `"expanded"` |

### 3.2 Session State Keys

| Key | Type | Default | Purpose |
|---|---|---|---|
| `messages` | `list[dict]` | `[]` | Chat history: `{"role": "user"\|"assistant", "content": str, "sources": list\|None}` |
| `pending_query` | `str \| None` | `None` | Triggered by suggested question button clicks |

### 3.3 Backend Integration

The UI calls Task 10's `generate_with_citation()`:

```python
from src.task10_generation import generate_with_citation

response = generate_with_citation(query, top_k=top_k)
answer = response.get("answer", "")
sources = response.get("sources", [])
retrieval_source = response.get("retrieval_source", "none")
```

---

## 4. UI Components

### 4.1 Sidebar

**Section 1 — Title & Branding**
```
🎓 University Services RAG
Trợ lý hỏi đáp về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện)
```

**Section 2 — Suggested Questions (5 buttons)**
Each button is a `st.button(use_container_width=True)`. When clicked, it sets `st.session_state["pending_query"]` to the question text, which triggers the query handler.

| # | Question (Vietnamese) |
|---|---|
| 1 | `"Học phí tại USSH là bao nhiêu?"` |
| 2 | `"Làm sao để đặt phòng học nhóm ở thư viện?"` |
| 3 | `"Điều kiện xin học bổng Academic Achievement?"` |
| 4 | `"Dịch vụ hỗ trợ chỗ ở cho sinh viên như thế nào?"` |
| 5 | `"Cách đăng ký học phần tại USSH như thế nào?"` |

**Section 3 — Settings**
- `top_k` slider: `st.slider("Số chunks retrieval (top_k)", 3, 10, 5)`
- Controls how many chunks are retrieved and passed to the LLM

**Section 4 — Architecture Info**
```
Kiến trúc hệ thống:
Hybrid Retrieval (Semantic + BM25) → RRF Rerank
→ PageIndex Fallback → LLM Generation có Citation
```

### 4.2 Main Chat Area

**Header**
```
🎓 University Services RAG Chatbot
Hệ thống hỏi đáp thông tin dịch vụ đại học
(Học phí, Học bổng, Ký túc xá, Thư viện)
```

**Chat History Rendering**
- Iterate through `st.session_state.messages`
- Each message displayed via `st.chat_message(msg["role"])`
- Assistant messages with sources show an expander `📚 Nguồn tham khảo (N chunks)`

**Source Reference Display (per chunk)**
```
[{i}] {source_name} {doc_type} | score: {score:.4f}
{content_preview[:300]}...
```

Where:
- `source_name` = `metadata.get("source", "Unknown")`
- `doc_type` = `metadata.get("type", "unknown")`
- `score` = `chunk.get("score", 0)`
- `content_preview` = `chunk.get("content", "")[:300]`

**Chat Input**
- `st.chat_input("Nhập câu hỏi của bạn về chính sách/dịch vụ đại học...")`
- Processes both typed input and `pending_query` from suggested buttons

### 4.3 Query Flow

```
User action (type or click suggestion)
  → Set query = user_input or pending_query
  → Clear pending_query
  → Append user message to session state
  → Display user message
  → Show spinner "Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."
  → Call generate_with_citation(query, top_k=top_k)
  → Display answer
  → If sources exist: show expander with source references
  → Append assistant message + sources to session state
```

---

## 5. Error Handling

| Scenario | UI Behavior |
|---|---|
| `generate_with_citation` raises `NotImplementedError` | `"⚠️ **Task 10 chưa được implement.** Hãy hoàn thành src/task10_generation.py để kết nối pipeline vào UI!"` |
| `generate_with_citation` raises any other `Exception` | `"❌ **Lỗi khi chạy RAG Pipeline:** {error_message}"` |
| Empty sources list | No source expander shown |
| API key missing | Exception propagates to UI with clear error message |
| `top_k` out of range | Clamped by slider (3–10) |

---

## 6. Verification

### 6.1 Run the app

```bash
streamlit run app.py
```

**Expected:** Chat interface loads in browser at `http://localhost:8501`

### 6.2 Manual tests

| Test Case | Steps | Expected Result |
|---|---|---|
| Type a question | Type in chat input, press Enter | Answer appears with cited sources |
| Click suggested question | Click any button in sidebar | Question appears in chat, answer loads |
| Adjust top_k | Move slider to 3, ask question | Fewer sources in expander |
| Multiple turns | Ask 2-3 questions consecutively | Full chat history visible |
| Missing Task 10 | Comment out `generate_with_citation` import | Warning message shown |
| Invalid API key | Set wrong key in `.env` | Error message visible |

### 6.3 Checkpoint 5 Pass Criteria

From LAB_GUIDE.md:

> ✅ Chatbot UI phản hồi chính xác kèm danh sách nguồn; báo cáo `results.md` hiển thị đầy đủ bảng điểm đánh giá A/B testing (`CP5 Passed`).

---

## 7. Data Flow

```
User Input
  │
  ▼
app.py (Streamlit)
  │
  ├── top_k from sidebar slider
  │
  ▼
generate_with_citation(query, top_k=top_k)  ← Task 10
  │
  ├── retrieve(query, top_k)                ← Task 9
  │     ├── semantic_search()               ← Task 5
  │     ├── lexical_search()                ← Task 6
  │     ├── rerank_rrf()                    ← Task 7
  │     └── pageindex_search() (fallback)   ← Task 8
  │
  ├── reorder_for_llm()                     ← Task 10
  ├── format_context()                      ← Task 10
  └── LLM call (OpenRouter)                 ← Task 10
  │
  ▼
Return: {"answer": str, "sources": list, "retrieval_source": str}
  │
  ▼
app.py renders:
  ├── Answer markdown with citations
  └── Source expander (N chunks with metadata)
```

---

## 8. Configuration

All configuration is handled through the Streamlit UI sidebar (no separate config file needed):

| Setting | UI Element | Range | Default |
|---|---|---|---|
| `top_k` | Slider | 3–10 | 5 |

The following are configured in `.env` (shared across all tasks):

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | LLM API key for generation |
| `PAGEINDEX_API_KEY` | PageIndex API key for fallback |

---

## 9. Edge Cases & Notes

| Scenario | Expected Behavior |
|---|---|
| First load (empty chat) | Welcome header + chat input visible |
| Rapid consecutive clicks | Each click processes independently; Streamlit reruns |
| Very long answer | Answer wraps naturally in markdown container |
| API returns empty answer | Show empty message; sources still displayed if available |
| `pending_query` + typed input simultaneously | `user_input` takes priority (typed after click) |
| Page refresh | Session state resets; chat history lost |

### Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'streamlit'` | Streamlit not installed | `pip install streamlit>=1.35.0` |
| `ModuleNotFoundError: No module named 'src.task10_generation'` | Task 10 not implemented | Implement `generate_with_citation()` |
| `streamlit run app.py` not found | Wrong working directory | Run from project root: `cd K3-Day08-RAG-Pipeline` |
| API error in generation | Missing/invalid `.env` key | Check `OPENROUTER_API_KEY` in `.env` |
| Sources not displayed | `sources` list empty | Check retrieval pipeline (Task 9) |

---

## 10. Checklist Triển khai

- [ ] `st.set_page_config()` with correct title, icon, layout
- [ ] Sidebar: title, caption, divider
- [ ] Sidebar: 5 suggested question buttons with `pending_query` state
- [ ] Sidebar: `top_k` slider (3–10, default 5)
- [ ] Sidebar: architecture info text
- [ ] Session state: `messages` list initialized
- [ ] Session state: `pending_query` initialized
- [ ] Chat history rendering: user + assistant messages
- [ ] Source expander for assistant messages with `sources`
- [ ] Chat input box with placeholder
- [ ] Query handler: typed input OR `pending_query`
- [ ] Integration: `generate_with_citation(query, top_k=top_k)`
- [ ] Error handling: `NotImplementedError` → warning message
- [ ] Error handling: `Exception` → error message
- [ ] `streamlit run app.py` works without import errors
- [ ] Suggested question buttons trigger query
- [ ] `top_k` slider changes affect retrieval count
- [ ] Source expander shows correct metadata per chunk
- [ ] Multiple chat turns maintain history