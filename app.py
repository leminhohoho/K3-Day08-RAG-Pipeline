"""
RAG Chatbot — University Services (Starter Template)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🎓 University Services RAG")
    st.caption("Trợ lý hỏi đáp về dịch vụ và chính sách đại học (học phí, học bổng, ký túc xá, thư viện)")

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Học phí tại Trường Đại học Khoa học Xã hội và Nhân văn (USSH) là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Điều kiện xin học bổng là gì?",
        "Dịch vụ hỗ trợ chỗ ở cho sinh viên như thế nào?",
        "Cách đăng ký học phần tại USSH như thế nào?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)
    retrieval_mode_label = st.selectbox(
        "Chế độ retrieval",
        ("Auto fallback", "Hybrid only", "PageIndex direct"),
        help=(
            "Auto dùng Hybrid và chỉ fallback sang PageIndex khi dense confidence thấp. "
            "PageIndex direct phục vụ demo vectorless route minh bạch."
        ),
    )
    retrieval_mode = {
        "Auto fallback": "auto",
        "Hybrid only": "hybrid",
        "PageIndex direct": "pageindex",
    }[retrieval_mode_label]

    from src.task8_pageindex_vectorless import get_pageindex_status

    pageindex_status = get_pageindex_status()
    if pageindex_status["available"]:
        st.success(
            f"PageIndex sẵn sàng: {pageindex_status['ready_documents']} tài liệu"
        )
    elif pageindex_status["configured"]:
        st.warning("PageIndex đã có key nhưng chưa có tài liệu retrieval-ready")
    else:
        st.info("PageIndex chưa được cấu hình; Auto vẫn dùng Hybrid an toàn")

    st.divider()
    st.caption("**Kiến trúc hệ thống:**")
    st.caption("Hybrid Retrieval (Semantic + BM25) → RRF Rerank → PageIndex Fallback → LLM Generation có Citation")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🎓 University Services RAG Chatbot")
st.caption("Hệ thống hỏi đáp thông tin dịch vụ đại học (Học phí, Học bổng, Ký túc xá, Thư viện)")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander(f"📚 Nguồn tham khảo ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "Unknown")
                    doc_type = meta.get("type", "unknown")
                    display_score = src.get("confidence_score") or src.get("score", 0)
                    score_label = (
                        "rank proxy"
                        if src.get("score_type") == "rank_proxy"
                        else "similarity"
                    )
                    page = meta.get("page_index")
                    page_label = f" | trang: `{page}`" if page is not None else ""
                    st.markdown(
                        f"**[{i}] {source_name}** `{doc_type}` | "
                        f"{score_label}: `{display_score:.4f}`{page_label}"
                    )
                    st.text(src.get("content", ""))
                    st.divider()

# =============================================================================
# QUERY HANDLING
# =============================================================================

# Xử lý khi bấm nút gợi ý hoặc nhập câu hỏi mới
user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/dịch vụ đại học...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            try:
                from src.task10_generation import generate_with_citation
                response = generate_with_citation(
                    query,
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                )
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])

            except NotImplementedError:
                answer = "⚠️ **Task 10 chưa được implement.** Hãy hoàn thành `src/task10_generation.py` để kết nối pipeline vào UI!"
                sources = []
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source", "Unknown")
                        doc_type = meta.get("type", "unknown")
                        display_score = src.get("confidence_score") or src.get("score", 0)
                        score_label = (
                            "rank proxy"
                            if src.get("score_type") == "rank_proxy"
                            else "similarity"
                        )
                        page = meta.get("page_index")
                        page_label = f" | trang: `{page}`" if page is not None else ""
                        st.markdown(
                            f"**[{i}] {source_name}** `{doc_type}` | "
                            f"{score_label}: `{display_score:.4f}`{page_label}"
                        )
                        st.text(src.get("content", ""))
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
