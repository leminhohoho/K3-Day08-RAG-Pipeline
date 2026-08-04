"""
Task 10 — Generation Có Citation.

Pipeline:
    1. Retrieve chunks từ Task 9
    2. Reorder để tránh "lost in the middle"
    3. Format context với source labels
    4. Build prompt (system + context + question)
    5. Call LLM (OpenRouter)
    6. Return answer + sources
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on path for direct script execution
_proj_root = Path(__file__).parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

try:
    from .task9_retrieval_pipeline import retrieve  # type: ignore
except ImportError:
    from src.task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_MODEL = "openai/gpt-4o-mini"  # Hoặc ":free" model nếu chưa có credit


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    Strategy: front + back[::-1]
    Input (by score):  [1, 2, 3, 4, 5]
    Output:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return list(chunks)

    front = chunks[::2]  # indices 0, 2, 4, ...
    back = chunks[1::2]  # indices 1, 3, ...
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n{chunk['content']}"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM (OpenRouter)
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số lượng chunks đưa vào context

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex' hoặc 'none'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Step 5: Call LLM (OpenRouter — OpenAI-compatible API)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY or OPENAI_API_KEY in .env")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content or ""

    # Step 6: Return
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    test_queries = [
        "Học phí tại USSH là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên quốc tế có những học bổng nào?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(
            f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]"
        )

