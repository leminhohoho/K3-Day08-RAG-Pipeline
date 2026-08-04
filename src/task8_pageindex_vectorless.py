"""
Task 8 — PageIndex Vectorless RAG Fallback.

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Vì không có PAGEINDEX_API_KEY thật, module này trả [] cho mọi query.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PROJECT_ROOT = Path(__file__).parent.parent
PAGEINDEX_REGISTRY_PATH = PROJECT_ROOT / "pageindex_doc_ids.json"


def upload_documents(force: bool = False, wait_until_ready: bool = True) -> dict[str, str]:
    """
    Upload documents lên PageIndex.

    Vì không có API key thật, trả {} và in hướng dẫn.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa được set trong .env")
        print("  Đăng ký tại: https://pageindex.ai/")
        print("  Upload sẽ bỏ qua.")
        return {}
    # TODO: Implement real upload khi có API key
    print("❌ Chưa implement upload_documents với API key thật")
    return {}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex (fallback).

    Vì không có API key thật, luôn trả [].
    Khi có key, gọi PageIndex API và parse response.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,         # rank proxy score
            'score_type': 'rank_proxy',
            'confidence_score': None,
            'source': 'pageindex',
            'metadata': dict,
            'raw_scores': dict
        }
    """
    if not query or top_k <= 0:
        return []

    if not PAGEINDEX_API_KEY:
        return []

    # TODO: Implement real PageIndex search khi có API key
    return []


def get_pageindex_status() -> dict:
    """
    Return sanitized availability/readiness diagnostics.
    """
    return {
        "available": bool(PAGEINDEX_API_KEY),
        "configured": bool(PAGEINDEX_API_KEY),
        "ready_documents": 0,
        "total_registered_documents": 0,
        "last_error_type": None,
        "last_error_message": None,
    }


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa được set trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
        print("  Module sẽ trả [] cho mọi query.")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")