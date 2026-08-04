"""Opt-in PageIndex live smoke test.

Run with:
    RUN_PAGEINDEX_LIVE_TESTS=1 python -m pytest tests/test_task8_live.py -v -s
"""

import os

import pytest


pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.getenv("RUN_PAGEINDEX_LIVE_TESTS") != "1",
    reason="set RUN_PAGEINDEX_LIVE_TESTS=1 to call PageIndex",
)
@pytest.mark.parametrize(
    ("query", "expected_document_id"),
    [
        (
            "Sinh viên ngành Tâm lý học phải đóng mức học lại bao nhiêu cho một tín chỉ?",
            "ussh-tuition-plan-semester-1-2025-2026",
        ),
        (
            "VNU-USSH sử dụng những phương thức nào để tuyển sinh đại học chính quy năm 2026?",
            "ussh-undergraduate-admissions-2026",
        ),
    ],
)
def test_pageindex_live_returns_real_evidence(query, expected_document_id):
    from src.task8_pageindex_vectorless import get_pageindex_status, pageindex_search

    status = get_pageindex_status()
    assert status["configured"] is True
    assert status["ready_documents"] >= 10

    results = pageindex_search(query, top_k=3)
    assert results
    assert all(result["source"] == "pageindex" for result in results)
    assert all(result["metadata"]["pageindex_doc_id"] for result in results)
    assert all(result["metadata"]["node_id"] != "unknown" for result in results)
    assert all(result["metadata"]["page_index"] is not None for result in results)
    assert expected_document_id in {
        result["metadata"]["document_id"] for result in results
    }
