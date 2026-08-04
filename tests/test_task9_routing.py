"""Offline tests for explicit hybrid/PageIndex demo routing."""

from src import task9_retrieval_pipeline as task9
from src import task10_generation as task10


def _hybrid_result(score: float = 0.2) -> dict:
    return {
        "content": "policy evidence",
        "score": score,
        "metadata": {"source": "policy.md"},
    }


def test_pageindex_direct_does_not_call_hybrid(monkeypatch):
    expected = [{"content": "pageindex", "score": 1.0, "source": "pageindex"}]
    monkeypatch.setattr(task9, "pageindex_search", lambda query, top_k: expected)
    monkeypatch.setattr(
        task9,
        "semantic_search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dense called")),
    )

    assert task9.retrieve("học phí", retrieval_mode="pageindex") == expected


def test_hybrid_only_never_calls_pageindex(monkeypatch):
    dense = [_hybrid_result(0.2)]
    sparse = [_hybrid_result(1.0)]
    monkeypatch.setattr(task9, "semantic_search", lambda *_args, **_kwargs: dense)
    monkeypatch.setattr(task9, "lexical_search", lambda *_args, **_kwargs: sparse)
    monkeypatch.setattr(
        task9,
        "rerank_rrf",
        lambda *_args, **_kwargs: [_hybrid_result(0.03)],
    )
    monkeypatch.setattr(
        task9,
        "pageindex_search",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("PageIndex called")
        ),
    )

    result = task9.retrieve(
        "học phí", score_threshold=0.99, retrieval_mode="hybrid"
    )
    assert result[0]["source"] == "hybrid"


def test_auto_uses_pageindex_below_dense_threshold(monkeypatch):
    dense = [_hybrid_result(0.2)]
    sparse = [_hybrid_result(1.0)]
    expected = [{"content": "fallback", "score": 1.0, "source": "pageindex"}]
    monkeypatch.setattr(task9, "semantic_search", lambda *_args, **_kwargs: dense)
    monkeypatch.setattr(task9, "lexical_search", lambda *_args, **_kwargs: sparse)
    monkeypatch.setattr(
        task9,
        "rerank_rrf",
        lambda *_args, **_kwargs: [_hybrid_result(0.03)],
    )
    monkeypatch.setattr(task9, "pageindex_search", lambda *_args, **_kwargs: expected)

    assert task9.retrieve(
        "học phí", score_threshold=0.99, retrieval_mode="auto"
    ) == expected


def test_generation_forwards_retrieval_mode_without_calling_llm(monkeypatch):
    captured = {}

    def fake_retrieve(query, top_k, retrieval_mode):
        captured.update(query=query, top_k=top_k, retrieval_mode=retrieval_mode)
        return []

    monkeypatch.setattr(task10, "retrieve", fake_retrieve)
    response = task10.generate_with_citation(
        "học phí", top_k=3, retrieval_mode="pageindex"
    )

    assert captured == {
        "query": "học phí",
        "top_k": 3,
        "retrieval_mode": "pageindex",
    }
    assert response["retrieval_source"] == "none"
