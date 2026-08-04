"""Contract tests for the reproducible Task 3 command entrypoint."""

from src import task3_convert_markdown as entrypoint


def test_news_entrypoint_runs_canonical_optimizer(monkeypatch):
    calls = []

    def fake_optimizer(**kwargs):
        calls.append(kwargs)
        return ["news-result"]

    monkeypatch.setattr(entrypoint, "_load_optimizer", lambda: fake_optimizer)

    result = entrypoint.convert_news_articles(dry_run=True)

    assert result == ["news-result"]
    assert calls == [{"branch": "news", "dry_run": True}]


def test_all_entrypoint_can_reuse_existing_legal_extraction(monkeypatch):
    calls = []

    def unexpected_extraction():
        raise AssertionError("legal extraction should have been reused")

    def fake_optimizer(**kwargs):
        calls.append(kwargs)
        return ["all-result"]

    monkeypatch.setattr(entrypoint, "extract_legal_docs", unexpected_extraction)
    monkeypatch.setattr(entrypoint, "_load_optimizer", lambda: fake_optimizer)

    result = entrypoint.convert_all(reuse_legal_extraction=True)

    assert result == ["all-result"]
    assert calls == [{"branch": "all", "dry_run": False}]


def test_all_entrypoint_extracts_legal_before_optimization(monkeypatch):
    calls = []

    monkeypatch.setattr(entrypoint, "extract_legal_docs", lambda: calls.append("extract"))

    def fake_optimizer(**kwargs):
        calls.append(("optimize", kwargs))
        return []

    monkeypatch.setattr(entrypoint, "_load_optimizer", lambda: fake_optimizer)

    entrypoint.convert_all()

    assert calls == ["extract", ("optimize", {"branch": "all", "dry_run": False})]
