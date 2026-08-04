"""Offline tests for the safe, model-free Task 2 crawler."""

import json
from pathlib import Path

import pytest

from src.task2_crawl_news import (
    NEWS_SOURCES,
    NewsSource,
    build_article_payload,
    choose_retrieval_markdown,
    extract_published_at,
    promote_staged_run,
    strip_related_sections,
    validate_article_payload,
)


class FakeMarkdown:
    def __init__(self, raw: str, fit: str):
        self.raw_markdown = raw
        self.fit_markdown = fit


class FakeResult:
    success = True
    status_code = 200
    error_message = ""

    def __init__(self, source: NewsSource, raw: str, fit: str):
        self.url = source.url
        self.markdown = FakeMarkdown(raw, fit)
        self.metadata = {"title": source.title}


def _source() -> NewsSource:
    return NewsSource(
        document_id="sample-news",
        filename="article_01.json",
        url="https://ussh.vnu.edu.vn/vi/news/sample.html",
        title="Bài viết mẫu",
        published_at="2026-01-01",
        organization="VNU-USSH",
    )


def test_registry_matches_current_landing_urls():
    landing = Path("data/landing/news")

    assert len(NEWS_SOURCES) == 7
    for source in NEWS_SOURCES:
        payload = json.loads((landing / source.filename).read_text(encoding="utf-8-sig"))
        assert payload["url"] == source.url


def test_fit_markdown_is_selected_only_when_substantial():
    raw = "Nội dung nguồn đầy đủ. " * 200
    good_fit = "Nội dung chính đã loại menu. " * 80
    tiny_fit = "Quá ngắn. " * 10

    selected, kind = choose_retrieval_markdown(raw, good_fit, min_content_chars=500)
    assert selected == good_fit.strip()
    assert kind == "fit_markdown"

    selected, kind = choose_retrieval_markdown(raw, tiny_fit, min_content_chars=500)
    assert selected == raw.strip()
    assert kind == "raw_markdown"


def test_related_story_list_is_removed_from_retrieval_content():
    article = "Nội dung bài chính.\n\n**Tin bài liên quan:**\n[Bài khác](https://example.com)"

    cleaned, transforms = strip_related_sections(article)

    assert cleaned == "Nội dung bài chính."
    assert transforms == ["strip_related_story_list"]


def test_published_date_prefers_official_page_metadata():
    metadata = {"article:published_time": "2026-06-01+0708:08:00"}

    value, source = extract_published_at(metadata, "2026-05-31")

    assert value == "2026-06-01"
    assert source == "page_metadata"


def test_payload_uses_compact_schema_with_raw_and_content_hashes():
    source = _source()
    raw = "Nội dung nguồn đầy đủ từ trang chính thức. " * 150
    fit = "Nội dung bài viết đã loại phần điều hướng. " * 80

    payload = build_article_payload(
        source,
        FakeResult(source, raw, fit),
        min_content_chars=500,
    )

    assert payload["content_kind"] == "source_extract"
    assert payload["raw_markdown"] == raw.strip()
    assert payload["content_markdown"] == raw.strip()
    assert payload["content_transforms"] == []
    assert len(payload["raw_content_sha256"]) == 64
    assert len(payload["content_sha256"]) == 64
    assert "fit_markdown" not in payload
    assert "quality" not in payload
    assert "content_selection" not in payload
    assert "canonical_url" not in payload


def test_validation_rejects_rag_generated_prompt_sections():
    source = _source()
    content = (
        "Nội dung nguồn chính thức đủ dài. " * 50
        + "\n\n## Các câu hỏi hệ thống RAG có thể trả lời\n- Câu hỏi mẫu"
    )
    payload = {
        "document_id": source.document_id,
        "url": source.url,
        "canonical_url": source.url,
        "title": source.title,
        "content_kind": "source_extract",
        "raw_markdown": content,
        "content_markdown": content,
        "content_transforms": [],
        "raw_content_sha256": __import__("hashlib").sha256(content.encode()).hexdigest(),
        "content_sha256": __import__("hashlib").sha256(content.encode()).hexdigest(),
    }

    with pytest.raises(ValueError, match="RAG-generated"):
        validate_article_payload(payload, source, min_content_chars=500)


def test_promote_keeps_backup_of_previous_landing(tmp_path):
    source = _source()
    run_dir = tmp_path / "run"
    landing_dir = tmp_path / "landing"
    run_dir.mkdir()
    landing_dir.mkdir()
    old_payload = {"old": True}
    (landing_dir / source.filename).write_text(json.dumps(old_payload), encoding="utf-8")

    raw = "Nội dung nguồn đầy đủ từ trang chính thức. " * 150
    fit = "Nội dung bài viết đã loại phần điều hướng. " * 80
    new_payload = build_article_payload(
        source,
        FakeResult(source, raw, fit),
        min_content_chars=500,
    )
    (run_dir / source.filename).write_text(
        json.dumps(new_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    backup_dir = promote_staged_run(
        run_dir,
        [source],
        landing_dir=landing_dir,
        min_content_chars=500,
    )

    promoted = json.loads((landing_dir / source.filename).read_text(encoding="utf-8"))
    backup = json.loads((backup_dir / source.filename).read_text(encoding="utf-8"))
    assert promoted["document_id"] == source.document_id
    assert backup == old_payload
