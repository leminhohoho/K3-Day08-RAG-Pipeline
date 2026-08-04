r"""Task 2: safely crawl official VNU-USSH pages for the RAG corpus.

The crawler is deliberately model-free. It targets the official article body
(``#news-bodyhtml``), stores Crawl4AI raw Markdown for auditability, and keeps
fit Markdown when the optional filtered version passes quality checks. New data
is always written to a timestamped staging directory;
``data/landing/news`` is changed only after every expected article validates.

Examples (PowerShell, after installing requirements-crawl.txt):

    .\.venv-crawl\Scripts\python.exe src\task2_crawl_news.py
    .\.venv-crawl\Scripts\python.exe src\task2_crawl_news.py --promote
    .\.venv-crawl\Scripts\python.exe src\task2_crawl_news.py --promote-run <run-dir>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
LANDING_DIR = DATA_DIR / "landing" / "news"
STAGING_ROOT = DATA_DIR / "staging" / "news"
MANIFEST_PATH = DATA_DIR / "sources_manifest.json"

ALLOWED_DOMAINS = {"ussh.vnu.edu.vn"}
ARTICLE_BODY_SELECTOR = "#news-bodyhtml"
MIN_ARTICLES = 5
DEFAULT_MIN_CONTENT_CHARS = 1_200
DEFAULT_RETRIES = 2
DEFAULT_REQUEST_DELAY = 0.75

RAG_LEAKAGE_RE = re.compile(
    r"(?:các?\s+câu\s+hỏi\s+hệ\s+thống\s+rag|"
    r"giá\s+trị\s+dữ\s+liệu\s+đối\s+với\s+hệ\s+thống\s+rag)",
    re.IGNORECASE,
)
BLOCK_PAGE_RE = re.compile(
    r"(?:access denied|page not found|404 not found|captcha|cloudflare challenge|"
    r"trang không tồn tại|không tìm thấy trang)",
    re.IGNORECASE,
)
RELATED_SECTION_RE = re.compile(
    r"(?im)^\s*\*{0,2}(?:tin\s+bài\s+liên\s+quan|bài\s+viết\s+liên\s+quan|"
    r"tin\s+liên\s+quan)\s*:?[ \t]*\*{0,2}\s*$"
)


@dataclass(frozen=True)
class NewsSource:
    document_id: str
    filename: str
    url: str
    title: str
    published_at: str
    organization: str
    language: str = "vi"
    min_content_chars: int | None = None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_news_sources(path: Path = MANIFEST_PATH) -> list[NewsSource]:
    """Use the provenance manifest as the single source of truth for URLs."""

    if not path.exists():
        raise FileNotFoundError(f"Missing source registry: {path}")
    entries = json.loads(path.read_text(encoding="utf-8-sig"))
    sources: list[NewsSource] = []
    for entry in entries:
        if entry.get("source_type") != "news":
            continue
        landing_path = Path(str(entry.get("landing_path", "")))
        source = NewsSource(
            document_id=str(entry.get("document_id", "")).strip(),
            filename=landing_path.name,
            url=str(entry.get("url", "")).strip(),
            title=str(entry.get("title", "")).strip(),
            published_at=str(entry.get("published_at", "")).strip(),
            organization=str(entry.get("organization", "")).strip(),
            language=str(entry.get("language") or "vi").strip(),
            min_content_chars=(
                int(entry["min_content_chars"])
                if entry.get("min_content_chars") is not None
                else None
            ),
        )
        sources.append(source)

    sources.sort(key=lambda item: item.filename)
    validate_source_registry(sources)
    return sources


def validate_source_registry(sources: list[NewsSource]) -> None:
    if len(sources) < MIN_ARTICLES:
        raise ValueError(f"Need at least {MIN_ARTICLES} news sources, found {len(sources)}")
    ids = [item.document_id for item in sources]
    filenames = [item.filename for item in sources]
    urls = [item.url for item in sources]
    if any(not value for value in ids + filenames + urls):
        raise ValueError("Every news source requires document_id, filename and URL")
    if len(ids) != len(set(ids)) or len(filenames) != len(set(filenames)):
        raise ValueError("News source registry contains duplicate IDs or filenames")
    if len(urls) != len(set(urls)):
        raise ValueError("News source registry contains duplicate URLs")
    for source in sources:
        if not re.fullmatch(r"article_\d{2}\.json", source.filename):
            raise ValueError(f"Unexpected news filename: {source.filename}")
        host = (urlparse(source.url).hostname or "").casefold()
        if host not in ALLOWED_DOMAINS:
            raise ValueError(f"Source is outside approved official domains: {source.url}")
        if source.min_content_chars is not None and source.min_content_chars < 500:
            raise ValueError(
                f"Source-specific min_content_chars must be >=500: {source.document_id}"
            )


def required_min_content_chars(source: NewsSource, fallback: int) -> int:
    """Return a documented per-source threshold, or the CLI/default fallback."""

    return source.min_content_chars if source.min_content_chars is not None else fallback


# Kept for compatibility with the starter code and simple lab demonstrations.
NEWS_SOURCES = load_news_sources()
ARTICLE_URLS = [source.url for source in NEWS_SOURCES]


def _markdown_value(markdown_result: Any, attribute: str) -> str:
    if markdown_result is None:
        return ""
    if isinstance(markdown_result, str):
        return markdown_result.strip() if attribute == "raw_markdown" else ""
    value = getattr(markdown_result, attribute, None)
    return value.strip() if isinstance(value, str) else ""


def extract_markdown_variants(markdown_result: Any) -> tuple[str, str]:
    """Return ``(raw_markdown, fit_markdown)`` across Crawl4AI result variants."""

    raw = _markdown_value(markdown_result, "raw_markdown")
    fit = _markdown_value(markdown_result, "fit_markdown")
    if not raw and markdown_result is not None and not isinstance(markdown_result, str):
        fallback = getattr(markdown_result, "markdown_with_citations", None)
        raw = fallback.strip() if isinstance(fallback, str) else ""
    if not raw and markdown_result is not None:
        rendered = str(markdown_result).strip()
        if rendered and not rendered.startswith("<"):
            raw = rendered
    return raw, fit


def extract_markdown(markdown_result: Any) -> str:
    """Backward-compatible helper returning fit Markdown when it is available."""

    raw, fit = extract_markdown_variants(markdown_result)
    return fit or raw


def clean_crawled_markdown(text: str) -> str:
    """Normalize encoding/control artifacts without summarizing source content."""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = "".join(
        char for char in text if char in "\n\t" or unicodedata.category(char) != "Cc"
    )
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{4,}", "\n\n\n", text).strip()


def strip_related_sections(text: str) -> tuple[str, list[str]]:
    """Remove in-page related-story lists that would corrupt source provenance."""

    match = RELATED_SECTION_RE.search(text)
    if not match:
        return text, []
    return text[: match.start()].rstrip(), ["strip_related_story_list"]


def choose_retrieval_markdown(
    raw: str,
    fit: str,
    *,
    min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS,
) -> tuple[str, str]:
    """Prefer fit Markdown only when it retains a meaningful source body."""

    raw = clean_crawled_markdown(raw)
    fit = clean_crawled_markdown(fit)
    if fit and len(fit) >= min_content_chars and (not raw or len(fit) >= len(raw) * 0.18):
        return fit, "fit_markdown"
    return raw, "raw_markdown"


def _crawler_version() -> str:
    try:
        return importlib.metadata.version("crawl4ai")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def extract_published_at(metadata: dict[str, Any], fallback: str) -> tuple[str, str]:
    """Read the local publication date from page metadata when Crawl4AI exposes it."""

    for key in (
        "article:published_time",
        "article_published_time",
        "published_time",
        "datePublished",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", str(value))
        if match:
            return match.group(1), "page_metadata"
    return fallback, "manifest"


def build_article_payload(
    source: NewsSource,
    result: Any,
    *,
    min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS,
) -> dict[str, Any]:
    if not getattr(result, "success", True):
        message = getattr(result, "error_message", "unknown crawl error")
        raise RuntimeError(f"Crawl failed for {source.url}: {message}")
    status_code = getattr(result, "status_code", None)
    if isinstance(status_code, int) and status_code >= 400:
        raise RuntimeError(f"HTTP {status_code} for {source.url}")

    raw, _fit = extract_markdown_variants(getattr(result, "markdown", None))
    clean_raw = clean_crawled_markdown(raw)
    selected = clean_raw
    selected, content_transforms = strip_related_sections(selected)
    metadata = getattr(result, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    published_at, _published_at_source = extract_published_at(metadata, source.published_at)
    crawled_title = str(
        metadata.get("title") or metadata.get("og:title") or source.title
    ).strip()
    final_url = str(getattr(result, "url", "") or source.url).strip()
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    payload: dict[str, Any] = {
        "schema_version": 2,
        "document_id": source.document_id,
        "url": source.url,
        "title": crawled_title,
        "published_at": published_at,
        "date_crawled": now,
        "language": source.language,
        "source_type": "news",
        "content_kind": "source_extract",
        "crawl_backend": "crawl4ai",
        "http_status": status_code if isinstance(status_code, int) else "unknown",
        "content_selector": ARTICLE_BODY_SELECTOR,
        "raw_markdown": clean_raw,
        "content_markdown": selected,
        "content_transforms": content_transforms,
        "raw_content_sha256": _sha256_text(clean_raw),
        "content_sha256": _sha256_text(selected),
    }
    if final_url.rstrip("/") != source.url.rstrip("/"):
        payload["canonical_url"] = final_url
    validate_article_payload(payload, source, min_content_chars=min_content_chars)
    return payload


def validate_article_payload(
    payload: dict[str, Any],
    source: NewsSource,
    *,
    min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS,
) -> None:
    if payload.get("document_id") != source.document_id:
        raise ValueError(f"document_id mismatch for {source.filename}")
    if payload.get("url") != source.url:
        raise ValueError(f"Source URL mismatch for {source.filename}")
    canonical_url = str(payload.get("canonical_url") or payload.get("url", ""))
    canonical_host = (urlparse(canonical_url).hostname or "").casefold()
    if canonical_host not in ALLOWED_DOMAINS:
        raise ValueError(f"Redirected outside approved domains: {canonical_url}")
    title = str(payload.get("title", "")).strip()
    content = str(payload.get("content_markdown", "")).strip()
    required_min_chars = required_min_content_chars(source, min_content_chars)
    if not title:
        raise ValueError(f"Missing title for {source.filename}")
    if len(content) < required_min_chars:
        raise ValueError(
            f"Content too short for {source.filename}: {len(content)} < {required_min_chars}"
        )
    if BLOCK_PAGE_RE.search((title + "\n" + content[:1_500])):
        raise ValueError(f"Possible error/block page for {source.filename}")
    if RAG_LEAKAGE_RE.search(content):
        raise ValueError(f"RAG-generated prompt section detected in {source.filename}")
    if payload.get("content_kind") != "source_extract":
        raise ValueError(f"Unexpected content_kind for {source.filename}")
    if payload.get("content_sha256") != _sha256_text(content):
        raise ValueError(f"Content checksum mismatch for {source.filename}")
    raw = str(payload.get("raw_markdown", ""))
    if payload.get("raw_content_sha256") != _sha256_text(raw):
        raise ValueError(f"Raw content checksum mismatch for {source.filename}")
    if not isinstance(payload.get("content_transforms"), list):
        raise ValueError(f"content_transforms must be a list for {source.filename}")


async def crawl_source(
    source: NewsSource,
    crawler: Any,
    run_config: Any,
    *,
    retries: int = DEFAULT_RETRIES,
    min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            result = await crawler.arun(url=source.url, config=run_config)
            return build_article_payload(
                source,
                result,
                min_content_chars=min_content_chars,
            )
        except Exception as error:  # retry browser/network and validation failures
            last_error = error
            if attempt <= retries:
                await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Failed after {retries + 1} attempts: {source.url}: {last_error}")


async def crawl_article(url: str, crawler: Any) -> dict[str, Any]:
    """Compatibility wrapper for the original lab starter API."""

    source = next((item for item in NEWS_SOURCES if item.url == url), None)
    if source is None:
        raise ValueError(f"URL is not registered in sources_manifest.json: {url}")
    return await crawl_source(source, crawler, run_config=None)


def validate_staged_run(
    run_dir: Path,
    sources: list[NewsSource] | None = None,
    *,
    min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS,
) -> list[dict[str, Any]]:
    sources = sources or NEWS_SOURCES
    payloads: list[dict[str, Any]] = []
    for source in sources:
        path = run_dir / source.filename
        if not path.exists():
            raise FileNotFoundError(f"Staged article is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        validate_article_payload(payload, source, min_content_chars=min_content_chars)
        payloads.append(payload)
    return payloads


def promote_staged_run(
    run_dir: Path,
    sources: list[NewsSource] | None = None,
    *,
    landing_dir: Path = LANDING_DIR,
    min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS,
) -> Path:
    """Promote a complete validated run with backups and best-effort rollback."""

    sources = sources or NEWS_SOURCES
    validate_staged_run(run_dir, sources, min_content_chars=min_content_chars)
    landing_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = run_dir / "_backup_before_promote"
    backup_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[NewsSource, Path, Path]] = []

    for source in sources:
        staged = run_dir / source.filename
        target = landing_dir / source.filename
        temporary = landing_dir / f".{source.filename}.pending"
        temporary.write_bytes(staged.read_bytes())
        if target.exists():
            shutil.copy2(target, backup_dir / source.filename)
        pending.append((source, temporary, target))

    replaced: list[tuple[NewsSource, Path]] = []
    try:
        for source, temporary, target in pending:
            os.replace(temporary, target)
            replaced.append((source, target))
    except Exception:
        for source, target in reversed(replaced):
            backup = backup_dir / source.filename
            if backup.exists():
                shutil.copy2(backup, target)
            elif target.exists():
                target.unlink()
        raise
    finally:
        for _, temporary, _ in pending:
            if temporary.exists():
                temporary.unlink()
    return backup_dir


async def crawl_all(
    *,
    promote: bool = False,
    retries: int = DEFAULT_RETRIES,
    min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> Path:
    """Crawl all registered sources into a new staging run."""

    # Imports follow the official Crawl4AI 0.9.x configuration API and stay
    # lazy so Task 1/3 and offline tests do not require the crawler environment.
    from crawl4ai import (
        AsyncWebCrawler,
        BrowserConfig,
        CacheMode,
        CrawlerRunConfig,
        DefaultMarkdownGenerator,
    )

    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    run_dir = STAGING_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        remove_overlay_elements=True,
        # All registered USSH pages expose their canonical article body through
        # this selector. Targeting it removes navigation, sharing controls and
        # related-news/footer text without summarising or rewriting the source.
        css_selector=ARTICLE_BODY_SELECTOR,
        markdown_generator=DefaultMarkdownGenerator(),
    )

    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "crawl_backend": "crawl4ai",
        "crawl_backend_version": _crawler_version(),
        "expected": len(NEWS_SOURCES),
        "succeeded": [],
        "failed": [],
        "promoted": False,
    }

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for index, source in enumerate(NEWS_SOURCES, start=1):
            print(f"[{index}/{len(NEWS_SOURCES)}] {source.document_id}: {source.url}")
            try:
                payload = await crawl_source(
                    source,
                    crawler,
                    run_config,
                    retries=retries,
                    min_content_chars=min_content_chars,
                )
                _atomic_write_json(run_dir / source.filename, payload)
                report["succeeded"].append(
                    {
                        "document_id": source.document_id,
                        "filename": source.filename,
                        "chars": len(payload["content_markdown"]),
                        "transforms": payload["content_transforms"],
                        "content_sha256": payload["content_sha256"],
                    }
                )
                print(
                    f"  OK {source.filename}: {len(payload['content_markdown'])} chars"
                )
            except Exception as error:
                report["failed"].append(
                    {
                        "document_id": source.document_id,
                        "filename": source.filename,
                        "error": str(error),
                    }
                )
                print(f"  ERROR: {error}")
            if request_delay > 0 and index < len(NEWS_SOURCES):
                await asyncio.sleep(request_delay)

    report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _atomic_write_json(run_dir / "_crawl_report.json", report)
    if report["failed"]:
        print(f"Run kept in staging; {len(report['failed'])} article(s) failed validation.")
        raise RuntimeError(
            f"Crawl run failed validation; inspect {run_dir / '_crawl_report.json'}"
        )

    validate_staged_run(run_dir, min_content_chars=min_content_chars)
    if promote:
        backup_dir = promote_staged_run(
            run_dir,
            min_content_chars=min_content_chars,
        )
        report["promoted"] = True
        report["promoted_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        report["backup_dir"] = str(backup_dir)
        _atomic_write_json(run_dir / "_crawl_report.json", report)
        print(f"Promoted {len(NEWS_SOURCES)} articles; backup: {backup_dir}")
    else:
        print(f"Validated staging run: {run_dir}")
        print(f"Review it, then promote with --promote-run \"{run_dir}\"")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true", help="promote only after all crawls pass")
    parser.add_argument("--promote-run", type=Path, help="promote an existing validated staging run")
    parser.add_argument("--list-sources", action="store_true", help="print the registered source set")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--min-content-chars", type=int, default=DEFAULT_MIN_CONTENT_CHARS)
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY)
    args = parser.parse_args()

    if args.retries < 0 or args.min_content_chars < 500 or args.request_delay < 0:
        parser.error("Require retries>=0, min-content-chars>=500 and request-delay>=0")
    if args.list_sources:
        for source in NEWS_SOURCES:
            print(f"{source.filename}\t{source.document_id}\t{source.url}")
        return
    if args.promote_run:
        backup = promote_staged_run(
            args.promote_run.resolve(),
            min_content_chars=args.min_content_chars,
        )
        print(f"Promoted validated run. Previous landing backup: {backup}")
        return
    asyncio.run(
        crawl_all(
            promote=args.promote,
            retries=args.retries,
            min_content_chars=args.min_content_chars,
            request_delay=args.request_delay,
        )
    )


if __name__ == "__main__":
    main()
