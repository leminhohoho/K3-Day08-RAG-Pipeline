"""
Task 2 — Crawl tối thiểu 5 bài viết từ website công khai của trường đại học.

Output:
    data/landing/news/article_01.json
    data/landing/news/article_02.json
    ...

Mỗi file JSON chứa:
    - url
    - title
    - date_crawled
    - content_markdown
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


# Thư mục gốc của project
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Nơi lưu dữ liệu crawl
DATA_DIR = PROJECT_DIR / "data" / "landing" / "news"


# 5 bài viết trên website chính thức của USSH - ĐHQGHN
ARTICLE_URLS = [
    (
        "https://ussh.vnu.edu.vn/vi/news/thong-bao/"
        "thong-bao-ke-hoach-thu-hoc-phi-doi-voi-sinh-vien-"
        "dai-hoc-chinh-quy-ky-1-nam-hoc-2025-2026-23755.html"
    ),
    (
        "https://ussh.vnu.edu.vn/vi/news/thong-bao/"
        "thong-bao-ke-hoach-thu-hoc-phi-ky-2-"
        "nam-hoc-2025-2026-23955.html"
    ),
    (
        "https://ussh.vnu.edu.vn/vi/news/thong-bao/"
        "thong-bao-ke-hoach-thu-hoc-phi-bac-sau-dai-hoc-"
        "sdh-nam-hoc-2025-2026-23627.html"
    ),
    (
        "https://ussh.vnu.edu.vn/vi/news/thong-bao/"
        "thong-bao-tuyen-sinh-dai-hoc-chinh-quy-cac-chuong-trinh-"
        "dao-tao-thu-hai-nam-2025-dot-2-23771.html"
    ),
    (
        "https://ussh.vnu.edu.vn/vi/news/thong-bao/"
        "thong-bao-tuyen-sinh-dai-hoc-chinh-quy-cac-chuong-trinh-"
        "dao-tao-thu-hai-nam-2026-dot-1-24222.html"
    ),
]


def setup_directory() -> None:
    """Tạo thư mục data/landing/news nếu chưa tồn tại."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_markdown(result) -> str:
    """
    Lấy nội dung Markdown từ Crawl4AI.

    Một số phiên bản trả result.markdown là chuỗi.
    Một số phiên bản mới trả về MarkdownGenerationResult.
    """
    markdown = getattr(result, "markdown", None)

    if markdown is None:
        return ""

    # Crawl4AI phiên bản mới
    raw_markdown = getattr(markdown, "raw_markdown", None)
    if raw_markdown:
        return str(raw_markdown)

    # Crawl4AI phiên bản cũ hoặc markdown đã là chuỗi
    if isinstance(markdown, str):
        return markdown

    return str(markdown)


def fallback_title(url: str) -> str:
    """Tạo title dự phòng từ phần cuối URL."""
    slug = Path(urlparse(url).path).stem
    return slug.replace("-", " ").strip().title()


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dictionary chứa metadata và nội dung.
    """
    from crawl4ai import (
        AsyncWebCrawler,
        BrowserConfig,
        CacheMode,
        CrawlerRunConfig,
    )

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_config,
        )

    if not getattr(result, "success", False):
        error_message = getattr(
            result,
            "error_message",
            "Không xác định được lỗi",
        )
        raise RuntimeError(
            f"Crawl thất bại: {url}\nLỗi: {error_message}"
        )

    content_markdown = extract_markdown(result).strip()

    if len(content_markdown) < 500:
        raise ValueError(
            f"Nội dung crawl quá ngắn: {len(content_markdown)} ký tự"
        )

    metadata = getattr(result, "metadata", None) or {}
    title = metadata.get("title") or fallback_title(url)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_all() -> None:
    """Crawl toàn bộ URL và lưu mỗi bài vào một file JSON."""
    setup_directory()

    success_count = 0
    failed_urls = []

    for index, url in enumerate(ARTICLE_URLS, start=1):
        print(f"\n[{index}/{len(ARTICLE_URLS)}] Crawling:")
        print(url)

        try:
            article = await crawl_article(url)

            filename = f"article_{index:02d}.json"
            filepath = DATA_DIR / filename

            filepath.write_text(
                json.dumps(
                    article,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            file_size = filepath.stat().st_size
            content_length = len(article["content_markdown"])

            print(f"  Saved: {filepath}")
            print(f"  Title: {article['title']}")
            print(f"  Content: {content_length} characters")
            print(f"  File size: {file_size} bytes")

            success_count += 1

        except Exception as error:
            failed_urls.append(url)
            print(f"  ERROR: {error}")

    print("\n" + "=" * 60)
    print(f"Crawl thành công: {success_count}/{len(ARTICLE_URLS)}")

    if failed_urls:
        print("\nCác URL crawl thất bại:")

        for failed_url in failed_urls:
            print(f"- {failed_url}")

    if success_count < 5:
        raise RuntimeError(
            "Chưa crawl đủ 5 bài. Hãy kiểm tra lỗi hiển thị phía trên."
        )


if __name__ == "__main__":
    asyncio.run(crawl_all())