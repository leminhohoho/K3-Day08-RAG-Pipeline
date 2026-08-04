"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

"""
Task 2 - Crawl bài viết từ website Trường ĐH KHXH&NV, ĐHQGHN.

Output:
    data/landing/news/article_01.json
    data/landing/news/article_02.json
    ...
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "landing" / "news"


ARTICLE_URLS = [
    "https://ussh.vnu.edu.vn/vi/gioi-thieu/lich-su-hinh-thanh-va-phat-trien/lich-su-phat-trien-19675.html",

    "https://ussh.vnu.edu.vn/vi/gioi-thieu/su-menh-va-muc-tieu-phat-trien/su-menh-va-muc-tieu-phat-trien-truong-dai-hoc-khoa-hoc-xa-hoi-va-nhan-van-19694.html",

    "https://ussh.vnu.edu.vn/vi/dao-tao/dao-tao-dai-hoc/dao-tao-dai-hoc-20685.html",

    "https://ussh.vnu.edu.vn/vi/news/sinh-vien/sinh-vien-vnu-ussh-trai-nghiem-hoc-tap-kien-thuc-ai-robot-va-giao-luu-van-hoa-truong-dh-khoa-hoc-ky-thuat-dien-tu-que-lam-tq-24282.html",

    "https://ussh.vnu.edu.vn/vi/news/sinh-vien/sinh-vien-truong-dh-khxh-nv-tu-tin-toa-sang-tai-trai-he-tri-tue-nhan-tao-thanh-nien-asean-va-trai-he-huu-nghi-trung-viet-duyen-duc-tai-nam-2026-24281.html",

    "https://ussh.vnu.edu.vn/vi/news/sinh-vien/sinh-vien-khoa-du-lich-hoc-tham-gia-chuong-trinh-he-du-lich-di-san-van-hoa-va-truyen-thong-2026-tai-dai-hoc-trung-son-trung-quoc-24278.html",

    "https://ussh.vnu.edu.vn/vi/news/hop-tac-phat-trien/hop-tac-dao-tao-nghien-cuu-khoa-hoc-va-phat-trien-nguon-nhan-luc-giua-truong-dh-khxh-nv-va-cong-ty-cp-dau-tu-va-cong-nghe-igo-24274.html",
]


def setup_directory() -> None:
    """Tạo thư mục lưu dữ liệu."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_markdown(markdown_result: Any) -> str:
    """
    Chuyển kết quả Markdown của Crawl4AI thành chuỗi.

    Crawl4AI phiên bản khác nhau có thể trả:
    - một string;
    - object có raw_markdown;
    - object có fit_markdown.
    """
    if markdown_result is None:
        return ""

    if isinstance(markdown_result, str):
        return markdown_result.strip()

    for attribute in (
        "fit_markdown",
        "raw_markdown",
        "markdown_with_citations",
    ):
        value = getattr(markdown_result, attribute, None)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return str(markdown_result).strip()


async def crawl_article(url: str, crawler: Any) -> dict:
    """Crawl một bài viết và trả về dictionary."""

    result = await crawler.arun(url=url)

    if not getattr(result, "success", True):
        error_message = getattr(
            result,
            "error_message",
            "Unknown crawl error",
        )

        raise RuntimeError(
            f"Không crawl được {url}: {error_message}"
        )

    content = extract_markdown(
        getattr(result, "markdown", None)
    )

    if len(content) < 500:
        raise ValueError(
            f"Nội dung quá ngắn: {len(content)} ký tự."
        )

    metadata = getattr(result, "metadata", {}) or {}

    if not isinstance(metadata, dict):
        metadata = {}

    title = (
        metadata.get("title")
        or metadata.get("og:title")
        or "Không xác định được tiêu đề"
    )

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
        "content_markdown": content,
    }


async def crawl_all() -> None:
    """Crawl toàn bộ URL và lưu thành các file JSON."""

    from crawl4ai import AsyncWebCrawler

    setup_directory()

    # Xóa file kết quả cũ
    for old_file in DATA_DIR.glob("article_*.json"):
        old_file.unlink()

    success_count = 0

    async with AsyncWebCrawler() as crawler:
        for index, url in enumerate(ARTICLE_URLS, start=1):
            print(
                f"\n[{index}/{len(ARTICLE_URLS)}] "
                f"Crawling: {url}"
            )

            try:
                article = await crawl_article(
                    url=url,
                    crawler=crawler,
                )

            except Exception as error:
                print(f"[ERROR] {error}")
                continue

            success_count += 1

            filename = f"article_{success_count:02d}.json"
            filepath = DATA_DIR / filename

            filepath.write_text(
                json.dumps(
                    article,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(f"[OK] Title: {article['title']}")
            print(
                f"[OK] Content: "
                f"{len(article['content_markdown'])} ký tự"
            )
            print(f"[OK] Saved: {filepath}")

    print("\n" + "=" * 60)
    print(
        f"Hoàn thành: "
        f"{success_count}/{len(ARTICLE_URLS)} bài."
    )

    if success_count < 5:
        raise RuntimeError(
            "Chưa đủ 5 bài. Kiểm tra các lỗi phía trên."
        )


if __name__ == "__main__":
    asyncio.run(crawl_all())