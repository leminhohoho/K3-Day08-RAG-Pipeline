"""
Task 3 — Chuẩn hóa dữ liệu về Markdown.

Input:
    data/landing/legal/     PDF, DOCX, DOC
    data/landing/news/      JSON, HTML, TXT, MD

Output:
    data/standardized/legal/*.md
    data/standardized/news/*.md
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from markitdown import MarkItDown


# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

LANDING_DIR = PROJECT_DIR / "data" / "landing"
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"

LEGAL_INPUT_DIR = LANDING_DIR / "legal"
NEWS_INPUT_DIR = LANDING_DIR / "news"

LEGAL_OUTPUT_DIR = STANDARDIZED_DIR / "legal"
NEWS_OUTPUT_DIR = STANDARDIZED_DIR / "news"


# Các định dạng được xử lý
LEGAL_EXTENSIONS = {".pdf", ".doc", ".docx"}
NEWS_EXTENSIONS = {".json", ".html", ".htm", ".txt", ".md"}


# ============================================================
# HÀM HỖ TRỢ
# ============================================================

def setup_directories() -> None:
    """Tạo các thư mục output nếu chưa tồn tại."""
    LEGAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NEWS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    """
    Chuyển tên file về dạng đơn giản, không dấu và không khoảng trắng.

    Ví dụ:
        "Quy chế sinh viên nội trú"
        -> "quy-che-sinh-vien-noi-tru"
    """
    text = text.replace("đ", "d").replace("Đ", "D")

    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")

    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    ascii_text = ascii_text.strip("-")

    return ascii_text or "document"


def yaml_value(value: Any) -> str:
    """
    Chuyển một giá trị thành chuỗi an toàn để đặt trong YAML front matter.
    """
    if value is None:
        value = ""

    return json.dumps(str(value), ensure_ascii=False)


def build_markdown_document(
    *,
    title: str,
    source: str,
    source_path: str,
    document_type: str,
    content: str,
    url: str = "",
    date_crawled: str = "",
) -> str:
    """
    Ghép metadata và nội dung thành một tài liệu Markdown hoàn chỉnh.
    """
    clean_content = content.strip()

    metadata = [
        "---",
        f"title: {yaml_value(title)}",
        f"source: {yaml_value(source)}",
        f"source_path: {yaml_value(source_path)}",
        f"type: {yaml_value(document_type)}",
    ]

    if url:
        metadata.append(f"url: {yaml_value(url)}")

    if date_crawled:
        metadata.append(f"date_crawled: {yaml_value(date_crawled)}")

    metadata.extend(
        [
            "---",
            "",
            f"# {title}",
            "",
            clean_content,
            "",
        ]
    )

    return "\n".join(metadata)


def extract_markitdown_text(result: Any) -> str:
    """
    Đọc nội dung từ kết quả MarkItDown.

    Hàm hỗ trợ một số phiên bản MarkItDown khác nhau.
    """
    text_content = getattr(result, "text_content", None)

    if isinstance(text_content, str) and text_content.strip():
        return text_content.strip()

    markdown = getattr(result, "markdown", None)

    if isinstance(markdown, str) and markdown.strip():
        return markdown.strip()

    raw_markdown = getattr(markdown, "raw_markdown", None)

    if isinstance(raw_markdown, str) and raw_markdown.strip():
        return raw_markdown.strip()

    return ""


def convert_with_markitdown(
    converter: MarkItDown,
    input_path: Path,
) -> str:
    """
    Chuyển một file local bằng MarkItDown.
    """
    convert_local = getattr(converter, "convert_local", None)

    if callable(convert_local):
        result = convert_local(str(input_path))
    else:
        # Tương thích với phiên bản MarkItDown cũ
        result = converter.convert(str(input_path))

    content = extract_markitdown_text(result)

    if len(content) < 200:
        raise ValueError(
            f"Nội dung sau khi convert quá ngắn: "
            f"{len(content)} ký tự"
        )

    return content


# ============================================================
# CONVERT LEGAL: PDF/DOCX -> MARKDOWN
# ============================================================

def convert_legal_files(converter: MarkItDown) -> int:
    """
    Chuyển tất cả tài liệu legal sang Markdown.

    Returns:
        Số file convert thành công.
    """
    if not LEGAL_INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục: {LEGAL_INPUT_DIR}"
        )

    input_files = sorted(
        file
        for file in LEGAL_INPUT_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in LEGAL_EXTENSIONS
    )

    if not input_files:
        raise FileNotFoundError(
            "Không tìm thấy PDF/DOCX trong data/landing/legal/"
        )

    success_count = 0

    print("\n" + "=" * 60)
    print("CONVERT LEGAL DOCUMENTS")
    print("=" * 60)

    for index, input_path in enumerate(input_files, start=1):
        print(
            f"\n[{index}/{len(input_files)}] "
            f"{input_path.name}"
        )

        content = convert_with_markitdown(
            converter=converter,
            input_path=input_path,
        )

        title = input_path.stem
        output_name = f"{slugify(input_path.stem)}.md"
        output_path = LEGAL_OUTPUT_DIR / output_name

        relative_source_path = input_path.relative_to(PROJECT_DIR)

        markdown_document = build_markdown_document(
            title=title,
            source=input_path.name,
            source_path=str(relative_source_path),
            document_type="legal",
            content=content,
        )

        output_path.write_text(
            markdown_document,
            encoding="utf-8",
        )

        print(f"  Saved: {output_path}")
        print(f"  Characters: {len(markdown_document)}")

        success_count += 1

    return success_count


# ============================================================
# CONVERT NEWS JSON -> MARKDOWN
# ============================================================

def get_json_content(data: dict[str, Any]) -> str:
    """
    Tìm trường chứa nội dung chính trong file JSON.
    """
    possible_fields = [
        "content_markdown",
        "markdown",
        "content",
        "text",
        "body",
    ]

    for field in possible_fields:
        value = data.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def convert_json_news(input_path: Path) -> tuple[str, str, str, str]:
    """
    Đọc một file news JSON.

    Returns:
        title, content, url, date_crawled
    """
    data = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"{input_path.name} không chứa JSON object"
        )

    title = str(
        data.get("title")
        or input_path.stem
    ).strip()

    content = get_json_content(data)

    if len(content) < 200:
        raise ValueError(
            f"Nội dung trong {input_path.name} quá ngắn: "
            f"{len(content)} ký tự"
        )

    url = str(data.get("url") or "").strip()
    date_crawled = str(
        data.get("date_crawled")
        or data.get("crawled_at")
        or ""
    ).strip()

    return title, content, url, date_crawled


def convert_news_files(converter: MarkItDown) -> int:
    """
    Chuyển toàn bộ dữ liệu news sang Markdown.

    JSON được xử lý trực tiếp để giữ metadata.
    HTML/TXT/MD được xử lý bằng MarkItDown.
    """
    if not NEWS_INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục: {NEWS_INPUT_DIR}"
        )

    input_files = sorted(
        file
        for file in NEWS_INPUT_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in NEWS_EXTENSIONS
    )

    if not input_files:
        raise FileNotFoundError(
            "Không tìm thấy bài viết trong data/landing/news/"
        )

    success_count = 0

    print("\n" + "=" * 60)
    print("CONVERT NEWS ARTICLES")
    print("=" * 60)

    for index, input_path in enumerate(input_files, start=1):
        print(
            f"\n[{index}/{len(input_files)}] "
            f"{input_path.name}"
        )

        if input_path.suffix.lower() == ".json":
            title, content, url, date_crawled = (
                convert_json_news(input_path)
            )
        else:
            title = input_path.stem
            content = convert_with_markitdown(
                converter=converter,
                input_path=input_path,
            )
            url = ""
            date_crawled = ""

        output_name = f"{slugify(input_path.stem)}.md"
        output_path = NEWS_OUTPUT_DIR / output_name

        relative_source_path = input_path.relative_to(PROJECT_DIR)

        markdown_document = build_markdown_document(
            title=title,
            source=input_path.name,
            source_path=str(relative_source_path),
            document_type="news",
            content=content,
            url=url,
            date_crawled=date_crawled,
        )

        output_path.write_text(
            markdown_document,
            encoding="utf-8",
        )

        print(f"  Saved: {output_path}")
        print(f"  Title: {title}")
        print(f"  Characters: {len(markdown_document)}")

        success_count += 1

    return success_count


# ============================================================
# CHẠY TOÀN BỘ TASK 3
# ============================================================

def main() -> None:
    """Chạy toàn bộ quá trình chuẩn hóa dữ liệu."""
    setup_directories()

    converter = MarkItDown()

    errors: list[str] = []

    legal_count = 0
    news_count = 0

    try:
        legal_count = convert_legal_files(converter)
    except Exception as error:
        errors.append(f"Legal: {error}")
        print(f"\nERROR LEGAL: {error}")

    try:
        news_count = convert_news_files(converter)
    except Exception as error:
        errors.append(f"News: {error}")
        print(f"\nERROR NEWS: {error}")

    print("\n" + "=" * 60)
    print("TASK 3 SUMMARY")
    print("=" * 60)
    print(f"Legal converted: {legal_count}")
    print(f"News converted:  {news_count}")
    print(f"Total converted: {legal_count + news_count}")

    if errors:
        print("\nCác lỗi gặp phải:")

        for error in errors:
            print(f"- {error}")

        raise RuntimeError(
            "Task 3 chưa hoàn thành đầy đủ. "
            "Xem lỗi phía trên."
        )

    print("\nTask 3 completed successfully.")


if __name__ == "__main__":
    main()