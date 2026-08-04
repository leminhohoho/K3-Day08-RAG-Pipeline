"""Focused tests for the deterministic Task 3 RAG optimization layer."""

import json

from src.task3_optimize_markdown import (
    add_legal_headings,
    add_news_headings,
    build_rag_markdown,
    clean_extracted_text,
    hydrate_news_manifest,
    normalize_news_link_lists,
    remove_legal_page_numbers,
    remove_rag_leakage_sections,
    stable_slug,
    strip_generated_legal_headings,
)


def test_news_headings_promote_explicit_sections_and_preserve_inline_bold():
    raw = (
        "**I. THÔNG TIN CHUNG**\n"
        "**1. Tên cơ sở đào tạo:** Trường Đại học mẫu.\n"
        "Ngày **10/10/1945** là một mốc lịch sử."
    )

    converted = add_news_headings(clean_extracted_text(raw))

    assert "## I. THÔNG TIN CHUNG" in converted
    assert "### 1. Tên cơ sở đào tạo\n\nTrường Đại học mẫu." in converted
    assert "Ngày **10/10/1945** là một mốc lịch sử." in converted


def test_empty_images_are_removed_but_captions_are_preserved():
    raw = "Nội dung chính.\n![](https://example.com/photo.jpg)\n_Chú thích ảnh_"

    cleaned = clean_extracted_text(raw)

    assert "![](" not in cleaned
    assert "_Chú thích ảnh_" in cleaned


def test_standalone_program_links_become_a_markdown_list():
    raw = "\n".join(
        [
            "[Ngành Báo chí](https://example.com/1)",
            "[Ngành Lịch sử](https://example.com/2)",
            "[Ngành Văn học](https://example.com/3)",
        ]
    )

    converted = normalize_news_link_lists(clean_extracted_text(raw))

    assert converted.count("\n- ") == 2
    assert converted.startswith("- [Ngành Báo chí]")


def test_legal_page_numbers_are_removed_without_losing_numbered_clauses():
    raw = "1. Khoản thứ nhất\n\n2\n\n2. Khoản thứ hai"

    cleaned = remove_legal_page_numbers(raw)

    assert "\n2\n" not in cleaned
    assert "1. Khoản thứ nhất" in cleaned
    assert "2. Khoản thứ hai" in cleaned


def test_stable_slug_supports_vietnamese():
    assert stable_slug("Nhân Văn - Học Bổng 2025") == "nhan-van-hoc-bong-2025"
    assert stable_slug("NhanVan_HocBong") == "nhan-van-hoc-bong"


def test_cleaning_removes_pdf_artifacts_without_losing_numbers():
    raw = (
        "Điều 1. Mức thu\n"
        "Mức học phí là 1.690.000đ/tháng/SV,\n"
        "áp dụng từ 05/11/2025.\f\n"
        "In ra\nĐóng cửa sổ này\n"
    )
    cleaned = clean_extracted_text(raw)

    assert "\f" not in cleaned
    assert "In ra" not in cleaned
    assert "Đóng cửa sổ này" not in cleaned
    assert "1.690.000đ/tháng/SV" in cleaned
    assert "05/11/2025" in cleaned


def test_cleaning_joins_wrapped_numbered_paragraphs():
    raw = (
        "1. Quy chế này áp dụng cho toàn bộ sinh viên đang học tập tại trường\n"
        "và các đơn vị có liên quan trong quá trình thực hiện.\n\n"
        "2. Khoản tiếp theo được giữ thành mục riêng."
    )
    cleaned = clean_extracted_text(raw)

    assert "tại trường và các đơn vị" in cleaned
    assert "\n\n2. Khoản tiếp theo" in cleaned


def test_rag_markdown_has_provenance_and_legal_headings():
    raw = """QUY CHẾ
CÔNG TÁC SINH VIÊN NỘI TRÚ

Chương I
QUY ĐỊNH CHUNG

Điều 1. Phạm vi áp dụng
Văn bản này áp dụng cho sinh viên nội trú tại ký túc xá. Nội dung chi tiết được
giữ nguyên để làm bằng chứng truy xuất. Sinh viên phải tuân thủ đầy đủ quy định.

Điều 2. Quyền của sinh viên
Sinh viên được sử dụng các dịch vụ hợp pháp và được cung cấp thông tin cần thiết.
"""
    document, metadata = build_rag_markdown(
        raw,
        {
            "document_id": "vnu-dormitory-regulation",
            "title": "QUY CHẾ CÔNG TÁC SINH VIÊN NỘI TRÚ",
            "url": "https://example.edu.vn/dormitory",
            "source_type": "legal",
            "language": "vi",
            "status": "official",
        },
        source_name="noi-tru.pdf",
        source_hash="a" * 64,
    )

    assert document.startswith("# QUY CHẾ CÔNG TÁC SINH VIÊN NỘI TRÚ")
    assert "**Source:** https://example.edu.vn/dormitory" in document
    assert "**Document ID:** vnu-dormitory-regulation" in document
    assert "## Chương I — QUY ĐỊNH CHUNG" in document
    assert "### Điều 1. Phạm vi áp dụng" in document
    assert metadata["content_hash"]


def test_news_removes_duplicate_h1_and_rag_prompt_section():
    raw = """# Bài viết mẫu

## Nội dung chính

Đây là nội dung nguồn đủ dài để kiểm tra bộ chuẩn hóa. Thông tin này được giữ lại
và không bị mô hình ngôn ngữ viết lại. Đoạn văn bổ sung giúp vượt ngưỡng tối thiểu
để phép kiểm thử phản ánh đúng một tài liệu tin tức thực tế trong pipeline RAG.

## Các câu hỏi hệ thống RAG có thể trả lời

- Câu hỏi bị loại khỏi tri thức nguồn.
"""
    document, _ = build_rag_markdown(
        raw,
        {
            "document_id": "sample-news",
            "title": "Bài viết mẫu",
            "url": "https://example.edu.vn/news",
            "source_type": "news",
            "language": "vi",
            "content_kind": "derived_summary",
        },
        source_name="article.json",
        source_hash="b" * 64,
    )

    assert document.count("# Bài viết mẫu") == 1
    assert "Câu hỏi hệ thống RAG" not in document
    assert "**Content Kind:** derived_summary" in document
    assert "Citation note:" in document


def test_leakage_removal_stops_at_equal_or_higher_heading():
    text = """## Nội dung
Giữ lại.

### Giá trị dữ liệu đối với hệ thống RAG
Bỏ đoạn này.

## Phần tiếp theo
Giữ phần này.
"""
    cleaned = remove_rag_leakage_sections(text)
    assert "Bỏ đoạn này" not in cleaned
    assert "Giữ phần này" in cleaned


def test_lowercase_legal_points_are_not_roman_headings():
    raw = """I. Quy định chung

c) Điểm c là nội dung của một khoản, không phải một chương mới.

d) Điểm d tiếp tục nội dung của cùng khoản đó.
"""
    rebuilt = add_legal_headings(strip_generated_legal_headings(raw))

    assert "## I. Quy định chung" in rebuilt
    assert "## c)" not in rebuilt
    assert "## d)" not in rebuilt


def test_old_generated_headings_are_rebuilt_with_new_rules():
    old_body = """## I. Mục chính

## c) Đây từng bị nhận diện sai

### Điều 1. Nội dung
"""
    rebuilt = add_legal_headings(strip_generated_legal_headings(old_body))

    assert "## I. Mục chính" in rebuilt
    assert "## c)" not in rebuilt
    assert "### Điều 1. Nội dung" in rebuilt


def test_wrapped_legal_heading_and_curated_chapter_title():
    raw = """TRÁCH NHIỆM CỦA CÁC ĐƠN VỊ TRONG CÔNG TÁC

Chương II

Điều 5. Trung tâm Hỗ trợ sinh viên và đơn vị quản lý Ký túc
xá

1. Nội dung của điều.
"""
    rebuilt = add_legal_headings(
        raw,
        chapter_titles={"II": "TRÁCH NHIỆM CỦA CÁC ĐƠN VỊ"},
        discard_standalone_lines=["TRÁCH NHIỆM CỦA CÁC ĐƠN VỊ TRONG CÔNG TÁC"],
    )

    assert "## Chương II — TRÁCH NHIỆM CỦA CÁC ĐƠN VỊ" in rebuilt
    assert "### Điều 5. Trung tâm Hỗ trợ sinh viên và đơn vị quản lý Ký túc xá" in rebuilt


def test_decimal_headings_follow_contextual_hierarchy():
    notice = add_legal_headings("""I. Mức thu
1.1. Sinh viên Việt Nam
1.1.2. Khóa tuyển sinh
1.1.2.1. Nhóm ngành được kiểm định
""")
    regulation = add_legal_headings("""Điều 5. Học bổng
1.1. Đối tượng và tiêu chí
""")

    assert "### 1.1. Sinh viên Việt Nam" in notice
    assert "#### 1.1.2. Khóa tuyển sinh" in notice
    assert "##### 1.1.2.1. Nhóm ngành được kiểm định" in notice
    assert "#### 1.1. Đối tượng và tiêu chí" in regulation


def test_source_extract_news_has_crawl_provenance_without_summary_warning():
    raw = """# Bài nguồn thật

## Nội dung

Nội dung được trích xuất trực tiếp từ website chính thức và đủ dài để kiểm tra.
Thông tin được giữ nguyên, không sử dụng mô hình ngôn ngữ để viết lại hoặc tóm tắt.
Đoạn bổ sung bảo đảm tài liệu vượt qua ngưỡng độ dài tối thiểu của bộ chuẩn hóa.
"""
    document, _ = build_rag_markdown(
        raw,
        {
            "document_id": "official-news",
            "title": "Bài nguồn thật",
            "url": "https://ussh.vnu.edu.vn/vi/news/official.html",
            "source_type": "news",
            "language": "vi",
            "content_kind": "source_extract",
            "crawl_backend": "crawl4ai",
            "content_selector": "#news-bodyhtml",
            "raw_content_sha256": "c" * 64,
            "content_sha256": "d" * 64,
        },
        source_name="article.json",
        source_hash="e" * 64,
    )

    assert "**Content Kind:** source_extract" in document
    assert "**Crawl Backend:** crawl4ai" in document
    assert "**Content Selector:** #news-bodyhtml" in document
    assert "**Crawl Backend Version:**" not in document
    assert "**Content Selection:**" not in document
    assert "**Effective:** N/A" not in document
    assert "**Decision Number:** N/A" not in document
    assert "Citation note:" not in document


def test_hydrate_news_manifest_promotes_valid_schema_v2_metadata(tmp_path, monkeypatch):
    import src.task3_optimize_markdown as optimizer

    landing = tmp_path / "data" / "landing" / "news"
    landing.mkdir(parents=True)
    content = "Nội dung nguồn thật đủ dài để kiểm tra. " * 20
    raw = "Raw: " + content
    payload = {
        "schema_version": 2,
        "document_id": "official-news",
        "url": "https://ussh.vnu.edu.vn/vi/news/official.html",
        "title": "Bài nguồn thật",
        "content_kind": "source_extract",
        "crawl_backend": "crawl4ai",
        "content_selector": "#news-bodyhtml",
        "content_transforms": [],
        "raw_markdown": raw,
        "content_markdown": content,
        "raw_content_sha256": optimizer.sha256_text(raw),
        "content_sha256": optimizer.sha256_text(content),
    }
    source = landing / "article_01.json"
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    entries = [
        {
            "document_id": "official-news",
            "url": payload["url"],
            "source_type": "news",
            "content_kind": "derived_summary",
            "landing_path": "data/landing/news/article_01.json",
        }
    ]
    monkeypatch.setattr(optimizer, "REPO_DIR", tmp_path)
    monkeypatch.setattr(optimizer, "LANDING_DIR", tmp_path / "data" / "landing")

    changed = hydrate_news_manifest(entries)

    assert changed is True
    assert entries[0]["content_kind"] == "source_extract"
    assert "crawl_backend" not in entries[0]
    assert entries[0]["sha256"] == optimizer.sha256_bytes(source.read_bytes())
