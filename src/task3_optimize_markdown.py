"""Deterministically optimize Task 3 Markdown for retrieval-augmented generation.

This module is intentionally separate from ``task3_convert_markdown.py`` so
the legal and news owners can keep working without editing the same file.

Inputs:
* legal: the MarkItDown extraction already present in ``standardized/legal``;
* news: ``content_markdown`` from each ``landing/news/*.json`` file;
* provenance: ``data/sources_manifest.json``.

The landing files are never modified. Output is written atomically and the
normalizer is idempotent, so it is safe to run again after a fresh extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
STANDARDIZED_DIR = DATA_DIR / "standardized"
MANIFEST_PATH = DATA_DIR / "sources_manifest.json"
MIN_CONTENT_CHARS = 200

CHAPTER_RE = re.compile(r"^(?:chương|chuong)\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^(?:điều|dieu)\s+\d+[a-zA-Z]?\s*[.:-]?", re.IGNORECASE)
# Deliberately case-sensitive: lowercase ``c)`` and ``d)`` are legal points,
# not Roman-numbered top-level sections.
ROMAN_RE = re.compile(r"^[IVXLCDM]{1,8}\s*[.)]\s+\S")
SUBSECTION_RE = re.compile(r"^\d+(?:\.\d+){1,4}\.?\s+\S")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LEADING_BOLD_RE = re.compile(r"^\*\*(.+?)\*\*\s*(.*)$")
STANDALONE_LINK_RE = re.compile(r"^\[[^\]]+\]\([^\n]+\)$")
EMPTY_IMAGE_RE = re.compile(r"!\[\s*\]\([^\n)]*\)")
LIST_RE = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+|[a-zđ][.)]\s+)", re.IGNORECASE)
LEAKAGE_RE = re.compile(
    r"(?:các?\s+câu\s+hỏi\s+hệ\s+thống\s+rag|"
    r"giá\s+trị\s+dữ\s+liệu\s+đối\s+với\s+hệ\s+thống\s+rag)",
    re.IGNORECASE,
)

BOILERPLATE_LINES = {
    "in ra",
    "đóng cửa sổ này",
    "print",
    "close this window",
}


@dataclass(frozen=True)
class OptimizedDocument:
    source: Path
    output: Path
    document_id: str
    title: str
    document_type: str
    chars: int
    headings: int
    written: bool


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def stable_slug(value: str) -> str:
    value = re.sub(r"(?<=[^\W\d_])(?=[A-ZĐ])", "-", value)
    value = value.replace("Đ", "D").replace("đ", "d")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower() or "document"


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("sources_manifest.json must contain a list of objects")

    ids = [str(item.get("document_id", "")) for item in payload]
    if any(not item for item in ids):
        raise ValueError("Every manifest entry must have document_id")
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate document_id values")
    return payload


def find_manifest_entry(source: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    relative = source.resolve().relative_to(REPO_DIR).as_posix().casefold()
    for entry in entries:
        landing_path = str(entry.get("landing_path", "")).replace("\\", "/").casefold()
        if landing_path == relative:
            return dict(entry)
    raise KeyError(f"No manifest entry for {relative}")


def _strip_existing_canonical_header(text: str) -> str:
    """Return the body if this optimizer already generated the document."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    prefix = normalized[:1500]
    if "**Document ID:**" not in prefix or "**Content Hash:**" not in prefix:
        return normalized
    parts = re.split(r"(?m)^---\s*$", normalized, maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else normalized


def clean_extracted_text(text: str) -> str:
    text = _strip_existing_canonical_header(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\f", "\n\n")
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    # Empty-alt image URLs contain no searchable evidence. Captions on their
    # neighbouring lines are preserved, while Task 2 landing JSON keeps raw
    # Markdown for provenance/audit.
    text = EMPTY_IMAGE_RE.sub("", text)
    text = "".join(
        char for char in text if char in "\n\t" or unicodedata.category(char) != "Cc"
    )

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line.casefold() not in BOILERPLATE_LINES]
    text = "\n".join(lines)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "-", text)
    return _join_soft_wrapped_lines(text.splitlines())


def _is_structure(line: str) -> bool:
    return bool(
        HEADING_RE.match(line)
        or CHAPTER_RE.match(line)
        or ARTICLE_RE.match(line)
        or ROMAN_RE.match(line)
        or SUBSECTION_RE.match(line)
        or LIST_RE.match(line)
        or line == "---"
        or _news_heading_parts(line) is not None
        or STANDALONE_LINK_RE.match(line)
    )


def _news_heading_parts(line: str) -> tuple[int, str, str] | None:
    """Recognize explicit bold section labels without guessing prose headings."""

    match = LEADING_BOLD_RE.match(line.strip())
    if not match:
        return None
    label = match.group(1).strip()
    remainder = match.group(2).strip()
    roman = re.match(r"^([IVXLCDM]{1,8})[.)]\s+\S", label)
    numbered = re.match(r"^(\d+(?:\.\d+)*)[.)]\s+\S", label)
    if roman:
        level = 2
    elif numbered:
        level = min(6, 2 + len(numbered.group(1).split(".")))
    elif not remainder and _uppercase_heading(label.rstrip(":;")):
        level = 2
    else:
        return None
    return level, label.rstrip(" :;"), remainder.lstrip(" :;")


def _should_join(previous: str, current: str, blank_between: bool) -> bool:
    if not previous or not current or _is_structure(current):
        return False
    # A numbered/bulleted paragraph may wrap across several PDF lines. Keep the
    # list marker, but join its continuation; true headings remain boundaries.
    previous_is_heading = bool(
        HEADING_RE.match(previous)
        or CHAPTER_RE.match(previous)
        or ARTICLE_RE.match(previous)
        or ROMAN_RE.match(previous)
        or SUBSECTION_RE.match(previous)
        or previous == "---"
    )
    if previous_is_heading:
        return False
    if previous.startswith("|") or current.startswith("|"):
        return False
    if previous.endswith((".", "?", "!", ":", ";")):
        return False
    if previous.endswith((",", "(", "/")) or current[:1].islower():
        return True
    return not blank_between and len(previous) >= 45


def _join_soft_wrapped_lines(lines: list[str]) -> str:
    output: list[str] = []
    pending = ""
    blank_between = False

    def flush() -> None:
        nonlocal pending, blank_between
        if pending:
            output.append(pending)
            pending = ""
        if blank_between and output and output[-1] != "":
            output.append("")
        blank_between = False

    for line in lines:
        line = line.strip()
        if not line:
            blank_between = True
            continue
        if not pending:
            pending = line
        elif _should_join(pending, line, blank_between):
            pending = f"{pending} {line}"
            blank_between = False
        else:
            flush()
            pending = line
    flush()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def _uppercase_heading(line: str) -> bool:
    letters = [char for char in line if char.isalpha()]
    return bool(letters) and len(line) <= 140 and all(not char.islower() for char in letters)


def strip_generated_legal_headings(text: str) -> str:
    """Remove optimizer-created H2-H6 markers before rebuilding hierarchy.

    MarkItDown's legal inputs in this lab contain plain extracted text rather
    than authored Markdown headings. Rebuilding makes the optimizer idempotent
    across rule upgrades and repairs headings produced by older versions.
    """

    return re.sub(r"(?m)^#{2,6}\s+", "", text)


def add_legal_headings(
    text: str,
    *,
    chapter_titles: dict[str, str] | None = None,
    discard_standalone_lines: list[str] | None = None,
) -> str:
    chapter_titles = {str(key).upper(): value for key, value in (chapter_titles or {}).items()}
    discarded = {
        re.sub(r"\s+", " ", value).strip().casefold()
        for value in (discard_standalone_lines or [])
    }
    lines = [
        line
        for line in text.splitlines()
        if re.sub(r"\s+", " ", line).strip().casefold() not in discarded
    ]
    output: list[str] = []
    index = 0
    in_article = False

    def consume_lowercase_continuation(heading: str, at: int) -> tuple[str, int]:
        next_index = at + 1
        while next_index < len(lines) and not lines[next_index].strip():
            next_index += 1
        if next_index >= len(lines):
            return heading, at
        candidate = lines[next_index].strip()
        if (
            (candidate[:1].islower() or heading.count("(") > heading.count(")"))
            and len(candidate) <= 180
            and not _is_structure(candidate)
        ):
            return f"{heading} {candidate}", next_index
        return heading, at

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            if output and output[-1] != "":
                output.append("")
            index += 1
            continue
        if HEADING_RE.match(line):
            output.append(line)
        elif CHAPTER_RE.match(line):
            in_article = False
            heading = line
            chapter_match = CHAPTER_RE.match(line)
            chapter_key = chapter_match.group(1).upper() if chapter_match else ""
            if chapter_key in chapter_titles:
                chapter_label = line[: chapter_match.end()].strip() if chapter_match else line
                heading = f"{chapter_label} — {chapter_titles[chapter_key]}"
            else:
                next_index = index + 1
                while next_index < len(lines) and not lines[next_index].strip():
                    next_index += 1
                if next_index < len(lines):
                    candidate = lines[next_index].strip()
                    if _uppercase_heading(candidate) and not _is_structure(candidate):
                        heading = f"{heading} — {candidate}"
                        index = next_index
            output.append(f"## {heading}")
        elif ROMAN_RE.match(line):
            in_article = False
            output.append(f"## {line}")
        elif ARTICLE_RE.match(line):
            in_article = True
            line, index = consume_lowercase_continuation(line, index)
            output.append(f"### {line}")
        elif SUBSECTION_RE.match(line):
            line, index = consume_lowercase_continuation(line, index)
            number = re.match(r"^(\d+(?:\.\d+){1,4})", line)
            depth = len(number.group(1).split(".")) if number else 2
            level = min(6, (depth + 2) if in_article else (depth + 1))
            output.append(f"{'#' * level} {line}")
        else:
            output.append(line)
        index += 1
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def add_news_headings(text: str) -> str:
    """Promote only explicit bold Roman/numbered labels to Markdown headings."""

    output: list[str] = []
    for line in text.splitlines():
        parts = _news_heading_parts(line)
        if parts is None:
            output.append(line)
            continue
        level, label, remainder = parts
        output.append(f"{'#' * level} {label}")
        if remainder:
            output.extend(["", remainder])
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def normalize_news_link_lists(text: str) -> str:
    """Turn runs of standalone links into real Markdown lists."""

    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        if not STANDALONE_LINK_RE.match(lines[index].strip()):
            output.append(lines[index])
            index += 1
            continue
        end = index
        while end < len(lines) and STANDALONE_LINK_RE.match(lines[end].strip()):
            end += 1
        run = lines[index:end]
        if len(run) >= 3:
            output.extend(f"- {line.strip()}" for line in run)
        else:
            output.extend(run)
        index = end
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def remove_legal_page_numbers(text: str) -> str:
    """Remove standalone PDF page counters without touching numbered clauses."""

    return re.sub(r"(?m)^\s*\d{1,3}\s*$\n?", "", text).strip()


def remove_rag_leakage_sections(text: str) -> str:
    output: list[str] = []
    skipped_level: int | None = None

    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if skipped_level is not None:
            if heading and len(heading.group(1)) <= skipped_level:
                skipped_level = None
            else:
                continue
        if heading and LEAKAGE_RE.search(heading.group(2)):
            skipped_level = len(heading.group(1))
            while output and not output[-1].strip():
                output.pop()
            continue
        output.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def remove_duplicate_title_h1(text: str, title: str) -> str:
    lines = text.splitlines()
    index = next((i for i, line in enumerate(lines) if line.strip()), None)
    if index is None:
        return text
    match = re.match(r"^#\s+(.+?)\s*$", lines[index])
    normalize = lambda value: re.sub(r"\W+", " ", value.casefold()).strip()
    if not match or normalize(match.group(1)) != normalize(title):
        return text
    del lines[index]
    while index < len(lines) and not lines[index].strip():
        del lines[index]
    return "\n".join(lines).strip()


def _value(value: Any, default: str = "N/A") -> str:
    if value is None or value == "":
        return default
    return re.sub(r"\s+", " ", str(value)).strip()


def build_rag_markdown(
    raw_text: str,
    metadata: dict[str, Any],
    *,
    source_name: str,
    source_hash: str,
) -> tuple[str, dict[str, Any]]:
    document_type = _value(metadata.get("source_type") or metadata.get("type"))
    body = clean_extracted_text(raw_text)
    if document_type == "legal":
        body = remove_legal_page_numbers(body)
        body = strip_generated_legal_headings(body)
        body = add_legal_headings(
            body,
            chapter_titles=metadata.get("chapter_titles"),
            discard_standalone_lines=metadata.get("discard_standalone_lines"),
        )
    elif document_type == "news":
        body = normalize_news_link_lists(body)
        body = add_news_headings(body)
    body = remove_rag_leakage_sections(body)

    resolved = dict(metadata)
    resolved["document_id"] = _value(
        resolved.get("document_id"), stable_slug(Path(source_name).stem)
    )
    resolved["title"] = _value(resolved.get("title"), Path(source_name).stem)
    resolved["url"] = _value(resolved.get("url"))
    resolved["source_type"] = document_type
    resolved["language"] = _value(resolved.get("language"), "vi")
    resolved["content_kind"] = _value(
        resolved.get("content_kind"),
        "source_extract" if document_type == "legal" else "derived_summary",
    )
    body = remove_duplicate_title_h1(body, resolved["title"])
    if len(body) < MIN_CONTENT_CHARS:
        raise ValueError(f"Content too short after cleaning: {len(body)} chars")

    resolved["source"] = source_name
    resolved["sha256"] = source_hash
    resolved["content_hash"] = sha256_text(body)

    header = [
        f"# {resolved['title']}",
        "",
        f"**Source:** {resolved['url']}",
        f"**Document ID:** {resolved['document_id']}",
        f"**Type:** {resolved['source_type']}",
        f"**Language:** {resolved['language']}",
    ]
    for label, key in (
        ("Organization", "organization"),
        ("Scope", "scope"),
        ("Published", "published_at"),
        ("Crawled", "date_crawled"),
        ("Effective", "effective_at"),
        ("Decision Number", "decision_number"),
    ):
        if resolved.get(key) not in (None, ""):
            header.append(f"**{label}:** {_value(resolved.get(key))}")
    header.extend(
        [
            f"**Content Kind:** {resolved['content_kind']}",
            f"**Status:** {_value(resolved.get('status'), 'unverified')}",
        ]
    )
    if resolved.get("crawl_backend"):
        for label, key in (
            ("Crawl Backend", "crawl_backend"),
            ("Crawl Backend Version", "crawl_backend_version"),
            ("Content Selector", "content_selector"),
            ("Content Selection", "content_selection"),
        ):
            if resolved.get(key) not in (None, ""):
                header.append(f"**{label}:** {_value(resolved.get(key))}")
        transforms = resolved.get("content_transforms")
        if isinstance(transforms, list) and transforms:
            header.append(f"**Content Transforms:** {', '.join(map(str, transforms))}")
        header.extend(
            [
                f"**Raw Content Hash:** {_value(resolved.get('raw_content_sha256'))}",
                f"**Landing Content Hash:** {_value(resolved.get('content_sha256'))}",
            ]
        )
    header.extend(
        [
            f"**Original File:** {source_name}",
            f"**Source SHA256:** {source_hash}",
            f"**Content Hash:** {resolved['content_hash']}",
        ]
    )
    if resolved["content_kind"] == "derived_summary":
        header.extend(
            [
                "",
                "> Citation note: Nội dung là bản diễn giải từ nguồn chính thức; "
                "đối chiếu URL nguồn khi trả lời dữ kiện nhạy cảm.",
            ]
        )
    document = "\n".join(header + ["", "---", "", body.rstrip(), ""])
    validate_rag_markdown(document, resolved)
    return document, resolved


def validate_rag_markdown(document: str, metadata: dict[str, Any]) -> None:
    required = (
        "# ",
        "**Source:** ",
        "**Document ID:** ",
        "**Type:** ",
        "**Content Hash:** ",
    )
    for marker in required:
        if marker not in document:
            raise ValueError(f"Missing canonical marker: {marker}")
    if metadata.get("url") in (None, "", "N/A"):
        raise ValueError("A real source URL is required")
    if "\ufffd" in document or "\f" in document:
        raise ValueError("Unicode/control extraction artifact remains")
    if LEAKAGE_RE.search(document):
        raise ValueError("RAG query leakage section remains")
    if len(re.findall(r"(?m)^#\s+", document)) != 1:
        raise ValueError("Canonical Markdown must have exactly one H1")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _optimize_one(
    source: Path,
    output: Path,
    raw_text: str,
    metadata: dict[str, Any],
    *,
    dry_run: bool,
) -> OptimizedDocument:
    document, resolved = build_rag_markdown(
        raw_text,
        metadata,
        source_name=source.name,
        source_hash=sha256_bytes(source.read_bytes()),
    )
    if not dry_run:
        _atomic_write(output, document)
    return OptimizedDocument(
        source=source,
        output=output,
        document_id=str(resolved["document_id"]),
        title=str(resolved["title"]),
        document_type=str(resolved["source_type"]),
        chars=len(document),
        headings=len(re.findall(r"(?m)^#{1,6}\s+", document)),
        written=not dry_run,
    )


def optimize_legal(
    entries: list[dict[str, Any]], *, dry_run: bool = False
) -> list[OptimizedDocument]:
    results: list[OptimizedDocument] = []
    for source in sorted((LANDING_DIR / "legal").glob("*")):
        if source.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        output = STANDARDIZED_DIR / "legal" / f"{source.stem}.md"
        if not output.exists():
            raise FileNotFoundError(
                f"Missing legal extraction {output}; run task3_convert_markdown.py first"
            )
        metadata = find_manifest_entry(source, entries)
        result = _optimize_one(
            source,
            output,
            output.read_text(encoding="utf-8"),
            metadata,
            dry_run=dry_run,
        )
        results.append(result)
        print(f"OK legal {result.output.name}: {result.headings} headings, {result.chars} chars")
    return results


def optimize_news(
    entries: list[dict[str, Any]], *, dry_run: bool = False
) -> list[OptimizedDocument]:
    results: list[OptimizedDocument] = []
    for source in sorted((LANDING_DIR / "news").glob("*.json")):
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        raw_text = payload.get("content_markdown")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(f"{source.name} has no content_markdown")
        metadata = find_manifest_entry(source, entries)
        # JSON is authoritative for crawl timestamp/title/url; manifest supplies
        # stable IDs, publication date, scope and content provenance.
        for key in (
            "title",
            "url",
            "published_at",
            "date_crawled",
            "language",
            "content_kind",
            "crawl_backend",
            "content_selector",
            "content_transforms",
            "raw_content_sha256",
            "content_sha256",
        ):
            if payload.get(key):
                metadata[key] = payload[key]
        output = STANDARDIZED_DIR / "news" / f"{source.stem}.md"
        result = _optimize_one(
            source,
            output,
            raw_text,
            metadata,
            dry_run=dry_run,
        )
        results.append(result)
        print(f"OK news {result.output.name}: {result.headings} headings, {result.chars} chars")
    return results


def hydrate_news_manifest(entries: list[dict[str, Any]]) -> bool:
    """Validate promoted landing JSON and refresh its provenance snapshot.

    Old lab summaries have no ``schema_version``/``content_kind`` and therefore
    retain their explicit ``derived_summary`` label. A newly crawled schema-v2
    payload is authoritative only after its stable ID, URL and hashes validate.
    """

    changed = False
    for source in sorted((LANDING_DIR / "news").glob("*.json")):
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        relative = source.resolve().relative_to(REPO_DIR).as_posix().casefold()
        entry = next(
            (
                item
                for item in entries
                if str(item.get("landing_path", "")).replace("\\", "/").casefold()
                == relative
            ),
            None,
        )
        if entry is None:
            raise KeyError(f"No manifest entry for {relative}")

        payload_id = str(payload.get("document_id", "")).strip()
        if payload_id and payload_id != str(entry.get("document_id", "")):
            raise ValueError(f"document_id mismatch in {source.name}")
        payload_url = str(payload.get("url", "")).strip()
        manifest_url = str(entry.get("url", "")).strip()
        if payload_url and payload_url.rstrip("/") != manifest_url.rstrip("/"):
            raise ValueError(f"Source URL mismatch in {source.name}")

        content = payload.get("content_markdown")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{source.name} has no content_markdown")
        expected_content_hash = str(payload.get("content_sha256", "")).strip()
        if expected_content_hash and sha256_text(content) != expected_content_hash:
            raise ValueError(f"content_sha256 mismatch in {source.name}")
        raw = payload.get("raw_markdown")
        expected_raw_hash = str(payload.get("raw_content_sha256", "")).strip()
        if expected_raw_hash and (
            not isinstance(raw, str) or sha256_text(raw) != expected_raw_hash
        ):
            raise ValueError(f"raw_content_sha256 mismatch in {source.name}")

        updates: dict[str, Any] = {"sha256": sha256_bytes(source.read_bytes())}
        for key in (
            "title",
            "url",
            "published_at",
            "language",
            "content_kind",
        ):
            if payload.get(key) not in (None, ""):
                updates[key] = payload[key]
        # Crawl-run details belong in landing JSON/report, not the stable source
        # registry. Remove values copied by older schema versions.
        for key in (
            "crawl_backend",
            "crawl_backend_version",
            "content_selector",
            "content_selection",
            "content_transforms",
            "raw_content_sha256",
            "content_sha256",
        ):
            if key in entry:
                del entry[key]
                changed = True
        for key, value in updates.items():
            if entry.get(key) != value:
                entry[key] = value
                changed = True
    return changed


def optimize_all(*, branch: str = "all", dry_run: bool = False, strict: bool = True) -> list[OptimizedDocument]:
    entries = load_manifest()
    results: list[OptimizedDocument] = []
    manifest_changed = False
    if branch in {"all", "news"}:
        manifest_changed = hydrate_news_manifest(entries)
    if branch in {"all", "legal"}:
        results.extend(optimize_legal(entries, dry_run=dry_run))
    if branch in {"all", "news"}:
        results.extend(optimize_news(entries, dry_run=dry_run))

    legal_count = sum(item.document_type == "legal" for item in results)
    news_count = sum(item.document_type == "news" for item in results)
    if strict and branch in {"all", "legal"} and legal_count < 3:
        raise RuntimeError(f"Expected at least 3 legal documents, found {legal_count}")
    if strict and branch in {"all", "news"} and news_count < 5:
        raise RuntimeError(f"Expected at least 5 news documents, found {news_count}")
    ids = [item.document_id for item in results]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate document_id in optimized corpus")
    if manifest_changed and not dry_run:
        _atomic_write(
            MANIFEST_PATH,
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        )
        print(f"Updated provenance manifest: {MANIFEST_PATH}")
    print(f"Done: legal={legal_count}, news={news_count}, dry_run={dry_run}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=("all", "legal", "news"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    optimize_all(branch=args.type, dry_run=args.dry_run, strict=not args.no_strict)


if __name__ == "__main__":
    main()
