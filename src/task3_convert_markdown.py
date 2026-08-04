"""Task 3: convert landing files into canonical Markdown for the RAG corpus.

Legal PDF/DOC/DOCX files are first extracted with Microsoft MarkItDown. News
JSON already contains source Markdown from Task 2, so it is normalized directly
by ``task3_optimize_markdown``. The final optimization pass adds canonical
metadata, provenance hashes and retrieval-friendly headings for both branches.

Examples (PowerShell):

    python src\task3_convert_markdown.py
    python src\task3_convert_markdown.py --type news
    python src\task3_convert_markdown.py --type all --reuse-legal-extraction
    python src\task3_convert_markdown.py --type all --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable


PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_DIR / "data" / "landing"
OUTPUT_DIR = PROJECT_DIR / "data" / "standardized"
LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}


def _load_optimizer() -> Callable[..., list[Any]]:
    """Support both ``python file.py`` and ``python -m src...`` invocation."""

    try:
        from src.task3_optimize_markdown import optimize_all
    except ModuleNotFoundError:
        from task3_optimize_markdown import optimize_all
    return optimize_all


def _load_markitdown() -> Any:
    try:
        from markitdown import MarkItDown
    except ImportError as error:
        raise RuntimeError(
            "MarkItDown with PDF support is required for legal extraction. "
            'Install it with: python -m pip install "markitdown[pdf]"'
        ) from error
    return MarkItDown


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _legal_sources() -> list[Path]:
    legal_dir = LANDING_DIR / "legal"
    if not legal_dir.exists():
        raise FileNotFoundError(f"Missing legal landing directory: {legal_dir}")
    sources = sorted(
        path
        for path in legal_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in LEGAL_EXTENSIONS
    )
    if not sources:
        raise FileNotFoundError(f"No PDF/DOC/DOCX files found in {legal_dir}")
    return sources


def extract_legal_docs() -> list[Path]:
    """Extract every legal landing file and fail the batch on any bad document."""

    MarkItDown = _load_markitdown()
    converter = MarkItDown()
    output_dir = OUTPUT_DIR / "legal"
    written: list[Path] = []
    failures: list[str] = []

    for source in _legal_sources():
        print(f"Extracting legal: {source.name}")
        try:
            result = converter.convert(str(source))
            content = str(getattr(result, "text_content", "") or "").strip()
            if len(content) < 200:
                raise ValueError(f"extracted content is too short ({len(content)} chars)")
            output = output_dir / f"{source.stem}.md"
            _atomic_write_text(output, content + "\n")
            written.append(output)
            print(f"  OK {output.relative_to(PROJECT_DIR)}")
        except Exception as error:
            failures.append(f"{source.name}: {type(error).__name__}: {error}")

    if failures:
        raise RuntimeError("Legal extraction failed:\n- " + "\n- ".join(failures))
    return written


def convert_legal_docs(*, reuse_extraction: bool = False, dry_run: bool = False) -> list[Any]:
    """Extract legal sources when needed, then create canonical legal Markdown."""

    if not reuse_extraction and not dry_run:
        extract_legal_docs()
    return _load_optimizer()(branch="legal", dry_run=dry_run)


def convert_news_articles(*, dry_run: bool = False) -> list[Any]:
    """Convert every Task 2 JSON payload to canonical news Markdown."""

    return _load_optimizer()(branch="news", dry_run=dry_run)


def convert_all(*, reuse_legal_extraction: bool = False, dry_run: bool = False) -> list[Any]:
    """Build both legal and news Markdown through one reproducible entrypoint."""

    if not reuse_legal_extraction and not dry_run:
        extract_legal_docs()
    return _load_optimizer()(branch="all", dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=("all", "legal", "news"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-legal-extraction",
        action="store_true",
        help="normalize existing legal Markdown without running MarkItDown again",
    )
    args = parser.parse_args()

    if args.type == "news":
        convert_news_articles(dry_run=args.dry_run)
    elif args.type == "legal":
        convert_legal_docs(
            reuse_extraction=args.reuse_legal_extraction,
            dry_run=args.dry_run,
        )
    else:
        convert_all(
            reuse_legal_extraction=args.reuse_legal_extraction,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
