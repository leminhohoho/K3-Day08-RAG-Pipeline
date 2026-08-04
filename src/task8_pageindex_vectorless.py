"""Task 8 — real PageIndex vectorless retrieval with a safe public fallback.

The query path never uploads documents. Run the explicit ``--upload`` setup
command first; uploaded document IDs and checksums are stored in the generated
``pageindex_doc_ids.json`` registry.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
PAGEINDEX_API_BASE = os.getenv(
    "PAGEINDEX_API_BASE", "https://api.pageindex.ai"
).rstrip("/")
PAGEINDEX_REGISTRY_PATH = PROJECT_ROOT / "pageindex_doc_ids.json"
SOURCES_MANIFEST_PATH = PROJECT_ROOT / "data" / "sources_manifest.json"
LANDING_DIR = PROJECT_ROOT / "data" / "landing"
PAGEINDEX_PDF_CACHE_DIR = PROJECT_ROOT / "pageindex_pdfs"


def _env_float(name: str, default: float, minimum: float) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


PAGEINDEX_REQUEST_TIMEOUT_SECONDS = _env_float(
    "PAGEINDEX_REQUEST_TIMEOUT_SECONDS", 30.0, 1.0
)
PAGEINDEX_PROCESSING_TIMEOUT_SECONDS = _env_float(
    "PAGEINDEX_PROCESSING_TIMEOUT_SECONDS", 600.0, 1.0
)
PAGEINDEX_RETRIEVAL_TIMEOUT_SECONDS = _env_float(
    "PAGEINDEX_RETRIEVAL_TIMEOUT_SECONDS", 90.0, 1.0
)
PAGEINDEX_POLL_INITIAL_SECONDS = _env_float(
    "PAGEINDEX_POLL_INITIAL_SECONDS", 1.0, 0.05
)
PAGEINDEX_POLL_MAX_SECONDS = _env_float(
    "PAGEINDEX_POLL_MAX_SECONDS", 8.0, PAGEINDEX_POLL_INITIAL_SECONDS
)
PAGEINDEX_MAX_DOCUMENTS_PER_QUERY = _env_int(
    "PAGEINDEX_MAX_DOCUMENTS_PER_QUERY", 3, 1
)
PAGEINDEX_QUERY_CACHE_TTL_SECONDS = _env_float(
    "PAGEINDEX_QUERY_CACHE_TTL_SECONDS", 3600.0, 0.0
)
PAGEINDEX_MAX_EVIDENCE_CHARS = _env_int(
    "PAGEINDEX_MAX_EVIDENCE_CHARS", 12_000, 500
)

BACKEND_NAME = "legacy_retrieval"
RETRIEVAL_METHOD = "pageindex_legacy_retrieval"
REGISTRY_SCHEMA_VERSION = 1
PARSER_VERSION = "1"
TERMINAL_FAILURES = {"failed", "error", "cancelled", "canceled"}


class PageIndexIntegrationError(RuntimeError):
    """Sanitized integration error safe to expose through diagnostics."""

    def __init__(self, kind: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


class RegistryError(RuntimeError):
    """Raised when an existing registry cannot be trusted."""


class PageIndexBackend(Protocol):
    def submit_document(self, path: Path) -> str: ...

    def get_document_status(self, doc_id: str) -> dict[str, Any]: ...

    def submit_retrieval(self, doc_id: str, query: str, thinking: bool) -> str: ...

    def get_retrieval(self, retrieval_id: str) -> dict[str, Any]: ...


class CloudLegacyRetrievalBackend:
    """Official PageIndex document and legacy retrieval REST adapter."""

    def __init__(
        self,
        api_key: str,
        *,
        api_base: str = PAGEINDEX_API_BASE,
        timeout: float = PAGEINDEX_REQUEST_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise PageIndexIntegrationError("not_configured", "PAGEINDEX_API_KEY is missing")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def _headers(self) -> dict[str, str]:
        return {"api_key": self.api_key}

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = self.session.request(
                method, f"{self.api_base}{path}", headers=self._headers, **kwargs
            )
        except requests.Timeout as error:
            raise PageIndexIntegrationError("timeout", "PageIndex request timed out") from error
        except requests.RequestException as error:
            raise PageIndexIntegrationError(
                "network_error", "PageIndex network request failed"
            ) from error

        if response.status_code not in {200, 201, 202}:
            if response.status_code in {401, 403}:
                kind = "auth_error"
            elif response.status_code == 429:
                kind = "rate_limited"
            elif response.status_code >= 500:
                kind = "remote_error"
            else:
                kind = "invalid_request"
            raise PageIndexIntegrationError(
                kind,
                f"PageIndex returned HTTP {response.status_code}",
                response.status_code,
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise PageIndexIntegrationError(
                "schema_error", "PageIndex returned non-JSON data"
            ) from error
        if not isinstance(payload, dict):
            raise PageIndexIntegrationError(
                "schema_error", "PageIndex response must be a JSON object"
            )
        return payload

    def submit_document(self, path: Path) -> str:
        with path.open("rb") as stream:
            payload = self._request_json(
                "POST",
                "/doc/",
                files={"file": (path.name, stream, "application/pdf")},
                data={"if_retrieval": "true"},
            )
        doc_id = payload.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise PageIndexIntegrationError(
                "schema_error", "Document submission did not return doc_id"
            )
        return doc_id.strip()

    def get_document_status(self, doc_id: str) -> dict[str, Any]:
        return self._request_json(
            "GET", f"/doc/{doc_id}/", params={"type": "tree", "summary": "false"}
        )

    def submit_retrieval(self, doc_id: str, query: str, thinking: bool) -> str:
        payload = self._request_json(
            "POST",
            "/retrieval/",
            json={"doc_id": doc_id, "query": query, "thinking": thinking},
        )
        retrieval_id = payload.get("retrieval_id")
        if not isinstance(retrieval_id, str) or not retrieval_id.strip():
            raise PageIndexIntegrationError(
                "schema_error", "Retrieval submission did not return retrieval_id"
            )
        return retrieval_id.strip()

    def get_retrieval(self, retrieval_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/retrieval/{retrieval_id}/")


_SERVICE_STATUS: dict[str, Any] = {
    "last_error_type": None,
    "last_error_message": None,
    "last_success_at": None,
    "last_route": "not_run",
}
_QUERY_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sdk_version() -> str | None:
    try:
        return importlib.metadata.version("pageindex")
    except importlib.metadata.PackageNotFoundError:
        return None


def _record_error(error: BaseException, route: str) -> None:
    kind = getattr(error, "kind", type(error).__name__)
    message = str(error).replace(PAGEINDEX_API_KEY, "***") if PAGEINDEX_API_KEY else str(error)
    _SERVICE_STATUS.update(
        {
            "last_error_type": str(kind),
            "last_error_message": message[:240],
            "last_route": route,
        }
    )


def _record_success(route: str) -> None:
    _SERVICE_STATUS.update(
        {
            "last_error_type": None,
            "last_error_message": None,
            "last_success_at": _now(),
            "last_route": route,
        }
    )


def _empty_registry() -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "sdk_version": _sdk_version(),
        "updated_at": _now(),
        "documents": {},
    }


def _load_registry(*, strict: bool = False) -> dict[str, Any]:
    if not PAGEINDEX_REGISTRY_PATH.exists():
        return _empty_registry()
    try:
        payload = json.loads(PAGEINDEX_REGISTRY_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        if strict:
            raise RegistryError(
                "Existing PageIndex registry is unreadable; refusing automatic upload"
            ) from error
        _record_error(error, "registry")
        return _empty_registry()
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or not isinstance(payload.get("documents"), dict)
    ):
        error = RegistryError("Unsupported or invalid PageIndex registry schema")
        if strict:
            raise error
        _record_error(error, "registry")
        return _empty_registry()
    return payload


def _write_registry(registry: dict[str, Any]) -> None:
    registry["schema_version"] = REGISTRY_SCHEMA_VERSION
    registry["sdk_version"] = _sdk_version()
    registry["updated_at"] = _now()
    PAGEINDEX_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PAGEINDEX_REGISTRY_PATH.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(PAGEINDEX_REGISTRY_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _unicode_font_path() -> Path:
    configured = os.getenv("PAGEINDEX_UNICODE_FONT", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No Vietnamese Unicode font found; set PAGEINDEX_UNICODE_FONT"
    )


def _build_markdown_pdf(item: dict[str, Any], markdown_path: Path) -> Path:
    """Create/reuse a Unicode PDF suitable for PageIndex document processing."""

    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    markdown_checksum = _sha256(markdown_path).removeprefix("sha256:")
    output = PAGEINDEX_PDF_CACHE_DIR / (
        f"{item['document_id']}-{markdown_checksum[:12]}.pdf"
    )
    if output.is_file() and output.stat().st_size >= 1024:
        return output

    PAGEINDEX_PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    font_path = _unicode_font_path()
    markdown = markdown_path.read_text(encoding="utf-8-sig")
    pdf = FPDF(format="A4")
    pdf.set_margins(16, 16, 16)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_title(str(item.get("title") or markdown_path.stem))
    pdf.set_author(str(item.get("organization") or "VNU-USSH"))
    pdf.add_font("PageIndexUnicode", fname=str(font_path))
    pdf.add_page()
    pdf.set_font("PageIndexUnicode", size=16)
    pdf.multi_cell(
        0,
        8,
        text=str(item.get("title") or markdown_path.stem),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(2)
    pdf.set_font("PageIndexUnicode", size=9)
    pdf.multi_cell(
        0,
        5,
        text=f"Document ID: {item['document_id']}",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.multi_cell(
        0,
        5,
        text=f"Source URL: {item.get('url') or 'N/A'}",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(3)

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            pdf.ln(2 if level <= 2 else 1)
            pdf.set_font("PageIndexUnicode", size=max(11, 16 - level))
            pdf.multi_cell(
                0,
                7,
                text=heading.group(2).strip(),
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            pdf.set_font("PageIndexUnicode", size=10)
        elif not line.strip():
            pdf.ln(2)
        else:
            pdf.set_font("PageIndexUnicode", size=10)
            pdf.multi_cell(
                0,
                5.5,
                text=line,
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )

    temporary = output.with_suffix(".pdf.tmp")
    try:
        pdf.output(str(temporary))
        if temporary.stat().st_size < 1024:
            raise ValueError(f"Generated PageIndex PDF is unexpectedly small: {output.name}")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _eligible_documents() -> list[dict[str, Any]]:
    payload = json.loads(SOURCES_MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("sources_manifest.json must contain a list")
    landing_root = LANDING_DIR.resolve()
    documents: list[dict[str, Any]] = []
    for item in payload:
        landing_path = item.get("landing_path")
        standardized_path = item.get("standardized_path")
        if isinstance(landing_path, str) and landing_path.lower().endswith(".pdf"):
            path = (PROJECT_ROOT / landing_path).resolve()
            try:
                path.relative_to(landing_root)
            except ValueError as error:
                raise ValueError(
                    f"Upload path is outside data/landing: {landing_path}"
                ) from error
            source_path = path
        elif isinstance(standardized_path, str) and standardized_path.lower().endswith(
            ".md"
        ):
            source_path = (PROJECT_ROOT / standardized_path).resolve()
            standardized_root = (PROJECT_ROOT / "data" / "standardized").resolve()
            try:
                source_path.relative_to(standardized_root)
            except ValueError as error:
                raise ValueError(
                    f"Markdown path is outside data/standardized: {standardized_path}"
                ) from error
            path = _build_markdown_pdf(item, source_path)
        else:
            continue
        if not source_path.is_file() or source_path.stat().st_size < 100:
            raise FileNotFoundError(f"Eligible source is missing or invalid: {source_path}")
        if not path.is_file() or path.stat().st_size < 1024:
            raise FileNotFoundError(f"Eligible PDF is missing or invalid: {path}")
        source_checksum = _sha256(source_path)
        input_checksum = _sha256(path)
        documents.append(
            {
                "document_id": str(item["document_id"]),
                "path": path,
                "source_path": source_path.relative_to(PROJECT_ROOT).as_posix(),
                "pageindex_input_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "source": source_path.name,
                "source_url": str(item.get("url") or ""),
                "title": str(item.get("title") or path.stem),
                "scope": str(item.get("scope") or ""),
                "source_type": str(item.get("source_type") or "unknown"),
                "language": str(item.get("language") or "vi"),
                "checksum": source_checksum,
                "input_checksum": input_checksum,
            }
        )
    return sorted(documents, key=lambda item: item["document_id"])


def _make_backend() -> PageIndexBackend:
    return CloudLegacyRetrievalBackend(PAGEINDEX_API_KEY)


def _poll_documents(
    backend: PageIndexBackend,
    registry: dict[str, Any],
    pending_ids: set[str],
) -> None:
    deadline = time.monotonic() + PAGEINDEX_PROCESSING_TIMEOUT_SECONDS
    delay = PAGEINDEX_POLL_INITIAL_SECONDS
    while pending_ids and time.monotonic() < deadline:
        for document_id in sorted(tuple(pending_ids)):
            entry = registry["documents"][document_id]
            try:
                state = backend.get_document_status(entry["pageindex_doc_id"])
                status = str(state.get("status") or "processing").casefold()
                retrieval_ready = bool(state.get("retrieval_ready", False))
                entry.update(
                    {
                        "status": status,
                        "retrieval_ready": retrieval_ready,
                        "last_checked_at": _now(),
                        "error": None,
                    }
                )
                if retrieval_ready:
                    entry["status"] = "completed"
                    pending_ids.remove(document_id)
                elif status in TERMINAL_FAILURES:
                    entry["error"] = f"remote_{status}"
                    pending_ids.remove(document_id)
            except PageIndexIntegrationError as error:
                entry.update(
                    {
                        "last_checked_at": _now(),
                        "error": error.kind,
                    }
                )
                _record_error(error, "upload_poll")
                if error.kind in {"auth_error", "invalid_request"}:
                    pending_ids.remove(document_id)
        _write_registry(registry)
        if pending_ids:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, PAGEINDEX_POLL_MAX_SECONDS)

    for document_id in sorted(pending_ids):
        entry = registry["documents"][document_id]
        entry.update(
            {
                "status": "timeout",
                "retrieval_ready": False,
                "last_checked_at": _now(),
                "error": "processing_timeout",
            }
        )
    if pending_ids:
        _write_registry(registry)


def upload_documents(force: bool = False, wait_until_ready: bool = True) -> dict[str, str]:
    """Upload changed eligible PDFs and return canonical ID -> PageIndex doc ID."""

    if not PAGEINDEX_API_KEY:
        _record_error(
            PageIndexIntegrationError("not_configured", "PAGEINDEX_API_KEY is missing"),
            "upload",
        )
        return {}
    try:
        registry = _load_registry(strict=True)
        documents = _eligible_documents()
        backend = _make_backend()
    except (OSError, ValueError, RegistryError, PageIndexIntegrationError) as error:
        _record_error(error, "upload")
        return {}

    pending_ids: set[str] = set()
    mapping: dict[str, str] = {}
    for item in documents:
        document_id = item["document_id"]
        existing = registry["documents"].get(document_id, {})
        reusable = (
            not force
            and existing.get("pageindex_doc_id")
            and existing.get("pageindex_input_checksum") == item["input_checksum"]
        )
        if reusable:
            existing.update(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"path", "checksum", "input_checksum"}
                }
            )
            existing["checksum"] = item["checksum"]
            registry["documents"][document_id] = existing
            mapping[document_id] = str(existing["pageindex_doc_id"])
            if wait_until_ready and not existing.get("retrieval_ready"):
                pending_ids.add(document_id)
            continue

        uploaded_at = _now()
        try:
            pageindex_doc_id = backend.submit_document(item["path"])
            registry["documents"][document_id] = {
                "document_id": document_id,
                "pageindex_doc_id": pageindex_doc_id,
                "source": item["source"],
                "source_path": item["source_path"],
                "pageindex_input_path": item["pageindex_input_path"],
                "source_url": item["source_url"],
                "title": item["title"],
                "scope": item["scope"],
                "source_type": item["source_type"],
                "language": item["language"],
                "checksum": item["checksum"],
                "pageindex_input_checksum": item["input_checksum"],
                "status": "submitted",
                "retrieval_ready": False,
                "uploaded_at": uploaded_at,
                "last_checked_at": uploaded_at,
                "error": None,
            }
            mapping[document_id] = pageindex_doc_id
            if wait_until_ready:
                pending_ids.add(document_id)
            _write_registry(registry)
        except PageIndexIntegrationError as error:
            if existing.get("pageindex_doc_id"):
                # A failed forced refresh must not discard a previously usable
                # remote version. Keep it available and record the attempt.
                existing["last_upload_attempt_at"] = _now()
                existing["last_upload_attempt_error"] = error.kind
                registry["documents"][document_id] = existing
                mapping[document_id] = str(existing["pageindex_doc_id"])
            else:
                registry["documents"][document_id] = {
                    "document_id": document_id,
                    "source": item["source"],
                    "source_path": item["source_path"],
                    "pageindex_input_path": item["pageindex_input_path"],
                    "source_url": item["source_url"],
                    "title": item["title"],
                    "scope": item["scope"],
                    "source_type": item["source_type"],
                    "language": item["language"],
                    "checksum": item["checksum"],
                    "pageindex_input_checksum": item["input_checksum"],
                    "status": "upload_failed",
                    "retrieval_ready": False,
                    "uploaded_at": None,
                    "last_checked_at": _now(),
                    "error": error.kind,
                }
            _write_registry(registry)
            _record_error(error, "upload")

    _write_registry(registry)
    if wait_until_ready and pending_ids:
        _poll_documents(backend, registry, pending_ids)
    ready_count = sum(
        bool(entry.get("retrieval_ready"))
        for entry in registry["documents"].values()
    )
    if ready_count:
        _record_success("upload")
    return mapping


def _normalize_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return {token for token in re.findall(r"[a-z0-9]+", ascii_text) if len(token) > 1}


def select_pageindex_documents(
    query: str,
    registry: dict[str, Any],
    max_documents: int = PAGEINDEX_MAX_DOCUMENTS_PER_QUERY,
) -> list[dict[str, Any]]:
    """Deterministically shortlist ready files using local metadata only."""

    query_tokens = _normalize_tokens(query)
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for document_id, entry in registry.get("documents", {}).items():
        if not entry.get("retrieval_ready") or not entry.get("pageindex_doc_id"):
            continue
        title_tokens = _normalize_tokens(str(entry.get("title") or ""))
        other_tokens = _normalize_tokens(
            " ".join(
                str(entry.get(key) or "")
                for key in ("document_id", "source", "scope", "source_type")
            )
        )
        score = 3.0 * len(query_tokens & title_tokens) + len(query_tokens & other_tokens)
        ranked.append((score, str(document_id), entry))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    positive = [item for item in ranked if item[0] > 0]
    selected = positive if positive else ranked
    output: list[dict[str, Any]] = []
    for rank, item in enumerate(selected[: max(1, max_documents)], start=1):
        entry = copy.deepcopy(item[2])
        entry["_shortlist_rank"] = rank
        output.append(entry)
    return output


def _iter_relevant_items(value: Any):
    if isinstance(value, dict):
        if any(key in value for key in ("relevant_content", "content", "text")):
            yield value
        else:
            for child in value.values():
                yield from _iter_relevant_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_relevant_items(child)


def _iter_nodes(value: Any):
    if isinstance(value, list):
        for child in value:
            yield from _iter_nodes(child)
    elif isinstance(value, dict):
        if "relevant_contents" in value or any(
            key in value for key in ("relevant_content", "content", "text")
        ):
            yield value
        children = value.get("nodes") or value.get("children")
        if children is not None:
            yield from _iter_nodes(children)


def _normalize_page_index(value: Any) -> int | str | None:
    """Normalize both documented page_index and live <physical_index_N>."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    match = re.search(r"(?:physical_index_)?(\d+)", text)
    return int(match.group(1)) if match else (text or None)


def _parse_retrieval_result(
    payload: dict[str, Any], entry: dict[str, Any]
) -> list[dict[str, Any]]:
    retrieved_nodes = payload.get("retrieved_nodes")
    if retrieved_nodes is None and isinstance(payload.get("result"), dict):
        retrieved_nodes = payload["result"].get("retrieved_nodes")
    if retrieved_nodes is None:
        raise PageIndexIntegrationError(
            "schema_error", "Completed retrieval response has no retrieved_nodes"
        )

    passages: list[dict[str, Any]] = []
    for node in _iter_nodes(retrieved_nodes):
        # The current live legacy response uses ``id`` while the documented
        # example and older captures use ``node_id``.
        node_id = str(node.get("node_id") or node.get("id") or "unknown")
        node_title = str(node.get("title") or entry.get("title") or "")
        relevant = node.get("relevant_contents", node)
        for item in _iter_relevant_items(relevant):
            content = next(
                (
                    str(item[key]).strip()
                    for key in ("relevant_content", "content", "text")
                    if item.get(key) is not None and str(item[key]).strip()
                ),
                "",
            )
            if not content:
                continue
            content = content[:PAGEINDEX_MAX_EVIDENCE_CHARS]
            section = str(
                item.get("section_title") or node_title or entry.get("title") or ""
            )
            page_index = _normalize_page_index(
                item.get(
                    "page_index",
                    item.get(
                        "physical_index",
                        node.get("page_index", node.get("physical_index")),
                    ),
                )
            )
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            passages.append(
                {
                    "content": content,
                    "node_id": node_id,
                    "section": section,
                    "page_index": page_index,
                    "content_hash": content_hash,
                }
            )
    return passages


def _cache_key(query: str, selected: list[dict[str, Any]]) -> str:
    payload = {
        "parser": PARSER_VERSION,
        "query": " ".join(query.casefold().split()),
        "documents": [
            [entry.get("pageindex_doc_id"), entry.get("pageindex_input_checksum")]
            for entry in selected
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _cached_results(key: str) -> list[dict[str, Any]] | None:
    cached = _QUERY_CACHE.get(key)
    if not cached:
        return None
    created, results = cached
    if time.monotonic() - created > PAGEINDEX_QUERY_CACHE_TTL_SECONDS:
        _QUERY_CACHE.pop(key, None)
        return None
    output = copy.deepcopy(results)
    for result in output:
        result["metadata"]["cache_hit"] = True
    return output


def pageindex_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Return real PageIndex evidence passages, or ``[]`` when unavailable."""

    query = str(query).strip()
    if not query or top_k <= 0:
        return []
    if not PAGEINDEX_API_KEY:
        _record_error(
            PageIndexIntegrationError("not_configured", "PAGEINDEX_API_KEY is missing"),
            "search",
        )
        return []
    registry = _load_registry()
    selected = select_pageindex_documents(query, registry)
    if not selected:
        _record_error(
            PageIndexIntegrationError("not_ready", "No retrieval-ready documents"),
            "search",
        )
        return []
    key = _cache_key(query, selected)
    cached = _cached_results(key)
    if cached is not None:
        _record_success("query_cache")
        return cached[:top_k]

    try:
        backend = _make_backend()
    except PageIndexIntegrationError as error:
        _record_error(error, "search")
        return []

    jobs: dict[str, dict[str, Any]] = {}
    for entry in selected:
        try:
            retrieval_id = backend.submit_retrieval(
                str(entry["pageindex_doc_id"]), query, False
            )
            jobs[retrieval_id] = entry
        except PageIndexIntegrationError as error:
            _record_error(error, "retrieval_submit")

    deadline = time.monotonic() + PAGEINDEX_RETRIEVAL_TIMEOUT_SECONDS
    delay = PAGEINDEX_POLL_INITIAL_SECONDS
    completed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    while jobs and time.monotonic() < deadline:
        for retrieval_id in sorted(tuple(jobs)):
            try:
                payload = backend.get_retrieval(retrieval_id)
                status = str(payload.get("status") or "processing").casefold()
                if status == "completed" or payload.get("retrieved_nodes") is not None:
                    completed.append((jobs.pop(retrieval_id), payload))
                elif status in TERMINAL_FAILURES:
                    jobs.pop(retrieval_id)
            except PageIndexIntegrationError as error:
                _record_error(error, "retrieval_poll")
                if error.kind in {"auth_error", "invalid_request", "schema_error"}:
                    jobs.pop(retrieval_id)
        if jobs:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, PAGEINDEX_POLL_MAX_SECONDS)

    evidence: list[tuple[dict[str, Any], dict[str, Any]]] = []
    completed.sort(key=lambda item: int(item[0].get("_shortlist_rank", 10_000)))
    for entry, payload in completed:
        try:
            for passage in _parse_retrieval_result(payload, entry):
                evidence.append((entry, passage))
        except PageIndexIntegrationError as error:
            _record_error(error, "retrieval_parse")

    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for entry, passage in evidence:
        dedupe_key = (
            str(entry["pageindex_doc_id"]),
            passage["node_id"],
            passage["content_hash"],
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rank = len(output) + 1
        score = 1.0 / rank
        chunk_id = (
            f"pageindex:{entry['pageindex_doc_id']}:{passage['node_id']}:"
            f"{passage['content_hash']}"
        )
        output.append(
            {
                "content": passage["content"],
                "score": score,
                "score_type": "rank_proxy",
                "confidence_score": None,
                "source": "pageindex",
                "retrieval_method": RETRIEVAL_METHOD,
                "metadata": {
                    "chunk_id": chunk_id,
                    "document_id": entry["document_id"],
                    "pageindex_doc_id": entry["pageindex_doc_id"],
                    "source": entry["source"],
                    "type": entry.get("source_type", "unknown"),
                    "language": entry.get("language", "vi"),
                    "title": entry["title"],
                    "url": entry.get("source_url", ""),
                    "section": passage["section"],
                    "node_id": passage["node_id"],
                    "page_index": passage["page_index"],
                    "rank": rank,
                    "shortlist_rank": entry.get("_shortlist_rank"),
                    "score_kind": "rank_proxy",
                    "cache_hit": False,
                },
                "raw_scores": {"pageindex_rank_proxy": score},
            }
        )
        if len(output) >= top_k:
            break

    if output:
        _QUERY_CACHE[key] = (time.monotonic(), copy.deepcopy(output))
        _record_success("pageindex_live")
    elif not _SERVICE_STATUS.get("last_error_type"):
        _record_error(
            PageIndexIntegrationError("empty_result", "PageIndex returned no evidence"),
            "search",
        )
    return output


def get_pageindex_status() -> dict[str, Any]:
    """Return sanitized configuration, readiness and last-run diagnostics."""

    registry = _load_registry()
    documents = registry.get("documents", {})
    ready = sum(bool(entry.get("retrieval_ready")) for entry in documents.values())
    return {
        "available": bool(PAGEINDEX_API_KEY) and ready > 0,
        "configured": bool(PAGEINDEX_API_KEY),
        "ready_documents": ready,
        "total_registered_documents": len(documents),
        "last_error_type": _SERVICE_STATUS.get("last_error_type"),
        "last_error_message": _SERVICE_STATUS.get("last_error_message"),
        "last_success_at": _SERVICE_STATUS.get("last_success_at"),
        "last_route": _SERVICE_STATUS.get("last_route"),
        "backend": BACKEND_NAME,
        "sdk_version": _sdk_version(),
        "registry_path": PAGEINDEX_REGISTRY_PATH.name,
    }


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help="upload/reuse eligible PDFs")
    parser.add_argument("--force", action="store_true", help="upload new remote versions")
    parser.add_argument("--no-wait", action="store_true", help="do not poll processing")
    parser.add_argument("--query", help="run one live PageIndex evidence query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.upload:
        mapping = upload_documents(force=args.force, wait_until_ready=not args.no_wait)
        print(json.dumps({"uploaded_or_reused": mapping}, ensure_ascii=False, indent=2))
    if args.query:
        results = pageindex_search(args.query, top_k=args.top_k)
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    if args.status or (not args.upload and not args.query):
        print(json.dumps(get_pageindex_status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
