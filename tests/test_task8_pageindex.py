"""Offline contract tests for the real PageIndex adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import task8_pageindex_vectorless as task8


class FakeBackend:
    def __init__(self) -> None:
        self.submit_calls: list[Path] = []
        self.retrieval_calls: list[tuple[str, str, bool]] = []
        self.fail_submit = False

    def submit_document(self, path: Path) -> str:
        self.submit_calls.append(path)
        if self.fail_submit:
            raise task8.PageIndexIntegrationError("remote_error", "upload failed")
        return f"pi-{len(self.submit_calls)}"

    def get_document_status(self, _doc_id: str) -> dict:
        return {"status": "completed", "retrieval_ready": True}

    def submit_retrieval(self, doc_id: str, query: str, thinking: bool) -> str:
        self.retrieval_calls.append((doc_id, query, thinking))
        return f"retrieval-{len(self.retrieval_calls)}"

    def get_retrieval(self, _retrieval_id: str) -> dict:
        return {
            "status": "completed",
            "retrieved_nodes": [
                {
                    "title": "Mức thu học phí",
                    "id": "0005",
                    "relevant_contents": [
                        [
                            {
                                "physical_index": "<physical_index_10>",
                                "relevant_content": "Mức thu học lại là 980.000 đồng mỗi tín chỉ.",
                            }
                        ]
                    ],
                }
            ],
        }


@pytest.fixture
def pageindex_sandbox(tmp_path, monkeypatch):
    project = tmp_path
    landing = project / "data" / "landing" / "legal"
    landing.mkdir(parents=True)
    pdf = landing / "tuition.pdf"
    pdf.write_bytes(b"%PDF-1.4\n" + b"x" * 2048)
    manifest = [
        {
            "document_id": "tuition-policy",
            "title": "Thông báo học phí và mức thu học lại",
            "source_type": "legal",
            "language": "vi",
            "scope": "Sinh viên đại học chính quy",
            "url": "https://example.edu/tuition",
            "landing_path": "data/landing/legal/tuition.pdf",
        }
    ]
    manifest_path = project / "data" / "sources_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    registry_path = project / "pageindex_doc_ids.json"

    monkeypatch.setattr(task8, "PROJECT_ROOT", project)
    monkeypatch.setattr(task8, "LANDING_DIR", project / "data" / "landing")
    monkeypatch.setattr(task8, "SOURCES_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(task8, "PAGEINDEX_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(task8, "PAGEINDEX_API_KEY", "test-key")
    monkeypatch.setattr(task8, "PAGEINDEX_POLL_INITIAL_SECONDS", 0.001)
    monkeypatch.setattr(task8, "PAGEINDEX_POLL_MAX_SECONDS", 0.001)
    monkeypatch.setattr(task8, "PAGEINDEX_PROCESSING_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(task8, "PAGEINDEX_RETRIEVAL_TIMEOUT_SECONDS", 0.1)
    task8._QUERY_CACHE.clear()
    task8._SERVICE_STATUS.update(
        {
            "last_error_type": None,
            "last_error_message": None,
            "last_success_at": None,
            "last_route": "not_run",
        }
    )
    return registry_path


def test_missing_key_is_safe(monkeypatch):
    monkeypatch.setattr(task8, "PAGEINDEX_API_KEY", "")
    assert task8.upload_documents() == {}
    assert task8.pageindex_search("học phí") == []
    assert task8.get_pageindex_status()["configured"] is False


def test_upload_persists_ready_registry_and_reuses_checksum(
    pageindex_sandbox, monkeypatch
):
    backend = FakeBackend()
    monkeypatch.setattr(task8, "_make_backend", lambda: backend)

    first = task8.upload_documents()
    second = task8.upload_documents()

    assert first == {"tuition-policy": "pi-1"}
    assert second == first
    assert len(backend.submit_calls) == 1
    registry = json.loads(pageindex_sandbox.read_text(encoding="utf-8"))
    entry = registry["documents"]["tuition-policy"]
    assert entry["retrieval_ready"] is True
    assert entry["status"] == "completed"
    assert entry["checksum"].startswith("sha256:")
    assert entry["pageindex_input_checksum"] == entry["checksum"]


def test_corrupt_registry_refuses_automatic_upload(pageindex_sandbox, monkeypatch):
    pageindex_sandbox.write_text("{broken", encoding="utf-8")
    backend = FakeBackend()
    monkeypatch.setattr(task8, "_make_backend", lambda: backend)

    assert task8.upload_documents() == {}
    assert backend.submit_calls == []
    assert pageindex_sandbox.read_text(encoding="utf-8") == "{broken"


def test_forced_upload_failure_preserves_previous_ready_document(
    pageindex_sandbox, monkeypatch
):
    backend = FakeBackend()
    monkeypatch.setattr(task8, "_make_backend", lambda: backend)
    task8.upload_documents()
    backend.fail_submit = True

    mapping = task8.upload_documents(force=True)
    registry = json.loads(pageindex_sandbox.read_text(encoding="utf-8"))
    entry = registry["documents"]["tuition-policy"]

    assert mapping == {"tuition-policy": "pi-1"}
    assert entry["pageindex_doc_id"] == "pi-1"
    assert entry["retrieval_ready"] is True
    assert entry["last_upload_attempt_error"] == "remote_error"


def test_shortlist_uses_ready_metadata_and_is_deterministic():
    registry = {
        "documents": {
            "dormitory": {
                "document_id": "dormitory",
                "title": "Quy chế nội trú ký túc xá",
                "pageindex_doc_id": "pi-dorm",
                "retrieval_ready": True,
            },
            "tuition": {
                "document_id": "tuition",
                "title": "Thông báo học phí",
                "pageindex_doc_id": "pi-tuition",
                "retrieval_ready": True,
            },
            "not-ready": {
                "document_id": "not-ready",
                "title": "Học phí",
                "pageindex_doc_id": "pi-pending",
                "retrieval_ready": False,
            },
        }
    }
    selected = task8.select_pageindex_documents("mức học phí", registry, 2)
    assert [item["document_id"] for item in selected] == ["tuition"]
    assert selected[0]["_shortlist_rank"] == 1


def test_live_shape_parser_and_query_cache(pageindex_sandbox, monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(task8, "_make_backend", lambda: backend)
    task8.upload_documents()

    first = task8.pageindex_search("mức học phí", top_k=2)
    second = task8.pageindex_search("mức học phí", top_k=2)

    assert len(first) == 1
    assert first[0]["source"] == "pageindex"
    assert first[0]["score_type"] == "rank_proxy"
    assert first[0]["confidence_score"] is None
    assert first[0]["metadata"]["document_id"] == "tuition-policy"
    assert first[0]["metadata"]["page_index"] == 10
    assert first[0]["metadata"]["cache_hit"] is False
    assert second[0]["metadata"]["cache_hit"] is True
    assert len(backend.retrieval_calls) == 1


def test_completed_response_without_nodes_is_safely_rejected(
    pageindex_sandbox, monkeypatch
):
    backend = FakeBackend()
    backend.get_retrieval = lambda _retrieval_id: {"status": "completed"}
    monkeypatch.setattr(task8, "_make_backend", lambda: backend)
    task8.upload_documents()

    assert task8.pageindex_search("học phí") == []
    assert task8.get_pageindex_status()["last_error_type"] == "schema_error"
