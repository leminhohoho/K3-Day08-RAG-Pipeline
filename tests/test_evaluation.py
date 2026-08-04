"""Offline tests for the group-project evaluation layer."""

from __future__ import annotations

import copy
import json

import pytest

from group_project.evaluation.eval_pipeline import (
    _load_existing_run,
    load_golden_dataset,
    select_cases,
    validate_golden_dataset,
)
from group_project.evaluation.metrics import (
    document_recall_at_k,
    duplicate_context_rate_at_k,
    evidence_coverage_at_k,
    ndcg_at_k,
    reciprocal_rank,
    source_hit_at_k,
)
from group_project.evaluation.pipeline_adapters import retrieve_configurations


def _case(*sources: str) -> dict:
    aliases = [[source.removesuffix(".md") + "-canonical", source.removesuffix(".md")] for source in sources]
    return {
        "expected_source_files": list(sources),
        "expected_document_id_aliases": aliases,
        "expected_context": "học phí mười triệu đồng trong một năm học",
    }


def _result(source: str, chunk: int, content: str = "") -> dict:
    return {
        "content": content,
        "score": 1.0,
        "metadata": {
            "source": source,
            "document_id": source.removesuffix(".md"),
            "chunk_id": f"{source}:{chunk}",
        },
    }


def test_real_golden_dataset_is_valid_and_covers_manifest():
    summary = validate_golden_dataset(load_golden_dataset())
    assert summary["valid"] is True
    assert summary["case_count"] == 28
    assert summary["document_coverage"] == "10/10"
    assert summary["splits"] == {"challenge": 8, "core": 18, "safety": 2}


def test_select_cases_filters_before_limit():
    cases = load_golden_dataset()
    selected = select_cases(cases, ["challenge"], 3)
    assert len(selected) == 3
    assert all(case["evaluation_split"] == "challenge" for case in selected)


def test_single_document_metrics_reward_first_rank():
    case = _case("correct.md")
    results = [_result("correct.md", 0), _result("other.md", 0)]
    assert source_hit_at_k(results, case, 1) == 1.0
    assert document_recall_at_k(results, case, 5) == 1.0
    assert reciprocal_rank(results, case, 5) == 1.0
    assert ndcg_at_k(results, case, 5) == 1.0


def test_wrong_source_metrics_are_zero():
    case = _case("correct.md")
    results = [_result("other.md", 0)]
    assert source_hit_at_k(results, case, 5) == 0.0
    assert document_recall_at_k(results, case, 5) == 0.0
    assert reciprocal_rank(results, case, 5) == 0.0
    assert ndcg_at_k(results, case, 5) == 0.0


def test_multi_document_recall_does_not_reward_duplicate_chunks():
    case = _case("a.md", "b.md")
    only_a = [_result("a.md", 0), _result("a.md", 1)]
    both = [_result("a.md", 0), _result("b.md", 0)]
    assert document_recall_at_k(only_a, case, 5) == 0.5
    assert document_recall_at_k(both, case, 5) == 1.0
    assert ndcg_at_k(only_a, case, 5) < ndcg_at_k(both, case, 5)


def test_evidence_coverage_and_duplicate_rate():
    case = _case("correct.md")
    results = [
        _result("correct.md", 0, "Học phí mười triệu đồng trong một năm học."),
        _result("correct.md", 0, "same chunk ID"),
    ]
    assert evidence_coverage_at_k(results, case, 5) == 1.0
    assert duplicate_context_rate_at_k(results, 5) == 0.5


def test_retrieval_configs_share_backends_and_do_not_mutate_inputs():
    dense = [_result("dense.md", 0)]
    bm25 = [_result("bm25.md", 0)]
    original_dense = copy.deepcopy(dense)
    original_bm25 = copy.deepcopy(bm25)
    calls = {"dense": 0, "bm25": 0, "rrf": 0}

    def dense_search(_query: str, _top_k: int):
        calls["dense"] += 1
        return dense

    def bm25_search(_query: str, _top_k: int):
        calls["bm25"] += 1
        return bm25

    def fuse(ranked_lists, top_k, k, weights):
        calls["rrf"] += 1
        assert k == 60
        assert weights == [1.0, 0.9]
        return (ranked_lists[0] + ranked_lists[1])[:top_k]

    outputs = retrieve_configurations(
        "query",
        ["dense_only", "bm25_only", "hybrid_rrf"],
        top_k=2,
        dense_search=dense_search,
        bm25_search=bm25_search,
        rrf_fusion=fuse,
    )
    assert calls == {"dense": 1, "bm25": 1, "rrf": 1}
    assert set(outputs) == {"dense_only", "bm25_only", "hybrid_rrf"}
    assert outputs["hybrid_rrf"]["results"][0]["source"] == "hybrid"
    assert dense == original_dense
    assert bm25 == original_bm25


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="Unknown evaluation configs"):
        retrieve_configurations("query", ["fake_config"])


def test_resume_loads_frozen_retrieval_artifacts(tmp_path):
    manifest = {"configs": {"dense_only": {}}}
    predictions = {"cases": []}
    retrieval_metrics = {"configs": {"dense_only": {}}}
    for name, payload in (
        ("run_manifest.json", manifest),
        ("predictions.json", predictions),
        ("retrieval_metrics.json", retrieval_metrics),
    ):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    loaded_manifest, loaded_predictions, loaded_metrics, configs = _load_existing_run(
        tmp_path, ["dense_only"]
    )
    assert loaded_manifest["resumed_stages"] == []
    assert loaded_predictions == predictions
    assert loaded_metrics == retrieval_metrics
    assert configs == ["dense_only"]


def test_resume_rejects_config_not_in_frozen_run(tmp_path):
    for name, payload in (
        ("run_manifest.json", {"configs": {"dense_only": {}}}),
        ("predictions.json", {"cases": []}),
        ("retrieval_metrics.json", {"configs": {"dense_only": {}}}),
    ):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain configurations"):
        _load_existing_run(tmp_path, ["hybrid_rrf"])
