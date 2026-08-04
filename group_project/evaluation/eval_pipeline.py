r"""Reproducible retrieval, RAGAS and safety evaluation for the RAG pipeline.

Examples (PowerShell):

    .\.venv\Scripts\python.exe group_project\evaluation\eval_pipeline.py --validate
    .\.venv\Scripts\python.exe group_project\evaluation\eval_pipeline.py --mode retrieval
    .\.venv\Scripts\python.exe group_project\evaluation\eval_pipeline.py --mode ragas --limit 3

The default retrieval evaluation is deterministic and model-free apart from the
embedding model already required by dense search. RAGAS and generation are lazy
optional paths and never run during validation/unit tests.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from group_project.evaluation.metrics import (  # noqa: E402
    aggregate_by_field,
    aggregate_case_metrics,
    evaluate_retrieval_case,
    evidence_tokens,
)
from group_project.evaluation.pipeline_adapters import (  # noqa: E402
    CONFIG_DESCRIPTIONS,
    RRF_K,
    RRF_WEIGHTS,
    SUPPORTED_CONFIGS,
    retrieve_configurations,
    validate_config_names,
)


EVALUATION_DIR = Path(__file__).resolve().parent
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "results.md"
RUNS_DIR = EVALUATION_DIR / "runs"
MANIFEST_PATH = PROJECT_DIR / "data" / "sources_manifest.json"
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"

REQUIRED_CASE_FIELDS = {
    "id",
    "question",
    "expected_answer",
    "expected_context",
    "expected_document_ids",
    "expected_document_id_aliases",
    "expected_source_files",
    "expected_sections",
    "category",
    "difficulty",
    "question_type",
    "answerable",
    "evaluation_split",
    "tags",
}
VALID_SPLITS = {"core", "challenge", "safety"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
REFUSAL_PHRASES = (
    "không thể xác minh",
    "không đủ thông tin",
    "nguồn hiện có không",
    "context không đủ",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _corpus_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        digest.update(path.relative_to(PROJECT_DIR).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_golden_dataset() -> list[dict[str, Any]]:
    """Load the list-shaped golden dataset used by starter and extended eval."""

    payload = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("golden_dataset.json must contain a JSON list")
    return payload


def _heading_labels(text: str) -> set[str]:
    headings: set[str] = set()
    for line in text.splitlines():
        if not re.match(r"^#{1,6}\s+", line):
            continue
        label = re.sub(r"^#{1,6}\s+", "", line).strip().strip("*").strip()
        headings.add(label)
    return headings


def validate_golden_dataset(
    golden_dataset: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate schema, manifest references, aliases, headings and coverage."""

    cases = golden_dataset if golden_dataset is not None else load_golden_dataset()
    errors: list[str] = []
    warnings: list[str] = []
    if len(cases) < 15:
        errors.append(f"Need at least 15 cases, found {len(cases)}")

    identifiers = [str(case.get("id", "")) for case in cases]
    duplicates = sorted(item for item, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        errors.append(f"Duplicate case IDs: {duplicates}")

    manifest_payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    manifest = {str(item["document_id"]): item for item in manifest_payload}
    covered_documents: set[str] = set()

    for index, case in enumerate(cases):
        case_id = str(case.get("id") or f"index-{index}")
        missing = sorted(REQUIRED_CASE_FIELDS - set(case))
        if missing:
            errors.append(f"{case_id}: missing fields {missing}")
            continue
        if not str(case["question"]).strip() or not str(case["expected_answer"]).strip():
            errors.append(f"{case_id}: question/expected_answer must be non-empty")
        if case["evaluation_split"] not in VALID_SPLITS:
            errors.append(f"{case_id}: invalid evaluation_split")
        if case["difficulty"] not in VALID_DIFFICULTIES:
            errors.append(f"{case_id}: invalid difficulty")
        if case["evaluation_split"] == "safety" and case["answerable"] is not False:
            errors.append(f"{case_id}: safety cases must set answerable=false")

        document_ids = case["expected_document_ids"]
        alias_groups = case["expected_document_id_aliases"]
        sources = case["expected_source_files"]
        if not (
            isinstance(document_ids, list)
            and isinstance(alias_groups, list)
            and isinstance(sources, list)
            and len(document_ids) == len(alias_groups) == len(sources)
            and len(document_ids) > 0
        ):
            errors.append(f"{case_id}: expected document/source arrays are misaligned")
            continue

        valid_headings: set[str] = set()
        source_texts: list[str] = []
        for document_id, aliases, source in zip(document_ids, alias_groups, sources):
            document_id = str(document_id)
            covered_documents.add(document_id)
            entry = manifest.get(document_id)
            if entry is None:
                errors.append(f"{case_id}: unknown canonical document_id {document_id}")
                continue
            standardized_path = PROJECT_DIR / str(entry["standardized_path"])
            if not standardized_path.exists():
                errors.append(f"{case_id}: missing source file {standardized_path}")
                continue
            if standardized_path.name != source:
                errors.append(
                    f"{case_id}: source {source} disagrees with manifest {standardized_path.name}"
                )
            expected_aliases = [document_id, Path(source).stem]
            if aliases != expected_aliases:
                errors.append(
                    f"{case_id}: aliases must be canonical/runtime pair {expected_aliases}"
                )
            text = standardized_path.read_text(encoding="utf-8-sig")
            source_texts.append(text)
            valid_headings.update(_heading_labels(text))
            header_match = re.search(r"(?m)^\*\*Document ID:\*\*\s*(.+?)\s*$", text)
            if not header_match or header_match.group(1).strip() != document_id:
                errors.append(f"{case_id}: Markdown Document ID does not match {document_id}")

        for section in case["expected_sections"]:
            if section not in valid_headings:
                errors.append(f"{case_id}: section is not a real heading/title: {section}")

        expected_tokens = evidence_tokens(case["expected_context"])
        source_tokens = evidence_tokens("\n".join(source_texts))
        overlap = len(expected_tokens & source_tokens) / max(1, len(expected_tokens))
        if overlap < 0.55:
            warnings.append(f"{case_id}: expected_context/source token overlap is {overlap:.2f}")

    uncovered = sorted(set(manifest) - covered_documents)
    if uncovered:
        errors.append(f"Golden dataset does not cover manifest documents: {uncovered}")
    if errors:
        raise ValueError("Golden dataset validation failed:\n- " + "\n- ".join(errors))

    return {
        "valid": True,
        "case_count": len(cases),
        "answerable_count": sum(bool(case["answerable"]) for case in cases),
        "document_coverage": f"{len(covered_documents)}/{len(manifest)}",
        "splits": dict(sorted(Counter(case["evaluation_split"] for case in cases).items())),
        "difficulty": dict(sorted(Counter(case["difficulty"] for case in cases).items())),
        "categories": dict(sorted(Counter(case["category"] for case in cases).items())),
        "warnings": warnings,
    }


def select_cases(
    cases: list[dict[str, Any]],
    splits: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = cases
    if splits and "all" not in splits:
        invalid = sorted(set(splits) - VALID_SPLITS)
        if invalid:
            raise ValueError(f"Unknown splits: {invalid}")
        selected = [case for case in selected if case["evaluation_split"] in splits]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    return selected


def _serialize_retrieval_result(item: dict[str, Any]) -> dict[str, Any]:
    return _json_safe(
        {
            "content": str(item.get("content", "")),
            "score": item.get("score"),
            "score_type": item.get("score_type"),
            "confidence_score": item.get("confidence_score"),
            "source": item.get("source"),
            "metadata": item.get("metadata") or {},
            "raw_scores": item.get("raw_scores") or {},
        }
    )


def run_retrieval_evaluation(
    cases: list[dict[str, Any]],
    config_names: list[str],
    *,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all configs over identical queries and compute deterministic metrics."""

    validate_config_names(config_names)
    records_by_config: dict[str, list[dict[str, Any]]] = {
        config: [] for config in config_names
    }
    prediction_cases: list[dict[str, Any]] = []

    for position, case in enumerate(cases, start=1):
        print(f"[{position:02d}/{len(cases):02d}] {case['id']}: {case['question']}")
        outputs = retrieve_configurations(case["question"], config_names, top_k=top_k)
        case_prediction: dict[str, Any] = {
            "id": case["id"],
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "expected_context": case["expected_context"],
            "expected_document_ids": case["expected_document_ids"],
            "expected_source_files": case["expected_source_files"],
            "category": case["category"],
            "difficulty": case["difficulty"],
            "question_type": case["question_type"],
            "answerable": case["answerable"],
            "evaluation_split": case["evaluation_split"],
            "configs": {},
        }
        for config in config_names:
            output = outputs[config]
            results = output["results"]
            metrics = evaluate_retrieval_case(
                case,
                results,
                top_k=top_k,
                latency_ms=output["latency_ms"],
            )
            record = {
                "id": case["id"],
                "category": case["category"],
                "difficulty": case["difficulty"],
                "question_type": case["question_type"],
                "evaluation_split": case["evaluation_split"],
                "answerable": case["answerable"],
                "metrics": metrics,
            }
            records_by_config[config].append(record)
            case_prediction["configs"][config] = {
                "description": CONFIG_DESCRIPTIONS[config],
                "latency_ms": output["latency_ms"],
                "component_latency_ms": output["component_latency_ms"],
                "metrics": metrics,
                "results": [_serialize_retrieval_result(item) for item in results],
            }
        prediction_cases.append(case_prediction)

    metrics_payload: dict[str, Any] = {"configs": {}}
    for config, records in records_by_config.items():
        metrics_payload["configs"][config] = {
            "description": CONFIG_DESCRIPTIONS[config],
            "overall": aggregate_case_metrics(records),
            "by_split": aggregate_by_field(records, "evaluation_split"),
            "by_category": aggregate_by_field(records, "category"),
            "by_difficulty": aggregate_by_field(records, "difficulty"),
            "cases": records,
        }
    predictions_payload = {
        "schema_version": 1,
        "top_k": top_k,
        "configs": config_names,
        "cases": prediction_cases,
    }
    return predictions_payload, metrics_payload


def _format_context(results: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}
        label = metadata.get("title") or metadata.get("source") or f"Source {index}"
        url = metadata.get("url") or "N/A"
        section = metadata.get("section") or label
        parts.append(
            f"[Document {index} | Title: {label} | Section: {section} | URL: {url}]\n"
            f"{result.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _openrouter_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("RAGAS/generation requires OPENROUTER_API_KEY or OPENAI_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1"), api_key


def generate_answers(
    predictions: dict[str, Any],
    config_names: list[str],
    *,
    model: str,
    answerable: bool | None = None,
    case_ids: set[str] | None = None,
) -> None:
    """Generate one grounded answer per case/config using already retrieved context."""

    from src.task10_generation import SYSTEM_PROMPT

    client, _ = _openrouter_client()
    for case_index, case in enumerate(predictions["cases"], start=1):
        if case_ids is not None and case["id"] not in case_ids:
            continue
        if answerable is not None and bool(case["answerable"]) is not answerable:
            continue
        for config in config_names:
            config_output = case["configs"][config]
            if config_output.get("answer"):
                continue
            results = config_output["results"]
            if not results:
                config_output["answer"] = "Tôi không thể xác minh thông tin này từ nguồn hiện có"
                config_output["generation_latency_ms"] = 0.0
                continue
            context = _format_context(results)
            user_message = f"Context:\n{context}\n\n---\n\nQuestion: {case['question']}"
            print(
                f"Generate [{case_index:02d}/{len(predictions['cases']):02d}] "
                f"{case['id']} / {config}"
            )
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                top_p=1.0,
            )
            config_output["generation_latency_ms"] = (
                time.perf_counter() - started
            ) * 1000
            config_output["answer"] = response.choices[0].message.content or ""


def evaluate_with_ragas(
    predictions: dict[str, Any],
    config_names: list[str],
    *,
    judge_model: str,
    embedding_model: str,
    judge_max_tokens: int,
    judge_max_workers: int,
    case_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Evaluate the four required RAGAS metrics for answerable cases only."""

    from datasets import Dataset
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from ragas.run_config import RunConfig

    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("RAGAS requires OPENROUTER_API_KEY or OPENAI_API_KEY")
    judge = ChatOpenAI(
        model=judge_model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        # RAGAS expects short structured judgements. Keeping this bounded
        # avoids reserving the provider's much larger default output budget.
        max_tokens=judge_max_tokens,
    )
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        # OpenRouter's BGE-M3 endpoint accepts strings, while older
        # langchain-openai releases otherwise convert unknown models to
        # token-ID arrays before sending the request.
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
    metric_names = (
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    )
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    payload: dict[str, Any] = {
        "case_ids": sorted(case_ids) if case_ids is not None else None,
        "configs": {},
    }
    for config in config_names:
        rows = [
            case
            for case in predictions["cases"]
            if case["answerable"]
            and (case_ids is None or case["id"] in case_ids)
        ]
        if not rows:
            raise ValueError("RAGAS requires at least one selected answerable case")
        dataset = Dataset.from_dict(
            {
                "question": [case["question"] for case in rows],
                "answer": [case["configs"][config]["answer"] for case in rows],
                "contexts": [
                    [item["content"] for item in case["configs"][config]["results"]]
                    for case in rows
                ],
                "ground_truth": [case["expected_answer"] for case in rows],
            }
        )
        print(f"RAGAS: {config} / {len(rows)} answerable cases")
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=judge,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=judge_max_workers, timeout=300),
            raise_exceptions=False,
        )
        frame = result.to_pandas()
        case_scores: list[dict[str, Any]] = []
        for row_index, case in enumerate(rows):
            score_row = {"id": case["id"]}
            for metric_name in metric_names:
                value = frame.iloc[row_index].get(metric_name)
                score_row[metric_name] = _json_safe(value)
            case_scores.append(score_row)
        overall = {}
        for metric_name in metric_names:
            values = [
                float(row[metric_name])
                for row in case_scores
                if isinstance(row[metric_name], (int, float))
                and math.isfinite(float(row[metric_name]))
            ]
            overall[metric_name] = mean(values) if values else None
        payload["configs"][config] = {
            "case_count": len(case_scores),
            "overall": overall,
            "valid_counts": {
                metric_name: sum(
                    isinstance(row[metric_name], (int, float))
                    and math.isfinite(float(row[metric_name]))
                    for row in case_scores
                )
                for metric_name in metric_names
            },
            "cases": case_scores,
        }
    return payload


def _merge_ragas_payload(
    existing: dict[str, Any] | None,
    update: dict[str, Any],
) -> dict[str, Any]:
    """Merge retry results without discarding previously valid case scores."""

    if not existing:
        return update

    merged = copy.deepcopy(existing)
    existing_case_ids = existing.get("case_ids")
    update_case_ids = update.get("case_ids")
    merged["case_ids"] = (
        None
        if existing_case_ids is None or update_case_ids is None
        else sorted(set(existing_case_ids) | set(update_case_ids))
    )
    metric_names = (
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    )
    for config, incoming in update.get("configs", {}).items():
        previous = merged.get("configs", {}).get(config, {})
        previous_by_id = {
            row["id"]: row for row in previous.get("cases", []) if "id" in row
        }
        incoming_by_id = {
            row["id"]: row for row in incoming.get("cases", []) if "id" in row
        }
        ordered_ids = list(previous_by_id)
        ordered_ids.extend(
            case_id for case_id in incoming_by_id if case_id not in previous_by_id
        )
        merged_cases = []
        for case_id in ordered_ids:
            row = incoming_by_id.get(case_id, {})
            combined = {"id": case_id}
            prior = previous_by_id.get(case_id, {})
            for metric_name in metric_names:
                value = row.get(metric_name)
                valid = isinstance(value, (int, float)) and math.isfinite(float(value))
                combined[metric_name] = value if valid else prior.get(metric_name)
            merged_cases.append(combined)

        valid_counts: dict[str, int] = {}
        overall: dict[str, float | None] = {}
        for metric_name in metric_names:
            values = [
                float(row[metric_name])
                for row in merged_cases
                if isinstance(row.get(metric_name), (int, float))
                and math.isfinite(float(row[metric_name]))
            ]
            valid_counts[metric_name] = len(values)
            overall[metric_name] = mean(values) if values else None
        merged.setdefault("configs", {})[config] = {
            "case_count": len(merged_cases),
            "overall": overall,
            "valid_counts": valid_counts,
            "cases": merged_cases,
        }
    return merged


def evaluate_safety(
    predictions: dict[str, Any], config_names: list[str]
) -> dict[str, Any]:
    """Score explicit refusal behavior separately from answerable RAGAS metrics."""

    safety_cases = [case for case in predictions["cases"] if not case["answerable"]]
    payload: dict[str, Any] = {"configs": {}}
    for config in config_names:
        records = []
        for case in safety_cases:
            answer = str(case["configs"][config].get("answer", ""))
            normalized = answer.casefold()
            refused = any(phrase in normalized for phrase in REFUSAL_PHRASES)
            records.append(
                {
                    "id": case["id"],
                    "refused": refused,
                    "unsupported_answer": not refused,
                    "answer": answer,
                }
            )
        payload["configs"][config] = {
            "case_count": len(records),
            "refusal_accuracy": (
                mean(float(record["refused"]) for record in records) if records else None
            ),
            "unsupported_answer_rate": (
                mean(float(record["unsupported_answer"]) for record in records)
                if records
                else None
            ),
            "cases": records,
        }
    return payload


def _package_versions() -> dict[str, str | None]:
    names = (
        "chromadb",
        "langchain-text-splitters",
        "rank-bm25",
        "scikit-learn",
        "ragas",
        "datasets",
        "openai",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_run_manifest(
    cases: list[dict[str, Any]],
    config_names: list[str],
    *,
    top_k: int,
    mode: str,
    generation_model: str,
    judge_model: str,
    judge_embedding_model: str,
) -> dict[str, Any]:
    index_info: dict[str, Any] = {}
    try:
        from src.task4_chunking_indexing import (
            CHUNK_OVERLAP,
            CHUNK_SIZE,
            COLLECTION_NAME,
            EMBEDDING_DIM,
            EMBEDDING_MODEL,
            get_collection,
        )

        collection = get_collection()
        sample = collection.get(limit=1, include=["metadatas"])
        index_info = {
            "collection": COLLECTION_NAME,
            "count": collection.count(),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "sample_metadata": (sample.get("metadatas") or [None])[0],
        }
    except Exception as error:  # Manifest diagnostics must not hide the main run.
        index_info = {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}

    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "corpus_hash": _corpus_hash(),
        "golden_dataset_hash": _sha256_file(GOLDEN_DATASET_PATH),
        "case_ids": [case["id"] for case in cases],
        "top_k": top_k,
        "configs": {
            name: {"description": CONFIG_DESCRIPTIONS[name]} for name in config_names
        },
        "rrf": {"k": RRF_K, "weights": {"dense": RRF_WEIGHTS[0], "bm25": RRF_WEIGHTS[1]}},
        "generation_model": generation_model if mode in {"ragas", "all", "safety"} else None,
        "judge_model": judge_model if mode in {"ragas", "all"} else None,
        "judge_embedding_model": (
            judge_embedding_model if mode in {"ragas", "all"} else None
        ),
        "index": index_info,
        "packages": _package_versions(),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _retrieval_table(metrics_payload: dict[str, Any], config_names: list[str]) -> str:
    headers = "| Metric | " + " | ".join(config_names) + " |"
    divider = "|---|" + "|".join("---:" for _ in config_names) + "|"
    rows = [headers, divider]
    metrics = (
        ("Source Hit@1", "source_hit_at_1"),
        ("Source Hit@3", "source_hit_at_3"),
        ("Source Hit@5", "source_hit_at_5"),
        ("Document Recall@5", "document_recall_at_5"),
        ("MRR@5", "mrr_at_5"),
        ("nDCG@5", "ndcg_at_5"),
        ("Evidence Coverage@5", "evidence_coverage_at_5"),
        ("Duplicate Context Rate@5", "duplicate_context_rate_at_5"),
        ("Empty Result Rate", "empty_result"),
        ("Latency p50 (ms)", "latency_p50_ms"),
        ("Latency p95 (ms)", "latency_p95_ms"),
    )
    for label, key in metrics:
        values = [
            _fmt(metrics_payload["configs"][config]["overall"].get(key), 1 if "Latency" in label else 3)
            for config in config_names
        ]
        rows.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(rows)


def _split_table(metrics_payload: dict[str, Any], config_names: list[str]) -> str:
    rows = [
        "| Split | Config | Cases | Hit@1 | Recall@5 | MRR@5 | nDCG@5 | Evidence@5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    split_names = sorted(
        {
            split
            for config in config_names
            for split in metrics_payload["configs"][config]["by_split"]
        }
    )
    for split in split_names:
        for config in config_names:
            values = metrics_payload["configs"][config]["by_split"].get(split)
            if not values:
                continue
            rows.append(
                f"| {split} | `{config}` | {values['case_count']} | "
                f"{_fmt(values.get('source_hit_at_1'))} | "
                f"{_fmt(values.get('document_recall_at_5'))} | "
                f"{_fmt(values.get('mrr_at_5'))} | "
                f"{_fmt(values.get('ndcg_at_5'))} | "
                f"{_fmt(values.get('evidence_coverage_at_5'))} |"
            )
    return "\n".join(rows)


def _ragas_table(ragas_payload: dict[str, Any] | None, config_names: list[str]) -> str:
    if not ragas_payload:
        return "Chưa chạy LLM judge; không tạo điểm RAGAS giả."
    ragas_configs = ragas_payload.get("configs", {})

    def case_count(name: str) -> int:
        config_payload = ragas_configs.get(name, {})
        return config_payload.get("case_count", len(config_payload.get("cases", [])))

    counts = ", ".join(
        f"`{name}`: {case_count(name)}" for name in config_names
    )
    selected_ids = ragas_payload.get("case_ids")
    subset_note = (
        f" Selected case IDs: {', '.join(f'`{case_id}`' for case_id in selected_ids)}. "
        "Treat these as smoke/diagnostic scores, not final full-corpus RAGAS scores."
        if selected_ids
        else ""
    )
    headers = "| Metric | " + " | ".join(config_names) + " |"
    divider = "|---|" + "|".join("---:" for _ in config_names) + "|"
    rows = [headers, divider]
    for label, key in (
        ("Faithfulness", "faithfulness"),
        ("Answer Relevance", "answer_relevancy"),
        ("Context Recall", "context_recall"),
        ("Context Precision", "context_precision"),
    ):
        values = []
        for config in config_names:
            config_payload = ragas_configs.get(config)
            if not config_payload:
                values.append("not run (0/0)")
                continue
            case_count = config_payload.get(
                "case_count", len(config_payload.get("cases", []))
            )
            valid_count = config_payload.get("valid_counts", {}).get(key)
            if valid_count is None:
                valid_count = sum(
                    isinstance(row.get(key), (int, float))
                    and math.isfinite(float(row[key]))
                    for row in config_payload.get("cases", [])
                )
            values.append(
                f"{_fmt(config_payload['overall'].get(key))} ({valid_count}/{case_count})"
            )
        rows.append(f"| {label} | " + " | ".join(values) + " |")
    return (
        f"Answerable cases evaluated: {counts}.{subset_note}\n\n"
        + "Each cell reports `mean (valid judgements/selected cases)`.\n\n"
        + "\n".join(rows)
    )


def _safety_table(safety_payload: dict[str, Any] | None, config_names: list[str]) -> str:
    if not safety_payload:
        return "Chưa chạy generation trên safety split."
    rows = [
        "| Metric | " + " | ".join(config_names) + " |",
        "|---|" + "|".join("---:" for _ in config_names) + "|",
    ]
    for label, key in (
        ("Refusal Accuracy", "refusal_accuracy"),
        ("Unsupported Answer Rate", "unsupported_answer_rate"),
    ):
        values = [
            _fmt(safety_payload.get("configs", {}).get(name, {}).get(key))
            for name in config_names
        ]
        rows.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(rows)


def export_results(
    run_dir: Path,
    manifest: dict[str, Any],
    retrieval_metrics: dict[str, Any],
    predictions: dict[str, Any],
    config_names: list[str],
    *,
    ragas_metrics: dict[str, Any] | None = None,
    safety_metrics: dict[str, Any] | None = None,
) -> None:
    """Export an evidence-backed report; absent stages are explicitly labelled."""

    resumed_stages = manifest.get("resumed_stages", [])

    def latest_model(key: str) -> str | None:
        return manifest.get(key) or next(
            (
                stage[key]
                for stage in reversed(resumed_stages)
                if stage.get(key)
            ),
            None,
        )

    generation_model = latest_model("generation_model")
    judge_model = latest_model("judge_model")
    judge_embedding_model = latest_model("judge_embedding_model")

    ranked_worst: list[tuple[float, str, str, str]] = []
    for case in predictions["cases"]:
        for config in config_names:
            metrics = case["configs"][config]["metrics"]
            quality = (
                float(metrics["document_recall_at_5"])
                + float(metrics["mrr_at_5"])
                + float(metrics["evidence_coverage_at_5"])
            ) / 3
            ranked_worst.append((quality, config, case["id"], case["question"]))
    ranked_worst.sort(key=lambda item: (item[0], item[1], item[2]))
    worst_rows = [
        f"| {case_id} | {config} | {_fmt(score)} | {question.replace('|', '/')} |"
        for score, config, case_id, question in ranked_worst[:5]
    ]

    winner = max(
        config_names,
        key=lambda name: (
            retrieval_metrics["configs"][name]["overall"].get("document_recall_at_5", 0),
            retrieval_metrics["configs"][name]["overall"].get("mrr_at_5", 0),
        ),
    )
    relative_run = run_dir.relative_to(PROJECT_DIR).as_posix()
    content = f"""# RAG Evaluation Results

## Run

- Timestamp: `{manifest['created_at']}`
- Git commit: `{manifest['git_commit']}`
- Git dirty: `{manifest['git_dirty']}`
- Corpus hash: `{manifest['corpus_hash']}`
- Golden dataset hash: `{manifest['golden_dataset_hash']}`
- Cases: `{len(predictions['cases'])}`
- Top-k: `{manifest['top_k']}`
- Answer generation model: `{generation_model or 'not run'}` via OpenRouter
- RAGAS LLM judge: `{judge_model or 'not run'}` via OpenRouter
- RAGAS embedding model: `{judge_embedding_model or 'not run'}` via OpenRouter
- Local raw artifacts (generated, gitignored): `{relative_run}/`

## Configurations

| Config | Description |
|---|---|
"""
    for name in config_names:
        content += f"| `{name}` | {CONFIG_DESCRIPTIONS[name]} |\n"
    content += f"""

## Retrieval Evaluation

{_retrieval_table(retrieval_metrics, config_names)}

Best retrieval configuration by Document Recall@5 then MRR@5: **`{winner}`**.

Metrics are deterministic source/document retrieval metrics. `Evidence Coverage@5` is a lexical proxy and is not presented as RAGAS Context Recall.

### Retrieval by split

{_split_table(retrieval_metrics, config_names)}

## Required RAGAS Metrics

{_ragas_table(ragas_metrics, config_names)}

## Safety

Safety cases are excluded from answerable RAGAS averages.

{_safety_table(safety_metrics, config_names)}

## Worst Retrieval Cases

| Case | Config | Composite retrieval score | Question |
|---|---|---:|---|
{chr(10).join(worst_rows)}

## Initial Recommendations

1. Use `{winner}` as the current evidence-backed retrieval baseline; do not select a configuration from one demo query.
2. Inspect the worst cases in `predictions.json` before changing chunk size, RRF weights or fallback threshold.
3. Re-run the same frozen golden dataset after Task 4 canonical metadata/index rebuild; compare by paired case ID.
4. Re-run all 26 answerable RAGAS cases after material pipeline changes, using `--resume` so retrieval stays frozen during judge retries.
5. Keep safety refusal metrics separate from answerable quality averages.
"""
    _atomic_write_text(RESULTS_PATH, content)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_run_dir(value: Path | None) -> Path:
    if value is not None:
        return value.resolve()
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return RUNS_DIR / run_id


def _load_existing_run(
    run_dir: Path,
    requested_configs: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    """Load frozen retrieval artifacts so LLM stages never retrieve twice."""

    required = (
        run_dir / "run_manifest.json",
        run_dir / "predictions.json",
        run_dir / "retrieval_metrics.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot resume {run_dir}; missing artifacts: {', '.join(missing)}"
        )
    manifest = json.loads(required[0].read_text(encoding="utf-8-sig"))
    predictions = json.loads(required[1].read_text(encoding="utf-8-sig"))
    retrieval_metrics = json.loads(required[2].read_text(encoding="utf-8-sig"))
    available = list(manifest.get("configs", {}))
    unavailable = [name for name in requested_configs if name not in available]
    if unavailable:
        raise ValueError(
            f"Run does not contain configurations: {', '.join(unavailable)}"
        )
    manifest.setdefault("resumed_stages", [])
    return manifest, predictions, retrieval_metrics, requested_configs


def main() -> None:
    # Some Windows terminals inherit cp1252 even when files are UTF-8. Keep
    # Vietnamese progress output readable without requiring shell-specific env.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("retrieval", "ragas", "safety", "all"),
        default="retrieval",
    )
    parser.add_argument("--validate", action="store_true", help="validate and exit")
    parser.add_argument(
        "--configs",
        default=",".join(SUPPORTED_CONFIGS),
        help=f"comma-separated: {', '.join(SUPPORTED_CONFIGS)}",
    )
    parser.add_argument("--split", default="all", help="core,challenge,safety or all")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--case-ids",
        help="comma-separated answerable case IDs for generation/RAGAS smoke subsets",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse retrieval artifacts in --run-dir for generation/RAGAS/safety",
    )
    parser.add_argument(
        "--generation-model",
        default=os.getenv("EVAL_GENERATION_MODEL", "openai/gpt-4o-mini"),
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-4o-mini"),
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=int(os.getenv("EVAL_JUDGE_MAX_TOKENS", "1024")),
        help="maximum output tokens per RAGAS judge call",
    )
    parser.add_argument(
        "--judge-max-workers",
        type=int,
        default=int(os.getenv("EVAL_JUDGE_MAX_WORKERS", "4")),
        help="maximum concurrent RAGAS judge jobs",
    )
    parser.add_argument(
        "--judge-embedding-model",
        default=os.getenv("EVAL_EMBEDDING_MODEL", "BAAI/bge-m3"),
    )
    args = parser.parse_args()

    cases = load_golden_dataset()
    validation = validate_golden_dataset(cases)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if args.validate:
        return

    config_names = _parse_csv(args.configs)
    validate_config_names(config_names)
    ragas_case_ids = set(_parse_csv(args.case_ids)) if args.case_ids else None
    run_dir = _resolve_run_dir(args.run_dir)
    if args.resume:
        if args.run_dir is None:
            parser.error("--resume requires --run-dir")
        manifest, predictions, retrieval_metrics, config_names = _load_existing_run(
            run_dir, config_names
        )
        available_case_ids = {case["id"] for case in predictions["cases"]}
        missing_case_ids = sorted((ragas_case_ids or set()) - available_case_ids)
        if missing_case_ids:
            raise ValueError(
                f"Run does not contain case IDs: {', '.join(missing_case_ids)}"
            )
        manifest["resumed_stages"].append(
            {
                "mode": args.mode,
                "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "generation_model": args.generation_model,
                "judge_model": args.judge_model if args.mode in {"ragas", "all"} else None,
                "judge_max_tokens": (
                    args.judge_max_tokens if args.mode in {"ragas", "all"} else None
                ),
                "judge_max_workers": (
                    args.judge_max_workers if args.mode in {"ragas", "all"} else None
                ),
                "judge_embedding_model": (
                    args.judge_embedding_model if args.mode in {"ragas", "all"} else None
                ),
            }
        )
    else:
        splits = _parse_csv(args.split)
        if args.mode == "safety":
            splits = ["safety"]
        selected = select_cases(cases, splits, args.limit)
        if not selected:
            raise ValueError("No cases selected")
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_run_manifest(
            selected,
            config_names,
            top_k=args.top_k,
            mode=args.mode,
            generation_model=args.generation_model,
            judge_model=args.judge_model,
            judge_embedding_model=args.judge_embedding_model,
        )
        predictions, retrieval_metrics = run_retrieval_evaluation(
            selected, config_names, top_k=args.top_k
        )
        predictions["created_at"] = manifest["created_at"]

    _atomic_write_json(run_dir / "run_manifest.json", manifest)
    _atomic_write_json(run_dir / "predictions.json", predictions)
    _atomic_write_json(run_dir / "retrieval_metrics.json", retrieval_metrics)

    ragas_path = run_dir / "ragas_metrics.json"
    safety_path = run_dir / "safety_metrics.json"
    ragas_payload = (
        json.loads(ragas_path.read_text(encoding="utf-8-sig"))
        if ragas_path.is_file()
        else None
    )
    safety_payload = (
        json.loads(safety_path.read_text(encoding="utf-8-sig"))
        if safety_path.is_file()
        else None
    )
    if args.mode in {"safety", "all"}:
        generate_answers(
            predictions,
            config_names,
            model=args.generation_model,
            answerable=False if args.mode == "safety" else None,
        )
        _atomic_write_json(run_dir / "predictions.json", predictions)
        safety_payload = evaluate_safety(predictions, config_names)
        _atomic_write_json(safety_path, safety_payload)
    if args.mode in {"ragas", "all"}:
        generate_answers(
            predictions,
            config_names,
            model=args.generation_model,
            answerable=True,
            case_ids=ragas_case_ids,
        )
        _atomic_write_json(run_dir / "predictions.json", predictions)
        ragas_payload = evaluate_with_ragas(
            predictions,
            config_names,
            judge_model=args.judge_model,
            embedding_model=args.judge_embedding_model,
            judge_max_tokens=args.judge_max_tokens,
            judge_max_workers=args.judge_max_workers,
            case_ids=ragas_case_ids,
        )
        ragas_payload = _merge_ragas_payload(
            json.loads(ragas_path.read_text(encoding="utf-8-sig"))
            if ragas_path.is_file()
            else None,
            ragas_payload,
        )
        _atomic_write_json(ragas_path, ragas_payload)

    report_config_names = (
        list(manifest.get("configs", {})) if args.resume else config_names
    )
    export_results(
        run_dir,
        manifest,
        retrieval_metrics,
        predictions,
        report_config_names,
        ragas_metrics=ragas_payload,
        safety_metrics=safety_payload,
    )
    print(f"Evaluation complete: {run_dir}")
    print(f"Report updated: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
