from __future__ import annotations

"""
Production CLI for final submission construction.

This script is intentionally strict. It:
- validates prediction artifacts before reading them into the submission boundary
- uses the online submission formatter contract wherever possible
- enforces deterministic ordering and exact Kaggle schema (id,answer)
- writes manifest/provenance/summary sidecars for auditability
- never silently drops malformed rows
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.constants import OFFLINE_ARTIFACT_VERSION
from src.common.schemas import FinalPrediction
from src.online.kaggle_runner import KaggleProblemRecord
from src.online.submission_formatter import (
    SubmissionDebugRow,
    SubmissionFormattingError,
    SubmissionFormatterConfig,
    SubmissionRow,
    format_debug_row,
    format_submission_batch,
    render_debug_sidecar_csv,
    render_submission_csv,
    sort_debug_rows,
    sort_submission_rows,
    submission_rows_from_records,
)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


SCHEMA_VERSION = "build_submission.v1"
ARTIFACT_KIND = "submission_build"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class SubmissionBuildError(RuntimeError):
    """Raised when submission construction cannot proceed safely."""


class SourceArtifact(StrictModel):
    path: str
    kind: str
    record_count: int = Field(ge=0)
    digest: str
    source_manifest_path: str | None = None
    source_manifest_digest: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReproducibilityMetadata(StrictModel):
    python_version: str
    platform: str
    cwd: str
    script_path: str
    timestamp_utc: str
    input_paths: tuple[str, ...]
    expected_ids_path: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)


class BuildSummary(StrictModel):
    row_count: int = Field(ge=0)
    expected_problem_count: int | None = Field(default=None, ge=0)
    prediction_count: int = Field(ge=0)
    prebuilt_row_count: int = Field(ge=0)
    duplicate_problem_ids: tuple[str, ...] = Field(default_factory=tuple)
    problem_ids: tuple[str, ...] = Field(default_factory=tuple)
    min_answer: int | None = None
    max_answer: int | None = None
    output_digest: str
    debug_row_count: int = Field(ge=0)


class SubmissionBuildManifest(StrictModel):
    artifact_kind: str = ARTIFACT_KIND
    artifact_version: str = OFFLINE_ARTIFACT_VERSION
    schema_version: str = SCHEMA_VERSION
    run_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    sources: tuple[SourceArtifact, ...] = Field(default_factory=tuple)
    output_csv_path: str
    debug_csv_path: str | None = None
    summary_path: str
    expected_ids_path: str | None = None
    reproducibility: ReproducibilityMetadata
    summary: BuildSummary


class SubmissionBuildStats(StrictModel):
    prediction_count: int = Field(default=0, ge=0)
    prebuilt_row_count: int = Field(default=0, ge=0)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_hash(prefix: str, payload: Any) -> str:
    token = f"{prefix}::{_stable_json(payload)}"
    return f"{prefix}_{hashlib.sha1(token.encode('utf-8')).hexdigest()[:16]}"


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _environment_snapshot() -> dict[str, str]:
    keys = [
        "PYTHONHASHSEED",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _repro_metadata(
    input_paths: Sequence[str],
    *,
    expected_ids_path: str | None,
) -> ReproducibilityMetadata:
    return ReproducibilityMetadata(
        python_version=sys.version.replace("\n", " "),
        platform=platform.platform(),
        cwd=str(Path.cwd()),
        script_path=str(Path(__file__).resolve()),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        input_paths=tuple(input_paths),
        expected_ids_path=expected_ids_path,
        environment=_environment_snapshot(),
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SubmissionBuildError(f"Invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SubmissionBuildError(f"Invalid JSONL row in {path} line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise SubmissionBuildError(f"Non-object JSONL row in {path} line {line_no}")
            rows.append(payload)
    return rows


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SubmissionBuildError(f"CSV file has no header: {path}")
        return [dict(row) for row in reader]


def _candidate_artifact_files(directory: Path, manifest_payload: Mapping[str, Any] | None) -> list[Path]:
    candidates: list[Path] = []

    def add_candidate(raw: Any) -> None:
        if not raw:
            return
        try:
            candidate = Path(str(raw))
        except Exception:
            return
        if not candidate.is_absolute():
            candidate = directory / candidate
        candidates.append(candidate)

    if manifest_payload is not None:
        for key in (
            "predictions_path",
            "prediction_path",
            "final_predictions_path",
            "results_path",
            "results_jsonl_path",
            "submission_rows_path",
            "submission_csv_path",
            "output_csv_path",
        ):
            if key in manifest_payload:
                add_candidate(manifest_payload[key])

    for filename in (
        "final_predictions.jsonl",
        "predictions.jsonl",
        "results.jsonl",
        "submission_rows.jsonl",
        "final_predictions.json",
        "predictions.json",
        "results.json",
        "submission_rows.json",
        "submission.csv",
    ):
        add_candidate(filename)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def _load_directory_artifact(path: Path) -> tuple[list[dict[str, Any]], SourceArtifact]:
    manifest_path = path / "manifest.json"
    manifest_payload: dict[str, Any] | None = None
    manifest_digest: str | None = None

    if manifest_path.exists():
        payload = _read_json(manifest_path)
        if not isinstance(payload, dict):
            raise SubmissionBuildError(f"Expected object-like manifest in {manifest_path}")
        manifest_payload = payload
        manifest_digest = _sha1_text(_stable_json(payload))

    for candidate in _candidate_artifact_files(path, manifest_payload):
        if candidate.exists() and candidate.is_file():
            records, kind = _load_records_from_file(candidate)
            digest = _sha1_text(_stable_json({"records": records, "kind": kind}))
            return records, SourceArtifact(
                path=str(path),
                kind=f"directory:{kind}",
                record_count=len(records),
                digest=digest,
                source_manifest_path=str(manifest_path) if manifest_path.exists() else None,
                source_manifest_digest=manifest_digest,
                metadata={
                    "resolved_file": str(candidate),
                    "manifest_present": manifest_path.exists(),
                },
            )

    raise SubmissionBuildError(
        f"Directory artifact {path} did not contain a supported prediction file or discoverable manifest pointer"
    )


def _load_records_from_file(path: Path) -> tuple[list[dict[str, Any]], str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path), "jsonl"
    if suffix == ".json":
        payload = _read_json(path)
        if isinstance(payload, list):
            rows = []
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    raise SubmissionBuildError(f"JSON list item {idx} in {path} is not an object")
                rows.append(item)
            return rows, "json"
        if isinstance(payload, dict):
            if "records" in payload and isinstance(payload["records"], list):
                rows = []
                for idx, item in enumerate(payload["records"]):
                    if not isinstance(item, dict):
                        raise SubmissionBuildError(f"JSON records[{idx}] in {path} is not an object")
                    rows.append(item)
                return rows, "json_records"
            if "submission_rows" in payload and isinstance(payload["submission_rows"], list):
                rows = []
                for idx, item in enumerate(payload["submission_rows"]):
                    if not isinstance(item, dict):
                        raise SubmissionBuildError(f"JSON submission_rows[{idx}] in {path} is not an object")
                    rows.append(item)
                return rows, "json_submission_rows"
            if "final_prediction" in payload or "submission_row" in payload or {"problem_id", "final_answer"} <= set(payload):
                return [payload], "json_single_record"
        raise SubmissionBuildError(f"Unsupported JSON artifact layout in {path}")
    if suffix == ".csv":
        return _read_csv_records(path), "csv"
    raise SubmissionBuildError(f"Unsupported artifact file type: {path}")


def _load_source_artifact(path_text: str) -> tuple[list[dict[str, Any]], SourceArtifact]:
    path = Path(path_text)
    if not path.exists():
        raise SubmissionBuildError(f"Prediction artifact path does not exist: {path_text}")

    if path.is_dir():
        return _load_directory_artifact(path)
    if path.name == "manifest.json":
        return _load_directory_artifact(path.parent)

    records, kind = _load_records_from_file(path)
    digest = _sha1_text(_stable_json({"records": records, "kind": kind}))
    return records, SourceArtifact(
        path=str(path),
        kind=kind,
        record_count=len(records),
        digest=digest,
        source_manifest_path=None,
        source_manifest_digest=None,
        metadata={},
    )


def _unwrap_prediction_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if "final_prediction" in record and isinstance(record["final_prediction"], Mapping):
        return _unwrap_prediction_record(record["final_prediction"])
    if "prediction" in record and isinstance(record["prediction"], Mapping):
        inner = record["prediction"]
        if {"problem_id", "final_answer"} <= set(inner.keys()):
            return inner
    return record


def _coerce_final_prediction(record: Mapping[str, Any]) -> FinalPrediction:
    payload = dict(_unwrap_prediction_record(record))
    try:
        return FinalPrediction.model_validate(payload)
    except ValidationError as exc:
        raise SubmissionBuildError(
            "Malformed prediction record: expected FinalPrediction-compatible payload. "
            f"problem_id={payload.get('problem_id')!r} error={exc}"
        ) from exc


def _looks_like_submission_row(record: Mapping[str, Any]) -> bool:
    return "id" in record and "answer" in record


def _looks_like_prediction(record: Mapping[str, Any]) -> bool:
    record = _unwrap_prediction_record(record)
    return "problem_id" in record and "final_answer" in record


def _load_expected_problem_ids(path_text: str | None) -> list[str] | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        raise SubmissionBuildError(f"Expected-ids path does not exist: {path_text}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _read_csv_records(path)
        if not rows:
            return []
        header = set(rows[0].keys())
        id_key = "id" if "id" in header else None
        if id_key is None:
            raise SubmissionBuildError(f"Expected-ids CSV must contain an 'id' column: {path}")
        out: list[str] = []
        for idx, row in enumerate(rows, start=1):
            value = str(row.get(id_key, "")).strip()
            if not value:
                raise SubmissionBuildError(f"Blank id at row {idx} in expected-ids CSV {path}")
            out.append(value)
        return out

    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows = _read_jsonl(path)
            out = []
            for idx, row in enumerate(rows, start=1):
                if "id" not in row:
                    raise SubmissionBuildError(f"Expected-ids JSONL row {idx} missing 'id' in {path}")
                out.append(str(row["id"]).strip())
            return out

        payload = _read_json(path)
        if isinstance(payload, list):
            out = []
            for idx, item in enumerate(payload, start=1):
                if isinstance(item, Mapping):
                    if "id" not in item:
                        raise SubmissionBuildError(f"Expected-ids JSON item {idx} missing 'id' in {path}")
                    out.append(str(item["id"]).strip())
                else:
                    out.append(str(item).strip())
            return out

        raise SubmissionBuildError(f"Unsupported expected-ids JSON structure in {path}")

    raise SubmissionBuildError(f"Unsupported expected-ids file type: {path}")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_output_targets_clear(paths: Sequence[Path], *, overwrite: bool) -> None:
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(path)
            seen.add(resolved)

    existing = [path for path in unique_paths if path.exists()]
    if existing and not overwrite:
        raise SubmissionBuildError(
            "One or more output targets already exist. "
            f"Re-run with --overwrite to replace them: {[str(path) for path in existing]}"
        )

    if overwrite:
        for path in existing:
            if path.is_dir():
                raise SubmissionBuildError(f"Refusing to overwrite directory output target: {path}")


def _format_debug_for_prebuilt_rows(rows: Sequence[SubmissionRow]) -> list[SubmissionDebugRow]:
    out: list[SubmissionDebugRow] = []
    for row in rows:
        out.append(
            SubmissionDebugRow(
                problem_id=row.id,
                submission_answer=row.answer,
                confidence=0.0,
                method_used="prebuilt_submission_row",
                winning_answer_raw=str(row.answer),
                winning_answer_canonical=str(row.answer),
                num_branches_generated=0,
                num_branches_survived=0,
                solve_time_sec=0.0,
                warnings=["constructed_from_prebuilt_row"],
                rationale_summary="Prebuilt submission row supplied directly to build_submission.py",
                metadata={},
            )
        )
    return out


def _merge_submission_rows(rows: Sequence[SubmissionRow]) -> list[SubmissionRow]:
    deduped: dict[str, SubmissionRow] = {}
    duplicates: list[str] = []
    for row in rows:
        if row.id in deduped:
            duplicates.append(row.id)
        deduped[row.id] = row
    if duplicates:
        dupes = tuple(sorted(set(duplicates)))
        raise SubmissionBuildError(f"Duplicate submission problem ids detected across sources: {list(dupes)}")
    return sort_submission_rows(list(deduped.values()))


def _merge_debug_rows(rows: Sequence[SubmissionDebugRow]) -> list[SubmissionDebugRow]:
    deduped: dict[str, SubmissionDebugRow] = {}
    duplicates: list[str] = []
    for row in rows:
        if row.problem_id in deduped:
            duplicates.append(row.problem_id)
        deduped[row.problem_id] = row
    if duplicates:
        dupes = tuple(sorted(set(duplicates)))
        raise SubmissionBuildError(f"Duplicate debug problem ids detected across sources: {list(dupes)}")
    return sort_debug_rows(list(deduped.values()))


def _build_submission(
    *,
    input_paths: Sequence[str],
    expected_problem_ids: Sequence[str] | None,
    include_debug_sidecar: bool,
) -> tuple[list[SubmissionRow], list[SubmissionDebugRow], tuple[SourceArtifact, ...], SubmissionBuildStats]:
    source_artifacts: list[SourceArtifact] = []
    all_prediction_records: list[FinalPrediction] = []
    all_prebuilt_row_records: list[dict[str, Any]] = []

    for path_text in input_paths:
        records, source_artifact = _load_source_artifact(path_text)
        source_artifacts.append(source_artifact)

        for idx, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                raise SubmissionBuildError(
                    f"Non-object record in source {path_text} at index {idx}: {type(record)!r}"
                )

            if "submission_row" in record and isinstance(record["submission_row"], Mapping):
                inner = dict(record["submission_row"])
                if not _looks_like_submission_row(inner):
                    raise SubmissionBuildError(
                        f"Malformed submission_row wrapper in source {path_text} index {idx}: "
                        f"keys={sorted(inner.keys())}"
                    )
                all_prebuilt_row_records.append(inner)
                continue

            if _looks_like_submission_row(record):
                all_prebuilt_row_records.append(dict(record))
                continue

            if _looks_like_prediction(record):
                all_prediction_records.append(_coerce_final_prediction(record))
                continue

            if "status" in record and "problem_id" in record and "final_prediction" not in record:
                raise SubmissionBuildError(
                    f"Malformed runtime-result record in source {path_text} index {idx}: "
                    "status/problem_id present but no usable final_prediction or submission_row payload"
                )

            raise SubmissionBuildError(
                f"Malformed prediction record in source {path_text} index {idx}: "
                f"keys={sorted(record.keys())}"
            )

    formatter_config = SubmissionFormatterConfig(
        enforce_answer_range=True,
        include_debug_sidecar=include_debug_sidecar,
        sort_rows=True,
        require_unique_problem_ids=True,
        strict_problem_id=True,
    )

    built_rows: list[SubmissionRow] = []
    built_debug_rows: list[SubmissionDebugRow] = []

    if all_prediction_records:
        bundle = format_submission_batch(
            all_prediction_records,
            expected_problem_ids=expected_problem_ids,
            config=formatter_config,
        )
        built_rows.extend(bundle.rows)
        built_debug_rows.extend(bundle.debug_rows)

    if all_prebuilt_row_records:
        rows = submission_rows_from_records(all_prebuilt_row_records)
        built_rows.extend(rows)
        if include_debug_sidecar:
            built_debug_rows.extend(_format_debug_for_prebuilt_rows(rows))

    merged_rows = _merge_submission_rows(built_rows)
    if expected_problem_ids is not None:
        actual_ids = [row.id for row in merged_rows]
        expected_ids = [str(item).strip() for item in expected_problem_ids]
        if len(actual_ids) != len(expected_ids):
            raise SubmissionBuildError(
                f"Submission row count {len(actual_ids)} does not match expected count {len(expected_ids)}"
            )
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        if missing or extra:
            raise SubmissionBuildError(f"Submission problem-id mismatch. missing={missing} extra={extra}")

    merged_debug_rows = _merge_debug_rows(built_debug_rows) if include_debug_sidecar else []
    return (
        merged_rows,
        merged_debug_rows,
        tuple(source_artifacts),
        SubmissionBuildStats(
            prediction_count=len(all_prediction_records),
            prebuilt_row_count=len(all_prebuilt_row_records),
        ),
    )


def _build_summary(
    rows: Sequence[SubmissionRow],
    debug_rows: Sequence[SubmissionDebugRow],
    *,
    expected_problem_ids: Sequence[str] | None,
    stats: SubmissionBuildStats,
) -> BuildSummary:
    ordered_rows = sort_submission_rows(list(rows))
    answers = [row.answer for row in ordered_rows]
    digest = _sha1_text(_stable_json([row.model_dump() for row in ordered_rows]))
    return BuildSummary(
        row_count=len(ordered_rows),
        expected_problem_count=(len(expected_problem_ids) if expected_problem_ids is not None else None),
        prediction_count=stats.prediction_count,
        prebuilt_row_count=stats.prebuilt_row_count,
        duplicate_problem_ids=tuple(),
        problem_ids=tuple(row.id for row in ordered_rows),
        min_answer=(min(answers) if answers else None),
        max_answer=(max(answers) if answers else None),
        output_digest=digest,
        debug_row_count=len(debug_rows),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic Kaggle submission from validated prediction artifacts")
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Prediction artifact path. Repeatable. Supports manifest-backed directories, JSONL, JSON, or prebuilt submission CSV.",
    )
    parser.add_argument("--output-csv", required=True, help="Path to output submission CSV")
    parser.add_argument("--debug-csv", default=None, help="Optional path to debug sidecar CSV")
    parser.add_argument("--summary-json", default=None, help="Optional path to summary JSON")
    parser.add_argument("--manifest-json", default=None, help="Optional path to manifest JSON")
    parser.add_argument(
        "--expected-ids-csv",
        default=None,
        help="Optional Kaggle input CSV (or JSON/JSONL id list) used to enforce exact submission ids",
    )
    parser.add_argument(
        "--include-debug-sidecar",
        action="store_true",
        help="Emit debug rows derived from FinalPrediction inputs and prebuilt rows",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files instead of failing on collisions",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        output_csv = Path(args.output_csv)
        debug_csv = Path(args.debug_csv) if args.debug_csv else None
        summary_json = Path(args.summary_json) if args.summary_json else output_csv.with_suffix(".summary.json")
        manifest_json = Path(args.manifest_json) if args.manifest_json else output_csv.with_suffix(".manifest.json")
        output_targets = [output_csv, summary_json, manifest_json]
        if debug_csv is not None:
            output_targets.append(debug_csv)
        _ensure_output_targets_clear(output_targets, overwrite=bool(args.overwrite))

        expected_problem_ids = _load_expected_problem_ids(args.expected_ids_csv)

        rows, debug_rows, sources, stats = _build_submission(
            input_paths=args.predictions,
            expected_problem_ids=expected_problem_ids,
            include_debug_sidecar=bool(args.include_debug_sidecar or debug_csv is not None),
        )

        submission_csv_text = render_submission_csv(rows)
        _write_text(output_csv, submission_csv_text)

        if debug_csv is not None:
            debug_csv_text = render_debug_sidecar_csv(debug_rows)
            _write_text(debug_csv, debug_csv_text)

        summary = _build_summary(rows, debug_rows, expected_problem_ids=expected_problem_ids, stats=stats)
        _write_json(summary_json, summary.model_dump(mode="json"))

        run_id = _stable_hash(
            "submission_build",
            {
                "inputs": list(args.predictions),
                "output_csv": str(output_csv),
                "summary_digest": summary.output_digest,
                "expected_ids_path": args.expected_ids_csv,
            },
        )
        manifest = SubmissionBuildManifest(
            artifact_version=OFFLINE_ARTIFACT_VERSION,
            run_id=run_id,
            config={
                "input_paths": list(args.predictions),
                "include_debug_sidecar": bool(args.include_debug_sidecar or debug_csv is not None),
            },
            sources=sources,
            output_csv_path=str(output_csv),
            debug_csv_path=(str(debug_csv) if debug_csv is not None else None),
            summary_path=str(summary_json),
            expected_ids_path=args.expected_ids_csv,
            reproducibility=_repro_metadata(args.predictions, expected_ids_path=args.expected_ids_csv),
            summary=summary,
        )
        _write_json(manifest_json, manifest.model_dump(mode="json"))

        print(f"[build_submission] run_id: {manifest.run_id}")
        print(f"[build_submission] rows: {summary.row_count}")
        if summary.expected_problem_count is not None:
            print(f"[build_submission] expected rows: {summary.expected_problem_count}")
        print(f"[build_submission] output csv: {output_csv}")
        if debug_csv is not None:
            print(f"[build_submission] debug csv: {debug_csv}")
        print(f"[build_submission] summary: {summary_json}")
        print(f"[build_submission] manifest: {manifest_json}")
        return 0

    except (
        SubmissionBuildError,
        SubmissionFormattingError,
        ValidationError,
        FileNotFoundError,
        ValueError,
        TypeError,
    ) as exc:
        print(f"[build_submission] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())