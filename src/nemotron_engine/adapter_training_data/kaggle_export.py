"""Export Sprint-3 curated data as a Kaggle input dataset."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import csv
import json
from pathlib import Path
import shutil
import zipfile
from typing import Any

from nemotron_engine.core.schemas import stable_hash


class Sprint4ExportError(ValueError):
    """Raised when Sprint-4 Kaggle export/config validation fails."""


REQUIRED_CURATED_FILES = (
    "train_direct.jsonl",
    "train_family_tagged.jsonl",
    "train_solver_distilled.jsonl",
    "train_hard_oversampled.jsonl",
    "validation_family_balanced.jsonl",
    "dataset_manifest.json",
)

KNOWN_VARIANTS = frozenset({"variant_a_family_tagged", "variant_b_distilled", "variant_c_hard_oversampled", "variant_d_direct_hard"})


@dataclass(frozen=True)
class Sprint4Config:
    parent_adapter_path: str | None
    curated_dataset_path: str
    output_adapter_dir: str
    submission_zip_path: str
    variant_name: str
    max_train_rows: int
    dataset_mixture: tuple[str, ...]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    num_epochs: float
    max_seq_len: int
    batch_size: int
    grad_accum: int
    seed: int
    config_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "curated_dataset_path", _text(self.curated_dataset_path, "curated_dataset_path"))
        object.__setattr__(self, "output_adapter_dir", _text(self.output_adapter_dir, "output_adapter_dir"))
        object.__setattr__(self, "submission_zip_path", _text(self.submission_zip_path, "submission_zip_path"))
        object.__setattr__(self, "variant_name", _text(self.variant_name, "variant_name"))
        if self.variant_name not in KNOWN_VARIANTS:
            raise Sprint4ExportError(f"unknown variant_name: {self.variant_name}")
        mixture = tuple(_text(item, "dataset_mixture") for item in self.dataset_mixture)
        if not mixture:
            raise Sprint4ExportError("dataset_mixture must be non-empty.")
        object.__setattr__(self, "dataset_mixture", mixture)
        rank = int(self.lora_rank)
        if rank <= 0 or rank > 32:
            raise Sprint4ExportError("lora_rank must be between 1 and 32.")
        object.__setattr__(self, "lora_rank", rank)
        for field_name in ("lora_alpha", "max_seq_len", "batch_size", "grad_accum", "seed"):
            value = int(getattr(self, field_name))
            if field_name != "seed" and value <= 0:
                raise Sprint4ExportError(f"{field_name} must be positive.")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "max_train_rows", int(self.max_train_rows))
        object.__setattr__(self, "lora_dropout", float(self.lora_dropout))
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "num_epochs", float(self.num_epochs))
        if self.parent_adapter_path is not None:
            object.__setattr__(self, "parent_adapter_path", str(self.parent_adapter_path))
        _set_or_check_hash(self, "config_hash")


def load_sprint4_config(path: str | Path, *, require_dataset_path_exists: bool = True) -> Sprint4Config:
    """Load the simple Sprint-4 YAML config without requiring PyYAML."""

    payload = _load_simple_yaml(Path(path))
    config = Sprint4Config(
        parent_adapter_path=_none_if_empty(payload.get("parent_adapter_path")),
        curated_dataset_path=str(payload.get("curated_dataset_path", "")),
        output_adapter_dir=str(payload.get("output_adapter_dir", "")),
        submission_zip_path=str(payload.get("submission_zip_path", "")),
        variant_name=str(payload.get("variant_name", "")),
        max_train_rows=int(payload.get("max_train_rows", 0)),
        dataset_mixture=tuple(payload.get("dataset_mixture", ())),
        lora_rank=int(payload.get("lora_rank", 0)),
        lora_alpha=int(payload.get("lora_alpha", 0)),
        lora_dropout=float(payload.get("lora_dropout", 0.0)),
        learning_rate=float(payload.get("learning_rate", 0.0)),
        num_epochs=float(payload.get("num_epochs", 0.0)),
        max_seq_len=int(payload.get("max_seq_len", 0)),
        batch_size=int(payload.get("batch_size", 0)),
        grad_accum=int(payload.get("grad_accum", 0)),
        seed=int(payload.get("seed", 0)),
    )
    if require_dataset_path_exists and not Path(config.curated_dataset_path).exists():
        raise Sprint4ExportError(f"curated_dataset_path does not exist: {config.curated_dataset_path}")
    return config


def export_sprint4_kaggle_input(
    source_dir: str | Path = "artifacts/adapter_training",
    output_dir: str | Path = "artifacts/sprint4_kaggle_input",
    zip_path: str | Path = "artifacts/sprint4_kaggle_input.zip",
    *,
    test_csv_path: str | Path = "data/nemotron_competition/test.csv",
) -> dict[str, Any]:
    """Copy curated artifacts into a Kaggle input folder and zip it."""

    source = Path(source_dir)
    output = Path(output_dir)
    target_zip = Path(zip_path)
    if not source.exists():
        raise Sprint4ExportError(f"source_dir does not exist: {source}")
    output.mkdir(parents=True, exist_ok=True)
    test_ids = _read_test_ids(Path(test_csv_path))
    copied_counts: dict[str, int] = {}
    copied_hashes: dict[str, str] = {}
    excluded_counts: dict[str, int] = {}

    for filename in REQUIRED_CURATED_FILES:
        src = source / filename
        if not src.exists():
            raise Sprint4ExportError(f"missing curated artifact: {filename}")
        dst = output / filename
        if filename.endswith(".jsonl"):
            kept, excluded = _copy_jsonl_filter_ids(src, dst, test_ids)
            copied_counts[filename] = kept
            excluded_counts[filename] = excluded
        else:
            shutil.copyfile(src, dst)
            copied_counts[filename] = 1
            excluded_counts[filename] = 0
        copied_hashes[filename] = _sha256_bytes(dst.read_bytes())

    readme = output / "README_SPRINT4_INPUT.md"
    readme.write_text(
        "Sprint-4 Kaggle input dataset\n\n"
        "Upload this folder or artifacts/sprint4_kaggle_input.zip as a Kaggle Dataset.\n"
        "It contains curated SFT JSONL files from train.csv only. Rows whose id appears in local test.csv are filtered out.\n"
        "This is not the competition submission.zip.\n",
        encoding="utf-8",
    )
    copied_hashes[readme.name] = _sha256_bytes(readme.read_bytes())
    copied_counts[readme.name] = 1
    excluded_counts[readme.name] = 0

    export_manifest = {
        "source_dir": str(source),
        "output_dir": str(output),
        "zip_path": str(target_zip),
        "copied_counts": copied_counts,
        "excluded_test_id_counts": excluded_counts,
        "artifact_hashes": copied_hashes,
        "test_ids_filtered_count": len(test_ids),
        "export_hash": stable_hash({"copied_counts": copied_counts, "excluded_counts": excluded_counts, "hashes": copied_hashes}),
    }
    (output / "sprint4_export_manifest.json").write_text(json.dumps(export_manifest, sort_keys=True, indent=2), encoding="utf-8")
    _write_zip_from_dir(output, target_zip)
    return export_manifest


def _copy_jsonl_filter_ids(src: Path, dst: Path, excluded_ids: set[str]) -> tuple[int, int]:
    kept = 0
    excluded = 0
    with src.open("r", encoding="utf-8") as handle, dst.open("w", encoding="utf-8", newline="\n") as out:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Sprint4ExportError(f"invalid JSONL in {src} line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise Sprint4ExportError(f"non-object JSONL row in {src} line {line_no}")
            if str(payload.get("problem_id", "")) in excluded_ids:
                excluded += 1
                continue
            out.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
            kept += 1
    return kept, excluded


def _read_test_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {str(row.get("id") or row.get("problem_id")) for row in reader if row.get("id") or row.get("problem_id")}


def _write_zip_from_dir(source: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Sprint4ExportError(f"config path does not exist: {path}")
    payload: dict[str, Any] = {}
    current_list: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_list:
            payload[current_list].append(_parse_scalar(line[4:].strip()))
            continue
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                payload[key] = []
                current_list = key
            else:
                payload[key] = _parse_scalar(value)
                current_list = None
    return payload


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"null", "none", ""}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(ch in value.lower() for ch in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _none_if_empty(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _set_or_check_hash(instance: object, hash_field: str) -> None:
    expected = stable_hash({item.name: getattr(instance, item.name) for item in fields(instance) if item.name != hash_field})
    current = getattr(instance, hash_field)
    if not current:
        object.__setattr__(instance, hash_field, expected)
    elif current != expected:
        raise Sprint4ExportError(f"{hash_field} does not match payload.")


def _text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise Sprint4ExportError(f"{field_name} must be non-empty.")
    return text


__all__ = [
    "KNOWN_VARIANTS",
    "REQUIRED_CURATED_FILES",
    "Sprint4Config",
    "Sprint4ExportError",
    "export_sprint4_kaggle_input",
    "load_sprint4_config",
]


def main() -> None:
    manifest = export_sprint4_kaggle_input()
    print(json.dumps(manifest, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
