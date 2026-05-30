from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

try:
    from kaggle_behavioral_eval import BehavioralEvalError, BehavioralGateConfig, load_report, validate_behavioral_report
except ImportError:  # pragma: no cover - supports package-style local tests
    from kaggle_sprint4.kaggle_behavioral_eval import BehavioralEvalError, BehavioralGateConfig, load_report, validate_behavioral_report


class AdapterPackagingError(ValueError):
    pass


FORBIDDEN_SUFFIXES = (".csv", ".jsonl", ".pt", ".pth", ".ckpt")
FORBIDDEN_NAMES = {"optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin", "pytorch_model.bin"}
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")


def validate_adapter_dir(adapter_dir: str | Path) -> tuple[Path, Path]:
    root = Path(adapter_dir)
    config = root / "adapter_config.json"
    model = root / "adapter_model.safetensors"
    if not config.is_file():
        raise AdapterPackagingError(f"missing adapter_config.json: {config}")
    if not model.is_file():
        raise AdapterPackagingError(f"missing adapter_model.safetensors: {model}")
    payload = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AdapterPackagingError("adapter_config.json must be a JSON object")
    rank = _extract_rank(payload)
    if rank is not None and rank > 32:
        raise AdapterPackagingError(f"LoRA rank exceeds 32: {rank}")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name in FORBIDDEN_NAMES or name.endswith(FORBIDDEN_SUFFIXES) and name not in REQUIRED_FILES:
            raise AdapterPackagingError(f"forbidden file in adapter dir: {path.relative_to(root)}")
    return config, model


def validate_behavior_gate(
    behavioral_report_path: str | Path | None,
    *,
    allow_no_behavioral_eval: bool = False,
    baseline_adapter_only: bool = False,
    min_child_exact_match_to_package: float = 0.0,
    max_allowed_regression: float = 0.0,
    max_family_regression: float = 0.10,
) -> None:
    if baseline_adapter_only:
        print("behavior_gate=baseline_adapter_only")
        return
    if behavioral_report_path is None:
        if allow_no_behavioral_eval:
            print("behavior_gate=bypassed_no_behavioral_eval")
            return
        raise AdapterPackagingError("behavioral eval report is required before packaging")
    path = Path(behavioral_report_path)
    if not path.is_file():
        if allow_no_behavioral_eval:
            print(f"behavior_gate=bypassed_missing_report path={path}")
            return
        raise AdapterPackagingError(f"missing behavioral eval report: {path}")
    try:
        report = load_report(path)
        validate_behavioral_report(
            report,
            gate=BehavioralGateConfig(
                min_child_exact_match_to_package=min_child_exact_match_to_package,
                max_allowed_regression_vs_parent=max_allowed_regression,
                max_family_regression=max_family_regression,
            ),
        )
    except BehavioralEvalError as exc:
        raise AdapterPackagingError(f"behavioral eval gate failed: {exc}") from exc
    print("behavior_gate=passed")


def create_submission_zip(
    adapter_dir: str | Path,
    zip_path: str | Path,
    *,
    behavioral_report_path: str | Path | None = None,
    allow_no_behavioral_eval: bool = False,
    baseline_adapter_only: bool = False,
    min_child_exact_match_to_package: float = 0.0,
    max_allowed_regression: float = 0.0,
    max_family_regression: float = 0.10,
) -> Path:
    validate_behavior_gate(
        behavioral_report_path,
        allow_no_behavioral_eval=allow_no_behavioral_eval,
        baseline_adapter_only=baseline_adapter_only,
        min_child_exact_match_to_package=min_child_exact_match_to_package,
        max_allowed_regression=max_allowed_regression,
        max_family_regression=max_family_regression,
    )
    config, model = validate_adapter_dir(adapter_dir)
    output = Path(zip_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(config, "adapter_config.json")
        archive.write(model, "adapter_model.safetensors")
    print_zip_summary(output)
    return output


def print_zip_summary(zip_path: str | Path) -> None:
    path = Path(zip_path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    print(f"submission_zip={path}")
    print(f"size_bytes={path.stat().st_size}")
    print("contents:")
    for name in names:
        print(f"- {name}")


def _extract_rank(payload: dict) -> int | None:
    for key in ("r", "rank", "lora_rank"):
        if key in payload:
            return int(payload[key])
    peft = payload.get("peft_config")
    if isinstance(peft, dict) and "r" in peft:
        return int(peft["r"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", default="/kaggle/working/custom_adapter")
    parser.add_argument("--zip-path", default="/kaggle/working/submission.zip")
    parser.add_argument("--behavioral-report-path", default="/kaggle/working/sprint4_behavioral_eval.json")
    parser.add_argument("--allow-no-behavioral-eval", action="store_true")
    parser.add_argument("--baseline-adapter-only", action="store_true")
    parser.add_argument("--min-child-exact-match-to-package", type=float, default=0.0)
    parser.add_argument("--max-allowed-regression", type=float, default=0.0)
    parser.add_argument("--max-family-regression", type=float, default=0.10)
    args = parser.parse_args()
    create_submission_zip(
        args.adapter_dir,
        args.zip_path,
        behavioral_report_path=args.behavioral_report_path,
        allow_no_behavioral_eval=args.allow_no_behavioral_eval,
        baseline_adapter_only=args.baseline_adapter_only,
        min_child_exact_match_to_package=args.min_child_exact_match_to_package,
        max_allowed_regression=args.max_allowed_regression,
        max_family_regression=args.max_family_regression,
    )


if __name__ == "__main__":
    main()
