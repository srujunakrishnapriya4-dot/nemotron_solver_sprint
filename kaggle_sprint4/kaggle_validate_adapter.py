from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

try:
    from kaggle_behavioral_eval import BehavioralEvalError, BehavioralGateConfig, load_report, validate_behavioral_report
except ImportError:  # pragma: no cover - supports package-style local tests
    from kaggle_sprint4.kaggle_behavioral_eval import BehavioralEvalError, BehavioralGateConfig, load_report, validate_behavioral_report


class AdapterValidationError(ValueError):
    pass


def validate_adapter_and_zip(
    adapter_dir: str | Path,
    zip_path: str | Path,
    *,
    behavioral_report_path: str | Path | None = None,
    allow_behavior_unknown: bool = False,
    min_child_exact_match_to_package: float = 0.0,
    max_allowed_regression: float = 0.0,
    max_family_regression: float = 0.10,
) -> bool:
    errors: list[str] = []
    root = Path(adapter_dir)
    config = root / "adapter_config.json"
    model = root / "adapter_model.safetensors"
    if not config.is_file():
        errors.append(f"missing adapter_config.json: {config}")
    if not model.is_file():
        errors.append(f"missing adapter_model.safetensors: {model}")
    if model.exists() and model.stat().st_size < 1024:
        errors.append("adapter_model.safetensors is too small to be a real adapter")
    if config.exists():
        try:
            payload = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                errors.append("adapter_config.json must be a JSON object")
            rank = _extract_rank(payload) if isinstance(payload, dict) else None
            if rank is not None and rank > 32:
                errors.append(f"LoRA rank exceeds 32: {rank}")
        except Exception as exc:
            errors.append(f"invalid adapter_config.json: {exc}")
    zip_file = Path(zip_path)
    if not zip_file.is_file():
        errors.append(f"missing submission.zip: {zip_file}")
    else:
        try:
            with zipfile.ZipFile(zip_file) as archive:
                names = sorted(archive.namelist())
            if names != ["adapter_config.json", "adapter_model.safetensors"]:
                errors.append(f"unexpected zip contents: {names}")
        except Exception as exc:
            errors.append(f"invalid zip: {exc}")

    structure_ready = not errors
    behavior_ready = False
    behavior_unknown = False
    if structure_ready:
        print("STRUCTURE_READY")
    report_path = Path(behavioral_report_path) if behavioral_report_path is not None else None
    if report_path is None or not report_path.is_file():
        behavior_unknown = True
        print("BEHAVIOR_UNKNOWN")
    else:
        try:
            report = load_report(report_path)
            validate_behavioral_report(
                report,
                gate=BehavioralGateConfig(
                    min_child_exact_match_to_package=min_child_exact_match_to_package,
                    max_allowed_regression_vs_parent=max_allowed_regression,
                    max_family_regression=max_family_regression,
                ),
            )
            behavior_ready = True
            print("BEHAVIOR_READY")
        except BehavioralEvalError as exc:
            errors.append(f"behavioral eval gate failed: {exc}")

    if errors:
        print("NOT_READY")
        for error in errors:
            print(f"- {error}")
        return False
    if behavior_ready or (behavior_unknown and allow_behavior_unknown):
        print("READY_TO_SUBMIT")
        return True
    print("NOT_READY")
    return False


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
    parser.add_argument("--allow-behavior-unknown", action="store_true")
    parser.add_argument("--min-child-exact-match-to-package", type=float, default=0.0)
    parser.add_argument("--max-allowed-regression", type=float, default=0.0)
    parser.add_argument("--max-family-regression", type=float, default=0.10)
    args = parser.parse_args()
    ok = validate_adapter_and_zip(
        args.adapter_dir,
        args.zip_path,
        behavioral_report_path=args.behavioral_report_path,
        allow_behavior_unknown=args.allow_behavior_unknown,
        min_child_exact_match_to_package=args.min_child_exact_match_to_package,
        max_allowed_regression=args.max_allowed_regression,
        max_family_regression=args.max_family_regression,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
