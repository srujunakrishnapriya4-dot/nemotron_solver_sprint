from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.kaggle_path_safety import safe_write_text


COMMANDS = [
    "python kaggle_anti086/training/real_tokenizer_collator_audit.py --config kaggle_anti086/training/configs/v2a_base_lora.yaml --out artifacts/sprint11/day8_1_real_collator_audit_report.json --sample-size 128 --kaggle-mode",
    "python kaggle_anti086/training/day8_1_backend_readiness.py --out artifacts/sprint11/day8_1_backend_readiness_report.json",
    "python kaggle_anti086/training/train_v2a_lora.py --config kaggle_anti086/training/configs/v2a_base_lora.yaml --collator-audit artifacts/sprint11/day8_1_real_collator_audit_report.json --out-manifest artifacts/sprint11/day8_1_v2a_train_manifest.json --out-summary artifacts/sprint11/day8_1_v2a_train_summary.json --kaggle-mode",
    "python kaggle_anti086/training/post_train_eval.py --base-model-path <resolved_base_model_path> --adapter-dir <adapter_dir_from_train_summary> --out-report artifacts/sprint11/day8_1_v2a_eval_report.json --out-predictions artifacts/sprint11/day8_1_v2a_eval_predictions.jsonl --kaggle-mode",
    "python kaggle_anti086/training/day8_candidate_report.py --train-summary artifacts/sprint11/day8_1_v2a_train_summary.json --eval-report artifacts/sprint11/day8_1_v2a_eval_report.json --out artifacts/sprint11/day8_1_candidate_report.json",
    "python kaggle_anti086/training/day8_decision_gate.py --collator-audit artifacts/sprint11/day8_1_real_collator_audit_report.json --train-summary artifacts/sprint11/day8_1_v2a_train_summary.json --eval-report artifacts/sprint11/day8_1_v2a_eval_report.json --candidate-report artifacts/sprint11/day8_1_candidate_report.json --out artifacts/sprint11/day8_1_decision_report.json",
]


def write_command_plan(path: str | Path) -> None:
    content = "# SPRINT-11G.1 Day 8.1 Kaggle command plan\n# No package command. No submit command.\n\n" + "\n\n".join(COMMANDS) + "\n"
    safe_write_text(path, content, field_name="day8_1_kaggle_command_plan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    write_command_plan(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
