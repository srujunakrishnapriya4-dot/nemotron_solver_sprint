from __future__ import annotations

import argparse, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.kaggle_path_safety import safe_write_text


COMMANDS = [
    "python kaggle_anti086/training/real_tokenizer_collator_audit.py --config kaggle_anti086/training/configs/v2a_base_lora.yaml --out artifacts/sprint11/day8_1_real_collator_audit_report.json --sample-size 128 --kaggle-mode",
    "python kaggle_anti086/training/weighted_sft_sampler.py --out artifacts/sprint11/day8_2_weighted_sampling_report.json",
    "python kaggle_anti086/training/truncation_audit.py --out artifacts/sprint11/day8_2_truncation_audit_report.json",
    "python kaggle_anti086/training/lora_target_inspector.py --config kaggle_anti086/training/configs/v2a_base_lora.yaml --out artifacts/sprint11/day8_2_lora_target_report.json --kaggle-mode",
    "python kaggle_anti086/training/trainable_parameter_audit.py --config kaggle_anti086/training/configs/v2a_base_lora.yaml --out artifacts/sprint11/day8_2_trainable_parameter_report.json --kaggle-mode",
    "python kaggle_anti086/training/supervised_label_decode_audit.py --out artifacts/sprint11/day8_2_supervised_label_decode_report.json",
    "python kaggle_anti086/training/final_sft_leakage_audit.py --out artifacts/sprint11/day8_2_final_sft_leakage_report.json",
    "python kaggle_anti086/training/day8_1_backend_readiness.py --out artifacts/sprint11/day8_1_backend_readiness_report.json",
    "python kaggle_anti086/training/day8_2_smoke_train.py --config kaggle_anti086/training/configs/v2a_base_lora.yaml --collator-audit artifacts/sprint11/day8_1_real_collator_audit_report.json --steps 5 --out-summary artifacts/sprint11/day8_2_smoke_train_summary.json --kaggle-mode",
    "python kaggle_anti086/training/adapter_artifact_audit.py --adapter-dir <smoke_adapter_dir> --out artifacts/sprint11/day8_2_adapter_artifact_report.json",
    "python kaggle_anti086/training/post_train_eval.py --base-model-path <resolved_base_model_path> --adapter-dir <smoke_adapter_dir> --out-report artifacts/sprint11/day8_2_smoke_eval_report.json --out-predictions artifacts/sprint11/day8_2_smoke_eval_predictions.jsonl --kaggle-mode",
    "python kaggle_anti086/training/output_drift_audit.py --predictions artifacts/sprint11/day8_2_smoke_eval_predictions.jsonl --out artifacts/sprint11/day8_2_output_drift_report.json",
    "python kaggle_anti086/training/day8_2_readiness.py --out artifacts/sprint11/day8_2_readiness_report.json",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    content = "# SPRINT-11G.2 Day 8.2 Kaggle smoke-train command plan\n# No full 500-step train. No package. No submit.\n\n" + "\n\n".join(COMMANDS) + "\n"
    safe_write_text(args.out, content, field_name="day8_2_kaggle_command_plan")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
