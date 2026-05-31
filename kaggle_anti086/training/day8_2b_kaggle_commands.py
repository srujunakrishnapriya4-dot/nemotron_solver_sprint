from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.kaggle_path_safety import safe_write_text


COMMAND_PLAN = """# SPRINT-11G.2b Kaggle smoke orchestrator and gated full-train plan
# No package. No submit. No v2b/v2c/v3.

python kaggle_anti086/training/day8_2b_smoke_orchestrator.py --config kaggle_anti086/training/configs/v2a_base_lora.yaml --out artifacts/sprint11/day8_2b_smoke_orchestrator_report.json --kaggle-mode

# RUN ONLY IF artifacts/sprint11/day8_2b_smoke_orchestrator_report.json decision is ALLOW_DAY8_3_FULL_V2A_TRAINING.
python kaggle_anti086/training/day8_3_full_v2a_train.py --config kaggle_anti086/training/configs/v2a_base_lora.yaml --smoke-report artifacts/sprint11/day8_2b_smoke_orchestrator_report.json --out-manifest artifacts/sprint11/day8_3_v2a_train_manifest.json --out-summary artifacts/sprint11/day8_3_v2a_train_summary.json --kaggle-mode
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    safe_write_text(args.out, COMMAND_PLAN, field_name="day8_2b_kaggle_command_plan")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
