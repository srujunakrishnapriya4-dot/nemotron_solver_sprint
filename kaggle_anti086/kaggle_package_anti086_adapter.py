from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_prepare_anti086_tokens import load_simple_yaml
from kaggle_path_safety import require_not_kaggle_input_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="anti086_config_micro.yaml")
    parser.add_argument("--eval-summary", default="/kaggle/working/anti086_eval/eval_summary.json")
    parser.add_argument("--zip-path", default="/kaggle/working/submission.zip")
    args = parser.parse_args()
    require_not_kaggle_input_path(args.zip_path, field_name="submission_zip_path")
    config = load_simple_yaml(args.config)
    summary = json.loads(Path(args.eval_summary).read_text(encoding="utf-8"))
    decision = {
        "decision": "REJECT",
        "reason": "SPRINT-10 v1b forbids packaging; require later v1/v2 evidence before any child submission.",
        "rank": int(config.get("rank", 999)),
        "exact_match": summary.get("exact_match", 0.0),
    }
    print(json.dumps(decision, sort_keys=True, indent=2))
    raise SystemExit("SPRINT-10 package path is intentionally disabled")


if __name__ == "__main__":
    main()
