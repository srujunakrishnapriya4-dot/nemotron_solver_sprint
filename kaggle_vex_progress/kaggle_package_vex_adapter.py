from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "/kaggle/working/src")

from kaggle_prepare_vex_corpus import load_simple_yaml
from nemotron_engine.vex_progress_bridge.vex_submission_gate import evaluate_vex_submission_gate, package_vex_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vex_config_main.yaml")
    parser.add_argument("--eval-summary", default="/kaggle/working/vex_eval/eval_summary.json")
    parser.add_argument("--zip-path", default="/kaggle/working/submission.zip")
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    eval_summary = json.loads(Path(args.eval_summary).read_text(encoding="utf-8"))
    gate = evaluate_vex_submission_gate(config["output_adapter_dir"], rank=int(config["rank"]), adapter_size_limit_mb=int(config["adapter_size_limit_mb"]), eval_summary=eval_summary)
    print(json.dumps(gate, sort_keys=True, indent=2))
    if gate["decision"] != "ACCEPT":
        raise SystemExit("VEX gate rejected adapter")
    manifest = package_vex_adapter(config["output_adapter_dir"], args.zip_path, "/kaggle/working/vex_submission_manifest.json")
    print(json.dumps(manifest, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
