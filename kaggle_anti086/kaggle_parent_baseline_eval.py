from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from kaggle_prepare_anti086_tokens import load_simple_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="anti086_winmode_micro.yaml")
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    parent = str(config.get("parent_adapter_path", ""))
    if parent in {"", "auto_or_none", "none", "None"}:
        raise SystemExit("parent_adapter_path is required for parent baseline eval")
    tmp_config = Path("/kaggle/working/anti086_parent_eval_config.yaml")
    text = Path(args.config).read_text(encoding="utf-8")
    text = text.replace(str(config["output_adapter_dir"]), parent)
    text = text.replace(str(config["eval_output_dir"]), "/kaggle/working/anti086_parent_eval")
    tmp_config.write_text(text, encoding="utf-8")
    subprocess.check_call([sys.executable, "kaggle_eval_stage.py", "--config", str(tmp_config), "--stage", "eval_micro"])
    summary = json.loads(Path("/kaggle/working/anti086_parent_eval/eval_summary.json").read_text(encoding="utf-8"))
    out = Path("/kaggle/working/parent_baseline_eval_report.json")
    out.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
