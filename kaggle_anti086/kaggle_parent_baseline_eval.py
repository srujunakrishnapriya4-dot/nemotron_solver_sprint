from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from kaggle_prepare_anti086_tokens import load_simple_yaml
from kaggle_path_safety import require_writable_output_path, safe_write_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="anti086_winmode_micro.yaml")
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    parent = str(config.get("parent_adapter_path", ""))
    if parent in {"", "auto_or_none", "none", "None"}:
        raise SystemExit("parent_adapter_path is required for parent baseline eval")
    tmp_config = require_writable_output_path("/kaggle/working/anti086_parent_eval_config.yaml", field_name="parent_baseline_tmp_config")
    text = Path(args.config).read_text(encoding="utf-8")
    text = text.replace(str(config["output_adapter_dir"]), parent)
    text = text.replace(str(config["eval_output_dir"]), "/kaggle/working/anti086_parent_eval")
    safe_write_text(tmp_config, text, field_name="parent_baseline_tmp_config")
    subprocess.check_call([sys.executable, "kaggle_eval_stage.py", "--config", str(tmp_config), "--stage", "eval_micro"])
    summary = json.loads(Path("/kaggle/working/anti086_parent_eval/eval_summary.json").read_text(encoding="utf-8"))
    out = require_writable_output_path("/kaggle/working/parent_baseline_eval_report.json", field_name="parent_baseline_report")
    safe_write_text(out, json.dumps(summary, sort_keys=True, indent=2), field_name="parent_baseline_report")
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
