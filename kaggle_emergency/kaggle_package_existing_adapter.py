from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "/kaggle/working/src")
sys.path.insert(0, "src")

from nemotron_engine.emergency_score_recovery.adapter_package_plan import package_existing_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--zip-path", default="/kaggle/working/submission.zip")
    args = parser.parse_args()
    result = package_existing_adapter(args.adapter_dir, args.zip_path)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
