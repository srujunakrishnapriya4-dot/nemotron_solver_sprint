from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="anti086_config_micro.yaml")
    parser.parse_args()
    root = Path("/kaggle/working/anti086_input")
    if not root.exists():
        root = Path("artifacts/anti086")
    manifest = {"self_contained": True, "root": str(root), "note": "SPRINT-10 v1b uses mounted corpus files; this helper does not import repository packages."}
    print(json.dumps(manifest, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
