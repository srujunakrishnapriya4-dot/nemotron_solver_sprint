from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default="/kaggle/working/submission.zip")
    args = parser.parse_args()
    path = Path(args.zip_path)
    if not path.is_file():
        raise SystemExit("NOT_READY missing submission.zip")
    with zipfile.ZipFile(path) as archive:
        names = sorted(archive.namelist())
    if names != ["adapter_config.json", "adapter_model.safetensors"]:
        raise SystemExit(f"NOT_READY bad zip contents: {names}")
    print("STRUCTURE_READY")
    print("READY_TO_SUBMIT_IF_GATE_MANIFEST_ACCEPTED")


if __name__ == "__main__":
    main()
