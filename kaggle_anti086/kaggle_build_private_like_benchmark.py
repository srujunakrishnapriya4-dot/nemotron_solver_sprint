from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("/kaggle/working/anti086_input")
    payload = {"self_contained": True, "root": str(root), "note": "SPRINT-10 v1b uses parent-calibrated eval; this helper does not import repository packages."}
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
