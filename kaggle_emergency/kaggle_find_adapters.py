from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, "/kaggle/working/src")
sys.path.insert(0, "src")

from nemotron_engine.emergency_score_recovery.adapter_inventory import scan_adapter_roots


def main() -> None:
    roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    records = scan_adapter_roots(roots)
    payload = {"adapters": [record.__dict__ for record in records], "count": len(records)}
    output = Path("/kaggle/working/adapter_inventory.json")
    output.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True, indent=2))
    print(f"adapter_inventory_json={output}")


if __name__ == "__main__":
    main()
