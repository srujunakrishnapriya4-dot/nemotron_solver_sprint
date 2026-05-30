from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    manifest = Path("artifacts/anti086/curriculum_manifest.json")
    if not manifest.exists():
        raise SystemExit("missing curriculum_manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("synthetic_ratio_v2", 1.0) > 0.35:
        raise SystemExit("synthetic ratio too high")
    if payload.get("contrastive_ratio_v3", 1.0) > 0.15:
        raise SystemExit("contrastive ratio too high")
    print("TOKEN_CORPUS_POLICY_READY")


if __name__ == "__main__":
    main()
