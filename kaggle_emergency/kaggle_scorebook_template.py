from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, "/kaggle/working/src")
sys.path.insert(0, "src")

from nemotron_engine.emergency_score_recovery.adapter_scorebook import write_scorebook_template


def main() -> None:
    output = write_scorebook_template(Path("/kaggle/working/adapter_scorebook.json"))
    print(f"adapter_scorebook_json={output}")
    print("Manual step: add public_score/private_score after each Kaggle submission. No API calls are made.")


if __name__ == "__main__":
    main()
