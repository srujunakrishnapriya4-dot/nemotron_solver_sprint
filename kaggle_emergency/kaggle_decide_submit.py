from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, "/kaggle/working/src")
sys.path.insert(0, "src")

from nemotron_engine.emergency_score_recovery.adapter_inventory import inspect_adapter_dir
from nemotron_engine.emergency_score_recovery.adapter_scorebook import load_scorebook
from nemotron_engine.emergency_score_recovery.submission_decision_gate import decide_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--probe-json")
    parser.add_argument("--scorebook-json")
    parser.add_argument("--training-metadata-json")
    parser.add_argument("--operator-selected", action="store_true")
    parser.add_argument("--known-good-baseline", action="store_true")
    args = parser.parse_args()
    adapter = inspect_adapter_dir(args.adapter_dir)
    scorebook = load_scorebook(args.scorebook_json) if args.scorebook_json else ()
    probe = _load_json(args.probe_json) if args.probe_json else None
    training = _load_json(args.training_metadata_json) if args.training_metadata_json else None
    decision = decide_submission(
        adapter,
        scorebook_entries=scorebook,
        probe_evidence=probe,
        training_metadata=training,
        operator_selected=args.operator_selected,
        known_good_baseline=args.known_good_baseline,
    )
    print(decision.decision)
    print(json.dumps(decision.__dict__, sort_keys=True, indent=2))


def _load_json(path: str | None) -> dict:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()
