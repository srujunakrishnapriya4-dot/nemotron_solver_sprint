from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_engine.vex_progress_bridge.verified_rule_exporter import export_verified_rules  # noqa: E402


def test_verified_rule_exporter_keeps_all_rows_and_marks_statuses(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "prompt", "answer"))
        writer.writeheader()
        writer.writerow({"id": "r1", "prompt": "In Alice's Wonderland, numbers are secretly converted into a different numeral system.\n\n11 -> XI\n12 -> XII\n\nNow, write the number 13 in the Wonderland numeral system.", "answer": "XIII"})
        writer.writerow({"id": "bad", "prompt": "malformed", "answer": "x"})

    manifest = export_verified_rules(csv_path, tmp_path / "out")
    rows = [json.loads(line) for line in (tmp_path / "out" / "verified_rules.jsonl").read_text(encoding="utf-8").splitlines()]

    assert manifest["total_rows"] == 2
    assert rows[0]["verified_status"] == "verified_correct"
    assert rows[1]["verified_status"] == "parse_failed"

