from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.eval.day5_eval_factory import build_answerable_rows, write_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day 5.1 answerable-only eval variants.")
    parser.add_argument("--source-eval", required=True, choices=["private_like", "rule_holdout", "family_hard"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1151)
    args = parser.parse_args(argv)
    rows = build_answerable_rows(args.source_eval, args.rows, args.seed)
    summary = write_rows(args.out, rows, field_name=f"day5_{args.source_eval}_answerable_eval")
    print(json.dumps({"status": "PASS", "source_eval": args.source_eval, **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
