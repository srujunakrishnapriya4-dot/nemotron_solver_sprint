from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.eval.day5_eval_factory import build_eval_rows, write_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day 5 anti-leak eval.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1108)
    args = parser.parse_args(argv)
    rows = build_eval_rows("anti_leak", args.rows, args.seed)
    write_rows(args.out, rows, field_name="day5_anti_leak_eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
