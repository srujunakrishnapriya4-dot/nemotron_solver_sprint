from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VENV_SITE_PACKAGES = REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(VENV_SITE_PACKAGES))

from src.offline.failure_mining import (
    FailureMiningConfig,
    FailureMiningStatus,
    load_failure_input_path,
    mine_failures,
    write_failure_artifacts,
)


LOGGER = logging.getLogger("run_failure_mining")
DEFAULT_OUTPUT_DIR = "artifacts/offline/failure_mining"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic offline failure mining over historical prediction artifacts."
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Input path. Repeat for multiple sources. Supports JSONL/JSON, distilled artifact dirs, and manifest paths.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--min-hard-problems",
        type=int,
        default=20,
        help="Minimum hard-problem count required for a fully valid mining run.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on historical records processed after deterministic sorting.",
    )
    parser.add_argument(
        "--allow-small-batch",
        action="store_true",
        help="Do not fail the process when fewer than --min-hard-problems are available.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level. Default: INFO",
    )
    return parser


def _load_sources(paths: Sequence[str]) -> list[object]:
    loaded: list[object] = []
    for raw_path in paths:
        loaded.extend(load_failure_input_path(raw_path))
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    config = FailureMiningConfig(
        min_hard_problems=max(1, int(args.min_hard_problems)),
        max_records=args.max_records,
    )
    sources = _load_sources(args.inputs)
    result = mine_failures(sources, config=config)
    paths = write_failure_artifacts(result, args.output_dir)

    LOGGER.info(
        "Failure mining completed: status=%s total_records=%s labeled=%s failed=%s hard=%s",
        result.manifest.status.value,
        result.manifest.total_records,
        result.manifest.labeled_records,
        result.manifest.failed_records,
        result.slice_metrics.get("hard_problem_count", 0),
    )
    LOGGER.info("Artifacts written to %s", Path(args.output_dir).resolve())
    for key in ("failures", "failure_summary", "slice_metrics", "suggested_repairs", "manifest"):
        LOGGER.info("%s: %s", key, paths[key])

    print(
        json.dumps(
            {
                "status": result.manifest.status.value,
                "output_dir": str(Path(args.output_dir).resolve()),
                "paths": paths,
                "summary": {
                    "total_records": result.manifest.total_records,
                    "labeled_records": result.manifest.labeled_records,
                    "failed_records": result.manifest.failed_records,
                    "hard_problem_count": result.slice_metrics.get("hard_problem_count", 0),
                },
            },
            sort_keys=True,
        )
    )

    if result.manifest.status is FailureMiningStatus.INSUFFICIENT_INPUT and not args.allow_small_batch:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
