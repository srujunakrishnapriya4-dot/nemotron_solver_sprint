from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


CREATED_BY = "DAY1X_PREFLIGHT"
ARTIFACTS = Path("artifacts/sprint11")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def count_jsonl(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_day1x_preflight_report(root: str | Path = ".") -> dict[str, Any]:
    repo_root = Path(root)
    artifacts = repo_root / ARTIFACTS
    blocked: list[str] = []
    paths = {
        "day1_closeout": artifacts / "day1_closeout_and_artifact_alias_report.json",
        "composed_report": artifacts / "day1x_composed_teacher_report.json",
        "public_report": artifacts / "day1x_public_style_prompt_report.json",
        "prompt_decision": artifacts / "day1x_decision_report_prompt_stage.json",
        "phase4_train": artifacts / "phase4_verified_sft_train.jsonl",
        "phase4_eval": artifacts / "phase4_verified_sft_eval.jsonl",
        "phase4_probe": artifacts / "phase4_verified_eval_probe.jsonl",
        "phase5_dpo_train": artifacts / "phase5_dpo_train_pairs.jsonl",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    blocked.extend(f"missing_{name}" for name in missing)

    closeout = load_json(paths["day1_closeout"]) if paths["day1_closeout"].exists() else {}
    composed = load_json(paths["composed_report"]) if paths["composed_report"].exists() else {}
    public = load_json(paths["public_report"]) if paths["public_report"].exists() else {}
    decision = load_json(paths["prompt_decision"]) if paths["prompt_decision"].exists() else {}
    safety = dict(closeout.get("safety") or {})

    flags = {
        "training_authorized": _flag(closeout, safety, "training_authorized"),
        "safe_to_train_lora": _flag(closeout, safety, "safe_to_train_lora"),
        "package_authorized": _flag(closeout, safety, "package_authorized"),
        "submission_authorized": _flag(closeout, safety, "submission_authorized"),
        "leaderboard_claim": _flag(closeout, safety, "leaderboard_claim"),
        "no_0_93_evidence": _flag(closeout, safety, "no_0_93_evidence"),
        "no_0_95_evidence": _flag(closeout, safety, "no_0_95_evidence"),
    }
    checks = {
        "day1_closeout_status": closeout.get("status"),
        "day1_complete": closeout.get("day1_complete") is True,
        **flags,
        "composed_status": composed.get("status"),
        "public_status": public.get("status"),
        "prompt_decision_status": decision.get("status"),
        "ready_for_day1x_100k_factory": decision.get("ready_for_day1x_100k_factory") is True,
    }
    if checks["day1_closeout_status"] != "PASS":
        blocked.append("day1_closeout_not_pass")
    for key in (
        "day1_complete",
        "training_authorized",
        "safe_to_train_lora",
        "no_0_93_evidence",
        "no_0_95_evidence",
        "ready_for_day1x_100k_factory",
    ):
        if checks[key] is not True:
            blocked.append(f"{key}_bad")
    for key in ("package_authorized", "submission_authorized", "leaderboard_claim"):
        if checks[key] is not False:
            blocked.append(f"{key}_bad")
    if composed.get("status") != "PASS":
        blocked.append("composed_teacher_report_not_pass")
    if public.get("status") != "PASS":
        blocked.append("public_style_prompt_report_not_pass")
    if decision.get("status") != "PASS":
        blocked.append("prompt_stage_decision_not_pass")

    base_counts = {
        "phase4_train_rows": _count_if_exists(paths["phase4_train"]),
        "phase4_eval_rows": _count_if_exists(paths["phase4_eval"]),
        "phase4_probe_rows": _count_if_exists(paths["phase4_probe"]),
        "phase5_dpo_train_pairs": _count_if_exists(paths["phase5_dpo_train"]),
    }
    minimums = {
        "phase4_train_rows": 10000,
        "phase4_eval_rows": 1000,
        "phase4_probe_rows": 1000,
        "phase5_dpo_train_pairs": 3000,
    }
    for key, minimum in minimums.items():
        if base_counts[key] < minimum:
            blocked.append(f"{key}_below_minimum")
    blocked = sorted(set(blocked))
    return {
        "schema_version": 1,
        "created_by": CREATED_BY,
        "status": "PASS" if not blocked else "FAIL",
        "repo_root": str(repo_root.resolve()),
        "day1_closeout_status": closeout.get("status", "MISSING"),
        "day1x_prompt_stage_status": decision.get("status", "MISSING"),
        "training_authorized": checks["training_authorized"],
        "safe_to_train_lora": checks["safe_to_train_lora"],
        "package_authorized": _flag(closeout, safety, "package_authorized"),
        "submission_authorized": _flag(closeout, safety, "submission_authorized"),
        "leaderboard_claim": _flag(closeout, safety, "leaderboard_claim"),
        "base_counts": base_counts,
        "blocked_reasons": blocked,
        "ready_for_expanded_factory": not blocked,
    }


def write_day1x_preflight_report(out_path: str | Path) -> dict[str, Any]:
    report = build_day1x_preflight_report(".")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def assert_day1x_preflight_pass(report: dict[str, Any]) -> None:
    if report.get("status") != "PASS" or report.get("ready_for_expanded_factory") is not True:
        raise ValueError(f"Day1X preflight failed: {report.get('blocked_reasons', [])}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Day1X preflight report.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_day1x_preflight_report(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "ready_for_expanded_factory": report["ready_for_expanded_factory"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


def _flag(closeout: dict[str, Any], safety: dict[str, Any], key: str) -> Any:
    return closeout.get(key, safety.get(key))


def _count_if_exists(path: Path) -> int:
    return count_jsonl(path) if path.exists() else 0


if __name__ == "__main__":
    raise SystemExit(main())
