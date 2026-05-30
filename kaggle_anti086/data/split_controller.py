from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random

from kaggle_anti086.data.schema import validate_row, is_trainable_row, RowValidationError


@dataclass(frozen=True)
class LeakageReport:
    status: str
    train_count: int
    eval_counts: dict[str, int]
    rule_overlap_count: int
    rule_overlap_examples: list[str]
    leakage_group_overlap_count: int
    leakage_group_overlap_examples: list[str]
    row_id_overlap_count: int
    row_id_overlap_examples: list[str]
    unverified_train_rows: list[str]
    unsupported_train_rows: list[str]
    unsafe_train_rows: list[str]
    abstained_train_rows: list[str]
    quarantined_train_rows: list[str]
    unsupported_equation_train_rows: list[str]
    failures: list[str]


@dataclass(frozen=True)
class SplitConfig:
    private_like_eval: int = 1000
    family_hard_eval: int = 512
    rule_holdout_eval: int = 512
    anti_leak_eval: int = 256
    parent_calibrated_eval: int = 64
    seed: int = 1102
    allow_failed_report: bool = False


@dataclass(frozen=True)
class SplitBundle:
    train: list[dict] = field(default_factory=list)
    private_like_eval: list[dict] = field(default_factory=list)
    family_hard_eval: list[dict] = field(default_factory=list)
    rule_holdout_eval: list[dict] = field(default_factory=list)
    anti_leak_eval: list[dict] = field(default_factory=list)
    parent_calibrated_eval: list[dict] = field(default_factory=list)
    quarantine: list[dict] = field(default_factory=list)
    report: LeakageReport | None = None


def check_split_leakage(train_rows, eval_bundles: dict[str, list[dict]]) -> LeakageReport:
    train_ids = {str(r.get("id")) for r in train_rows}
    train_rules = {str(r.get("rule_id")) for r in train_rows}
    train_groups = {str(r.get("leakage_group")) for r in train_rows}
    eval_counts = {name: len(rows) for name, rows in sorted(eval_bundles.items())}

    rule_holdout_rules = {str(r.get("rule_id")) for r in eval_bundles.get("rule_holdout_eval", [])}
    anti_leak_groups = {str(r.get("leakage_group")) for r in eval_bundles.get("anti_leak_eval", [])}
    eval_ids = {str(r.get("id")) for rows in eval_bundles.values() for r in rows}

    rule_overlap = sorted(train_rules & rule_holdout_rules)
    group_overlap = sorted(train_groups & anti_leak_groups)
    id_overlap = sorted(train_ids & eval_ids)

    unverified = []
    unsupported = []
    unsafe = []
    abstained = []
    quarantined = []
    unsupported_equation = []
    for row in train_rows:
        row_id = str(row.get("id"))
        status = str(row.get("verification_status"))
        family = str(row.get("family"))
        if status == "unverified":
            unverified.append(row_id)
        if status == "unsupported":
            unsupported.append(row_id)
        if status == "unsafe":
            unsafe.append(row_id)
        if status == "abstained":
            abstained.append(row_id)
        if status == "quarantined":
            quarantined.append(row_id)
        if family == "equation_operator" and status != "verified":
            unsupported_equation.append(row_id)

    failures = []
    if rule_overlap:
        failures.append("train/rule_holdout rule_id overlap")
    if group_overlap:
        failures.append("train/anti_leak leakage_group overlap")
    if id_overlap:
        failures.append("train/eval row id overlap")
    for label, values in (
        ("unverified train rows", unverified),
        ("unsupported train rows", unsupported),
        ("unsafe train rows", unsafe),
        ("abstained train rows", abstained),
        ("quarantined train rows", quarantined),
        ("unsupported equation/operator train rows", unsupported_equation),
    ):
        if values:
            failures.append(label)

    return LeakageReport(
        status="FAIL" if failures else "PASS",
        train_count=len(train_rows),
        eval_counts=eval_counts,
        rule_overlap_count=len(rule_overlap),
        rule_overlap_examples=rule_overlap[:10],
        leakage_group_overlap_count=len(group_overlap),
        leakage_group_overlap_examples=group_overlap[:10],
        row_id_overlap_count=len(id_overlap),
        row_id_overlap_examples=id_overlap[:10],
        unverified_train_rows=unverified,
        unsupported_train_rows=unsupported,
        unsafe_train_rows=unsafe,
        abstained_train_rows=abstained,
        quarantined_train_rows=quarantined,
        unsupported_equation_train_rows=unsupported_equation,
        failures=failures,
    )


def build_splits(rows: list[dict], config: SplitConfig | None = None) -> SplitBundle:
    config = config or SplitConfig()
    rng = random.Random(config.seed)
    buckets = {
        "train": [],
        "private_like_eval": [],
        "family_hard_eval": [],
        "rule_holdout_eval": [],
        "anti_leak_eval": [],
        "parent_calibrated_eval": [],
        "quarantine": [],
    }
    for raw in rows:
        row = dict(raw)
        if not is_trainable_row(row) and row.get("split") == "train":
            row["split"] = "quarantine"
            row["verification_status"] = row.get("verification_status") or "quarantined"
        try:
            parsed = validate_row(row)
        except RowValidationError:
            row["split"] = "quarantine"
            row.setdefault("verification_status", "quarantined")
            row.setdefault("answer", None)
            parsed = validate_row(row)
        buckets[parsed.split].append(row)

    for key in buckets:
        rng.shuffle(buckets[key])

    report = check_split_leakage(
        buckets["train"],
        {k: buckets[k] for k in buckets if k != "train" and k != "quarantine"},
    )
    if report.status == "FAIL" and not config.allow_failed_report:
        raise SystemExit(f"split leakage check failed: {report.failures}")
    return SplitBundle(**buckets, report=report)


def write_split_report(report: LeakageReport, path: str | Path) -> None:
    payload = {k: v for k, v in report.__dict__.items()}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
