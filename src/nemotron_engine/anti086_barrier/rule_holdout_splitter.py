from __future__ import annotations

from collections import defaultdict
from typing import Any


def split_rule_holdout_v2(rows: list[dict[str, Any]], *, holdout_fraction: float = 0.20) -> dict[str, list[dict[str, Any]]]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rule[str(row.get("rule_id", "unknown"))].append(row)
    rules = sorted(by_rule)
    holdout_rules = set(rules[: max(1, int(len(rules) * holdout_fraction))])
    return {
        "train": [row for rule, vals in by_rule.items() if rule not in holdout_rules for row in vals],
        "holdout": [row for rule, vals in by_rule.items() if rule in holdout_rules for row in vals],
        "holdout_rule_ids": sorted(holdout_rules),
    }
