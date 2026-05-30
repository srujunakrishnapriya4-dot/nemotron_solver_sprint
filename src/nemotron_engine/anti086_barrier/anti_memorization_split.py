from __future__ import annotations

from collections import defaultdict
from typing import Any


class AntiMemorizationSplitError(ValueError):
    pass


def split_rule_holdout(rows: list[dict[str, Any]], *, holdout_fraction: float = 0.2) -> dict[str, list[dict[str, Any]]]:
    if not 0 < holdout_fraction < 1:
        raise AntiMemorizationSplitError("holdout_fraction must be in (0,1)")
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rule[str(row.get("rule_id", "unknown"))].append(row)
    rules = sorted(by_rule)
    holdout_n = max(1, int(round(len(rules) * holdout_fraction))) if rules else 0
    holdout_rules = set(rules[-holdout_n:])
    train = [row for rule in rules if rule not in holdout_rules for row in by_rule[rule]]
    holdout = [row for rule in rules if rule in holdout_rules for row in by_rule[rule]]
    if {row.get("rule_id") for row in train} & {row.get("rule_id") for row in holdout}:
        raise AntiMemorizationSplitError("rule holdout leakage")
    return {"train": train, "rule_holdout": holdout}


def exclude_validation_hashes(train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validation_ids = {row.get("generation_hash") or row.get("row_hash") or row.get("id") for row in validation_rows}
    return [row for row in train_rows if (row.get("generation_hash") or row.get("row_hash") or row.get("id")) not in validation_ids]
