from __future__ import annotations


def emergency_notebook_steps() -> tuple[str, ...]:
    return (
        "Run kaggle_find_adapters.py to inventory all public, tinker, and working adapter directories.",
        "Package the known-good public parent adapter first so the 0.85 fallback path is preserved.",
        "Submit the baseline manually if no clearly better candidate exists.",
        "For each existing public/tinker adapter, package it without training, submit manually, and record the public score in adapter_scorebook.json.",
        "Do not train unless a micro-run has assistant-only loss, low LR, adapter-size guard, fast probe, and decision-gate approval.",
        "Run kaggle_probe_fast.py on at most 3 rows before any custom child packaging.",
        "Run kaggle_decide_submit.py and obey DO_NOT_SUBMIT unless the operator explicitly selects a candidate.",
        "Final selection should prefer the best public score unless private-risk notes justify a safer baseline.",
    )


def emergency_strategy_verdict() -> dict:
    return {
        "can_guarantee_0_95": False,
        "realistic_near_term_band": "0.85-0.87 unless a genuinely stronger public/tinker adapter is found",
        "highest_probability_24h_action": "adapter sweep and safe packaging of existing adapters, with the known-good parent as fallback",
        "stop_immediately": "blind 30B continuation training, full-prompt loss, huge child adapters without evidence, slow broad behavioral eval",
        "submit_if_no_better_candidate": "known-good public parent adapter direct package",
        "required_to_chase_0_95": "a proven stronger adapter or a validated training recipe with cheap non-regression probes and real public/private robustness evidence",
    }
