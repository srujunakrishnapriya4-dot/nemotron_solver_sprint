from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_anti086.kaggle_path_safety import safe_write_text


COMMAND_PLAN = """DAY 11A Kaggle Command Plan - Real v2 Candidate + Baseline Ranking

Stage A - Verify repo and PASS 10E runtime fixes
git branch --show-current
git log --oneline -5
python - <<'PY'
from pathlib import Path
checks = [
    ('trainer_api_compat', 'kaggle_anti086/training/lora_backend.py', '_trainer_tokenizer_kwargs'),
    ('callback_compat', 'kaggle_anti086/training/training_safety_callbacks.py', 'TrainerCallback'),
    ('bf16_qv_smoke_config', 'kaggle_anti086/training/configs/v4_solver_teacher_lora_smoke_bf16_qv.yaml', 'smoke_bf16_runtime_only'),
    ('nemotron_diagnostics', 'kaggle_anti086/training/day10_nemotron_lora_target_diagnostics.py', 'peft_linear4bit_shape_incompatible'),
]
missing = []
for name, path, token in checks:
    text = Path(path).read_text(encoding='utf-8') if Path(path).exists() else ''
    if token not in text:
        missing.append(name)
if missing:
    raise SystemExit('PASS10E_RUNTIME_FIX_MISSING:' + ','.join(missing))
print('PASS10E_RUNTIME_FIXES_PRESENT')
PY

Stage B - Run Day 10 regression tests
python -m pytest tests/test_day10_*.py -q -p no:cacheprovider

Stage C - Verify Day 10 corpus artifacts exist
python kaggle_anti086/training/day10_build_solver_teacher_corpus.py --out-direct artifacts/sprint11/day10_solver_teacher_direct.jsonl --out-abstain-policy artifacts/sprint11/day10_solver_teacher_abstain_policy.jsonl --out-manifest artifacts/sprint11/day10_solver_teacher_manifest.json --out-audit artifacts/sprint11/day10_solver_teacher_audit.json --out-overlap artifacts/sprint11/day10_teacher_overlap_audit.json --out-learnability artifacts/sprint11/day10_teacher_learnability_audit.json --out-mix-repair artifacts/sprint11/day10_corpus_mix_repair_report.json

Stage D - Run BF16 Q/V capacity audit
python kaggle_anti086/training/day10_lora_capacity_audit.py --config kaggle_anti086/training/configs/v4_solver_teacher_lora_smoke_bf16_qv.yaml --out artifacts/sprint11/day10_lora_capacity_audit_smoke_bf16_qv.json

Stage E - Dry-run v2a_50 config
python kaggle_anti086/training/day11a_real_v2_candidate.py --config kaggle_anti086/training/configs/v2a_bf16_qv_50.yaml --teacher-audit artifacts/sprint11/day10_solver_teacher_audit.json --overlap-audit artifacts/sprint11/day10_teacher_overlap_audit.json --learnability-audit artifacts/sprint11/day10_teacher_learnability_audit.json --mix-repair-report artifacts/sprint11/day10_corpus_mix_repair_report.json --capacity-audit artifacts/sprint11/day10_lora_capacity_audit_smoke_bf16_qv.json --out-summary artifacts/sprint11/day11a_v2a_50_train_summary.json --out-manifest artifacts/sprint11/day11a_v2a_50_train_manifest.json --dry-run --no-submit

Stage F - Train v2a_50 on Kaggle GPU
python kaggle_anti086/training/day11a_real_v2_candidate.py --config kaggle_anti086/training/configs/v2a_bf16_qv_50.yaml --teacher-audit artifacts/sprint11/day10_solver_teacher_audit.json --overlap-audit artifacts/sprint11/day10_teacher_overlap_audit.json --learnability-audit artifacts/sprint11/day10_teacher_learnability_audit.json --mix-repair-report artifacts/sprint11/day10_corpus_mix_repair_report.json --capacity-audit artifacts/sprint11/day10_lora_capacity_audit_smoke_bf16_qv.json --out-summary artifacts/sprint11/day11a_v2a_50_train_summary.json --out-manifest artifacts/sprint11/day11a_v2a_50_train_manifest.json --kaggle-mode --train --no-submit

Stage G - Validate adapter files without packaging
export ADAPTER_DIR=$(python - <<'PY'
import json
print(json.load(open('artifacts/sprint11/day11a_v2a_50_train_summary.json', encoding='utf-8')).get('adapter_dir',''))
PY
)
python kaggle_anti086/training/day10_adapter_package_guard.py --adapter-dir "$ADAPTER_DIR" --out artifacts/sprint11/day11a_v2a_50_package_guard.json --dry-run-package

Stage H - Run solver-only eval
python kaggle_anti086/eval/run_solver_eval.py --input artifacts/sprint11/day5_private_like_answerable_512.jsonl --out-report artifacts/sprint11/day11a_solver_only_eval_report.json --out-predictions artifacts/sprint11/day11a_solver_only_predictions.jsonl --allow-invalid-rows

Stage I - Run base model eval
python kaggle_anti086/training/day10_adapter_eval.py --base-model-path /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1 --day10-adapter-dir "$ADAPTER_DIR" --eval-path artifacts/sprint11/day5_private_like_answerable_512.jsonl --out-report artifacts/sprint11/day11a_base_eval_report.json --out-predictions artifacts/sprint11/day11a_base_eval_predictions.jsonl --max-rows 512 --kaggle-mode

Stage J - Run adapter-only v2a_50 eval
python kaggle_anti086/training/day10_adapter_eval.py --base-model-path /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1 --day10-adapter-dir "$ADAPTER_DIR" --eval-path artifacts/sprint11/day5_private_like_answerable_512.jsonl --out-report artifacts/sprint11/day11a_v2a_50_adapter_eval_report.json --out-predictions artifacts/sprint11/day11a_v2a_50_adapter_eval_predictions.jsonl --max-rows 512 --kaggle-mode

Stage K - Run combined solver plus adapter eval
python kaggle_anti086/training/day10_adapter_eval.py --base-model-path /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1 --day10-adapter-dir "$ADAPTER_DIR" --eval-path artifacts/sprint11/day5_private_like_answerable_512.jsonl --out-report artifacts/sprint11/day11a_combined_v2a_50_report.json --out-predictions artifacts/sprint11/day11a_combined_v2a_50_predictions.jsonl --max-rows 512 --kaggle-mode

Stage L - Build candidate ranking
python kaggle_anti086/training/day11a_candidate_ranking.py --base-report artifacts/sprint11/day11a_base_eval_report.json --solver-report artifacts/sprint11/day11a_solver_only_eval_report.json --adapter-report artifacts/sprint11/day11a_v2a_50_adapter_eval_report.json --combined-report artifacts/sprint11/day11a_combined_v2a_50_report.json --out artifacts/sprint11/day11a_candidate_ranking.json --decision-out artifacts/sprint11/day11a_decision_report.json

Stage M - Decision
python - <<'PY'
import json
d = json.load(open('artifacts/sprint11/day11a_decision_report.json', encoding='utf-8'))
print(json.dumps({'scale_to_150_allowed': d.get('scale_to_150_allowed'), 'fix_router_first': d.get('fix_router_first'), 'focus_solver_first': d.get('focus_solver_first'), 'submit_recommended': d.get('submit_recommended')}, sort_keys=True))
PY

Stage N - Save evidence metadata only
python - <<'PY'
import json
from pathlib import Path
paths = sorted(str(p) for p in Path('artifacts/sprint11').glob('day11a_*'))
Path('artifacts/sprint11/day11a_evidence_manifest.txt').write_text('\\n'.join(paths) + '\\n', encoding='utf-8')
print('DAY11A_EVIDENCE_METADATA_WRITTEN_NO_ARCHIVE')
PY

Stage O - No package / no submit
python - <<'PY'
print('NO_PUBLIC_SUBMISSION_COMMAND')
print('NO_ADAPTER_PACKAGE_CREATED')
print('NO_LEADERBOARD_CLAIM')
PY

Optional note: a public Tinker adapter may be evaluated as a baseline only if locally available and explicitly allowed by future rules. It is not a primary solution and is not copied, trained from, or used as a deliverable here.
"""


def write_command_plan(out: str | Path) -> None:
    safe_write_text(out, COMMAND_PLAN, field_name="day11a_kaggle_command_plan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Day 11A Kaggle command plan.")
    parser.add_argument("--out", default="artifacts/sprint11/day11a_kaggle_command_plan.txt")
    args = parser.parse_args(argv)
    write_command_plan(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
