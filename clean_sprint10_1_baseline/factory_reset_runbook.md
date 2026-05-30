# Sprint 11 Day 1.2 Factory Reset Runbook

## Purpose

Day 1.2 freezes the clean Sprint 10.1 baseline, hardens every configured Kaggle output path, and verifies the generated Kaggle cells before Sprint 11 work. This does not improve score directly. It prevents corrupted scripts, fake eval, broken packaging, hidden stale cells, and untraceable file mutation.

## Required Kaggle Inputs

- Nemotron base model mounted at `/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1` or an equivalent local Kaggle model path in the config.
- A parent adapter input containing both `adapter_config.json` and `adapter_model.safetensors`.
- A clean Anti-0.86 / win-system input dataset containing `corpus_anti086_v1.jsonl` or `win_v1.jsonl`, plus manifest files when available.

Parent adapter discovery scans `/kaggle/input` and `/kaggle/working`, chooses an adapter directory containing the two adapter files, and patches `anti086_winmode_v1b.yaml`. If no parent adapter is found, the parent-calibrated eval must fail.

## Local Freeze Checks

Run from the repository root:

```bash
python -m py_compile kaggle_anti086/kaggle_*.py
python tools/day1_repo_guard.py --root . --out artifacts/day1/day1_guard_report.json
python tools/hash_audit.py --check artifacts/day1/day1_hash_manifest.json
python -m pytest tests/test_day1_* -q -p no:cacheprovider
```

The protected file list is recorded in `artifacts/day1/day1_hash_manifest.json` and the snapshot manifest is `clean_sprint10_1_baseline/baseline_manifest.json`.

Artifact policy: `artifacts/day1/*.json` reports are committed freeze artifacts for this Day 1 baseline and can be regenerated with the commands above. If a fresh clone is missing or changes these reports, rerun the guard/hash commands before trusting the notebook cells.

## Kaggle Factory Reset Cell Order

Copy and run only these cells from `artifacts/win_system_kaggle_cells/`:

1. `CELL_01_write_configs.py`
2. `CELL_02_write_runtime_patches.py`
3. `CELL_03_write_prepare_tokens.py`
4. `CELL_04_write_train.py`
5. `CELL_05_write_eval.py`
6. `CELL_06_write_parent_calibrated_eval.py`
7. `CELL_07_write_orchestrator.py`
8. `CELL_08_verify_files.py`
9. `CELL_09_v1b_commands.py`

Each writer cell uses `write_file_checked(...)` and prints JSON containing file path, size bytes, and SHA256 after writing. If `artifacts/win_system_kaggle_cells/` is missing, do not use the zip blindly. Regenerate or fail.

Every configured output path must be under `/kaggle/working` in Kaggle. Output paths under `/kaggle/input` are invalid even when the input dataset itself is mounted there.

## Kaggle Smoke Command

Run parent-calibrated eval only:

```bash
PYTHONPATH="/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages:/kaggle/working:/kaggle/working/src:$PYTHONPATH" \
TRITON_PTXAS_PATH="/tmp/ptxas-blackwell" \
TRITON_PTXAS_BLACKWELL_PATH="/tmp/ptxas-blackwell" \
python kaggle_build_parent_calibrated_eval.py --config anti086_winmode_v1b.yaml 2>&1 | tee /kaggle/working/build_parent_calibrated_eval.log
```

Expected artifacts:

- `/kaggle/working/anti086_eval/v1b/parent_calibration_report.json`
- `/kaggle/working/anti086_eval/v1b/parent_calibrated_eval.jsonl`
- `/kaggle/working/build_parent_calibrated_eval.log`

This is diagnostic only. Parent score may be weak. This does not prove 0.95. Do not train on Day 1.

## What Not To Run

Do not run:

- `train_v1`
- `train_v1b`
- `train_v2`
- `train_v3`
- `package`
- `submission`
- old Sprint-8/Sprint-9 cells
- manual notebook patch cells
- any cell or script that writes outputs to `/kaggle/input`

## Expected Success State

- all active `kaggle_*.py` compile
- `day1_repo_guard` reports `PASS`
- no active Kaggle script imports `nemotron_engine`
- no active Kaggle script writes to `/kaggle/input`
- no active config sends outputs to `/kaggle/input`
- `artifacts/win_system_kaggle_cells/` exists and matches `artifacts/win_system_kaggle_cells.zip` by SHA256
- `artifacts/day1/day1_generated_cells_manifest.json` exists
- `clean_sprint10_1_baseline/baseline_manifest.json` exists
- `artifacts/day1/day1_parent_calibrated_eval_smoke_command.txt` exists

## Recovery If Files Are Overwritten

Restore from:

- `clean_sprint10_1_baseline/`
- `clean_sprint10_1_baseline/baseline_manifest.json`

Compare file hashes before copying back. Do not restore model weights, adapters, token files, or `submission.zip` from any snapshot.

## Reality Check

Day 1.2 still provides zero 0.95 evidence. It only protects the later Sprint 11 solver/eval/training plan from infrastructure corruption. 0.95 evidence is still absent.

## Sprint 11A Day 2 Foundation

Day 2 adds the non-training correctness foundation that later solver/data/eval work must obey:

- answer normalization in `kaggle_anti086/solvers/answer_normalizer.py`
- strict row schema in `kaggle_anti086/data/schema.py`
- leakage-aware split controller in `kaggle_anti086/data/split_controller.py`
- rule-bank registry in `kaggle_anti086/data/build_rule_bank.py`
- solver interface contracts in `kaggle_anti086/solvers/types.py` and `kaggle_anti086/solvers/base.py`

Validate Day 2 from the repository root:

```bash
python -m py_compile kaggle_anti086/solvers/*.py kaggle_anti086/data/*.py
python -m pytest tests/test_sprint11_answer_normalizer.py tests/test_sprint11_schema.py tests/test_sprint11_split_controller.py tests/test_sprint11_rule_bank.py tests/test_sprint11_solver_contracts.py -q -p no:cacheprovider
python tools/day1_repo_guard.py --root . --out artifacts/day1/day1_guard_report.json
python tools/hash_audit.py --check artifacts/day1/day1_hash_manifest.json
```

PowerShell does not reliably expand `*.py` for `py_compile`; use explicit file lists or `Get-ChildItem` when needed.

Day 2 intentionally does not implement full deterministic solvers, synthetic generation, LoRA training, packaging, or submission. It connects to Day 3 by making rule IDs, leakage groups, verification status, and answer normalization mandatory before any solver-correct corpus can be trusted.

Day 2 creates no 0.95 evidence. It only prevents fake local scores caused by bad normalization, invalid rows, train/eval leakage, and unregistered rules.

## Sprint 11B Day 3 Deterministic Solver Smoke

Day 3 adds the first non-model deterministic solver layer:

- Roman numeral solver
- unit conversion fitter
- numeric/gravity formula fitter
- word-level cipher substitution solver
- deterministic solver ensemble
- solver-only eval smoke runner

Validate Day 3 from the repository root:

```bash
python -m py_compile kaggle_anti086/solvers/*.py kaggle_anti086/data/*.py kaggle_anti086/eval/*.py
python -m pytest tests/test_sprint11_roman_solver.py tests/test_sprint11_unit_conversion_solver.py tests/test_sprint11_numeric_formula_solver.py tests/test_sprint11_word_cipher_solver.py tests/test_sprint11_solver_ensemble.py tests/test_sprint11_solver_eval.py -q -p no:cacheprovider
```

PowerShell wildcard fallback:

```powershell
Get-ChildItem kaggle_anti086\solvers\*.py | ForEach-Object { python -m py_compile $_.FullName }
Get-ChildItem kaggle_anti086\data\*.py | ForEach-Object { python -m py_compile $_.FullName }
Get-ChildItem kaggle_anti086\eval\*.py | ForEach-Object { python -m py_compile $_.FullName }
```

Run the solver eval smoke manually with:

```bash
python -m kaggle_anti086.eval.run_solver_eval --input path.jsonl --out-report artifacts/sprint11/day3_solver_eval_report.json --out-predictions artifacts/sprint11/day3_solver_eval_predictions.jsonl
```

Day 3 intentionally does not implement bit transforms, symbol mapping, char cipher, equation/operator solving, private-like split generation, synthetic corpora, LoRA training, packaging, or submission. It creates no leaderboard evidence and no 0.95 evidence. Its job is to prove that deterministic solvers can abstain safely, return schema-compatible candidates, and be scored with the Day 2 normalizer.

## Sprint 11B.1 Day 3 Hardening

Day 3.1 repairs the first deterministic solver layer before Day 4 expansion:

- raw `Decimal` numeric parsing preserves output precision from prompt examples
- unit and numeric formula solvers abstain on unsafe ambiguity instead of two-point overfit
- Roman query parsing handles additional natural query forms
- word cipher abstentions include mapping coverage and conflict diagnostics
- solver ensemble merges same-answer candidates and refuses sub-threshold confidence
- solver eval uses path-safe output writes and separates `execution_status` from `quality_status`
- `kaggle_anti086/eval/build_day3_solver_smoke.py` generates a 60-row in-scope solver smoke set

Validate Day 3.1:

```bash
python kaggle_anti086/eval/build_day3_solver_smoke.py --out artifacts/sprint11/day3_solver_smoke_eval.jsonl
python kaggle_anti086/eval/run_solver_eval.py --input artifacts/sprint11/day3_solver_smoke_eval.jsonl --out-report artifacts/sprint11/day3_solver_eval_report.json --out-predictions artifacts/sprint11/day3_solver_eval_predictions.jsonl --min-exact-match 0.80 --min-attempt-rate 0.80 --fail-on-quality-gate
python -m pytest tests/test_sprint11_unit_conversion_precision.py tests/test_sprint11_numeric_formula_precision.py tests/test_sprint11_roman_query_parsing.py tests/test_sprint11_word_cipher_diagnostics.py tests/test_sprint11_solver_ensemble_hardening.py tests/test_sprint11_solver_eval_hardening.py tests/test_sprint11_day3_solver_smoke_builder.py -q -p no:cacheprovider
```

This still creates no 0.95 evidence. The smoke rows are generated inside the supported Day 3 solver scope and are only a regression gate.

## Sprint 11C Day 4 Hard Solver Expansion

Day 4 adds the second deterministic solver layer and safety foundation:

- bit transform solver for fixed-width binary transformations
- symbol mapping solver for punctuation-heavy substitution tasks
- char cipher solver for Caesar, reversal, and safe monoalphabetic mappings
- router that selects solver order without emitting answers
- verifier that rejects unsafe candidate answers before ensemble ranking
- adversarial solver eval builder with answerable and expected-abstain rows
- solver eval metrics for unsafe answers, correct abstains, wrong abstains, and quality gates

Validate Day 4:

```bash
python kaggle_anti086/eval/build_day4_adversarial_solver_eval.py --out artifacts/sprint11/day4_adversarial_solver_eval.jsonl
python kaggle_anti086/eval/run_solver_eval.py --input artifacts/sprint11/day4_adversarial_solver_eval.jsonl --out-report artifacts/sprint11/day4_adversarial_solver_report.json --out-predictions artifacts/sprint11/day4_adversarial_solver_predictions.jsonl --min-exact-match 0.70 --min-attempt-rate 0.50 --max-unsafe-answer-rate 0.05 --min-correct-abstain-rate 0.80 --fail-on-quality-gate
python -m pytest tests/test_sprint11_bit_transform_solver.py tests/test_sprint11_symbol_mapping_solver.py tests/test_sprint11_char_cipher_solver.py tests/test_sprint11_router.py tests/test_sprint11_verifier.py tests/test_sprint11_solver_ensemble_day4.py tests/test_sprint11_day4_adversarial_eval.py -q -p no:cacheprovider
```

Day 4 improves deterministic solver coverage and safety only. It is still not leaderboard evidence, not public/private readiness, and not 0.95 evidence. The next blocker is private-like, rule-holdout, and family-hard eval generation plus a solver-correct v2 corpus.
