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
