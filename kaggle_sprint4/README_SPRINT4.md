# SPRINT-4 Kaggle Continuation LoRA Workflow

This folder contains a Kaggle-ready workflow for continuation training a real LoRA adapter from the Sprint-3 curated datasets.

It does not call Kaggle APIs and does not auto-submit. The training script is intended to run inside a Kaggle Notebook with GPU access.

## Local export

From the repository root:

```bash
PYTHONPATH=src python -m nemotron_engine.adapter_training_data.kaggle_export
```

or call:

```python
from nemotron_engine.adapter_training_data.kaggle_export import export_sprint4_kaggle_input
export_sprint4_kaggle_input()
```

This creates:

```text
artifacts/sprint4_kaggle_input/
artifacts/sprint4_kaggle_input.zip
```

Upload `artifacts/sprint4_kaggle_input.zip` as a Kaggle Dataset. This zip is only a training input dataset, not the competition submission.

## Kaggle sequence

1. Upload `artifacts/sprint4_kaggle_input.zip` as a Kaggle Dataset, for example mounted at `/kaggle/input/sprint4-kaggle-input/`.
2. Add the best parent adapter as a Kaggle input if continuing from the public 0.85 adapter.
3. Copy or upload the scripts in `kaggle_sprint4/` to the notebook.
4. Run the retention-safe first pass:

```bash
python kaggle_train_curated_adapter.py --config sprint4_config_retention.yaml
```

5. Run behavioral validation before packaging. The packaging script now blocks by default unless a behavioral report shows the continued child adapter does not regress against the parent on `validation_family_balanced.jsonl`.

```bash
python kaggle_behavioral_eval.py \
  --mode parent \
  --validation-file /kaggle/input/datasets/surtr19/sprint4-kaggle-input/validation_family_balanced.jsonl \
  --base-model-path /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1 \
  --parent-adapter-path /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20 \
  --predictions-output-json /kaggle/working/parent_predictions.json
python kaggle_behavioral_eval.py \
  --mode child \
  --validation-file /kaggle/input/datasets/surtr19/sprint4-kaggle-input/validation_family_balanced.jsonl \
  --base-model-path /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1 \
  --child-adapter-path /kaggle/working/custom_adapter \
  --predictions-output-json /kaggle/working/child_predictions.json
python kaggle_behavioral_eval.py \
  --validation-file /kaggle/input/datasets/surtr19/sprint4-kaggle-input/validation_family_balanced.jsonl \
  --parent-predictions-json /kaggle/working/parent_predictions.json \
  --child-predictions-json /kaggle/working/child_predictions.json \
  --output-report-path /kaggle/working/sprint4_behavioral_eval.json
python kaggle_package_adapter.py \
  --adapter-dir /kaggle/working/custom_adapter \
  --zip-path /kaggle/working/submission.zip \
  --behavioral-report-path /kaggle/working/sprint4_behavioral_eval.json
python kaggle_validate_adapter.py \
  --adapter-dir /kaggle/working/custom_adapter \
  --zip-path /kaggle/working/submission.zip \
  --behavioral-report-path /kaggle/working/sprint4_behavioral_eval.json
```

6. Manually submit `/kaggle/working/submission.zip` only if validation prints `STRUCTURE_READY`, `BEHAVIOR_READY`, and `READY_TO_SUBMIT`.

For emergency public-baseline preservation, package the parent adapter directly with:

```bash
python kaggle_package_adapter.py --adapter-dir /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20 --zip-path /kaggle/working/submission.zip --baseline-adapter-only
```

## Presets

Use `variant_name` and mapping-style `dataset_mixture` weights in the YAML configs. All retention-safe configs use assistant-only loss and raw answer format by default, so prompt tokens are not supervised.

Retention-safe first run:

```yaml
variant_name: retention_raw_family_tagged
dataset_mixture:
  train_family_tagged.jsonl: 1.0
learning_rate: 0.000001
assistant_only_loss: true
answer_format: raw
```

Distilled low-LR run:

```yaml
variant_name: distilled_low_lr
dataset_mixture:
  train_solver_distilled.jsonl: 1.0
learning_rate: 0.000001
assistant_only_loss: true
answer_format: raw
```

Hard low-LR run:

```yaml
variant_name: hard_low_lr
dataset_mixture:
  train_solver_distilled.jsonl: 0.7
  train_hard_oversampled.jsonl: 0.3
learning_rate: 0.0000005
assistant_only_loss: true
answer_format: raw
```

Legacy variants are still available for controlled experiments, but learning rates above `2e-6` require `allow_high_lr: true`.

Variant A:

```yaml
variant_name: variant_a_family_tagged
dataset_mixture:
  - train_family_tagged.jsonl
```

Variant B:

```yaml
variant_name: variant_b_distilled
dataset_mixture:
  - train_family_tagged.jsonl
  - train_solver_distilled.jsonl
```

Variant C:

```yaml
variant_name: variant_c_hard_oversampled
dataset_mixture:
  - train_family_tagged.jsonl
  - train_solver_distilled.jsonl
  - train_hard_oversampled.jsonl
```

Variant D:

```yaml
variant_name: variant_d_direct_hard
dataset_mixture:
  - train_direct.jsonl
  - train_hard_oversampled.jsonl
```

## Expected risks

- The 30B base model may not fit every Kaggle GPU configuration.
- CUTLASS/CUTE import setup is required for the NVIDIA runtime path.
- Behavioral validation is local to the curated validation split; it cannot guarantee public leaderboard improvement.
- A public 0.85 baseline does not guarantee improvement; package only when the child is not worse than the parent locally.
- The workflow saves adapter-only outputs and avoids full model checkpoints.
