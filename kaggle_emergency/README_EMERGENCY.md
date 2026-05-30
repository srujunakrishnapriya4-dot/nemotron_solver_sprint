# Emergency Score Recovery Runbook

This is the low-risk path after the failed SPRINT-4 continuation runs.

## Blunt Strategy

Current code cannot guarantee `0.95` public or private. Under the remaining time/GPU constraints, the realistic high-probability band is the known `0.85` parent adapter plus any better public/tinker adapter discovered by sweep. A real `0.95` chase needs a genuinely better adapter or a validated training recipe, not another blind 30B continuation run.

## Notebook Sequence

1. Find every adapter available in Kaggle inputs and working storage:

```bash
python kaggle_find_adapters.py
```

2. Package the known-good public parent adapter first:

```bash
python kaggle_package_existing_adapter.py \
  --adapter-dir /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20 \
  --zip-path /kaggle/working/submission.zip
```

3. Submit the baseline manually if no better candidate exists.

4. Create and maintain the manual scorebook:

```bash
python kaggle_scorebook_template.py
```

5. For each available public/tinker adapter:
   - package it without training,
   - submit manually,
   - record public score in `/kaggle/working/adapter_scorebook.json`.

6. Do not train unless the path is already proven. If training is attempted:
   - micro-run only,
   - assistant-only loss,
   - low LR,
   - adapter size guard,
   - fast probe,
   - decision gate.

7. Fast probe before any custom child submission:

```bash
python kaggle_probe_fast.py \
  --adapter-dir /kaggle/working/custom_adapter \
  --base-model-path /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1 \
  --validation-file /kaggle/input/datasets/surtr19/sprint4-kaggle-input/validation_family_balanced.jsonl \
  --output-json /kaggle/working/fast_probe.json
```

8. Decision gate:

```bash
python kaggle_decide_submit.py \
  --adapter-dir /kaggle/working/custom_adapter \
  --probe-json /kaggle/working/fast_probe.json \
  --scorebook-json /kaggle/working/adapter_scorebook.json \
  --operator-selected
```

Obey `DO_NOT_SUBMIT`.

## Stop Immediately

- Blind 30B continuation training.
- Full-prompt loss.
- Packaging child adapters with unknown behavior.
- Huge custom adapters without known-good structure.
- Slow 300-row generation eval as the first gate.

## Fallback

If no better candidate appears, submit the known-good public parent adapter direct package.
