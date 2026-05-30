# VEX/Progress-Style Nemotron Training Bridge

This is the replacement architecture for the failed generic SPRINT-4 Trainer continuation. It does not guarantee `0.95`. It exists to make a real `0.95` chase technically sane: verified corpus, pretokenized masked training rows, weighted token loss, VEX-style token-swap recipe support, fast vLLM-style eval, and a strict package gate.

The real VEX replica mode requires the private/Kaggle token corpora and swap inputs extracted from the notebooks:

- huikang base token corpus
- train order index
- cryptarithm THK swap
- V6 cryptarithm swap
- binary-format swap
- hyp-revise binary synth
- VEX heldout ID file

If these are missing, `vex_replica_if_inputs_available` must refuse to run. Use `public_safe_micro` only as a pipeline sanity check.

## Sequence

1. Upload `artifacts/vex_progress/vex_kaggle_input.zip` as a Kaggle Dataset.
2. Prepare token corpus before model load:

```bash
python kaggle_prepare_vex_corpus.py --config vex_config_micro.yaml
```

3. Run micro training only:

```bash
python kaggle_train_vex_style_sft.py --config vex_config_micro.yaml
```

4. Smoke eval:

```bash
python kaggle_eval_vex_style_vllm.py --mode smoke_16 --config vex_config_micro.yaml
```

5. Only if micro loss, adapter size, and smoke eval are sane, and the VEX private/token-swap inputs are mounted, run main config:

```bash
python kaggle_prepare_vex_corpus.py --config vex_config_main.yaml
python kaggle_train_vex_style_sft.py --config vex_config_main.yaml
python kaggle_eval_vex_style_vllm.py --mode family_60 --config vex_config_main.yaml
```

6. Package only if gate passes:

```bash
python kaggle_package_vex_adapter.py --config vex_config_main.yaml
python kaggle_validate_vex_adapter.py --zip-path /kaggle/working/submission.zip
```

## Hard Rules

- No full-prompt loss path exists.
- No generic HuggingFace `Trainer`.
- No fake adapters.
- No test labels.
- No public/private/test ID hardcoding.
- No packaging without rank, adapter size, loss, and eval evidence.
- No 4GB custom adapter unless it matches a known-good VEX/Tinker structure and passes the package gate.
- No silent replacement of VEX private data with public-safe corpus unless `public_safe_fallback` is explicitly selected.
