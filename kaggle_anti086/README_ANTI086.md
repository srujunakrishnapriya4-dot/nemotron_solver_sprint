# Anti-0.86 Barrier System

This system does not break 0.86 by itself. It creates the mechanism needed to test whether a custom adapter can break it: private-like validation, hard verified synthetic rows, contrastive anti-shortcut examples, and gated promotion.

## Sequence

1. Build private-like benchmark and anti086 corpus:

```bash
python kaggle_build_private_like_benchmark.py
python kaggle_build_anti086_corpus.py --config anti086_config_micro.yaml
```

2. Prepare masked tokens:

```bash
python kaggle_prepare_anti086_tokens.py --config anti086_config_micro.yaml
```

3. Train micro only:

```bash
python kaggle_train_anti086_adapter.py --config anti086_config_micro.yaml
```

4. Eval with vLLM:

```bash
python kaggle_eval_anti086_vllm.py --config anti086_config_micro.yaml --mode smoke_16
```

5. Promote to v1/v2/v3 only if gates pass. Do not package without private-like validation.

## Stop Conditions

- token mask invalid
- loss not finite
- empty or prompt-copy smoke outputs
- public-like collapse
- private-like no improvement
- synthetic-only gains
- package not clean
