# Anti-0.86 Win-Mode Notebook Flow

This is a gated execution plan, not a blind training script.

1. Run `python kaggle_backend_probe.py`.
2. Run `python kaggle_winmode_orchestrator.py --stage validate_artifacts`.
3. Run `python kaggle_winmode_orchestrator.py --stage prepare_micro`.
4. Run `python kaggle_winmode_orchestrator.py --stage train_micro`.
5. Run `python kaggle_winmode_orchestrator.py --stage eval_micro`.
6. Continue to v1/v2/v3 only when the previous gate report exists and passes.
7. Package only through `python kaggle_winmode_orchestrator.py --stage package_final`.

Forbidden:
- no full-prompt loss
- no generic HuggingFace Trainer final path
- no run-all training command
- no submission automation
- no packaging custom adapters without eval gates

The credible target is hard-family lift over the known parent, not synthetic volume.
