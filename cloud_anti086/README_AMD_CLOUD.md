# AMD Cloud Anti-0.86 Use

Use AMD credits for deterministic data work:

- build adversarial synthetic rows
- build corpora
- validate masks and token manifests if tokenizer is available
- run unit tests

Do not use AMD for final Nemotron training unless `cloud_backend_probe.py` proves ROCm, PyTorch, PEFT, vLLM, tokenizer, and model stack compatibility.
