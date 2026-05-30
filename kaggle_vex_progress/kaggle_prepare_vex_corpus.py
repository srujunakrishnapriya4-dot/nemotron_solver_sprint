from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_simple_yaml(path: str | Path) -> dict:
    payload = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = parse_scalar(value.strip())
    return payload


def parse_scalar(value: str):
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"none", "null", ""}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(ch in value.lower() for ch in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def tokenize_corpus(config: dict) -> dict:
    mode = str(config.get("mode", "public_safe_micro"))
    if mode == "vex_replica_if_inputs_available":
        validate_vex_private_inputs(config)
        raise SystemExit("VEX replica mode expects pretokenized VEX corpus/swap inputs; use the replica merge notebook path, not public-safe tokenizer fallback.")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = []
    source = Path(config["corpus_path"])
    max_rows = int(config["max_rows"])
    max_len = int(config["max_seq_len"])
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                rows.append(payload)
                if len(rows) >= max_rows:
                    break
    output = Path(config["token_corpus_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = {"rows": 0, "zero_supervised_rows": 0, "total_supervised_tokens": 0, "decoded_supervised_spans": []}
    with output.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            text = row["text"]
            answer = row["answer"]
            answer_index = text.rfind(answer)
            if answer_index < 0 or text.rfind("Assistant:") > answer_index:
                raise SystemExit(f"masking failure for row {row.get('row_id')}: answer absent from assistant span")
            prefix = text[:answer_index]
            prefix_ids = tokenizer(prefix, truncation=True, max_length=max_len, padding=False)["input_ids"]
            full = tokenizer(text, truncation=True, max_length=max_len, padding=False)["input_ids"]
            weights = [0.0] * len(full)
            for idx in range(min(len(prefix_ids), len(full)), len(full)):
                weights[idx] = 1.0
            supervised = sum(1 for value in weights if value > 0)
            if supervised <= 0:
                diagnostics["zero_supervised_rows"] += 1
            diagnostics["rows"] += 1
            diagnostics["total_supervised_tokens"] += supervised
            if len(diagnostics["decoded_supervised_spans"]) < 2:
                supervised_ids = [token for token, weight in zip(full, weights) if weight > 0]
                diagnostics["decoded_supervised_spans"].append(tokenizer.decode(supervised_ids))
            out.write(json.dumps({"input_ids": full, "target_ids": list(full), "loss_weights": weights, "metadata": row}, sort_keys=True, separators=(",", ":")) + "\n")
    if diagnostics["zero_supervised_rows"]:
        raise SystemExit(f"zero supervised rows: {diagnostics['zero_supervised_rows']}")
    manifest = {"token_corpus_path": str(output), "diagnostics": diagnostics}
    Path("/kaggle/working/vex_tokens/token_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return manifest


def validate_vex_private_inputs(config: dict) -> None:
    required = [
        "corpus_path",
        "train_order_path",
        "swap_cryptarithm_thk_path",
        "swap_v6_path",
        "swap_binary_fmt_path",
        "extra_hyprevise_synth_path",
        "heldout_file",
    ]
    missing = [str(config.get(key)) for key in required if not config.get(key) or not Path(str(config.get(key))).exists()]
    if missing:
        raise SystemExit(f"missing VEX private/token-swap inputs; full replica mode blocked: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vex_config_micro.yaml")
    args = parser.parse_args()
    tokenize_corpus(load_simple_yaml(args.config))


if __name__ == "__main__":
    main()
