from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def load_simple_yaml(path: str | Path) -> dict:
    payload = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = _scalar(value.strip())
    return payload


def stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_token_mask_rows(rows: list[dict]) -> dict:
    row_count = len(rows)
    zero_supervised = 0
    prompt_leakage = 0
    supervised_tokens = 0
    total_tokens = 0
    missing_loss_weights = 0
    for row in rows:
        input_ids = row.get("input_ids") or []
        target_ids = row.get("target_ids") or []
        weights = row.get("loss_weights")
        if weights is None:
            missing_loss_weights += 1
            continue
        if len(input_ids) != len(target_ids) or len(input_ids) != len(weights):
            raise SystemExit("token mask contract failed: length mismatch")
        prompt_count = int(row.get("prompt_token_count", 0))
        sup = sum(1 for value in weights if float(value) > 0)
        supervised_tokens += sup
        total_tokens += len(input_ids)
        zero_supervised += sup == 0
        if any(float(value) > 0 for value in weights[:prompt_count]):
            prompt_leakage += 1
    if missing_loss_weights:
        raise SystemExit(f"token mask contract failed: missing loss weights in {missing_loss_weights} rows")
    if zero_supervised:
        raise SystemExit(f"token mask contract failed: zero supervised rows {zero_supervised}")
    if prompt_leakage:
        raise SystemExit(f"token mask contract failed: prompt leakage {prompt_leakage}")
    return {
        "row_count": row_count,
        "total_tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "supervised_token_count": supervised_tokens,
        "zero_supervised_rows": zero_supervised,
        "prompt_leakage": prompt_leakage,
    }


def _scalar(value: str):
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(ch in value.lower() for ch in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="anti086_config_micro.yaml")
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    from transformers import AutoTokenizer

    root = _resolve_input_root(config)
    source = root / str(config["corpus_file"])
    if not source.exists() and str(config["corpus_file"]) == "winmode_micro.jsonl":
        source = root / "win_micro.jsonl"
    if not source.exists() and str(config["corpus_file"]) == "corpus_anti086_v1.jsonl":
        source = root / "win_v1.jsonl"
    if not source.exists() and str(config["corpus_file"]) == "corpus_anti086_micro.jsonl":
        source = root / "win_micro.jsonl"
    if not source.exists():
        source = Path("artifacts/anti086") / str(config["corpus_file"])
    if not source.exists():
        source = Path("artifacts/win_system") / str(config["corpus_file"]).replace("winmode_", "win_")
    if not source.exists():
        raise SystemExit(f"missing micro corpus: {source}")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"], trust_remote_code=True)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= int(config["max_rows"]):
                    break
    output = Path(config["token_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    zero = 0
    with output.open("w", encoding="utf-8", newline="\n") as out:
        token_rows = []
        for row in rows:
            row = transform_direct_answer_row(row) if bool(config.get("direct_answer_only", False)) else row
            text = row["text"]
            answer = str(row["answer"])
            if bool(config.get("direct_answer_only", False)) and _assistant_has_reasoning(text, answer):
                raise SystemExit("direct-answer target contains reasoning or extra assistant text")
            idx = text.rfind(answer)
            if idx < text.rfind("Assistant:"):
                raise SystemExit("answer absent from assistant span")
            prefix_ids = tokenizer(text[:idx], truncation=True, max_length=int(config["max_seq_len"]), padding=False)["input_ids"]
            input_ids = tokenizer(text, truncation=True, max_length=int(config["max_seq_len"]), padding=False)["input_ids"]
            weights = [0.0] * len(input_ids)
            for pos in range(min(len(prefix_ids), len(input_ids)), len(input_ids)):
                weights[pos] = 1.0
            zero += sum(weights) == 0
            token_row = {"input_ids": input_ids, "target_ids": input_ids, "loss_weights": weights, "prompt_token_count": min(len(prefix_ids), len(input_ids)), "metadata": row}
            token_rows.append(token_row)
            out.write(json.dumps(token_row, sort_keys=True, separators=(",", ":")) + "\n")
    if zero:
        raise SystemExit(f"zero supervised rows: {zero}")
    manifest = validate_token_mask_rows(token_rows)
    manifest.update({"max_seq_len": int(config["max_seq_len"]), "corpus_hash": stable_hash([row.get("id") or row.get("source_id") for row in rows]), "token_output": str(output)})
    manifest_path = Path(str(config.get("token_manifest_path", output.parent / "token_manifest.json")))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


def transform_direct_answer_row(row: dict) -> dict:
    answer = str(row.get("answer", "")).strip()
    if not answer:
        raise SystemExit("direct-answer row missing answer")
    prompt = str(row.get("prompt") or _extract_user_prompt(str(row.get("text", "")))).strip()
    if not prompt:
        raise SystemExit("direct-answer row missing prompt")
    out = dict(row)
    out["text"] = f"User:\n{prompt}\nRespond with only the final answer. No explanation.\nAssistant:\n{answer}"
    out["answer"] = answer
    out["corpus_type"] = "direct_answer_only"
    return out


def _extract_user_prompt(text: str) -> str:
    if "Assistant:" in text:
        text = text.split("Assistant:", 1)[0]
    if text.startswith("User:"):
        text = text[len("User:") :]
    return text.strip()


def _assistant_has_reasoning(text: str, answer: str) -> bool:
    assistant = text.split("Assistant:", 1)[-1].strip()
    if assistant != answer.strip():
        return True
    lowered = assistant.lower()
    return any(marker in lowered for marker in ("because", "therefore", "rule:", "explanation", "answer:"))


def _resolve_input_root(config: dict) -> Path:
    return resolve_anti086_input_root(config)


def _write_conservative_quarantine_overlay(root: Path) -> None:
    manifest = root / "equation_quarantine_manifest.json"
    if manifest.exists():
        return
    manifest.write_text(
        json.dumps(
            {
                "equation_status": "OFFLINE_SYNTHESIS_REQUIRED",
                "overlay_created": True,
                "safe_verified_trace_count": 0,
                "unsafe_trace_excluded_count": 0,
                "direct_answer_only_count": 0,
                "note": "Conservative Kaggle overlay only; missing manifest does not mark unsafe equation traces safe.",
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def resolve_anti086_input_root(config: dict | None = None) -> Path:
    config = config or {}
    configured = str(config.get("anti086_input_root", "auto"))
    if configured != "auto":
        root = Path(configured)
        root.mkdir(parents=True, exist_ok=True) if str(root).startswith("/kaggle/working") else None
        _write_conservative_quarantine_overlay(root)
        return root
    work = Path("/kaggle/working/anti086_input")
    marker_names = {
        "curriculum_manifest.json",
        "win_corpus_manifest.json",
        "win_micro.jsonl",
        "corpus_anti086_micro.jsonl",
        "corpus_anti086_v1.jsonl",
        "win_v1.jsonl",
    }
    input_base = Path("/kaggle/input")
    if input_base.exists():
        for zip_path in sorted(input_base.rglob("*.zip")):
            if "anti086" in zip_path.name or "win_system" in zip_path.name:
                work.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as archive:
                    archive.extractall(work)
                _write_conservative_quarantine_overlay(work)
                return work
        for marker in marker_names:
            for path in sorted(input_base.rglob(marker)):
                _write_conservative_quarantine_overlay(path.parent)
                return path.parent
    candidates = [
        work,
        Path("/kaggle/input/anti086"),
        Path("/kaggle/input/win-system"),
        Path("artifacts/anti086"),
        Path("artifacts/win_system"),
    ]
    for candidate in candidates:
        if candidate.exists() and any((candidate / name).exists() for name in marker_names):
            _write_conservative_quarantine_overlay(candidate)
            return candidate
    work.mkdir(parents=True, exist_ok=True)
    _write_conservative_quarantine_overlay(work)
    return work


if __name__ == "__main__":
    main()
