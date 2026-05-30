"""Core schemas, manifests, and registry APIs for Pass 2."""

from .manifest import (
    create_manifest,
    hash_file,
    hash_json_obj,
    hash_text,
    read_manifest,
    validate_manifest_chain,
    write_manifest,
)
from .registry import (
    load_registry_jsonl,
    quarantine_problem,
    register_problem,
    split_allows_eval,
    split_allows_sft,
    validate_registry,
    write_registry_jsonl,
)
from .schemas import *

__all__ = [
    "create_manifest",
    "hash_file",
    "hash_json_obj",
    "hash_text",
    "load_registry_jsonl",
    "quarantine_problem",
    "read_manifest",
    "register_problem",
    "split_allows_eval",
    "split_allows_sft",
    "validate_manifest_chain",
    "validate_registry",
    "write_manifest",
    "write_registry_jsonl",
] + [name for name in globals() if name[0].isupper() or name in {"stable_hash", "stable_json_dumps"}]
