from __future__ import annotations

from .hard_family_router import RuleHypothesis


def solve_cipher_symbol_problem(examples: tuple[tuple[str, str], ...], target_input: str) -> RuleHypothesis:
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for enc, dec in examples:
        if len(enc) != len(dec):
            return _token_bijection(examples, target_input)
        for e, d in zip(enc, dec):
            if e in mapping and mapping[e] != d:
                return _reject("mapping_collision")
            if d in reverse and reverse[d] != e:
                return _reject("reverse_mapping_collision")
            mapping[e] = d
            reverse[d] = e
    if any(ch not in mapping for ch in target_input):
        return _reject("unseen_target_symbol")
    pred = "".join(mapping[ch] for ch in target_input)
    return RuleHypothesis("cipher_symbol_solver", "char_bijection", pred, True, "verified_unique", {"mapping_size": len(mapping)})


def _token_bijection(examples, target_input):
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for enc, dec in examples:
        et = enc.split()
        dt = dec.split()
        if len(et) != len(dt):
            return _reject("unsupported_token_shape")
        for e, d in zip(et, dt):
            if e in mapping and mapping[e] != d:
                return _reject("mapping_collision")
            if d in reverse and reverse[d] != e:
                return _reject("reverse_mapping_collision")
            mapping[e] = d
            reverse[d] = e
    toks = target_input.split()
    if any(tok not in mapping for tok in toks):
        return _reject("unseen_target_token")
    return RuleHypothesis("cipher_symbol_solver", "token_bijection", " ".join(mapping[t] for t in toks), True, "verified_unique", {"mapping_size": len(mapping)})


def _reject(reason: str) -> RuleHypothesis:
    return RuleHypothesis("cipher_symbol_solver", "reject", None, False, reason, {})
