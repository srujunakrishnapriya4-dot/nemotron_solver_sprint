from __future__ import annotations

import itertools
import math
import re
import statistics
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


BITS = 8
MASK = 0xFF


def norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# =============================================================================
# Shared numeric teachers already strong enough: roman / unit / gravity
# =============================================================================

ROMAN_TABLE = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def roman_from_int(n: int) -> str:
    if n <= 0 or n >= 4000:
        raise ValueError(f"roman_out_of_range:{n}")
    out = []
    r = n
    for val, sym in ROMAN_TABLE:
        while r >= val:
            out.append(sym)
            r -= val
    return "".join(out)


def parse_last_int_query(prompt: str) -> Optional[int]:
    tail = prompt[-900:]
    nums = re.findall(r"-?\d+", tail)
    if not nums:
        return None
    return int(nums[-1])


def roman_teacher(prompt: str, gold: str) -> Dict[str, Any]:
    q = parse_last_int_query(prompt)
    if q is None:
        return {"covered": False, "reason": "roman_query_int_not_parsed"}
    try:
        pred = roman_from_int(q)
    except Exception as e:
        return {"covered": False, "reason": str(e)}
    ok = pred == norm_text(gold)
    return {"covered": ok, "prediction": pred, "reason": "verified" if ok else "prediction_mismatch"}


def decimal_places_from_gold(gold: str) -> int:
    gold = norm_text(gold)
    if "." not in gold:
        return 0
    return len(gold.split(".")[-1])


def format_float_like_gold(value: float, gold: str) -> str:
    places = decimal_places_from_gold(gold)
    if places == 0:
        return str(int(round(value)))
    return f"{value:.{places}f}"


def parse_becomes_examples(prompt: str) -> List[Tuple[float, float]]:
    pairs = []
    for m in re.finditer(
        r"(-?\d+(?:\.\d+)?)\s*[A-Za-z/%°]*\s+becomes\s+(-?\d+(?:\.\d+)?)",
        prompt,
        flags=re.IGNORECASE,
    ):
        a = safe_float(m.group(1))
        b = safe_float(m.group(2))
        if a is not None and b is not None and abs(a) > 1e-12:
            pairs.append((a, b))
    return pairs


def parse_unit_query(prompt: str) -> Optional[float]:
    tail = prompt[-700:]
    m = re.search(r"Now\s+convert\s*:?\s*(-?\d+(?:\.\d+)?)", tail, flags=re.IGNORECASE)
    if m:
        return safe_float(m.group(1))
    nums = re.findall(r"-?\d+(?:\.\d+)?", tail)
    return safe_float(nums[-1]) if nums else None


def unit_conversion_teacher(prompt: str, gold: str) -> Dict[str, Any]:
    pairs = parse_becomes_examples(prompt)
    q = parse_unit_query(prompt)
    if len(pairs) < 2:
        return {"covered": False, "reason": "unit_not_enough_examples"}
    if q is None:
        return {"covered": False, "reason": "unit_query_not_parsed"}

    ratios = [b / a for a, b in pairs if abs(a) > 1e-12]
    if not ratios:
        return {"covered": False, "reason": "unit_no_valid_ratios"}

    # Median is robust to round-off noise.
    k = statistics.median(ratios)
    pred = format_float_like_gold(q * k, gold)
    ok = pred == norm_text(gold)
    return {
        "covered": ok,
        "prediction": pred,
        "scale": k,
        "reason": "verified" if ok else "prediction_mismatch",
    }


def parse_gravity_examples(prompt: str) -> List[Tuple[float, float]]:
    examples = []
    for m in re.finditer(
        r"t\s*=\s*(-?\d+(?:\.\d+)?)\s*s?[^.\n\r]*?distance\s*=\s*(-?\d+(?:\.\d+)?)",
        prompt,
        flags=re.IGNORECASE,
    ):
        t = safe_float(m.group(1))
        d = safe_float(m.group(2))
        if t is not None and d is not None and abs(t) > 1e-12:
            examples.append((t, d))
    return examples


def parse_gravity_query(prompt: str) -> Optional[float]:
    tail = prompt[-800:]
    vals = re.findall(r"t\s*=\s*(-?\d+(?:\.\d+)?)\s*s?", tail, flags=re.IGNORECASE)
    return safe_float(vals[-1]) if vals else None


def gravity_teacher(prompt: str, gold: str) -> Dict[str, Any]:
    examples = parse_gravity_examples(prompt)
    q = parse_gravity_query(prompt)
    if len(examples) < 2:
        return {"covered": False, "reason": "gravity_not_enough_examples"}
    if q is None:
        return {"covered": False, "reason": "gravity_query_not_parsed"}

    gs = [2.0 * d / (t * t) for t, d in examples if abs(t) > 1e-12]
    if not gs:
        return {"covered": False, "reason": "gravity_no_valid_g"}

    g = statistics.median(gs)
    pred = format_float_like_gold(0.5 * g * q * q, gold)
    ok = pred == norm_text(gold)
    return {
        "covered": ok,
        "prediction": pred,
        "g": g,
        "reason": "verified" if ok else "prediction_mismatch",
    }


# =============================================================================
# Word cipher teacher
# =============================================================================

def _between_examples_and_now(prompt: str) -> str:
    p = str(prompt)
    m = re.search(r"examples?\s*:?\s*(.*?)(?:Now[, ]+|Now\s+)", p, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: everything before Now.
    return re.split(r"\bNow\b", p, flags=re.IGNORECASE)[0]


def _query_after(prompt: str, phrase: str) -> Optional[str]:
    # Generic fallback used by older tests.
    m = re.search(phrase + r"\s*:?\s*(.+)$", prompt, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    q = m.group(1)
    q = re.split(r"\n|Answer\s*:", q, flags=re.IGNORECASE)[0]
    return norm_text(q)


def parse_word_cipher_query(prompt: str) -> Optional[str]:
    # Public train style includes variants like:
    #   Now, decrypt the following text: ...
    #   Now decrypt: ...
    #   Now, decrypt: ...
    patterns = [
        r"Now\s*,?\s*decrypt\s+the\s+following\s+text\s*:?\s*(.+)$",
        r"Now\s*,?\s*decrypt\s*:?\s*(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, prompt, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        q = m.group(1)
        q = re.split(r"\n|Answer\s*:", q, flags=re.IGNORECASE)[0]
        q = norm_text(q)
        if q:
            return q
    return None


def _words(x: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", x.lower())


def _update_bijection(cmap: Dict[str, str], rmap: Dict[str, str], enc: str, dec: str) -> bool:
    for a, b in zip(enc, dec):
        if a.isspace() and b.isspace():
            continue
        if a.isspace() != b.isspace():
            return False
        if not a.isalpha() or not b.isalpha():
            continue
        if a in cmap and cmap[a] != b:
            return False
        if b in rmap and rmap[b] != a:
            return False
        cmap[a] = b
        rmap[b] = a
    return True


def _candidate_cipher_pair_splits(body: str) -> List[List[Tuple[str, str]]]:
    """
    Public word_cipher prompts are often one paragraph:
      enc words -> plain words enc words -> plain words ... Now decrypt: ...

    This parser searches phrase split points instead of assuming line breaks.
    It is intentionally conservative: it keeps only splits where each encrypted word
    aligns length-wise with each plaintext word, enabling character substitution.
    """
    body = norm_text(body)
    if "->" not in body:
        return []

    parts = [p.strip() for p in body.split("->")]
    if len(parts) < 2:
        return []

    token_parts = [_words(p) for p in parts]
    if not token_parts[0] or not token_parts[-1]:
        return []

    # Initial left encrypted phrase: last k words before first arrow.
    states = []
    for k in range(1, min(7, len(token_parts[0])) + 1):
        enc0 = token_parts[0][-k:]
        states.append({
            "pairs": [],
            "pending_enc": enc0,
            "cmap": {},
            "rmap": {},
        })

    # Intermediate segments: dec_i + enc_{i+1}
    for idx in range(1, len(token_parts) - 1):
        toks = token_parts[idx]
        new_states = []
        for st in states:
            pending_enc = st["pending_enc"]
            # split toks into dec words then next enc words
            for dec_len in range(1, min(7, len(toks)) + 1):
                dec = toks[:dec_len]
                enc_next = toks[dec_len:]
                if not enc_next:
                    continue
                if len(dec) != len(pending_enc):
                    continue
                if any(len(a) != len(b) for a, b in zip(pending_enc, dec)):
                    continue

                cmap = dict(st["cmap"])
                rmap = dict(st["rmap"])
                if not _update_bijection(cmap, rmap, " ".join(pending_enc), " ".join(dec)):
                    continue

                new_states.append({
                    "pairs": st["pairs"] + [(" ".join(pending_enc), " ".join(dec))],
                    "pending_enc": enc_next,
                    "cmap": cmap,
                    "rmap": rmap,
                })
        states = new_states[:500]
        if not states:
            return []

    # Last segment is final plaintext for pending encrypted phrase.
    last_toks = token_parts[-1]
    final_states = []
    for st in states:
        pending_enc = st["pending_enc"]
        dec = last_toks
        if len(dec) != len(pending_enc):
            continue
        if any(len(a) != len(b) for a, b in zip(pending_enc, dec)):
            continue

        cmap = dict(st["cmap"])
        rmap = dict(st["rmap"])
        if not _update_bijection(cmap, rmap, " ".join(pending_enc), " ".join(dec)):
            continue

        final_states.append(st["pairs"] + [(" ".join(pending_enc), " ".join(dec))])

    return final_states[:200]


def _word_cipher_phrase_pair_fallback(body: str, query: str, gold: str) -> Optional[Dict[str, Any]]:
    """
    Conservative fallback for compact examples such as:
      abc def -> cat dog ghi jkl -> red sun

    It does not guess using the target answer alone. It uses the query/gold only
    to verify a candidate word map after building candidate phrase alignments.
    """
    body = norm_text(body).lower()
    query_words = query.lower().split()
    gold_words = gold.lower().split()

    if not query_words or not gold_words or len(query_words) != len(gold_words):
        return None
    if "->" not in body:
        return None

    parts = [p.strip() for p in body.split("->")]
    toks = [_words(p) for p in parts]
    if len(toks) < 2:
        return None

    candidates: List[List[Tuple[str, str]]] = []

    # Case: exactly two arrows:
    #   enc1 -> dec1 enc2 -> dec2
    # Try all splits of the middle segment into dec1 + enc2.
    if len(toks) == 3:
        left = toks[0]
        mid = toks[1]
        right = toks[2]
        for enc1_len in range(1, min(8, len(left)) + 1):
            enc1 = left[-enc1_len:]
            for dec1_len in range(1, min(8, len(mid)) + 1):
                dec1 = mid[:dec1_len]
                enc2 = mid[dec1_len:]
                dec2 = right
                if not enc2:
                    continue
                if len(enc1) != len(dec1):
                    continue
                if len(enc2) != len(dec2):
                    continue
                candidates.append([
                    (" ".join(enc1), " ".join(dec1)),
                    (" ".join(enc2), " ".join(dec2)),
                ])

    # More general fallback: use the stronger split parser too.
    candidates.extend(_candidate_cipher_pair_splits(body))

    for pairs in candidates:
        word_map: Dict[str, str] = {}
        conflict = False
        for enc, dec in pairs:
            ew = enc.split()
            dw = dec.split()
            if len(ew) != len(dw):
                conflict = True
                break
            for a, b in zip(ew, dw):
                if a in word_map and word_map[a] != b:
                    conflict = True
                    break
                word_map[a] = b
            if conflict:
                break
        if conflict:
            continue

        if all(w in word_map for w in query_words):
            pred = norm_text(" ".join(word_map[w] for w in query_words)).lower()
            if pred == gold:
                return {
                    "covered": True,
                    "prediction": pred,
                    "mode": "phrase_word_map_fallback",
                    "reason": "verified",
                    "example_pairs": pairs[:8],
                }

    return None


def word_cipher_teacher(prompt: str, gold: str) -> Dict[str, Any]:
    gold = norm_text(gold).lower()
    query = parse_word_cipher_query(prompt)
    if not query:
        return {"covered": False, "reason": "word_cipher_query_not_parsed"}

    body = _between_examples_and_now(prompt)

    # First: phrase-level fallback. This catches compact inline public examples
    # and the synthetic test case without weakening verification.
    fb = _word_cipher_phrase_pair_fallback(body, query, gold)
    if fb is not None:
        return fb

    split_candidates = _candidate_cipher_pair_splits(body)
    if not split_candidates:
        return {"covered": False, "reason": "word_cipher_examples_not_parsed"}

    best_failure = None

    for pairs in split_candidates:
        # Word map first.
        word_map = {}
        for enc, dec in pairs:
            ew = enc.split()
            dw = dec.split()
            if len(ew) == len(dw):
                for a, b in zip(ew, dw):
                    word_map[a] = b

        qwords = query.lower().split()
        if qwords and all(w in word_map for w in qwords):
            pred = norm_text(" ".join(word_map[w] for w in qwords)).lower()
            if pred == gold:
                return {
                    "covered": True,
                    "prediction": pred,
                    "mode": "word_map",
                    "reason": "verified",
                    "example_pairs": pairs[:6],
                }
            best_failure = {"prediction": pred, "reason": "prediction_mismatch_word_map"}

        # Character substitution.
        cmap: Dict[str, str] = {}
        rmap: Dict[str, str] = {}
        ok = True
        for enc, dec in pairs:
            if len(enc) != len(dec):
                ok = False
                break
            if not _update_bijection(cmap, rmap, enc, dec):
                ok = False
                break
        if not ok:
            continue

        out = []
        missing = []
        for ch in query.lower():
            if ch.isspace():
                out.append(ch)
            elif ch in cmap:
                out.append(cmap[ch])
            else:
                missing.append(ch)
                out.append("?")

        if missing:
            best_failure = {
                "prediction": norm_text("".join(out)),
                "reason": "missing_query_chars",
                "missing_chars": sorted(set(missing)),
            }
            continue

        pred = norm_text("".join(out)).lower()
        if pred == gold:
            return {
                "covered": True,
                "prediction": pred,
                "mode": "char_substitution",
                "reason": "verified",
                "example_pairs": pairs[:6],
            }

        best_failure = {"prediction": pred, "reason": "prediction_mismatch_char_map"}

    return {
        "covered": False,
        "reason": best_failure.get("reason", "word_cipher_no_verified_candidate") if best_failure else "word_cipher_no_verified_candidate",
        **(best_failure or {}),
    }


# =============================================================================
# Bit manipulation teacher
# =============================================================================

def bits_to_int(s: str) -> int:
    return int(s, 2)


def int_to_bits(x: int) -> str:
    return f"{x & MASK:08b}"


def reverse_bits(x: int) -> int:
    b = f"{x & MASK:08b}"[::-1]
    return int(b, 2)


def rol(x: int, k: int) -> int:
    k %= 8
    x &= MASK
    return ((x << k) | (x >> (8 - k))) & MASK


def ror(x: int, k: int) -> int:
    k %= 8
    x &= MASK
    return ((x >> k) | (x << (8 - k))) & MASK


def parse_bit_pairs(prompt: str) -> List[Tuple[int, int]]:
    pairs = []
    for m in re.finditer(r"\b([01]{8})\s*->\s*([01]{8})\b", prompt):
        pairs.append((bits_to_int(m.group(1)), bits_to_int(m.group(2))))
    return pairs


def parse_bit_query(prompt: str) -> Optional[int]:
    tail = prompt[-700:]
    # Query is usually after "Now transform:" or the last binary string in the prompt.
    m = re.search(r"Now[^:]{0,80}:\s*([01]{8})", tail, flags=re.IGNORECASE)
    if m:
        return bits_to_int(m.group(1))
    bins = re.findall(r"\b[01]{8}\b", tail)
    if bins:
        return bits_to_int(bins[-1])
    return None


def base_bit_transforms() -> List[Tuple[str, Callable[[int], int]]]:
    funcs: List[Tuple[str, Callable[[int], int]]] = []
    funcs.append(("identity", lambda x: x & MASK))
    funcs.append(("not", lambda x: (~x) & MASK))
    funcs.append(("reverse", reverse_bits))
    for k in range(1, 8):
        funcs.append((f"rol{k}", lambda x, k=k: rol(x, k)))
        funcs.append((f"ror{k}", lambda x, k=k: ror(x, k)))
        funcs.append((f"shl{k}", lambda x, k=k: (x << k) & MASK))
        funcs.append((f"shr{k}", lambda x, k=k: (x >> k) & MASK))
    return funcs


def bit_manipulation_teacher(prompt: str, gold: str) -> Dict[str, Any]:
    gold = norm_text(gold)
    if not re.fullmatch(r"[01]{8}", gold):
        return {"covered": False, "reason": "gold_not_8bit"}

    pairs = parse_bit_pairs(prompt)
    q = parse_bit_query(prompt)

    if len(pairs) < 2:
        return {"covered": False, "reason": "bit_not_enough_pairs"}
    if q is None:
        return {"covered": False, "reason": "bit_query_not_parsed"}

    candidates = []

    for base_name, f in base_bit_transforms():
        # Plain base transform.
        if all(f(x) == y for x, y in pairs):
            candidates.append((base_name, lambda z, f=f: f(z)))

        # XOR constant after base.
        c = f(pairs[0][0]) ^ pairs[0][1]
        if all((f(x) ^ c) & MASK == y for x, y in pairs):
            candidates.append((f"{base_name}_xor_{c:02x}", lambda z, f=f, c=c: (f(z) ^ c) & MASK))

        # ADD constant after base.
        c = (pairs[0][1] - f(pairs[0][0])) & MASK
        if all((f(x) + c) & MASK == y for x, y in pairs):
            candidates.append((f"{base_name}_add_{c:02x}", lambda z, f=f, c=c: (f(z) + c) & MASK))

        # SUB constant after base.
        c = (f(pairs[0][0]) - pairs[0][1]) & MASK
        if all((f(x) - c) & MASK == y for x, y in pairs):
            candidates.append((f"{base_name}_sub_{c:02x}", lambda z, f=f, c=c: (f(z) - c) & MASK))

    if not candidates:
        return {"covered": False, "reason": "bit_no_candidate_verified"}

    preds: Dict[str, List[str]] = {}
    for name, fn in candidates:
        pred = int_to_bits(fn(q))
        preds.setdefault(pred, []).append(name)

    if len(preds) != 1:
        return {
            "covered": False,
            "reason": "bit_ambiguous_query_predictions",
            "predictions": preds,
        }

    pred = next(iter(preds))
    ok = pred == gold
    return {
        "covered": ok,
        "prediction": pred,
        "candidate_count": len(candidates),
        "candidate_names": list(next(iter(preds.values())))[:10],
        "reason": "verified" if ok else "prediction_mismatch",
    }


# =============================================================================
# Equation transform teacher
# =============================================================================

def parse_equation_transform(prompt: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    p = str(prompt)
    before_now, *after = re.split(r"\bNow\b", p, maxsplit=1, flags=re.IGNORECASE)
    query = None
    if after:
        m = re.search(r"result\s+for\s*:?\s*(\S+)", after[0], flags=re.IGNORECASE)
        if m:
            query = m.group(1).strip()

    pairs = []
    # LHS and RHS are compact no-space tokens in public train.
    for m in re.finditer(r"(\S+)\s*=\s*(\S+)", before_now):
        lhs, rhs = m.group(1).strip(), m.group(2).strip()
        # Skip accidental text fragments.
        if len(lhs) > 0 and len(rhs) > 0 and "http" not in lhs.lower():
            pairs.append((lhs, rhs))

    return pairs, query


def parse_two_number_expr(expr: str) -> Optional[Tuple[int, str, int]]:
    m = re.fullmatch(r"(-?\d+)(\D)(-?\d+)", expr)
    if not m:
        return None
    a = safe_int(m.group(1))
    b = safe_int(m.group(3))
    if a is None or b is None:
        return None
    return a, m.group(2), b



def numeric_expr_candidates(a: int, op: str, b: int) -> Dict[str, str]:
    vals: Dict[str, Any] = {}

    vals["add"] = a + b
    vals["sub"] = a - b
    vals["rsub"] = b - a
    vals["abs_sub"] = abs(a - b)
    vals["mul"] = a * b
    vals["add_plus_1"] = a + b + 1
    vals["add_minus_1"] = a + b - 1
    vals["mul_plus_1"] = a * b + 1
    vals["mul_minus_1"] = a * b - 1

    if b != 0 and a % b == 0:
        vals["div"] = a // b
    if a != 0 and b % a == 0:
        vals["rdiv"] = b // a
    if b != 0:
        vals["mod"] = a % b
    if a != 0:
        vals["rmod"] = b % a

    sa, sb = str(abs(a)), str(abs(b))
    expr = f"{sa}{op}{sb}"
    expr_noop = f"{sa}{sb}"

    # String/digit transforms. These matter more than arithmetic here.
    vals["left"] = sa
    vals["right"] = sb
    vals["concat"] = sa + sb
    vals["rconcat"] = sb + sa
    vals["concat_rev"] = (sa + sb)[::-1]
    vals["rconcat_rev"] = (sb + sa)[::-1]
    vals["left_rev"] = sa[::-1]
    vals["right_rev"] = sb[::-1]
    vals["first_left"] = sa[0]
    vals["last_left"] = sa[-1]
    vals["first_right"] = sb[0]
    vals["last_right"] = sb[-1]
    vals["outer_digits"] = sa[0] + sb[-1]
    vals["inner_digits"] = sa[-1] + sb[0]
    vals["outer_digits_rev"] = sb[-1] + sa[0]
    vals["inner_digits_rev"] = sb[0] + sa[-1]

    # Digit sums/products.
    vals["sum_digits_ab"] = sum(map(int, sa + sb))
    vals["sum_a_plus_sum_b"] = sum(map(int, sa)) + sum(map(int, sb))
    prod = 1
    for ch in sa + sb:
        prod *= int(ch)
    vals["prod_digits_ab"] = prod

    # Preserve operator with computed/string values.
    vals["absdiff_with_op"] = f"{abs(a-b)}{op}"
    vals["op_absdiff"] = f"{op}{abs(a-b)}"
    vals["add_with_op"] = f"{a+b}{op}"
    vals["op_add"] = f"{op}{a+b}"
    vals["sub_with_op"] = f"{a-b}{op}"
    vals["rsub_with_op"] = f"{b-a}{op}"
    vals["left_with_op"] = f"{sa}{op}"
    vals["right_with_op"] = f"{sb}{op}"
    vals["op_left"] = f"{op}{sa}"
    vals["op_right"] = f"{op}{sb}"
    vals["concat_with_op"] = f"{sa}{sb}{op}"
    vals["op_concat"] = f"{op}{sa}{sb}"

    # Expression-level projections.
    vals["expr"] = expr
    vals["expr_rev"] = expr[::-1]
    vals["expr_noop"] = expr_noop
    vals["expr_noop_rev"] = expr_noop[::-1]
    vals["drop_op"] = sa + sb
    vals["drop_left"] = op + sb
    vals["drop_right"] = sa + op

    # Fixed index projections from expression string, including operator.
    chars = list(expr)
    for i in range(len(chars)):
        vals[f"expr_pos_{i}"] = chars[i]
    for i in range(len(chars)):
        for j in range(len(chars)):
            vals[f"expr_pos_{i}_{j}"] = chars[i] + chars[j]
    for i in range(len(chars)):
        for j in range(len(chars)):
            for k in range(len(chars)):
                vals[f"expr_pos_{i}_{j}_{k}"] = chars[i] + chars[j] + chars[k]

    return {k: str(v) for k, v in vals.items()}



def equation_numeric_per_operator_teacher(
    pairs: List[Tuple[str, str]],
    query: str,
    gold: str,
) -> Optional[Dict[str, Any]]:
    """
    Phase 8.6d critical fix.

    Many public equation_transform rows define a SET of operator-specific rules.
    Do not force one arithmetic/string transform across all operators.

    Rule:
      - parse query as a OP b
      - use only training examples with the same OP
      - infer candidate transform from those examples
      - apply to query only if all same-OP examples verify
      - reject if candidate predictions disagree
    """
    q = parse_two_number_expr(query)
    if q is None:
        return None

    qa, qop, qb = q

    same_op_examples: List[Tuple[Tuple[int, str, int], str]] = []
    for lhs, rhs in pairs:
        t = parse_two_number_expr(lhs)
        if t is None:
            continue
        if t[1] == qop:
            same_op_examples.append((t, rhs))

    if not same_op_examples:
        return {
            "covered": False,
            "reason": "equation_perop_no_same_operator_examples",
        }

    candidate_names: Optional[set[str]] = None

    for (a, op, b), rhs in same_op_examples:
        cands = numeric_expr_candidates(a, op, b)
        passing = {name for name, val in cands.items() if val == rhs}
        if candidate_names is None:
            candidate_names = passing
        else:
            candidate_names &= passing

    if not candidate_names:
        return {
            "covered": False,
            "reason": "equation_perop_no_candidate_verified",
            "same_operator_examples": len(same_op_examples),
        }

    q_cands = numeric_expr_candidates(qa, qop, qb)
    preds: Dict[str, List[str]] = {}

    for name in candidate_names:
        if name in q_cands:
            preds.setdefault(q_cands[name], []).append(name)

    if not preds:
        return {
            "covered": False,
            "reason": "equation_perop_no_query_prediction",
            "candidate_count": len(candidate_names),
        }

    if len(preds) != 1:
        return {
            "covered": False,
            "reason": "equation_perop_ambiguous_query_predictions",
            "prediction_count": len(preds),
            "sample_predictions": {k: v[:5] for k, v in list(preds.items())[:8]},
            "same_operator_examples": len(same_op_examples),
        }

    pred = next(iter(preds))
    ok = pred == norm_text(gold)

    return {
        "covered": ok,
        "prediction": pred,
        "candidate_count": len(candidate_names),
        "candidate_names": list(next(iter(preds.values())))[:10],
        "same_operator_examples": len(same_op_examples),
        "reason": "verified" if ok else "prediction_mismatch",
    }


def equation_numeric_teacher(pairs: List[Tuple[str, str]], query: str, gold: str) -> Optional[Dict[str, Any]]:
    parsed = []
    for lhs, rhs in pairs:
        t = parse_two_number_expr(lhs)
        if t is None:
            return None
        parsed.append((t, rhs))

    q = parse_two_number_expr(query)
    if q is None:
        return None

    candidate_names = None
    for (a, op, b), rhs in parsed:
        cands = numeric_expr_candidates(a, op, b)
        passing = {name for name, val in cands.items() if val == rhs}
        if candidate_names is None:
            candidate_names = passing
        else:
            candidate_names &= passing

    if not candidate_names:
        return {
            "covered": False,
            "reason": "equation_numeric_no_candidate_verified",
        }

    preds = {}
    q_cands = numeric_expr_candidates(q[0], q[1], q[2])
    for name in candidate_names:
        if name in q_cands:
            preds.setdefault(q_cands[name], []).append(name)

    if len(preds) != 1:
        return {
            "covered": False,
            "reason": "equation_numeric_ambiguous_query_predictions",
            "predictions": preds,
        }

    pred = next(iter(preds))
    ok = pred == norm_text(gold)
    return {
        "covered": ok,
        "prediction": pred,
        "candidate_names": list(next(iter(preds.values()))),
        "reason": "verified" if ok else "prediction_mismatch",
    }


def learn_symbol_map_for_positions(
    pairs: List[Tuple[str, str]],
    positions: Tuple[int, ...],
) -> Optional[Dict[str, str]]:
    cmap: Dict[str, str] = {}
    rmap: Dict[str, str] = {}

    for lhs, rhs in pairs:
        if max(positions) >= len(lhs):
            return None
        if len(rhs) != len(positions):
            return None

        selected = "".join(lhs[i] for i in positions)
        for a, b in zip(selected, rhs):
            if a in cmap and cmap[a] != b:
                return None
            if b in rmap and rmap[b] != a:
                return None
            cmap[a] = b
            rmap[b] = a

    return cmap



def _learn_map_for_selected(
    pairs: List[Tuple[str, str]],
    selector: Callable[[str], Optional[str]],
) -> Optional[Dict[str, str]]:
    cmap: Dict[str, str] = {}
    rmap: Dict[str, str] = {}

    for lhs, rhs in pairs:
        selected = selector(lhs)
        if selected is None:
            return None
        if len(selected) != len(rhs):
            return None

        for a, b in zip(selected, rhs):
            if a in cmap and cmap[a] != b:
                return None
            if b in rmap and rmap[b] != a:
                return None
            cmap[a] = b
            rmap[b] = a

    return cmap


def _apply_char_map(seq: str, cmap: Dict[str, str]) -> Optional[str]:
    out = []
    for ch in seq:
        if ch not in cmap:
            return None
        out.append(cmap[ch])
    return "".join(out)


def _all_same_rhs(pairs: List[Tuple[str, str]]) -> Optional[str]:
    vals = {rhs for _, rhs in pairs}
    if len(vals) == 1:
        return next(iter(vals))
    return None


def equation_symbolic_teacher(pairs: List[Tuple[str, str]], query: str, gold: str) -> Optional[Dict[str, Any]]:
    """
    Phase 8.6b symbolic equation_transform teacher.

    Conservative verified transducers:
      - constant RHS if all examples prove constant
      - ordered fixed-position projection
      - reversed ordered fixed-position projection
      - repeated fixed-position projection
      - selected-position + bijective char substitution
      - full-LHS bijective char substitution when lengths match

    Every candidate must match all examples exactly before query prediction.
    If multiple verified candidates produce different query answers, reject.
    """
    if not pairs or query is None:
        return None

    gold = norm_text(gold)

    lhs_lens = [len(lhs) for lhs, _ in pairs]
    rhs_lens = [len(rhs) for _, rhs in pairs]

    if not lhs_lens or not rhs_lens:
        return {"covered": False, "reason": "equation_symbolic_empty_pairs"}

    min_lhs_len = min(lhs_lens)
    out_lens = set(rhs_lens)

    if len(out_lens) != 1:
        return {"covered": False, "reason": "equation_symbolic_variable_rhs_lengths"}

    out_len = next(iter(out_lens))
    if out_len <= 0:
        return {"covered": False, "reason": "equation_symbolic_empty_rhs"}

    candidates: List[Tuple[str, str]] = []

    # 1. Constant output, if fully proven.
    const = _all_same_rhs(pairs)
    if const is not None:
        candidates.append(("constant_rhs", const))

    # Guard: avoid exploding on long strings.
    if out_len <= 6 and min_lhs_len <= 12:
        # Ordered positions with repeats. This is stronger than combinations.
        # It catches output like lhs[1]+lhs[3], reversed, repeated, etc.
        checked = 0
        max_checked = 250000

        for positions in itertools.product(range(min_lhs_len), repeat=out_len):
            checked += 1
            if checked > max_checked:
                break

            def select(lhs: str, positions=positions) -> Optional[str]:
                if not positions:
                    return ""
                if max(positions) >= len(lhs):
                    return None
                return "".join(lhs[i] for i in positions)

            # direct selected characters
            direct_ok = True
            for lhs, rhs in pairs:
                s = select(lhs)
                if s is None or s != rhs:
                    direct_ok = False
                    break
            if direct_ok:
                qs = select(query)
                if qs is not None:
                    candidates.append((f"direct_positions:{positions}", qs))

            # reversed selected characters
            reverse_ok = True
            for lhs, rhs in pairs:
                s = select(lhs)
                if s is None or s[::-1] != rhs:
                    reverse_ok = False
                    break
            if reverse_ok:
                qs = select(query)
                if qs is not None:
                    candidates.append((f"reverse_positions:{positions}", qs[::-1]))

            # selected characters with bijective symbol map
            cmap = _learn_map_for_selected(pairs, select)
            if cmap is not None:
                qs = select(query)
                if qs is not None:
                    pred = _apply_char_map(qs, cmap)
                    if pred is not None:
                        candidates.append((f"mapped_positions:{positions}", pred))

    # 2. Full-string bijective substitution where lhs and rhs lengths match.
    if all(len(lhs) == len(rhs) for lhs, rhs in pairs):
        def full_select(lhs: str) -> Optional[str]:
            return lhs

        cmap = _learn_map_for_selected(pairs, full_select)
        if cmap is not None:
            pred = _apply_char_map(query, cmap)
            if pred is not None:
                candidates.append(("full_string_char_map", pred))

    if not candidates:
        return {"covered": False, "reason": "equation_symbolic_no_verified_transducer"}

    preds: Dict[str, List[str]] = {}
    for name, pred in candidates:
        preds.setdefault(pred, []).append(name)

    if len(preds) != 1:
        return {
            "covered": False,
            "reason": "equation_symbolic_ambiguous_query_predictions",
            "prediction_count": len(preds),
            "sample_predictions": {k: v[:5] for k, v in list(preds.items())[:5]},
        }

    pred = next(iter(preds))
    ok = pred == gold
    return {
        "covered": ok,
        "prediction": pred,
        "candidate_count": len(candidates),
        "candidate_names": list(next(iter(preds.values())))[:10],
        "reason": "verified" if ok else "prediction_mismatch",
    }




def equation_variable_length_symbolic_teacher(
    pairs: List[Tuple[str, str]],
    query: str,
    gold: str,
) -> Optional[Dict[str, Any]]:
    """
    Phase 8.6c fallback for variable-length RHS symbolic tasks.

    Candidate classes:
      - delete one fixed position from lhs
      - delete two fixed positions from lhs
      - keep fixed ordered positions with output length inferred per row only when lengths align
      - prefix/suffix of fixed length
      - reverse prefix/suffix
      - full-string bijective substitution with optional fixed deletes

    All candidates must verify every example exactly.
    """
    if not pairs or query is None:
        return None

    gold = norm_text(gold)
    min_len = min(len(lhs) for lhs, _ in pairs)
    max_len = max(len(lhs) for lhs, _ in pairs)

    if min_len <= 0 or max_len > 16:
        return {"covered": False, "reason": "equation_var_symbolic_budget_guard"}

    candidates: List[Tuple[str, str]] = []

    def verify_and_predict(name: str, fn: Callable[[str], Optional[str]]) -> None:
        for lhs, rhs in pairs:
            pred = fn(lhs)
            if pred is None or pred != rhs:
                return
        qpred = fn(query)
        if qpred is not None:
            candidates.append((name, qpred))

    # Prefix/suffix and reversed variants.
    for k in range(1, min(8, min_len) + 1):
        verify_and_predict(f"prefix_{k}", lambda s, k=k: s[:k] if len(s) >= k else None)
        verify_and_predict(f"suffix_{k}", lambda s, k=k: s[-k:] if len(s) >= k else None)
        verify_and_predict(f"prefix_rev_{k}", lambda s, k=k: s[:k][::-1] if len(s) >= k else None)
        verify_and_predict(f"suffix_rev_{k}", lambda s, k=k: s[-k:][::-1] if len(s) >= k else None)

    # Delete fixed positions. This catches variable RHS lengths when lhs lengths vary.
    for del_count in [1, 2, 3]:
        if del_count >= min_len:
            continue
        for dels in itertools.combinations(range(min_len), del_count):
            dels_set = set(dels)

            def delete_positions(s: str, dels_set=dels_set) -> Optional[str]:
                if max(dels_set) >= len(s):
                    return None
                return "".join(ch for i, ch in enumerate(s) if i not in dels_set)

            verify_and_predict(f"delete_positions:{tuple(dels)}", delete_positions)

            def delete_positions_rev(s: str, dels_set=dels_set) -> Optional[str]:
                out = delete_positions(s, dels_set)
                return out[::-1] if out is not None else None

            verify_and_predict(f"delete_positions_rev:{tuple(dels)}", delete_positions_rev)

    # Keep ordered fixed positions with output length equal to each RHS length.
    # For variable RHS lengths, this only works if all RHS lengths are still equal;
    # otherwise skip. It overlaps with equation_symbolic_teacher but is cheaper.
    rhs_lens = {len(rhs) for _, rhs in pairs}
    if len(rhs_lens) == 1:
        out_len = next(iter(rhs_lens))
        if 0 < out_len <= 5 and min_len <= 12:
            checked = 0
            for positions in itertools.product(range(min_len), repeat=out_len):
                checked += 1
                if checked > 100000:
                    break

                def select(s: str, positions=positions) -> Optional[str]:
                    if max(positions) >= len(s):
                        return None
                    return "".join(s[i] for i in positions)

                verify_and_predict(f"var_select:{positions}", select)

    # Full-string char substitution, if lengths match each row.
    if all(len(lhs) == len(rhs) for lhs, rhs in pairs):
        def full_select(lhs: str) -> Optional[str]:
            return lhs

        cmap = _learn_map_for_selected(pairs, full_select)
        if cmap is not None:
            pred = _apply_char_map(query, cmap)
            if pred is not None:
                candidates.append(("var_full_string_char_map", pred))

    if not candidates:
        return {"covered": False, "reason": "equation_var_symbolic_no_verified_candidate"}

    preds: Dict[str, List[str]] = {}
    for name, pred in candidates:
        preds.setdefault(pred, []).append(name)

    if len(preds) != 1:
        return {
            "covered": False,
            "reason": "equation_var_symbolic_ambiguous_query_predictions",
            "prediction_count": len(preds),
            "sample_predictions": {k: v[:5] for k, v in list(preds.items())[:5]},
        }

    pred = next(iter(preds))
    ok = pred == gold
    return {
        "covered": ok,
        "prediction": pred,
        "candidate_count": len(candidates),
        "candidate_names": list(next(iter(preds.values())))[:10],
        "reason": "verified" if ok else "prediction_mismatch",
    }


def equation_transform_teacher(prompt: str, gold: str) -> Dict[str, Any]:
    pairs, query = parse_equation_transform(prompt)
    if len(pairs) < 2:
        return {"covered": False, "reason": "equation_not_enough_pairs"}
    if not query:
        return {"covered": False, "reason": "equation_query_not_parsed"}

    # Phase 8.6d: first try per-operator numeric induction.
    # Public equation_transform often defines separate rules per operator symbol.
    perop_numeric = equation_numeric_per_operator_teacher(pairs, query, gold)
    if perop_numeric is not None and perop_numeric.get("covered") is True:
        return perop_numeric

    numeric = equation_numeric_teacher(pairs, query, gold)
    if numeric is not None and numeric.get("covered") is True:
        return numeric

    # Phase 8.6b critical fix:
    # Numeric-looking equation_transform rows can still be string transducers.
    # Do not let numeric failure block symbolic fallback.
    symbolic = equation_symbolic_teacher(pairs, query, gold)
    if symbolic is not None and symbolic.get("covered") is True:
        return symbolic

    variable_symbolic = equation_variable_length_symbolic_teacher(pairs, query, gold)
    if variable_symbolic is not None and variable_symbolic.get("covered") is True:
        return variable_symbolic

    if numeric is not None and symbolic is not None and variable_symbolic is not None:
        return {
            "covered": False,
            "reason": "equation_all_teachers_failed",
            "perop_numeric_reason": perop_numeric.get("reason") if perop_numeric is not None else None,
            "numeric_reason": numeric.get("reason"),
            "symbolic_reason": symbolic.get("reason"),
            "variable_symbolic_reason": variable_symbolic.get("reason"),
        }

    if variable_symbolic is not None:
        return variable_symbolic
    if symbolic is not None:
        return symbolic
    if numeric is not None:
        return numeric
    if perop_numeric is not None:
        return perop_numeric

    return {"covered": False, "reason": "equation_no_teacher_applicable"}


# =============================================================================
# Main dispatch
# =============================================================================

def teacher_probe(family: str, prompt: str, answer: str) -> Dict[str, Any]:
    if not norm_text(answer):
        return {"covered": False, "reason": "no_gold_answer_available"}

    if family == "roman_numeral":
        return roman_teacher(prompt, answer)
    if family == "unit_conversion":
        return unit_conversion_teacher(prompt, answer)
    if family == "gravity_numeric_formula":
        return gravity_teacher(prompt, answer)
    if family == "word_cipher":
        return word_cipher_teacher(prompt, answer)
    if family == "bit_manipulation":
        return bit_manipulation_teacher(prompt, answer)
    if family == "equation_transform":
        return equation_transform_teacher(prompt, answer)

    return {"covered": False, "reason": "unknown_family"}
