from __future__ import annotations


def supervised_span_has_prompt_leak(decoded_supervised_span: str) -> bool:
    lowered = decoded_supervised_span.lower()
    return "user:" in lowered or "now, determine" in lowered or "here are some examples" in lowered


def validate_no_prompt_loss(decoded_spans: list[str]) -> dict:
    leaks = [span for span in decoded_spans if supervised_span_has_prompt_leak(span)]
    if leaks:
        raise ValueError("decoded supervised span contains prompt text")
    return {"prompt_leakage_count": 0, "span_count": len(decoded_spans)}
