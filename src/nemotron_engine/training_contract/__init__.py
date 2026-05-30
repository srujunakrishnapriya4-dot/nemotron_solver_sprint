"""Training loss/mask contracts for safe Nemotron adapter SFT."""

from .token_mask_contract import validate_token_mask_rows

__all__ = ["validate_token_mask_rows"]
