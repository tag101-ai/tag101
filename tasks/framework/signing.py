"""Wallet signing helpers for task-server requests."""

from __future__ import annotations

import json
from typing import Any


def wallet_hotkey(wallet: Any) -> str:
    hotkey = getattr(wallet, "hotkey", None)
    ss58 = getattr(hotkey, "ss58_address", None)
    if ss58:
        return str(ss58)
    return str(getattr(wallet, "ss58_address", ""))


def sign_model(wallet: Any, model: Any) -> str:
    hotkey = getattr(wallet, "hotkey", None)
    signer = getattr(hotkey, "sign", None)
    if signer is None:
        return ""
    payload = canonical_model_payload(model)
    signed = signer(payload)
    if hasattr(signed, "hex"):
        return signed.hex()
    return str(signed)


def verify_model_signature(hotkey: str, signature: str, model: Any) -> bool:
    if not signature:
        return False

    keypair = _verification_keypair(hotkey)
    if keypair is None:
        return False

    message = canonical_model_payload(model)
    message_forms = (message.encode("utf-8"), message)
    for candidate in signature_candidates(signature):
        for encoded_message in message_forms:
            try:
                if bool(keypair.verify(encoded_message, candidate)):
                    return True
            except Exception:
                continue
    return False


def _verification_keypair(hotkey: str) -> Any:
    """Build a public-key-only keypair for signature verification."""
    from bittensor.sp_core import Keypair

    try:
        return Keypair(ss58_address=hotkey)
    except Exception:
        return None


def canonical_model_payload(model: Any) -> str:
    if hasattr(model, "model_dump_json"):
        value = json.loads(model.model_dump_json())
    else:
        value = model
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except TypeError:
        return str(model)


def signature_candidates(signature: str) -> list[str | bytes]:
    value = signature.strip()
    candidates: list[str | bytes] = [value]
    hex_value = value[2:] if value.startswith("0x") else value
    if not value.startswith("0x"):
        candidates.append(f"0x{value}")
    try:
        candidates.append(bytes.fromhex(hex_value))
    except ValueError:
        pass
    return candidates
