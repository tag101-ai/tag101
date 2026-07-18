"""Helpers for replacing public metagraph axons with signed private endpoints."""

from __future__ import annotations

import copy
from ipaddress import ip_address
from typing import Any

from ..tasks import SignedMinerAxonAnnouncement
from ..tasks.framework.signing import verify_model_signature


def private_axons_by_hotkey(
    *,
    metagraph: Any,
    netuid: int,
    announcements: list[SignedMinerAxonAnnouncement],
) -> dict[str, Any]:
    hotkeys = [str(value) for value in getattr(metagraph, "hotkeys", [])]
    axons = list(getattr(metagraph, "axons", []))
    private: dict[str, Any] = {}
    newest_timestamp: dict[str, float] = {}

    for signed in announcements:
        payload = signed.payload
        hotkey = str(payload.hotkey)
        uid = int(getattr(payload, "uid", -1))
        if uid < 0 or uid >= len(hotkeys) or uid >= len(axons):
            continue
        if hotkeys[uid] != hotkey:
            continue
        if int(payload.netuid) != int(netuid):
            continue
        if not _valid_endpoint(payload.ip, payload.port):
            continue
        if not verify_model_signature(hotkey, signed.signature, payload):
            continue
        timestamp = float(getattr(payload, "timestamp", 0.0))
        if timestamp < newest_timestamp.get(hotkey, float("-inf")):
            continue
        newest_timestamp[hotkey] = timestamp
        private[hotkey] = _copy_axon_with_endpoint(axons[uid], payload)
    return private


def is_usable_axon(axon: Any) -> bool:
    if axon is None:
        return False
    if hasattr(axon, "is_serving"):
        return bool(getattr(axon, "is_serving"))
    ip = str(getattr(axon, "ip", "") or "")
    port = getattr(axon, "port", None)
    return bool(ip) and ip != "0.0.0.0" and port is not None and int(port) > 0


def _copy_axon_with_endpoint(axon: Any, payload: Any) -> Any:
    private = copy.copy(axon)
    setattr(private, "ip", payload.ip)
    setattr(private, "port", int(payload.port))
    setattr(private, "ip_type", int(payload.ip_type))
    setattr(private, "protocol", int(payload.protocol))
    if int(payload.version) > 0:
        setattr(private, "version", int(payload.version))
    setattr(private, "hotkey", str(payload.hotkey))
    return private


def _valid_endpoint(ip: str, port: int) -> bool:
    try:
        ip_address(ip)
        port = int(port)
    except (TypeError, ValueError):
        return False
    return 1 <= port <= 65535
