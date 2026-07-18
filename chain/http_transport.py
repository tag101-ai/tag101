"""Validator-side signed HTTP transport."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp

from .._bt import require_bittensor

#: HTTP route both sides agree on; the path is part of the signed payload, so
#: the miner server MUST expose exactly this path.
QUERY_PATH = "/TaskEnvelope"

_DEFAULT_TIMEOUT = 10.0


class HttpTransport:
    def __init__(
        self,
        wallet: Any,
        *,
        path: str = QUERY_PATH,
        max_response_bytes: int = 128 * 1024,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.wallet = wallet
        self.path = path
        self.max_response_bytes = int(max_response_bytes)
        self.timeout = float(timeout)

    async def query(self, *, axons: list[Any], synapse: Any) -> list[Any]:
        if not axons:
            return []
        response_cls = type(synapse)
        dump = synapse.model_dump() if hasattr(synapse, "model_dump") else dict(synapse)
        body = json.dumps(dump, separators=(",", ":"), sort_keys=True).encode("utf-8")
        timeout = float(
            getattr(synapse, "timeout", getattr(synapse, "time_limit", self.timeout))
            or self.timeout
        )
        connector = aiohttp.TCPConnector(limit=256)
        async with aiohttp.ClientSession(connector=connector) as session:
            return await asyncio.gather(
                *[
                    self._query_one(session, axon, body, timeout, response_cls)
                    for axon in axons
                ]
            )

    async def _query_one(
        self,
        session: aiohttp.ClientSession,
        axon: Any,
        body: bytes,
        timeout: float,
        response_cls: Any,
    ) -> Any:
        ip = str(getattr(axon, "ip", "") or "")
        port = int(getattr(axon, "port", 0) or 0)
        receiver = str(getattr(axon, "hotkey", "") or "") or None
        if not ip or port <= 0:
            return None
        url = f"http://{ip}:{port}{self.path}"
        try:
            headers = self._sign(body, receiver)
        except Exception:
            return None
        headers["Content-Type"] = "application/json"
        try:
            async with session.post(
                url,
                data=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                declared = int(response.headers.get("Content-Length", 0) or 0)
                if declared and declared > self.max_response_bytes:
                    return None
                raw = await response.read()
                if len(raw) > self.max_response_bytes or response.status != 200:
                    return None
                data = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        try:
            return response_cls.model_validate(data)
        except Exception:
            return None

    def _sign(self, body: bytes, receiver: str | None) -> dict[str, str]:
        bt = require_bittensor()
        return bt.http_auth.sign(
            self.wallet,
            method="POST",
            path=self.path,
            body=body,
            receiver_ss58=receiver,
        )
