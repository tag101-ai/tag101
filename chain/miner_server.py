"""Miner-side signed HTTP server."""

from __future__ import annotations

import threading
from typing import Any, Awaitable, Callable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .._bt import require_bittensor
from ..protocol import TaskEnvelope
from .http_transport import QUERY_PATH

ForwardFn = Callable[[TaskEnvelope], Awaitable[TaskEnvelope]]
BlacklistFn = Callable[[TaskEnvelope, str], Awaitable[tuple[bool, str]]]


class MinerServer:
    def __init__(
        self,
        *,
        wallet: Any,
        host: str,
        port: int,
        forward_fn: ForwardFn,
        blacklist_fn: BlacklistFn | None = None,
        max_request_bytes: int = 128 * 1024,
    ) -> None:
        self.wallet = wallet
        self.host = str(host)
        self.port = int(port)
        self.forward_fn = forward_fn
        self.blacklist_fn = blacklist_fn
        self.max_request_bytes = int(max_request_bytes)
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        bt = require_bittensor()
        self._http_auth = bt.http_auth
        self._nonce_store = bt.http_auth.InMemoryNonceStore()
        self.app = self._build_app()

    @property
    def hotkey_ss58(self) -> str:
        return str(self.wallet.hotkey.ss58_address)

    def _build_app(self) -> FastAPI:
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @app.post(QUERY_PATH)
        async def _forward(request: Request) -> Any:
            raw = await request.body()
            if len(raw) > self.max_request_bytes:
                return JSONResponse({"detail": "request too large"}, status_code=413)
            try:
                caller = self._http_auth.verify(
                    dict(request.headers),
                    raw,
                    method="POST",
                    path=QUERY_PATH,
                    self_hotkey_ss58=self.hotkey_ss58,
                    nonce_store=self._nonce_store,
                )
            except Exception as exc:
                return JSONResponse(
                    {"detail": f"unauthorized: {type(exc).__name__}"}, status_code=401
                )
            caller_hotkey = str(getattr(caller, "hotkey_ss58", ""))
            try:
                envelope = TaskEnvelope.model_validate_json(raw)
            except Exception:
                return JSONResponse({"detail": "invalid task envelope"}, status_code=400)
            if self.blacklist_fn is not None:
                blocked, reason = await self.blacklist_fn(envelope, caller_hotkey)
                if blocked:
                    return JSONResponse(
                        {"detail": reason or "blacklisted"}, status_code=403
                    )
            result = await self.forward_fn(envelope)
            payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            return JSONResponse(payload)

        return app

    def start(self) -> None:
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            loop="asyncio",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._server = None
