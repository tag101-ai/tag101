"""Async client for the task server."""

from __future__ import annotations

import time
from ipaddress import ip_address
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

from .models import (
    LeaseRequest,
    MinerAxonAnnouncement,
    MinerAxonRequest,
    MinerAxonResponse,
    ResultReport,
    ScoreboardRequest,
    ScoreboardSnapshot,
    SignedLeaseRequest,
    SignedMinerAxonAnnouncement,
    SignedMinerAxonRequest,
    SignedResultReport,
    SignedScoreboardRequest,
    SignedScoreboardSnapshot,
    TaskLease,
)
from .signing import sign_model, wallet_hotkey


TASK_SERVER_RETRY = retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(60),
    reraise=True,
)


class TaskServerClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = bool(verify_ssl)
        self.transport = transport

    @TASK_SERVER_RETRY
    async def lease(
        self,
        *,
        wallet: Any,
        netuid: int,
        uid: int = -1,
        block: int = 0,
        version: int = 1,
        profile: dict[str, Any] | None = None,
    ) -> TaskLease:
        payload = LeaseRequest(
            version=int(version),
            timestamp=time.time(),
            hotkey=wallet_hotkey(wallet),
            netuid=netuid,
            uid=int(uid),
            block=int(block),
            profile=profile or {},
        )
        body = SignedLeaseRequest(
            payload=payload,
            signature=sign_model(wallet, payload),
        )
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            verify=self.verify_ssl,
            transport=self.transport,
        ) as client:
            response = await client.post("/tasks/lease", json=body.model_dump())
            response.raise_for_status()
            return TaskLease.model_validate(response.json())

    @TASK_SERVER_RETRY
    async def report(
        self,
        *,
        wallet: Any,
        netuid: int,
        task_id: str,
        miner_uids: list[int],
        rewards: list[float],
        uid: int = -1,
        version: int = 1,
        miner_hotkeys: list[str] | None = None,
        miner_answers: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = ResultReport(
            version=int(version),
            timestamp=time.time(),
            task_id=task_id,
            hotkey=wallet_hotkey(wallet),
            netuid=netuid,
            uid=int(uid),
            miner_uids=miner_uids,
            miner_hotkeys=miner_hotkeys or [],
            miner_answers=miner_answers or [],
            rewards=rewards,
            metadata=metadata or {},
        )
        body = SignedResultReport(payload=payload, signature=sign_model(wallet, payload))
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            verify=self.verify_ssl,
            transport=self.transport,
        ) as client:
            response = await client.post("/tasks/report", json=body.model_dump())
            response.raise_for_status()

    @TASK_SERVER_RETRY
    async def upload_scoreboard(
        self,
        *,
        wallet: Any,
        netuid: int,
        block: int,
        scores: list[float],
        uid: int = -1,
        version: int = 1,
        miner_hotkeys: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = ScoreboardSnapshot(
            version=int(version),
            timestamp=time.time(),
            hotkey=wallet_hotkey(wallet),
            netuid=netuid,
            uid=int(uid),
            block=int(block),
            scores=scores,
            miner_hotkeys=miner_hotkeys or [],
            metadata=metadata or {},
        )
        body = SignedScoreboardSnapshot(
            payload=payload,
            signature=sign_model(wallet, payload),
        )
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            verify=self.verify_ssl,
            transport=self.transport,
        ) as client:
            response = await client.post("/scoreboards", json=body.model_dump())
            response.raise_for_status()
            return response.json()

    @TASK_SERVER_RETRY
    async def latest_scoreboards(
        self,
        *,
        wallet: Any,
        netuid: int,
        uid: int = -1,
        version: int = 1,
    ) -> list[SignedScoreboardSnapshot]:
        payload = ScoreboardRequest(
            version=int(version),
            timestamp=time.time(),
            hotkey=wallet_hotkey(wallet),
            netuid=netuid,
            uid=int(uid),
        )
        body = SignedScoreboardRequest(
            payload=payload,
            signature=sign_model(wallet, payload),
        )
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            verify=self.verify_ssl,
            transport=self.transport,
        ) as client:
            response = await client.post("/scoreboards/latest", json=body.model_dump())
            response.raise_for_status()
            payloads = response.json().get("scoreboards", [])
        return [SignedScoreboardSnapshot.model_validate(item) for item in payloads]

    @TASK_SERVER_RETRY
    async def announce_miner_axon(
        self,
        *,
        wallet: Any,
        netuid: int,
        uid: int,
        block: int,
        ip: str,
        port: int,
        ip_type: int | None = None,
        protocol: int = 4,
        version: int = 1,
    ) -> dict[str, Any]:
        payload = MinerAxonAnnouncement(
            version=int(version),
            timestamp=time.time(),
            hotkey=wallet_hotkey(wallet),
            netuid=netuid,
            uid=int(uid),
            block=int(block),
            ip=ip,
            port=int(port),
            ip_type=int(ip_type or ip_address(ip).version),
            protocol=int(protocol),
        )
        body = SignedMinerAxonAnnouncement(
            payload=payload,
            signature=sign_model(wallet, payload),
        )
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            verify=self.verify_ssl,
            transport=self.transport,
        ) as client:
            response = await client.post("/miners/axon", json=body.model_dump())
            response.raise_for_status()
            return response.json()

    @TASK_SERVER_RETRY
    async def miner_axons(
        self,
        *,
        wallet: Any,
        netuid: int,
        uid: int,
        block: int,
        version: int = 1,
    ) -> MinerAxonResponse:
        payload = MinerAxonRequest(
            version=int(version),
            timestamp=time.time(),
            hotkey=wallet_hotkey(wallet),
            netuid=netuid,
            uid=int(uid),
            block=int(block),
        )
        body = SignedMinerAxonRequest(
            payload=payload,
            signature=sign_model(wallet, payload),
        )
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            verify=self.verify_ssl,
            transport=self.transport,
        ) as client:
            response = await client.post("/miners/axons", json=body.model_dump())
            response.raise_for_status()
            return MinerAxonResponse.model_validate(response.json())
