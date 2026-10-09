"""Data models shared with the task server."""

from __future__ import annotations

from typing import Any

from ipaddress import ip_address

from pydantic import BaseModel, Field, field_validator

from .corpus_hashing import DEFAULT_CORPUS_PREFIX
from .scoring import ScoreBreakdown


class LeaseRequest(BaseModel):
    version: int = 1
    timestamp: float
    hotkey: str
    netuid: int
    uid: int = -1
    block: int = 0
    profile: dict[str, Any] = Field(default_factory=dict)


class SignedLeaseRequest(BaseModel):
    payload: LeaseRequest
    signature: str = ""


class TaskLease(BaseModel):
    task_id: str = Field(description="Server-issued question UUID.")
    task_kind: str
    spec_version: str = "v1"
    time_limit: float = 10.0
    payload: dict[str, Any] = Field(default_factory=dict)
    scoring: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultReport(BaseModel):
    version: int = 1
    timestamp: float
    task_id: str = Field(description="Server-issued question UUID from the task lease.")
    hotkey: str
    netuid: int
    uid: int = -1
    miner_uids: list[int] = Field(default_factory=list)
    miner_hotkeys: list[str] = Field(default_factory=list)
    miner_answers: list[dict[str, Any]] = Field(default_factory=list)
    rewards: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignedResultReport(BaseModel):
    payload: ResultReport
    signature: str = ""


class ScoreboardSnapshot(BaseModel):
    version: int = 1
    timestamp: float
    hotkey: str
    netuid: int
    uid: int = -1
    block: int
    scores: list[float] = Field(default_factory=list)
    miner_hotkeys: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignedScoreboardSnapshot(BaseModel):
    payload: ScoreboardSnapshot
    signature: str = ""


class ScoreboardRequest(BaseModel):
    version: int = 1
    timestamp: float
    hotkey: str
    netuid: int
    uid: int = -1


class SignedScoreboardRequest(BaseModel):
    payload: ScoreboardRequest
    signature: str = ""


class MinerAxonAnnouncement(BaseModel):
    version: int = 1
    timestamp: float
    hotkey: str
    netuid: int
    uid: int = -1
    block: int
    ip: str
    port: int
    ip_type: int = 4
    protocol: int = 4

    @field_validator("ip")
    @classmethod
    def _valid_ip(cls, value: str) -> str:
        return str(ip_address(value))

    @field_validator("port")
    @classmethod
    def _valid_port(cls, value: int) -> int:
        value = int(value)
        if value < 1 or value > 65535:
            raise ValueError("port must be between 1 and 65535")
        return value

    @field_validator("ip_type")
    @classmethod
    def _valid_ip_type(cls, value: int) -> int:
        value = int(value)
        if value not in {4, 6}:
            raise ValueError("ip_type must be 4 or 6")
        return value


class SignedMinerAxonAnnouncement(BaseModel):
    payload: MinerAxonAnnouncement
    signature: str = ""


class MinerAxonRequest(BaseModel):
    version: int = 1
    timestamp: float
    hotkey: str
    netuid: int
    uid: int = -1
    block: int


class SignedMinerAxonRequest(BaseModel):
    payload: MinerAxonRequest
    signature: str = ""


class MinerAxonResponse(BaseModel):
    axons: list[SignedMinerAxonAnnouncement] = Field(default_factory=list)


class CorpusCredentialRequest(BaseModel):
    version: int = 1
    timestamp: float
    hotkey: str
    netuid: int
    uid: int = -1
    block: int = 0


class SignedCorpusCredentialRequest(BaseModel):
    payload: CorpusCredentialRequest
    signature: str = ""


class CorpusCredentialResponse(BaseModel):
    """Read-only corpus access material handed to a qualified validator."""

    rotation_id: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    session_token: str | None = None
    region: str = ""
    endpoint_url: str | None = None
    bucket: str = ""
    prefix: str = DEFAULT_CORPUS_PREFIX
    object_key_template: str = "{prefix}/tweets/{tweet_uuid}.json"
    digest_url: str = ""
    digest_key: str | None = None
    expires_at: float | None = None


class MinerResult(BaseModel):
    uid: int
    answer: dict[str, Any] = Field(default_factory=dict)
    latency: float | None = None
    error: str | None = None


__all__ = [
    "LeaseRequest",
    "SignedLeaseRequest",
    "TaskLease",
    "ResultReport",
    "SignedResultReport",
    "ScoreboardSnapshot",
    "SignedScoreboardSnapshot",
    "ScoreboardRequest",
    "SignedScoreboardRequest",
    "MinerAxonAnnouncement",
    "SignedMinerAxonAnnouncement",
    "MinerAxonRequest",
    "SignedMinerAxonRequest",
    "MinerAxonResponse",
    "CorpusCredentialRequest",
    "SignedCorpusCredentialRequest",
    "CorpusCredentialResponse",
    "MinerResult",
    "ScoreBreakdown",
]
