"""Validator/miner message contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, model_validator


class TaskEnvelope(BaseModel):
    """Generic work message sent from validators to miners."""

    task_id: str = Field(default="", description="Server-issued question UUID.")
    task_kind: str = Field(default="", description="Task handler key.")
    spec_version: str = Field(default="v1", description="Task schema version.")
    time_limit: float = Field(default=10.0, description="Expected miner time budget.")
    payload: dict[str, Any] = Field(default_factory=dict)
    scoring: dict[str, Any] = Field(default_factory=dict)
    answer: dict[str, Any] = Field(default_factory=dict)
    timeout: float = Field(default=10.0)

    @model_validator(mode="before")
    @classmethod
    def _reject_miner_note(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "miner_note" in data:
            raise ValueError("miner_note is not supported on TaskEnvelope")
        return data

    def deserialize(self) -> dict[str, Any]:
        return self.answer


def envelope_from_lease(lease: Any) -> TaskEnvelope:
    """Build a synapse from a pydantic lease object or compatible mapping."""

    if hasattr(lease, "model_dump"):
        data = lease.model_dump()
    else:
        data = dict(lease)
    time_limit = float(data.get("time_limit", 10.0))
    return TaskEnvelope(
        task_id=data["task_id"],
        task_kind=data["task_kind"],
        spec_version=data.get("spec_version", "v1"),
        time_limit=time_limit,
        payload=data.get("payload", {}),
        scoring=data.get("scoring", {}),
        timeout=time_limit,
    )
