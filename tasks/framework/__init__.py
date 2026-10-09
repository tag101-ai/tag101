"""Shared task framework.

Task authors usually import names from ``tag101.tasks``. This subpackage
keeps reusable schemas, signing, registry, client, and scoring code away from
the top-level task modules that project teams customize.
"""

from .base import (
    AnswerScorer,
    LocalLeaseBuilder,
    LocalLeaseContext,
    MinerSolver,
    TaskAnswer,
    TaskHandler,
    TaskPayload,
    TaskProfile,
)
from .models import (
    CorpusCredentialRequest,
    CorpusCredentialResponse,
    LeaseRequest,
    MinerAxonAnnouncement,
    MinerAxonRequest,
    MinerAxonResponse,
    MinerResult,
    ResultReport,
    ScoreboardRequest,
    ScoreboardSnapshot,
    SignedCorpusCredentialRequest,
    SignedLeaseRequest,
    SignedMinerAxonAnnouncement,
    SignedMinerAxonRequest,
    SignedResultReport,
    SignedScoreboardRequest,
    SignedScoreboardSnapshot,
    TaskLease,
)
from .registry import TaskRegistry, default_registry
from .scoring import MinerScore, ScoreBreakdown


def __getattr__(name: str):
    if name == "TaskServerClient":
        from .client import TaskServerClient

        return TaskServerClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AnswerScorer",
    "CorpusCredentialRequest",
    "CorpusCredentialResponse",
    "LeaseRequest",
    "LocalLeaseBuilder",
    "LocalLeaseContext",
    "MinerAxonAnnouncement",
    "MinerAxonRequest",
    "MinerAxonResponse",
    "MinerResult",
    "MinerScore",
    "MinerSolver",
    "ResultReport",
    "ScoreboardRequest",
    "ScoreboardSnapshot",
    "ScoreBreakdown",
    "SignedCorpusCredentialRequest",
    "SignedLeaseRequest",
    "SignedMinerAxonAnnouncement",
    "SignedMinerAxonRequest",
    "SignedResultReport",
    "SignedScoreboardRequest",
    "SignedScoreboardSnapshot",
    "TaskAnswer",
    "TaskHandler",
    "TaskPayload",
    "TaskProfile",
    "TaskRegistry",
    "TaskServerClient",
    "TaskLease",
    "default_registry",
]
