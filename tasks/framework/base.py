"""Generic task framework used by miners and validators.

Task authors provide one handler per task kind. Public handlers solve payloads
for miners and score miner answers for validators; task generation lives in
the task server.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ...chain.runtime import ChainRuntime
from ...protocol import TaskEnvelope
from .models import TaskLease
from .scoring import ScoreBreakdown


TaskPayload = Mapping[str, Any]
TaskAnswer = Mapping[str, Any]
TaskProfile = Mapping[str, Any]

MinerSolver = Callable[[TaskEnvelope, ChainRuntime], dict[str, Any]]
AnswerScorer = Callable[[TaskPayload, Mapping[str, Any], Sequence[TaskAnswer]], ScoreBreakdown]


@dataclass(frozen=True)
class LocalLeaseContext:
    """Inputs for a validator building its own lease from an external corpus.

    ``corpus`` is a ``CorpusReader``-like object exposing ``random_tweet(rng)``.
    """

    task_id: str
    corpus: Any
    rng: random.Random = field(default_factory=random.Random)
    profile: Mapping[str, Any] = field(default_factory=dict)


LocalLeaseBuilder = Callable[[LocalLeaseContext], TaskLease]

# Name aliases for task modules that still use the older hook labels.
SolveFn = MinerSolver
ScoreFn = AnswerScorer


@dataclass(frozen=True, init=False)
class TaskHandler:
    """Public miner/validator contract for one task kind.

    New task modules should provide two project-specific hooks:

    - ``solve_problem``: turns a task envelope and chain runtime into an answer.
    - ``score_answers``: scores miner answers using the scoring context.
    """

    kind: str
    spec_version: str
    solve_problem: MinerSolver
    score_answers: AnswerScorer
    build_lease: LocalLeaseBuilder | None
    description: str = ""

    def __init__(
        self,
        *,
        kind: str,
        spec_version: str,
        solve_problem: MinerSolver | None = None,
        score_answers: AnswerScorer | None = None,
        solve: MinerSolver | None = None,
        score_batch: AnswerScorer | None = None,
        build_lease: LocalLeaseBuilder | None = None,
        description: str = "",
    ):
        solver = solve_problem or solve
        scorer = score_answers or score_batch
        if solver is None:
            raise ValueError("TaskHandler requires solve_problem")
        if scorer is None:
            raise ValueError("TaskHandler requires score_answers")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "spec_version", spec_version)
        object.__setattr__(self, "solve_problem", solver)
        object.__setattr__(self, "score_answers", scorer)
        object.__setattr__(self, "build_lease", build_lease)
        object.__setattr__(self, "description", description)

    @property
    def solve(self) -> MinerSolver:
        return self.solve_problem

    @property
    def score_batch(self) -> AnswerScorer:
        return self.score_answers

    def build_local_lease(self, context: LocalLeaseContext) -> TaskLease:
        """Build a lease locally from an external corpus (AWS-corpus flow)."""

        if self.build_lease is None:
            raise RuntimeError(
                f"task kind {self.kind!r} does not support local corpus leasing"
            )
        return self.build_lease(context)
