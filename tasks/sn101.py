"""Task adapter for the SN101 reference implementation.

The SN101 scoring/miner logic lives under ``tag101.tasks.sn101_reference``.
This module intentionally stays thin: it converts the wire-format answer into
the reference scorer input, and converts the reference scorer output into
``ScoreBreakdown``. Task payloads are leased from the task server.
"""

from __future__ import annotations

import random
import re
from typing import Any, Mapping, Sequence

from ..chain.runtime import ChainRuntime
from ..protocol import TaskEnvelope
from .framework import (
    LocalLeaseContext,
    ScoreBreakdown,
    TaskHandler,
    TaskLease,
)
from .sn101_reference.core.miner import (
    Miner as ReferenceMiner,
)


KIND = "sn101.tags.v1"
SPEC_VERSION = "v1"
REFERENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# SN101 constants are intentionally fixed here, not provided through
# task_profile_json. Keep them in code so validators cannot change task shape.
SN101_MAX_TAGS = 3
SN101_TIME_LIMIT = 120.0
SN101_TEST_FALLBACK_TAGS = (
    "bitcoin",
    "ethereum",
    "crypto market",
    "defi",
    "stablecoin",
    "regulation",
    "etf",
    "trading",
)


def build_local_lease(context: LocalLeaseContext) -> TaskLease:
    """Build an SN101 lease locally from a self-selected corpus tweet.

    Task-shape constants stay fixed in code so validators cannot change them.
    """

    selection = context.corpus.random_tweet(context.rng)
    tweet = selection.tweet
    return TaskLease(
        task_id=context.task_id,
        task_kind=KIND,
        spec_version=SPEC_VERSION,
        time_limit=SN101_TIME_LIMIT,
        payload={
            "tweet_id": str(tweet.get("tweet_id", "")),
            "tweet_uuid": str(tweet.get("tweet_uuid", "")),
            "author": str(tweet.get("author", "")),
            "created_at": str(tweet.get("created_at", "")),
            "url": str(tweet.get("url", "")),
            "text": str(tweet.get("text", "")),
            "content_sha256": selection.content_sha256,
            "max_tags": SN101_MAX_TAGS,
            "time_limit": SN101_TIME_LIMIT,
        },
        scoring={
            "method": "tao_sn101_exp.core.scoring.TagScorer",
            "model_name": str(context.profile.get("model_name", REFERENCE_MODEL_NAME)),
            "n_tags_per_miner": SN101_MAX_TAGS,
            "proximity_rank_decay": float(
                context.profile.get("proximity_rank_decay", 1.0)
            ),
        },
        metadata={
            "dataset": "aws-corpus",
            "tweet_uuid": str(tweet.get("tweet_uuid", "")),
            "content_sha256": selection.content_sha256,
            "digest_root_hash": selection.root_hash,
            "s3_key": str(selection.entry.get("key", "")),
            "whitelist_author": str(tweet.get("author", "")),
        },
    )


def solve_problem(
    envelope: TaskEnvelope,
    chain_runtime: ChainRuntime,
) -> dict[str, Any]:
    task_payload = dict(envelope.payload)
    post = str(task_payload.get("text", ""))
    miner = ReferenceMiner(
        n_tags=SN101_MAX_TAGS,
        timeout_sec=int(SN101_TIME_LIMIT),
    )
    try:
        tags = miner.generate_tags(post)
    except RuntimeError as exc:
        if "Set OPENAI_API_KEY" not in str(exc):
            raise
        tags = [random.choice(SN101_TEST_FALLBACK_TAGS)]
    return {"tags": tags}


def score_answers(
    payload: Mapping[str, Any],
    scoring: Mapping[str, Any],
    answers: Sequence[Mapping[str, Any]],
) -> ScoreBreakdown:
    post = str(payload.get("text", ""))
    responses = [_tags_from_answer(answer) for answer in answers]
    result = _score_with_reference(
        post=post,
        responses=responses,
        n_tags_per_miner=SN101_MAX_TAGS,
        model_name=str(scoring.get("model_name", REFERENCE_MODEL_NAME)),
    )

    rewards = [float(value) for value in result.get("miner_scores", [])]
    metrics = _miner_metrics(result, len(answers))
    return ScoreBreakdown(
        rewards=rewards,
        metrics=metrics,
        details={
            "method": scoring.get(
                "method",
                "tag101.tasks.sn101_reference.core.scoring.TagScorer",
            ),
            "tweet_id": payload.get("tweet_id", ""),
            "clusters": result.get("clusters", []),
            "scored_cluster_ids": result.get("scored_cluster_ids", []),
            "spans": result.get("spans", []),
        },
    )


def _score_with_reference(
    *,
    post: str,
    responses: list[list[str]],
    n_tags_per_miner: int,
    model_name: str,
) -> dict[str, Any]:
    # Clustering/signal thresholds are fixed as class defaults inside the
    # reference scorer (TagClusterer / SignalScorer); they are intentionally
    # not surfaced here so validators cannot change scoring shape.
    from .sn101_reference.core.scoring import TagScorer

    return TagScorer(
        model_name=model_name,
        n_tags_per_miner=n_tags_per_miner,
    ).score(post, responses)


def _miner_metrics(
    result: Mapping[str, Any],
    miner_count: int,
) -> list[dict[str, Any]]:
    per_miner_keys = (
        "normalized_responses",
        "tag_scores",
        "signal_scores",
        "validity_scores",
        "utility_scores",
        "validity_details",
        "utility_details",
        "cluster_labels",
        "raw_miner_scores",
        "repeated_set_counts",
        "repeated_set_adjustments",
    )
    metrics: list[dict[str, Any]] = []
    for index in range(miner_count):
        item: dict[str, Any] = {}
        for key in per_miner_keys:
            values = result.get(key, [])
            if isinstance(values, list) and index < len(values):
                item[key] = values[index]
        metrics.append(item)
    return metrics


def _tags_from_answer(answer: Mapping[str, Any]) -> list[str]:
    raw = (
        answer.get("tags")
        or answer.get("labels")
        or answer.get("keywords")
        or []
    )
    if isinstance(raw, str):
        values: Sequence[Any] = re.split(r"[,;\n|]+", raw)
    elif isinstance(raw, Sequence):
        values = raw
    else:
        values = []

    tags: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        # Keep normalization inside the reference scorer. Importing its
        # preprocessing module here would execute scoring/__init__.py and
        # force miner startup to import sklearn even though miners do not
        # score.
        tag = value
        if tag:
            tags.append(tag)
    return tags


solve = solve_problem
score_batch = score_answers


def handler() -> TaskHandler:
    return TaskHandler(
        kind=KIND,
        spec_version=SPEC_VERSION,
        solve_problem=solve_problem,
        score_answers=score_answers,
        build_lease=build_local_lease,
        description=(
            "SN101 semantic tagging task backed by the reference scorer."
        ),
    )
