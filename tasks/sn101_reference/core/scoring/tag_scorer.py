from __future__ import annotations

import math
from typing import Any

from .consensus import ConsensusScorer
from .diversity import DiversityScorer
from .preprocessing import (
    aggregate_miner_score,
    build_scoring_context,
    unflatten_scores,
)
from .validity import ValidityScorer


class TagScorer:
    """Phase 1 scorer composed from independent tag scorers."""

    N_TAGS_PER_MINER = 3
    CONSENSUS_WEIGHT = 0.60
    VALIDITY_DIVERSITY_WEIGHT = 0.40
    DUPLICATE_PENALTY_K = 0.1
    DUPLICATE_PENALTY_C = 50.0

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_tags_per_miner: int = N_TAGS_PER_MINER,
        proximity_rank_decay: float = 1.0,
        duplicate_penalty_enabled: bool = True,
        duplicate_penalty_k: float = DUPLICATE_PENALTY_K,
        duplicate_penalty_c: float = DUPLICATE_PENALTY_C,
    ) -> None:
        self.model_name = model_name
        self.n_tags_per_miner = n_tags_per_miner
        self.proximity_rank_decay = proximity_rank_decay
        self.duplicate_penalty_enabled = duplicate_penalty_enabled
        self.duplicate_penalty_k = duplicate_penalty_k
        self.duplicate_penalty_c = duplicate_penalty_c
        self._model = self._load_model(model_name)
        self._consensus_scorer = ConsensusScorer(
            model_name=model_name,
            n_tags_per_miner=n_tags_per_miner,
            proximity_rank_decay=proximity_rank_decay,
            model=self._model,
        )
        self._validity_scorer = ValidityScorer(
            model_name=model_name,
            n_tags_per_miner=n_tags_per_miner,
            model=self._model,
        )
        self._diversity_scorer = DiversityScorer(
            model_name=model_name,
            n_tags_per_miner=n_tags_per_miner,
            model=self._model,
        )

    def score(self, post: str, responses: list[list[str]]) -> dict[str, Any]:
        context = build_scoring_context(
            post=post,
            responses=responses,
            n_tags_per_miner=self.n_tags_per_miner,
        )
        consensus_result = self._consensus_scorer.score_from_context(context)
        validity_result = self._validity_scorer.score_from_context(context)
        diversity_result = self._diversity_scorer.score_from_context(context)

        normalized_responses = consensus_result["normalized_responses"]
        consensus_scores = consensus_result["consensus_scores"]
        validity_scores = validity_result["validity_scores"]
        diversity_scores = diversity_result["diversity_scores"]

        if not normalized_responses:
            return {
                "tag_scores": [],
                "miner_scores": [],
                "consensus_scores": [],
                "validity_scores": [],
                "diversity_scores": [],
                "clusters": [],
                "normalized_responses": [],
            }

        flat_consensus = self._flatten_scores(consensus_scores)
        flat_validity = self._flatten_scores(validity_scores)
        flat_diversity = self._flatten_scores(diversity_scores)
        flat_tag_scores = [
            (self.CONSENSUS_WEIGHT * c + self.VALIDITY_DIVERSITY_WEIGHT * v * d)
            for c, v, d in zip(flat_consensus, flat_validity, flat_diversity)
        ]

        tag_scores, _, _ = unflatten_scores(
            normalized_responses=normalized_responses,
            flat_tag_scores=flat_tag_scores,
            flat_consensus_scores=flat_consensus,
            flat_validity_scores=flat_validity,
        )
        miner_scores = [
            aggregate_miner_score(scores=scores, top_k=self.n_tags_per_miner)
            for scores in tag_scores
        ]
        penalty_result = self._apply_duplicate_penalty(
            miner_scores=miner_scores,
            normalized_responses=normalized_responses,
        )
        return {
            "tag_scores": tag_scores,
            "miner_scores": penalty_result["miner_scores"],
            "raw_miner_scores": miner_scores,
            "duplicate_counts": penalty_result["duplicate_counts"],
            "duplicate_decays": penalty_result["duplicate_decays"],
            "consensus_scores": consensus_scores,
            "validity_scores": validity_scores,
            "diversity_scores": diversity_scores,
            "clusters": consensus_result["clusters"],
            "normalized_responses": normalized_responses,
            "spans": validity_result["spans"],
            "validity_details": validity_result["validity_details"],
            "diversity_details": diversity_result["diversity_details"],
        }

    def _load_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install it with "
                "'pip install sentence-transformers'."
            ) from exc
        return SentenceTransformer(model_name)

    def _flatten_scores(self, scores: list[list[float]]) -> list[float]:
        return [score for miner_scores in scores for score in miner_scores]

    def _apply_duplicate_penalty(
        self,
        *,
        miner_scores: list[float],
        normalized_responses: list[list[str]],
    ) -> dict[str, Any]:
        if not self.duplicate_penalty_enabled:
            return {
                "miner_scores": miner_scores,
                "duplicate_counts": [1 for _ in normalized_responses],
                "duplicate_decays": [1.0 for _ in normalized_responses],
            }

        duplicate_counts_by_key: dict[tuple[str, ...], int] = {}
        response_keys = [
            self._duplicate_key(response) for response in normalized_responses
        ]
        for key in response_keys:
            duplicate_counts_by_key[key] = duplicate_counts_by_key.get(key, 0) + 1

        duplicate_counts = [duplicate_counts_by_key[key] for key in response_keys]
        duplicate_decays = [self._duplicate_decay(count) for count in duplicate_counts]
        adjusted_scores = [
            score * decay for score, decay in zip(miner_scores, duplicate_decays)
        ]
        return {
            "miner_scores": adjusted_scores,
            "duplicate_counts": duplicate_counts,
            "duplicate_decays": duplicate_decays,
        }

    def _duplicate_key(self, response: list[str]) -> tuple[str, ...]:
        return tuple(sorted(" ".join(tag.split()) for tag in response if tag))

    def _duplicate_decay(self, count: int) -> float:
        exponent = self.duplicate_penalty_k * (float(count) - self.duplicate_penalty_c)
        return 1.0 / (1.0 + math.exp(exponent))
