from __future__ import annotations

from typing import Any

from .clustering import TagClusterer
from .preprocessing import (
    aggregate_miner_score_by_coverage,
    build_scoring_context,
    repeated_set_adjustment,
    unflatten_scores,
)
from .signal import SignalScorer
from .utility import UtilityScorer
from .validity import ValidityScorer


class TagScorer:
    """TagScore = mean(signal, validity, utility) per tag over a shared clustering."""

    N_TAGS_PER_MINER = 3
    TOKEN_ID_CAP = 2

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_tags_per_miner: int = N_TAGS_PER_MINER,
        distance_threshold: float = TagClusterer.DISTANCE_THRESHOLD,
        min_clusters: int = TagClusterer.MIN_CLUSTERS,
        max_clusters: int = TagClusterer.MAX_CLUSTERS,
        support_alpha: float = SignalScorer.SUPPORT_ALPHA,
    ) -> None:
        self.model_name = model_name
        self.n_tags_per_miner = n_tags_per_miner
        self._model = self._load_model(model_name)
        self._clusterer = TagClusterer(
            model=self._model,
            model_name=model_name,
            n_tags=n_tags_per_miner,
            distance_threshold=distance_threshold,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )
        self._signal_scorer = SignalScorer(
            model_name=model_name,
            n_tags_per_miner=n_tags_per_miner,
            support_alpha=support_alpha,
            model=self._model,
            clusterer=self._clusterer,
        )
        self._validity_scorer = ValidityScorer(
            model_name=model_name,
            n_tags_per_miner=n_tags_per_miner,
            model=self._model,
        )
        self._utility_scorer = UtilityScorer(
            model_name=model_name,
            n_tags_per_miner=n_tags_per_miner,
            model=self._model,
            clusterer=self._clusterer,
        )

    def score(self, post: str, responses: list[list[str]]) -> dict[str, Any]:
        context = build_scoring_context(
            post=post,
            responses=responses,
            n_tags_per_miner=self.n_tags_per_miner,
        )
        clustering = self._clusterer.build(context)
        signal_result = self._signal_scorer.score_from_context(
            context, clustering=clustering
        )
        validity_result = self._validity_scorer.score_from_context(
            context, tag_embeddings=clustering.tag_embeddings
        )
        utility_result = self._utility_scorer.score_from_context(
            context, clustering=clustering
        )

        normalized_responses = signal_result["normalized_responses"]
        signal_scores = signal_result["signal_scores"]
        validity_scores = validity_result["validity_scores"]
        utility_scores = utility_result["utility_scores"]

        if not normalized_responses:
            return self._empty_result()

        flat_signal = self._flatten_scores(signal_scores)
        flat_validity = self._flatten_scores(validity_scores)
        flat_utility = self._flatten_scores(utility_scores)
        flat_tag_scores = [
            (s + v + u) / 3.0
            for s, v, u in zip(flat_signal, flat_validity, flat_utility)
        ]

        tag_scores, _, _ = unflatten_scores(
            normalized_responses=normalized_responses,
            flat_tag_scores=flat_tag_scores,
            flat_signal_scores=flat_signal,
            flat_validity_scores=flat_validity,
        )
        cluster_labels = self._unflatten_labels(
            normalized_responses=normalized_responses,
            labels=[int(label) for label in clustering.labels],
        )
        coverage_scores = aggregate_miner_score_by_coverage(
            flat_tag_scores=flat_tag_scores,
            flat_miner_idx=clustering.flat_miner_idx,
            labels=[int(label) for label in clustering.labels],
            scored_cluster_ids=clustering.scored_cluster_ids,
            miner_count=len(normalized_responses),
            n_tags=self.n_tags_per_miner,
        )
        adjustment = self._apply_repeated_set_adjustment(
            coverage_scores=coverage_scores,
            normalized_responses=normalized_responses,
        )

        return {
            "tag_scores": tag_scores,
            "miner_scores": adjustment["miner_scores"],
            "raw_miner_scores": coverage_scores,
            "repeated_set_counts": adjustment["repeated_set_counts"],
            "repeated_set_adjustments": adjustment["repeated_set_adjustments"],
            "signal_scores": signal_scores,
            "validity_scores": validity_scores,
            "utility_scores": utility_scores,
            "cluster_labels": cluster_labels,
            "scored_cluster_ids": list(clustering.scored_cluster_ids),
            "clusters": signal_result["clusters"],
            "normalized_responses": normalized_responses,
            "spans": validity_result["spans"],
            "validity_details": validity_result["validity_details"],
            "utility_details": utility_result["utility_details"],
        }

    def _empty_result(self) -> dict[str, Any]:
        return {
            "tag_scores": [],
            "miner_scores": [],
            "raw_miner_scores": [],
            "repeated_set_counts": [],
            "repeated_set_adjustments": [],
            "signal_scores": [],
            "validity_scores": [],
            "utility_scores": [],
            "cluster_labels": [],
            "scored_cluster_ids": [],
            "clusters": [],
            "normalized_responses": [],
            "spans": [],
            "validity_details": [],
            "utility_details": [],
        }

    def _apply_repeated_set_adjustment(
        self,
        *,
        coverage_scores: list[float],
        normalized_responses: list[list[str]],
    ) -> dict[str, Any]:
        """Multiply each miner's coverage score by RepeatedSetAdjustment(k)."""
        counts_by_key: dict[tuple[tuple[int, ...], ...], int] = {}
        keys = [self._set_key(response) for response in normalized_responses]
        for key in keys:
            counts_by_key[key] = counts_by_key.get(key, 0) + 1

        counts = [counts_by_key[key] for key in keys]
        adjustments = [repeated_set_adjustment(count) for count in counts]
        miner_scores = [
            float(score * adj) for score, adj in zip(coverage_scores, adjustments)
        ]
        return {
            "miner_scores": miner_scores,
            "repeated_set_counts": counts,
            "repeated_set_adjustments": adjustments,
        }

    def _set_key(self, response: list[str]) -> tuple[tuple[int, ...], ...]:
        tokenizer = self._model.tokenizer
        max_length = int(getattr(self._model, "max_seq_length", 256))
        cap = self.TOKEN_ID_CAP
        keys = []
        for tag in response:
            if not tag:
                continue
            ids = tokenizer(
                tag,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(ids)
            seen: dict[int, int] = {}
            capped: list[int] = []
            for token_id, token in zip(ids, tokens):
                if not any(ch.isalnum() for ch in token):
                    continue
                token_id = int(token_id)
                n = seen.get(token_id, 0)
                if n < cap:
                    capped.append(token_id)
                    seen[token_id] = n + 1
            keys.append(tuple(capped))
        return tuple(sorted(keys))

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

    def _unflatten_labels(
        self,
        normalized_responses: list[list[str]],
        labels: list[int],
    ) -> list[list[int]]:
        out: list[list[int]] = []
        cursor = 0
        for tags in normalized_responses:
            count = len(tags)
            out.append(labels[cursor : cursor + count])
            cursor += count
        return out
