from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_samples

from .clustering import ClusterView, TagClusterer
from .preprocessing import (
    ScoringContext,
    _load_span_nlp,
    build_scoring_context,
)


CONTENT_POS = frozenset({"NOUN", "PROPN", "ADJ", "NUM", "X"})
LABEL_ROOT_POS = frozenset({"NOUN", "PROPN", "X"})
VERBAL_POS = frozenset({"VERB", "AUX"})
_VOWELS = frozenset("aeiou")
_MODEL_TOKEN = re.compile(r"[a-z]+-?[0-9]+|[0-9]+[a-z]+")


class UtilityScorer:
    """Utility scorer: 0.70*SilhouetteUtility + 0.30*LabelFitness."""

    SILHOUETTE_WEIGHT = 0.70
    LABEL_FITNESS_WEIGHT = 0.30
    NEUTRAL_UTILITY = 0.5

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_tags_per_miner: int = 3,
        silhouette_weight: float = SILHOUETTE_WEIGHT,
        label_fitness_weight: float = LABEL_FITNESS_WEIGHT,
        model: Any | None = None,
        clusterer: TagClusterer | None = None,
    ) -> None:
        self.model_name = model_name
        self.n_tags_per_miner = n_tags_per_miner
        self.silhouette_weight = silhouette_weight
        self.label_fitness_weight = label_fitness_weight
        self._clusterer = clusterer or TagClusterer(
            model=model, model_name=model_name, n_tags=n_tags_per_miner
        )
        self._nlp = _load_span_nlp()

    def score(self, post: str, responses: list[list[str]]) -> dict[str, Any]:
        context = build_scoring_context(
            post=post,
            responses=responses,
            n_tags_per_miner=self.n_tags_per_miner,
        )
        return self.score_from_context(context)

    def score_from_context(
        self,
        context: ScoringContext,
        clustering: ClusterView | None = None,
    ) -> dict[str, Any]:
        if context.miner_count == 0:
            return {
                "utility_scores": [],
                "utility_details": [],
                "normalized_responses": [],
            }

        if not context.flat_tags:
            return {
                "utility_scores": [[] for _ in range(context.miner_count)],
                "utility_details": [[] for _ in range(context.miner_count)],
                "normalized_responses": context.normalized_responses,
            }

        view = clustering if clustering is not None else self._clusterer.build(context)

        silhouette_values, silhouette_utilities = self._silhouette_utilities(view)
        label_fitness = self._label_fitness(view.flat_tags)

        flat_scores: list[float] = []
        flat_details: list[dict[str, float]] = []
        for idx in range(view.n_flat):
            utility = (
                self.silhouette_weight * silhouette_utilities[idx]
                + self.label_fitness_weight * label_fitness[idx]
            )
            flat_scores.append(float(np.clip(utility, 0.0, 1.0)))
            flat_details.append(
                {
                    "silhouette": silhouette_values[idx],
                    "silhouette_utility": silhouette_utilities[idx],
                    "label_fitness": label_fitness[idx],
                }
            )

        return {
            "utility_scores": self._unflatten(
                context.normalized_responses, flat_scores
            ),
            "utility_details": self._unflatten(
                context.normalized_responses, flat_details
            ),
            "normalized_responses": context.normalized_responses,
        }

    def _silhouette_utilities(
        self, view: ClusterView
    ) -> tuple[list[float | None], list[float]]:
        n_flat = view.n_flat
        n_clusters = view.n_clusters
        # silhouette_samples needs 2 <= n_labels <= n_samples - 1.
        if n_clusters < 2 or n_clusters >= n_flat:
            return (
                [None] * n_flat,
                [self.NEUTRAL_UTILITY] * n_flat,
            )

        samples = silhouette_samples(view.tag_embeddings, view.labels, metric="cosine")
        values: list[float | None] = []
        utilities: list[float] = []
        for idx in range(n_flat):
            cluster_id = int(view.labels[idx])
            s = float(samples[idx])
            if view.cluster_size(cluster_id) <= 1 or not math.isfinite(s):
                # Singleton clusters (or any non-finite silhouette) cannot
                # yield a reliable value; fall back to neutral utility.
                values.append(None)
                utilities.append(self.NEUTRAL_UTILITY)
                continue
            values.append(s)
            # Map the cosine silhouette straight to utility, flooring s<=0 at 0.
            utilities.append(float(np.clip(s, 0.0, 1.0)))
        return values, utilities

    def _label_fitness(self, flat_tags: list[str]) -> list[float]:
        cache: dict[str, float] = {}
        scores: list[float] = []
        for tag in flat_tags:
            if tag not in cache:
                cache[tag] = label_fitness(tag, self._nlp)
            scores.append(cache[tag])
        return scores

    def _unflatten(
        self,
        normalized_responses: list[list[str]],
        flat_values: list[Any],
    ) -> list[list[Any]]:
        out: list[list[Any]] = []
        cursor = 0
        for tags in normalized_responses:
            count = len(tags)
            out.append(flat_values[cursor : cursor + count])
            cursor += count
        return out


def valid_tokens(doc: Any) -> list[Any]:
    return [tok for tok in doc if not tok.is_punct and not tok.is_space]


def is_acronym_like(tag: str) -> bool:
    compact = tag.strip()
    if " " in compact or not compact.isalpha():
        return False
    if not 2 <= len(compact) <= 5:
        return False
    vowels = sum(1 for ch in compact.lower() if ch in _VOWELS)
    return (vowels / len(compact)) < 0.4


def is_model_or_product_like(tag: str) -> bool:
    return any(_MODEL_TOKEN.fullmatch(token) for token in tag.split())


def entity_coverage_score(doc: Any) -> float:
    toks = valid_tokens(doc)
    if not toks:
        return 0.0
    covered = set()
    for ent in doc.ents:
        covered.update(range(ent.start, ent.end))
    hits = sum(1 for tok in toks if tok.i in covered)
    return hits / len(toks)


def noun_chunk_coverage_score(doc: Any) -> float:
    toks = valid_tokens(doc)
    if not toks:
        return 0.0
    covered = set()
    for chunk in getattr(doc, "noun_chunks", []):
        covered.update(range(chunk.start, chunk.end))
    hits = sum(1 for tok in toks if tok.i in covered)
    return hits / len(toks)


def label_fitness(tag: str, nlp: Any | None) -> float:
    stripped = tag.strip()
    if not stripped:
        return 0.0
    if nlp is None:
        return _lexical_label_fitness(stripped)

    doc = nlp(stripped.title())
    toks = valid_tokens(doc)
    if not toks:
        return 0.0

    # Strong semantic-label signals (case-insensitive checks on the raw tag).
    if is_acronym_like(stripped) or is_model_or_product_like(stripped):
        return 1.0
    if entity_coverage_score(doc) >= 0.8:
        return 1.0
    if noun_chunk_coverage_score(doc) >= 0.8:
        return 0.9

    content_tokens = [
        tok for tok in toks if tok.pos_ in CONTENT_POS and not tok.is_stop
    ]
    content_ratio = len(content_tokens) / len(toks)

    has_verb_or_aux = any(tok.pos_ in VERBAL_POS for tok in toks)
    is_sentence_like = has_verb_or_aux and len(toks) >= 3
    if is_sentence_like:
        return 0.4 if content_ratio >= 0.5 else 0.25

    root = next((tok for tok in toks if tok.dep_ == "ROOT"), toks[-1])
    if root.pos_ in LABEL_ROOT_POS:
        return 0.85 if content_ratio >= 0.6 else 0.65

    if content_ratio >= 0.8:
        return 0.8
    if content_ratio >= 0.6:
        return 0.65
    if content_ratio >= 0.4:
        return 0.45
    return 0.25


def _lexical_label_fitness(tag: str) -> float:
    """spaCy-free fallback biased toward a neutral score."""
    if is_acronym_like(tag) or is_model_or_product_like(tag):
        return 0.85
    tokens = re.findall(r"[a-z0-9]+", tag.lower())
    if not tokens:
        return 0.0
    if len(tokens) > 5:
        return 0.35
    return 0.5
