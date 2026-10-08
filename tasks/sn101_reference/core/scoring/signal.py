from __future__ import annotations

import math
import re
from typing import Any

import inflect
import numpy as np

from .clustering import ClusterView, TagClusterer
from .preprocessing import ScoringContext, build_scoring_context, _load_span_nlp


# Shared inflect engine for singular<->plural perturbations.
_INFLECT = inflect.engine()


# SimilarityLevel tiers: (min cosine to centroid, level).
SIMILARITY_TIERS: tuple[tuple[float, float], ...] = (
    (0.90, 0.85),
    (0.80, 0.70),
    (0.70, 0.55),
    (0.60, 0.35),
)

# Articles dropped when generating a meaning-preserving perturbation.
ARTICLES: frozenset[str] = frozenset({"a", "an", "the"})

# Content POS tags kept by lexical-core normalization.
_CONTENT_POS: frozenset[str] = frozenset({"NOUN", "PROPN", "ADJ", "NUM"})


class SignalScorer:

    SUPPORT_ALPHA = 9.0
    SIMILARITY_WEIGHT = 0.75
    DENSITY_WEIGHT = 0.25
    LOCAL_DENSITY_THRESHOLD = 0.75

    # Reliability exponents: exp(Phi) = R^PROXIMITY_EXP * P^PERTURB_EXP * C^CANON_EXP.
    PROXIMITY_EXP = 0.5
    PERTURB_EXP = 0.25
    CANON_EXP = 0.25

    # PerturbationStability linear map of raw cosine -> [0, 1].
    PERTURB_THETA_LOW = 0.65
    PERTURB_THETA_HIGH = 0.95

    # Measurement confidence
    PERTURB_CONFIDENCE_FULL = 3.0
    CANON_CONFIDENCE_FULL = 1.0

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_tags_per_miner: int = 3,
        support_alpha: float = SUPPORT_ALPHA,
        similarity_weight: float = SIMILARITY_WEIGHT,
        density_weight: float = DENSITY_WEIGHT,
        local_density_threshold: float = LOCAL_DENSITY_THRESHOLD,
        model: Any | None = None,
        clusterer: TagClusterer | None = None,
    ) -> None:
        self.model_name = model_name
        self.n_tags_per_miner = n_tags_per_miner
        self.support_alpha = support_alpha
        self.similarity_weight = similarity_weight
        self.density_weight = density_weight
        self.local_density_threshold = local_density_threshold
        self._model = model
        self._clusterer = clusterer or TagClusterer(
            model=model, model_name=model_name, n_tags=n_tags_per_miner
        )

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
                "signal_scores": [],
                "clusters": [],
                "normalized_responses": [],
            }

        if not context.flat_tags:
            return {
                "signal_scores": [[] for _ in range(context.miner_count)],
                "clusters": [],
                "normalized_responses": context.normalized_responses,
            }

        view = clustering if clustering is not None else self._clusterer.build(context)

        flat_signal_scores = self._compute_signal_scores(view)
        signal_scores = self._unflatten_scores(
            normalized_responses=context.normalized_responses,
            flat_scores=flat_signal_scores,
        )
        clusters = self._build_cluster_records(view)
        return {
            "signal_scores": signal_scores,
            "clusters": clusters,
            "normalized_responses": context.normalized_responses,
        }

    def _log_support(self, raw_support: float) -> float:
        alpha = self.support_alpha
        if alpha <= 0:
            return raw_support
        return math.log1p(alpha * raw_support) / math.log1p(alpha)

    def _cluster_proximity(
        self,
        indices: list[int],
        flat_tags: list[str],
        tag_embeddings: np.ndarray,
        centroid: np.ndarray,
    ) -> dict[str, float]:
        """Proximity per unique tag string within a cluster."""
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        tag_vectors: dict[str, list[np.ndarray]] = {}
        for idx in indices:
            tag_vectors.setdefault(flat_tags[idx], []).append(tag_embeddings[idx])

        unique_tags = list(tag_vectors)
        mean_vectors: dict[str, np.ndarray] = {}
        centroid_sim: dict[str, float] = {}
        for tag, vectors in tag_vectors.items():
            vector = np.mean(vectors, axis=0)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            mean_vectors[tag] = vector
            centroid_sim[tag] = float(np.dot(vector, centroid))

        closest_tag = max(centroid_sim, key=centroid_sim.get) if centroid_sim else None

        proximity: dict[str, float] = {}
        for tag in unique_tags:
            similarity_level = self._similarity_level(
                tag=tag,
                sim=centroid_sim[tag],
                is_closest=(tag == closest_tag),
            )
            local_density = self._local_density(
                tag=tag,
                unique_tags=unique_tags,
                mean_vectors=mean_vectors,
            )
            proximity[tag] = (
                self.similarity_weight * similarity_level
                + self.density_weight * local_density
            )
        return proximity

    def _similarity_level(
        self,
        *,
        tag: str,
        sim: float,
        is_closest: bool,
    ) -> float:
        if is_closest:
            return 1.0
        for threshold, level in SIMILARITY_TIERS:
            if sim >= threshold:
                return level
        return 0.0

    def _local_density(
        self,
        *,
        tag: str,
        unique_tags: list[str],
        mean_vectors: dict[str, np.ndarray],
    ) -> float:
        if len(unique_tags) <= 1:
            return 0.0
        tag_vector = mean_vectors[tag]
        threshold = self.local_density_threshold
        neighbors = sum(
            1
            for other in unique_tags
            if other != tag
            and float(np.dot(tag_vector, mean_vectors[other])) >= threshold
        )
        return neighbors / (len(unique_tags) - 1)

    def _compute_signal_scores(self, view: ClusterView) -> list[float]:
        cluster_proximity: dict[int, dict[str, float]] = {}
        for cluster_id, indices in view.cluster_to_indices.items():
            cluster_proximity[cluster_id] = self._cluster_proximity(
                indices=indices,
                flat_tags=view.flat_tags,
                tag_embeddings=view.tag_embeddings,
                centroid=view.cluster_centroids[cluster_id],
            )

        # P(t) and C(t) are cluster-independent, so resolve them once per tag.
        reliability = self._reliability_by_tag(view)

        signal_scores: list[float] = []
        for idx, label in enumerate(view.labels):
            cluster_id = int(label)
            log_support = self._log_support(view.cluster_support[cluster_id])
            tag = view.flat_tags[idx]
            proximity = cluster_proximity[cluster_id].get(tag, 1.0)
            perturbation, canonical = reliability[tag]
            signal = (
                log_support
                * (proximity**self.PROXIMITY_EXP)
                * (perturbation**self.PERTURB_EXP)
                * (canonical**self.CANON_EXP)
            )
            signal_scores.append(float(np.clip(signal, 0.0, 1.0)))
        return signal_scores

    def _reliability_by_tag(self, view: ClusterView) -> dict[str, tuple[float, float]]:
        """Return ``{tag: (PerturbationStability, CanonicalConsistency)}``."""
        unique_tags = list(dict.fromkeys(view.flat_tags))
        variants_by_tag = {tag: self._perturbation_variants(tag) for tag in unique_tags}
        views_by_tag = {tag: self._canonical_views(tag) for tag in unique_tags}

        embeddings = self._embedding_lookup(view)
        needed = {
            text
            for tag in unique_tags
            for text in (*variants_by_tag[tag], *views_by_tag[tag])
            if text not in embeddings
        }
        if needed:
            ordered = sorted(needed)
            vectors = self._embed_texts(ordered)
            for text, vector in zip(ordered, vectors):
                embeddings[text] = vector

        reliability: dict[str, tuple[float, float]] = {}
        for tag in unique_tags:
            tag_vector = embeddings[tag]
            variant_vectors = [embeddings[text] for text in variants_by_tag[tag]]
            view_vectors = [embeddings[text] for text in views_by_tag[tag]]
            reliability[tag] = (
                self._perturbation_stability(tag_vector, variant_vectors),
                self._canonical_consistency(view_vectors),
            )
        return reliability

    def _embedding_lookup(self, view: ClusterView) -> dict[str, np.ndarray]:
        lookup: dict[str, np.ndarray] = {}
        for tag, vector in zip(view.flat_tags, view.tag_embeddings):
            if tag not in lookup:
                lookup[tag] = vector
        return lookup

    def _perturbation_variants(self, tag: str) -> list[str]:
        """Meaning-preserving *surface* rewrites of an already-normalized tag."""
        variants: set[str] = set()
        variants.update(self._punctuation_variants(tag))
        variants.update(self._article_variants(tag))
        variants.update(self._number_variants(tag))
        variants.update(self._spacing_variants(tag))
        variants.discard(tag)
        return sorted(variant for variant in variants if variant)

    def _punctuation_variants(self, tag: str) -> list[str]:
        stripped = self._norm(re.sub(r"[^\w\s]", "", tag))
        return [stripped] if stripped else []

    def _article_variants(self, tag: str) -> list[str]:
        without = self._norm(
            " ".join(token for token in tag.split() if token not in ARTICLES)
        )
        return [without] if without else []

    def _number_variants(self, tag: str) -> list[str]:
        tokens = tag.split()
        if not tokens:
            return []
        toggled = self._toggle_number(tokens[-1])
        if not toggled or toggled == tokens[-1]:
            return []
        return [self._norm(" ".join(tokens[:-1] + [toggled]))]

    @staticmethod
    def _toggle_number(word: str) -> str:
        """Singular<->plural toggle of one word via inflect."""
        if len(word) <= 2:
            return ""
        toggled = _INFLECT.singular_noun(word) or _INFLECT.plural_noun(word)
        return toggled if isinstance(toggled, str) else ""

    def _spacing_variants(self, tag: str) -> list[str]:
        if "-" not in tag:
            return []
        candidates = {
            self._norm(tag.replace("-", " ")),
            self._norm(tag.replace("-", "")),
        }
        return [candidate for candidate in candidates if candidate]

    def _canonical_views(self, tag: str) -> list[str]:
        """Canonical tag views: normalized form, lemma, lexical core (content words)."""
        views = [tag]
        for transform in (self._lemmatize, self._lexical_core):
            view = transform(tag)
            if view:
                views.append(view)
        return list(dict.fromkeys(view for view in views if view))

    def _lemmatize(self, text: str) -> str:
        nlp = _load_span_nlp()
        if nlp is None:
            return ""
        doc = nlp(text)
        lemmas = [token.lemma_.lower() for token in doc if not token.is_space]
        return self._norm(" ".join(lemmas))

    def _lexical_core(self, tag: str) -> str:
        nlp = _load_span_nlp()
        if nlp is None:
            return ""
        doc = nlp(tag)
        kept = [
            token.text.casefold()
            for token in doc
            if not token.is_space and not token.is_punct and token.pos_ in _CONTENT_POS
        ]
        result = self._norm(" ".join(kept))
        original = self._norm(tag.casefold())
        return result if result and result != original else ""

    def _perturbation_stability(
        self,
        tag_vector: np.ndarray,
        variant_vectors: list[np.ndarray],
    ) -> float:
        """Confidence-blended stability under lexical perturbation."""
        if not variant_vectors:
            return 1.0
        sims = [float(np.dot(tag_vector, vector)) for vector in variant_vectors]
        raw = sum(sims) / len(sims)
        span = self.PERTURB_THETA_HIGH - self.PERTURB_THETA_LOW
        observed = (
            self._clip01((raw - self.PERTURB_THETA_LOW) / span) if span > 0 else 1.0
        )
        confidence = self._confidence(
            len(variant_vectors), self.PERTURB_CONFIDENCE_FULL
        )
        return 1.0 - confidence * (1.0 - observed)

    def _canonical_consistency(self, view_vectors: list[np.ndarray]) -> float:
        """Confidence-blended agreement across canonical views."""
        k = len(view_vectors)
        if k <= 1:
            return 1.0
        sims = [
            float(np.dot(view_vectors[i], view_vectors[j]))
            for i in range(k)
            for j in range(i + 1, k)
        ]
        observed = self._clip01(sum(sims) / len(sims))
        confidence = self._confidence(k - 1, self.CANON_CONFIDENCE_FULL)
        return 1.0 - confidence * (1.0 - observed)

    @staticmethod
    def _confidence(count: int, full: float) -> float:
        if full <= 0:
            return 1.0
        return min(1.0, count / full)

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            self._model = self._load_model(self.model_name)
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    def _load_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install it with "
                "'pip install sentence-transformers'."
            ) from exc
        return SentenceTransformer(model_name)

    def _unflatten_scores(
        self,
        normalized_responses: list[list[str]],
        flat_scores: list[float],
    ) -> list[list[float]]:
        scores: list[list[float]] = []
        cursor = 0
        for tags in normalized_responses:
            count = len(tags)
            scores.append(flat_scores[cursor : cursor + count])
            cursor += count
        return scores

    def _build_cluster_records(self, view: ClusterView) -> list[dict[str, Any]]:
        scored = set(view.scored_cluster_ids)
        records: list[dict[str, Any]] = []
        for cluster_id in sorted(view.cluster_to_indices):
            indices = view.cluster_to_indices[cluster_id]
            support = view.cluster_support[cluster_id]
            records.append(
                {
                    "cluster_id": cluster_id,
                    "support": support,
                    "log_support": self._log_support(support),
                    "in_topk": cluster_id in scored,
                    "tags": [view.flat_tags[i] for i in indices],
                    "miner_indices": view.cluster_miner_indices[cluster_id],
                }
            )
        return records
