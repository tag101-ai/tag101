from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .preprocessing import ScoringContext, build_scoring_context


@dataclass(frozen=True)
class ClusterView:
    """Immutable view of the shared clustering for a single task.

    Signal (support/proximity/reliability), utility (silhouette) and the
    coverage-based miner aggregation all read the *same* labels, embeddings and
    centroids from this view so there is a single source of truth per task.
    """

    flat_tags: list[str]
    flat_miner_idx: list[int]
    miner_count: int
    tag_embeddings: np.ndarray  # [n_flat, d], L2-normalized (dot == cosine)
    labels: np.ndarray  # cluster id per flat tag
    cluster_to_indices: dict[int, list[int]]
    cluster_centroids: dict[int, np.ndarray]  # normalized centroid per cluster
    cluster_support: dict[int, float]  # RAW distinct-miner fraction
    cluster_miner_indices: dict[int, list[int]]
    scored_cluster_ids: list[int] = field(default_factory=list)

    @property
    def n_flat(self) -> int:
        return len(self.flat_tags)

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_to_indices)

    def cluster_size(self, cluster_id: int) -> int:
        return len(self.cluster_to_indices.get(cluster_id, []))

    def distinct_miner_count(self, cluster_id: int) -> int:
        return len(self.cluster_miner_indices.get(cluster_id, []))


class TagClusterer:
    """Embed tags once and cluster them once, shared across sub-scorers."""

    DISTANCE_THRESHOLD = 0.30
    # Bounds on the number of clusters the dynamic threshold may produce.
    MIN_CLUSTERS = 3
    MAX_CLUSTERS = 8

    def __init__(
        self,
        model: Any | None = None,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_tags: int = 3,
        distance_threshold: float = DISTANCE_THRESHOLD,
        min_clusters: int = MIN_CLUSTERS,
        max_clusters: int = MAX_CLUSTERS,
    ) -> None:
        self.model_name = model_name
        self.n_tags = n_tags
        self.distance_threshold = distance_threshold
        self.min_cluster_size = n_tags
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self._model = model or self._load_model(model_name)

    def build_from_responses(
        self,
        post: str,
        responses: list[list[str]],
        n_tags_per_miner: int,
    ) -> ClusterView:
        context = build_scoring_context(
            post=post,
            responses=responses,
            n_tags_per_miner=n_tags_per_miner,
        )
        return self.build(context)

    def build(self, context: ScoringContext) -> ClusterView:
        if not context.flat_tags:
            return ClusterView(
                flat_tags=[],
                flat_miner_idx=[],
                miner_count=context.miner_count,
                tag_embeddings=np.empty((0, 0)),
                labels=np.array([], dtype=int),
                cluster_to_indices={},
                cluster_centroids={},
                cluster_support={},
                cluster_miner_indices={},
                scored_cluster_ids=[],
            )

        tag_embeddings = self._embed_unique(context.flat_tags)
        labels = self._cluster(tag_embeddings)

        cluster_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            cluster_to_indices.setdefault(int(label), []).append(idx)

        cluster_centroids: dict[int, np.ndarray] = {}
        cluster_support: dict[int, float] = {}
        cluster_miner_indices: dict[int, list[int]] = {}
        for cluster_id, indices in cluster_to_indices.items():
            cluster_centroids[cluster_id] = self._centroid(tag_embeddings[indices])
            miner_set = {context.flat_miner_idx[i] for i in indices}
            cluster_miner_indices[cluster_id] = sorted(miner_set)
            cluster_support[cluster_id] = len(miner_set) / max(context.miner_count, 1)

        scored_cluster_ids = self._select_scored_clusters(
            cluster_to_indices=cluster_to_indices,
            cluster_miner_indices=cluster_miner_indices,
            cluster_support=cluster_support,
        )

        return ClusterView(
            flat_tags=context.flat_tags,
            flat_miner_idx=context.flat_miner_idx,
            miner_count=context.miner_count,
            tag_embeddings=tag_embeddings,
            labels=labels,
            cluster_to_indices=cluster_to_indices,
            cluster_centroids=cluster_centroids,
            cluster_support=cluster_support,
            cluster_miner_indices=cluster_miner_indices,
            scored_cluster_ids=scored_cluster_ids,
        )

    def _select_scored_clusters(
        self,
        *,
        cluster_to_indices: dict[int, list[int]],
        cluster_miner_indices: dict[int, list[int]],
        cluster_support: dict[int, float],
    ) -> list[int]:
        qualifying = [
            cluster_id
            for cluster_id in cluster_to_indices
            if len(cluster_miner_indices[cluster_id]) >= self.min_cluster_size
        ]
        # Sorted by support for a stable, meaningful order in the output records.
        qualifying.sort(
            key=lambda cid: (
                -cluster_support[cid],
                -len(cluster_to_indices[cid]),
                cid,
            )
        )
        return qualifying

    def _cluster(self, tag_embeddings: np.ndarray) -> np.ndarray:
        n_tags = tag_embeddings.shape[0]
        if n_tags <= 1:
            # distance_threshold cannot fit a single sample.
            return np.zeros(n_tags, dtype=int)

        # Dynamic threshold first, then clamp the cluster count to
        # [min_clusters, max_clusters] (bounded by n_tags) by re-fitting with
        # a fixed n_clusters when the threshold over- or under-splits.
        labels = self._agglomerative(
            n_clusters=None, distance_threshold=self.distance_threshold
        ).fit_predict(tag_embeddings)
        n_clusters = len(set(labels.tolist()))
        low = min(self.min_clusters, n_tags)
        high = min(self.max_clusters, n_tags)
        if low <= n_clusters <= high:
            return labels
        target = low if n_clusters < low else high
        return self._agglomerative(
            n_clusters=target, distance_threshold=None
        ).fit_predict(tag_embeddings)

    def _agglomerative(
        self, *, n_clusters: int | None, distance_threshold: float | None
    ) -> AgglomerativeClustering:
        try:
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average",
            )
        except TypeError:
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                distance_threshold=distance_threshold,
                affinity="cosine",
                linkage="average",
            )

    def _centroid(self, vectors: np.ndarray) -> np.ndarray:
        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid

    def _load_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install it with "
                "'pip install sentence-transformers'."
            ) from exc
        return SentenceTransformer(model_name)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    def _embed_unique(self, texts: list[str]) -> np.ndarray:
        """Embed distinct strings once, then broadcast to the full order."""
        unique = list(dict.fromkeys(texts))
        index = {tag: i for i, tag in enumerate(unique)}
        unique_vectors = self._embed_texts(unique)
        return unique_vectors[[index[tag] for tag in texts]]
