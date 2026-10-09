from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np

from .preprocessing import ScoringContext, build_scoring_context


class ValidityScorer:
    """Independent validity scorer with span extraction."""

    _URL_PATTERN = re.compile(
        r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(com|org|net|io|ai|co)\b)",
        re.IGNORECASE,
    )
    MAX_TAG_CHARS = 100

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_tags_per_miner: int = 3,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.n_tags_per_miner = n_tags_per_miner
        self._model = model or self._load_model(model_name)

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
        tag_embeddings: Any | None = None,
    ) -> dict[str, Any]:
        if context.miner_count == 0:
            return {
                "validity_scores": [],
                "validity_details": [],
                "spans": [],
                "normalized_responses": [],
            }

        if not context.flat_tags:
            return {
                "validity_scores": [[] for _ in range(context.miner_count)],
                "validity_details": [[] for _ in range(context.miner_count)],
                "spans": context.spans,
                "normalized_responses": context.normalized_responses,
            }

        post_embedding = self._embed_texts([context.post])[0]
        span_embeddings = self._embed_texts(context.spans)
        # Reuse the shared tag embeddings when provided (avoids re-encoding
        # the same flat tags the clusterer already embedded).
        if tag_embeddings is None:
            tag_embeddings = self._embed_texts(context.flat_tags)

        flat_validity_scores, raw_details = self._compute_flat_scores(
            tags=context.flat_tags,
            post_embedding=post_embedding,
            spans=context.spans,
            span_embeddings=span_embeddings,
            tag_embeddings=tag_embeddings,
        )
        validity_scores = self._unflatten_scores(
            normalized_responses=context.normalized_responses,
            flat_scores=flat_validity_scores,
        )
        return {
            "validity_scores": validity_scores,
            "validity_details": self._unflatten_details(
                normalized_responses=context.normalized_responses,
                flat_details=raw_details,
            ),
            "spans": context.spans,
            "normalized_responses": context.normalized_responses,
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

    def _compute_flat_scores(
        self,
        tags: list[str],
        post_embedding: np.ndarray,
        spans: list[str],
        span_embeddings: np.ndarray,
        tag_embeddings: np.ndarray,
    ) -> tuple[list[float], list[dict[str, float]]]:
        scores: list[float] = []
        details: list[dict[str, float]] = []

        for idx, tag in enumerate(tags):
            tag_embedding = tag_embeddings[idx]
            sim_post = self._scale_similarity(
                float(np.dot(tag_embedding, post_embedding))
            )
            sim_span = self._scale_similarity(
                float(np.max(np.dot(span_embeddings, tag_embedding)))
            )
            lexical_overlap = self._lexical_overlap_score(tag, spans)
            base_score = max(sim_post, sim_span, lexical_overlap)

            format_score = self._format_score(tag)
            raw_validity = base_score * format_score
            validity = self._tier_map_validity(raw_validity)

            scores.append(validity)
            details.append(
                {
                    "base_score": base_score,
                    "sim_post": sim_post,
                    "sim_span": sim_span,
                    "lexical_overlap": lexical_overlap,
                    "format_score": format_score,
                    "raw_validity": raw_validity,
                }
            )
        return scores, details

    def _scale_similarity(
        self,
        sim: float,
        low: float = 0.30,
        high: float = 0.75,
    ) -> float:
        if sim <= low:
            return 0.0
        if sim >= high:
            return 1.0
        return (sim - low) / (high - low)

    def _tokenize(self, text: str) -> list[str]:
        raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [self._normalize_token(token) for token in raw_tokens]

    def _normalize_token(self, token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return f"{token[:-3]}y"
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    def _lexical_overlap_score(self, tag: str, spans: list[str]) -> float:
        tag_tokens = self._tokenize(tag)
        if not tag_tokens:
            return 0.0

        best = 0.0
        for span in spans:
            span_tokens_set = set(self._tokenize(span))
            if not span_tokens_set:
                continue
            hits = sum(1 for token in tag_tokens if token in span_tokens_set)
            best = max(best, hits / len(tag_tokens))
        return best

    def _format_score(self, tag: str) -> float:
        normalized = tag.strip()
        if not normalized:
            return 0.0
        if len(normalized) > self.MAX_TAG_CHARS:
            return 0.0
        if self._URL_PATTERN.search(normalized):
            return 0.0

        token_count = len(self._tokenize(normalized))
        if token_count < 1 or token_count > 5:
            return 0.0

        if self._is_numeric_only(normalized):
            return 0.0
        if self._is_punctuation_only(normalized):
            return 0.0
        if self._is_emoji_only(normalized):
            return 0.0
        if self._has_too_many_special_symbols(normalized):
            return 0.0
        return 1.0

    def _is_numeric_only(self, text: str) -> bool:
        return bool(re.fullmatch(r"[\d\s.,:/%+\-]+", text))

    def _is_punctuation_only(self, text: str) -> bool:
        return bool(re.fullmatch(r"[^\w\s]+", text, flags=re.UNICODE))

    def _is_emoji_only(self, text: str) -> bool:
        compact = "".join(ch for ch in text if not ch.isspace())
        if not compact:
            return False
        if any(ch.isalnum() for ch in compact):
            return False
        return all(unicodedata.category(ch).startswith(("S", "P")) for ch in compact)

    def _has_too_many_special_symbols(self, text: str) -> bool:
        compact = [ch for ch in text if not ch.isspace()]
        if not compact:
            return False
        special_count = sum(
            1
            for ch in compact
            if not ch.isalnum() and ch not in {"-", "_", "&", "/", "+"}
        )
        return (special_count / len(compact)) > 0.3

    def _tier_map_validity(self, raw_validity: float) -> float:
        if raw_validity < 0.15:
            return 0.0
        if raw_validity < 0.35:
            return 0.3
        if raw_validity < 0.65:
            return 0.6
        return 1.0

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

    def _unflatten_details(
        self,
        normalized_responses: list[list[str]],
        flat_details: list[dict[str, float]],
    ) -> list[list[dict[str, float]]]:
        details: list[list[dict[str, float]]] = []
        cursor = 0
        for tags in normalized_responses:
            count = len(tags)
            details.append(flat_details[cursor : cursor + count])
            cursor += count
        return details
