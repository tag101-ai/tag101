from __future__ import annotations

import importlib
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

SPAN_ENTITY_MODEL = "en_core_web_sm"
SPAN_ENTITY_LABELS = frozenset(
    {
        "PERSON",
        "NORP",
        "FAC",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
    }
)


@dataclass
class ScoringContext:
    post: str
    responses: list[list[str]]
    normalized_responses: list[list[str]]
    flat_tags: list[str]
    flat_miner_idx: list[int]
    spans: list[str]

    @property
    def miner_count(self) -> int:
        return len(self.normalized_responses)


def normalize_tag(tag: str) -> str:
    tag = unicodedata.normalize("NFKC", tag)
    tag = "".join(ch for ch in tag if unicodedata.category(ch) != "Cf")
    return re.sub(r"\s+", " ", tag.strip().lower())


def preprocess_responses(
    responses: list[list[str]],
    n_tags_per_miner: int,
) -> list[list[str]]:
    processed: list[list[str]] = []
    for tags in responses:
        deduped: list[str] = []
        seen: set[str] = set()
        for raw_tag in tags[:n_tags_per_miner]:
            tag = normalize_tag(raw_tag)
            if not tag or tag in seen:
                continue
            seen.add(tag)
            deduped.append(tag)
        processed.append(deduped)
    return processed


def flatten_responses(
    normalized_responses: list[list[str]],
) -> tuple[list[str], list[int]]:
    flat_tags: list[str] = []
    flat_miner_idx: list[int] = []
    for miner_idx, tags in enumerate(normalized_responses):
        for tag in tags:
            flat_tags.append(tag)
            flat_miner_idx.append(miner_idx)
    return flat_tags, flat_miner_idx


def build_spans(post: str) -> list[str]:
    spans: list[str] = []
    seen: set[str] = set()
    post_clean = post.strip()
    if post_clean:
        _append_unique_span(spans, seen, post_clean)

    for sentence in re.split(r"[.!?\n]+", post):
        sentence = sentence.strip()
        if sentence:
            _append_unique_span(spans, seen, sentence)

    for span in extract_entity_and_noun_chunk_spans(post):
        _append_unique_span(spans, seen, span)

    return spans if spans else [post]


def extract_entity_and_noun_chunk_spans(post: str) -> list[str]:
    nlp = _load_span_nlp()
    if nlp is None or not post.strip():
        return []

    doc = nlp(post)
    spans: list[str] = []
    seen: set[str] = set()

    for ent in doc.ents:
        if ent.label_ in SPAN_ENTITY_LABELS:
            _append_unique_span(spans, seen, ent.text)

    for chunk in getattr(doc, "noun_chunks", []):
        _append_unique_span(spans, seen, _trim_stop_tokens(chunk))

    return spans


@lru_cache(maxsize=1)
def _load_span_nlp() -> Any | None:
    try:
        spacy = importlib.import_module("spacy")
        return spacy.load(SPAN_ENTITY_MODEL)
    except (ModuleNotFoundError, OSError):
        return None


def _append_unique_span(spans: list[str], seen: set[str], text: str) -> None:
    span = _normalize_span(text)
    if not span:
        return
    key = span.lower()
    if key in seen:
        return
    seen.add(key)
    spans.append(span)


def _normalize_span(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("\"'`.,;:!?()[]{}"))


def _trim_stop_tokens(span: Any) -> str:
    tokens = [token for token in span if not token.is_punct and not token.is_space]
    while tokens and tokens[0].is_stop:
        tokens.pop(0)
    while tokens and tokens[-1].is_stop:
        tokens.pop()
    return " ".join(token.text for token in tokens)


def unflatten_scores(
    normalized_responses: list[list[str]],
    flat_tag_scores: list[float],
    flat_signal_scores: list[float],
    flat_validity_scores: list[float],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    tag_scores: list[list[float]] = []
    signal_scores: list[list[float]] = []
    validity_scores: list[list[float]] = []

    cursor = 0
    for tags in normalized_responses:
        count = len(tags)
        tag_scores.append(flat_tag_scores[cursor : cursor + count])
        signal_scores.append(flat_signal_scores[cursor : cursor + count])
        validity_scores.append(flat_validity_scores[cursor : cursor + count])
        cursor += count

    return tag_scores, signal_scores, validity_scores


MAX_TAG_SCORE = 1.0


def aggregate_miner_score(scores: list[float], top_k: int) -> float:
    if not scores:
        return 0.0
    top_scores = sorted(scores, reverse=True)[:top_k]
    return float(sum(top_scores) / top_k)


def aggregate_miner_score_by_coverage(
    *,
    flat_tag_scores: list[float],
    flat_miner_idx: list[int],
    labels: list[int],
    scored_cluster_ids: list[int],
    miner_count: int,
    n_tags: int,
) -> list[float]:
    if miner_count == 0:
        return []

    scored = list(scored_cluster_ids)
    if scored:
        # Per-miner best TagScore within each qualifying cluster.
        miner_best: list[dict[int, float]] = [{} for _ in range(miner_count)]
        scored_set = set(scored)
        for score, miner_idx, label in zip(flat_tag_scores, flat_miner_idx, labels):
            cid = int(label)
            if cid not in scored_set:
                continue
            current = miner_best[miner_idx].get(cid)
            if current is None or score > current:
                miner_best[miner_idx][cid] = score

        # Denominator = the absolute ceiling for an N-tag miner.
        denominator = n_tags * MAX_TAG_SCORE
        return [
            float(
                np.clip(
                    sum(sorted(miner_best[m].values(), reverse=True)[:n_tags])
                    / denominator,
                    0.0,
                    1.0,
                )
            )
            for m in range(miner_count)
        ]

    # Fallback: no qualifying cluster -> pad missing tags as 0, then top-k mean.
    per_miner_tag_scores: list[list[float]] = [[] for _ in range(miner_count)]
    for score, miner_idx in zip(flat_tag_scores, flat_miner_idx):
        per_miner_tag_scores[miner_idx].append(score)
    return [
        aggregate_miner_score(per_miner_tag_scores[m], n_tags)
        for m in range(miner_count)
    ]


def repeated_set_adjustment(
    count: int,
    *,
    c: float = 50.0,
    p: float = 5.5,
    floor: float = 0.0,
) -> float:
    """Hill multiplier that dampens miners submitting an identical tag set."""
    k = max(int(count), 1)
    if c <= 0:
        return 1.0
    hill = 1.0 / (1.0 + (k / c) ** p)
    return float(floor + (1.0 - floor) * hill)


def build_scoring_context(
    post: str,
    responses: list[list[str]],
    n_tags_per_miner: int,
) -> ScoringContext:
    normalized_responses = preprocess_responses(
        responses=responses,
        n_tags_per_miner=n_tags_per_miner,
    )
    flat_tags, flat_miner_idx = flatten_responses(normalized_responses)
    spans = build_spans(post)
    return ScoringContext(
        post=post,
        responses=responses,
        normalized_responses=normalized_responses,
        flat_tags=flat_tags,
        flat_miner_idx=flat_miner_idx,
        spans=spans,
    )
