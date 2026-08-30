from __future__ import annotations

import importlib
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

SPAN_ENTITY_MODEL = "en_core_web_sm"
DISALLOWED_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
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
    """Return the canonical representation used by every scoring stage."""
    value = unicodedata.normalize("NFKC", tag).casefold()
    characters: list[str] = []
    for character in value:
        category = unicodedata.category(character)
        if character.isspace() or category.startswith("Z"):
            characters.append(" ")
        elif category not in DISALLOWED_UNICODE_CATEGORIES:
            characters.append(character)

    # Case folding can produce decomposed output, so normalize once more after it.
    canonical = unicodedata.normalize("NFKC", "".join(characters))
    return re.sub(r"\s+", " ", canonical).strip()


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
    tokens = [
        token
        for token in span
        if not token.is_punct and not token.is_space
    ]
    while tokens and tokens[0].is_stop:
        tokens.pop(0)
    while tokens and tokens[-1].is_stop:
        tokens.pop()
    return " ".join(token.text for token in tokens)


def unflatten_scores(
    normalized_responses: list[list[str]],
    flat_tag_scores: list[float],
    flat_consensus_scores: list[float],
    flat_validity_scores: list[float],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    tag_scores: list[list[float]] = []
    consensus_scores: list[list[float]] = []
    validity_scores: list[list[float]] = []

    cursor = 0
    for tags in normalized_responses:
        count = len(tags)
        tag_scores.append(flat_tag_scores[cursor: cursor + count])
        consensus_scores.append(flat_consensus_scores[cursor: cursor + count])
        validity_scores.append(flat_validity_scores[cursor: cursor + count])
        cursor += count

    return tag_scores, consensus_scores, validity_scores


def aggregate_miner_score(scores: list[float], top_k: int) -> float:
    if not scores:
        return 0.0
    top_scores = sorted(scores, reverse=True)[:top_k]
    return float(sum(top_scores) / top_k)


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
