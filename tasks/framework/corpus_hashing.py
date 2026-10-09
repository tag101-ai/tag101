"""Deterministic corpus hashing and digest construction (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


DIGEST_SCHEMA = "sn101-corpus-digest"
DIGEST_VERSION = 1
ROOT_HASH_ALGO = "sha256(sorted('{tweet_uuid}:{sha256}') joined by '\\n')"
DEFAULT_CORPUS_PREFIX = "corpus"
DEFAULT_OBJECT_KEY_TEMPLATE = "{prefix}/tweets/{tweet_uuid}.json"
DIGEST_LATEST_KEY_TEMPLATE = "{prefix}/digest/latest.json"
DIGEST_SNAPSHOT_KEY_TEMPLATE = "{prefix}/digest/{root_hash}.json"

CORPUS_TWEET_FIELDS: tuple[str, ...] = (
    "tweet_uuid",
    "tweet_id",
    "source_tweet_id",
    "author",
    "created_at",
    "url",
    "text",
    "retweet_count",
)


@dataclass(frozen=True)
class CorpusBuild:
    """Digest plus per-tweet object bytes, ready to publish."""

    digest: dict[str, Any]
    objects: list[tuple[str, bytes]] = field(default_factory=list)

    @property
    def root_hash(self) -> str:
        return str(self.digest.get("root_hash", ""))


def _clean_str(value: Any) -> str:
    return "" if value is None else str(value)


def build_corpus_tweet(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize a canonical/DB tweet into a stable corpus object, or None."""

    text = _clean_str(record.get("text")).strip()
    tweet_uuid = _clean_str(record.get("tweet_uuid")).strip()
    if not text or not tweet_uuid:
        return None
    source_tweet_id = _clean_str(
        record.get("source_tweet_id") or record.get("tweet_id")
    )
    try:
        retweet_count = int(record.get("retweet_count") or 0)
    except (TypeError, ValueError):
        retweet_count = 0
    return {
        "tweet_uuid": tweet_uuid,
        "tweet_id": source_tweet_id,
        "source_tweet_id": source_tweet_id,
        "author": _clean_str(record.get("author")),
        "created_at": _clean_str(record.get("created_at")),
        "url": _clean_str(record.get("url")),
        "text": text,
        "retweet_count": retweet_count,
    }


def serialize_tweet_object(tweet: Mapping[str, Any]) -> bytes:
    """Canonical byte serialization for a stored corpus tweet object."""

    return json.dumps(
        dict(tweet),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_key(prefix: str, tweet_uuid: str, *, template: str = DEFAULT_OBJECT_KEY_TEMPLATE) -> str:
    key = template.format(prefix=prefix, tweet_uuid=tweet_uuid)
    return key.lstrip("/")


def digest_latest_key(prefix: str) -> str:
    return DIGEST_LATEST_KEY_TEMPLATE.format(prefix=prefix).lstrip("/")


def digest_snapshot_key(prefix: str, root_hash: str) -> str:
    return DIGEST_SNAPSHOT_KEY_TEMPLATE.format(
        prefix=prefix, root_hash=root_hash
    ).lstrip("/")


def compute_root_hash(entries: Sequence[Mapping[str, Any]]) -> str:
    """Recompute the corpus root hash from digest entries (uuid + sha256)."""

    lines = sorted(
        f"{_clean_str(entry.get('tweet_uuid'))}:{_clean_str(entry.get('sha256'))}"
        for entry in entries
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def build_corpus(
    records: Sequence[Mapping[str, Any]],
    *,
    prefix: str = DEFAULT_CORPUS_PREFIX,
    generated_at: float | None = None,
    object_key_template: str = DEFAULT_OBJECT_KEY_TEMPLATE,
) -> CorpusBuild:
    """Build the digest and per-tweet object bytes from tweet records."""

    normalized: dict[str, dict[str, Any]] = {}
    for record in records:
        tweet = build_corpus_tweet(record)
        if tweet is None:
            continue
        normalized[tweet["tweet_uuid"]] = tweet

    objects: list[tuple[str, bytes]] = []
    entries: list[dict[str, Any]] = []
    for tweet_uuid in sorted(normalized):
        tweet = normalized[tweet_uuid]
        data = serialize_tweet_object(tweet)
        key = object_key(prefix, tweet_uuid, template=object_key_template)
        sha256 = compute_sha256(data)
        objects.append((key, data))
        # Public digest keeps only what verification/fetch need; source
        # metadata and full text stay in the private per-tweet objects.
        entries.append(
            {
                "tweet_uuid": tweet_uuid,
                "sha256": sha256,
                "key": key,
            }
        )

    digest = {
        "schema": DIGEST_SCHEMA,
        "version": DIGEST_VERSION,
        "generated_at": float(time.time() if generated_at is None else generated_at),
        "prefix": prefix,
        "object_key_template": object_key_template,
        "root_hash_algo": ROOT_HASH_ALGO,
        "count": len(entries),
        "root_hash": compute_root_hash(entries),
        "tweets": entries,
    }
    return CorpusBuild(digest=digest, objects=objects)


class DigestError(ValueError):
    """Raised when a digest fails integrity verification."""


def verify_digest(digest: Mapping[str, Any]) -> str:
    """Recompute and check the digest ``root_hash``; return it or raise."""

    tweets = digest.get("tweets")
    if not isinstance(tweets, Sequence):
        raise DigestError("digest is missing a 'tweets' list")
    recorded = _clean_str(digest.get("root_hash"))
    if not recorded:
        raise DigestError("digest is missing 'root_hash'")
    recomputed = compute_root_hash(list(tweets))
    if recomputed != recorded:
        raise DigestError(
            f"digest root_hash mismatch: recorded={recorded} recomputed={recomputed}"
        )
    return recomputed


def verify_object_bytes(entry: Mapping[str, Any], data: bytes) -> None:
    """Check downloaded object bytes against the digest entry ``sha256``."""

    expected = _clean_str(entry.get("sha256"))
    actual = compute_sha256(data)
    if expected != actual:
        raise DigestError(
            f"tweet object sha256 mismatch for {entry.get('tweet_uuid')!r}: "
            f"expected={expected} actual={actual}"
        )
