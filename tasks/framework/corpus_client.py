"""Validator-side reader for the S3 tweet corpus.

Verifies the public digest, self-selects a tweet, and verifies the fetched
object before use. boto3 and the digest loader are injectable for testing.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .corpus_hashing import DigestError, verify_digest, verify_object_bytes


class CorpusLeaseError(RuntimeError):
    """Base for validator-side corpus lease failures (distinct failure domains)."""


class CredentialFetchError(CorpusLeaseError):
    """Corpus credentials could not be fetched from the task server, or are invalid."""


class DigestFetchError(CorpusLeaseError):
    """The corpus digest could not be downloaded."""


class DigestVerificationError(CorpusLeaseError):
    """The corpus digest failed integrity or pinned-root_hash verification."""


class CorpusObjectError(CorpusLeaseError):
    """A corpus tweet object could not be fetched, hash-verified, or parsed."""


@dataclass(frozen=True)
class CorpusSelection:
    """One self-selected, hash-verified tweet plus its provenance."""

    tweet: dict[str, Any]
    entry: dict[str, Any]
    content_sha256: str
    root_hash: str


def _require_boto3() -> Any:
    try:
        import boto3  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via install extras
        raise RuntimeError(
            "boto3 is required to read the S3 corpus. Install it with the "
            "validator requirements (tag101) which include boto3."
        ) from exc
    return boto3


class CorpusReader:
    """Caches a verified digest and reads/verifies individual tweet objects.

    A changed credential ``rotation_id``/``bucket`` invalidates the cache.
    """

    def __init__(
        self,
        *,
        digest_ttl: float = 15 * 60,
        expected_root_hash: str = "",
        s3_client_factory: Callable[[Mapping[str, Any]], Any] | None = None,
        digest_loader: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
        http_timeout: float = 30.0,
        time_fn: Callable[[], float] = time.time,
    ):
        self._digest_ttl = max(0.0, float(digest_ttl))
        # Pin: when set, a downloaded digest whose root_hash differs is rejected.
        self._expected_root_hash = str(expected_root_hash or "").strip()
        self._s3_client_factory = s3_client_factory
        self._digest_loader = digest_loader
        self._http_timeout = float(http_timeout)
        self._time_fn = time_fn

        self._creds: dict[str, Any] = {}
        self._creds_key: tuple[str, str] | None = None
        self._s3_client: Any = None
        self._digest: dict[str, Any] | None = None
        self._entries: list[dict[str, Any]] = []
        self._root_hash: str = ""
        self._loaded_at: float = 0.0

    def update(self, credentials: Mapping[str, Any]) -> None:
        creds = dict(credentials)
        if not creds.get("bucket"):
            raise ValueError("corpus credentials are missing a bucket")
        key = (str(creds.get("rotation_id", "")), str(creds.get("bucket", "")))
        if key != self._creds_key:
            # Rotation (or first use): drop the cached client + digest.
            self._s3_client = None
            self._digest = None
            self._entries = []
            self._root_hash = ""
            self._loaded_at = 0.0
        self._creds = creds
        self._creds_key = key

    @property
    def root_hash(self) -> str:
        return self._root_hash

    @property
    def digest(self) -> dict[str, Any] | None:
        return self._digest

    def ensure_ready(self) -> None:
        if not self._creds:
            raise RuntimeError("corpus credentials have not been provided")
        now = self._time_fn()
        fresh = (
            self._digest is not None
            and (now - self._loaded_at) < self._digest_ttl
        )
        if fresh:
            return
        try:
            digest = self._load_digest()
        except Exception as exc:
            raise DigestFetchError(f"failed to download corpus digest: {exc}") from exc
        try:
            root_hash = verify_digest(digest)
        except DigestError as exc:
            raise DigestVerificationError(str(exc)) from exc
        if self._expected_root_hash and root_hash != self._expected_root_hash:
            raise DigestVerificationError(
                f"corpus root_hash {root_hash} does not match the pinned "
                f"expected_root_hash {self._expected_root_hash}"
            )
        entries = list(digest.get("tweets", []))
        if not entries:
            raise DigestVerificationError("corpus digest lists no tweets")
        self._root_hash = root_hash
        self._digest = digest
        self._entries = entries
        self._loaded_at = now

    def random_tweet(self, rng: random.Random | None = None) -> CorpusSelection:
        self.ensure_ready()
        chooser = rng or random
        entry = chooser.choice(self._entries)
        return self.fetch_entry(entry)

    def fetch_entry(self, entry: Mapping[str, Any]) -> CorpusSelection:
        key = str(entry.get("key") or "")
        if not key:
            raise DigestVerificationError("digest entry is missing an object key")
        try:
            data = self._s3_get(key)
        except Exception as exc:
            raise CorpusObjectError(f"failed to fetch corpus object {key!r}: {exc}") from exc
        try:
            verify_object_bytes(entry, data)
        except DigestError as exc:
            raise CorpusObjectError(str(exc)) from exc
        try:
            tweet = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise CorpusObjectError(f"corpus object {key!r} is not valid JSON: {exc}") from exc
        return CorpusSelection(
            tweet=tweet,
            entry=dict(entry),
            content_sha256=str(entry.get("sha256") or ""),
            root_hash=self._root_hash,
        )

    def _client(self) -> Any:
        if self._s3_client is None:
            if self._s3_client_factory is not None:
                self._s3_client = self._s3_client_factory(self._creds)
            else:
                self._s3_client = self._default_s3_client(self._creds)
        return self._s3_client

    def _default_s3_client(self, creds: Mapping[str, Any]) -> Any:
        boto3 = _require_boto3()
        kwargs: dict[str, Any] = {}
        if creds.get("region"):
            kwargs["region_name"] = str(creds["region"])
        if creds.get("endpoint_url"):
            kwargs["endpoint_url"] = str(creds["endpoint_url"])
        if creds.get("access_key_id") and creds.get("secret_access_key"):
            kwargs["aws_access_key_id"] = str(creds["access_key_id"])
            kwargs["aws_secret_access_key"] = str(creds["secret_access_key"])
            if creds.get("session_token"):
                kwargs["aws_session_token"] = str(creds["session_token"])
        return boto3.client("s3", **kwargs)

    def _s3_get(self, key: str) -> bytes:
        response = self._client().get_object(Bucket=str(self._creds["bucket"]), Key=key)
        body = response["Body"]
        return body.read()

    def _load_digest(self) -> dict[str, Any]:
        if self._digest_loader is not None:
            return self._digest_loader(self._creds)
        digest_url = str(self._creds.get("digest_url") or "")
        if digest_url:
            return self._load_digest_http(self._pinned_location(digest_url))
        digest_key = str(self._creds.get("digest_key") or "")
        if digest_key:
            return json.loads(self._s3_get(self._pinned_location(digest_key)).decode("utf-8"))
        raise DigestError(
            "corpus credentials provide neither digest_url nor digest_key"
        )

    def _pinned_location(self, latest: str) -> str:
        """Resolve the digest URL/key to load.

        When pinned, read the immutable ``digest/<root_hash>.json`` snapshot
        (same directory as ``latest.json``) so a newly published corpus does
        not change what a pinned validator sees until its pin is updated.
        """

        if not self._expected_root_hash:
            return latest
        base = latest.rsplit("/", 1)[0]
        return f"{base}/{self._expected_root_hash}.json"

    def _load_digest_http(self, url: str) -> dict[str, Any]:
        import httpx

        response = httpx.get(url, timeout=self._http_timeout)
        response.raise_for_status()
        return response.json()
