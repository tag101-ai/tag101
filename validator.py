"""Bittensor validator entry point."""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from pathlib import Path
from typing import Any

from . import SCOREBOARD_VERSION
from ._logging import configure_logging, get_logger
from .chain.http_transport import HttpTransport
from .chain.private_axons import is_usable_axon, private_axons_by_hotkey
from .chain.runtime import ChainRuntime
from .chain.scoreboard import ScoreBoard
from .chain.selector import CandidatePolicy, miner_candidates
from .chain.settings import build_config
from .chain.weights import (
    WeightSubmissionTimeout,
    aggregate_validator_scores,
    qualified_validator_stake,
    submit_weights_with_deadline,
)
from .protocol import envelope_from_lease
from .tasks import LocalLeaseContext, TaskRegistry, TaskServerClient, default_registry
from .tasks.framework.corpus_client import (
    CorpusObjectError,
    CorpusReader,
    CredentialFetchError,
    DigestFetchError,
    DigestVerificationError,
)

# Trust aggregated scoreboards only above this fraction of qualified validator stake.
MIN_AGGREGATED_STAKE_FRACTION = 0.5

# Corpus provenance forwarded from the local lease into the report so each
# scored question is auditable against the public digest.
_LEASE_PROVENANCE_KEYS = (
    "dataset",
    "tweet_uuid",
    "content_sha256",
    "digest_root_hash",
    "s3_key",
)


def _lease_corpus_provenance(lease: Any) -> dict[str, Any]:
    meta = getattr(lease, "metadata", {}) or {}
    return {key: meta[key] for key in _LEASE_PROVENANCE_KEYS if key in meta}


def _lease_failure_label(exc: BaseException) -> str:
    """Map a corpus lease failure to a distinct, greppable log label."""

    if isinstance(exc, CredentialFetchError):
        return "corpus credential fetch failed"
    if isinstance(exc, DigestFetchError):
        return "corpus digest fetch failed"
    if isinstance(exc, DigestVerificationError):
        return "corpus digest verification failed"
    if isinstance(exc, CorpusObjectError):
        return "corpus object fetch/verification failed"
    return "local lease build failed"


class SolverValidator:
    def __init__(self, *, config: Any | None = None, registry: TaskRegistry | None = None):
        self.config = config or build_config("validator")
        configure_logging(self.config)
        self.log = get_logger("validator")
        self.runtime = ChainRuntime(self.config, role="validator")
        self.registry = registry or default_registry()
        self.client = TaskServerClient(
            self.config.task_server.url,
            timeout=float(self.config.task_server.timeout),
            verify_ssl=bool(self.config.task_server.verify_ssl),
        )
        # Task handlers can opt into self-selected questions from the AWS corpus.
        self._corpus = CorpusReader(
            digest_ttl=float(getattr(self.config.corpus, "digest_ttl", 900.0)),
            expected_root_hash=str(
                getattr(self.config.corpus, "expected_root_hash", "") or ""
            ),
        )
        self._corpus_creds: dict[str, Any] | None = None
        self._corpus_creds_at = 0.0
        handler = self._configured_task_handler()
        if handler.build_lease is not None:
            self.log.info(
                "validator leases from the AWS tweet corpus (self-selected)"
            )
        else:
            self.log.info(
                f"validator leases {handler.kind} tasks from the task server"
            )
        self.transport = HttpTransport(
            self.runtime.wallet,
            max_response_bytes=int(self.config.validator.max_response_bytes),
        )
        self.state_path = Path(self.config.neuron.storage_dir) / "scoreboard.json"
        self.scoreboard = ScoreBoard.load(
            self.state_path,
            size=int(getattr(self.runtime.metagraph, "n", 0)),
            alpha=float(self.config.validator.ema_alpha),
            strategy=str(self.config.validator.score_strategy),
            raw_history_max_events=int(self.config.validator.raw_history_max_events),
        )
        score_log = (
            f"validator scoreboard strategy={self.scoreboard.strategy.name} "
            f"alpha={self.scoreboard.alpha} "
            f"raw_history_max_events={self.scoreboard.raw_history_max_events}"
        )
        if self.scoreboard.strategy.uses_raw_history:
            score_log = (
                f"{score_log} window={self.scoreboard.window_seconds}s "
                f"softmax_beta={self.scoreboard.softmax_beta}"
            )
        self.log.info(score_log)
        self.step = 0
        self.last_weight_block = 0
        self.should_exit = False

    def _task_profile(self) -> dict[str, Any]:
        base = getattr(self.config.task, "profile", {}) or {}
        profile: dict[str, Any] = dict(base)
        if getattr(self.config.task, "kind", None):
            profile["task_kind"] = self.config.task.kind
        if self.config.task.node_count is not None:
            profile["node_count"] = self.config.task.node_count
        if self.config.task.edge_probability is not None:
            profile["edge_probability"] = self.config.task.edge_probability
        if self.config.task.time_limit is not None:
            profile["time_limit"] = self.config.task.time_limit
        return profile

    def _configured_task_handler(self) -> Any:
        kind = str(
            getattr(getattr(self.config, "task", None), "kind", "")
            or self.registry.default_kind
        )
        return self.registry.handler_for(kind)

    async def _lease_from_corpus(self) -> Any:
        credentials = await self._corpus_credentials()
        task_id = str(uuid.uuid4())
        profile = self._task_profile()
        # boto3/S3 reads are blocking; run the build off the event loop.
        return await asyncio.to_thread(
            self._build_corpus_lease, credentials, task_id, profile
        )

    def _build_corpus_lease(
        self,
        credentials: dict[str, Any],
        task_id: str,
        profile: dict[str, Any],
    ) -> Any:
        try:
            self._corpus.update(credentials)
        except ValueError as exc:
            raise CredentialFetchError(f"invalid corpus credentials: {exc}") from exc
        handler = self._configured_task_handler()
        context = LocalLeaseContext(
            task_id=task_id,
            corpus=self._corpus,
            rng=random.Random(),
            profile=profile,
        )
        return handler.build_local_lease(context)

    async def _corpus_credentials(self) -> dict[str, Any]:
        now = time.time()
        ttl = float(getattr(self.config.corpus, "credentials_ttl", 900.0))
        if self._corpus_creds is not None and (now - self._corpus_creds_at) < ttl:
            return self._corpus_creds
        try:
            response = await self.client.corpus_credentials(
                wallet=self.runtime.wallet,
                netuid=int(self.config.netuid),
                uid=int(self.runtime.uid),
                block=int(self.runtime.block),
            )
        except Exception as exc:
            raise CredentialFetchError(
                f"failed to fetch corpus credentials from task server: {exc}"
            ) from exc
        self._corpus_creds = response.model_dump()
        self._corpus_creds_at = now
        return self._corpus_creds

    async def forward_once(self) -> float:
        """Run one validator round and return the seconds to wait before the next round."""
        started = time.perf_counter()
        handler = self._configured_task_handler()
        uses_corpus = handler.build_lease is not None
        try:
            if uses_corpus:
                lease = await self._lease_from_corpus()
            else:
                lease = await self.client.lease(
                    wallet=self.runtime.wallet,
                    netuid=int(self.config.netuid),
                    uid=int(self.runtime.uid),
                    block=int(self.runtime.block),
                    profile=self._task_profile(),
                )
        except Exception as exc:
            label = (
                _lease_failure_label(exc)
                if uses_corpus
                else "task lease failed after task server retries"
            )
            self.log.warning(
                f"{label}; skipping validator round "
                f"error={type(exc).__name__}: {exc}"
            )
            return 60.0
        self.log.info(
            f"VALIDATOR_LEASED_TASK task={lease.task_id} kind={lease.task_kind}"
        )
        synapse = envelope_from_lease(lease)
        selected_uids = self._select_uids()
        if not selected_uids:
            self.log.warning("no miner candidates available")
            return float(self.config.validator.forward_interval)

        selected_uids, axons = await self._resolve_axons(selected_uids)
        if not selected_uids:
            self.log.warning("no usable miner axons available")
            return float(self.config.validator.forward_interval)
        responses = await self.transport.query(axons=axons, synapse=synapse)
        answers = [self._answer_from_response(response) for response in responses]
        miner_hotkeys = self._hotkeys_for_uids(selected_uids)
        handler = self.registry.handler_for(lease.task_kind)
        scores = handler.score_answers(lease.payload, lease.scoring, answers)
        self.scoreboard.resize(int(getattr(self.runtime.metagraph, "n", len(self.scoreboard.ema))))
        self.scoreboard.observe(selected_uids, scores.rewards)
        self.scoreboard.save(self.state_path)
        try:
            await self._upload_scoreboard()
        except Exception as exc:
            self.log.warning(
                "scoreboard upload failed after task server retries "
                f"error={type(exc).__name__}: {exc}"
            )

        try:
            await self.client.report(
                wallet=self.runtime.wallet,
                netuid=int(self.config.netuid),
                uid=int(self.runtime.uid),
                task_id=lease.task_id,
                miner_uids=selected_uids,
                miner_hotkeys=miner_hotkeys,
                miner_answers=answers,
                rewards=scores.rewards,
                metadata={
                    "task_kind": lease.task_kind,
                    "spec_version": lease.spec_version,
                    **_lease_corpus_provenance(lease),
                    **scores.report_metadata(),
                },
            )
        except Exception as exc:
            self.log.warning(
                "task report failed after task server retries "
                f"error={type(exc).__name__}: {exc}"
            )

        elapsed = time.perf_counter() - started
        self.log.info(
            f"VALIDATOR_SCORED_TASK task={lease.task_id} miners={selected_uids} rewards={scores.rewards} "
            f"elapsed={elapsed:.3f}s"
        )
        return float(self.config.validator.forward_interval)

    def _select_uids(self) -> list[int]:
        policy = CandidatePolicy(
            exclude_uid=self.runtime.uid,
            exclude_validator_permit=True,
            exclude_positive_validator_trust=True,
        )
        return miner_candidates(self.runtime.metagraph, policy)

    async def _resolve_axons(self, uids: list[int]) -> tuple[list[int], list[Any]]:
        public_axons = [self.runtime.metagraph.axons[uid] for uid in uids]
        hotkeys = self._hotkeys_for_uids(uids)

        try:
            response = await self.client.miner_axons(
                wallet=self.runtime.wallet,
                netuid=int(self.config.netuid),
                uid=int(self.runtime.uid),
                block=self.runtime.block,
            )
        except Exception as exc:
            self.log.warning(
                "private miner axon lookup failed; falling back to metagraph axons "
                f"error={type(exc).__name__}: {exc}"
            )
            return self._filter_usable_public_axons(uids, public_axons)

        private_by_hotkey = private_axons_by_hotkey(
            metagraph=self.runtime.metagraph,
            netuid=int(self.config.netuid),
            announcements=response.axons,
        )
        resolved_uids: list[int] = []
        resolved_axons: list[Any] = []
        private_used = 0
        public_fallbacks = 0
        for uid, hotkey, public_axon in zip(uids, hotkeys, public_axons):
            private_axon = private_by_hotkey.get(hotkey)
            if private_axon is not None:
                private_used += 1
                resolved_uids.append(uid)
                resolved_axons.append(private_axon)
                continue
            if is_usable_axon(public_axon):
                public_fallbacks += 1
                resolved_uids.append(uid)
                resolved_axons.append(public_axon)

        if private_used or public_fallbacks:
            self.log.info(
                "VALIDATOR_RESOLVED_MINER_AXONS "
                f"private={private_used} "
                f"metagraph_fallback={public_fallbacks}"
            )
        return resolved_uids, resolved_axons

    @staticmethod
    def _filter_usable_public_axons(
        uids: list[int],
        axons: list[Any],
    ) -> tuple[list[int], list[Any]]:
        filtered_uids: list[int] = []
        filtered_axons: list[Any] = []
        for uid, axon in zip(uids, axons):
            if is_usable_axon(axon):
                filtered_uids.append(uid)
                filtered_axons.append(axon)
        return filtered_uids, filtered_axons

    @staticmethod
    def _answer_from_response(response: Any) -> dict[str, Any]:
        if response is None:
            return {}
        if hasattr(response, "deserialize"):
            data = response.deserialize()
            return data if isinstance(data, dict) else {}
        answer = getattr(response, "answer", None)
        return answer if isinstance(answer, dict) else {}

    def _hotkeys_for_uids(self, uids: list[int]) -> list[str]:
        hotkeys = self._all_hotkeys()
        return [
            str(hotkeys[uid]) if 0 <= uid < len(hotkeys) else ""
            for uid in uids
        ]

    def _all_hotkeys(self) -> list[str]:
        return [str(value) for value in getattr(self.runtime.metagraph, "hotkeys", [])]

    async def _upload_scoreboard(self) -> None:
        block = self.runtime.block
        result = await self.client.upload_scoreboard(
            wallet=self.runtime.wallet,
            netuid=int(self.config.netuid),
            block=block,
            uid=int(self.runtime.uid),
            version=SCOREBOARD_VERSION,
            scores=self.scoreboard.scores.astype(float).tolist(),
            miner_hotkeys=self._all_hotkeys(),
            metadata={
                "score_strategy": self.scoreboard.strategy.name,
                "alpha": self.scoreboard.alpha,
                "score_state": self.scoreboard.ema.astype(float).tolist(),
                "counts": self.scoreboard.counts.astype(int).tolist(),
            },
        )
        if not result.get("accepted") or not result.get("updated", True):
            self.log.warning(
                "scoreboard upload returned unexpected response "
                f"block={block} response={result}"
            )

    async def maybe_set_weights(self) -> None:
        if bool(self.config.validator.disable_set_weights):
            return
        if self.step < int(self.config.validator.min_steps_before_set_weights):
            return
        epoch = int(self.config.neuron.epoch_length)
        if self.runtime.block - self.last_weight_block < max(1, epoch // 2):
            return

        try:
            scoreboards = await self.client.latest_scoreboards(
                wallet=self.runtime.wallet,
                netuid=int(self.config.netuid),
                uid=int(self.runtime.uid),
            )
        except Exception as exc:
            self.log.warning(
                "latest scoreboards lookup failed after task server retries "
                f"error={type(exc).__name__}: {exc}"
            )
            return
        current_block = self.runtime.block
        aggregated = aggregate_validator_scores(
            metagraph=self.runtime.metagraph,
            netuid=int(self.config.netuid),
            scoreboards=scoreboards,
            current_block=current_block,
        )
        if aggregated.rejected:
            self.log.warning(
                "ignored signed validator scoreboards: "
                f"{list(aggregated.rejected[:10])}"
            )

        qualified_stake = qualified_validator_stake(self.runtime.metagraph)
        coverage = (
            aggregated.total_stake / qualified_stake if qualified_stake > 0.0 else 0.0
        )
        if coverage <= MIN_AGGREGATED_STAKE_FRACTION:
            self.log.warning(
                "aggregated validator scoreboard stake does not exceed the "
                "required fraction of qualified validator stake; setting weights "
                "from the local scoreboard instead "
                f"accepted={len(aggregated.accepted_hotkeys)} "
                f"aggregated_stake={aggregated.total_stake:.6f} "
                f"qualified_stake={qualified_stake:.6f} "
                f"coverage={coverage:.4f} min_fraction={MIN_AGGREGATED_STAKE_FRACTION:.4f} "
                f"rejected={list(aggregated.rejected[:10])}"
            )
            self._set_weights_from_local_scoreboard()
            return

        submission = self._submit_weights(aggregated.weights)
        if submission is None:
            return
        ok, message = submission
        if ok:
            self.last_weight_block = self.runtime.block
            self.log.info(
                "VALIDATOR_SET_WEIGHTS_SUCCESS weights submitted "
                f"validators={len(aggregated.accepted_hotkeys)} "
                f"stake={aggregated.total_stake:.6f} coverage={coverage:.4f}"
            )
        else:
            self.log.error(f"weight submission failed: {message}")

    def _set_weights_from_local_scoreboard(self) -> None:
        submission = self._submit_weights(self.scoreboard.scores)
        if submission is None:
            return
        ok, message = submission
        if ok:
            self.last_weight_block = self.runtime.block
            self.log.info(
                "VALIDATOR_SET_WEIGHTS_SUCCESS local scoreboard submitted "
                "(insufficient aggregated validator stake coverage)"
            )
        else:
            self.log.error(f"local weight submission failed: {message}")

    def _submit_weights(self, scores: Any) -> tuple[bool, str] | None:
        try:
            return submit_weights_with_deadline(
                timeout=float(self.config.validator.set_weights_timeout),
                subtensor=self.runtime.subtensor,
                wallet=self.runtime.wallet,
                netuid=int(self.config.netuid),
                metagraph=self.runtime.metagraph,
                scores=scores,
                normalize=False,
            )
        except WeightSubmissionTimeout as exc:
            self.log.error(f"VALIDATOR_SET_WEIGHTS_TIMEOUT {exc}")
            self.runtime.reconnect()
            return None

    def serve_validator_axon(self) -> Any | None:
        return None

    def run(self) -> None:
        axon = self.serve_validator_axon()
        try:
            while not self.should_exit:
                try:
                    started = time.perf_counter()
                    delay = asyncio.run(self.forward_once())
                    self.runtime.sync_metagraph()
                    self.scoreboard.resize(int(getattr(self.runtime.metagraph, "n", len(self.scoreboard.ema))))
                    asyncio.run(self.maybe_set_weights())
                    self.step += 1
                    remaining = float(delay) - (time.perf_counter() - started)
                    if remaining > 0:
                        time.sleep(remaining)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    self.log.warning(
                        f"validator step failed, continuing: {type(exc).__name__}: {exc}"
                    )
                    time.sleep(2)
        except KeyboardInterrupt:
            self.log.info("validator interrupted")
        finally:
            self.scoreboard.save(self.state_path)
            if axon is not None:
                axon.stop()


def main() -> None:
    SolverValidator().run()


if __name__ == "__main__":
    main()
