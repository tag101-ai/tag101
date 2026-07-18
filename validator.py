"""Bittensor validator entry point."""

from __future__ import annotations

import asyncio
import time
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
from .chain.weights import aggregate_validator_scores, submit_weights
from .protocol import envelope_from_lease
from .tasks import TaskRegistry, TaskServerClient, default_registry


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

    async def forward_once(self) -> float:
        """Run one validator round and return the seconds to wait before the next round."""
        started = time.perf_counter()
        try:
            lease = await self.client.lease(
                wallet=self.runtime.wallet,
                netuid=int(self.config.netuid),
                uid=int(self.runtime.uid),
                profile=self._task_profile(),
            )
        except Exception as exc:
            self.log.warning(
                "task lease failed after task server retries; skipping validator round "
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
        if not aggregated.accepted_hotkeys:
            self.log.error(
                "no valid signed validator scoreboards available for set_weights "
                f"scoreboards={len(scoreboards)} current_block={current_block} "
                f"rejected={list(aggregated.rejected[:10])}"
            )
            return
        if aggregated.rejected:
            self.log.warning(
                "ignored signed validator scoreboards: "
                f"{list(aggregated.rejected[:10])}"
            )

        ok, message = submit_weights(
            subtensor=self.runtime.subtensor,
            wallet=self.runtime.wallet,
            netuid=int(self.config.netuid),
            metagraph=self.runtime.metagraph,
            scores=aggregated.weights,
            normalize=False,
        )
        if ok:
            self.last_weight_block = self.runtime.block
            self.log.info(
                "VALIDATOR_SET_WEIGHTS_SUCCESS weights submitted "
                f"validators={len(aggregated.accepted_hotkeys)} stake={aggregated.total_stake:.6f}"
            )
        else:
            self.log.error(f"weight submission failed: {message}")

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
