"""Bittensor miner entry point."""

import asyncio
import time
from typing import Any, Tuple

from ._logging import configure_logging, get_logger
from .chain.miner_server import MinerServer
from .chain.runtime import ChainRuntime
from .chain.settings import build_config
from .protocol import TaskEnvelope
from .tasks import TaskRegistry, TaskServerClient, build_task_registry


class SolverMiner:
    def __init__(self, *, config: Any | None = None, registry: TaskRegistry | None = None):
        self.config = config or build_config("miner")
        configure_logging(self.config)
        self.log = get_logger("miner")
        self.runtime = ChainRuntime(self.config, role="miner")
        self.registry = registry or self._task_registry()
        self.client = self._task_server_client()
        host, port = self._axon_bind()
        self.server = MinerServer(
            wallet=self.runtime.wallet,
            host=host,
            port=port,
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            max_request_bytes=int(
                getattr(getattr(self.config, "axon", None), "max_request_bytes", None) or 128 * 1024
            ),
        )
        self.should_exit = False

    async def forward(self, synapse: TaskEnvelope) -> TaskEnvelope:
        started = time.perf_counter()
        try:
            handler = self.registry.handler_for(synapse.task_kind)
            synapse.answer = handler.solve_problem(synapse, self.runtime)
        except Exception:
            synapse.answer = {}
        elapsed = time.perf_counter() - started
        self.log.info(
            f"MINER_SOLVED_TASK task={synapse.task_id} kind={synapse.task_kind} "
            f"elapsed={elapsed:.3f}s answer_keys={list(synapse.answer)}"
        )
        return synapse

    async def blacklist(self, synapse: TaskEnvelope, caller_hotkey: str) -> Tuple[bool, str]:
        hotkey = str(caller_hotkey or "")
        if not hotkey:
            allow_empty = bool(getattr(self.config.miner, "allow_empty_hotkey", False))
            return (not allow_empty), "missing caller hotkey"

        hotkeys = list(getattr(self.runtime.metagraph, "hotkeys", []))
        if hotkey not in hotkeys:
            if bool(getattr(self.config.blacklist, "allow_non_registered", False)):
                return False, "unregistered caller allowed by config"
            return True, "caller is not registered"

        uid = hotkeys.index(hotkey)
        if bool(getattr(self.config.blacklist, "force_validator_permit", False)):
            permits = getattr(self.runtime.metagraph, "validator_permit", [])
            if uid >= len(permits) or not bool(permits[uid]):
                return True, "caller has no validator permit"
        return False, "accepted"

    def run(self) -> None:
        self.runtime.ensure_registered()
        self.server.start()
        external_ip, external_port = self._axon_external()
        if external_ip and external_port:
            try:
                self.runtime.publish_axon(external_ip, external_port)
            except Exception as exc:
                self.log.warning(
                    f"on-chain serve-axon failed: {type(exc).__name__}: {exc}"
                )
        self.log.info(f"miner serving at block {self.runtime.block}")
        self._announce_private_axon_at_startup()
        try:
            while not self.should_exit:
                self.runtime.sync_metagraph()
                time.sleep(12)
        except KeyboardInterrupt:
            self.log.info("miner interrupted")
        finally:
            self.server.stop()

    def _axon_bind(self) -> tuple[str, int]:
        axon = getattr(self.config, "axon", None)
        host = str(getattr(axon, "ip", "0.0.0.0") or "0.0.0.0")
        port = int(getattr(axon, "port", 8091) or 8091)
        return host, port

    def _axon_external(self) -> tuple[str | None, int | None]:
        axon = getattr(self.config, "axon", None)
        ip = getattr(axon, "external_ip", None) or getattr(axon, "ip", None)
        port = getattr(axon, "external_port", None) or getattr(axon, "port", None)
        ip = str(ip) if ip else None
        if ip in ("0.0.0.0", ""):
            ip = None
        return ip, (int(port) if port else None)

    def _task_server_client(self) -> TaskServerClient | None:
        url = str(getattr(self.config.task_server, "url", "") or "").strip()
        if not url or bool(getattr(self.config.miner, "axon_to_public_metagraph", False)):
            return None
        return TaskServerClient(
            url,
            timeout=float(getattr(self.config.task_server, "timeout", 30.0)),
            verify_ssl=bool(getattr(self.config.task_server, "verify_ssl", True)),
        )

    def _task_registry(self) -> TaskRegistry:
        miner_module = str(getattr(getattr(self.config, "task", None), "miner_module", "") or "")
        override_modules = [
            module.strip()
            for module in miner_module.split(",")
            if module.strip()
        ]
        return build_task_registry(override_modules=override_modules)

    def _announce_private_axon_at_startup(self) -> None:
        if self.client is None:
            return

        try:
            result = asyncio.run(self._announce_private_axon_once())
        except Exception as exc:
            self.log.warning(
                "private miner axon startup announcement failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        if not result.get("accepted") or not result.get("updated", True):
            self.log.warning(
                "private miner axon announcement returned unexpected response "
                f"response={result}"
            )
        else:
            self.log.info(
                "MINER_PRIVATE_AXON_ANNOUNCED "
                f"netuid={int(self.config.netuid)} block={self.runtime.block}"
            )

    async def _announce_private_axon_once(self) -> dict[str, Any]:
        if self.client is None:
            return {"accepted": False, "updated": False}
        external_ip, external_port = self._axon_external()
        if not external_ip or not external_port:
            raise ValueError("private axon announcement requires external ip and port")
        axon = getattr(self.config, "axon", None)
        return await self.client.announce_miner_axon(
            wallet=self.runtime.wallet,
            netuid=int(self.config.netuid),
            uid=int(self.runtime.uid),
            block=self.runtime.block,
            ip=str(external_ip),
            port=int(external_port),
            ip_type=None,
            protocol=int(getattr(axon, "protocol", 4) or 4),
            version=int(getattr(axon, "version", 0) or 0),
        )

    def __enter__(self) -> "SolverMiner":
        self._task = asyncio.get_event_loop().run_in_executor(None, self.run)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.should_exit = True


def main() -> None:
    SolverMiner().run()


if __name__ == "__main__":
    main()
