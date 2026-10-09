"""Construction and synchronization of Bittensor runtime objects."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .._bt import bittensor_attr, require_bittensor
from . import _v11

_log = logging.getLogger("tag101.chain.runtime")


@dataclass
class ChainRuntime:
    config: Any
    role: str
    wallet: Any = field(init=False)
    subtensor: Any = field(init=False)
    metagraph: Any = field(init=False)
    uid: int | None = field(default=None, init=False)
    _cached_block: int | None = field(default=None, init=False)
    _cached_at: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        bt = require_bittensor()
        wallet_cls = bittensor_attr("wallet", "Wallet")
        self.wallet = _build_wallet(wallet_cls, self.config)
        network_arg = _subtensor_network_arg(self.config)
        self.subtensor = _v11.build_subtensor(bt, network_arg)
        self.metagraph = _v11.fetch_metagraph_snapshot(self.subtensor, self.config.netuid)
        if not getattr(self.config.neuron, "no_registration_check", False):
            self.ensure_registered()
        self.uid = self.find_uid()
        _log.info(
            f"{self.role} ready: netuid={self.config.netuid} uid={self.uid} "
            f"endpoint={network_arg}"
        )

    @property
    def hotkey(self) -> str:
        return self.wallet.hotkey.ss58_address

    @property
    def block(self) -> int:
        ttl = float(getattr(self.config.neuron, "block_cache_ttl", 0.0))
        now = time.time()
        if ttl <= 0 or self._cached_block is None or now - self._cached_at >= ttl:
            self._cached_block = _v11.current_block(self.subtensor)
            self._cached_at = now
        return self._cached_block

    def ensure_registered(self) -> None:
        registered = _v11.is_hotkey_registered(
            self.subtensor,
            netuid=self.config.netuid,
            hotkey_ss58=self.hotkey,
        )
        if not registered:
            raise RuntimeError(
                f"Hotkey {self.hotkey} is not registered on netuid {self.config.netuid}."
            )

    def find_uid(self) -> int | None:
        hotkeys = list(getattr(self.metagraph, "hotkeys", []))
        if self.hotkey not in hotkeys:
            return None
        return hotkeys.index(self.hotkey)

    def sync_metagraph(self) -> None:
        self.metagraph = _v11.fetch_metagraph_snapshot(self.subtensor, self.config.netuid)
        self.uid = self.find_uid()

    def reconnect(self) -> None:
        """Rebuild the subtensor client after the chain connection wedges.

        A timed-out call leaves its worker parked on the old socket, so the old
        client is closed (bounded teardown: a 5s loop call plus a 5s thread
        join) to release it, and a cold replacement is built in its place.
        """
        bt = require_bittensor()
        network_arg = _subtensor_network_arg(self.config)
        stale = self.subtensor
        self.subtensor = _v11.build_subtensor(bt, network_arg)
        self._cached_block = None
        self._cached_at = 0.0
        try:
            stale.close()
        except Exception as exc:
            _log.warning(
                f"discarded subtensor did not close cleanly: {type(exc).__name__}: {exc}"
            )
        _log.warning(f"{self.role} rebuilt subtensor connection endpoint={network_arg}")

    def publish_axon(self, ip: str, port: int) -> Any:
        """Publish this neuron's HTTP endpoint on-chain."""
        bt = require_bittensor()
        intent = bt.ServeAxon(netuid=int(self.config.netuid), ip=str(ip), port=int(port))
        result = self.subtensor.execute(intent, self.wallet)
        if getattr(result, "success", True) is False:
            raise RuntimeError(
                f"failed to serve axon: {getattr(result, 'message', result)}"
            )
        return result


def _subtensor_network_arg(config: Any) -> str | None:
    if hasattr(config, "is_set") and config.is_set("subtensor.chain_endpoint"):
        return config.subtensor.chain_endpoint
    if hasattr(config, "is_set") and config.is_set("subtensor.network"):
        return config.subtensor.network
    subtensor = getattr(config, "subtensor", None)
    endpoint = getattr(subtensor, "chain_endpoint", None)
    if endpoint:
        return endpoint
    return getattr(subtensor, "network", None)


def _build_wallet(wallet_cls: Any, config: Any) -> Any:
    """Construct a wallet from a hotkey seed, or from the configured name/hotkey/path."""
    wallet_config = getattr(config, "wallet", None)
    kwargs = {
        "name": getattr(wallet_config, "name", None),
        "hotkey": getattr(wallet_config, "hotkey", None),
        "path": getattr(wallet_config, "path", None),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    secret = str(getattr(wallet_config, "hotkey_seed", None) or "").strip()
    if not secret:
        return wallet_cls(**kwargs)

    # The name/hotkey/path still identify the wallet in logs and storage paths;
    # with a seed configured, nothing is ever read from that path.
    wallet = wallet_cls(**kwargs)
    _v11.load_wallet_from_secret(
        wallet,
        hotkey_secret=secret,
        coldkey_ss58=getattr(wallet_config, "coldkey_ss58", None),
    )
    _log.info(
        f"wallet derived from secret: hotkey={kwargs.get('hotkey')} "
        f"ss58={wallet.hotkey.ss58_address} (keys stay in memory)"
    )
    return wallet
