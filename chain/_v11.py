"""bittensor 11 compatibility adapters for chain reads and mutations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._bt import require_bittensor

_KNOWN_NETWORKS = frozenset({"archive", "devnet", "finney", "local", "test"})


def normalize_network_endpoint(network: str | None) -> str | None:
    """Coerce a configured network into a form bittensor 11 accepts."""
    if network is None:
        return None
    value = str(network).strip()
    if not value or value in _KNOWN_NETWORKS or "://" in value:
        return value
    return f"wss://{value}"


def build_subtensor(bt: Any, network: str | None) -> Any:
    normalized = normalize_network_endpoint(network)
    if normalized:
        return bt.Subtensor(network=normalized)
    return bt.Subtensor()


def current_block(subtensor: Any) -> int:
    return int(subtensor.block())


def read_metagraph(subtensor: Any, netuid: int, block: int | None = None) -> Any:
    """Fetch the raw per-neuron metagraph for a subnet."""
    return subtensor.subnets.metagraph(netuid=int(netuid), block=block)


def is_hotkey_registered(subtensor: Any, *, netuid: int, hotkey_ss58: str) -> bool:
    """Whether a hotkey holds a uid on the subnet."""
    snapshot = snapshot_from_metagraph(read_metagraph(subtensor, netuid))
    return hotkey_ss58 in snapshot.hotkeys


def balance_to_float(value: Any) -> float:
    """Coerce a bittensor Balance (or plain number) to float."""
    for attr in ("alpha", "tao"):
        try:
            candidate = getattr(value, attr)
        except Exception:
            continue
        if candidate is None:
            continue
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    try:
        rao = getattr(value, "rao")
    except Exception:
        rao = None
    if rao is not None:
        try:
            return float(rao) / 1e9
        except (TypeError, ValueError):
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class AxonEndpoint:
    """Minimal axon endpoint (bittensor 11 exposes only ip:port)."""

    ip: str = ""
    port: int = 0
    ip_type: int = 4
    protocol: int = 4
    version: int = 0
    hotkey: str = ""

    @property
    def is_serving(self) -> bool:
        return bool(self.ip) and self.ip not in ("0.0.0.0",) and int(self.port) > 0


def parse_axon(value: Any) -> AxonEndpoint:
    """Build an :class:`AxonEndpoint` from a bittensor 11 ``"ip:port"`` string (or ``None``)."""
    if not value:
        return AxonEndpoint()
    text = str(value).strip()
    if not text:
        return AxonEndpoint()
    host, sep, port = text.rpartition(":")
    if not sep:
        return AxonEndpoint(ip=text, port=0)
    try:
        return AxonEndpoint(ip=host, port=int(port))
    except ValueError:
        return AxonEndpoint(ip=text, port=0)


@dataclass
class MetagraphSnapshot:
    """The aggregate metagraph view the validator/miner code reads, built from bittensor 11 neurons."""

    block: int
    hotkeys: list[str]
    coldkeys: list[str]
    uids: list[int]
    total_stake: list[float]
    incentive: list[float]
    validator_permit: list[bool]
    axons: list[AxonEndpoint]

    @property
    def S(self) -> list[float]:  # noqa: N802 - mirrors bittensor's field name
        return self.total_stake

    @property
    def I(self) -> list[float]:  # noqa: E743,N802 - mirrors bittensor's field name
        return self.incentive

    @property
    def n(self) -> int:
        return len(self.hotkeys)

    def by_hotkey(self, hotkey: str) -> int | None:
        try:
            return self.hotkeys.index(str(hotkey))
        except ValueError:
            return None


def snapshot_from_metagraph(raw: Any, *, block: int | None = None) -> MetagraphSnapshot:
    """Build a :class:`MetagraphSnapshot` from a bittensor 11 per-neuron metagraph."""
    if isinstance(raw, MetagraphSnapshot):
        return raw

    blk = int(block if block is not None else getattr(raw, "block", 0) or 0)
    neurons = list(getattr(raw, "neurons", []) or [])
    hotkeys = [str(value) for value in getattr(raw, "hotkeys", []) or []]
    size = max(len(hotkeys), len(neurons))
    hotkeys = (hotkeys + [""] * size)[:size]
    coldkeys = [""] * size
    total_stake = [0.0] * size
    incentive = [0.0] * size
    validator_permit = [False] * size
    axons = [AxonEndpoint() for _ in range(size)]
    for neuron in neurons:
        try:
            uid = int(getattr(neuron, "uid"))
        except (TypeError, ValueError, AttributeError):
            continue
        if not (0 <= uid < size):
            continue
        hotkey = getattr(neuron, "hotkey", None)
        if hotkey is not None:
            hotkeys[uid] = str(hotkey)
        coldkey = getattr(neuron, "coldkey", None)
        if coldkey is not None:
            coldkeys[uid] = str(coldkey)
        total_stake[uid] = balance_to_float(getattr(neuron, "total_stake", 0.0))
        incentive[uid] = balance_to_float(getattr(neuron, "incentive", 0.0))
        validator_permit[uid] = bool(getattr(neuron, "validator_permit", False))
        axon = parse_axon(getattr(neuron, "axon", None))
        axon.hotkey = hotkeys[uid]
        axons[uid] = axon
    return MetagraphSnapshot(
        block=blk,
        hotkeys=hotkeys,
        coldkeys=coldkeys,
        uids=list(range(size)),
        total_stake=total_stake,
        incentive=incentive,
        validator_permit=validator_permit,
        axons=axons,
    )


def fetch_metagraph_snapshot(
    subtensor: Any, netuid: int, block: int | None = None
) -> MetagraphSnapshot:
    raw = read_metagraph(subtensor, netuid, block)
    return snapshot_from_metagraph(raw, block=block)


# bittensor 11 mutation / provisioning helpers


def _intent_result(result: Any) -> tuple[bool, str]:
    ok = getattr(result, "success", getattr(result, "is_success", True))
    message = getattr(result, "message", None)
    if message is None:
        message = getattr(result, "error", None) or getattr(result, "error_message", None)
    if message is None:
        message = str(result)
    return bool(ok), str(message)


def execute_intent(subtensor: Any, intent: Any, wallet: Any) -> tuple[bool, str]:
    return _intent_result(subtensor.execute(intent, wallet))


def subnet_exists(subtensor: Any, netuid: int) -> bool:
    try:
        return int(netuid) in {int(s.netuid) for s in subtensor.subnets.all()}
    except Exception:
        try:
            fetch_metagraph_snapshot(subtensor, netuid)
            return True
        except Exception:
            return False


def get_balance(subtensor: Any, ss58_address: str) -> Any:
    return subtensor.balances.get(ss58_address)


def import_wallet_from_uris(
    wallet: Any,
    *,
    coldkey_uri: str,
    hotkey_uri: str,
    overwrite: bool = False,
) -> None:
    """Persist deterministic development keys with the bittensor 11 wallet API.

    Bittensor 11 removed the public URI import helpers and public keypair
    setters. These are the same setters its create/regenerate methods use.
    """
    from bittensor.sp_core import Keypair

    coldkey = Keypair.create_from_uri(str(coldkey_uri))
    hotkey = Keypair.create_from_uri(str(hotkey_uri))
    wallet._set_coldkey(
        coldkey,
        encrypt=False,
        overwrite=bool(overwrite),
        save_coldkey_to_env=False,
        coldkey_password=None,
    )
    wallet._set_coldkeypub(coldkey, overwrite=bool(overwrite))
    wallet._set_hotkey(
        hotkey,
        encrypt=False,
        overwrite=bool(overwrite),
        save_hotkey_to_env=False,
        hotkey_password=None,
    )
    wallet._set_hotkeypub(hotkey, overwrite=bool(overwrite))


def transfer(subtensor: Any, wallet: Any, dest_ss58: str, amount_tao: float) -> tuple[bool, str]:
    bt = require_bittensor()
    intent = bt.Transfer(dest_ss58=str(dest_ss58), amount_tao=float(amount_tao))
    return execute_intent(subtensor, intent, wallet)


def add_stake(
    subtensor: Any, wallet: Any, hotkey_ss58: str, netuid: int, amount_tao: float
) -> tuple[bool, str]:
    bt = require_bittensor()
    intent = bt.AddStake(
        hotkey_ss58=str(hotkey_ss58), netuid=int(netuid), amount_tao=float(amount_tao)
    )
    return execute_intent(subtensor, intent, wallet)


def register_subnet(subtensor: Any, wallet: Any, hotkey_ss58: str | None = None) -> tuple[bool, str]:
    bt = require_bittensor()
    return execute_intent(subtensor, bt.RegisterSubnet(hotkey_ss58=hotkey_ss58), wallet)


def burned_register(
    subtensor: Any, wallet: Any, netuid: int, hotkey_ss58: str | None = None
) -> tuple[bool, str]:
    bt = require_bittensor()
    intent = bt.BurnedRegister(netuid=int(netuid), hotkey_ss58=hotkey_ss58)
    return execute_intent(subtensor, intent, wallet)


def start_call(subtensor: Any, wallet: Any, netuid: int) -> tuple[bool, str]:
    bt = require_bittensor()
    return execute_intent(subtensor, bt.StartCall(netuid=int(netuid)), wallet)


def serve_axon(
    subtensor: Any,
    wallet: Any,
    *,
    netuid: int,
    ip: str,
    port: int,
) -> tuple[bool, str]:
    bt = require_bittensor()
    intent = bt.ServeAxon(netuid=int(netuid), ip=str(ip), port=int(port))
    return execute_intent(subtensor, intent, wallet)


def set_hyperparameter(
    subtensor: Any, wallet: Any, netuid: int, name: str, value: Any
) -> tuple[bool, str]:
    """Set a subnet hyperparameter via the SetHyperparameter intent."""
    bt = require_bittensor()
    try:
        intent = bt.SetHyperparameter(netuid=int(netuid), name=str(name), value=value)
    except Exception as exc:
        return False, f"invalid hyperparameter {name}={value!r}: {exc}"
    return execute_intent(subtensor, intent, wallet)
