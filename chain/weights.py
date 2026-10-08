"""Weight preparation and submission."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import __version__, semver_to_int_version
from .._bt import require_bittensor
from .metagraph import (
    DEFAULT_SCOREBOARD_MAX_AGE_BLOCKS,
    block_within_window,
    non_owner_has_incentive,
)
from ..tasks.framework.models import SignedScoreboardSnapshot
from ..tasks.framework.signing import verify_model_signature

# Sent with every weight submission. Every release from 1.1 encodes above the
# 100100 that all earlier releases sent, and keys only grow, so the subnet's
# WeightsVersionKey hyperparameter can reject validators that have not upgraded.
VERSION_KEY = semver_to_int_version(__version__)

# Share of each weight row routed to BURN_UID before miners split the rest.
# 0.0 turns the burn off: submit_weights then leaves the row untouched.
BURN_UID = 0
BURN_WEIGHT_SHARE = 0.0
# A healthy finney submission returns in ~30s; anything past this is a wedged
# websocket, not a slow chain.
DEFAULT_SET_WEIGHTS_TIMEOUT = 180.0


@dataclass(frozen=True)
class AggregatedValidatorWeights:
    weights: np.ndarray
    accepted_hotkeys: tuple[str, ...]
    rejected: tuple[str, ...]
    total_stake: float


def normalized_weights(scores: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0)
    clean = np.maximum(clean, 0.0)
    if not np.any(clean > 0):
        return np.zeros_like(clean)
    shifted = clean - np.max(clean)
    exp_scores = np.exp(shifted)
    exp_scores[clean <= 0] = 0.0
    total = float(np.sum(exp_scores))
    if total <= 0:
        return np.zeros_like(clean)
    return exp_scores / total


def aggregate_validator_scores(
    *,
    metagraph: Any,
    netuid: int,
    scoreboards: list[SignedScoreboardSnapshot | dict[str, Any]],
    current_block: int | None = None,
    max_age_blocks: int = DEFAULT_SCOREBOARD_MAX_AGE_BLOCKS,
) -> AggregatedValidatorWeights:
    hotkeys = [str(value) for value in getattr(metagraph, "hotkeys", [])]
    size = int(getattr(metagraph, "n", len(hotkeys)))
    if size <= 0:
        size = len(hotkeys)
    uid_by_hotkey = {hotkey: uid for uid, hotkey in enumerate(hotkeys)}
    stakes = _stake_values(metagraph, size)
    weighted_weights = np.zeros(size, dtype=np.float64)
    accepted: list[str] = []
    rejected: list[str] = []
    total_stake = 0.0

    for raw in scoreboards:
        try:
            signed = SignedScoreboardSnapshot.model_validate(raw)
        except Exception:
            rejected.append("<invalid>:malformed")
            continue
        payload = signed.payload
        signer = str(payload.hotkey)
        reason_prefix = signer or "<missing-hotkey>"
        if not signer:
            rejected.append(f"{reason_prefix}:missing-hotkey")
            continue
        if int(payload.netuid) != int(netuid):
            rejected.append(f"{reason_prefix}:wrong-netuid")
            continue
        if current_block is not None and not block_within_window(
            current_block=int(current_block),
            signed_block=int(payload.block),
            max_age_blocks=max_age_blocks,
        ):
            rejected.append(f"{reason_prefix}:stale-block")
            continue
        if not verify_model_signature(signer, signed.signature, payload):
            rejected.append(f"{reason_prefix}:invalid-signature")
            continue
        signer_uid = uid_by_hotkey.get(signer)
        if signer_uid is None:
            rejected.append(f"{reason_prefix}:not-in-metagraph")
            continue
        if int(getattr(payload, "uid", -1)) != signer_uid:
            rejected.append(f"{reason_prefix}:uid-hotkey-mismatch")
            continue
        if non_owner_has_incentive(metagraph, signer_uid):
            rejected.append(f"{reason_prefix}:has-incentive")
            continue
        stake = stakes[signer_uid] if signer_uid < len(stakes) else 0.0
        if stake <= 0.0:
            rejected.append(f"{reason_prefix}:zero-stake")
            continue

        aligned_scores = _align_scores_to_metagraph(
            scores=payload.scores,
            miner_hotkeys=payload.miner_hotkeys,
            size=size,
            uid_by_hotkey=uid_by_hotkey,
        )
        weighted_weights += stake * _nonnegative_distribution(aligned_scores)
        total_stake += stake
        accepted.append(signer)

    if total_stake <= 0.0:
        return AggregatedValidatorWeights(
            weights=np.zeros(size, dtype=np.float64),
            accepted_hotkeys=tuple(accepted),
            rejected=tuple(rejected),
            total_stake=0.0,
        )
    return AggregatedValidatorWeights(
        weights=_nonnegative_distribution(weighted_weights / total_stake),
        accepted_hotkeys=tuple(accepted),
        rejected=tuple(rejected),
        total_stake=total_stake,
    )


def qualified_validator_stake(metagraph: Any) -> float:
    """Total stake of neurons that qualify as validators: hold a validator
    permit, have positive stake, and carry no miner incentive (owner exempt)."""
    hotkeys = [str(value) for value in getattr(metagraph, "hotkeys", [])]
    size = int(getattr(metagraph, "n", len(hotkeys)))
    if size <= 0:
        size = len(hotkeys)
    stakes = _stake_values(metagraph, size)
    validator_permit = getattr(metagraph, "validator_permit", None)
    total = 0.0
    for uid in range(size):
        if not _has_validator_permit(validator_permit, uid):
            continue
        stake = float(stakes[uid]) if uid < len(stakes) else 0.0
        if stake <= 0.0:
            continue
        if non_owner_has_incentive(metagraph, uid):
            continue
        total += stake
    return total


def _has_validator_permit(validator_permit: Any, uid: int) -> bool:
    if validator_permit is None:
        return False
    try:
        return bool(validator_permit[uid])
    except (IndexError, KeyError, TypeError):
        return False


def submit_weights(
    *,
    subtensor: Any,
    wallet: Any,
    netuid: int,
    metagraph: Any,
    scores: np.ndarray,
    version_key: int = VERSION_KEY,
    normalize: bool = True,
) -> tuple[bool, str]:
    uids = np.asarray(getattr(metagraph, "uids", np.arange(len(scores))), dtype=np.int64)
    weights = normalized_weights(scores) if normalize else _nonnegative_distribution(scores)
    weights = _with_burn_weight_share(weights, burn_uid=BURN_UID, share=BURN_WEIGHT_SHARE)
    uid_list = [int(value) for value in np.asarray(uids).ravel().tolist()]
    weight_list = [float(value) for value in np.asarray(weights).ravel().tolist()]

    return _submit_weights_intent(
        subtensor=subtensor,
        wallet=wallet,
        netuid=int(netuid),
        uids=uid_list,
        weights=weight_list,
        version_key=int(version_key),
    )


class WeightSubmissionTimeout(TimeoutError):
    """Raised when a weight submission outlives its deadline."""


def submit_weights_with_deadline(
    *,
    timeout: float = DEFAULT_SET_WEIGHTS_TIMEOUT,
    **kwargs: Any,
) -> tuple[bool, str]:
    """``submit_weights``, but it always returns.

    ``subtensor.execute`` waits on a finalization subscription with no deadline
    of its own (``bittensor.sync._Loop.call`` accepts a timeout, but nothing on
    the public path ever passes one), so a half-open connection to the chain
    endpoint blocks the caller forever rather than raising. Run it on a daemon
    thread instead and abandon the thread if it overruns.

    A ``timeout`` of zero or less runs the submission inline, undeadlined.

    The abandoned thread cannot be killed and stays parked on the dead socket,
    so callers must rebuild the subtensor connection before submitting again —
    see ``ChainRuntime.reconnect``, whose ``close()`` releases that thread.
    """
    if timeout <= 0:
        return submit_weights(**kwargs)

    outcome: list[tuple[bool, str]] = []
    failure: list[BaseException] = []

    def _run() -> None:
        try:
            outcome.append(submit_weights(**kwargs))
        except BaseException as exc:  # re-raised on the calling thread below
            failure.append(exc)

    worker = threading.Thread(target=_run, name="tag101-submit-weights", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise WeightSubmissionTimeout(
            f"weight submission did not return within {timeout:.0f}s"
        )
    if failure:
        raise failure[0]
    return outcome[0]


def _commit_reveal_enabled(subtensor: Any, netuid: int) -> bool:
    subnets = getattr(subtensor, "subnets", None)
    checker = getattr(subnets, "commit_reveal_enabled", None) if subnets is not None else None
    if not callable(checker):
        return False
    try:
        return bool(checker(netuid=netuid))
    except Exception:
        return False


def _submit_weights_intent(
    *,
    subtensor: Any,
    wallet: Any,
    netuid: int,
    uids: list[int],
    weights: list[float],
    version_key: int,
) -> tuple[bool, str]:
    bt = require_bittensor()
    if _commit_reveal_enabled(subtensor, netuid):
        intent = bt.CommitWeights(netuid=netuid, uids=uids, weights=weights, version_key=version_key)
    else:
        intent = bt.SetWeights(netuid=netuid, uids=uids, weights=weights, version_key=version_key)

    try:
        result = subtensor.execute(intent, wallet)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    ok = getattr(result, "success", getattr(result, "is_success", True))
    message = getattr(result, "message", None)
    if message is None:
        message = getattr(result, "error", "") or str(result)
    return bool(ok), str(message)



def _nonnegative_distribution(values: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    clean = np.maximum(clean, 0.0)
    total = float(np.sum(clean))
    if total <= 0.0:
        return np.zeros_like(clean)
    return clean / total


def _with_burn_weight_share(
    values: np.ndarray,
    *,
    burn_uid: int,
    share: float,
) -> np.ndarray:
    weights = _nonnegative_distribution(values)
    if weights.size == 0 or burn_uid < 0 or burn_uid >= weights.size:
        return weights

    burn_share = min(max(float(share), 0.0), 1.0)
    if burn_share <= 0.0:
        # Burn off: keep the row as scored, burn uid's own slot included.
        return weights
    remaining_share = 1.0 - burn_share
    adjusted = np.zeros_like(weights, dtype=np.float64)
    adjusted[burn_uid] = burn_share

    non_burn = weights.copy()
    non_burn[burn_uid] = 0.0
    non_burn = _nonnegative_distribution(non_burn)
    if remaining_share > 0.0 and not np.any(non_burn > 0):
        non_burn = np.ones_like(weights, dtype=np.float64)
        non_burn[burn_uid] = 0.0
        non_burn = _nonnegative_distribution(non_burn)
    adjusted += remaining_share * non_burn
    return _nonnegative_distribution(adjusted)


def _align_scores_to_metagraph(
    *,
    scores: list[float],
    miner_hotkeys: list[str],
    size: int,
    uid_by_hotkey: dict[str, int],
) -> np.ndarray:
    aligned = np.zeros(size, dtype=np.float64)
    clean_scores = np.nan_to_num(
        np.asarray(scores, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    for index, score in enumerate(clean_scores):
        uid = None
        if index < len(miner_hotkeys):
            miner_hotkey = str(miner_hotkeys[index] or "")
            uid = uid_by_hotkey.get(miner_hotkey)
        elif index < size:
            uid = index
        if uid is not None and 0 <= uid < size:
            aligned[uid] = max(float(score), 0.0)
    return aligned


def _stake_values(metagraph: Any, size: int) -> np.ndarray:
    raw_stakes = getattr(metagraph, "total_stake", getattr(metagraph, "S", []))
    stakes = np.zeros(size, dtype=np.float64)
    for index, value in enumerate(list(raw_stakes)[:size]):
        stakes[index] = _stake_as_float(value)
    return stakes


def _stake_as_float(value: Any) -> float:
    tao = getattr(value, "tao", None)
    if tao is not None:
        return float(tao)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
