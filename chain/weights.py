"""Weight preparation and submission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._bt import require_bittensor
from .metagraph import (
    DEFAULT_SCOREBOARD_MAX_AGE_BLOCKS,
    block_within_window,
    non_owner_has_incentive,
)
from ..tasks.framework.models import SignedScoreboardSnapshot
from ..tasks.framework.signing import verify_model_signature

VERSION_KEY = 100100


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
