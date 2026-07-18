"""Command-line configuration for miner and validator processes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from munch import DefaultMunch

from .scoreboard import DEFAULT_RAW_HISTORY_MAX_EVENTS


SN101_TASK_INTERVAL_SECONDS = 15 * 60
SN101_ONE_DAY_EMA_ALPHA = 2.0 / ((24 * 60 // 15) + 1)
DEFAULT_TASK_SERVER_URL = "https://crawler.tag101.ai"


def _add_bittensor_args(parser: argparse.ArgumentParser) -> None:
    wallet = parser.add_argument_group("wallet")
    wallet.add_argument("--wallet.name", dest="wallet.name",
                        default=os.getenv("BT_WALLET_NAME", "default"))
    wallet.add_argument("--wallet.hotkey", dest="wallet.hotkey",
                        default=os.getenv("BT_WALLET_HOTKEY", "default"))
    wallet.add_argument("--wallet.path", dest="wallet.path",
                        default=os.getenv("BT_WALLET_PATH", "~/.bittensor/wallets/"))

    subtensor = parser.add_argument_group("subtensor")
    subtensor.add_argument("--subtensor.network", dest="subtensor.network",
                           default=os.getenv("BT_NETWORK", os.getenv("SUBTENSOR_NETWORK", "finney")))
    subtensor.add_argument("--subtensor.chain_endpoint", dest="subtensor.chain_endpoint",
                           default=os.getenv("BT_CHAIN_ENDPOINT", None))

    axon = parser.add_argument_group("axon")
    axon.add_argument("--axon.port", dest="axon.port", type=int,
                      default=int(os.getenv("AXON_PORT", "8091")))
    axon.add_argument("--axon.ip", dest="axon.ip", default=os.getenv("AXON_IP", "0.0.0.0"))
    axon.add_argument("--axon.external_ip", dest="axon.external_ip",
                      default=os.getenv("AXON_EXTERNAL_IP", None))
    axon.add_argument("--axon.external_port", dest="axon.external_port", type=int,
                      default=(int(os.environ["AXON_EXTERNAL_PORT"]) if os.getenv("AXON_EXTERNAL_PORT") else None))

    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument("--logging.info", dest="logging.info",
                               action="store_true", default=_env_bool("BT_LOGGING_INFO", False),
                               help="Log at INFO level (the default); accepted for compatibility.")
    logging_group.add_argument("--logging.debug", dest="logging.debug",
                               action="store_true", default=_env_bool("BT_LOGGING_DEBUG", False))
    logging_group.add_argument("--logging.trace", dest="logging.trace",
                               action="store_true", default=_env_bool("BT_LOGGING_TRACE", False))
    logging_group.add_argument("--logging.logging_dir", dest="logging.logging_dir",
                               default=os.getenv("BT_LOGGING_DIR", "~/.bittensor/sn101"))


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--netuid", type=int, default=101, help="Subnet netuid.")
    parser.add_argument("--neuron.epoch_length", type=int, default=360)
    parser.add_argument("--neuron.block_cache_ttl", type=float, default=12.0)
    parser.add_argument("--neuron.storage_dir", type=str, default=None)
    parser.add_argument("--neuron.no_registration_check", action="store_true", default=False)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        if name.isupper() and "_" in name:
            return default
        raw = name
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _add_task_server_args(parser: argparse.ArgumentParser, *, default_url: str) -> None:
    parser.add_argument(
        "--task-server.url",
        dest="task_server.url",
        type=str,
        default=os.getenv("TASK_SERVER_URL", default_url),
    )
    parser.add_argument(
        "--task-server.timeout",
        dest="task_server.timeout",
        type=float,
        default=float(os.getenv("TASK_SERVER_TIMEOUT", "30.0")),
    )
    parser.add_argument(
        "--task-server.verify_ssl",
        dest="task_server.verify_ssl",
        type=_env_bool,
        default=_env_bool("TASK_SERVER_VERIFY_SSL", True),
        help="Verify HTTPS certificates for task-server requests.",
    )


def _add_miner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--neuron.name", type=str, default="miner")
    parser.add_argument("--blacklist.allow_non_registered", action="store_true", default=False)
    parser.add_argument("--blacklist.force_validator_permit", action="store_true", default=False)
    parser.add_argument("--miner.allow_empty_hotkey", action="store_true", default=False)
    parser.add_argument(
        "--task.miner_module",
        type=str,
        default=os.getenv("TASK_MINER_MODULE", ""),
        help=(
            "Optional task handler module used by miners to override a default "
            "task solver."
        ),
    )
    _add_task_server_args(parser, default_url=DEFAULT_TASK_SERVER_URL)
    parser.add_argument(
        "--miner.axon_to_public_metagraph",
        action="store_true",
        default=_env_bool("MINER_AXON_TO_PUBLIC_METAGRAPH", False),
        help="Use only the public metagraph axon and skip task-server private axon announcements.",
    )


def _add_validator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--neuron.name", type=str, default="validator")
    _add_task_server_args(parser, default_url=DEFAULT_TASK_SERVER_URL)
    parser.add_argument(
        "--task.kind",
        type=str,
        default=os.getenv("TASK_KIND", "sn101.tags.v1"),
        help="Optional task kind requested from the task server, for example sn101.tags.v1.",
    )
    parser.add_argument(
        "--task.profile_json",
        type=str,
        default=os.getenv("TASK_PROFILE_JSON", "{}"),
        help="JSON object merged into the task lease profile.",
    )
    parser.add_argument("--task.node_count", type=int, default=None)
    parser.add_argument("--task.edge_probability", type=float, default=None)
    parser.add_argument("--task.time_limit", type=float, default=None)
    parser.add_argument(
        "--validator.forward_interval",
        type=float,
        default=float(os.getenv("VALIDATOR_FORWARD_INTERVAL", str(SN101_TASK_INTERVAL_SECONDS))),
    )
    parser.add_argument(
        "--validator.ema_alpha",
        type=float,
        default=float(os.getenv("VALIDATOR_EMA_ALPHA", str(SN101_ONE_DAY_EMA_ALPHA))),
    )
    parser.add_argument(
        "--validator.score_strategy",
        type=str,
        default=os.getenv("VALIDATOR_SCORE_STRATEGY", "rolling_ma_softmax"),
        help=(
            "Score accumulation strategy from tag101.chain.scoreboard. "
            "Add custom strategies there and pass the registered name here."
        ),
    )
    parser.add_argument(
        "--validator.raw_history_max_events",
        type=int,
        default=int(
            os.getenv(
                "VALIDATOR_RAW_HISTORY_MAX_EVENTS",
                str(DEFAULT_RAW_HISTORY_MAX_EVENTS),
            )
        ),
    )
    parser.add_argument("--validator.disable_set_weights", action="store_true", default=False)
    parser.add_argument("--validator.min_steps_before_set_weights", type=int, default=5)
    parser.add_argument("--validator.max_response_bytes", type=int, default=128 * 1024)
    parser.add_argument("--validator.axon_off", action="store_true", default=False)


class _Config(DefaultMunch):
    """Munch-based config replacing the removed ``bittensor.Config``."""

    def is_set(self, dotted: str) -> bool:
        marker = DefaultMunch.get(self, "_explicit_keys", None)
        return bool(marker) and dotted in marker


def build_config(role: str) -> Any:
    parser = argparse.ArgumentParser(prog=f"tag101-{role}")
    _add_bittensor_args(parser)
    _add_common_args(parser)
    parser.add_argument("--config", default=None, help="Reserved for Bittensor-style config files.")
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--no_version_checking", action="store_true", default=False)
    if role == "miner":
        _add_miner_args(parser)
    elif role == "validator":
        _add_validator_args(parser)
    else:
        raise ValueError(f"unknown role: {role}")

    args = sys.argv[1:]
    namespace = parser.parse_args(args)
    explicit = _explicit_destinations(parser, args)
    config = _Config(None, {})
    for key, value in vars(namespace).items():
        _set_config_value(config, key, value)
    config["_explicit_keys"] = explicit
    _prepare_storage(config, role)
    if role == "validator":
        _prepare_task_profile(config)
    return config


def _prepare_task_profile(config: Any) -> None:
    raw = _config_value(config, "task.profile_json", "{}") or "{}"
    try:
        profile = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--task.profile_json must be a JSON object: {exc}") from exc
    if not isinstance(profile, dict):
        raise ValueError("--task.profile_json must decode to a JSON object")
    config.task.profile = profile


def _explicit_destinations(parser: argparse.ArgumentParser, args: list[str]) -> set[str]:
    by_option: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            by_option[option] = action.dest

    explicit: set[str] = set()
    for raw in args:
        option = raw.split("=", 1)[0]
        if option in by_option:
            explicit.add(by_option[option])
    return explicit


def _set_config_value(config: Any, dotted: str, value: Any) -> None:
    current = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not hasattr(current, part) or getattr(current, part) is None:
            setattr(current, part, DefaultMunch(None, {}))
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def _config_value(config: Any, dotted: str, default: Any = None) -> Any:
    current = config
    for part in dotted.split("."):
        if not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


def _prepare_storage(config: Any, role: str) -> None:
    explicit = _config_value(config, "neuron.storage_dir")
    if explicit:
        path = Path(os.path.expanduser(explicit))
    else:
        logging_dir = _config_value(config, "logging.logging_dir", "~/.bittensor/sn101")
        wallet_name = _config_value(config, "wallet.name", "default")
        hotkey = _config_value(config, "wallet.hotkey", "default")
        netuid = _config_value(config, "netuid", 0)
        neuron_name = _config_value(config, "neuron.name", role) or role
        path = Path(os.path.expanduser(logging_dir)) / wallet_name / hotkey / f"netuid{netuid}" / neuron_name
    path.mkdir(parents=True, exist_ok=True)
    config.neuron.storage_dir = str(path)
