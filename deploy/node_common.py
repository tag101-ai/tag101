"""Shared helpers for node deployment launchers."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS = 60.0
CANONICAL_WALLET_PATH_OPTIONS = {"--wallet.path"}


def load_env_files(paths: Iterable[Path]) -> dict[str, str]:
    env: dict[str, str] = {}
    for path in paths:
        env.update(load_env_file(path))
    return env


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_number}: empty key")
        values[key] = parse_env_value(value.strip())
    return values


def parse_env_value(value: str) -> str:
    if not value:
        return ""
    try:
        tokens = shlex.split(value, comments=False, posix=True)
    except ValueError:
        return value
    if len(tokens) == 1:
        return tokens[0]
    return value


def split_env_args(value: str) -> list[str]:
    if not value:
        return []
    return shlex.split(value)


def normalize_wallet_path_options(args: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for arg in args:
        if arg == "--wallet-path":
            normalized.append("--wallet.path")
        elif arg.startswith("--wallet-path="):
            normalized.append("--wallet.path=" + arg.split("=", 1)[1])
        else:
            normalized.append(arg)
    return normalized


def last_option_value(args: Sequence[str], option_names: set[str]) -> str | None:
    value: str | None = None
    index = 0
    while index < len(args):
        arg = args[index]
        if "=" in arg:
            option, raw_value = arg.split("=", 1)
            if option in option_names:
                value = raw_value
            index += 1
            continue
        if arg in option_names and index + 1 < len(args):
            value = args[index + 1]
            index += 2
            continue
        index += 1
    return value


def rewrite_option_values(args: Sequence[str], option_names: set[str], value: str) -> list[str]:
    rewritten: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if "=" in arg:
            option, _raw_value = arg.split("=", 1)
            if option in option_names:
                rewritten.append(f"{option}={value}")
            else:
                rewritten.append(arg)
            index += 1
            continue
        rewritten.append(arg)
        if arg in option_names and index + 1 < len(args):
            rewritten.append(value)
            index += 2
            continue
        index += 1
    return rewritten


def strip_remainder_separator(args: Sequence[str]) -> list[str]:
    if args and args[0] == "--":
        return list(args[1:])
    return list(args)


def resolve_host_path(value: str, cwd: Path) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = cwd / path
    return str(path)


def current_git_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() or "unknown"
