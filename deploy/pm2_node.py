#!/usr/bin/env python3
"""Manage one validator or miner with PM2."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    from . import git_autoupdate
    from .node_common import (
        DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS,
        load_env_files,
        normalize_wallet_path_options,
        resolve_host_path,
        split_env_args,
        strip_remainder_separator,
    )
except ImportError:  # pragma: no cover - used when this file is executed directly.
    import git_autoupdate  # type: ignore
    from node_common import (  # type: ignore
        DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS,
        load_env_files,
        normalize_wallet_path_options,
        resolve_host_path,
        split_env_args,
        strip_remainder_separator,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
PM2_MONITOR_SUFFIX = "auto-update"
ROLE_MODULES = {
    "validator": "tag101.validator",
    "miner": "tag101.miner",
}


@dataclass(frozen=True)
class PlannedCommand:
    args: list[str]
    check: bool = True
    env: dict[str, str] | None = None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "monitor":
        monitor_loop(args)
        return 0
    commands = planned_commands(args)
    run_planned_commands(commands, dry_run=bool(args.dry_run), cwd=REPO_ROOT)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single validator or miner under PM2.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start a validator or miner with PM2.")
    add_dry_run_arg(start)
    add_node_args(start)

    restart = subparsers.add_parser("restart", help="Restart a validator or miner with PM2.")
    add_dry_run_arg(restart)
    add_node_args(restart)

    stop = subparsers.add_parser("stop", help="Stop a validator or miner and its auto-update monitor.")
    add_dry_run_arg(stop)
    stop.add_argument("--name", required=True)
    stop.add_argument("--no-save", action="store_true", help="Do not run pm2 save after deleting processes.")

    stop_monitor = subparsers.add_parser("stop-monitor", help="Stop only the PM2 auto-update monitor.")
    add_dry_run_arg(stop_monitor)
    stop_monitor.add_argument("--name", required=True)
    stop_monitor.add_argument("--no-save", action="store_true", help="Do not run pm2 save after deleting the monitor.")

    status = subparsers.add_parser("status", help="Show PM2 status for a node and its monitor.")
    add_dry_run_arg(status)
    status.add_argument("--name", required=True)

    monitor = subparsers.add_parser("monitor", help="Run the PM2 auto-update monitor loop.")
    add_dry_run_arg(monitor)
    add_monitor_args(monitor)

    return parser.parse_args(argv)


def add_dry_run_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS, help="Print commands without running them.")


def add_node_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--role", choices=sorted(ROLE_MODULES), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--env-file", action="append", default=[], help="Env file to pass to PM2 and read deployment defaults from.")
    parser.add_argument("--python", default=None, help="Python executable PM2 should run. Defaults to PYTHON_BIN/PM2_PYTHON or this interpreter.")
    parser.add_argument("--no-auto-update", action="store_true", help="Skip git upstream checks and do not start the PM2 monitor.")
    parser.add_argument("--auto-update-upstream", default="@{u}", help="Git ref to compare before updating. Defaults to the current branch upstream.")
    parser.add_argument("--auto-update-no-fetch", action="store_true", help="Do not run git fetch before update checks.")
    parser.add_argument("--auto-update-interval-seconds", type=float, default=DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS)
    parser.add_argument("--no-install", action="store_true", help="After git pull, skip python -m pip install -e.")
    parser.add_argument("--no-save", action="store_true", help="Do not run pm2 save after changing the process list.")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Arguments after -- are passed to tag101.validator/miner.")


def add_monitor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--role", choices=sorted(ROLE_MODULES), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--env-file", action="append", default=[])
    parser.add_argument("--python", default=None)
    parser.add_argument("--auto-update-upstream", default="@{u}")
    parser.add_argument("--auto-update-no-fetch", action="store_true")
    parser.add_argument("--auto-update-interval-seconds", type=float, default=DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS)
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)


def planned_commands(args: argparse.Namespace) -> list[PlannedCommand]:
    if args.command in {"start", "restart"}:
        env_files = [Path(path) for path in args.env_file]
        auto_update_enabled = not bool(args.no_auto_update)
        return start_commands(
            role=args.role,
            name=args.name,
            env_files=env_files,
            python_bin=node_python(args.python, env_files),
            cli_args=strip_remainder_separator(args.passthrough),
            cwd=REPO_ROOT,
            save=not bool(args.no_save),
            pre_commands=[
                *stop_monitor_commands(args.name),
                *(auto_update_commands_from_args(args, env_files) if auto_update_enabled else []),
            ],
            post_commands=monitor_commands_from_args(args, env_files) if auto_update_enabled else [],
        )
    if args.command == "stop":
        return [
            *stop_monitor_commands(args.name),
            *stop_commands(args.name),
            *save_commands(enabled=not bool(args.no_save)),
        ]
    if args.command == "stop-monitor":
        return [
            *stop_monitor_commands(args.name),
            *save_commands(enabled=not bool(args.no_save)),
        ]
    if args.command == "status":
        return status_commands(args.name)
    if args.command == "monitor":
        return []
    raise ValueError(f"unknown command: {args.command}")


def start_commands(
    *,
    role: str,
    name: str,
    env_files: Sequence[Path] = (),
    python_bin: str | None = None,
    cli_args: Sequence[str] = (),
    cwd: Path = REPO_ROOT,
    pre_commands: Sequence[PlannedCommand] = (),
    post_commands: Sequence[PlannedCommand] = (),
    save: bool = True,
) -> list[PlannedCommand]:
    env = load_env_files(env_files)
    return [
        *pre_commands,
        *stop_commands(name),
        PlannedCommand(
            build_start_command(
                role=role,
                name=name,
                env_files=env_files,
                python_bin=python_bin,
                cli_args=cli_args,
                cwd=cwd,
            ),
            env=env or None,
        ),
        *post_commands,
        *save_commands(enabled=save),
    ]


def restart_commands(
    *,
    role: str,
    name: str,
    env_files: Sequence[Path] = (),
    python_bin: str | None = None,
    cli_args: Sequence[str] = (),
    cwd: Path = REPO_ROOT,
    save: bool = True,
) -> list[PlannedCommand]:
    return start_commands(
        role=role,
        name=name,
        env_files=env_files,
        python_bin=python_bin,
        cli_args=cli_args,
        cwd=cwd,
        save=save,
    )


def build_start_command(
    *,
    role: str,
    name: str,
    env_files: Sequence[Path] = (),
    python_bin: str | None = None,
    cli_args: Sequence[str] = (),
    cwd: Path = REPO_ROOT,
) -> list[str]:
    if role not in ROLE_MODULES:
        raise ValueError(f"unsupported role: {role}")

    env = load_env_files(env_files)
    selected_python = python_bin or node_python(None, env_files)
    env_process_args = normalize_wallet_path_options(split_env_args(env.get("NODE_ARGS", "")))
    cli_process_args = normalize_wallet_path_options(cli_args)
    process_args = [
        *auto_process_args(env, cwd),
        *env_process_args,
        *cli_process_args,
    ]
    return [
        "pm2",
        "start",
        selected_python,
        "--name",
        name,
        "--cwd",
        str(cwd),
        "--interpreter",
        "none",
        "--time",
        *split_env_args(env.get("PM2_START_ARGS", "")),
        "--",
        "-m",
        ROLE_MODULES[role],
        *process_args,
    ]


def auto_process_args(env: Mapping[str, str], cwd: Path) -> list[str]:
    args: list[str] = []
    wallet_host = env.get("PM2_WALLET_PATH") or env.get("WALLET_HOST_PATH")
    if wallet_host:
        args.extend(["--wallet.path", resolve_host_path(wallet_host, cwd)])

    state_host = env.get("PM2_STATE_DIR") or env.get("STATE_HOST_DIR")
    if state_host:
        args.extend(["--neuron.storage_dir", resolve_host_path(state_host, cwd)])

    axon_port = env.get("AXON_PORT") or env.get("AXON_HOST_PORT")
    if axon_port:
        args.extend(["--axon.ip", env.get("AXON_IP", "0.0.0.0")])
        args.extend(["--axon.port", axon_port])
        external_port = env.get("AXON_EXTERNAL_PORT") or axon_port
        args.extend(["--axon.external_port", external_port])

    external_ip = env.get("AXON_EXTERNAL_IP")
    if external_ip:
        args.extend(["--axon.external_ip", external_ip])
    return args


def auto_update_commands_from_args(args: argparse.Namespace, env_files: Sequence[Path]) -> list[PlannedCommand]:
    return auto_update_commands(
        upstream=str(args.auto_update_upstream),
        fetch=not bool(args.auto_update_no_fetch) and not bool(args.dry_run),
        install=not bool(args.no_install),
        python_bin=node_python(args.python, env_files),
    )


def auto_update_commands(
    *,
    upstream: str = "@{u}",
    fetch: bool = True,
    install: bool = True,
    python_bin: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> list[PlannedCommand]:
    check = git_autoupdate.inspect_repository(repo=repo_root, upstream=upstream, fetch=fetch)
    print(git_autoupdate.format_check(check), flush=True)
    if not check.should_update:
        return []

    commands = [PlannedCommand(["git", "pull", "--ff-only"])]
    if install:
        commands.append(PlannedCommand([python_bin or sys.executable, "-m", "pip", "install", "-e", str(repo_root)]))
    return commands


def monitor_commands_from_args(args: argparse.Namespace, env_files: Sequence[Path]) -> list[PlannedCommand]:
    selected_python = node_python(args.python, env_files)
    process_command = monitor_process_command(args, env_files, selected_python)
    command = [
        "pm2",
        "start",
        selected_python,
        "--name",
        monitor_name(args.name),
        "--cwd",
        str(REPO_ROOT),
        "--interpreter",
        "none",
        "--time",
        "--",
        *process_command[1:],
    ]
    env = load_env_files(env_files)
    return [PlannedCommand(command, env=env or None)]


def monitor_process_command(args: argparse.Namespace, env_files: Sequence[Path], selected_python: str) -> list[str]:
    command = [
        selected_python,
        "-m",
        "tag101.deploy.pm2_node",
        "monitor",
        "--role",
        args.role,
        "--name",
        args.name,
        "--python",
        selected_python,
        "--auto-update-upstream",
        str(args.auto_update_upstream),
        "--auto-update-interval-seconds",
        f"{float(args.auto_update_interval_seconds):g}",
    ]
    for env_file in env_files:
        command.extend(["--env-file", str(env_file)])
    if bool(args.auto_update_no_fetch):
        command.append("--auto-update-no-fetch")
    if bool(args.no_install):
        command.append("--no-install")
    if bool(args.no_save):
        command.append("--no-save")
    passthrough = strip_remainder_separator(args.passthrough)
    if passthrough:
        command.extend(["--", *passthrough])
    return command


def monitor_loop(args: argparse.Namespace) -> None:
    env_files = [Path(path) for path in args.env_file]
    cli_args = strip_remainder_separator(args.passthrough)
    selected_python = node_python(args.python, env_files)
    print(
        f"pm2_auto_update_monitor name={args.name} status=started interval={float(args.auto_update_interval_seconds):g}s",
        flush=True,
    )
    while True:
        try:
            commands = auto_update_commands(
                upstream=str(args.auto_update_upstream),
                fetch=not bool(args.auto_update_no_fetch),
                install=not bool(args.no_install),
                python_bin=selected_python,
            )
            if commands:
                run_planned_commands(commands, dry_run=False, cwd=REPO_ROOT)
                restart_without_monitor = restart_commands(
                    role=args.role,
                    name=args.name,
                    env_files=env_files,
                    python_bin=selected_python,
                    cli_args=cli_args,
                    cwd=REPO_ROOT,
                    save=not bool(args.no_save),
                )
                run_planned_commands(restart_without_monitor, dry_run=False, cwd=REPO_ROOT)
                reexec_monitor(args, env_files, selected_python)
        except Exception as exc:
            print(f"pm2_auto_update_monitor name={args.name} error={type(exc).__name__}: {exc}", flush=True)
        time.sleep(max(1.0, float(args.auto_update_interval_seconds)))


def reexec_monitor(args: argparse.Namespace, env_files: Sequence[Path], selected_python: str) -> None:
    command = monitor_process_command(args, env_files, selected_python)
    print(f"pm2_auto_update_monitor name={args.name} status=reexecing_after_update", flush=True)
    os.execv(command[0], command)


def node_python(python_bin: str | None, env_files: Sequence[Path]) -> str:
    if python_bin:
        return python_bin
    env = load_env_files(env_files)
    return env.get("PYTHON_BIN") or env.get("PM2_PYTHON") or sys.executable


def monitor_name(name: str) -> str:
    return f"{name}-{PM2_MONITOR_SUFFIX}"


def stop_commands(name: str) -> list[PlannedCommand]:
    return [PlannedCommand(["pm2", "delete", name], check=False)]


def stop_monitor_commands(name: str) -> list[PlannedCommand]:
    return [PlannedCommand(["pm2", "delete", monitor_name(name)], check=False)]


def save_commands(*, enabled: bool) -> list[PlannedCommand]:
    return [PlannedCommand(["pm2", "save"])] if enabled else []


def status_commands(name: str) -> list[PlannedCommand]:
    return [
        PlannedCommand(["pm2", "status", name], check=False),
        PlannedCommand(["pm2", "status", monitor_name(name)], check=False),
    ]


def run_planned_commands(commands: Sequence[PlannedCommand], *, dry_run: bool, cwd: Path) -> None:
    for planned in commands:
        printable = shlex.join(planned.args)
        if dry_run:
            print(f"DRY RUN: {printable}")
            continue
        env = None
        if planned.env:
            env = os.environ.copy()
            env.update(planned.env)
        subprocess.run(planned.args, cwd=cwd, env=env, check=planned.check)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
