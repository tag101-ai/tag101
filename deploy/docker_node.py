#!/usr/bin/env python3
"""Manage one validator or miner in a Docker container."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .node_common import (
        CANONICAL_WALLET_PATH_OPTIONS,
        DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS,
        current_git_sha,
        last_option_value,
        load_env_file,
        load_env_files,
        normalize_wallet_path_options,
        parse_env_value,
        resolve_host_path,
        rewrite_option_values,
        split_env_args,
        strip_remainder_separator,
    )
except ImportError:  # pragma: no cover - used when this file is executed directly.
    from node_common import (  # type: ignore
        CANONICAL_WALLET_PATH_OPTIONS,
        DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS,
        current_git_sha,
        last_option_value,
        load_env_file,
        load_env_files,
        normalize_wallet_path_options,
        parse_env_value,
        resolve_host_path,
        rewrite_option_values,
        split_env_args,
        strip_remainder_separator,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
DOCKER_CONTEXT = REPO_ROOT
DEFAULT_IMAGE = "tag101:latest"
DEFAULT_WALLET_CONTAINER_PATH = "/home/node/.bittensor/wallets"
DEFAULT_STATE_CONTAINER_PATH = "/home/node/state"
DEFAULT_DOCKER_NETWORK = "host"
DEPLOY_STATE_DIR = REPO_ROOT / ".deploy-state"
ROLE_COMMANDS = {
    "validator": "tag101-validator",
    "miner": "tag101-miner",
}


@dataclass(frozen=True)
class PlannedCommand:
    args: list[str]
    check: bool = True
    background: bool = False
    log_file: Path | None = None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    commands = planned_commands(args)
    if args.command == "monitor":
        monitor_loop(args)
        return 0
    if args.command == "stop-monitor":
        stop_monitor(args.name)
        return 0
    run_planned_commands(commands, dry_run=bool(args.dry_run), cwd=REPO_ROOT)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run a single Docker node.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the Docker image.")
    add_dry_run_arg(build)
    build.add_argument("--image", default=None, help=f"Docker image tag. Defaults to {DEFAULT_IMAGE}.")
    build.add_argument("--dockerfile", default=str(DOCKERFILE_PATH))
    build.add_argument("--context", default=str(DOCKER_CONTEXT))
    build.add_argument("--git-sha", default=None, help="Override IMAGE_REVISION build arg.")

    start = subparsers.add_parser("start", help="Start a validator or miner container.")
    add_dry_run_arg(start)
    add_node_args(start)

    restart = subparsers.add_parser("restart", help="Restart a validator or miner container.")
    add_dry_run_arg(restart)
    add_node_args(restart)

    stop = subparsers.add_parser("stop", help="Stop and remove a container by name.")
    add_dry_run_arg(stop)
    stop.add_argument("--name", required=True)

    stop_monitor = subparsers.add_parser("stop-monitor", help="Stop the built-in auto-update monitor for a container.")
    add_dry_run_arg(stop_monitor)
    stop_monitor.add_argument("--name", required=True)

    status = subparsers.add_parser("status", help="Show Docker status for a container by name.")
    add_dry_run_arg(status)
    status.add_argument("--name", required=True)

    monitor = subparsers.add_parser("monitor", help="Run the built-in auto-update monitor loop.")
    add_dry_run_arg(monitor)
    add_monitor_args(monitor)

    return parser.parse_args(argv)


def add_dry_run_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS, help="Print commands without running them.")


def add_node_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--role", choices=sorted(ROLE_COMMANDS), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--env-file", action="append", default=[], help="Env file to pass to Docker and read deployment defaults from.")
    parser.add_argument("--image", default=None, help=f"Docker image tag. Defaults to {DEFAULT_IMAGE}.")
    parser.add_argument("--no-auto-update", action="store_true", help="Skip the built-in git upstream update check before starting.")
    parser.add_argument("--auto-update-upstream", default="@{u}", help="Git ref to compare before starting. Defaults to the current branch upstream.")
    parser.add_argument("--auto-update-no-fetch", action="store_true", help="Do not run git fetch before the built-in update check.")
    parser.add_argument("--auto-update-interval-seconds", type=float, default=DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS)
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Arguments after -- are passed to tag101-validator/miner.")


def add_monitor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--role", choices=sorted(ROLE_COMMANDS), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--env-file", action="append", default=[])
    parser.add_argument("--image", default=None)
    parser.add_argument("--auto-update-upstream", default="@{u}")
    parser.add_argument("--auto-update-no-fetch", action="store_true")
    parser.add_argument("--auto-update-interval-seconds", type=float, default=DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS)
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)


def planned_commands(args: argparse.Namespace) -> list[PlannedCommand]:
    if args.command == "build":
        image = args.image or DEFAULT_IMAGE
        return [
            PlannedCommand(
                build_image_command(
                    image=image,
                    dockerfile=Path(args.dockerfile),
                    context=Path(args.context),
                    git_sha=args.git_sha,
                )
            )
        ]
    if args.command == "start":
        env_files = [Path(path) for path in args.env_file]
        auto_update_enabled = not bool(args.no_auto_update)
        return start_commands(
            role=args.role,
            name=args.name,
            env_files=env_files,
            image=args.image,
            cli_args=strip_remainder_separator(args.passthrough),
            cwd=REPO_ROOT,
            pre_commands=[
                *stop_monitor_commands(args.name),
                *(auto_update_commands_from_args(args, env_files) if auto_update_enabled else []),
            ],
            post_commands=monitor_commands_from_args(args, env_files) if auto_update_enabled else [],
        )
    if args.command == "restart":
        env_files = [Path(path) for path in args.env_file]
        auto_update_enabled = not bool(args.no_auto_update)
        return restart_commands(
            role=args.role,
            name=args.name,
            env_files=env_files,
            image=args.image,
            cli_args=strip_remainder_separator(args.passthrough),
            cwd=REPO_ROOT,
            pre_commands=[
                *stop_monitor_commands(args.name),
                *(auto_update_commands_from_args(args, env_files) if auto_update_enabled else []),
            ],
            post_commands=monitor_commands_from_args(args, env_files) if auto_update_enabled else [],
        )
    if args.command == "stop":
        return [*stop_monitor_commands(args.name), *stop_commands(args.name)]
    if args.command == "stop-monitor":
        return stop_monitor_commands(args.name)
    if args.command == "status":
        return status_commands(args.name)
    if args.command == "monitor":
        return []
    raise ValueError(f"unknown command: {args.command}")


def build_image_command(
    *,
    image: str = DEFAULT_IMAGE,
    dockerfile: Path = DOCKERFILE_PATH,
    context: Path = DOCKER_CONTEXT,
    git_sha: str | None = None,
) -> list[str]:
    sha = git_sha or current_git_sha(REPO_ROOT)
    return [
        "docker",
        "build",
        "-f",
        str(dockerfile),
        "--build-arg",
        f"IMAGE_REVISION={sha}",
        "-t",
        image,
        str(context),
    ]


def start_commands(
    *,
    role: str,
    name: str,
    env_files: Sequence[Path] = (),
    image: str | None = None,
    cli_args: Sequence[str] = (),
    cwd: Path = REPO_ROOT,
    pre_commands: Sequence[PlannedCommand] = (),
    post_commands: Sequence[PlannedCommand] = (),
) -> list[PlannedCommand]:
    return [
        *pre_commands,
        *stop_commands(name),
        PlannedCommand(
            build_run_command(
                role=role,
                name=name,
                env_files=env_files,
                image=image,
                cli_args=cli_args,
                cwd=cwd,
            )
        ),
        *post_commands,
    ]


def restart_commands(
    *,
    role: str,
    name: str,
    env_files: Sequence[Path] = (),
    image: str | None = None,
    cli_args: Sequence[str] = (),
    cwd: Path = REPO_ROOT,
    pre_commands: Sequence[PlannedCommand] = (),
    post_commands: Sequence[PlannedCommand] = (),
) -> list[PlannedCommand]:
    return start_commands(
        role=role,
        name=name,
        env_files=env_files,
        image=image,
        cli_args=cli_args,
        cwd=cwd,
        pre_commands=pre_commands,
        post_commands=post_commands,
    )


def auto_update_commands_from_args(args: argparse.Namespace, env_files: Sequence[Path]) -> list[PlannedCommand]:
    return auto_update_commands(
        image=node_image(image=args.image, env_files=env_files),
        upstream=str(args.auto_update_upstream),
        fetch=not bool(args.auto_update_no_fetch) and not bool(args.dry_run),
    )


def monitor_commands_from_args(args: argparse.Namespace, env_files: Sequence[Path]) -> list[PlannedCommand]:
    command = monitor_process_command(args, env_files)
    return [PlannedCommand(command, background=True, log_file=monitor_log_path(args.name))]


def monitor_process_command(args: argparse.Namespace, env_files: Sequence[Path]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tag101.deploy.docker_node",
        "monitor",
        "--role",
        args.role,
        "--name",
        args.name,
        "--auto-update-upstream",
        str(args.auto_update_upstream),
        "--auto-update-interval-seconds",
        f"{float(args.auto_update_interval_seconds):g}",
    ]
    for env_file in env_files:
        command.extend(["--env-file", str(env_file)])
    if args.image:
        command.extend(["--image", args.image])
    if bool(args.auto_update_no_fetch):
        command.append("--auto-update-no-fetch")
    passthrough = strip_remainder_separator(args.passthrough)
    if passthrough:
        command.extend(["--", *passthrough])
    return command


def auto_update_commands(
    *,
    image: str,
    upstream: str = "@{u}",
    fetch: bool = True,
    repo_root: Path = REPO_ROOT,
) -> list[PlannedCommand]:
    try:
        from . import git_autoupdate
    except ImportError:  # pragma: no cover - used when this file is executed directly.
        import git_autoupdate  # type: ignore

    check = git_autoupdate.inspect_repository(repo=repo_root, upstream=upstream, fetch=fetch)
    print(git_autoupdate.format_check(check), flush=True)
    if not check.should_update:
        return []
    return [
        PlannedCommand(["git", "pull", "--ff-only"]),
        PlannedCommand([sys.executable, "-m", "tag101.deploy.docker_node", "build", "--image", image]),
    ]


def monitor_loop(args: argparse.Namespace) -> None:
    env_files = [Path(path) for path in args.env_file]
    cli_args = strip_remainder_separator(args.passthrough)
    lock_path = monitor_lock_path(args.name)
    pid_path = monitor_pid_path(args.name)
    DEPLOY_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"auto_update_monitor name={args.name} status=already_running", flush=True)
            return
        pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
        print(
            f"auto_update_monitor name={args.name} status=started interval={float(args.auto_update_interval_seconds):g}s",
            flush=True,
        )
        try:
            while True:
                try:
                    commands = auto_update_commands(
                        image=node_image(image=args.image, env_files=env_files),
                        upstream=str(args.auto_update_upstream),
                        fetch=not bool(args.auto_update_no_fetch),
                    )
                    if commands:
                        run_planned_commands(commands, dry_run=False, cwd=REPO_ROOT)
                        restart_without_monitor = restart_commands(
                            role=args.role,
                            name=args.name,
                            env_files=env_files,
                            image=args.image,
                            cli_args=cli_args,
                            cwd=REPO_ROOT,
                        )
                        run_planned_commands(restart_without_monitor, dry_run=False, cwd=REPO_ROOT)
                        reexec_monitor(args, env_files)
                except Exception as exc:
                    print(f"auto_update_monitor name={args.name} error={type(exc).__name__}: {exc}", flush=True)
                time.sleep(max(1.0, float(args.auto_update_interval_seconds)))
        finally:
            try:
                pid_path.unlink()
            except FileNotFoundError:
                pass


def reexec_monitor(args: argparse.Namespace, env_files: Sequence[Path]) -> None:
    command = monitor_process_command(args, env_files)
    print(f"auto_update_monitor name={args.name} status=reexecing_after_update", flush=True)
    os.execv(command[0], command)


def node_image(*, image: str | None, env_files: Sequence[Path]) -> str:
    env = load_env_files(env_files)
    return image or env.get("NODE_IMAGE") or DEFAULT_IMAGE


def stop_monitor_commands(name: str) -> list[PlannedCommand]:
    return [PlannedCommand([sys.executable, "-m", "tag101.deploy.docker_node", "stop-monitor", "--name", name], check=False)]


def stop_commands(name: str) -> list[PlannedCommand]:
    return [PlannedCommand(["docker", "rm", "-f", name], check=False)]


def status_commands(name: str) -> list[PlannedCommand]:
    return [
        PlannedCommand(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"name=^/{name}$",
                "--format",
                "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}",
            ]
        )
    ]


def stop_monitor(name: str) -> None:
    pid_path = monitor_pid_path(name)
    try:
        raw_pid = pid_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return
    if not raw_pid:
        return
    try:
        pid = int(raw_pid)
    except ValueError:
        pid_path.unlink(missing_ok=True)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_path.unlink(missing_ok=True)
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    pid_path.unlink(missing_ok=True)


def monitor_lock_path(name: str) -> Path:
    return DEPLOY_STATE_DIR / f"{safe_name(name)}.monitor.lock"


def monitor_pid_path(name: str) -> Path:
    return DEPLOY_STATE_DIR / f"{safe_name(name)}.monitor.pid"


def monitor_log_path(name: str) -> Path:
    return DEPLOY_STATE_DIR / f"{safe_name(name)}.monitor.log"


def safe_name(name: str) -> str:
    value = "".join(char if char.isalnum() or char in "_.-" else "-" for char in name.strip())
    return value.strip("-.") or "node"


def build_run_command(
    *,
    role: str,
    name: str,
    env_files: Sequence[Path] = (),
    image: str | None = None,
    cli_args: Sequence[str] = (),
    cwd: Path = REPO_ROOT,
) -> list[str]:
    if role not in ROLE_COMMANDS:
        raise ValueError(f"unsupported role: {role}")

    env = load_env_files(env_files)
    selected_image = image or env.get("NODE_IMAGE") or DEFAULT_IMAGE
    command = [
        "docker",
        "run",
        "-d",
        "--restart",
        "unless-stopped",
        "--name",
        name,
    ]

    for env_file in env_files:
        command.extend(["--env-file", str(env_file)])

    docker_network = env.get("DOCKER_NETWORK", DEFAULT_DOCKER_NETWORK)
    if docker_network:
        command.extend(["--network", docker_network])

    env_process_args = normalize_wallet_path_options(split_env_args(env.get("NODE_ARGS", "")))
    cli_process_args = normalize_wallet_path_options(cli_args)
    inferred_wallet_host = last_option_value(
        [*env_process_args, *cli_process_args],
        CANONICAL_WALLET_PATH_OPTIONS,
    )
    wallet_host = env.get("WALLET_HOST_PATH") or inferred_wallet_host
    wallet_container = env.get("WALLET_CONTAINER_PATH", DEFAULT_WALLET_CONTAINER_PATH)

    command.extend(volume_args(env, cwd, wallet_host=wallet_host))
    command.extend(port_args(env))
    command.extend(gpu_args(env))
    command.extend(split_env_args(env.get("DOCKER_RUN_ARGS", "")))

    if wallet_host:
        env_process_args = rewrite_option_values(env_process_args, CANONICAL_WALLET_PATH_OPTIONS, wallet_container)
        cli_process_args = rewrite_option_values(cli_process_args, CANONICAL_WALLET_PATH_OPTIONS, wallet_container)

    process_args = [
        *auto_process_args(env),
        *env_process_args,
        *cli_process_args,
    ]
    return [
        *command,
        selected_image,
        ROLE_COMMANDS[role],
        *process_args,
    ]


def volume_args(env: Mapping[str, str], cwd: Path, *, wallet_host: str | None = None) -> list[str]:
    args: list[str] = []
    if wallet_host:
        wallet_container = env.get("WALLET_CONTAINER_PATH", DEFAULT_WALLET_CONTAINER_PATH)
        args.extend(["-v", f"{resolve_host_path(wallet_host, cwd)}:{wallet_container}"])

    state_host = env.get("STATE_HOST_DIR")
    if state_host:
        state_container = env.get("STATE_CONTAINER_DIR", DEFAULT_STATE_CONTAINER_PATH)
        args.extend(["-v", f"{resolve_host_path(state_host, cwd)}:{state_container}"])
    return args


def port_args(env: Mapping[str, str]) -> list[str]:
    if env.get("DOCKER_NETWORK", DEFAULT_DOCKER_NETWORK) == "host":
        return []
    host_port = env.get("AXON_HOST_PORT") or env.get("AXON_PORT")
    container_port = env.get("AXON_CONTAINER_PORT") or env.get("AXON_PORT")
    if not host_port or not container_port:
        return []
    return ["-p", f"{host_port}:{container_port}"]


def gpu_args(env: Mapping[str, str]) -> list[str]:
    gpus = env.get("DOCKER_GPUS")
    if not gpus:
        return []
    return ["--gpus", gpus]


def auto_process_args(env: Mapping[str, str]) -> list[str]:
    args: list[str] = []
    wallet_host = env.get("WALLET_HOST_PATH")
    if wallet_host:
        wallet_container = env.get("WALLET_CONTAINER_PATH", DEFAULT_WALLET_CONTAINER_PATH)
        args.extend(["--wallet.path", wallet_container])

    state_host = env.get("STATE_HOST_DIR")
    if state_host:
        state_container = env.get("STATE_CONTAINER_DIR", DEFAULT_STATE_CONTAINER_PATH)
        args.extend(["--neuron.storage_dir", state_container])

    axon_port = env.get("AXON_CONTAINER_PORT") or env.get("AXON_PORT")
    if axon_port:
        args.extend(["--axon.ip", env.get("AXON_IP", "0.0.0.0")])
        args.extend(["--axon.port", axon_port])
        external_port = env.get("AXON_EXTERNAL_PORT") or env.get("AXON_HOST_PORT") or env.get("AXON_PORT")
        args.extend(["--axon.external_port", external_port])

    external_ip = env.get("AXON_EXTERNAL_IP")
    if external_ip:
        args.extend(["--axon.external_ip", external_ip])
    return args


def run_planned_commands(commands: Sequence[PlannedCommand], *, dry_run: bool, cwd: Path) -> None:
    for planned in commands:
        printable = shlex.join(planned.args)
        if dry_run:
            prefix = "DRY RUN background" if planned.background else "DRY RUN"
            suffix = f" > {planned.log_file} 2>&1" if planned.background and planned.log_file else ""
            print(f"{prefix}: {printable}{suffix}")
            continue
        if planned.background:
            log_file = planned.log_file or (DEPLOY_STATE_DIR / "background.log")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("ab") as handle:
                process = subprocess.Popen(
                    planned.args,
                    cwd=cwd,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            print(f"started background pid={process.pid} log={log_file}", flush=True)
            continue
        subprocess.run(planned.args, cwd=cwd, check=planned.check)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
