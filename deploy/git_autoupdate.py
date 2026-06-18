#!/usr/bin/env python3
"""Auto-update a Docker node when the current git branch advances."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from . import docker_node
    from .node_common import DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS, load_env_files, strip_remainder_separator
except ImportError:  # pragma: no cover - used when this file is executed directly.
    import docker_node  # type: ignore
    from node_common import DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS, load_env_files, strip_remainder_separator  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_NODE_MODULE = "tag101.deploy.docker_node"


@dataclass(frozen=True)
class UpdateCheck:
    status: str
    reason: str
    head: str | None = None
    upstream_ref: str | None = None
    upstream_sha: str | None = None

    @property
    def should_update(self) -> bool:
        return self.status == "needs_update"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    while True:
        check = inspect_repository(repo=Path(args.repo), upstream=args.upstream, fetch=not args.no_fetch)
        print(format_check(check), flush=True)
        if check.should_update:
            commands = update_commands(args)
            run_commands(commands, cwd=Path(args.repo), dry_run=bool(args.dry_run))
        if args.once:
            return 0
        time.sleep(float(args.interval_seconds))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch git upstream and restart one Dockerized node after updates.")
    parser.add_argument("--repo", default=str(REPO_ROOT), help="Repository root. Defaults to this checkout.")
    parser.add_argument("--upstream", default="@{u}", help="Git ref to compare against. Defaults to the current branch upstream.")
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_AUTO_UPDATE_INTERVAL_SECONDS)
    parser.add_argument("--role", choices=sorted(docker_node.ROLE_COMMANDS), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--env-file", action="append", default=[])
    parser.add_argument("--image", default=None)
    parser.add_argument("--once", action="store_true", help="Check once and exit.")
    parser.add_argument("--no-fetch", action="store_true", help="Skip git fetch before comparison.")
    parser.add_argument("--dry-run", action="store_true", help="Print update commands without running them.")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Arguments after -- are passed to the restarted node.")
    return parser.parse_args(argv)


def inspect_repository(*, repo: Path, upstream: str = "@{u}", fetch: bool = True) -> UpdateCheck:
    if is_detached_head(repo):
        return UpdateCheck("detached", "HEAD is detached; skipping auto-update", upstream_ref=upstream)

    upstream_ref = resolve_upstream(repo, upstream)
    if upstream_ref is None:
        return UpdateCheck("no_upstream", "current branch has no upstream", upstream_ref=upstream)

    if fetch:
        run_git(["fetch", "--prune"], repo=repo, capture=False, check=True)

    head = git_stdout(["rev-parse", "HEAD"], repo=repo)
    upstream_sha = git_stdout(["rev-parse", upstream_ref], repo=repo)
    can_fast_forward = is_ancestor(repo, head, upstream_sha) if head != upstream_sha else False
    can_apply_fast_forward = can_apply_fast_forward_update(repo, head, upstream_sha) if can_fast_forward else False
    return classify_update(
        head=head,
        upstream_ref=upstream_ref,
        upstream_sha=upstream_sha,
        dirty=is_worktree_dirty(repo),
        can_fast_forward=can_fast_forward,
        can_apply_fast_forward=can_apply_fast_forward,
    )


def classify_update(
    *,
    head: str,
    upstream_ref: str,
    upstream_sha: str,
    dirty: bool,
    can_fast_forward: bool,
    can_apply_fast_forward: bool | None = None,
) -> UpdateCheck:
    if head == upstream_sha:
        return UpdateCheck("up_to_date", "HEAD already matches upstream", head, upstream_ref, upstream_sha)
    if not can_fast_forward:
        return UpdateCheck("diverged", "HEAD cannot fast-forward to upstream; skipping auto-update", head, upstream_ref, upstream_sha)
    if can_apply_fast_forward is None:
        can_apply_fast_forward = not dirty
    if not can_apply_fast_forward:
        return UpdateCheck("dirty", "fast-forward would overwrite local worktree changes; skipping auto-update", head, upstream_ref, upstream_sha)
    if dirty:
        return UpdateCheck(
            "needs_update",
            "upstream has new commits and git can fast-forward with local changes",
            head,
            upstream_ref,
            upstream_sha,
        )
    return UpdateCheck("needs_update", "upstream has new commits and HEAD can fast-forward", head, upstream_ref, upstream_sha)


def is_worktree_dirty(repo: Path) -> bool:
    return bool(git_stdout(["status", "--porcelain"], repo=repo))


def can_apply_fast_forward_update(repo: Path, head: str, upstream_sha: str) -> bool:
    result = run_git(["read-tree", "-n", "-m", "-u", head, upstream_sha], repo=repo, capture=True, check=False)
    return result.returncode == 0


def resolve_upstream(repo: Path, upstream: str) -> str | None:
    result = run_git(
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", upstream],
        repo=repo,
        capture=True,
        check=False,
    )
    value = result.stdout.strip()
    return value or None


def is_detached_head(repo: Path) -> bool:
    result = run_git(["symbolic-ref", "--quiet", "--short", "HEAD"], repo=repo, capture=True, check=False)
    return result.returncode != 0


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = run_git(["merge-base", "--is-ancestor", ancestor, descendant], repo=repo, capture=False, check=False)
    return result.returncode == 0


def update_commands(args: argparse.Namespace) -> list[list[str]]:
    env_files = [Path(path) for path in args.env_file]
    env = load_env_files(env_files)
    image = args.image or env.get("NODE_IMAGE") or docker_node.DEFAULT_IMAGE
    passthrough = strip_remainder_separator(args.passthrough)

    build_command = [
        sys.executable,
        "-m",
        DOCKER_NODE_MODULE,
        "build",
        "--image",
        image,
    ]
    restart_command = [
        sys.executable,
        "-m",
        DOCKER_NODE_MODULE,
        "restart",
        "--role",
        args.role,
        "--name",
        args.name,
    ]
    for env_file in env_files:
        restart_command.extend(["--env-file", str(env_file)])
    if args.image:
        restart_command.extend(["--image", args.image])
    if passthrough:
        restart_command.extend(["--", *passthrough])

    return [
        ["git", "pull", "--ff-only"],
        build_command,
        restart_command,
    ]


def run_commands(commands: Sequence[Sequence[str]], *, cwd: Path, dry_run: bool) -> None:
    for command in commands:
        printable = shlex.join(list(command))
        if dry_run:
            print(f"DRY RUN: {printable}", flush=True)
            continue
        subprocess.run(command, cwd=cwd, check=True)


def git_stdout(args: Sequence[str], *, repo: Path) -> str:
    result = run_git(args, repo=repo, capture=True, check=True)
    return result.stdout.strip()


def run_git(
    args: Sequence[str],
    *,
    repo: Path,
    capture: bool,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=check,
    )


def format_check(check: UpdateCheck) -> str:
    parts = [f"git_update status={check.status}", f"reason={shlex.quote(check.reason)}"]
    if check.upstream_ref:
        parts.append(f"upstream={check.upstream_ref}")
    if check.head:
        parts.append(f"head={check.head[:12]}")
    if check.upstream_sha:
        parts.append(f"upstream_sha={check.upstream_sha[:12]}")
    return " ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
