from __future__ import annotations

import argparse
import shutil
import subprocess

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.runlog import summarize_run_log


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("commit", help="Experiment commit hash or prefix.")
    parser.add_argument(
        "--lines",
        type=int,
        default=40,
        help="Number of trailing lines to print before following.",
    )
    parser.add_argument(
        "--follow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Follow the log after printing. Defaults to yes only while the run is still active.",
    )


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    record = app.store.get_node_record(args.commit)
    if record is None:
        raise ArcError(f"Unknown commit: {args.commit}")

    log_path = app.node_log_path(record.node)
    if not log_path.is_file():
        raise ArcError(f"No run log found for `{record.node.commit}` at {app.relative_path(log_path)}.")

    tail_path = shutil.which("tail")
    if tail_path is None:
        raise ArcError("`tail` is not on PATH.")

    summary = summarize_run_log(log_path)
    follow = args.follow if args.follow is not None else summary.state == "running"
    cmd = [tail_path, "-n", str(args.lines)]
    if follow:
        cmd.append("-f")
    cmd.append(str(log_path))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


COMMAND = CommandSpec(
    name="tail",
    help="Print or follow a node's run.log without looking up the worktree path manually.",
    register=register,
    run=run,
)
