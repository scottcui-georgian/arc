from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.git import current_head
from arc.models import Node
from arc.timeutil import utc_now_iso


def register(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Initialize arc in the current repository, create a baseline node at the current HEAD, "
        "and configure the primary metric used by tree views and promotion decisions."
    )
    parser.add_argument(
        "--metric",
        help="Single primary metric used by `arc tree`, `arc report`, and promotion decisions.",
    )
    parser.add_argument(
        "--direction",
        choices=("min", "max"),
        default="min",
        help="Whether lower or higher values are better for the primary metric.",
    )


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    if app.store.exists():
        raise ArcError("Arc is already initialized in this repository.")

    app.ensure_directories()
    app.store.initialize(app.paths.repo_root)

    root_commit = current_head(app.paths.repo_root)
    created_at = utc_now_iso()
    app.store.insert_node(
        Node(
            commit=root_commit,
            parent=None,
            name="baseline",
            status="committed",
            hypothesis=None,
            analysis=None,
            worktree=".",
            created_at=created_at,
            completed_at=None,
            verdict=None,
            archived_at=None,
        )
    )
    app.store.set_meta("main", root_commit)
    if args.metric:
        app.store.set_meta("main_metric", args.metric)
    app.store.set_meta("main_metric_direction", args.direction)
    app.store.set_meta("task", getattr(app.task, "name", "default"))
    app.store.set_meta("last_status_check", created_at)

    print(f"Initialized arc in {app.paths.repo_root}")
    print(f"Root node: {root_commit}")
    if args.metric:
        direction = "lower is better" if args.direction == "min" else "higher is better"
        print(f"Main metric: {args.metric} ({direction})")
    return 0


COMMAND = CommandSpec(
    name="init",
    help="Initialize arc, create a baseline node, and set the primary metric.",
    register=register,
    run=run,
)
