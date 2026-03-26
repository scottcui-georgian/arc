from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.render import render_tree


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--status",
        choices=("running", "completed", "failed", "committed"),
        help="Filter visible nodes by status.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="Limit rendered depth below the root.",
    )
    parser.add_argument(
        "--leaves",
        action="store_true",
        help="Show only leaf nodes and their ancestors.",
    )


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()
    records = app.store.list_node_records()
    print(
        render_tree(
            records,
            metric_name=app.main_metric(),
            direction=app.metric_direction(),
            main_commit=app.main_commit(),
            status_filter=args.status,
            depth=args.depth,
            leaves_only=args.leaves,
        )
    )
    return 0


COMMAND = CommandSpec(
    name="tree",
    help="Print the experiment tree.",
    register=register,
    run=run,
)
