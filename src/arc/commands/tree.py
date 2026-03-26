from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.render import render_tree


def register(parser: argparse.ArgumentParser) -> None:
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = (
        "Tree markers:\n"
        "  ● committed\n"
        "  ◌ running\n"
        "  ✓ completed + promising\n"
        "  ○ completed + unsupported\n"
        "  ✗ hard failure\n"
        "  ◦ archived prefix\n"
        "  (main) current main node\n"
        "  (best) best completed leaf on the main metric"
    )
    parser.add_argument(
        "--status",
        choices=("running", "completed", "failed", "committed", "archived"),
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
    parser.add_argument(
        "--concise",
        action="store_true",
        help="Hide archived nodes unless they are explicitly requested.",
    )


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()
    archived_only = args.status == "archived"
    status_filter = None if archived_only else args.status
    include_archived = archived_only or not args.concise
    records = app.store.list_node_records(include_archived=include_archived)
    print(
        render_tree(
            records,
            metric_name=app.main_metric(),
            direction=app.metric_direction(),
            main_commit=app.main_commit(),
            status_filter=status_filter,
            archived_only=archived_only,
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
