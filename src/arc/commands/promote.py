from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.git import try_fast_forward_main


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("commit", help="Experiment commit hash or prefix.")


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    target = app.store.get_node_record(args.commit)
    if target is None:
        raise ArcError(f"Unknown commit: {args.commit}")
    if target.node.archived_at is not None:
        raise ArcError("Archived experiments cannot be promoted.")
    if target.node.status != "completed":
        raise ArcError("Only completed experiments can be promoted.")
    if target.node.verdict == "unsupported":
        raise ArcError("Unsupported completed experiments cannot be promoted.")

    previous_main_commit = app.main_commit()
    previous_main = (
        app.store.get_node_record(previous_main_commit) if previous_main_commit else None
    )
    metric_name = app.main_metric()
    app.store.set_meta("main", target.node.commit)
    branch_updated = try_fast_forward_main(app.paths.repo_root, target.node.commit)

    print(f"Promoted {target.node.commit} ({target.node.name}) to main.")
    if previous_main is not None:
        if metric_name and metric_name in previous_main.metrics:
            print(
                f"Previous main: {previous_main.node.commit} "
                f"({previous_main.metrics[metric_name]:g})"
            )
        else:
            print(f"Previous main: {previous_main.node.commit}")
    if metric_name and metric_name in target.metrics:
        print(f"New main:      {target.node.commit} ({target.metrics[metric_name]:g})")
    else:
        print(f"New main:      {target.node.commit}")
    print(f"Git main:      {'fast-forwarded' if branch_updated else 'unchanged'}")
    return 0


COMMAND = CommandSpec(
    name="promote",
    help="Promote a completed node to main.",
    register=register,
    run=run,
)
