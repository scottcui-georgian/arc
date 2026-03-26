from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.render import render_show


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("commit", help="Experiment commit hash or prefix.")


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()
    record = app.store.get_node_record(args.commit)
    if record is None:
        raise ArcError(f"Unknown commit: {args.commit}")

    parent = app.store.get_node_record(record.node.parent) if record.node.parent else None
    main_commit = app.main_commit()
    main = app.store.get_node_record(main_commit) if main_commit else None
    print(render_show(record, parent=parent, main=main))
    return 0


COMMAND = CommandSpec(
    name="show",
    help="Show full details for a single node.",
    register=register,
    run=run,
)
