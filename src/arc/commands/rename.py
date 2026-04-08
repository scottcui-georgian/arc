from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.paths import validate_name


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("commit", help="Experiment commit hash or prefix.")
    parser.add_argument("name", help="New experiment name.")


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    record = app.store.get_node_record(args.commit)
    if record is None:
        raise ArcError(f"Unknown commit: {args.commit}")
    if record.node.archived_at is not None:
        raise ArcError(f"Cannot rename archived node `{record.node.commit}`.")

    new_name = validate_name(args.name)
    previous_name = record.node.name
    if previous_name == new_name:
        print(f"Name unchanged for {app.display_commit(record.node.commit)} ({record.node.name})")
        print(f"Name: {previous_name}")
        return 0

    app.store.update_node(record.node.commit, name=new_name)
    print(f"Renamed {app.display_commit(record.node.commit)}")
    print(f"Name: {previous_name} → {new_name}")
    return 0


COMMAND = CommandSpec(
    name="rename",
    help="Update the display name for an existing experiment.",
    register=register,
    run=run,
)
