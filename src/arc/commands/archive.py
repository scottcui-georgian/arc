from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.timeutil import utc_now_iso


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("commit", help="Experiment commit hash or prefix.")


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    record = app.store.get_node_record(args.commit)
    if record is None:
        raise ArcError(f"Unknown commit: {args.commit}")
    if record.node.archived_at is not None:
        raise ArcError(f"Node `{record.node.commit}` is already archived.")
    if record.node.parent is None:
        raise ArcError("Cannot archive the root baseline node.")
    if record.node.status == "running":
        raise ArcError("Cannot archive a running node.")
    if app.main_commit() == record.node.commit:
        raise ArcError("Cannot archive the current main node.")
    if app.store.has_children(record.node.commit):
        raise ArcError("Only leaf nodes can be archived.")

    archived_at = utc_now_iso()
    app.store.archive_node(record.node.commit, archived_at)

    print(f"Archived {app.display_commit(record.node.commit)} ({record.node.name})")
    print(f"Archived: {archived_at}")
    return 0


COMMAND = CommandSpec(
    name="archive",
    help="Hide a stale leaf node from the default tree view.",
    register=register,
    run=run,
)
