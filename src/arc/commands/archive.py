from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.timeutil import utc_now_iso


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("commit", help="Experiment commit hash or prefix.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Archive the node and all its descendants.",
    )


def _collect_descendants(app: ArcApp, commit: str) -> list[str]:
    """Return all descendant commits of *commit* (not including *commit* itself)."""
    all_nodes = app.store.list_nodes(include_archived=True)
    children_map: dict[str, list[str]] = {}
    for node in all_nodes:
        if node.parent:
            children_map.setdefault(node.parent, []).append(node.commit)

    descendants: list[str] = []
    queue = list(children_map.get(commit, []))
    while queue:
        current = queue.pop()
        descendants.append(current)
        queue.extend(children_map.get(current, []))
    return descendants


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    record = app.store.get_node_record(args.commit)
    if record is None:
        raise ArcError(f"Unknown commit: {args.commit}")
    if record.node.archived_at is not None and not args.recursive:
        raise ArcError(f"Node `{record.node.commit}` is already archived.")
    if record.node.parent is None:
        raise ArcError("Cannot archive the root baseline node.")
    if record.node.status == "running":
        raise ArcError("Cannot archive a running node.")
    main_commit = app.main_commit()

    if args.recursive:
        descendants = _collect_descendants(app, record.node.commit)
        targets = descendants + [record.node.commit]
        archived_at = utc_now_iso()
        count = 0
        skipped: list[str] = []
        for commit in targets:
            if commit == main_commit:
                skipped.append(f"{app.display_commit(commit)} (main)")
                continue
            node = app.store.get_node(commit)
            if node is None:
                continue
            if node.status == "running":
                skipped.append(f"{app.display_commit(commit)} (running)")
                continue
            if node.archived_at is not None:
                continue
            app.store.archive_node(commit, archived_at)
            count += 1
        print(f"Archived {count} node(s) under {app.display_commit(record.node.commit)} ({record.node.name})")
        if skipped:
            print(f"Skipped: {', '.join(skipped)}")
        return 0

    if main_commit == record.node.commit:
        raise ArcError("Cannot archive the current main node.")
    if app.store.has_children(record.node.commit):
        raise ArcError("Only leaf nodes can be archived. Use --recursive to archive the subtree.")

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
