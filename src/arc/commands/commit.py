from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.git import commit_all, commit_parent
from arc.models import Node
from arc.paths import validate_name
from arc.timeutil import utc_now_iso


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", help="Experiment name.")


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    name = validate_name(args.name)
    worktree = app.resolve_worktree_by_name(name)
    hypothesis = app.read_hypothesis(name)
    commit = commit_all(app.paths.repo_root, worktree, name)
    parent = commit_parent(app.paths.repo_root, commit)
    if parent is None:
        raise ArcError("Expected a parent commit for experiment commit.")
    if app.store.get_node_record(parent) is None:
        raise ArcError(f"Parent commit {parent} is not tracked by arc.")

    if app.store.get_node(commit) is not None:
        raise ArcError(f"Commit {commit} is already tracked.")

    app.store.insert_node(
        Node(
            commit=commit,
            parent=parent,
            name=name,
            status="committed",
            hypothesis=hypothesis,
            analysis=None,
            worktree=app.relative_path(worktree),
            created_at=utc_now_iso(),
            completed_at=None,
        )
    )
    app.hypothesis_path(name).unlink()

    print(f"Committed: {commit}")
    print(f"Parent:    {parent}")
    print(f"Name:      {name}")
    print("Status:    committed")
    return 0


COMMAND = CommandSpec(
    name="commit",
    help="Commit a worktree and create a node in the database.",
    register=register,
    run=run,
)
