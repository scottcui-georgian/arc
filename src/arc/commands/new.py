from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.git import create_worktree
from arc.paths import validate_name, worktree_dir_name, worktree_stamp


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("parent", help="Parent commit hash or `main`.")
    parser.add_argument("name", help="Experiment name.")


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    name = validate_name(args.name)
    parent_commit = app.resolve_main_or_commit(args.parent)
    parent_record = app.store.get_node_record(parent_commit)
    if parent_record is None:
        raise ArcError(f"Unknown parent node: {args.parent}")

    app.ensure_directories()
    dirname = worktree_dir_name(name)
    worktree_path = app.paths.worktrees_dir / dirname
    if worktree_path.exists():
        raise ArcError(f"Worktree already exists: {app.relative_path(worktree_path)}")

    branch_name = f"arc/{worktree_stamp()}-{name}"
    create_worktree(app.paths.repo_root, parent_record.node.commit, branch_name, worktree_path)

    print(f"Created worktree: {app.relative_path(worktree_path)}")
    print(f"Parent: {app.display_commit(parent_record.node.commit)}")
    print("")
    print(f"cd {app.relative_path(worktree_path)} to start working.")
    return 0


COMMAND = CommandSpec(
    name="new",
    help="Create a new worktree branching from a parent node.",
    register=register,
    run=run,
)
