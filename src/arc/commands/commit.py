from __future__ import annotations

from arc.app import ArcApp
from arc.errors import ArcError
from arc.git import commit_all, commit_parent
from arc.models import Node, NodeRecord
from arc.paths import validate_name
from arc.timeutil import utc_now_iso


def commit_worktree(app: ArcApp, name: str) -> NodeRecord:
    app.store.require_initialized()

    name = validate_name(name)
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
            verdict=None,
            archived_at=None,
        )
    )
    app.hypothesis_path(name).unlink()

    record = app.store.get_node_record(commit)
    if record is None:
        raise ArcError(f"Commit {commit} was created but could not be reloaded from arc.")
    return record
