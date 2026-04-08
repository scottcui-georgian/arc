from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace

from arc.commands.rehyp import run as run_rehyp
from arc.commands.rename import run as run_rename
from arc.errors import ArcError
from arc.models import Node, NodeRecord


def _record(*, name: str = "old-name", hypothesis: str | None = "old hypothesis") -> NodeRecord:
    return NodeRecord(
        node=Node(
            commit="a" * 40,
            parent="b" * 40,
            name=name,
            status="completed",
            hypothesis=hypothesis,
            analysis=None,
            worktree=".arc/worktrees/example",
            created_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:10:00Z",
            verdict="promising",
            archived_at=None,
        ),
        metrics={},
    )


class FakeStore:
    def __init__(self, record: NodeRecord) -> None:
        self.record = record
        self.updated: list[tuple[str, dict[str, object]]] = []

    def require_initialized(self) -> None:
        return None

    def get_node_record(self, commit_prefix: str) -> NodeRecord | None:
        if self.record.node.commit.startswith(commit_prefix):
            return self.record
        return None

    def update_node(self, commit: str, **kwargs: object) -> None:
        self.updated.append((commit, kwargs))


class EditCommandTests(unittest.TestCase):
    def test_rename_updates_node_name(self) -> None:
        store = FakeStore(_record())
        app = SimpleNamespace(store=store, display_commit=lambda commit: commit[:12])
        args = SimpleNamespace(commit="a" * 12, name="half-batch-focal")
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = run_rename(app, args, [])

        self.assertEqual(result, 0)
        self.assertEqual(
            store.updated,
            [("a" * 40, {"name": "half-batch-focal"})],
        )
        self.assertIn("Name: old-name → half-batch-focal", output.getvalue())

    def test_rename_rejects_invalid_names(self) -> None:
        store = FakeStore(_record())
        app = SimpleNamespace(store=store, display_commit=lambda commit: commit[:12])
        args = SimpleNamespace(commit="a" * 12, name="Bad Name")

        with self.assertRaises(ArcError):
            run_rename(app, args, [])

        self.assertEqual(store.updated, [])

    def test_rehyp_updates_hypothesis(self) -> None:
        store = FakeStore(_record())
        app = SimpleNamespace(store=store, display_commit=lambda commit: commit[:12])
        args = SimpleNamespace(commit="a" * 12, text="new hypothesis")
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = run_rehyp(app, args, [])

        self.assertEqual(result, 0)
        self.assertEqual(
            store.updated,
            [("a" * 40, {"hypothesis": "new hypothesis"})],
        )
        self.assertIn("Previous hypothesis: old hypothesis", output.getvalue())
        self.assertIn("New hypothesis: new hypothesis", output.getvalue())

    def test_rehyp_reports_unchanged_text(self) -> None:
        store = FakeStore(_record(hypothesis="same text"))
        app = SimpleNamespace(store=store, display_commit=lambda commit: commit[:12])
        args = SimpleNamespace(commit="a" * 12, text="same text")
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = run_rehyp(app, args, [])

        self.assertEqual(result, 0)
        self.assertEqual(store.updated, [])
        self.assertIn("Hypothesis unchanged", output.getvalue())


if __name__ == "__main__":
    unittest.main()
