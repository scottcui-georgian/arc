from __future__ import annotations

import unittest

from arc.models import Node, NodeRecord
from arc.render import render_tree


def _record(commit: str, parent: str | None, name: str) -> NodeRecord:
    return NodeRecord(
        node=Node(
            commit=commit,
            parent=parent,
            name=name,
            status="committed",
            hypothesis=None,
            analysis=None,
            worktree=".",
            created_at=f"2026-01-01T00:00:0{len(name)}Z",
            completed_at=None,
            verdict=None,
            archived_at=None,
        ),
        metrics={},
    )


class RenderTreeTests(unittest.TestCase):
    def test_single_child_chain_uses_unary_marker(self) -> None:
        records = [
            _record("a" * 12, None, "root"),
            _record("b" * 12, "a" * 12, "child"),
            _record("c" * 12, "b" * 12, "grandchild"),
        ]

        tree = render_tree(
            records,
            metric_name=None,
            direction="min",
            main_commit=None,
            status_filter=None,
            archived_only=False,
            depth=None,
            leaves_only=False,
        )

        self.assertEqual(
            tree,
            "\n".join(
                [
                    "• aaaaaaaaaaaa root",
                    "↳ • bbbbbbbbbbbb child",
                    "↳ • cccccccccccc grandchild",
                ]
            ),
        )

    def test_single_child_chain_keeps_parentage_clear_with_siblings(self) -> None:
        records = [
            _record("a" * 12, None, "root"),
            _record("b" * 12, "a" * 12, "child"),
            _record("c" * 12, "b" * 12, "grandchild"),
            _record("d" * 12, "a" * 12, "sibling"),
        ]

        tree = render_tree(
            records,
            metric_name=None,
            direction="min",
            main_commit=None,
            status_filter=None,
            archived_only=False,
            depth=None,
            leaves_only=False,
        )

        self.assertEqual(
            tree,
            "\n".join(
                [
                    "• aaaaaaaaaaaa root",
                    "├── • bbbbbbbbbbbb child",
                    "│   ↳ • cccccccccccc grandchild",
                    "└── • dddddddddddd sibling",
                ]
            ),
        )

    def test_branching_after_compressed_chain_reindents_children(self) -> None:
        records = [
            _record("a" * 12, None, "root"),
            _record("b" * 12, "a" * 12, "child"),
            _record("c" * 12, "b" * 12, "branch"),
            _record("d" * 12, "c" * 12, "left"),
            _record("e" * 12, "c" * 12, "right"),
        ]

        tree = render_tree(
            records,
            metric_name=None,
            direction="min",
            main_commit=None,
            status_filter=None,
            archived_only=False,
            depth=None,
            leaves_only=False,
        )

        self.assertEqual(
            tree,
            "\n".join(
                [
                    "• aaaaaaaaaaaa root",
                    "↳ • bbbbbbbbbbbb child",
                    "↳ • cccccccccccc branch",
                    "    ├── • dddddddddddd left",
                    "    └── • eeeeeeeeeeee right",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
