from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from arc.commands.interrupt import run as run_interrupt
from arc.errors import ArcError
from arc.models import Node, NodeRecord
from arc.runlog import latest_modal_app_id, summarize_run_log


def _record(worktree: str) -> NodeRecord:
    return NodeRecord(
        node=Node(
            commit="a" * 40,
            parent="b" * 40,
            name="probe-run",
            status="running",
            hypothesis=None,
            analysis=None,
            worktree=worktree,
            created_at="2026-01-01T00:00:00Z",
            completed_at=None,
            verdict=None,
            archived_at=None,
        ),
        metrics={},
    )


class FakeStore:
    def __init__(self, record: NodeRecord) -> None:
        self.record = record

    def require_initialized(self) -> None:
        return None

    def get_node_record(self, commit_prefix: str) -> NodeRecord | None:
        if self.record.node.commit.startswith(commit_prefix):
            return self.record
        return None


class InterruptCommandTests(unittest.TestCase):
    def test_latest_modal_app_id_uses_latest_submission_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[2026-01-01T00:00:00.000Z] submitting Parameter Golf train via Modal",
                        "[2026-01-01T00:00:00.100Z] https://modal.com/apps/ws/main/ap-old123",
                        "[2026-01-01T00:01:00.000Z] modal run exited with code 1",
                        "[2026-01-01T00:02:00.000Z] submitting Parameter Golf train via Modal",
                        "[2026-01-01T00:02:00.100Z] https://modal.com/apps/ws/main/ap-new456",
                    ],
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(latest_modal_app_id(log_path), "ap-new456")

    def test_interrupt_stops_modal_app_and_marks_log_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp) / "worktree"
            worktree.mkdir()
            (worktree / ".env").write_text("MODAL_ENVIRONMENT=research\n", encoding="utf-8")
            log_path = worktree / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[2026-01-01T00:00:00.000Z] submitting Parameter Golf train via Modal",
                        "[2026-01-01T00:00:00.100Z] https://modal.com/apps/ws/main/ap-test123",
                    ],
                )
                + "\n",
                encoding="utf-8",
            )

            app = SimpleNamespace(
                store=FakeStore(_record(str(worktree.relative_to(Path(tmp))))),
                node_log_path=lambda node: worktree / "run.log",
                node_worktree_path=lambda node: worktree,
                relative_path=lambda path: str(path.relative_to(Path(tmp))),
                display_commit=lambda commit: commit[:12],
            )
            args = SimpleNamespace(commit="a" * 12)
            output = io.StringIO()

            with (
                mock.patch("arc.commands.interrupt.shutil.which", return_value="/usr/bin/modal"),
                mock.patch("arc.commands.interrupt.subprocess.run") as run_mock,
                contextlib.redirect_stdout(output),
            ):
                run_mock.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
                result = run_interrupt(app, args, [])

            self.assertEqual(result, 0)
            run_mock.assert_called_once()
            call_args = run_mock.call_args
            self.assertEqual(call_args.args[0], ["/usr/bin/modal", "app", "stop", "ap-test123"])
            self.assertEqual(call_args.kwargs["env"]["MODAL_ENVIRONMENT"], "research")
            self.assertEqual(summarize_run_log(log_path).state, "failed")
            self.assertIn("ap-test123", log_path.read_text(encoding="utf-8"))
            self.assertIn("Interrupted aaaaaaaaaaaa (probe-run)", output.getvalue())

    def test_interrupt_rejects_logs_without_app_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp) / "worktree"
            worktree.mkdir()
            log_path = worktree / "run.log"
            log_path.write_text(
                "[2026-01-01T00:00:00.000Z] submitting Parameter Golf train via Modal\n",
                encoding="utf-8",
            )

            app = SimpleNamespace(
                store=FakeStore(_record(str(worktree.relative_to(Path(tmp))))),
                node_log_path=lambda node: log_path,
                node_worktree_path=lambda node: worktree,
                relative_path=lambda path: str(path.relative_to(Path(tmp))),
                display_commit=lambda commit: commit[:12],
            )
            args = SimpleNamespace(commit="a" * 12)

            with self.assertRaises(ArcError):
                run_interrupt(app, args, [])


if __name__ == "__main__":
    unittest.main()
