from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from arc.commands.hyp import run as run_hyp
from arc.commands.unhyp import run as run_unhyp
from arc.errors import ArcError


class HypothesisCommandTests(unittest.TestCase):
    def test_hyp_saves_pending_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            saved: list[tuple[str, str]] = []

            class FakeStore:
                def require_initialized(self) -> None:
                    return None

            class FakeApp:
                store = FakeStore()

                def save_hypothesis(self, name: str, text: str) -> Path:
                    saved.append((name, text))
                    return Path(tmp) / f"{name}.md"

            output = io.StringIO()
            args = SimpleNamespace(name="test-hyp", text="hello world")

            with contextlib.redirect_stdout(output):
                result = run_hyp(FakeApp(), args, [])

            self.assertEqual(result, 0)
            self.assertEqual(saved, [("test-hyp", "hello world")])
            self.assertIn("Hypothesis saved for test-hyp.", output.getvalue())

    def test_unhyp_removes_pending_hypothesis(self) -> None:
        removed: list[str] = []

        class FakeStore:
            def require_initialized(self) -> None:
                return None

        class FakeApp:
            store = FakeStore()

            def consume_hypothesis(self, name: str) -> str:
                removed.append(name)
                return "old hypothesis"

        output = io.StringIO()
        args = SimpleNamespace(name="test-hyp")

        with contextlib.redirect_stdout(output):
            result = run_unhyp(FakeApp(), args, [])

        self.assertEqual(result, 0)
        self.assertEqual(removed, ["test-hyp"])
        self.assertIn("Hypothesis removed for test-hyp.", output.getvalue())

    def test_unhyp_rejects_invalid_names(self) -> None:
        class FakeStore:
            def require_initialized(self) -> None:
                return None

        class FakeApp:
            store = FakeStore()

            def consume_hypothesis(self, name: str) -> str:
                raise AssertionError("should not be called")

        args = SimpleNamespace(name="Bad Name")
        with self.assertRaises(ArcError):
            run_unhyp(FakeApp(), args, [])


if __name__ == "__main__":
    unittest.main()
