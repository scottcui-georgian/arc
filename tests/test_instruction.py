from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from arc.commands.instruction import run


class InstructionCommandTests(unittest.TestCase):
    def test_parameter_golf_base_program_instruction(self) -> None:
        output = io.StringIO()
        args = SimpleNamespace(task="parameter_golf", name="base_program")

        with contextlib.redirect_stdout(output):
            run(SimpleNamespace(), args, [])

        self.assertIn("# Base program", output.getvalue())

    def test_parameter_golf_named_instruction(self) -> None:
        output = io.StringIO()
        args = SimpleNamespace(task="parameter_golf", name="create_submission")
        seen_names: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instruction.md"
            path.write_text("named instruction\n", encoding="utf-8")

            class FakeTask:
                def instruction_path(self, name: str):
                    seen_names.append(name)
                    return path

            with patch("arc.commands.instruction.load_task_module", return_value=FakeTask()):
                with contextlib.redirect_stdout(output):
                    run(SimpleNamespace(), args, [])

        self.assertEqual(seen_names, ["create_submission"])
        self.assertIn("named instruction", output.getvalue())


if __name__ == "__main__":
    unittest.main()
