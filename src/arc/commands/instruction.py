from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.tasks.registry import load_task_module


def register(parser: argparse.ArgumentParser) -> None:
    parser.description = "Print a task's bundled instruction file."
    parser.add_argument("task", help="Task module name, for example `parameter_golf`.")
    parser.add_argument(
        "name",
        nargs="?",
        default="base_program",
        help="Instruction name within that task. Defaults to `base_program`.",
    )


def _read_instruction_path(task_name: str, instruction_name: str) -> Path:
    task = load_task_module(task_name)
    path = task.instruction_path(instruction_name)
    if path is None:
        raise ArcError(
            f"Task module `{task_name}` does not provide instruction `{instruction_name}`."
        )
    if not path.is_file():
        raise ArcError(f"Instruction file not found: {path}")
    return path


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    del app
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")

    path = _read_instruction_path(args.task, args.name)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArcError(f"Failed to read instruction file: {path}") from exc
    sys.stdout.write(text)
    return 0


COMMAND = CommandSpec(
    name="instruction",
    help="Print a task's bundled instruction text.",
    register=register,
    run=run,
)
