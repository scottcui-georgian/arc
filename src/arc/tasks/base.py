from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from arc.executors.base import SubmitResult
from arc.models import Node
from arc.text import format_float


@dataclass(frozen=True)
class TaskModule:
    name: str

    def register_commands(self, subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
        del subparsers

    def format_metric(self, name: str, value: float) -> str:
        del name
        return format_float(value)

    def submit(self, node: Node, worktree_root: Path, log_path: Path) -> SubmitResult | None:
        del node, worktree_root, log_path
        return None
