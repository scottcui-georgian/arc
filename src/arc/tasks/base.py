from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from arc.executors.base import SubmitResult
from arc.models import Node
from arc.text import format_float

if TYPE_CHECKING:
    from arc.models import NodeRecord


@dataclass(frozen=True)
class TaskModule:
    name: str

    def register_commands(self, subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
        del subparsers

    def format_metric(self, name: str, value: float) -> str:
        del name
        return format_float(value)

    def process_result_metrics(
        self,
        node: Node,
        *,
        verdict: str,
        metrics: dict[str, float],
        completed_at: str,
    ) -> tuple[str, dict[str, float], list[str]]:
        del node, completed_at
        return verdict, dict(metrics), []

    def derive_result_metrics(
        self,
        node: Node,
        log_path: Path,
    ) -> tuple[dict[str, float], list[str]]:
        del node, log_path
        return {}, []

    def tree_metric_suffix(self, record: NodeRecord, *, metric_name: str | None) -> str:
        if metric_name and metric_name in record.metrics:
            return f" ({self.format_metric(metric_name, record.metrics[metric_name])})"
        return ""

    def submit(self, node: Node, worktree_root: Path, log_path: Path) -> SubmitResult | None:
        del node, worktree_root, log_path
        return None
