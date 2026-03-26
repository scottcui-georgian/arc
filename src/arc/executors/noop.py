from __future__ import annotations

from pathlib import Path

from arc.executors.base import Executor, SubmitResult
from arc.models import Node


class NoopExecutor(Executor):
    name = "noop"

    def submit(self, node: Node, log_path: Path) -> SubmitResult:
        self.append_log(log_path, f"Submitted {node.commit} ({node.name}) via noop executor.")
        return SubmitResult(backend=self.name, log_path=log_path)
