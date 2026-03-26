from __future__ import annotations

import os

from arc.errors import ArcError
from arc.executors.base import Executor
from arc.executors.noop import NoopExecutor


def load_executor() -> Executor:
    name = os.environ.get("ARC_EXECUTOR", "noop")
    if name == "noop":
        return NoopExecutor()
    raise ArcError(f"Unknown executor backend: {name}")
