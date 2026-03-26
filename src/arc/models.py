from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Status = Literal["committed", "running", "completed", "failed"]
Direction = Literal["min", "max"]


@dataclass(frozen=True)
class Node:
    commit: str
    parent: str | None
    name: str
    status: Status
    hypothesis: str | None
    analysis: str | None
    worktree: str
    created_at: str
    completed_at: str | None


@dataclass(frozen=True)
class NodeRecord:
    node: Node
    metrics: dict[str, float] = field(default_factory=dict)
