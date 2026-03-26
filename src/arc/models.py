from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Status = Literal["committed", "running", "completed", "failed"]
Direction = Literal["min", "max"]
Verdict = Literal["promising", "unsupported"]
RemoteRunState = Literal["missing", "running", "finished", "failed"]


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
    verdict: Verdict | None
    archived_at: str | None


@dataclass(frozen=True)
class NodeRecord:
    node: Node
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteRunRecord:
    record: NodeRecord
    state: RemoteRunState
    log_path: str
    metrics: dict[str, float] = field(default_factory=dict)
