from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from arc.models import Node
from arc.timeutil import utc_now_iso


@dataclass(frozen=True)
class SubmitResult:
    backend: str
    log_path: Path
    process_id: int | None = None


class Executor:
    name = "base"

    def submit(self, node: Node, log_path: Path) -> SubmitResult:
        raise NotImplementedError

    def append_log(self, log_path: Path, message: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now_iso()}] {message}\n")
