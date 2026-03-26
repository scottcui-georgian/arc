from __future__ import annotations

from arc.tasks.base import TaskModule


class DefaultTaskModule(TaskModule):
    def __init__(self) -> None:
        super().__init__(name="default")
