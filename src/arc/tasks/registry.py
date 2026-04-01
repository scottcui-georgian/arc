from __future__ import annotations

import importlib
from pathlib import Path

from arc.errors import ArcError
from arc.tasks.base import TaskModule
from arc.tasks.parameter_golf.runtime import is_parameter_golf_repo


def load_task_module(name: str) -> TaskModule:
    module_name = "arc.tasks.default" if name == "default" else f"arc.tasks.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise ArcError(
                f"Failed to import task module `{name}`: missing dependency `{exc.name}`."
            ) from exc
        raise ArcError(f"Unknown task module: {name}") from exc

    if name == "default":
        return module.DefaultTaskModule()

    factory = getattr(module, "build_task_module", None)
    if factory is None:
        raise ArcError(f"Task module `{name}` must define build_task_module().")
    task = factory()
    if not isinstance(task, TaskModule):
        raise ArcError(f"Task module `{name}` returned an invalid task module.")
    return task


def infer_task_module_name(repo_root: Path) -> str:
    if is_parameter_golf_repo(repo_root):
        return "parameter_golf"
    return "default"
