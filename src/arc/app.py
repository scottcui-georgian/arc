from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from arc.errors import ArcError
from arc.models import Direction
from arc.paths import ArcPaths, build_paths, relative_to_repo, validate_name
from arc.store import ArcStore
from arc.tasks.registry import infer_task_module_name, load_task_module


@dataclass
class ArcApp:
    paths: ArcPaths
    store: ArcStore
    task: object

    @classmethod
    def discover(cls, start: Path | None = None) -> ArcApp:
        paths = build_paths(start)
        store = ArcStore(paths.db_path)
        task_name = os.environ.get("ARC_TASK")
        if task_name is None and store.exists():
            task_name = store.get_meta("task")
        if task_name is None:
            task_name = infer_task_module_name(paths.repo_root)
        task = load_task_module(task_name)
        return cls(paths=paths, store=store, task=task)

    def ensure_directories(self) -> None:
        self.paths.arc_dir.mkdir(parents=True, exist_ok=True)
        self.paths.hypotheses_dir.mkdir(parents=True, exist_ok=True)
        self.paths.worktrees_dir.mkdir(parents=True, exist_ok=True)

    def main_metric(self) -> str | None:
        return self.store.get_meta("main_metric")

    def metric_direction(self) -> Direction:
        value = self.store.get_meta("main_metric_direction")
        return "max" if value == "max" else "min"

    def main_commit(self) -> str | None:
        return self.store.get_meta("main")

    def hypothesis_path(self, name: str) -> Path:
        return self.paths.hypotheses_dir / f"{validate_name(name)}.md"

    def save_hypothesis(self, name: str, text: str) -> Path:
        self.ensure_directories()
        path = self.hypothesis_path(name)
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return path

    def read_hypothesis(self, name: str) -> str:
        path = self.hypothesis_path(name)
        if not path.exists():
            raise ArcError(f"No hypothesis saved for `{name}`.")
        return path.read_text(encoding="utf-8").strip()

    def consume_hypothesis(self, name: str) -> str:
        text = self.read_hypothesis(name)
        self.hypothesis_path(name).unlink()
        return text

    def list_hypotheses(self) -> list[tuple[str, str]]:
        if not self.paths.hypotheses_dir.exists():
            return []
        items: list[tuple[str, str]] = []
        for path in sorted(self.paths.hypotheses_dir.glob("*.md")):
            items.append((path.stem, path.read_text(encoding="utf-8").strip()))
        return items

    def relative_path(self, path: Path) -> str:
        return relative_to_repo(self.paths.repo_root, path)

    def resolve_main_or_commit(self, value: str) -> str:
        if value == "main":
            main = self.main_commit()
            if main is None:
                raise ArcError("No promoted main commit is set.")
            return main
        record = self.store.get_node_record(value)
        if record is not None:
            return record.node.commit
        return value

    def resolve_worktree_by_name(self, name: str) -> Path:
        validate_name(name)
        if not self.paths.worktrees_dir.exists():
            raise ArcError("No worktrees have been created yet.")
        matches = sorted(self.paths.worktrees_dir.glob(f"*-{name}"))
        if not matches:
            raise ArcError(f"No worktree found for `{name}`.")
        if len(matches) > 1:
            raise ArcError(f"Multiple worktrees match `{name}`; clean up older copies first.")
        return matches[0]
