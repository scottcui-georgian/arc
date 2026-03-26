from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from arc.errors import ArcError

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class ArcPaths:
    repo_root: Path
    arc_dir: Path
    db_path: Path
    hypotheses_dir: Path
    worktrees_dir: Path


def discover_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    result = subprocess.run(
        [
            "git",
            "-C",
            str(current),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ArcError("Not inside a git repository.")

    common_dir = Path(result.stdout.strip())
    if common_dir.name != ".git":
        raise ArcError("Could not resolve the repository root.")
    return common_dir.parent


def build_paths(start: Path | None = None) -> ArcPaths:
    repo_root = discover_repo_root(start)
    arc_dir = repo_root / ".arc"
    return ArcPaths(
        repo_root=repo_root,
        arc_dir=arc_dir,
        db_path=arc_dir / "arc.db",
        hypotheses_dir=arc_dir / "hypotheses",
        worktrees_dir=arc_dir / "worktrees",
    )


def validate_name(name: str) -> str:
    if not NAME_PATTERN.fullmatch(name):
        raise ArcError(
            "Invalid experiment name. Use lowercase letters, numbers, '.', '_' or '-'."
        )
    return name


def worktree_stamp(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.strftime("%Y%m%d")


def worktree_dir_name(name: str, now: datetime | None = None) -> str:
    return f"{worktree_stamp(now)}-{validate_name(name)}"


def relative_to_repo(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)
