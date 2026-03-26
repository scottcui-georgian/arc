from __future__ import annotations

import subprocess
from pathlib import Path

from arc.errors import ArcError


def _run_git(
    repo_root: Path,
    *args: str,
    cwd: Path | None = None,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd or repo_root),
        check=False,
        capture_output=capture,
        text=True,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "git command failed"
        raise ArcError(detail)
    return completed


def short_rev(repo_root: Path, rev: str, cwd: Path | None = None) -> str:
    result = _run_git(repo_root, "rev-parse", "--short=7", rev, cwd=cwd)
    return result.stdout.strip()


def current_head(repo_root: Path) -> str:
    if not rev_exists(repo_root, "HEAD"):
        raise ArcError("Repository has no commits yet; create an initial commit before `arc init`.")
    return short_rev(repo_root, "HEAD")


def rev_exists(repo_root: Path, rev: str, cwd: Path | None = None) -> bool:
    result = _run_git(repo_root, "rev-parse", "--verify", rev, cwd=cwd, check=False)
    return result.returncode == 0


def commit_parent(repo_root: Path, rev: str) -> str | None:
    result = _run_git(repo_root, "rev-list", "--parents", "-n", "1", rev)
    parts = result.stdout.strip().split()
    if len(parts) <= 1:
        return None
    return short_rev(repo_root, parts[1])


def create_worktree(repo_root: Path, parent: str, branch: str, worktree: Path) -> None:
    _run_git(
        repo_root,
        "worktree",
        "add",
        "-b",
        branch,
        str(worktree),
        parent,
        capture=True,
    )


def commit_all(repo_root: Path, worktree: Path, message: str) -> str:
    _run_git(repo_root, "add", "-A", cwd=worktree)
    _run_git(repo_root, "commit", "-m", message, cwd=worktree)
    return short_rev(repo_root, "HEAD", cwd=worktree)


def current_branch(repo_root: Path, cwd: Path | None = None) -> str:
    result = _run_git(repo_root, "branch", "--show-current", cwd=cwd)
    branch = result.stdout.strip()
    if not branch:
        raise ArcError("Detached HEAD; expected a branch.")
    return branch


def try_fast_forward_main(repo_root: Path, target_commit: str) -> bool:
    has_main = _run_git(
        repo_root,
        "show-ref",
        "--verify",
        "--quiet",
        "refs/heads/main",
        check=False,
        capture=True,
    )
    if has_main.returncode != 0:
        return False

    main_sha = short_rev(repo_root, "refs/heads/main")
    is_ancestor = _run_git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        main_sha,
        target_commit,
        check=False,
        capture=True,
    )
    if is_ancestor.returncode != 0:
        return False

    try:
        branch = _run_git(
            repo_root,
            "branch",
            "--show-current",
            check=False,
        ).stdout.strip()
        # `git branch -f main <rev>` fails when `main` is checked out in this worktree
        # (Git refuses to force-move the branch you are on). Reset moves HEAD and the branch.
        if branch == "main":
            _run_git(repo_root, "reset", "--hard", target_commit)
        else:
            _run_git(repo_root, "branch", "-f", "main", target_commit)
    except ArcError:
        return False
    return True
