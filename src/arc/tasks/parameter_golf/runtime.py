from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from arc.errors import ArcError
from arc.executors.base import SubmitResult
from arc.timeutil import utc_now_iso


@dataclass(frozen=True)
class ParameterGolfLayout:
    pyproject_file: Path
    train_file: Path
    prepare_file: Path | None
    downloader_file: Path | None


def detect_task_layout(repo_root: Path) -> ParameterGolfLayout | None:
    pyproject_file = repo_root / "pyproject.toml"
    train_candidates = [
        repo_root / "train_gpt.py",
        repo_root / "workspace" / "train_gpt.py",
    ]
    prepare_candidates = [
        repo_root / "prepare.py",
        repo_root / "workspace" / "prepare.py",
    ]
    downloader_candidates = [
        repo_root / "data" / "cached_challenge_fineweb.py",
    ]

    if not pyproject_file.is_file():
        return None
    train_file = next((path for path in train_candidates if path.is_file()), None)
    if train_file is None:
        return None
    prepare_file = next((path for path in prepare_candidates if path.is_file()), None)
    downloader_file = next((path for path in downloader_candidates if path.is_file()), None)
    if prepare_file is None and downloader_file is None:
        return None
    return ParameterGolfLayout(
        pyproject_file=pyproject_file,
        train_file=train_file,
        prepare_file=prepare_file,
        downloader_file=downloader_file,
    )


def is_parameter_golf_repo(repo_root: Path) -> bool:
    return detect_task_layout(repo_root) is not None


def ensure_task_layout(repo_root: Path) -> ParameterGolfLayout:
    layout = detect_task_layout(repo_root)
    if layout is None:
        raise ArcError(
            "Parameter Golf task layout is incomplete. Expected `train_gpt.py` or "
            "`workspace/train_gpt.py`, a repo-root `pyproject.toml`, plus a prepare "
            "entrypoint or downloader."
        )
    return layout


def require_cmd(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ArcError(f"`{name}` is not on PATH.")
    return path


def load_dotenv_into(env: dict[str, str], dotenv_path: Path) -> None:
    if not dotenv_path.is_file():
        return
    try:
        text = dotenv_path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in env:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        env[key] = value


def normalize_action_args(args: list[str]) -> list[str]:
    if args[:1] == ["--"]:
        return args[1:]
    return args


class ParameterGolfModalRunner:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def run(self, action: str, action_args: list[str], *, quiet: bool) -> int:
        cmd, env = self._build_invocation(action, action_args, quiet=quiet)
        proc = subprocess.run(cmd, cwd=str(self.repo_root), env=env, check=False)
        return proc.returncode

    def submit_train(self, log_path: Path) -> SubmitResult:
        cmd, env = self._build_invocation("train", [], quiet=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now_iso()}] submitting Parameter Golf train via Modal\n")
            handle.flush()

        log_wrapper = Path(__file__).with_name("log_wrapper.py")
        wrapper_cmd = [
            sys.executable,
            str(log_wrapper),
            "--log-path",
            str(log_path),
            "--cwd",
            str(self.repo_root),
            "--",
            *cmd,
        ]
        process = subprocess.Popen(
            wrapper_cmd,
            cwd=str(self.repo_root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return SubmitResult(
            backend="parameter-golf-modal",
            log_path=log_path,
            process_id=process.pid,
        )

    def _build_invocation(
        self,
        action: str,
        action_args: list[str],
        *,
        quiet: bool,
    ) -> tuple[list[str], dict[str, str]]:
        valid_actions = {"prepare", "train"}
        if action not in valid_actions:
            valid = ", ".join(sorted(valid_actions))
            raise ArcError(f"Unknown Parameter Golf action `{action}`. Valid actions: {valid}")

        ensure_task_layout(self.repo_root)
        modal_path = require_cmd("modal")
        env = os.environ.copy()
        load_dotenv_into(env, self.repo_root / ".env")
        env["ARC_PARAMETER_GOLF_REPO_ROOT"] = str(self.repo_root)
        env["ARC_PARAMETER_GOLF_RUN_ID"] = self.repo_root.name
        env["ARC_PARAMETER_GOLF_ACTION_ARGS"] = json.dumps(normalize_action_args(action_args))
        env["ARC_PARAMETER_GOLF_QUIET"] = "1" if quiet else "0"

        modal_app = Path(__file__).with_name("modal_app.py")
        cmd = [modal_path, "run"]
        if quiet:
            cmd.append("-q")
        cmd.extend([str(modal_app), "--action", action])
        return cmd, env
