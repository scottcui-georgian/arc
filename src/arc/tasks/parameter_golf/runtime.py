from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, Mapping

import yaml

from arc.errors import ArcError
from arc.executors.base import SubmitResult
from arc.timeutil import utc_now_iso

DEFAULT_GPU_TYPE = "H100"
DEFAULT_REMOTE_CPU = 8.0
DEFAULT_REMOTE_MEMORY_GB = 8.0
SUBMIT_MAX_WALLCLOCK_SECONDS = 180
SUBMIT_GRAD_ACCUM_STEPS = 1
SUBMIT_GRAD_ACCUM_ALLOWED = frozenset({1, 2, 4})
SUBMIT_CONFIG_CHOICES: tuple[str, ...] = ("proxy", "full")
SUBMIT_CONFIG_DIR = "configs"
ALLOWED_SUBMIT_GPU_TYPES = frozenset({"H100", "H100:8"})
MODAL_CONFIG_ENV_VAR = "ARC_PARAMETER_GOLF_MODAL_CONFIG"

_RUN_FORWARD_ENV_KEYS = frozenset(
    {
        "RUN_ID",
        "SEED",
        "DATA_PATH",
        "TOKENIZER_PATH",
        "ITERATIONS",
        "VAL_BATCH_SIZE",
        "VAL_LOSS_EVERY",
        "TRAIN_LOG_EVERY",
        "TRAIN_BATCH_TOKENS",
        "TRAIN_SEQ_LEN",
        "MAX_WALLCLOCK_SECONDS",
        "WARMUP_STEPS",
        "WARMDOWN_ITERS",
        "VOCAB_SIZE",
        "NUM_LAYERS",
        "MODEL_DIM",
        "NUM_HEADS",
        "NUM_KV_HEADS",
        "MLP_MULT",
        "TIE_EMBEDDINGS",
        "ROPE_BASE",
        "LOGIT_SOFTCAP",
        "QK_GAIN_INIT",
        "EMBED_LR",
        "HEAD_LR",
        "TIED_EMBED_LR",
        "TIED_EMBED_INIT_STD",
        "MATRIX_LR",
        "SCALAR_LR",
        "MUON_MOMENTUM",
        "MUON_BACKEND_STEPS",
        "MUON_MOMENTUM_WARMUP_START",
        "MUON_MOMENTUM_WARMUP_STEPS",
        "BETA1",
        "BETA2",
        "ADAM_EPS",
        "GRAD_CLIP_NORM",
        "GRAD_ACCUM_STEPS",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "MATCHED_FINEWEB_REPO_ID",
        "MATCHED_FINEWEB_REMOTE_ROOT_PREFIX",
    }
)
_SUBMIT_FORWARD_ENV_KEYS = frozenset(
    {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }
)


@dataclass(frozen=True)
class ParameterGolfSubmitConfig:
    """Resolved values from a worktree-local proxy.yaml / full.yaml."""

    name: str
    gpu_type: str
    train_wallclock: int
    env: dict[str, str] = field(default_factory=dict)
    modal_timeout: int | None = None
    cpu: float | None = None
    memory_gb: float | None = None


def _submit_config_path(repo_root: Path, name: str) -> Path:
    return repo_root / SUBMIT_CONFIG_DIR / f"{name}.yaml"


def load_submit_config(repo_root: Path, name: str) -> ParameterGolfSubmitConfig:
    """Load and validate ``configs/<name>.yaml`` from the worktree root.

    The YAML file must contain:
      gpu: "H100" | "H100:8"
      train_wallclock: <int seconds, 60..3600>
      env: { KEY: "value", ... }   # optional
      modal_timeout: <int seconds> # optional override for the Modal job timeout
    """
    if name not in SUBMIT_CONFIG_CHOICES:
        allowed = ", ".join(SUBMIT_CONFIG_CHOICES)
        raise ArcError(f"Unknown submit config `{name}`. Choose from: {allowed}.")
    path = _submit_config_path(repo_root, name)
    if not path.is_file():
        raise ArcError(
            f"Missing submit config at `{path.relative_to(repo_root)}`. "
            f"Each worktree must ship `{SUBMIT_CONFIG_DIR}/proxy.yaml` and "
            f"`{SUBMIT_CONFIG_DIR}/full.yaml`."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ArcError(f"Could not parse `{path}`: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ArcError(f"`{path}` must be a YAML mapping.")

    gpu_type = raw.get("gpu")
    if not isinstance(gpu_type, str) or gpu_type not in ALLOWED_SUBMIT_GPU_TYPES:
        allowed = ", ".join(sorted(ALLOWED_SUBMIT_GPU_TYPES))
        raise ArcError(
            f"`gpu` in `{path}` must be one of: {allowed}. Got: {gpu_type!r}."
        )

    wallclock_raw = raw.get("train_wallclock")
    if not isinstance(wallclock_raw, int) or isinstance(wallclock_raw, bool):
        raise ArcError(f"`train_wallclock` in `{path}` must be an integer seconds value.")
    if wallclock_raw < 60 or wallclock_raw > 3600:
        raise ArcError(
            f"`train_wallclock` in `{path}` must be between 60 and 3600 seconds."
        )

    env_raw = raw.get("env") or {}
    if not isinstance(env_raw, dict):
        raise ArcError(f"`env` in `{path}` must be a mapping of KEY: value.")
    env: dict[str, str] = {}
    for key, value in env_raw.items():
        if not isinstance(key, str):
            raise ArcError(f"`env` keys in `{path}` must be strings.")
        env[key] = str(value)

    modal_timeout = raw.get("modal_timeout")
    if modal_timeout is not None:
        if not isinstance(modal_timeout, int) or isinstance(modal_timeout, bool):
            raise ArcError(f"`modal_timeout` in `{path}` must be an integer seconds value.")
        if modal_timeout <= 0:
            raise ArcError(f"`modal_timeout` in `{path}` must be positive.")

    cpu_raw = raw.get("cpu")
    cpu: float | None = None
    if cpu_raw is not None:
        if isinstance(cpu_raw, bool) or not isinstance(cpu_raw, (int, float)):
            raise ArcError(f"`cpu` in `{path}` must be a positive number.")
        if cpu_raw <= 0:
            raise ArcError(f"`cpu` in `{path}` must be positive.")
        cpu = float(cpu_raw)

    memory_raw = raw.get("memory_gb")
    memory_gb: float | None = None
    if memory_raw is not None:
        if isinstance(memory_raw, bool) or not isinstance(memory_raw, (int, float)):
            raise ArcError(f"`memory_gb` in `{path}` must be a positive number.")
        if memory_raw <= 0:
            raise ArcError(f"`memory_gb` in `{path}` must be positive.")
        memory_gb = float(memory_raw)

    return ParameterGolfSubmitConfig(
        name=name,
        gpu_type=gpu_type,
        train_wallclock=int(wallclock_raw),
        env=env,
        modal_timeout=modal_timeout,
        cpu=cpu,
        memory_gb=memory_gb,
    )


def submit_gpu_count(gpu_type: str) -> int:
    """Parse an 'H100' / 'H100:8' spec into GPU count (matches modal_app)."""
    _, sep, suffix = gpu_type.rpartition(":")
    if not sep:
        return 1
    try:
        count = int(suffix)
    except ValueError:
        return 1
    return count if count > 0 else 1


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


def should_use_flash3(gpu_type: str | None) -> bool:
    return bool(gpu_type) and "H100" in gpu_type


@dataclass(frozen=True)
class ParameterGolfModalConfig:
    mode: Literal["run", "submit"]
    action: Literal["prepare", "train"]
    quiet: bool
    repo_root: str
    gpu_type: str
    cpu: float
    memory_gb: float
    train_entrypoint: str | None
    extra_args: list[str]
    run_id: str
    use_flash3: bool
    forwarded_env: dict[str, str]
    submission_outputs: bool = False
    modal_timeout: int | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def resolve_train_entrypoint(repo_root: Path, entrypoint: str) -> tuple[Path, str]:
    path = (repo_root / entrypoint).resolve()
    try:
        relative = path.relative_to(repo_root)
    except ValueError as exc:
        raise ArcError("Train entrypoint must be inside the repository.") from exc
    if path.suffix != ".py":
        raise ArcError("Train entrypoint must be a Python file.")
    if not path.is_file():
        raise ArcError(f"Train entrypoint does not exist: {entrypoint}")
    return path, relative.as_posix()


def submission_run_root_from_entrypoint(relative_entrypoint: str) -> str | None:
    """If the train script is under `.arc/submissions/<folder>/`, return `<folder>`."""
    normalized = relative_entrypoint.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if len(parts) < 4:
        return None
    if parts[0] != ".arc" or parts[1] != "submissions":
        return None
    folder = parts[2]
    if not folder or folder.endswith(".py"):
        return None
    return folder


def _env_text(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    text = value.strip()
    return text or None


def _env_positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _env_text(env, name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise ArcError(f"`{name}` must be a positive number.") from exc
    if parsed <= 0:
        raise ArcError(f"`{name}` must be greater than 0.")
    return parsed


def _forward_env(env: Mapping[str, str], keys: frozenset[str]) -> dict[str, str]:
    return {key: env[key] for key in keys if key in env}


class ParameterGolfModalRunner:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def run(
        self,
        action: str,
        action_args: list[str],
        *,
        quiet: bool,
        gpu: str | None = None,
        cpu: float | None = None,
        memory_gb: float | None = None,
    ) -> int:
        config = self._build_run_config(
            action,
            action_args,
            quiet=quiet,
            gpu=gpu,
            cpu=cpu,
            memory_gb=memory_gb,
        )
        cmd, env = self._build_invocation(config)
        proc = subprocess.run(cmd, cwd=str(self.repo_root), env=env, check=False)
        return proc.returncode

    def submit_train(
        self,
        log_path: Path,
        *,
        config_name: str,
        train_wallclock: int | None = None,
        grad_accum_steps: int | None = None,
    ) -> tuple[SubmitResult, ParameterGolfSubmitConfig]:
        submit_config = load_submit_config(self.repo_root, config_name)
        config = self._build_submit_train_config(
            submit_config=submit_config,
            train_wallclock=train_wallclock,
            grad_accum_steps=grad_accum_steps,
        )
        cmd, env = self._build_invocation(config)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[{utc_now_iso()}] submitting Parameter Golf train via Modal "
                f"(config={config_name}, gpu={config.gpu_type})\n"
            )
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
        return (
            SubmitResult(
                backend="parameter-golf-modal",
                log_path=log_path,
                process_id=process.pid,
            ),
            submit_config,
        )

    def _source_env(self) -> dict[str, str]:
        env = os.environ.copy()
        load_dotenv_into(env, self.repo_root / ".env")
        return env

    def _resolve_run_train_entrypoint(
        self,
        source_env: Mapping[str, str],
        action_args: list[str],
    ) -> tuple[str | None, list[str]]:
        normalized_args = normalize_action_args(action_args)
        if normalized_args:
            first_arg = normalized_args[0]
            if first_arg.endswith(".py"):
                _, relative_entrypoint = resolve_train_entrypoint(self.repo_root, first_arg)
                return relative_entrypoint, normalize_action_args(normalized_args[1:])
        entrypoint = _env_text(source_env, "ARC_PARAMETER_GOLF_TRAIN_ENTRYPOINT")
        if entrypoint is None:
            return None, normalized_args
        _, relative_entrypoint = resolve_train_entrypoint(self.repo_root, entrypoint)
        return relative_entrypoint, normalized_args

    def _build_run_config(
        self,
        action: str,
        action_args: list[str],
        *,
        quiet: bool,
        gpu: str | None = None,
        cpu: float | None = None,
        memory_gb: float | None = None,
    ) -> ParameterGolfModalConfig:
        valid_actions = {"prepare", "train"}
        if action not in valid_actions:
            valid = ", ".join(sorted(valid_actions))
            raise ArcError(f"Unknown Parameter Golf action `{action}`. Valid actions: {valid}")

        ensure_task_layout(self.repo_root)
        source_env = self._source_env()
        train_entrypoint: str | None = None
        extra_args = normalize_action_args(action_args)
        if action == "train":
            train_entrypoint, extra_args = self._resolve_run_train_entrypoint(source_env, action_args)
        submission_folder = (
            submission_run_root_from_entrypoint(train_entrypoint)
            if action == "train" and train_entrypoint
            else None
        )
        submission_outputs = submission_folder is not None
        gpu_type = gpu or _env_text(source_env, "ARC_PARAMETER_GOLF_GPU") or DEFAULT_GPU_TYPE
        cpu_value = cpu if cpu is not None else _env_positive_float(
            source_env,
            "ARC_PARAMETER_GOLF_CPU",
            DEFAULT_REMOTE_CPU,
        )
        memory_gb_value = memory_gb if memory_gb is not None else _env_positive_float(
            source_env,
            "ARC_PARAMETER_GOLF_MEMORY_GB",
            DEFAULT_REMOTE_MEMORY_GB,
        )
        if submission_outputs:
            assert submission_folder is not None
            run_id = submission_folder
        else:
            run_id = (
                _env_text(source_env, "ARC_PARAMETER_GOLF_RUN_ID")
                or _env_text(source_env, "RUN_ID")
                or self.repo_root.name
            )
        return ParameterGolfModalConfig(
            mode="run",
            action=action,
            quiet=quiet,
            repo_root=str(self.repo_root),
            gpu_type=gpu_type,
            cpu=cpu_value,
            memory_gb=memory_gb_value,
            train_entrypoint=train_entrypoint,
            extra_args=extra_args,
            run_id=run_id,
            use_flash3=should_use_flash3(gpu_type),
            forwarded_env=_forward_env(source_env, _RUN_FORWARD_ENV_KEYS),
            submission_outputs=submission_outputs,
        )

    def _build_submit_train_config(
        self,
        *,
        submit_config: ParameterGolfSubmitConfig,
        train_wallclock: int | None = None,
        grad_accum_steps: int | None = None,
    ) -> ParameterGolfModalConfig:
        ensure_task_layout(self.repo_root)
        source_env = self._source_env()
        # Forwarding order (lowest priority → highest): base source env (HF creds only),
        # YAML env block, then CLI overrides. CLI > YAML > .env.
        forwarded_env = _forward_env(source_env, _SUBMIT_FORWARD_ENV_KEYS)
        forwarded_env.update(submit_config.env)

        accum = SUBMIT_GRAD_ACCUM_STEPS if grad_accum_steps is None else grad_accum_steps
        if accum not in SUBMIT_GRAD_ACCUM_ALLOWED:
            allowed = ", ".join(str(v) for v in sorted(SUBMIT_GRAD_ACCUM_ALLOWED))
            raise ArcError(f"submit grad accumulation must be one of: {allowed}.")
        forwarded_env["GRAD_ACCUM_STEPS"] = str(accum)

        wallclock = (
            train_wallclock if train_wallclock is not None else submit_config.train_wallclock
        )
        forwarded_env["MAX_WALLCLOCK_SECONDS"] = str(wallclock)

        gpu_type = submit_config.gpu_type
        return ParameterGolfModalConfig(
            mode="submit",
            action="train",
            quiet=False,
            repo_root=str(self.repo_root),
            gpu_type=gpu_type,
            cpu=submit_config.cpu if submit_config.cpu is not None else DEFAULT_REMOTE_CPU,
            memory_gb=(
                submit_config.memory_gb
                if submit_config.memory_gb is not None
                else DEFAULT_REMOTE_MEMORY_GB
            ),
            train_entrypoint=None,
            extra_args=[],
            run_id=self.repo_root.name,
            use_flash3=should_use_flash3(gpu_type),
            forwarded_env=forwarded_env,
            submission_outputs=False,
            modal_timeout=submit_config.modal_timeout,
        )

    def _build_invocation(self, config: ParameterGolfModalConfig) -> tuple[list[str], dict[str, str]]:
        modal_path = require_cmd("modal")
        env = self._source_env()
        env.pop(MODAL_CONFIG_ENV_VAR, None)
        env[MODAL_CONFIG_ENV_VAR] = config.to_json()
        modal_app = Path(__file__).with_name("modal_app.py")
        cmd = [modal_path, "run"]
        if config.quiet:
            cmd.append("-q")
        cmd.append(str(modal_app))
        return cmd, env
