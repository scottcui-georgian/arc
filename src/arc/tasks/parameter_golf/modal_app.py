"""
Modal backend for the Parameter Golf task.

This module is launched via `modal run`, but it must remain self-contained:
the remote container should depend only on task files and task dependencies.
"""

from __future__ import annotations

import codecs
import json
import os
import selectors
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import modal

APP_NAME = "autoresearch-parameter-golf"
DEFAULT_GPU_TYPE = "H100"
DEFAULT_REMOTE_CPU = 8.0
DEFAULT_REMOTE_MEMORY_GB = 8.0
MODAL_CONFIG_ENV_VAR = "ARC_PARAMETER_GOLF_MODAL_CONFIG"
FLASH_ATTENTION_3_WHEEL_INDEX = (
    "https://windreamer.github.io/flash-attention3-wheels/cu128_torch291"
)
UV_PYTHON = "/.uv/.venv/bin/python"
REMOTE_TASK_DIR = "/root/task"
VOLUME_ROOT = "/cache-home"
VOLUME_RUNS_ROOT = f"{VOLUME_ROOT}/parameter-golf-runs"
VOLUME_SUBMISSIONS_ROOT = f"{VOLUME_ROOT}/parameter-golf-submissions"


@dataclass(frozen=True)
class ModalLaunchConfig:
    mode: Literal["run", "submit"]
    action: Literal["prepare", "train"]
    quiet: bool
    repo_root: Path
    gpu_type: str
    cpu: float
    memory_gb: float
    train_entrypoint: str | None
    extra_args: list[str]
    run_id: str
    use_flash3: bool
    forwarded_env: dict[str, str]
    submission_outputs: bool
    modal_timeout: int | None = None


def _payload_positive_float(payload: dict[str, Any], name: str, default: float) -> float:
    raw = payload.get(name)
    if raw is None:
        return default
    if isinstance(raw, bool):
        raise RuntimeError(f"{name} must be a positive number.")
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive number.") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be greater than 0.")
    return parsed


def _payload_string(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a string.")
    text = value.strip()
    if not text:
        raise RuntimeError(f"{name} must not be empty.")
    return text


def _payload_optional_string(payload: dict[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a string when provided.")
    text = value.strip()
    return text or None


def _payload_bool(payload: dict[str, Any], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise RuntimeError(f"{name} must be a boolean.")
    return value


def _payload_bool_default(payload: dict[str, Any], name: str, default: bool) -> bool:
    value = payload.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise RuntimeError(f"{name} must be a boolean.")
    return value


def _payload_choice(
    payload: dict[str, Any],
    name: str,
    valid_values: set[str],
) -> str:
    value = _payload_string(payload, name)
    if value not in valid_values:
        valid = ", ".join(sorted(valid_values))
        raise RuntimeError(f"{name} must be one of: {valid}.")
    return value


def _payload_string_list(payload: dict[str, Any], name: str) -> list[str]:
    value = payload.get(name)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"{name} must be a JSON array of strings.")
    return value


def _payload_env_dict(payload: dict[str, Any], name: str) -> dict[str, str]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object of string values.")
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise RuntimeError(f"{name} must contain only string keys and string values.")
    return dict(value)


def _load_modal_config_from_env() -> ModalLaunchConfig:
    raw = os.environ.get(MODAL_CONFIG_ENV_VAR)
    if not raw:
        raise RuntimeError(f"{MODAL_CONFIG_ENV_VAR} is not set.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid {MODAL_CONFIG_ENV_VAR} payload.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{MODAL_CONFIG_ENV_VAR} must decode to a JSON object.")

    mode = _payload_choice(payload, "mode", {"run", "submit"})
    action = _payload_choice(payload, "action", {"prepare", "train"})
    if mode == "submit" and action != "train":
        raise RuntimeError("Submit mode only supports the train action.")

    timeout_raw = payload.get("modal_timeout")
    if timeout_raw is None:
        modal_timeout: int | None = None
    elif isinstance(timeout_raw, bool) or not isinstance(timeout_raw, int):
        raise RuntimeError("modal_timeout must be an integer number of seconds.")
    else:
        if timeout_raw <= 0:
            raise RuntimeError("modal_timeout must be positive.")
        modal_timeout = timeout_raw

    return ModalLaunchConfig(
        mode=mode,
        action=action,
        quiet=_payload_bool(payload, "quiet"),
        repo_root=Path(_payload_string(payload, "repo_root")).resolve(),
        gpu_type=_payload_string(payload, "gpu_type"),
        cpu=_payload_positive_float(payload, "cpu", DEFAULT_REMOTE_CPU),
        memory_gb=_payload_positive_float(payload, "memory_gb", DEFAULT_REMOTE_MEMORY_GB),
        train_entrypoint=_payload_optional_string(payload, "train_entrypoint"),
        extra_args=_payload_string_list(payload, "extra_args"),
        run_id=_payload_string(payload, "run_id"),
        use_flash3=_payload_bool(payload, "use_flash3"),
        forwarded_env=_payload_env_dict(payload, "forwarded_env"),
        submission_outputs=_payload_bool_default(payload, "submission_outputs", False),
        modal_timeout=modal_timeout,
    )


CONFIG = _load_modal_config_from_env()
GPU_TYPE = CONFIG.gpu_type
REMOTE_CPU = CONFIG.cpu
REMOTE_MEMORY_GB = CONFIG.memory_gb
REMOTE_MEMORY_MIB = int(round(REMOTE_MEMORY_GB * 1024))

# gpu_remote Modal wall clock: `arc submit` stays bounded; `arc run train` allows longer interactive runs.
# `submit` defaults to 1800s (proxy-sized); the submit YAML can override via `modal_timeout`
# for 8×H100 full runs that need more headroom for eval + TTT.
if CONFIG.mode == "run":
    GPU_REMOTE_TIMEOUT_SECONDS = 10800
elif CONFIG.modal_timeout is not None:
    GPU_REMOTE_TIMEOUT_SECONDS = CONFIG.modal_timeout
else:
    GPU_REMOTE_TIMEOUT_SECONDS = 1800


def _requested_gpu_count(gpu_type: str) -> int:
    _, sep, suffix = gpu_type.rpartition(":")
    if not sep:
        return 1
    try:
        count = int(suffix)
    except ValueError:
        return 1
    if count <= 0:
        raise RuntimeError(f"GPU count must be positive in ARC_PARAMETER_GOLF_GPU, got {gpu_type!r}.")
    return count


def _image_build_gpu_type(gpu_type: str) -> str:
    _, sep, suffix = gpu_type.rpartition(":")
    if not sep:
        return gpu_type
    try:
        int(suffix)
    except ValueError:
        return gpu_type
    return gpu_type[: -(len(suffix) + 1)]


def _find_first(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _install_common_training_deps(image: modal.Image) -> modal.Image:
    """Packages needed for train_gpt on any GPU (tokenizer + compressed data)."""
    return image.run_commands(
        f"uv pip install --python {UV_PYTHON} sentencepiece zstandard brotli",
        (
            f'{UV_PYTHON} -c "import sentencepiece, zstandard; print(\'common deps OK\')"'
        ),
    )


def _install_flash3_deps(image: modal.Image) -> modal.Image:
    """Flash Attention 3 wheels (e.g. H100); use only when USE_FLASH3 is enabled."""
    return image.run_commands(
        (
            f"uv pip install --python {UV_PYTHON} "
            f"flash_attn_3 --find-links {FLASH_ATTENTION_3_WHEEL_INDEX}"
        ),
        (
            f'{UV_PYTHON} -c "from flash_attn_interface import flash_attn_func; '
            "print('flash-attn OK')\""
        ),
    )


def _build_local_image() -> tuple[modal.Image, str]:
    task_root = CONFIG.repo_root
    pyproject_path = task_root / "pyproject.toml"
    if not pyproject_path.is_file():
        raise RuntimeError(f"Parameter Golf task is missing {pyproject_path}.")

    default_train_file = _find_first(
        [
            task_root / "train_gpt.py",
            task_root / "workspace" / "train_gpt.py",
        ]
    )
    if default_train_file is None:
        raise RuntimeError("Parameter Golf task is missing `train_gpt.py`.")
    train_entrypoint = CONFIG.train_entrypoint
    if train_entrypoint is None:
        train_file = default_train_file
        train_relative = "train_gpt.py"
    else:
        train_file = (task_root / train_entrypoint).resolve()
        try:
            train_relative = train_file.relative_to(task_root).as_posix()
        except ValueError as exc:
            raise RuntimeError("ARC_PARAMETER_GOLF_TRAIN_ENTRYPOINT must stay inside the repo.") from exc
        if not train_file.is_file():
            raise RuntimeError(f"Parameter Golf task is missing `{train_relative}`.")

    prepare_file = _find_first(
        [
            task_root / "prepare.py",
            task_root / "workspace" / "prepare.py",
        ]
    )
    downloader_file = task_root / "data" / "cached_challenge_fineweb.py"
    if prepare_file is None and not downloader_file.is_file():
        raise RuntimeError(
            "Parameter Golf task is missing both `prepare.py` and `data/cached_challenge_fineweb.py`."
        )

    image = modal.Image.debian_slim(python_version="3.12").uv_sync(
        uv_project_dir=str(task_root),
        gpu=_image_build_gpu_type(GPU_TYPE),
    )
    image = _install_common_training_deps(image)
    if CONFIG.use_flash3:
        image = _install_flash3_deps(image)
    image = image.add_local_file(
        train_file,
        remote_path=f"{REMOTE_TASK_DIR}/{train_relative}",
    )

    prepare_entrypoint = "prepare.py"
    if prepare_file is not None:
        image = image.add_local_file(
            prepare_file,
            remote_path=f"{REMOTE_TASK_DIR}/prepare.py",
        )
    else:
        prepare_entrypoint = "data/cached_challenge_fineweb.py"

    if downloader_file.is_file():
        image = image.add_local_file(
            downloader_file,
            remote_path=f"{REMOTE_TASK_DIR}/data/cached_challenge_fineweb.py",
        )

    return image, prepare_entrypoint


def _remote_env_from_config() -> dict[str, str]:
    result = dict(CONFIG.forwarded_env)
    result[MODAL_CONFIG_ENV_VAR] = os.environ[MODAL_CONFIG_ENV_VAR]
    result.update(
        {
            "ARC_PARAMETER_GOLF_GPU": GPU_TYPE,
            "ARC_PARAMETER_GOLF_RUN_ID": CONFIG.run_id,
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
            "TORCHINDUCTOR_CACHE_DIR": f"{VOLUME_ROOT}/torch_cache",
        }
    )
    if CONFIG.use_flash3:
        result["USE_FLASH3"] = "1"
    return result


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(f"{APP_NAME}-cache", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12")
prepare_entrypoint: str | None = None
if modal.is_local():
    image, prepare_entrypoint = _build_local_image()


def _python_command(args: list[str], *, distributed: bool) -> list[str]:
    if not distributed:
        return ["python", *args]
    return [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(_requested_gpu_count(GPU_TYPE)),
        *args,
    ]


def _abs_entrypoint_if_needed(relative_or_abs: str) -> str:
    """Train scripts are mounted under REMOTE_TASK_DIR; use absolute path when cwd is elsewhere."""
    if relative_or_abs.startswith("/"):
        return relative_or_abs
    return f"{REMOTE_TASK_DIR}/{relative_or_abs}"


def _run_python(args: list[str], *, distributed: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop(MODAL_CONFIG_ENV_VAR, None)
    env["HOME"] = VOLUME_ROOT
    env["PYTHONUNBUFFERED"] = "1"
    data_root = f"{VOLUME_ROOT}/parameter-golf-data"
    env["PARAMETER_GOLF_MODAL_DATA_ROOT"] = data_root
    if "DATA_PATH" not in env:
        env["DATA_PATH"] = f"{data_root}/datasets/fineweb10B_sp1024"
    if "TOKENIZER_PATH" not in env:
        env["TOKENIZER_PATH"] = f"{data_root}/tokenizers/fineweb_1024_bpe.model"
    out_root = VOLUME_SUBMISSIONS_ROOT if CONFIG.submission_outputs else VOLUME_RUNS_ROOT
    env["PARAMETER_GOLF_OUTPUT_ROOT"] = out_root
    if CONFIG.submission_outputs:
        run_fallback = env.get("ARC_PARAMETER_GOLF_RUN_ID", "").strip() or CONFIG.run_id
        env["RUN_ID"] = run_fallback
    elif "RUN_ID" not in env:
        run_id = env.get("ARC_PARAMETER_GOLF_RUN_ID", "").strip()
        if run_id:
            env["RUN_ID"] = run_id

    # Submission train scripts often write logs/artifacts with relative paths (like `arc submit`'s
    # root train_gpt using PARAMETER_GOLF_OUTPUT_ROOT). chdir to the volume run dir so those paths
    # persist on the cache volume without changing train_gpt.py.
    if CONFIG.submission_outputs:
        work_dir = f"{out_root}/{CONFIG.run_id}"
        os.makedirs(work_dir, exist_ok=True)
        run_args = list(args)
        if run_args:
            run_args[0] = _abs_entrypoint_if_needed(run_args[0])
    else:
        work_dir = REMOTE_TASK_DIR
        run_args = args

    cmd = _python_command(run_args, distributed=distributed)
    proc = subprocess.Popen(
        cmd,
        cwd=work_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    decoders = {
        "stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"),
        "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace"),
    }
    outputs: dict[str, list[str]] = {"stdout": [], "stderr": []}
    writers = {"stdout": sys.stdout, "stderr": sys.stderr}

    while selector.get_map():
        for key, _ in selector.select():
            chunk = os.read(key.fileobj.fileno(), 4096)
            stream_name = key.data
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            text = decoders[stream_name].decode(chunk)
            if text:
                outputs[stream_name].append(text)
                writers[stream_name].write(text)
                writers[stream_name].flush()

    for stream_name, decoder in decoders.items():
        tail = decoder.decode(b"", final=True)
        if tail:
            outputs[stream_name].append(tail)
            writers[stream_name].write(tail)
            writers[stream_name].flush()

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.wait(),
        stdout="".join(outputs["stdout"]),
        stderr="".join(outputs["stderr"]),
    )


def _tail(text: str, max_lines: int = 50) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _validate_gpu() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal GPU container.")
    device_count = torch.cuda.device_count()
    expected_device_count = _requested_gpu_count(GPU_TYPE)
    if device_count != expected_device_count:
        raise RuntimeError(
            f"Expected {expected_device_count} visible GPU(s) for {GPU_TYPE}, found {device_count}."
        )
    return {
        "device_name": torch.cuda.get_device_name(0),
        "device_count": device_count,
        "capability": torch.cuda.get_device_capability(0),
    }


@app.function(
    image=image,
    cpu=REMOTE_CPU,
    memory=REMOTE_MEMORY_MIB,
    timeout=7200,
    volumes={VOLUME_ROOT: cache_volume},
    env=_remote_env_from_config(),
)
def cpu_remote(entrypoint_file: str, extra_args: list[str] | None = None) -> dict[str, Any]:
    cache_volume.reload()
    proc = _run_python([entrypoint_file, *(extra_args or [])])
    cache_volume.commit()
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=REMOTE_CPU,
    memory=REMOTE_MEMORY_MIB,
    timeout=GPU_REMOTE_TIMEOUT_SECONDS,
    volumes={VOLUME_ROOT: cache_volume},
    env=_remote_env_from_config(),
)
def gpu_remote(entrypoint_file: str, extra_args: list[str] | None = None) -> dict[str, Any]:
    cache_volume.reload()
    gpu_info = _validate_gpu()
    proc = _run_python([entrypoint_file, *(extra_args or [])], distributed=gpu_info["device_count"] > 1)
    cache_volume.commit()
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "gpu": gpu_info,
    }


@app.local_entrypoint()
def main() -> None:
    action = CONFIG.action
    entrypoint_file = CONFIG.train_entrypoint or "train_gpt.py"
    extra_args = CONFIG.extra_args
    remote_function = gpu_remote
    if action == "prepare":
        if prepare_entrypoint is None:
            raise SystemExit("Prepare entrypoint is not configured for local Modal launch.")
        entrypoint_file = prepare_entrypoint
        remote_function = cpu_remote

    with modal.enable_output() as output_manager:
        output_manager.set_quiet_mode(CONFIG.quiet)
        result = remote_function.remote(entrypoint_file, extra_args)

    if result["returncode"] != 0:
        tail = _tail(
            result["stdout"]
            + ("\n" if result["stdout"] and result["stderr"] else "")
            + result["stderr"]
        )
        if tail:
            sys.stderr.write("\n--- remote tail ---\n")
            sys.stderr.write(tail)
            if not tail.endswith("\n"):
                sys.stderr.write("\n")
        raise SystemExit(result["returncode"])
