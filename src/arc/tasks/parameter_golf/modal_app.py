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
from pathlib import Path
from typing import Any

import modal

APP_NAME = "autoresearch-parameter-golf"
DEFAULT_GPU_TYPE = "A100-40GB"
DEFAULT_REMOTE_CPU = 8.0
DEFAULT_REMOTE_MEMORY_GB = 8.0
GPU_TYPE = os.environ.get("ARC_PARAMETER_GOLF_GPU", DEFAULT_GPU_TYPE).strip() or DEFAULT_GPU_TYPE
FLASH_ATTENTION_3_WHEEL_INDEX = (
    "https://windreamer.github.io/flash-attention3-wheels/cu128_torch291"
)
UV_PYTHON = "/.uv/.venv/bin/python"
REMOTE_TASK_DIR = "/root/task"
VOLUME_ROOT = "/cache-home"
VOLUME_RUNS_ROOT = f"{VOLUME_ROOT}/parameter-golf-runs"


def _env_positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive number.") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be greater than 0.")
    return parsed


REMOTE_CPU = _env_positive_float("ARC_PARAMETER_GOLF_CPU", DEFAULT_REMOTE_CPU)
REMOTE_MEMORY_GB = _env_positive_float("ARC_PARAMETER_GOLF_MEMORY_GB", DEFAULT_REMOTE_MEMORY_GB)
REMOTE_MEMORY_MIB = int(round(REMOTE_MEMORY_GB * 1024))


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


def _train_entrypoint_override_from_env() -> str | None:
    value = os.environ.get("ARC_PARAMETER_GOLF_TRAIN_ENTRYPOINT")
    if value is None:
        return None
    value = value.strip()
    return value or None


def _task_root_from_env() -> Path:
    value = os.environ.get("ARC_PARAMETER_GOLF_REPO_ROOT")
    if not value:
        raise RuntimeError("ARC_PARAMETER_GOLF_REPO_ROOT is not set.")
    return Path(value).resolve()


def _find_first(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _should_use_flash3(gpu_type: str | None) -> bool:
    return bool(gpu_type) and "H100" in gpu_type


def _is_h100_gpu() -> bool:
    return "H100" in GPU_TYPE


def _install_h100_deps(image: modal.Image) -> modal.Image:
    return image.run_commands(
        (
            f"uv pip install --python {UV_PYTHON} "
            f"flash_attn_3 --find-links {FLASH_ATTENTION_3_WHEEL_INDEX}"
        ),
        f"uv pip install --python {UV_PYTHON} sentencepiece zstandard",
        (
            f'{UV_PYTHON} -c "from flash_attn_interface import flash_attn_func; '
            "import sentencepiece, zstandard; print('deps OK')\""
        ),
    )


def _build_local_image() -> tuple[modal.Image, str]:
    task_root = _task_root_from_env()
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
    train_entrypoint = _train_entrypoint_override_from_env()
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
    if _is_h100_gpu():
        image = _install_h100_deps(image)
    else:
        image = image.uv_pip_install("zstandard")
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


def _runtime_env_from_client() -> dict[str, str]:
    result: dict[str, str] = {
        "ARC_PARAMETER_GOLF_GPU": GPU_TYPE,
    }
    run_id = os.environ.get("ARC_PARAMETER_GOLF_RUN_ID")
    if run_id:
        result["ARC_PARAMETER_GOLF_RUN_ID"] = run_id
    if _should_use_flash3(GPU_TYPE):
        result["USE_FLASH3"] = "1"
    if _requested_gpu_count(GPU_TYPE) == 1:
        result["GRAD_ACCUM_STEPS"] = "4"
    result["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    result["TORCHINDUCTOR_CACHE_DIR"] = f"{VOLUME_ROOT}/torch_cache"
    return result


def _extra_args_from_env() -> list[str]:
    raw = os.environ.get("ARC_PARAMETER_GOLF_ACTION_ARGS", "[]")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid ARC_PARAMETER_GOLF_ACTION_ARGS payload.") from exc
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise RuntimeError("ARC_PARAMETER_GOLF_ACTION_ARGS must be a JSON array of strings.")
    return payload


def _quiet_mode_from_env() -> bool:
    return os.environ.get("ARC_PARAMETER_GOLF_QUIET", "1") != "0"

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


def _run_python(args: list[str], *, distributed: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = VOLUME_ROOT
    env["PYTHONUNBUFFERED"] = "1"
    if "RUN_ID" not in env:
        run_id = env.get("ARC_PARAMETER_GOLF_RUN_ID", "").strip()
        if run_id:
            env["RUN_ID"] = run_id
    data_root = f"{VOLUME_ROOT}/parameter-golf-data"
    env["PARAMETER_GOLF_MODAL_DATA_ROOT"] = data_root
    if "DATA_PATH" not in env:
        env["DATA_PATH"] = f"{data_root}/datasets/fineweb10B_sp1024"
    if "TOKENIZER_PATH" not in env:
        env["TOKENIZER_PATH"] = f"{data_root}/tokenizers/fineweb_1024_bpe.model"
    env["PARAMETER_GOLF_OUTPUT_ROOT"] = VOLUME_RUNS_ROOT
    cmd = _python_command(args, distributed=distributed)
    proc = subprocess.Popen(
        cmd,
        cwd=REMOTE_TASK_DIR,
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
    env=_runtime_env_from_client(),
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
    timeout=1800,
    volumes={VOLUME_ROOT: cache_volume},
    env=_runtime_env_from_client(),
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
def main(action: str) -> None:
    valid_actions = {"prepare", "train"}
    if action not in valid_actions:
        valid = ", ".join(sorted(valid_actions))
        raise SystemExit(f"Unknown action '{action}'. Valid actions: {valid}")

    extra_args = _extra_args_from_env()
    quiet_mode = _quiet_mode_from_env()
    entrypoint_file = _train_entrypoint_override_from_env() or "train_gpt.py"
    remote_function = gpu_remote
    if action == "prepare":
        if prepare_entrypoint is None:
            raise SystemExit("Prepare entrypoint is not configured for local Modal launch.")
        entrypoint_file = prepare_entrypoint
        remote_function = cpu_remote

    with modal.enable_output() as output_manager:
        output_manager.set_quiet_mode(quiet_mode)
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
