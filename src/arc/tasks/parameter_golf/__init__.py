from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arc.app import ArcApp
    from arc.executors.base import SubmitResult
    from arc.models import Node, NodeRecord
    from arc.tasks.base import TaskModule
else:
    from arc.tasks.base import TaskModule

from arc.text import format_float
ARTIFACT_LIMIT_BYTES = 16_000_000
TIMESTAMP_PREFIX = re.compile(r"^\[(?P<ts>[^\]]+)\]\s")
FINAL_EXACT_PATTERN = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) "
    r"val_bpb:(?P<val_bpb>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
FINAL_PATTERN = re.compile(
    r"final_int8_zlib_roundtrip val_loss:(?P<val_loss>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) "
    r"val_bpb:(?P<val_bpb>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) "
    r"eval_time:(?P<eval_time_ms>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)ms"
)
SUBMISSION_BYTES_PATTERN = re.compile(r"Total submission size int8\+zlib: (?P<bytes>\d+) bytes")
PEAK_VRAM_PATTERN = re.compile(
    r"peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB"
)


def _parse_timestamped_line(line: str) -> tuple[datetime | None, str]:
    match = TIMESTAMP_PREFIX.match(line)
    if not match:
        return None, line
    raw = match.group("ts")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(raw)
    except ValueError:
        return None, line
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC), line[match.end() :]


def _last_run_segment(text: str) -> list[str]:
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if "submitting Parameter Golf train via Modal" in line:
            start = index
    return lines[start:]


def _derive_metrics_from_run_log(log_path: Path) -> tuple[dict[str, float], list[str]]:
    if not log_path.is_file():
        return {}, []
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}, []

    lines = _last_run_segment(text)
    metrics: dict[str, float] = {}
    notes: list[str] = []
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None

    for line in lines:
        timestamp, payload = _parse_timestamped_line(line)
        if timestamp is not None:
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp

        match = FINAL_EXACT_PATTERN.search(payload)
        if match:
            metrics["val_loss"] = float(match.group("val_loss"))
            metrics["val_bpb"] = float(match.group("val_bpb"))
            continue

        match = FINAL_PATTERN.search(payload)
        if match:
            metrics.setdefault("val_loss", float(match.group("val_loss")))
            metrics.setdefault("val_bpb", float(match.group("val_bpb")))
            metrics["eval_time_ms"] = float(match.group("eval_time_ms"))
            continue

        match = SUBMISSION_BYTES_PATTERN.search(payload)
        if match:
            metrics["submission_bytes"] = float(match.group("bytes"))
            continue

        match = PEAK_VRAM_PATTERN.search(payload)
        if match:
            metrics["peak_vram_mb"] = float(match.group("allocated"))

    if first_timestamp is not None and last_timestamp is not None and last_timestamp > first_timestamp:
        runtime_minutes = (last_timestamp - first_timestamp).total_seconds() / 60.0
        metrics["runtime_minutes"] = round(runtime_minutes, 2)
    else:
        notes.append("Runtime could not be derived from `run.log`; no per-line timestamps were available.")

    return metrics, notes


def _register_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reduce Modal progress noise when supported.",
    )


def _hide_subparser_from_help(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
) -> None:
    subparsers._choices_actions = [
        action for action in subparsers._choices_actions if action.dest != name
    ]
    visible = [action.metavar for action in subparsers._choices_actions]
    if visible:
        subparsers.metavar = "{" + ",".join(visible) + "}"


def _run_action(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    from arc.errors import ArcError
    from arc.tasks.parameter_golf.runtime import ParameterGolfModalRunner

    action = getattr(args, "_parameter_golf_action", None)
    if action not in {"train", "prepare"}:
        raise ArcError("Parameter Golf action is not set.")
    quiet = args.quiet if args.quiet is not None else (action != "train")
    runner = ParameterGolfModalRunner(app.paths.repo_root)
    return runner.run(action, list(extras), quiet=quiet)

class ParameterGolfTaskModule(TaskModule):
    def __init__(self) -> None:
        super().__init__(name="parameter_golf")

    def register_commands(self, subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
        from arc.commands.base import CommandSpec

        run_command = CommandSpec(
            name="run",
            help="Run task-specific Modal jobs.",
            register=lambda parser: None,
            run=_run_action,
        )
        run_parser = subparsers.add_parser(
            "run",
            help=argparse.SUPPRESS,
        )
        _hide_subparser_from_help(subparsers, "run")
        run_subparsers = run_parser.add_subparsers(dest="run_action", required=True)

        train_parser = run_subparsers.add_parser(
            "train",
            help="Run the task train entrypoint on Modal GPU.",
        )
        _register_run_parser(train_parser)
        train_parser.set_defaults(
            _arc_command=run_command,
            _parameter_golf_action="train",
        )

        prepare_parser = run_subparsers.add_parser(
            "prepare",
            help="Run the task prepare entrypoint on Modal CPU.",
        )
        _register_run_parser(prepare_parser)
        prepare_parser.set_defaults(
            _arc_command=run_command,
            _parameter_golf_action="prepare",
        )

    def format_metric(self, name: str, value: float) -> str:
        if name == "artifact_mb":
            return f"{value:.2f} MB"
        if name == "runtime_minutes":
            return f"{value:.2f} min"
        if name == "submission_bytes":
            return f"{int(round(value)):,}"
        if name == "peak_vram_mb":
            return f"{value:.0f} MB"
        if name == "eval_time_ms":
            return f"{value:.0f} ms"
        return format_float(value)

    def derive_result_metrics(
        self,
        node: Node,
        log_path: Path,
    ) -> tuple[dict[str, float], list[str]]:
        del node
        return _derive_metrics_from_run_log(log_path)

    def process_result_metrics(
        self,
        node: Node,
        *,
        verdict: str,
        metrics: dict[str, float],
        completed_at: str,
    ) -> tuple[str, dict[str, float], list[str]]:
        processed = dict(metrics)
        notes: list[str] = []

        if "submission_bytes" in processed:
            submission_bytes = processed["submission_bytes"]
            processed["artifact_mb"] = round(submission_bytes / 1_000_000, 2)
            if submission_bytes > ARTIFACT_LIMIT_BYTES and verdict != "invalid":
                verdict = "invalid"
                notes.append(
                    "Artifact exceeds 16,000,000 bytes; forcing verdict to `invalid`."
                )

        return verdict, processed, notes

    def tree_metric_suffix(self, record: NodeRecord, *, metric_name: str | None) -> str:
        parts: list[str] = []
        if metric_name and metric_name in record.metrics:
            parts.append(self.format_metric(metric_name, record.metrics[metric_name]))
        artifact_mb = record.metrics.get("artifact_mb")
        if artifact_mb is not None:
            parts.append(f"{artifact_mb:.2f}MB")
        runtime_minutes = record.metrics.get("runtime_minutes")
        if runtime_minutes is not None:
            parts.append(f"{runtime_minutes:.2f}m")
        else:
            parts.append("runtime:N/A")
        if not parts:
            return ""
        return " (" + " | ".join(parts) + ")"

    def submit(self, node: Node, worktree_root: Path, log_path: Path) -> SubmitResult | None:
        from arc.tasks.parameter_golf.runtime import ParameterGolfModalRunner

        del node
        runner = ParameterGolfModalRunner(worktree_root)
        return runner.submit_train(log_path)


def build_task_module() -> TaskModule:
    return ParameterGolfTaskModule()
