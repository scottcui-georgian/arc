from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arc.app import ArcApp
    from arc.models import Node, NodeRecord
    from arc.tasks.base import SubmitOutcome, TaskModule
else:
    from arc.tasks.base import SubmitOutcome, TaskModule

from arc.errors import ArcError
from arc.text import format_float

ARTIFACT_LIMIT_BYTES = 16_000_000


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _register_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reduce Modal progress noise when supported.",
    )
    parser.add_argument(
        "--cpu",
        type=_positive_float,
        help="Modal CPU count override. Defaults to ARC_PARAMETER_GOLF_CPU or 8.",
    )
    parser.add_argument(
        "--memory-gb",
        type=_positive_float,
        help="Modal memory override in GiB. Defaults to ARC_PARAMETER_GOLF_MEMORY_GB or 8.",
    )


def _register_train_parser(parser: argparse.ArgumentParser) -> None:
    _register_run_parser(parser)
    parser.add_argument(
        "--gpu",
        help="Modal GPU type override. Defaults to ARC_PARAMETER_GOLF_GPU or H100.",
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
    from arc.paths import discover_worktree_root
    from arc.tasks.parameter_golf.runtime import ParameterGolfModalRunner

    action = getattr(args, "_parameter_golf_action", None)
    if action not in {"train", "prepare"}:
        raise ArcError("Parameter Golf action is not set.")
    quiet = args.quiet if args.quiet is not None else (action != "train")
    runner = ParameterGolfModalRunner(discover_worktree_root())
    gpu = getattr(args, "gpu", None)
    cpu = getattr(args, "cpu", None)
    memory_gb = getattr(args, "memory_gb", None)
    return runner.run(action, list(extras), quiet=quiet, gpu=gpu, cpu=cpu, memory_gb=memory_gb)


class ParameterGolfTaskModule(TaskModule):
    def __init__(self) -> None:
        super().__init__(name="parameter_golf")

    def instruction_path(self, name: str) -> Path | None:
        path = Path(__file__).with_name("instructions") / f"{name}.md"
        if path.is_file():
            return path
        return None

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
        _register_train_parser(train_parser)
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
        submit_gpu_count = record.metrics.get("submit_gpu_count")
        submit_train_wallclock = record.metrics.get("submit_train_wallclock")
        if submit_gpu_count is not None and submit_train_wallclock is not None:
            parts.append(f"{int(submit_gpu_count)}×H100 {int(submit_train_wallclock)}s")
        if not parts:
            return ""
        return " (" + " | ".join(parts) + ")"

    def submit(
        self,
        node: Node,
        worktree_root: Path,
        log_path: Path,
        *,
        config_name: str | None = None,
        train_wallclock: int | None = None,
        grad_accum_steps: int | None = None,
    ) -> SubmitOutcome | None:
        from arc.tasks.parameter_golf.runtime import (
            ParameterGolfModalRunner,
            submit_gpu_count,
        )

        del node
        if config_name is None:
            raise ArcError(
                "Parameter Golf submit requires --config {proxy,full}."
            )
        runner = ParameterGolfModalRunner(worktree_root)
        result, submit_config = runner.submit_train(
            log_path,
            config_name=config_name,
            train_wallclock=train_wallclock,
            grad_accum_steps=grad_accum_steps,
        )
        effective_wallclock = (
            train_wallclock if train_wallclock is not None else submit_config.train_wallclock
        )
        metrics: dict[str, float] = {
            "submit_gpu_count": float(submit_gpu_count(submit_config.gpu_type)),
            "submit_train_wallclock": float(effective_wallclock),
        }
        return SubmitOutcome(result=result, metrics=metrics)


def build_task_module() -> TaskModule:
    return ParameterGolfTaskModule()
