from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arc.app import ArcApp
    from arc.executors.base import SubmitResult
    from arc.models import Node
    from arc.tasks.base import TaskModule
else:
    from arc.tasks.base import TaskModule


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
        return super().format_metric(name, value)

    def submit(self, node: Node, worktree_root: Path, log_path: Path) -> SubmitResult | None:
        from arc.tasks.parameter_golf.runtime import ParameterGolfModalRunner

        del node
        runner = ParameterGolfModalRunner(worktree_root)
        return runner.submit_train(log_path)


def build_task_module() -> TaskModule:
    return ParameterGolfTaskModule()
