from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.render import render_report


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("commit", help="Experiment commit hash or prefix.")


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()
    path = app.store.path_to_root(args.commit)
    if not path:
        raise ArcError(f"Unknown commit: {args.commit}")
    print(render_report(path, metric_name=app.main_metric()))
    return 0


COMMAND = CommandSpec(
    name="report",
    help="Show the path report from root to a node.",
    register=register,
    run=run,
)
