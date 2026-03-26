from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.render import render_status
from arc.timeutil import utc_now_iso


def register(parser: argparse.ArgumentParser) -> None:
    del parser


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    del args
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    last_check = app.store.get_meta("last_status_check")
    running = app.store.list_by_status(["running"])
    completed = app.store.list_completed_since(last_check)
    print(render_status(running, completed, metric_name=app.main_metric()))
    app.store.set_meta("last_status_check", utc_now_iso())
    return 0


COMMAND = CommandSpec(
    name="status",
    help="Show active experiments and recently finished runs.",
    register=register,
    run=run,
)
