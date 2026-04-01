from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.models import RemoteRunRecord
from arc.render import render_status
from arc.runlog import summarize_run_log
from arc.timeutil import utc_now_iso


def register(parser: argparse.ArgumentParser) -> None:
    del parser


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    del args
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    last_check = app.store.get_meta("last_status_check")
    running_records = app.store.list_by_status(["running"])
    running: list[RemoteRunRecord] = []
    finished_remote: list[RemoteRunRecord] = []
    failed_remote: list[RemoteRunRecord] = []
    missing_remote: list[RemoteRunRecord] = []
    for record in running_records:
        log_path = app.node_log_path(record.node)
        summary = summarize_run_log(log_path)
        item = RemoteRunRecord(
            record=record,
            state=summary.state,
            log_path=app.relative_path(log_path),
            metrics=summary.metrics,
        )
        if summary.state == "finished":
            finished_remote.append(item)
        elif summary.state == "failed":
            failed_remote.append(item)
        elif summary.state == "missing":
            missing_remote.append(item)
        else:
            running.append(item)

    completed = app.store.list_recent_by_status(["completed"], last_check)
    failed = app.store.list_recent_by_status(["failed"], last_check)
    print(
        render_status(
            running,
            finished_remote,
            failed_remote,
            missing_remote,
            completed,
            failed,
            metric_name=app.main_metric(),
        )
    )
    app.store.set_meta("last_status_check", utc_now_iso())
    return 0


COMMAND = CommandSpec(
    name="status",
    help="Show active experiments and recently finished runs.",
    register=register,
    run=run,
)
