from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.text import parse_metric_flags, read_text_argument
from arc.timeutil import utc_now_iso


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("commit", help="Experiment commit hash or prefix.")
    parser.add_argument("analysis", help="Failure analysis or `-` for stdin.")


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    app.store.require_initialized()
    record = app.store.get_node_record(args.commit)
    if record is None:
        raise ArcError(f"Unknown commit: {args.commit}")
    if record.node.archived_at is not None:
        raise ArcError(f"Cannot record failure for archived node `{record.node.commit}`.")
    if record.node.status not in {"committed", "running"}:
        raise ArcError(f"Cannot record failure from status `{record.node.status}`.")

    analysis = read_text_argument(args.analysis)
    metrics = parse_metric_flags(extras)
    completed_at = utc_now_iso()
    _, metrics, _ = app.task.process_result_metrics(
        record.node,
        verdict="invalid",
        metrics=metrics,
        completed_at=completed_at,
    )
    app.store.upsert_metrics(record.node.commit, metrics)
    app.store.update_node(
        record.node.commit,
        status="failed",
        analysis=analysis,
        completed_at=completed_at,
        verdict=None,
    )

    print(f"Recorded failure for {record.node.commit} ({record.node.name})")
    print(f"Status: {record.node.status} → failed")
    return 0


COMMAND = CommandSpec(
    name="fail",
    help="Record a failed experiment.",
    register=register,
    run=run,
)
