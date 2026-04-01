from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.text import parse_metric_flags, read_text_argument
from arc.timeutil import utc_now_iso


def register(parser: argparse.ArgumentParser) -> None:
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.description = "Record a hard experiment failure such as a crash, OOM, timeout, or infra error."
    parser.epilog = (
        "Metric flags:\n"
        "  Pass metrics as --name=value, for example --peak_vram_mb=6144.\n"
        "  Task-specific metrics may be inferred automatically from run.log.\n"
        "  Explicit flags override inferred values when both are present."
    )
    parser.add_argument("commit", help="Experiment commit hash or prefix.")
    parser.add_argument("analysis", help="Failure analysis, or `-` to read it from stdin.")


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
    inferred_metrics, inferred_notes = app.task.derive_result_metrics(
        record.node,
        app.node_log_path(record.node),
    )
    metrics = {**inferred_metrics, **metrics}
    completed_at = utc_now_iso()
    _, metrics, notes = app.task.process_result_metrics(
        record.node,
        verdict="invalid",
        metrics=metrics,
        completed_at=completed_at,
    )
    notes = [*inferred_notes, *notes]
    app.store.upsert_metrics(record.node.commit, metrics)
    app.store.update_node(
        record.node.commit,
        status="failed",
        analysis=analysis,
        completed_at=completed_at,
        verdict=None,
    )

    print(f"Recorded failure for {app.display_commit(record.node.commit)} ({record.node.name})")
    print(f"Status: {record.node.status} → failed")
    for note in notes:
        print(f"Note: {note}")
    for name, value in sorted(metrics.items()):
        print(f"{name}: {app.task.format_metric(name, value)}")
    return 0


COMMAND = CommandSpec(
    name="fail",
    help="Record a hard failure; infer metrics from run.log when possible.",
    register=register,
    run=run,
)
