from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("commit", help="Experiment commit hash or prefix.")
    parser.add_argument(
        "verdict",
        choices=("promising", "regression", "neutral", "inconclusive", "invalid"),
        help="Updated verdict for a completed experiment.",
    )


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    record = app.store.get_node_record(args.commit)
    if record is None:
        raise ArcError(f"Unknown commit: {args.commit}")
    if record.node.archived_at is not None:
        raise ArcError(f"Cannot update verdict for archived node `{record.node.commit}`.")
    if record.node.status != "completed":
        raise ArcError(
            f"Can only update verdict for completed nodes, got `{record.node.status}`."
        )

    completed_at = record.node.completed_at or record.node.created_at
    verdict, metrics, notes = app.task.process_result_metrics(
        record.node,
        verdict=args.verdict,
        metrics=record.metrics,
        completed_at=completed_at,
    )
    previous = record.node.verdict or "-"
    if record.node.verdict == verdict:
        print(f"Verdict unchanged for {record.node.commit} ({record.node.name})")
        print(f"Verdict: {previous}")
        for note in notes:
            print(f"Note: {note}")
        return 0

    app.store.upsert_metrics(record.node.commit, metrics)
    app.store.update_node(record.node.commit, verdict=verdict)
    print(f"Updated verdict for {record.node.commit} ({record.node.name})")
    print(f"Verdict: {previous} → {verdict}")
    for note in notes:
        print(f"Note: {note}")
    return 0


COMMAND = CommandSpec(
    name="verdict",
    help="Update the verdict for a completed experiment.",
    register=register,
    run=run,
)
