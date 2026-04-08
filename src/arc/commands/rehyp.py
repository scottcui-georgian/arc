from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.text import read_text_argument


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("commit", help="Experiment commit hash or prefix.")
    parser.add_argument("text", help="Replacement hypothesis text, or `-` to read it from stdin.")


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    record = app.store.get_node_record(args.commit)
    if record is None:
        raise ArcError(f"Unknown commit: {args.commit}")
    if record.node.archived_at is not None:
        raise ArcError(f"Cannot update hypothesis for archived node `{record.node.commit}`.")

    new_hypothesis = read_text_argument(args.text)
    previous_hypothesis = record.node.hypothesis or "-"
    if record.node.hypothesis == new_hypothesis:
        print(f"Hypothesis unchanged for {app.display_commit(record.node.commit)} ({record.node.name})")
        print(f"Hypothesis: {previous_hypothesis}")
        return 0

    app.store.update_node(record.node.commit, hypothesis=new_hypothesis)
    print(f"Updated hypothesis for {app.display_commit(record.node.commit)} ({record.node.name})")
    print(f"Previous hypothesis: {previous_hypothesis}")
    print(f"New hypothesis: {new_hypothesis}")
    return 0


COMMAND = CommandSpec(
    name="rehyp",
    help="Replace the stored hypothesis text for an existing experiment.",
    register=register,
    run=run,
)
