from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.paths import validate_name


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", help="Pending hypothesis name.")


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    name = validate_name(args.name)
    app.consume_hypothesis(name)
    print(f"Hypothesis removed for {name}.")
    return 0


COMMAND = CommandSpec(
    name="unhyp",
    help="Remove a pending hypothesis from the board.",
    register=register,
    run=run,
)
