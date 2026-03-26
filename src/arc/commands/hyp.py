from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.paths import validate_name
from arc.text import indent_block, read_text_argument


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", nargs="?", help="Hypothesis name.")
    parser.add_argument("text", nargs="?", help="Hypothesis text or `-` for stdin.")


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    if args.name is None:
        hypotheses = app.list_hypotheses()
        if not hypotheses:
            print("No pending hypotheses.")
            return 0
        print("Pending hypotheses:")
        for name, text in hypotheses:
            print(f"  {name}")
            print(indent_block(text))
            print("")
        return 0

    if args.text is None:
        raise ArcError("Usage: arc hyp <name> <text | ->")

    name = validate_name(args.name)
    text = read_text_argument(args.text)
    app.save_hypothesis(name, text)
    print(f"Hypothesis saved for {name}.")
    return 0


COMMAND = CommandSpec(
    name="hyp",
    help="Store or list pending hypotheses.",
    register=register,
    run=run,
)
