from __future__ import annotations

import argparse
import sys

from arc.app import ArcApp
from arc.commands import BUILTIN_COMMANDS
from arc.errors import ArcError


def build_parser(app: ArcApp) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arc", description="Autoresearch experiment CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in BUILTIN_COMMANDS:
        subparser = subparsers.add_parser(command.name, help=command.help)
        command.register(subparser)
        subparser.set_defaults(_arc_command=command)

    app.task.register_commands(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    try:
        app = ArcApp.discover()
        parser = build_parser(app)
        args, extras = parser.parse_known_args(args_list)
        command = getattr(args, "_arc_command", None)
        if command is None:
            parser.print_help()
            return 1
        return int(command.run(app, args, extras) or 0)
    except ArcError as exc:
        print(f"arc: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
