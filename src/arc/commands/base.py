from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from arc.app import ArcApp


CommandHandler = Callable[[ArcApp, argparse.Namespace, list[str]], int]
CommandRegistrar = Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    help: str
    register: CommandRegistrar
    run: CommandHandler
