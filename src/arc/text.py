from __future__ import annotations

import sys

from arc.errors import ArcError


def read_text_argument(value: str) -> str:
    if value != "-":
        text = value.strip()
    else:
        text = sys.stdin.read().strip()
    if not text:
        raise ArcError("Expected non-empty text input.")
    return text


def parse_metric_flags(tokens: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for token in tokens:
        if not token.startswith("--") or "=" not in token:
            raise ArcError(f"Invalid metric flag: {token}")
        name, raw_value = token[2:].split("=", 1)
        if not name:
            raise ArcError(f"Invalid metric name in flag: {token}")
        try:
            metrics[name] = float(raw_value)
        except ValueError as exc:
            raise ArcError(f"Metric `{name}` must be numeric.") from exc
    return metrics


def indent_block(text: str, prefix: str = "  ") -> str:
    return "\n".join(f"{prefix}{line}" if line else "" for line in text.splitlines())


def format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:g}"


def format_signed_delta(delta: float) -> str:
    return f"{delta:+.3g}"


def format_commit(commit: str, length: int = 12) -> str:
    if len(commit) <= length:
        return commit
    return commit[:length]
