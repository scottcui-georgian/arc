from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from arc.models import RemoteRunState

SUCCESS_MARKERS = (
    "Stopping app - local entrypoint completed.",
    "✓ App completed.",
)
FAILURE_MARKERS = (
    "Runner failed with exception",
    "Traceback (most recent call last):",
    "RemoteError(",
    "ModuleNotFoundError:",
)
METRIC_PATTERN = re.compile(
    r"\b(?P<name>val_loss|val_bpb):(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
EXIT_CODE_PATTERN = re.compile(r"modal run exited with code (?P<code>\d+)")


def _latest_submission_lines(lines: list[str]) -> list[str]:
    start = 0
    for index, line in enumerate(lines):
        if "submitting " in line:
            start = index
    return lines[start:]


@dataclass(frozen=True)
class RunLogSummary:
    state: RemoteRunState
    metrics: dict[str, float] = field(default_factory=dict)


def summarize_run_log(path: Path) -> RunLogSummary:
    if not path.is_file():
        return RunLogSummary(state="missing")

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return RunLogSummary(state="missing")

    lines = _latest_submission_lines(lines)
    metrics: dict[str, float] = {}
    saw_success = False
    saw_failure = False
    for line in lines:
        found = {
            match.group("name"): float(match.group("value"))
            for match in METRIC_PATTERN.finditer(line)
        }
        if found:
            metrics.update(found)
        if any(marker in line for marker in SUCCESS_MARKERS):
            saw_success = True
        if any(marker in line for marker in FAILURE_MARKERS):
            saw_failure = True
        match = EXIT_CODE_PATTERN.search(line)
        if match:
            if int(match.group("code")) == 0:
                saw_success = True
            else:
                saw_failure = True

    if saw_success:
        return RunLogSummary(state="finished", metrics=metrics)
    if saw_failure:
        return RunLogSummary(state="failed", metrics=metrics)
    return RunLogSummary(state="running", metrics=metrics)
