from __future__ import annotations

from collections.abc import Callable
from collections import defaultdict

from arc.models import Direction, Node, NodeRecord, RemoteRunRecord
from arc.text import format_commit, format_float, format_signed_delta, indent_block
from arc.timeutil import format_elapsed

STATUS_MARKERS = {
    "committed": "•",
    "running": "◌",
    "completed": "◆",
    "failed": "✗",
}
VERDICT_MARKERS = {
    "regression": "○",
    "neutral": "~",
    "inconclusive": "?",
    "invalid": "!",
    "unsupported": "○",
}


def _default_metric_formatter(name: str, value: float) -> str:
    del name
    return format_float(value)


def metric_delta(
    current: float | None,
    previous: float | None,
) -> str | None:
    if current is None or previous is None:
        return None
    return format_signed_delta(current - previous)


def choose_best_leaf(
    records: list[NodeRecord],
    *,
    metric_name: str | None,
    direction: Direction,
) -> str | None:
    if metric_name is None:
        return None
    children: dict[str | None, list[NodeRecord]] = defaultdict(list)
    for record in records:
        children[record.node.parent].append(record)

    leaves = [
        record
        for record in records
        if (
            not children.get(record.node.commit)
            and record.node.status == "completed"
            and record.node.verdict == "promising"
            and record.node.archived_at is None
        )
    ]
    candidates = [
        record
        for record in leaves
        if metric_name in record.metrics
    ]
    if not candidates:
        return None
    reverse = direction == "max"
    ranked = sorted(
        candidates,
        key=lambda item: item.metrics[metric_name],
        reverse=reverse,
    )
    return ranked[0].node.commit


def render_tree(
    records: list[NodeRecord],
    *,
    metric_name: str | None,
    direction: Direction,
    main_commit: str | None,
    status_filter: str | None,
    archived_only: bool,
    depth: int | None,
    leaves_only: bool,
    tree_metric_suffix: Callable[[NodeRecord, str | None], str] = lambda record, metric_name: (
        f" ({format_float(record.metrics[metric_name])})"
        if metric_name and metric_name in record.metrics
        else ""
    ),
) -> str:
    if not records:
        return "No experiments."

    nodes_by_commit = {record.node.commit: record for record in records}
    children: dict[str | None, list[NodeRecord]] = defaultdict(list)
    for record in records:
        children[record.node.parent].append(record)
    for siblings in children.values():
        siblings.sort(key=lambda item: (item.node.created_at, item.node.commit))

    selected: set[str] = set()
    if status_filter or archived_only or leaves_only:
        for record in records:
            is_leaf = not children.get(record.node.commit)
            if archived_only and record.node.archived_at is None:
                continue
            if status_filter and record.node.status != status_filter:
                continue
            if leaves_only and not is_leaf:
                continue
            selected.add(record.node.commit)
        expanded: set[str] = set()
        for commit in selected:
            current: Node | None = nodes_by_commit[commit].node
            while current is not None:
                expanded.add(current.commit)
                if current.parent is None:
                    break
                parent_record = nodes_by_commit.get(current.parent)
                current = None if parent_record is None else parent_record.node
        selected = expanded
    else:
        selected = {record.node.commit for record in records}

    best_leaf = choose_best_leaf(records, metric_name=metric_name, direction=direction)
    roots = [record for record in records if record.node.parent is None]
    lines: list[str] = []

    def visible_children(record: NodeRecord) -> list[NodeRecord]:
        return [
            child
            for child in children.get(record.node.commit, [])
            if child.node.commit in selected
        ]

    def visit(record: NodeRecord, prefix: str, is_last: bool, level: int) -> None:
        if record.node.commit not in selected:
            return

        current = record
        current_prefix = prefix
        current_is_last = is_last
        current_level = level
        chain_prefix: str | None = None

        while True:
            branch = ""
            if current_level > 0:
                branch = "↳ " if chain_prefix is not None else ("└── " if current_is_last else "├── ")
            line_prefix = current_prefix if chain_prefix is None else chain_prefix
            lines.append(
                f"{line_prefix}{branch}{_tree_label(current, metric_name, best_leaf, main_commit, tree_metric_suffix)}"
            )

            if depth is not None and current_level >= depth:
                return

            current_children = visible_children(current)
            if len(current_children) != 1:
                break

            current = current_children[0]
            current_level += 1
            current_is_last = True
            if chain_prefix is None:
                chain_prefix = "" if level == 0 else prefix + ("    " if is_last else "│   ")

        final_children = visible_children(current)
        next_prefix = (
            chain_prefix + "    "
            if chain_prefix is not None
            else ("" if current_level == 0 else current_prefix + ("    " if current_is_last else "│   "))
        )
        for index, child in enumerate(final_children):
            visit(
                child,
                next_prefix,
                index == len(final_children) - 1,
                current_level + 1,
            )

    for index, root in enumerate(roots):
        visit(root, "", index == len(roots) - 1, 0)
    if not lines:
        return "No experiments."
    return "\n".join(lines)


def _tree_label(
    record: NodeRecord,
    metric_name: str | None,
    best_leaf: str | None,
    main_commit: str | None,
    tree_metric_suffix: Callable[[NodeRecord, str | None], str],
) -> str:
    node = record.node
    status_symbol = STATUS_MARKERS.get(node.status, "•")
    if node.status == "completed" and node.verdict in VERDICT_MARKERS:
        status_symbol = VERDICT_MARKERS[node.verdict]
    prefix = status_symbol if node.archived_at is None else f"◦{status_symbol}"

    metric = tree_metric_suffix(record, metric_name)

    labels: list[str] = []
    if node.commit == main_commit:
        labels.append("main")
    if node.commit == best_leaf:
        labels.append("best")
    label_prefix = f" ({', '.join(labels)})" if labels else ""

    metric_prefix = f"{metric.strip()} " if metric else ""
    return f"{prefix}{label_prefix} {format_commit(node.commit)} {metric_prefix}{node.name}".rstrip()


def render_show(
    record: NodeRecord,
    *,
    parent: NodeRecord | None,
    main: NodeRecord | None,
    format_metric_value: Callable[[str, float], str] = _default_metric_formatter,
) -> str:
    lines = [
        f"Commit:      {record.node.commit}",
        f"Parent:      {record.node.parent or '-'}",
        f"Name:        {record.node.name}",
        f"Status:      {record.node.status}",
        f"Verdict:     {record.node.verdict or '-'}",
        f"Archived:    {record.node.archived_at or '-'}",
        f"Worktree:    {record.node.worktree}",
        f"Created:     {record.node.created_at}",
        f"Completed:   {record.node.completed_at or '-'}",
        "",
        "Metrics:",
    ]
    if record.metrics:
        for name, value in sorted(record.metrics.items()):
            details: list[str] = []
            if parent and name in parent.metrics:
                details.append(f"parent: {format_metric_value(name, parent.metrics[name])}")
            if main and name in main.metrics:
                details.append(f"main: {format_metric_value(name, main.metrics[name])}")
            detail_suffix = f"  ({', '.join(details)})" if details else ""
            lines.append(f"  {name}: {format_metric_value(name, value)}{detail_suffix}")
    else:
        lines.append("  -")

    if record.node.hypothesis:
        lines.extend(["", "Hypothesis:", indent_block(record.node.hypothesis)])
    if record.node.analysis:
        lines.extend(["", "Analysis:", indent_block(record.node.analysis)])
    return "\n".join(lines)


def render_report(
    path: list[NodeRecord],
    *,
    metric_name: str | None,
    format_metric_value: Callable[[str, float], str] = _default_metric_formatter,
) -> str:
    commits = " → ".join(format_commit(record.node.commit) for record in path)
    start_metric = path[0].metrics.get(metric_name) if metric_name else None
    end_metric = path[-1].metrics.get(metric_name) if metric_name else None
    summary = f"{max(0, len(path) - 1)} experiments"
    if metric_name and start_metric is not None and end_metric is not None:
        summary += (
            f", {format_metric_value(metric_name, start_metric)}"
            f" → {format_metric_value(metric_name, end_metric)}"
        )

    lines = [
        f"═══ Path: {commits} ═══",
        f"═══ {summary} ═══",
        "",
    ]

    for index, record in enumerate(path):
        node = record.node
        lines.append(f"── {format_commit(node.commit)} {node.name} " + "─" * 30)
        if node.verdict:
            lines.append(f"verdict: {node.verdict}")
        if metric_name and metric_name in record.metrics:
            current = record.metrics[metric_name]
            previous = path[index - 1].metrics.get(metric_name) if index > 0 else None
            delta = metric_delta(current, previous)
            suffix = f" ({delta} from parent)" if delta is not None and index > 0 else ""
            lines.append(f"{metric_name}: {format_metric_value(metric_name, current)}{suffix}")

        extra_metrics = [
            (name, value)
            for name, value in sorted(record.metrics.items())
            if name != metric_name
        ]
        for name, value in extra_metrics:
            lines.append(f"{name}: {format_metric_value(name, value)}")

        if node.hypothesis:
            lines.extend(["", "Hypothesis:", indent_block(node.hypothesis)])
        if node.analysis:
            lines.extend(["", "Analysis:", indent_block(node.analysis)])
        if index != len(path) - 1:
            lines.append("")

    return "\n".join(lines)


def render_status(
    running: list[RemoteRunRecord],
    finished_remote: list[RemoteRunRecord],
    failed_remote: list[RemoteRunRecord],
    missing_remote: list[RemoteRunRecord],
    completed: list[NodeRecord],
    failed: list[NodeRecord],
    *,
    metric_name: str | None,
) -> str:
    lines: list[str] = []
    lines.append(f"Running ({len(running)}):")
    if running:
        for item in running:
            record = item.record
            elapsed = format_elapsed(record.node.created_at)
            lines.append(
                f"  {format_commit(record.node.commit)}  {record.node.name}  {elapsed} elapsed  log: {item.log_path}"
            )
    else:
        lines.append("  -")

    lines.append("")
    lines.append(f"Remote Finished, Awaiting `arc result` ({len(finished_remote)}):")
    if finished_remote:
        for item in finished_remote:
            record = item.record
            metric = (
                f"{metric_name}: {format_float(item.metrics.get(metric_name))}"
                if metric_name and metric_name in item.metrics
                else "finished"
            )
            lines.append(
                f"  {format_commit(record.node.commit)}  {record.node.name}  {metric}  log: {item.log_path}"
            )
    else:
        lines.append("  -")

    lines.append("")
    lines.append(f"Remote Missing Log / Unknown ({len(missing_remote)}):")
    if missing_remote:
        for item in missing_remote:
            record = item.record
            lines.append(f"  {format_commit(record.node.commit)}  {record.node.name}  log: {item.log_path}")
    else:
        lines.append("  -")

    lines.append("")
    lines.append(f"Remote Failed, Awaiting `arc fail` ({len(failed_remote)}):")
    if failed_remote:
        for item in failed_remote:
            record = item.record
            lines.append(f"  {format_commit(record.node.commit)}  {record.node.name}  log: {item.log_path}")
    else:
        lines.append("  -")

    lines.append("")
    lines.append(f"Completed since last check ({len(completed)}):")
    if completed:
        for record in completed:
            metric = (
                f"{metric_name}: {format_float(record.metrics[metric_name])}"
                if metric_name and metric_name in record.metrics
                else record.node.status
            )
            verdict = f"  [{record.node.verdict}]" if record.node.verdict else ""
            lines.append(f"  {format_commit(record.node.commit)}  {record.node.name}  {metric}{verdict}")
    else:
        lines.append("  -")

    lines.append("")
    lines.append(f"Failed since last check ({len(failed)}):")
    if failed:
        for record in failed:
            metric = (
                f"{metric_name}: {format_float(record.metrics[metric_name])}"
                if metric_name and metric_name in record.metrics
                else record.node.status
            )
            lines.append(f"  {format_commit(record.node.commit)}  {record.node.name}  {metric}")
    else:
        lines.append("  -")
    return "\n".join(lines)
