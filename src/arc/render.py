from __future__ import annotations

from collections import defaultdict

from arc.models import Direction, Node, NodeRecord
from arc.text import format_float, format_signed_delta, indent_block
from arc.timeutil import format_elapsed


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
        if not children.get(record.node.commit) and record.node.status == "completed"
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
    depth: int | None,
    leaves_only: bool,
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
    if status_filter or leaves_only:
        for record in records:
            is_leaf = not children.get(record.node.commit)
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

    def visit(record: NodeRecord, prefix: str, is_last: bool, level: int) -> None:
        if record.node.commit not in selected:
            return
        branch = ""
        if level > 0:
            branch = "└── " if is_last else "├── "
        lines.append(f"{prefix}{branch}{_tree_label(record, metric_name, best_leaf, main_commit)}")

        if depth is not None and level >= depth:
            return

        visible_children = [
            child for child in children.get(record.node.commit, [])
            if child.node.commit in selected
        ]
        next_prefix = "" if level == 0 else prefix + ("    " if is_last else "│   ")
        for index, child in enumerate(visible_children):
            visit(
                child,
                next_prefix,
                index == len(visible_children) - 1,
                level + 1,
            )

    for index, root in enumerate(roots):
        visit(root, "", index == len(roots) - 1, 0)
    return "\n".join(lines)


def _tree_label(
    record: NodeRecord,
    metric_name: str | None,
    best_leaf: str | None,
    main_commit: str | None,
) -> str:
    node = record.node
    symbol = ""
    if node.parent is None:
        symbol = "●"
    elif node.commit == main_commit:
        symbol = "★"
    elif node.status == "running":
        symbol = "◌"
    elif node.commit == best_leaf:
        symbol = "✓"

    metric = ""
    if metric_name and metric_name in record.metrics:
        metric = f" ({format_float(record.metrics[metric_name])})"

    suffix = ""
    if node.commit == best_leaf and node.parent is not None:
        suffix = "best"
    if node.commit == main_commit and node.parent is not None:
        suffix = "main"
    if node.status == "failed":
        suffix = "failed"

    left = f"{symbol} " if symbol else ""
    right = f"{metric}{f' {suffix}' if suffix else ''}"
    return f"{left}{node.commit} {node.name}{right}".rstrip()


def render_show(
    record: NodeRecord,
    *,
    parent: NodeRecord | None,
    main: NodeRecord | None,
) -> str:
    lines = [
        f"Commit:      {record.node.commit}",
        f"Parent:      {record.node.parent or '-'}",
        f"Name:        {record.node.name}",
        f"Status:      {record.node.status}",
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
                details.append(f"parent: {format_float(parent.metrics[name])}")
            if main and name in main.metrics:
                details.append(f"main: {format_float(main.metrics[name])}")
            detail_suffix = f"  ({', '.join(details)})" if details else ""
            lines.append(f"  {name}: {format_float(value)}{detail_suffix}")
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
) -> str:
    commits = " → ".join(record.node.commit for record in path)
    start_metric = path[0].metrics.get(metric_name) if metric_name else None
    end_metric = path[-1].metrics.get(metric_name) if metric_name else None
    summary = f"{max(0, len(path) - 1)} experiments"
    if metric_name and start_metric is not None and end_metric is not None:
        summary += f", {format_float(start_metric)} → {format_float(end_metric)}"

    lines = [
        f"═══ Path: {commits} ═══",
        f"═══ {summary} ═══",
        "",
    ]

    for index, record in enumerate(path):
        node = record.node
        lines.append(f"── {node.commit} {node.name} " + "─" * 30)
        if metric_name and metric_name in record.metrics:
            current = record.metrics[metric_name]
            previous = path[index - 1].metrics.get(metric_name) if index > 0 else None
            delta = metric_delta(current, previous)
            suffix = f" ({delta} from parent)" if delta is not None and index > 0 else ""
            lines.append(f"{metric_name}: {format_float(current)}{suffix}")

        extra_metrics = [
            (name, value)
            for name, value in sorted(record.metrics.items())
            if name != metric_name
        ]
        for name, value in extra_metrics:
            lines.append(f"{name}: {format_float(value)}")

        if node.hypothesis:
            lines.extend(["", "Hypothesis:", indent_block(node.hypothesis)])
        if node.analysis:
            lines.extend(["", "Analysis:", indent_block(node.analysis)])
        if index != len(path) - 1:
            lines.append("")

    return "\n".join(lines)


def render_status(
    running: list[NodeRecord],
    completed: list[NodeRecord],
    *,
    metric_name: str | None,
) -> str:
    lines: list[str] = []
    lines.append(f"Running ({len(running)}):")
    if running:
        for record in running:
            elapsed = format_elapsed(record.node.created_at)
            lines.append(f"  {record.node.commit}  {record.node.name}  {elapsed} elapsed")
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
            lines.append(f"  {record.node.commit}  {record.node.name}  {metric}")
    else:
        lines.append("  -")
    return "\n".join(lines)
