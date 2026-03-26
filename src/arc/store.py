from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

from arc.errors import ArcError
from arc.models import Node, NodeRecord, Status

STATUSES: tuple[Status, ...] = ("committed", "running", "completed", "failed")


class ArcStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def exists(self) -> bool:
        return self.db_path.exists()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    "commit" TEXT PRIMARY KEY,
                    "parent" TEXT REFERENCES nodes("commit"),
                    name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('committed', 'running', 'completed', 'failed')),
                    hypothesis TEXT,
                    analysis TEXT,
                    worktree TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes("parent");
                CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);

                CREATE TABLE IF NOT EXISTS metrics (
                    "commit" TEXT NOT NULL REFERENCES nodes("commit") ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY ("commit", name)
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def require_initialized(self) -> None:
        if not self.exists():
            raise ArcError("Arc is not initialized in this repository. Run `arc init` first.")

    def get_meta(self, key: str) -> str | None:
        self.require_initialized()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM meta WHERE key = ?",
                (key,),
            ).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        self.require_initialized()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO meta(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def insert_node(self, node: Node) -> None:
        self.require_initialized()
        with self.connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO nodes(
                        "commit", "parent", name, status, hypothesis, analysis,
                        worktree, created_at, completed_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node.commit,
                        node.parent,
                        node.name,
                        node.status,
                        node.hypothesis,
                        node.analysis,
                        node.worktree,
                        node.created_at,
                        node.completed_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ArcError(str(exc)) from exc

    def update_node(
        self,
        commit: str,
        *,
        status: Status | None = None,
        hypothesis: str | None = None,
        analysis: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        self.require_initialized()
        assignments: list[str] = []
        params: list[str | None] = []
        if status is not None:
            assignments.append("status = ?")
            params.append(status)
        if hypothesis is not None:
            assignments.append("hypothesis = ?")
            params.append(hypothesis)
        if analysis is not None:
            assignments.append("analysis = ?")
            params.append(analysis)
        if completed_at is not None:
            assignments.append("completed_at = ?")
            params.append(completed_at)
        if not assignments:
            return

        params.append(commit)
        with self.connect() as connection:
            connection.execute(
                f'UPDATE nodes SET {", ".join(assignments)} WHERE "commit" = ?',
                params,
            )

    def get_node(self, commit_prefix: str) -> Node | None:
        self.require_initialized()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM nodes
                WHERE "commit" = ? OR "commit" LIKE ?
                ORDER BY created_at
                """,
                (commit_prefix, f"{commit_prefix}%"),
            ).fetchall()

        if not rows:
            return None
        if len(rows) > 1:
            raise ArcError(f"Ambiguous commit prefix: {commit_prefix}")
        return _row_to_node(rows[0])

    def get_node_record(self, commit_prefix: str) -> NodeRecord | None:
        node = self.get_node(commit_prefix)
        if node is None:
            return None
        return NodeRecord(node=node, metrics=self.get_metrics(node.commit))

    def list_nodes(self) -> list[Node]:
        self.require_initialized()
        with self.connect() as connection:
            rows = connection.execute(
                'SELECT * FROM nodes ORDER BY created_at, "commit"'
            ).fetchall()
        return [_row_to_node(row) for row in rows]

    def list_node_records(self) -> list[NodeRecord]:
        nodes = self.list_nodes()
        metrics = self.metrics_by_commit([node.commit for node in nodes])
        return [NodeRecord(node=node, metrics=metrics.get(node.commit, {})) for node in nodes]

    def path_to_root(self, commit_prefix: str) -> list[NodeRecord]:
        record = self.get_node_record(commit_prefix)
        if record is None:
            raise ArcError(f"Unknown commit: {commit_prefix}")

        chain: list[Node] = []
        current = record.node
        while current is not None:
            chain.append(current)
            if current.parent is None:
                break
            parent = self.get_node(current.parent)
            if parent is None:
                raise ArcError(f"Missing parent node for commit {current.commit}")
            current = parent
        chain.reverse()

        metrics = self.metrics_by_commit([node.commit for node in chain])
        return [NodeRecord(node=node, metrics=metrics.get(node.commit, {})) for node in chain]

    def upsert_metrics(self, commit: str, metrics: dict[str, float]) -> None:
        if not metrics:
            return
        self.require_initialized()
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO metrics("commit", name, value)
                VALUES(?, ?, ?)
                ON CONFLICT("commit", name) DO UPDATE SET value = excluded.value
                """,
                [(commit, name, value) for name, value in metrics.items()],
            )

    def get_metrics(self, commit: str) -> dict[str, float]:
        self.require_initialized()
        with self.connect() as connection:
            rows = connection.execute(
                'SELECT name, value FROM metrics WHERE "commit" = ? ORDER BY name',
                (commit,),
            ).fetchall()
        return {str(row["name"]): float(row["value"]) for row in rows}

    def metrics_by_commit(self, commits: Sequence[str]) -> dict[str, dict[str, float]]:
        if not commits:
            return {}
        self.require_initialized()
        placeholders = ", ".join("?" for _ in commits)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT "commit", name, value
                FROM metrics
                WHERE "commit" IN ({placeholders})
                ORDER BY "commit", name
                """,
                tuple(commits),
            ).fetchall()

        results: dict[str, dict[str, float]] = {}
        for row in rows:
            commit = str(row["commit"])
            results.setdefault(commit, {})[str(row["name"])] = float(row["value"])
        return results

    def list_by_status(self, statuses: Iterable[Status]) -> list[NodeRecord]:
        wanted = tuple(statuses)
        if not wanted:
            return []
        placeholders = ", ".join("?" for _ in wanted)
        self.require_initialized()
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM nodes
                WHERE status IN ({placeholders})
                ORDER BY created_at, "commit"
                """,
                wanted,
            ).fetchall()

        commits = [str(row["commit"]) for row in rows]
        metrics = self.metrics_by_commit(commits)
        return [
            NodeRecord(node=_row_to_node(row), metrics=metrics.get(str(row["commit"]), {}))
            for row in rows
        ]

    def list_completed_since(self, timestamp: str | None) -> list[NodeRecord]:
        self.require_initialized()
        query = """
            SELECT *
            FROM nodes
            WHERE status IN ('completed', 'failed')
        """
        params: tuple[str, ...] = ()
        if timestamp:
            query += " AND completed_at > ?"
            params = (timestamp,)
        query += ' ORDER BY completed_at, "commit"'

        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        commits = [str(row["commit"]) for row in rows]
        metrics = self.metrics_by_commit(commits)
        return [
            NodeRecord(node=_row_to_node(row), metrics=metrics.get(str(row["commit"]), {}))
            for row in rows
        ]


def _row_to_node(row: sqlite3.Row) -> Node:
    return Node(
        commit=str(row["commit"]),
        parent=None if row["parent"] is None else str(row["parent"]),
        name=str(row["name"]),
        status=str(row["status"]),
        hypothesis=None if row["hypothesis"] is None else str(row["hypothesis"]),
        analysis=None if row["analysis"] is None else str(row["analysis"]),
        worktree=str(row["worktree"]),
        created_at=str(row["created_at"]),
        completed_at=None if row["completed_at"] is None else str(row["completed_at"]),
    )
