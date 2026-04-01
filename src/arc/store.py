from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

from arc.errors import ArcError
from arc.models import Node, NodeRecord, Status, Verdict

STATUSES: tuple[Status, ...] = ("committed", "running", "completed", "failed")
VERDICTS: tuple[Verdict, ...] = (
    "promising",
    "regression",
    "neutral",
    "inconclusive",
    "invalid",
    "unsupported",
)
NODE_COLUMNS = """
    nodes."commit",
    nodes."parent",
    nodes.name,
    nodes.status,
    nodes.hypothesis,
    nodes.analysis,
    nodes.worktree,
    nodes.created_at,
    nodes.completed_at,
    nodes.verdict,
    archived_nodes.archived_at AS archived_at
"""
NODE_JOIN = """
    FROM nodes
    LEFT JOIN archived_nodes ON archived_nodes."commit" = nodes."commit"
"""
UNSET = object()


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

    def initialize(self, repo_root: Path | None = None) -> None:
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
                    completed_at TEXT,
                    verdict TEXT CHECK (
                        verdict IN (
                            'promising',
                            'regression',
                            'neutral',
                            'inconclusive',
                            'invalid',
                            'unsupported'
                        )
                    )
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

                CREATE TABLE IF NOT EXISTS archived_nodes (
                    "commit" TEXT PRIMARY KEY REFERENCES nodes("commit") ON DELETE CASCADE,
                    archived_at TEXT NOT NULL
                );
                """
            )
            self._ensure_nodes_schema(connection)
            if repo_root is not None:
                self._normalize_commit_ids(connection, repo_root)

    def _normalize_commit_ids(self, connection: sqlite3.Connection, repo_root: Path) -> None:
        from arc.git import full_rev

        rows = connection.execute('SELECT "commit", "parent" FROM nodes').fetchall()
        if not rows:
            return

        mapping: dict[str, str] = {}
        for row in rows:
            commit = str(row["commit"])
            mapping.setdefault(commit, full_rev(repo_root, commit))
            if row["parent"] is not None:
                parent = str(row["parent"])
                mapping.setdefault(parent, full_rev(repo_root, parent))

        changed = {old: new for old, new in mapping.items() if old != new}
        if not changed:
            return

        placeholders = {old: f"__arc_migrate__{index}__" for index, old in enumerate(changed, start=1)}
        connection.execute("PRAGMA foreign_keys = OFF")
        try:
            for old, placeholder in placeholders.items():
                connection.execute('UPDATE nodes SET "commit" = ? WHERE "commit" = ?', (placeholder, old))
                connection.execute('UPDATE nodes SET "parent" = ? WHERE "parent" = ?', (placeholder, old))
                connection.execute('UPDATE metrics SET "commit" = ? WHERE "commit" = ?', (placeholder, old))
                connection.execute(
                    'UPDATE archived_nodes SET "commit" = ? WHERE "commit" = ?',
                    (placeholder, old),
                )
                connection.execute(
                    'UPDATE meta SET value = ? WHERE key = ? AND value = ?',
                    (placeholder, "main", old),
                )

            for old, new in changed.items():
                placeholder = placeholders[old]
                connection.execute('UPDATE nodes SET "commit" = ? WHERE "commit" = ?', (new, placeholder))
                connection.execute('UPDATE nodes SET "parent" = ? WHERE "parent" = ?', (new, placeholder))
                connection.execute('UPDATE metrics SET "commit" = ? WHERE "commit" = ?', (new, placeholder))
                connection.execute(
                    'UPDATE archived_nodes SET "commit" = ? WHERE "commit" = ?',
                    (new, placeholder),
                )
                connection.execute(
                    'UPDATE meta SET value = ? WHERE key = ? AND value = ?',
                    (new, "main", placeholder),
                )
        finally:
            connection.execute("PRAGMA foreign_keys = ON")

    def _ensure_nodes_schema(self, connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute('PRAGMA table_info("nodes")').fetchall()
        }
        if "verdict" not in columns or not self._nodes_table_supports_verdicts(connection):
            self._rebuild_nodes_tables(connection, has_verdict="verdict" in columns)

    def _nodes_table_supports_verdicts(self, connection: sqlite3.Connection) -> bool:
        row = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND name = 'nodes'
            """
        ).fetchone()
        if row is None or row["sql"] is None:
            return False
        create_sql = str(row["sql"]).lower()
        if "check" not in create_sql:
            return True
        return all(
            verdict in create_sql
            for verdict in ("'regression'", "'neutral'", "'inconclusive'", "'invalid'")
        )

    def _rebuild_nodes_tables(
        self,
        connection: sqlite3.Connection,
        *,
        has_verdict: bool,
    ) -> None:
        connection.execute("PRAGMA foreign_keys = OFF")
        try:
            verdict_backup = ", verdict" if has_verdict else ", NULL AS verdict"
            connection.executescript(
                f"""
                CREATE TABLE nodes_backup AS
                SELECT
                    "commit",
                    "parent",
                    name,
                    status,
                    hypothesis,
                    analysis,
                    worktree,
                    created_at,
                    completed_at
                    {verdict_backup}
                FROM nodes;

                CREATE TABLE metrics_backup AS
                SELECT "commit", name, value
                FROM metrics;

                CREATE TABLE archived_nodes_backup AS
                SELECT "commit", archived_at
                FROM archived_nodes;

                DROP TABLE metrics;
                DROP TABLE archived_nodes;
                DROP TABLE nodes;

                CREATE TABLE nodes (
                    "commit" TEXT PRIMARY KEY,
                    "parent" TEXT REFERENCES nodes("commit"),
                    name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('committed', 'running', 'completed', 'failed')),
                    hypothesis TEXT,
                    analysis TEXT,
                    worktree TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    verdict TEXT CHECK (
                        verdict IN (
                            'promising',
                            'regression',
                            'neutral',
                            'inconclusive',
                            'invalid',
                            'unsupported'
                        )
                    )
                );

                CREATE INDEX idx_nodes_parent ON nodes("parent");
                CREATE INDEX idx_nodes_status ON nodes(status);

                CREATE TABLE metrics (
                    "commit" TEXT NOT NULL REFERENCES nodes("commit") ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY ("commit", name)
                );

                CREATE TABLE archived_nodes (
                    "commit" TEXT PRIMARY KEY REFERENCES nodes("commit") ON DELETE CASCADE,
                    archived_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """
                INSERT INTO nodes(
                    "commit", "parent", name, status, hypothesis, analysis,
                    worktree, created_at, completed_at, verdict
                )
                SELECT
                    "commit", "parent", name, status, hypothesis, analysis,
                    worktree, created_at, completed_at, verdict
                FROM nodes_backup
                """
            )
            connection.executescript(
                """
                INSERT INTO metrics("commit", name, value)
                SELECT "commit", name, value
                FROM metrics_backup;

                INSERT INTO archived_nodes("commit", archived_at)
                SELECT "commit", archived_at
                FROM archived_nodes_backup;

                DROP TABLE nodes_backup;
                DROP TABLE metrics_backup;
                DROP TABLE archived_nodes_backup;
                """
            )
        finally:
            connection.execute("PRAGMA foreign_keys = ON")

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
                        worktree, created_at, completed_at, verdict
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        node.verdict,
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
        verdict: Verdict | None | object = UNSET,
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
        if verdict is not UNSET:
            assignments.append("verdict = ?")
            params.append(verdict)
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
                SELECT
                """
                + NODE_COLUMNS
                + NODE_JOIN
                + """
                WHERE nodes."commit" = ? OR nodes."commit" LIKE ?
                ORDER BY nodes.created_at
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

    def list_nodes(self, *, include_archived: bool = True) -> list[Node]:
        self.require_initialized()
        where = "" if include_archived else 'WHERE archived_nodes."commit" IS NULL'
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                """
                + NODE_COLUMNS
                + NODE_JOIN
                + f"""
                {where}
                ORDER BY nodes.created_at, nodes."commit"
                """
            ).fetchall()
        return [_row_to_node(row) for row in rows]

    def list_node_records(self, *, include_archived: bool = True) -> list[NodeRecord]:
        nodes = self.list_nodes(include_archived=include_archived)
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

    def list_by_status(
        self,
        statuses: Iterable[Status],
        *,
        include_archived: bool = False,
    ) -> list[NodeRecord]:
        wanted = tuple(statuses)
        if not wanted:
            return []
        placeholders = ", ".join("?" for _ in wanted)
        archived_clause = "" if include_archived else 'AND archived_nodes."commit" IS NULL'
        self.require_initialized()
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                {NODE_COLUMNS}
                {NODE_JOIN}
                WHERE nodes.status IN ({placeholders})
                {archived_clause}
                ORDER BY nodes.created_at, nodes."commit"
                """,
                wanted,
            ).fetchall()

        commits = [str(row["commit"]) for row in rows]
        metrics = self.metrics_by_commit(commits)
        return [
            NodeRecord(node=_row_to_node(row), metrics=metrics.get(str(row["commit"]), {}))
            for row in rows
        ]

    def list_recent_by_status(
        self,
        statuses: Iterable[Status],
        timestamp: str | None,
        *,
        include_archived: bool = False,
    ) -> list[NodeRecord]:
        wanted = tuple(statuses)
        if not wanted:
            return []
        self.require_initialized()
        placeholders = ", ".join("?" for _ in wanted)
        query = """
            SELECT
        """
        query += NODE_COLUMNS
        query += NODE_JOIN
        query += f"""
            WHERE nodes.status IN ({placeholders})
        """
        if not include_archived:
            query += ' AND archived_nodes."commit" IS NULL'
        params: tuple[str, ...] = wanted
        if timestamp:
            query += " AND nodes.completed_at > ?"
            params = (*wanted, timestamp)
        query += ' ORDER BY nodes.completed_at, nodes."commit"'

        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        commits = [str(row["commit"]) for row in rows]
        metrics = self.metrics_by_commit(commits)
        return [
            NodeRecord(node=_row_to_node(row), metrics=metrics.get(str(row["commit"]), {}))
            for row in rows
        ]

    def has_children(self, commit: str) -> bool:
        self.require_initialized()
        with self.connect() as connection:
            row = connection.execute(
                'SELECT 1 FROM nodes WHERE "parent" = ? LIMIT 1',
                (commit,),
            ).fetchone()
        return row is not None

    def archive_node(self, commit: str, archived_at: str) -> None:
        self.require_initialized()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO archived_nodes("commit", archived_at)
                VALUES(?, ?)
                ON CONFLICT("commit") DO UPDATE SET archived_at = excluded.archived_at
                """,
                (commit, archived_at),
            )


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
        verdict=None if row["verdict"] is None else str(row["verdict"]),
        archived_at=None if row["archived_at"] is None else str(row["archived_at"]),
    )
