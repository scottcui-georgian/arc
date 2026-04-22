from __future__ import annotations

import argparse
import os
import shutil
import subprocess

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.runlog import latest_modal_app_id, summarize_run_log
from arc.tasks.parameter_golf.runtime import load_dotenv_into
from arc.timeutil import utc_now_iso


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("commit", help="Experiment commit hash or prefix.")


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    record = app.store.get_node_record(args.commit)
    if record is None:
        raise ArcError(f"Unknown commit: {args.commit}")
    if record.node.status != "running":
        raise ArcError(f"Expected status `running`, got `{record.node.status}`.")

    log_path = app.node_log_path(record.node)
    if not log_path.is_file():
        raise ArcError(f"No run log found for `{record.node.commit}` at {app.relative_path(log_path)}.")

    summary = summarize_run_log(log_path)
    if summary.state == "finished":
        raise ArcError("Remote job already finished; use `arc result` to record it.")
    if summary.state == "failed":
        raise ArcError("Remote job already failed; use `arc fail` to record it.")
    if summary.state == "missing":
        raise ArcError(f"Run log is missing at {app.relative_path(log_path)}.")

    app_id = latest_modal_app_id(log_path)
    if app_id is None:
        raise ArcError(
            "Could not find a Modal app ID in the latest run log section. "
            "The run may have exited before Modal initialized."
        )

    modal_path = shutil.which("modal")
    if modal_path is None:
        raise ArcError("`modal` is not on PATH.")

    env = os.environ.copy()
    load_dotenv_into(env, app.node_worktree_path(record.node) / ".env")
    proc = subprocess.run(
        [modal_path, "app", "stop", app_id],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        if detail:
            raise ArcError(f"Modal app stop failed: {detail}")
        raise ArcError(f"Modal app stop failed with exit code {proc.returncode}.")

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now_iso()}] modal app stop requested for {app_id} via arc interrupt\n")

    print(f"Interrupted {app.display_commit(record.node.commit)} ({record.node.name})")
    print(f"Modal app: {app_id}")
    print(f"Log:       {app.relative_path(log_path)}")
    print("Next:      use `arc status`, then record the interrupted run with `arc fail`.")
    return 0


COMMAND = CommandSpec(
    name="interrupt",
    help="Stop a running Modal-backed experiment by commit.",
    register=register,
    run=run,
)
