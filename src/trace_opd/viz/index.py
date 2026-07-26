"""SQLite index over runs/ artifacts (incremental, read-only towards runs/).

Artifacts are append-only/immutable (dumps carry a fresh uuid, research dirs a
timestamp), so incremental scanning is set-difference on paths with directory
mtime pruning. Corrupt/half-written files are skipped and retried next scan.
Filter/aggregate columns live in the DB; large text (thoughts, docs) is read
from the files on demand.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from . import plugins, schema

# Bump whenever the derived schema below changes shape. The DB is a pure cache
# over runs/, so a mismatch just drops it and rescans — much safer than leaving
# a stale layout to fail on a column count at insert time.
_SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rollout(
  run TEXT, file TEXT, step INTEGER, validate INTEGER, split TEXT,
  task_id TEXT, family TEXT, outcome TEXT, won INTEGER, env_steps INTEGER,
  assistant_turns INTEGER, invalid_format INTEGER, truncated INTEGER,
  doc_words INTEGER, research_final_success REAL, research_error INTEGER,
  doc_missing INTEGER, research_archive TEXT,
  n_steps INTEGER, n_invalid INTEGER, n_noact INTEGER, max_repeat INTEGER,
  sample_idx INTEGER, fail_label TEXT, plugin_ver TEXT, tags TEXT, ts REAL,
  sample_role TEXT,
  PRIMARY KEY(run, file));
CREATE INDEX IF NOT EXISTS idx_rollout_step ON rollout(run, validate, step);
CREATE INDEX IF NOT EXISTS idx_rollout_task ON rollout(run, task_id);
CREATE TABLE IF NOT EXISTS research(
  run TEXT, dir TEXT, task_id TEXT, family TEXT, policy_version INTEGER,
  ts REAL, doc_present INTEGER, doc_words INTEGER, experiments_used INTEGER,
  episodes_consumed INTEGER, final_success REAL, finalized INTEGER,
  session_error TEXT, researcher_model TEXT,
  PRIMARY KEY(run, dir));
CREATE INDEX IF NOT EXISTS idx_research_task ON research(run, task_id);
CREATE TABLE IF NOT EXISTS dir_state(
  run TEXT, dir TEXT, mtime REAL, PRIMARY KEY(run, dir));
CREATE TABLE IF NOT EXISTS scan_state(
  run TEXT PRIMARY KEY, last_scan REAL, n_skipped INTEGER);
"""


def discover_runs(runs_root: Path) -> list[str]:
    if not runs_root.is_dir():
        return []
    return sorted(
        d.name for d in runs_root.iterdir()
        if d.is_dir() and ((d / "rollouts").is_dir() or (d / "research").is_dir())
    )


class Index:
    def __init__(self, runs_root: Path, db_path: Path) -> None:
        self.runs_root = runs_root.resolve()
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version != _SCHEMA_VERSION:
                for table in ("rollout", "research", "dir_state", "scan_state"):
                    db.execute(f"DROP TABLE IF EXISTS {table}")
                db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            db.executescript(_SCHEMA)

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    # ---------------------------------------------------------------- scan

    def scan(self) -> dict[str, Any]:
        report: dict[str, Any] = {}
        for run in discover_runs(self.runs_root):
            report[run] = self._scan_run(run)
        return report

    def _scan_run(self, run: str) -> dict[str, int]:
        added = skipped = 0
        with self.connect() as db:
            added_r, skipped_r = self._scan_rollouts(db, run)
            added_s, skipped_s = self._scan_research(db, run)
            self._reclassify_stale(db, run)
            added, skipped = added_r + added_s, skipped_r + skipped_s
            db.execute(
                "INSERT OR REPLACE INTO scan_state VALUES (?,?,?)",
                (run, time.time(), skipped),
            )
            db.commit()
        return {"added": added, "skipped": skipped}

    def _dir_changed(self, db: sqlite3.Connection, run: str, path: Path) -> bool:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return False
        row = db.execute(
            "SELECT mtime FROM dir_state WHERE run=? AND dir=?", (run, str(path))
        ).fetchone()
        if row is not None and row["mtime"] == mtime:
            return False
        db.execute(
            "INSERT OR REPLACE INTO dir_state VALUES (?,?,?)", (run, str(path), mtime)
        )
        return True

    def _scan_rollouts(self, db: sqlite3.Connection, run: str) -> tuple[int, int]:
        root = self.runs_root / run / "rollouts"
        added = skipped = 0
        if not root.is_dir():
            return 0, 0
        for step_dir in sorted(root.iterdir()):
            parsed = schema.parse_step_dir(step_dir.name)
            if parsed is None or not self._dir_changed(db, run, step_dir):
                continue
            step, validate = parsed
            known = {
                r["file"]
                for r in db.execute(
                    "SELECT file FROM rollout WHERE run=? AND step=? AND validate=?",
                    (run, step, int(validate)),
                )
            }
            for f in step_dir.iterdir():
                rel = f"rollouts/{step_dir.name}/{f.name}"
                if not f.name.endswith(".json") or rel in known:
                    continue
                row = self._rollout_row(run, rel, f, step, validate)
                if row is None:  # half-written dump; forget the dir mtime so
                    skipped += 1  # the next scan re-lists and retries it
                    db.execute(
                        "DELETE FROM dir_state WHERE run=? AND dir=?",
                        (run, str(step_dir)),
                    )
                    continue
                db.execute(
                    f"INSERT OR REPLACE INTO rollout VALUES ({','.join('?' * len(row))})",
                    row,
                )
                added += 1
        return added, skipped

    def _rollout_row(
        self, run: str, rel: str, path: Path, step: int, validate: bool
    ) -> tuple | None:
        d = schema.read_json(path)
        if d is None or "task_id" not in d:
            return None
        traj = d.get("trajectory") or {}
        stats = schema.traj_stats(traj)
        task_id = str(d["task_id"])
        won = bool(d.get("won"))
        plugin = plugins.plugin_for(task_id)
        tags = ",".join(plugin.traj_tags(traj)) if plugin is not None else ""
        research = d.get("research") or {}
        return (
            run, rel, step, int(validate), str(d.get("split", "")),
            task_id, plugins.task_family(task_id), str(d.get("outcome", "")),
            int(won), int(d.get("env_steps") or 0),
            int(d.get("assistant_turns") or 0), int(d.get("invalid_format") or 0),
            int(bool(d.get("truncated"))), int(d.get("doc_words") or 0),
            float(research.get("research_final_success") or 0.0),
            int(research.get("research_error") or 0),
            int(research.get("doc_missing") or 0),
            str(research.get("research_archive") or ""),
            stats.n_steps, stats.n_invalid, stats.n_noact, stats.max_repeat,
            int(traj.get("sample_idx") or 0),
            plugins.classify(task_id, traj, won), plugins.classifier_version(task_id),
            tags, float(d.get("ts") or 0.0),
            # "bare" = an on-policy student rollout; "sft" = an expert rollout of
            # the mixed objective, produced WITH the privileged document. Every
            # aggregate in queries.py reads on-policy rows only — expert
            # trajectories succeed ~96% of the time and would otherwise show up
            # as a jump in the score curve that the policy never earned.
            str(d.get("sample_role") or "bare"),
        )

    def _scan_research(self, db: sqlite3.Connection, run: str) -> tuple[int, int]:
        root = self.runs_root / run / "research"
        added = skipped = 0
        if not root.is_dir():
            return 0, 0
        for task_dir in sorted(root.iterdir()):
            if not task_dir.is_dir() or not self._dir_changed(db, run, task_dir):
                continue
            known = {
                r["dir"]
                for r in db.execute(
                    "SELECT dir FROM research WHERE run=? AND dir LIKE ?",
                    (run, f"research/{task_dir.name}/%"),
                )
            }
            for session_dir in task_dir.iterdir():
                rel = f"research/{task_dir.name}/{session_dir.name}"
                if not session_dir.is_dir() or rel in known:
                    continue
                meta = schema.research_meta(session_dir)
                if meta is None:  # half-written session; retry next scan
                    skipped += 1
                    # forget the task-dir mtime so the next scan re-lists it
                    db.execute(
                        "DELETE FROM dir_state WHERE run=? AND dir=?",
                        (run, str(task_dir)),
                    )
                    continue
                task_id = str(meta.get("task_id", task_dir.name))
                att = meta.get("last_exp_attempts") or 0
                final = (meta.get("last_exp_successes") or 0) / att if att else 0.0
                db.execute(
                    "INSERT OR REPLACE INTO research VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run, rel, task_id, plugins.task_family(task_id),
                        meta.get("policy_version"), float(meta.get("ts") or 0.0),
                        int((session_dir / "doc.md").is_file()),
                        int(meta.get("doc_words") or 0),
                        int(meta.get("experiments_used") or 0),
                        int(meta.get("episodes_consumed") or 0),
                        float(final), int(bool(meta.get("finalized"))),
                        str(meta.get("session_error") or ""),
                        str(meta.get("researcher_model") or ""),
                    ),
                )
                added += 1
        return added, skipped

    def _reclassify_stale(self, db: sqlite3.Connection, run: str) -> None:
        """Plugin version bumps invalidate cached failure labels."""
        stale = [
            (r["file"], r["task_id"])
            for r in db.execute(
                "SELECT file, task_id, plugin_ver FROM rollout WHERE run=? AND won=0",
                (run,),
            )
            if r["plugin_ver"] != plugins.classifier_version(r["task_id"])
        ]
        for rel, task_id in stale:
            d = schema.read_json(self.runs_root / run / rel)
            if d is None:
                continue
            traj = d.get("trajectory") or {}
            plugin = plugins.plugin_for(task_id)
            db.execute(
                "UPDATE rollout SET fail_label=?, plugin_ver=?, tags=? WHERE run=? AND file=?",
                (
                    plugins.classify(task_id, traj, False),
                    plugins.classifier_version(task_id),
                    ",".join(plugin.traj_tags(traj)) if plugin is not None else "",
                    run, rel,
                ),
            )
