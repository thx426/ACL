"""Query layer: every aggregate ships provenance (value, n, refs, criteria).

Views, deep-links and (P1) figure specs all serialize the same query shapes,
so a number on screen, its URL, and its exported citation can never drift
apart. Success/failure stratification is built in — aggregate-only medians
misled once already (CD-006's env_steps=30 incident).
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from . import plugins, schema
from .index import Index

# Whitelisted per-step curve definitions over the rollout table.
# expr must be a pure SQL aggregate over rollout columns.
CURVES: dict[str, dict[str, str]] = {
    "score": {"expr": "AVG(won)", "label": "success rate (sampled)"},
    "env_steps": {"expr": "AVG(env_steps)", "label": "env steps (all)"},
    "env_steps_success": {
        "expr": "AVG(CASE WHEN won=1 THEN env_steps END)",
        "label": "env steps (success only)",
    },
    "env_steps_failed": {
        "expr": "AVG(CASE WHEN won=0 THEN env_steps END)",
        "label": "env steps (failed only)",
    },
    "invalid_format": {"expr": "AVG(invalid_format)", "label": "invalid-format count"},
    "truncated": {"expr": "AVG(truncated)", "label": "truncation rate"},
    "max_repeat": {"expr": "AVG(max_repeat)", "label": "longest action repeat"},
    "invalid_rate": {
        "expr": "AVG(CAST(n_invalid AS REAL)/MAX(n_steps,1))",
        "label": "invalid-step share",
    },
    "research_final_success": {
        "expr": "AVG(research_final_success)",
        "label": "researcher final success (doc quality)",
    },
    "doc_words": {"expr": "AVG(doc_words)", "label": "doc words"},
    "doc_missing": {"expr": "AVG(doc_missing)", "label": "doc missing rate"},
    "research_error": {"expr": "AVG(research_error)", "label": "research error rate"},
}


def _median(values: list) -> float | None:
    return round(statistics.median(values), 2) if values else None


def _stratified(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The always-on success/failure split every listing view shows."""
    won = [r for r in rows if r["won"]]
    lost = [r for r in rows if not r["won"]]
    return {
        "n": len(rows),
        "n_success": len(won),
        "success_rate": round(len(won) / len(rows), 4) if rows else None,
        "env_steps_success_median": _median([r["env_steps"] for r in won]),
        "env_steps_failed_median": _median([r["env_steps"] for r in lost]),
        "fail_labels": _distribution(lost, "fail_label"),
        "families": _distribution(rows, "family"),
    }


def _distribution(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for r in rows:
        value = r.get(key) or ""
        counts[value] = counts.get(value, 0) + 1
    return [
        {"value": k, "count": v}
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
    ]


def citation(value: Any, n: int, scope: str, criteria: str) -> str:
    return f"{value} (n={n}; {scope}; {criteria})"


class Queries:
    def __init__(self, index: Index) -> None:
        self.index = index

    # ------------------------------------------------------------- run level

    def runs(self) -> list[dict[str, Any]]:
        out = []
        with self.index.connect() as db:
            for run in sorted(
                {r["run"] for r in db.execute("SELECT DISTINCT run FROM rollout")}
                | {r["run"] for r in db.execute("SELECT DISTINCT run FROM research")}
            ):
                agg = db.execute(
                    "SELECT COUNT(*) AS n, MAX(step) AS last_step, MAX(ts) AS last_ts,"
                    " AVG(CASE WHEN validate=0 THEN won END) AS train_score"
                    " FROM rollout WHERE run=?",
                    (run,),
                ).fetchone()
                scan = db.execute(
                    "SELECT last_scan FROM scan_state WHERE run=?", (run,)
                ).fetchone()
                out.append(
                    {
                        "run": run,
                        "samples": agg["n"],
                        "last_step": agg["last_step"],
                        "last_ts": agg["last_ts"],
                        "train_score": agg["train_score"],
                        "last_scan": scan["last_scan"] if scan else None,
                    }
                )
        return out

    def overview(self, run: str) -> dict[str, Any]:
        with self.index.connect() as db:
            per_step = [
                dict(r)
                for r in db.execute(
                    "SELECT step, validate, COUNT(*) AS n, SUM(won) AS n_success"
                    " FROM rollout WHERE run=? GROUP BY step, validate ORDER BY step",
                    (run,),
                )
            ]
            research = db.execute(
                "SELECT COUNT(*) AS n, SUM(doc_present) AS docs,"
                " AVG(final_success) AS final_success, SUM(experiments_used) AS experiments"
                " FROM research WHERE run=?",
                (run,),
            ).fetchone()
            scan = db.execute(
                "SELECT last_scan, n_skipped FROM scan_state WHERE run=?", (run,)
            ).fetchone()
        train = [r for r in per_step if not r["validate"]]
        # Integrity audit: expected per-step count = mode of observed counts;
        # deviations flag silently missing dumps (or a partial current step).
        counts = [r["n"] for r in train]
        expected = statistics.mode(counts) if counts else None
        for r in per_step:
            r["short"] = bool(expected and not r["validate"] and r["n"] < expected)
        return {
            "run": run,
            "steps": per_step,
            "expected_per_step": expected,
            "research": dict(research) if research else {},
            "scan": dict(scan) if scan else {},
            "curve_keys": list(CURVES),
        }

    # ---------------------------------------------------------------- curves

    def curves(self, run: str, keys: list[str], mode: str) -> dict[str, Any]:
        bad = [k for k in keys if k not in CURVES]
        if bad:
            raise KeyError(f"unknown curve keys: {bad}")
        series: dict[str, Any] = {}
        with self.index.connect() as db:
            for key in keys:
                expr = CURVES[key]["expr"]
                if mode == "train":
                    rows = db.execute(
                        f"SELECT step, {expr} AS v, COUNT(*) AS n FROM rollout"
                        " WHERE run=? AND validate=0 GROUP BY step ORDER BY step",
                        (run,),
                    ).fetchall()
                    series[key] = {
                        "label": CURVES[key]["label"],
                        "points": [
                            {"step": r["step"], "value": r["v"], "n": r["n"]}
                            for r in rows
                            if r["v"] is not None
                        ],
                    }
                else:  # val: one series per split
                    rows = db.execute(
                        f"SELECT step, split, {expr} AS v, COUNT(*) AS n FROM rollout"
                        " WHERE run=? AND validate=1 GROUP BY step, split ORDER BY step",
                        (run,),
                    ).fetchall()
                    by_split: dict[str, list] = {}
                    for r in rows:
                        if r["v"] is not None:
                            by_split.setdefault(r["split"], []).append(
                                {"step": r["step"], "value": r["v"], "n": r["n"]}
                            )
                    series[key] = {
                        "label": CURVES[key]["label"],
                        "splits": by_split,
                    }
        return {"run": run, "mode": mode, "series": series}

    # ------------------------------------------------------------ step level

    def step_rollouts(
        self, run: str, step: int, validate: bool, filters: dict[str, str]
    ) -> dict[str, Any]:
        where = ["run=?", "step=?", "validate=?"]
        args: list[Any] = [run, step, int(validate)]
        for col in ("outcome", "family", "fail_label", "split", "task_id"):
            if filters.get(col):
                where.append(f"{col}=?")
                args.append(filters[col])
        if filters.get("won") in ("0", "1"):
            where.append("won=?")
            args.append(int(filters["won"]))
        with self.index.connect() as db:
            rows = [
                dict(r)
                for r in db.execute(
                    "SELECT file, task_id, family, split, outcome, won, env_steps,"
                    " assistant_turns, invalid_format, truncated, n_invalid, n_noact,"
                    " max_repeat, sample_idx, fail_label, plugin_ver, tags,"
                    " research_final_success, research_archive"
                    f" FROM rollout WHERE {' AND '.join(where)}"
                    " ORDER BY task_id, sample_idx",
                    args,
                )
            ]
        stats = _stratified(rows)
        crit = sorted({r["plugin_ver"] for r in rows}) if rows else []
        scope = f"runs/{run}/rollouts/step_{step:05d}{'_val' if validate else ''}/"
        return {
            "run": run,
            "step": step,
            "validate": validate,
            "filters": {k: v for k, v in filters.items() if v},
            "stats": stats,
            "rollouts": rows,
            "provenance": {
                "scope": scope,
                "criteria": f"classifier={'+'.join(crit)}",
                "citation": citation(
                    f"success {stats['success_rate']}",
                    stats["n"],
                    scope,
                    f"classifier={'+'.join(crit)}",
                ),
            },
        }

    # ------------------------------------------------------------- detail

    def rollout_detail(self, run: str, rel: str) -> dict[str, Any] | None:
        d = schema.read_json(self.index.runs_root / run / rel)
        if d is None:
            return None
        traj = d.get("trajectory") or {}
        task_id = str(d.get("task_id", ""))
        plugin = plugins.plugin_for(task_id)
        prev = None
        annotated = []
        for s in traj.get("steps") or []:
            tags = schema.generic_step_tags(s, prev)
            if plugin is not None:
                tags += plugin.step_tags(s)
            annotated.append({**s, "tags": tags})
            if s.get("acted", True):
                prev = s.get("action_norm")
        return {
            "run": run,
            "file": rel,
            "payload": {k: v for k, v in d.items() if k != "trajectory"},
            "task_id": task_id,
            "family": plugins.task_family(task_id),
            "traj_tags": plugin.traj_tags(traj) if plugin is not None else [],
            "initial_obs": traj.get("initial_obs", ""),
            "steps": annotated,
            "outcome": traj.get("outcome"),
            "won": traj.get("won"),
            "error": traj.get("error"),
            "research_archive": (d.get("research") or {}).get("research_archive") or "",
        }

    def research_detail(self, run: str, rel: str) -> dict[str, Any] | None:
        session_dir = self.index.runs_root / run / rel
        meta = schema.research_meta(session_dir)
        if meta is None:
            return None
        doc_path = session_dir / "doc.md"
        doc = doc_path.read_text(encoding="utf-8") if doc_path.is_file() else None
        with self.index.connect() as db:
            consumers = [
                dict(r)
                for r in db.execute(
                    "SELECT file, step, validate, won, outcome, env_steps, fail_label"
                    " FROM rollout WHERE run=? AND research_archive=?"
                    " ORDER BY validate, step, file",
                    (run, rel),
                )
            ]
        return {
            "run": run,
            "dir": rel,
            "meta": meta,
            "doc": doc,
            "consumers": consumers,
        }

    def task_history(self, run: str, task_id: str) -> dict[str, Any]:
        with self.index.connect() as db:
            rows = [
                dict(r)
                for r in db.execute(
                    "SELECT file, step, validate, won, outcome, env_steps, fail_label,"
                    " family, max_repeat, sample_idx FROM rollout WHERE run=? AND task_id=?"
                    " ORDER BY validate, step, sample_idx",
                    (run, task_id),
                )
            ]
            sessions = [
                dict(r)
                for r in db.execute(
                    "SELECT dir, policy_version, doc_present, doc_words,"
                    " experiments_used, episodes_consumed, final_success, session_error"
                    " FROM research WHERE run=? AND task_id=? ORDER BY policy_version",
                    (run, task_id),
                )
            ]
        by_step: dict[tuple, list] = {}
        for r in rows:
            by_step.setdefault((r["validate"], r["step"]), []).append(r)
        history = [
            {
                "step": step,
                "validate": bool(validate),
                **_stratified(group),
                "rollouts": group,
            }
            for (validate, step), group in sorted(by_step.items())
        ]
        return {
            "run": run,
            "task_id": task_id,
            "family": plugins.task_family(task_id),
            "history": history,
            "research_sessions": sessions,
        }
