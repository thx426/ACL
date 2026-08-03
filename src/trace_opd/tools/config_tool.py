"""Derive an experiment's config from the mechanism template, and keep it honest.

    python -m trace_opd.tools.config_tool new  162-4b-a3   # copy the template
    python -m trace_opd.tools.config_tool diff 162-4b-a3   # compare against it
    python -m trace_opd.tools.config_tool list             # every experiment's drift

Each experiment owns a COMPLETE copy of ``configs/pgs.yaml`` under
``configs/experiments/``. Not a diff against it — a copy. A run's config is then
self-contained: it can be read on its own months later, diffed against any other
experiment, and a template edit can never retroactively change what a finished
experiment meant.

``new`` copies the file verbatim, comments and all: the template's value is
mostly in its comments, and every rewrite through a YAML parser would strip
them. For the same reason ``diff`` only reports — bringing a copy up to date is
a paste, not an automated rewrite.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TEMPLATE = REPO / "configs" / "pgs.yaml"
EXPERIMENTS = REPO / "configs" / "experiments"

sys.path.insert(0, str(REPO / "src"))

from trace_opd.common.config import load_config, template_drift  # noqa: E402


def _path(name: str) -> Path:
    return EXPERIMENTS / (name if name.endswith(".yaml") else f"{name}.yaml")


def cmd_new(name: str, force: bool) -> int:
    target = _path(name)
    if target.exists() and not force:
        print(f"[config] {target} already exists (use --force to overwrite)")
        return 1
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE, target)
    print(f"[config] {target}\n[config] a complete copy of {TEMPLATE.name} — edit it freely; "
          "it is this experiment's whole definition")
    return 0


def cmd_diff(name: str) -> int:
    target = _path(name)
    if not target.exists():
        print(f"[config] no such experiment config: {target}")
        return 1
    drift = template_drift(target, TEMPLATE)
    if not drift["missing"] and not drift["extra"]:
        print(f"[config] {target.name}: in step with the template")
        return 0
    for key in drift["missing"]:
        print(f"[config] {target.name}: MISSING {key} — the template grew this setting after "
              "the copy was taken; the run falls back to the code default")
    for key in drift["extra"]:
        print(f"[config] {target.name}: EXTRA   {key} — not in the template (a setting the "
              "mechanism removed?)")
    return 0


def cmd_list() -> int:
    if not EXPERIMENTS.is_dir():
        print(f"[config] no experiments yet — start with: new <name>")
        return 0
    for path in sorted(EXPERIMENTS.glob("*.yaml")):
        drift = template_drift(path, TEMPLATE)
        n_missing, n_extra = len(drift["missing"]), len(drift["extra"])
        state = "in step" if not (n_missing or n_extra) else f"{n_missing} missing, {n_extra} extra"
        loads = "ok"
        try:
            load_config(path)
        except Exception as err:  # noqa: BLE001 — reporting tool
            loads = f"FAILS TO LOAD: {type(err).__name__}"
        print(f"  {path.stem:<28} {state:<24} {loads}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_new = sub.add_parser("new", help="copy the template into a new experiment config")
    p_new.add_argument("name")
    p_new.add_argument("--force", action="store_true")
    p_diff = sub.add_parser("diff", help="compare one experiment config against the template")
    p_diff.add_argument("name")
    sub.add_parser("list", help="drift and load status for every experiment config")
    args = parser.parse_args()
    if args.cmd == "new":
        return cmd_new(args.name, args.force)
    if args.cmd == "diff":
        return cmd_diff(args.name)
    return cmd_list()


if __name__ == "__main__":
    raise SystemExit(main())
