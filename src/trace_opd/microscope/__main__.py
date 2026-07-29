"""python -m trace_opd.microscope — serve | audit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(prog="trace_opd.microscope")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="workbench server (ssh -L <port> to browse)")
    s.add_argument("--obs-dir", required=True)
    s.add_argument("--tokenizer", default=None, help="model path for token text")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8643)
    a = sub.add_parser("audit", help="re-derive the trained objective from packs "
                                     "and compare against the stored truth anchor")
    a.add_argument("--obs-dir", required=True)
    a.add_argument("--atol", type=float, default=0.05)
    args = parser.parse_args()

    if args.cmd == "serve":
        from .server import serve

        serve(args.obs_dir, args.tokenizer, args.host, args.port)
        return 0

    from . import derive
    from .reader import list_steps, load_step

    bad = checked_rows = 0
    steps = list_steps(Path(args.obs_dir))
    for step in steps:
        pack = load_step(args.obs_dir, step)
        for row in pack.rows():
            res = derive.audit_row(
                row, atol=args.atol, loss_mode=pack.loss_mode,
                clamp=pack.log_prob_min_clamp,
            )
            checked_rows += 1
            if not res["ok"]:
                bad += 1
                # CD-006 req 3-V: say HOW MANY tokens exceeded, not just the
                # max — 3/3000 must never read as "the whole row is wrong".
                print(f"AUDIT FAIL step={step} hash={row['resp_hash']} "
                      f"exceed={res['n_exceed']}/{res['checked']} "
                      f"({res['exceed_frac']:.2%}) max_err={res['max_err']:.4f}")
    if not steps or not checked_rows:
        # codex review P1-16: no packs / no rows is a data-integrity FAILURE,
        # not a pass — the P0 failure mode looked exactly like this.
        print(f"audit FAILED: steps={len(steps)} rows={checked_rows} — no data to audit")
        return 2
    print(f"audit done: {'OK' if bad == 0 else f'{bad} rows FAILED'} "
          f"({checked_rows} rows over {len(steps)} steps)")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
