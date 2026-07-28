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

    bad = 0
    for step in list_steps(Path(args.obs_dir)):
        pack = load_step(args.obs_dir, step)
        for row in pack.rows():
            res = derive.audit_row(row, atol=args.atol, loss_mode=pack.loss_mode)
            if not res["ok"]:
                bad += 1
                print(f"AUDIT FAIL step={step} hash={row['resp_hash']} "
                      f"max_err={res['max_err']:.4f} checked={res['checked']}")
    print(f"audit done: {'OK' if bad == 0 else f'{bad} rows FAILED'}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
