"""CLI: python -m trace_opd.viz --runs-root <...>/runs [--port 8642]"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .index import Index
from .queries import Queries
from .server import make_server, start_rescan_thread


def main() -> int:
    ap = argparse.ArgumentParser(description="TRACE-OPD observation system (read-only)")
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback by default; access remotely via ssh -L")
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--data-dir", type=Path,
                    default=Path.home() / ".trace_opd_viz",
                    help="index/cache location (never inside runs/)")
    ap.add_argument("--rescan-interval", type=float, default=30.0)
    ap.add_argument("--scan-only", action="store_true",
                    help="build/refresh the index and exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    index = Index(args.runs_root, args.data_dir / "index.sqlite3")
    report = index.scan()
    for run, stats in report.items():
        logging.info("scanned %s: +%s (skipped %s)", run, stats["added"], stats["skipped"])
    if args.scan_only:
        return 0

    start_rescan_thread(index, args.rescan_interval)
    server = make_server(args.host, args.port, Queries(index))
    logging.info("serving http://%s:%s  (runs-root=%s, read-only)",
                 args.host, args.port, args.runs_root)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
