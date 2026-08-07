#!/usr/bin/env bash
# 162 max-open bundle entry: inherit the verified 1.7B SGLang toolchain, then
# point imports and outputs at the independent dynamic-researcher bundle.
set -euo pipefail

BASE_ENTRY=/data4/ws/PGS/trace_opd_1p7b_s2_bundle/env_pgs_sglang_1p7b_162.sh
[[ -f "$BASE_ENTRY" ]] || { printf 'FATAL: missing verified base entry: %s\n' "$BASE_ENTRY" >&2; return 2; }
# shellcheck disable=SC1090
source "$BASE_ENTRY"

export PGS_RESEARCHER_MAXOPEN_BUNDLE=/data4/ws/PGS/trace_opd_1p7b_researcher_maxopen_bundle
export PGS_REPO=$PGS_RESEARCHER_MAXOPEN_BUNDLE
export PGS_ROOT=$PGS_REPO/trace_opd
export PYTHONPATH=$PGS_ROOT/src:$PGS_ROOT/verl
export PGS_MAXOPEN_ENTRY_SOURCED=1
# The verified base entry sets offline when no key is in the login shell. This
# bundle receives the key later through stdin, so remove only that automatic
# fallback; run_162_researcher_maxopen.py still rejects an explicit offline mode.
unset LD_PRELOAD
unset SWANLAB_MODE
