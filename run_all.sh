#!/usr/bin/env bash
# Run every experiment in order, logging each step. Usage:
#   ./run_all.sh          full run (1000 trees) -> outputs/
#   DML_FAST=1 ./run_all.sh   smoke run (100 trees) -> outputs_fast/
cd "$(dirname "$0")"
PY=./.venv/Scripts/python
[ -x "$PY" ] || PY=./.venv/bin/python
OUTDIR=outputs; [ "${DML_FAST:-0}" = "1" ] && OUTDIR=outputs_fast
mkdir -p "$OUTDIR/logs"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
status=0
for s in 01_eda_baseline 02_dml_by_hand 03_dml_main 04_robustness 05_report; do
  echo "=== $s  $(date '+%F %T') ==="
  if (cd src && "../$PY" "$s.py") > "$OUTDIR/logs/$s.log" 2>&1; then
    echo "    ok"
  else
    echo "    FAILED (see $OUTDIR/logs/$s.log)"; status=1
  fi
done
echo "=== finished $(date '+%F %T') status=$status ==="
exit $status
