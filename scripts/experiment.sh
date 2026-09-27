#!/usr/bin/env bash
# Score both builds against the same dataset. This is the command that
# reproduces the comparison.
#
#   scripts/experiment.sh            both variants
#   scripts/experiment.sh v2         just one
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VARIANTS=("$@"); [ "$#" -eq 0 ] && VARIANTS=(v1 v2)

# Upsert once, before either run: both variants must score the same examples.
"$ROOT/.venv/bin/python" "$ROOT/evals/dataset.py"

for v in "${VARIANTS[@]}"; do
  echo "  running $v"
  "$ROOT/.venv/bin/python" "$ROOT/evals/run_experiment.py" --variant "$v" --skip-upsert \
    2>&1 | grep -E "EXPERIMENT_NAME|Error|error" || true
done
echo
echo "compare with:  scripts/report.py"
