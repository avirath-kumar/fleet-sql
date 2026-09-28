#!/usr/bin/env bash
# Score both builds against the same dataset. This is the command that
# reproduces the comparison.
#
#   scripts/experiment.sh            all three arms, 3 repetitions
#   REPETITIONS=1 scripts/experiment.sh   a quick pass
#
# v1 is the naive build, v2 the offload architecture, and v2-regressed is v2
# with a real regression: a receipt cleanup that dropped distinct counts.
#   scripts/experiment.sh v2         just one
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VARIANTS=("$@"); [ "$#" -eq 0 ] && VARIANTS=(v1 v2-regressed v2)
# Three by default. One pass cannot separate a real failure from a coin flip,
# and both failures in this suite have been intermittent at some point -- the
# distinct-count question has scored 0/3, 2/3 and 3/3 on the same build.
REPS="${REPETITIONS:-3}"

# Upsert once, before either run: both variants must score the same examples.
"$ROOT/.venv/bin/python" "$ROOT/evals/dataset.py"

for v in "${VARIANTS[@]}"; do
  echo "  running $v"
  "$ROOT/.venv/bin/python" "$ROOT/evals/run_experiment.py" --variant "$v" --skip-upsert \
    --repetitions "$REPS" \
    2>&1 | grep -E "EXPERIMENT_NAME|Error|error" || true
done
echo
echo "compare with:  scripts/report.py"
