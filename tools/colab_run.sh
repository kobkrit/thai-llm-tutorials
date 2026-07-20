#!/usr/bin/env bash
# Run a course notebook on a real Colab GPU, recreating the session if Colab
# reclaimed it.
#
# Colab drops idle runtimes without warning; a bare `colab exec` then dies with
# "Session appears to be lost (404/401)" having executed nothing. This wrapper
# makes that recoverable instead of a manual restart.
#
#   tools/colab_run.sh 02                # notebook 02 on a T4
#   GPU=A100 tools/colab_run.sh 07       # bigger GPU when a T4 genuinely won't do
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
SESSION="${SESSION:-llm}"
GPU="${GPU:-T4}"
TIMEOUT="${TIMEOUT:-3000}"
ID="${1:?usage: colab_run.sh <notebook-number, e.g. 02>}"

cd "$(dirname "$0")/.."
NB=$(ls notebooks/"${ID}"_*.ipynb 2>/dev/null | head -1)
[ -z "$NB" ] && { echo "no notebook matching ${ID}_*.ipynb"; exit 2; }
LOG="/tmp/nb${ID}.log"

ensure_session() {
  if colab sessions 2>/dev/null | grep -q "\[$SESSION\]"; then
    echo "[run] reusing session '$SESSION'"
    return 0
  fi
  echo "[run] creating session '$SESSION' on $GPU ..."
  colab stop -s "$SESSION" >/dev/null 2>&1   # clear a stale local entry
  colab new -s "$SESSION" --gpu "$GPU" 2>&1 | tail -2
  colab sessions 2>/dev/null | grep -q "\[$SESSION\]"
}

for attempt in 1 2; do
  ensure_session || { echo "[run] could not obtain a $GPU runtime"; exit 3; }
  echo "[run] executing $NB (attempt $attempt, timeout ${TIMEOUT}s)"
  colab exec -s "$SESSION" -f "$NB" --timeout "$TIMEOUT" > "$LOG" 2>&1
  if grep -q 'appears to be lost' "$LOG"; then
    echo "[run] session was reclaimed mid-run; retrying once"
    colab stop -s "$SESSION" >/dev/null 2>&1
    continue
  fi
  break
done

echo "[run] last cell: $(grep -o 'Executing cell [0-9]*/[0-9]*' "$LOG" | tail -1)"
if grep -qE '^[A-Za-z]*Error|Traceback \(most recent' "$LOG"; then
  echo "[run] RESULT: FAIL"
  grep -nE '^[A-Za-z]*Error|OutOfMemoryError' "$LOG" | head -3
else
  echo "[run] RESULT: no exception detected"
fi
echo "[run] log: $LOG"
