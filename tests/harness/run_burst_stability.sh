#!/usr/bin/env bash
# Burst / stability load test for the Theorizer A2A server — the load-level
# confirmation for the 2026-06-15 OOM-under-burst incident.
#
# Runs the container under a *capped* memory limit (to make a leak fail fast as an
# OOM instead of silently bloating), fires N concurrent tasks, samples container
# memory throughout, and asserts every task reaches a terminal state (the incident
# stranded all of them).
#
# Usage:
#   tests/harness/run_burst_stability.sh                  # N=16 cheap real tasks (~$1), 3g cap
#   tests/harness/run_burst_stability.sh --rebuild        # force image rebuild (do this after src changes)
#
#   # Scale check: 22 concurrent FULL-pipeline requests, realistic resident memory, $0 tokens.
#   # Canned LLM (FAKE=1) + injected papers reproduce the OOM conditions without paying for tokens,
#   # under the production 8Gi cap. This is the cheap-but-realistic repro.
#   FAKE=1 N=22 SKILL=literature-theory-generation BYO_PAPERS=12 PAPER_KB=300 MEM=8g \
#       tests/harness/run_burst_stability.sh --rebuild
#
#   MEM=2g FAKE=1 N=22 ... run_burst_stability.sh        # tighten the cap to find the ceiling
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
IMAGE="asta-theorizer:e2e"
NAME="theorizer-burst"
PORT=9101
MEM="${MEM:-3g}"
N="${N:-16}"
SKILL="${SKILL:-build-extraction-schema}"
MAXP="${MAXP:-3}"
# QUERIES_CSV fires one task per unique query from a CSV (deduped). MAX_SIM caps how many
# the server processes at once (queues the rest) — set it to exercise queuing + concurrent OOM.
QUERIES_CSV="${QUERIES_CSV:-}"
MAX_SIM="${MAX_SIM:-}"
# Per-task client poll timeout. With MAX_SIM queuing, a queued task's wall-clock is its own
# runtime PLUS the wait for a slot, so this must exceed ~ceil(N/MAX_SIM) single-run times or
# queued tasks falsely time out client-side while the server is still draining the queue.
TIMEOUT="${TIMEOUT:-1800}"
# FAKE=1 runs the full pipeline with a canned LLM (THEORIZER_FAKE_LLM) — real server,
# real memory, $0 tokens. BYO_PAPERS/PAPER_KB inject resident paper payloads per request.
FAKE="${FAKE:-0}"
FAKE_DELAY="${FAKE_DELAY:-4}"
BYO_PAPERS="${BYO_PAPERS:-0}"
PAPER_KB="${PAPER_KB:-200}"

cd "$REPO"

REBUILD=0; KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --rebuild) REBUILD=1; shift ;;
    --keep)    KEEP=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -f "$REPO/.env" ] || { echo "FATAL: $REPO/.env not found"; exit 1; }
API_KEY="$(grep '^API_KEY=' "$REPO/.env" | cut -d= -f2-)"
[ -n "$API_KEY" ] || { echo "FATAL: API_KEY empty in .env"; exit 1; }

echo "==> [1/5] build image ($IMAGE)"
if [ "$REBUILD" = 1 ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" .
else
  echo "    using existing $IMAGE (pass --rebuild to pick up src changes)"
fi

echo "==> [2/5] run container on :$PORT (mem cap=$MEM, FilePersistence)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
FAKE_ENV=()
if [ "$FAKE" = 1 ]; then
  FAKE_ENV=(-e THEORIZER_FAKE_LLM=1 -e THEORIZER_FAKE_LLM_DELAY_S="$FAKE_DELAY")
  echo "    THEORIZER_FAKE_LLM=1 (canned LLM, \$0 tokens; per-call delay=${FAKE_DELAY}s)"
fi
[ -n "$MAX_SIM" ] && echo "    MAX_SIMULTANEOUS_WORKFLOWS=$MAX_SIM (server processes this many at once; rest queue)"
docker run -d --name "$NAME" -p "$PORT:9000" \
  --memory "$MEM" --memory-swap "$MEM" \
  --env-file "$REPO/.env" -e DATABASE_URL= \
  ${MAX_SIM:+-e MAX_SIMULTANEOUS_WORKFLOWS="$MAX_SIM"} \
  ${FAKE_ENV[@]+"${FAKE_ENV[@]}"} \
  "$IMAGE" >/dev/null

# Liveness: this service has no /healthz (returns 404); POST / returns 401/200 once up.
echo -n "    waiting for server"
for _ in $(seq 1 45); do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 -X POST http://localhost:$PORT/ || true)"
  case "$code" in 200|401) echo " up (POST / -> $code)"; break ;; *) echo -n "."; sleep 2 ;; esac
done

echo "==> [3/5] sample memory while bursting"
MEMLOG="$(mktemp)"
( while docker inspect -f '{{.State.Running}}' "$NAME" >/dev/null 2>&1; do
    docker stats --no-stream --format '{{.MemUsage}}' "$NAME" 2>/dev/null
    sleep 3
  done ) > "$MEMLOG" 2>/dev/null &
SAMPLER=$!

rc=0
PYTHONPATH="$REPO/tests/harness" "$PY" "$REPO/tests/harness/burst_stability.py" \
  --url "http://localhost:$PORT" --api-key "$API_KEY" \
  --n "$N" --skill "$SKILL" --max-papers "$MAXP" \
  --byo-papers "$BYO_PAPERS" --paper-kb "$PAPER_KB" --timeout "$TIMEOUT" \
  ${QUERIES_CSV:+--queries-csv "$QUERIES_CSV"} || rc=$?

kill "$SAMPLER" >/dev/null 2>&1 || true
echo "==> [4/5] memory profile (container cap=$MEM)"
if [ -s "$MEMLOG" ]; then
  echo "    samples: $(wc -l < "$MEMLOG" | tr -d ' ')  |  first: $(head -1 "$MEMLOG")  |  peak: $(sort -h "$MEMLOG" | tail -1)  |  last: $(tail -1 "$MEMLOG")"
fi
# OOMKilled would mean a leak still bites under the cap.
echo "    OOMKilled: $(docker inspect -f '{{.State.OOMKilled}}' "$NAME" 2>/dev/null || echo unknown)"

echo "==> [5/5] teardown"
rm -f "$MEMLOG"
if [ "$KEEP" = 1 ]; then
  echo "    leaving '$NAME' up (--keep); logs: docker logs $NAME"
else
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

[ "$rc" = 0 ] && echo "BURST STABILITY PASSED" || echo "BURST STABILITY FAILED (rc=$rc)"
exit "$rc"
