#!/usr/bin/env bash
# Self-contained E2E matrix for the Theorizer A2A server.
#
# Builds the image with the pinned published SDK and runs the API-surface matrix
# (tests/harness/e2e_matrix.py), including rejection of URI paper-store inputs.
#
# Secrets come from the repo .env (OPENAI/ANTHROPIC/MISTRAL/S2 keys + API_KEY).
# DATABASE_URL is intentionally unset so the container uses FilePersistence —
# run_id resume then works against the container's own disk.
#
# Usage:
#   tests/harness/run_e2e_matrix.sh                 # all tiers, build if needed
#   tests/harness/run_e2e_matrix.sh --tier cheap medium
#   tests/harness/run_e2e_matrix.sh --rebuild       # force image rebuild
#   tests/harness/run_e2e_matrix.sh --keep          # leave container running
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
IMAGE="asta-theorizer:e2e"
NAME="theorizer-e2e"
PORT=9100

cd "$REPO"

REBUILD=0; KEEP=0; TIER_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --rebuild) REBUILD=1; shift ;;
    --keep)    KEEP=1; shift ;;
    --tier)    shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do TIER_ARGS+=("$1"); shift; done ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -f "$REPO/.env" ] || { echo "FATAL: $REPO/.env not found (needs API_KEY + provider keys)"; exit 1; }
API_KEY="$(grep '^API_KEY=' "$REPO/.env" | cut -d= -f2-)"
[ -n "$API_KEY" ] || { echo "FATAL: API_KEY empty in .env"; exit 1; }

echo "==> [1/4] build image ($IMAGE)"
if [ "$REBUILD" = 1 ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" .
else
  echo "    using existing $IMAGE (pass --rebuild to force)"
fi

# FAKE=1 runs the container with a canned LLM ($0 tokens) — for validating plumbing /
# the query-list pass end-to-end without paying. QUERIES_CSV drives the matrix from a
# CSV of real queries (deduped) instead of the fixed CASES.
FAKE="${FAKE:-0}"
FAKE_ENV=()
if [ "$FAKE" = 1 ]; then
  FAKE_ENV=(-e THEORIZER_FAKE_LLM=1)
  echo "    THEORIZER_FAKE_LLM=1 (canned LLM, \$0 tokens)"
fi

echo "==> [2/4] run container on :$PORT (FilePersistence)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "$PORT:9000" \
  --env-file "$REPO/.env" -e DATABASE_URL= \
  ${FAKE_ENV[@]+"${FAKE_ENV[@]}"} \
  "$IMAGE" >/dev/null
for _ in $(seq 1 30); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://localhost:$PORT/healthz || true)" = "200" ] && break
  sleep 2
done
echo "    healthz: $(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/healthz)"

echo "==> [3/4] run matrix"
rc=0
# Query-list mode (QUERIES_CSV set) replays a deduped CSV of real queries instead of the
# fixed CASES; QUERY_BYO_PAPERS>0 injects synthetic papers + skips PaperFinder (pair with FAKE=1
# for a $0 end-to-end pass). Otherwise run the standard CASES matrix.
QUERY_ARGS=()
if [ -n "${QUERIES_CSV:-}" ]; then
  QUERY_ARGS=(--queries-csv "$QUERIES_CSV" --query-byo-papers "${QUERY_BYO_PAPERS:-0}" --query-paper-kb "${QUERY_PAPER_KB:-50}")
fi
PYTHONPATH="$REPO/src" "$PY" "$REPO/tests/harness/e2e_matrix.py" \
  --url "http://localhost:$PORT" --api-key "$API_KEY" \
  ${QUERY_ARGS[@]+"${QUERY_ARGS[@]}"} \
  ${TIER_ARGS:+--tier "${TIER_ARGS[@]}"} || rc=$?

echo "==> [4/4] teardown"
if [ "$KEEP" = 1 ]; then
  echo "    leaving container '$NAME' up (--keep); logs: docker logs $NAME"
else
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

[ "$rc" = 0 ] && echo "MATRIX PASSED" || echo "MATRIX FAILED (rc=$rc) — see output above"
exit "$rc"
