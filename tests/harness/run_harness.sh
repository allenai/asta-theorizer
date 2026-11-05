#!/usr/bin/env bash
# Theorizer ↔ Nora UI confidence harness.
#
# Builds the theorizer image, runs it locally exactly as nora's a2a_handler
# reaches it in local mode (downstream_url = http://host.docker.internal:9000),
# and runs two checks that together build confidence the paper_store change
# will not break the UI:
#
#   1. transform_contract_test.py — nora's UI→theorizer transform emits no
#      paper_store (so the change is invisible to the UI path).
#   2. docker_smoke.py — against the running container: UI-shaped and valid-dict
#      requests pass inject_papers; a list-shaped paper_store fails FAST with a
#      clear error (the fix) instead of hanging.
#
# Usage:  tests/harness/run_harness.sh   [--keep]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
IMAGE="asta-theorizer:local"
NAME="theorizer-harness"
KEEP="${1:-}"

cd "$REPO"

echo "==> [1/4] build image (if missing)"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" .
else
  echo "    using existing $IMAGE"
fi

echo "==> [2/4] run container on :9000 (mirrors nora local downstream_url)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p 9000:9000 \
  -e API_KEY=localtest -e MISTRAL_API_KEY=dummy-not-used-for-shape-tests \
  "$IMAGE" >/dev/null
# wait for health
for _ in $(seq 1 30); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://localhost:9000/healthz || true)" = "200" ] && break
  sleep 2
done
echo "    healthz: $(curl -s -o /dev/null -w '%{http_code}' http://localhost:9000/healthz)"

rc=0
echo "==> [3/4] UI contract test (nora transform emits no paper_store)"
"$PY" tests/harness/transform_contract_test.py || rc=1

echo "==> [4/4] docker smoke (request shapes against the live container)"
"$PY" tests/harness/docker_smoke.py || rc=1

if [ "$KEEP" = "--keep" ]; then
  echo "    leaving container '$NAME' running (--keep)"
else
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

if [ "$rc" = 0 ]; then
  echo "HARNESS PASSED — change is safe for the UI path; paper_store fix fails fast."
else
  echo "HARNESS FAILED — see output above."
fi
exit "$rc"
