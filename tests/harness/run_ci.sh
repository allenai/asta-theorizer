#!/usr/bin/env bash
# PR checks: unit tests + docker smoke + fake-LLM E2E, all against the production image.
# No provider keys or spend: THEORIZER_FAKE_LLM=1 and dummy keys throughout.
#
# Usage:  tests/harness/run_ci.sh [--no-build]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="asta-theorizer:ci"
# docker_smoke.py reads this container's logs by name.
NAME="theorizer-harness"
PY="${PYTHON:-python3}"
# The other matrix cases need real LLM output (theories) or network hydration, so they can't pass on canned responses.
FAKE_CASES=(build-extraction-schema graceful-empty validation-errors
            find-extract-byo-fulltext find-extract-byo-schema find-extract-uri)

cd "$REPO"
[ "${1:-}" = "--no-build" ] || docker build -t "$IMAGE" .

SEED_DIR="$(mktemp -d)"
cleanup() {
  rc=$?
  [ "$rc" = 0 ] || docker logs "$NAME" 2>&1 | tail -200 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  rm -rf "$SEED_DIR"
  exit "$rc"
}
trap cleanup EXIT

echo "==> unit tests"
docker run --rm -e MISTRAL_API_KEY=ci-dummy -v "$REPO/tests:/app/tests:ro" -v "$REPO/pytest.ini:/app/pytest.ini:ro" \
  -w /app "$IMAGE" sh -c "pip install -q pytest && python -m pytest -q -p no:cacheprovider"

echo "==> start container (fake LLM)"
"$PY" - "$SEED_DIR/seed_paperstore.json" <<'PYEOF'
import json, sys
from tests.harness.e2e_matrix import fulltext_paperstore
with open(sys.argv[1], "w") as f:
    json.dump(fulltext_paperstore(), f)
PYEOF
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p 9000:9000 \
  -e API_KEY=localtest -e THEORIZER_FAKE_LLM=1 -e DATABASE_URL= -e USE_ASTA_PAPERS=false \
  -e MISTRAL_API_KEY=ci-dummy -e OPENAI_API_KEY=ci-dummy -e ANTHROPIC_API_KEY=ci-dummy \
  -v "$SEED_DIR:/seed:ro" "$IMAGE" >/dev/null
for _ in $(seq 1 30); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 http://localhost:9000/healthz || true)" = "200" ] && break
  sleep 2
done
curl -sf http://localhost:9000/healthz >/dev/null || { echo "container never became healthy"; exit 1; }

echo "==> docker smoke"
"$PY" tests/harness/docker_smoke.py

echo "==> fake-LLM E2E"
"$PY" tests/harness/e2e_matrix.py --url http://localhost:9000 --api-key localtest \
  --paperstore-uri file:///seed/seed_paperstore.json --timeout 600 --include "${FAKE_CASES[@]}"
