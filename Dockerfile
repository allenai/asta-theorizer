FROM python:3.12-slim

WORKDIR /app

# Vendor wheels for packages no longer available on public PyPI (mistralai was
# quarantined; see https://pypi.org/simple/mistralai/). pip --find-links pulls
# matching dists from ./vendor/ in preference to (failing) network lookups.
COPY vendor/ ./vendor/
COPY requirements.txt .
RUN pip install --no-cache-dir --find-links ./vendor/ -r requirements.txt uvicorn

COPY src/ ./src/
ENV PYTHONPATH=/app/src

EXPOSE 9000

# Listen on Cloud Run's injected $PORT when present; default 9000 for local/E2E and the
# existing prod service (whose Cloud Run container port is 9000, so $PORT=9000 there too).
# `exec` so uvicorn replaces the shell as PID 1 and receives SIGTERM directly (graceful
# shutdown on Cloud Run scale-down / revision swap); the sh wrapper is only for $PORT expansion.
CMD ["sh", "-c", "exec uvicorn asta.server:app --host 0.0.0.0 --port ${PORT:-9000}"]
