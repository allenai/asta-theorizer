"""Docker smoke test: submit the request shapes that matter to a LOCAL theorizer
container and assert the paper_store handling behaves — proving the fixed image
is safe to deploy and the UI path is unaffected.

Assumes a container is running and reachable (see run_harness.sh):
    docker run -d --name theorizer-harness -p 9000:9000 \\
        -e API_KEY=localtest -e MISTRAL_API_KEY=dummy asta-theorizer:local

Three submissions via A2A message/send, then poll tasks/get:
  1. UI-shaped DataPart (theory_query only — exactly what nora's transform
     emits): must be ACCEPTED and must NOT fail with a paper_store error.
  2. Valid dict paper_store (1 full-text paper): must get PAST inject_papers
     (no paper_store error) — i.e. the fix accepts a valid dict.
  3. List-shaped paper_store: must FAIL FAST with a clear "paper_store must be
     a PaperStore object" error — NOT hang (this is the fix; pre-fix it crashed
     mid-task and the task hung in submitted).

Exit non-zero if any contract is violated.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from urllib import request

URL = "http://localhost:9000"
BEARER = "localtest"
POLL_SECONDS = 90
PAPERSTORE_ERR = "paper_store must be a PaperStore object"


def _rpc(method: str, params: dict) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": "1", "method": method, "params": params}).encode()
    req = request.Request(URL + "/", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {BEARER}"})
    with request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def submit(data: dict) -> str:
    msg = {"message": {"messageId": str(uuid.uuid4()), "role": "user",
                       "parts": [{"kind": "data", "data": {
                           "kind": "literature-theory-generation", "data": data}}]}}
    return _rpc("message/send", msg)["result"]["id"]


def task_text(task_id: str) -> tuple[str, str]:
    """Return (state, concatenated message text) for a task."""
    d = _rpc("tasks/get", {"id": task_id})["result"]
    state = d.get("status", {}).get("state", "?")
    texts = []
    for h in d.get("history", []) or []:
        for p in h.get("parts", []) or []:
            if p.get("kind") == "text":
                texts.append(p.get("text", ""))
    st_msg = (d.get("status", {}) or {}).get("message") or {}
    for p in (st_msg.get("parts", []) if isinstance(st_msg, dict) else []) or []:
        if p.get("kind") == "text":
            texts.append(p.get("text", ""))
    return state, " || ".join(texts)


def poll_for(task_id: str, predicate, secs: int):
    """Poll until predicate(state, text) is True or terminal/timeout. Returns (state, text, hit)."""
    deadline = time.time() + secs
    last = ("", "")
    while time.time() < deadline:
        state, text = task_text(task_id)
        last = (state, text)
        if predicate(state, text):
            return state, text, True
        if state in {"completed", "failed", "canceled", "rejected"}:
            return state, text, predicate(state, text)
        time.sleep(3)
    return last[0], last[1], predicate(*last)


def main() -> int:
    UI_QUERY = "What is the causal role of sanitation in poliovirus transmission?"
    results = {}

    t1 = submit({"theory_query": UI_QUERY, "max_papers_to_retrieve": 5,
                 "do_qualified_novelty_evaluation": False})
    t2 = submit({"theory_query": UI_QUERY, "max_papers_to_retrieve": 5,
                 "search_additional_papers": False, "do_qualified_novelty_evaluation": False,
                 "paper_store": {"paperstore": {"P1": {"title": "Seed", "status": "full-text",
                                                       "paper_markdown": "# Seed\n\nbody text"}},
                                 "paperstore_lut": {"P1": "P1"}}})
    t3 = submit({"theory_query": UI_QUERY, "max_papers_to_retrieve": 5,
                 "do_qualified_novelty_evaluation": False,
                 "paper_store": [{"corpus_id": "13756489", "status": "metadata-only"}]})

    print(f"submitted: UI={t1}  valid-dict={t2}  list={t3}")

    # 3 must FAIL FAST (state->failed, not hang); A2A status hides the error text, so confirm the cause via container logs.
    s3, txt3, _ = poll_for(t3, lambda st, tx: st == "failed", POLL_SECONDS)
    import subprocess
    logs = subprocess.run(["docker", "logs", "theorizer-harness"],
                          capture_output=True, text=True).stdout + \
           subprocess.run(["docker", "logs", "theorizer-harness"],
                          capture_output=True, text=True).stderr
    err_in_logs = PAPERSTORE_ERR in logs and t3 in logs
    print(f"[3 list paper_store] state={s3} (failed=fast-fail, not hang) "
          f"paperstore_error_in_logs={err_in_logs}")
    results["list_fails_clean"] = (s3 == "failed") and err_in_logs

    # 1 and 2 must get PAST inject_papers; SDK masks the error text, so assert on STATE (fail-fast -> 'failed', getting past -> 'working').
    def progressed(st, tx):
        return st in {"working", "completed"}
    s1, _, ok1 = poll_for(t1, progressed, 30)
    s2, _, ok2 = poll_for(t2, progressed, 30)
    print(f"[1 UI-shaped]      state={s1} progressed_past_inject={ok1}")
    print(f"[2 valid dict]     state={s2} progressed_past_inject={ok2}")
    results["ui_progressed_past_inject"] = ok1
    results["valid_dict_progressed_past_inject"] = ok2

    ok = (results["list_fails_clean"]
          and results["ui_progressed_past_inject"]
          and results["valid_dict_progressed_past_inject"])
    print("\nRESULTS:", json.dumps(results))
    if ok:
        print("SMOKE OK: list paper_store fails fast with a clear error (no hang); "
              "UI-shaped and valid-dict requests pass inject_papers cleanly.")
        return 0
    print("SMOKE FAILED: see results above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
