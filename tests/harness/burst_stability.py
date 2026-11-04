"""Burst / stability load test — the concurrency gap the sequential E2E matrix can't cover.

Reproduces the shape of the 2026-06-15 incident: many tasks submitted at once. The
matrix runs one task at a time, so it never exercised concurrent load and never caught
the memory leak that OOM-killed the container, stranding every in-flight task.

This driver submits N tasks concurrently and asserts the property the incident violated:
**every task reaches a terminal state** (none stranded in `submitted`/`working` forever).
Pair it with a memory-capped container (see run_burst_stability.sh) to confirm the box
survives the burst instead of OOMing.

Default skill is `build-extraction-schema`: one LLM call, no PaperFinder/OCR, so it's
cheap and fast, yet still spins up a per-workflow PaperStore — which is exactly the
object/thread that leaked. A high N of cheap tasks maximizes the leak signal per dollar.
Use `--skill literature-theory-generation` for a heavier, full-pipeline burst.

    PYTHONPATH=tests/harness .venv/bin/python tests/harness/burst_stability.py \
        --url http://localhost:9101 --api-key "$API_KEY" --n 16
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time

from e2e_matrix import (
    send_skill, poll_until_done, TERMINAL_STATES, load_queries_csv, _dedupe_queries,
    task_error_text,
)

# Distinct queries so the burst isn't deduped/cached into a single code path.
QUERIES = [
    "What mechanisms link chronic inflammation to insulin resistance?",
    "How does sleep deprivation affect working memory consolidation?",
    "What drives catastrophic forgetting in continual learning?",
    "How do mycorrhizal networks transfer carbon between trees?",
    "What cellular pathways underlie exercise-induced angiogenesis?",
    "How does urban heat island intensity scale with city density?",
    "What governs the stability of perovskite solar cell interfaces?",
    "How do gut microbiota influence host circadian rhythms?",
]


def run_one(url, api_key, skill, data, timeout):
    t0 = time.time()
    task_id = send_skill(url, api_key, skill, data)
    task = poll_until_done(url, api_key, task_id, timeout)
    state = (task.get("status") or {}).get("state")
    # Capture the failure reason + which query, so a `failed` task is self-diagnosing.
    err = "" if state == "completed" else task_error_text(task)[:400]
    return {"task_id": task_id, "state": state, "secs": round(time.time() - t0, 1),
            "query": data.get("theory_query", "")[:60], "err": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-key")
    ap.add_argument("--n", type=int, default=16, help="number of concurrent tasks (ignored with --queries-csv)")
    ap.add_argument("--queries-csv", default=None,
                    help="Fire one task per UNIQUE query from this CSV's theory_query column "
                         "(deduped); sets N to the unique-query count.")
    ap.add_argument("--skill", default="build-extraction-schema")
    ap.add_argument("--max-papers", type=int, default=3, help="only used by paper-fetching skills")
    ap.add_argument("--byo-papers", type=int, default=0,
                    help="attach N synthetic full-text papers per request (resident memory, no PaperFinder)")
    ap.add_argument("--paper-kb", type=int, default=200,
                    help="approx KB of markdown per synthetic paper — sets the resident footprint")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    # Query source: deduped CSV (one task per unique query) or the built-in distinct set.
    if args.queries_csv:
        queries = _dedupe_queries(load_queries_csv(args.queries_csv))
        args.n = len(queries)
        print(f"==> {args.n} unique queries from {args.queries_csv} "
              f"(deduped; firing all concurrently, server caps in-flight via MAX_SIMULTANEOUS_WORKFLOWS)", flush=True)
    else:
        queries = QUERIES

    def byo_paperstore(i):
        # Synthetic full-text papers with bloated markdown, so each workflow holds a
        # realistic resident footprint — the thing that actually fills memory under load.
        blob = ("This sentence pads the paper body to a realistic on-disk size. " * 16)
        body = (blob * max(1, (args.paper_kb * 1024) // len(blob)))
        papers, lut = {}, {}
        for k in range(args.byo_papers):
            key = f"burst-{i}-paper-{k}"
            papers[key] = {
                "corpus_id": f"9{i:02d}{k:04d}",
                "status": "full-text",
                "title": f"Synthetic load-test paper {i}.{k}",
                "paper_markdown": f"# Synthetic load-test paper {i}.{k}\n\n{body}",
            }
            lut[key] = key
        return {"paperstore": papers, "paperstore_lut": lut}

    def payload(i):
        d = {"theory_query": queries[i % len(queries)]}
        if args.skill in ("literature-theory-generation", "find-and-extract"):
            d["max_papers_to_retrieve"] = args.max_papers
            if args.byo_papers:
                # BYO papers + no search/novelty = the heavy in-memory path with zero external fetches.
                d["paper_store"] = byo_paperstore(i)
                d["search_additional_papers"] = False
                if args.skill == "literature-theory-generation":
                    d["do_qualified_novelty_evaluation"] = False
        return d

    print(f"==> firing {args.n} concurrent '{args.skill}' tasks at {args.url}", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n) as ex:
        futs = {ex.submit(run_one, args.url, args.api_key, args.skill, payload(i), args.timeout): i
                for i in range(args.n)}
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001  (a stranded/timed-out task surfaces here)
                r = {"task_id": None, "state": f"EXC:{e!r}", "secs": None, "query": "", "err": repr(e)}
            results.append(r)
            extra = f"  <{r['query']!r}> {r['err']}" if r.get("err") else ""
            print(f"    [{len(results):>2}/{args.n}] state={r['state']} ({r['secs']}s) {r['task_id']}{extra}", flush=True)

    terminal = [r for r in results if r["state"] in TERMINAL_STATES]
    completed = [r for r in results if r["state"] == "completed"]
    failed = [r for r in results if r["state"] == "failed"]
    stranded = [r for r in results if r["state"] not in TERMINAL_STATES]

    print("\n" + "=" * 60)
    print(f"reached terminal: {len(terminal)}/{args.n}   completed: {len(completed)}   "
          f"failed: {len(failed)}   stranded: {len(stranded)}")
    for r in stranded:
        print(f"  STRANDED: {r}")
    for r in failed:
        print(f"  FAILED: <{r['query']!r}> ({r['secs']}s) {r['err']}")
    # Stranding (never reaching terminal) is the incident signature and the hard gate.
    # `failed` tasks ran to a clean terminal error — surfaced for triage; they also fail the run
    # so the E2E goal treats them as issues to fix.
    ok = len(stranded) == 0 and len(failed) == 0
    if len(stranded):
        print("BURST FAILED — tasks stranded (incident signature)")
    elif len(failed):
        print(f"BURST FAILED — {len(failed)} task(s) reached state=failed (see reasons above)")
    else:
        print("BURST PASSED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
