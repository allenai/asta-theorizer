# Theorizer test harnesses

Two harnesses live here:

1. **Full API-surface E2E matrix** (`e2e_matrix.py` + `run_e2e_matrix.sh`) — the
   primary confidence check before shipping. Builds the image on the published
   SDK version pinned in `requirements.txt`, runs it, and exercises every skill and every
   bring-your-own (BYO) input combination against it.
2. **Nora UI confidence harness** (`docker_smoke.py` + `transform_contract_test.py`
   + `run_harness.sh`) — narrow check that the `paper_store` change is invisible
   to the Nora UI path. Documented further down.

## E2E matrix (`run_e2e_matrix.sh`)

Self-contained: builds `asta-theorizer:e2e`, boots it with `FilePersistence`
(so `run_id` resume works in-container), runs `e2e_matrix.py`, and checks that URI
paper stores are rejected. Secrets come from the repo `.env`.

```bash
tests/harness/run_e2e_matrix.sh                 # all tiers
tests/harness/run_e2e_matrix.sh --tier cheap medium   # fast plumbing only
tests/harness/run_e2e_matrix.sh --rebuild --keep      # force rebuild, leave container up
```

Tiers: `cheap` (validation + schema, no paper retrieval), `medium` (BYO-dict
find/extract + BYO schema/results/theory chains + run_id resume, no PaperFinder),
`expensive` (metadata-only hydration + full `literature-theory-generation`).
Cases run in dependency order and reuse each other's `run_id` / folded
`theory_store`, so paper-finding and extraction are paid for once.

Coverage: BYO full-text dict paper_store · metadata-only hydration · paper_store
URI rejection · BYO extraction_schema · BYO extraction_results · BYO
theory_store · step-by-step via `run_id` · resume-extraction · full pipeline ·
fail-fast validation (list-shaped paper_store / mutually-exclusive args).

To run against an already-running server instead of building a container:

```bash
PYTHONPATH=src .venv/bin/python tests/harness/e2e_matrix.py \
    --url http://localhost:9100 --api-key "$API_KEY" --tier medium
```

---

# Theorizer ↔ Nora UI confidence harness

Purpose: build confidence that the theorizer `paper_store` change does not break
the Nora UI, by running the theorizer in Docker exactly as Nora reaches it
locally and submitting the request shapes that matter.

## How Nora reaches the theorizer

Nora's `a2a_handler` proxies the UI to the theorizer:

```
Nora UI (TextPart: the research question)
  → a2a_handler  (app/transforms/theorizer.py reshapes it)
  → downstream theorizer   (local.conf: http://host.docker.internal:9000)
```

The transform emits a fixed DataPart — `theory_query` + `max_papers_to_retrieve`
+ `do_qualified_novelty_evaluation`. **It never sends `paper_store`.** So the
`paper_store` fix (the `isinstance` guard in `inject_papers`) is invisible to the
UI path; it only fires for direct API/CLI callers that supply a `paper_store`.

## What the harness checks

| Check | File | Asserts |
|---|---|---|
| UI contract | `transform_contract_test.py` | Nora's transform → theorizer DataPart with **no `paper_store`**, correct query + defaults/overrides |
| Live request shapes | `docker_smoke.py` | Against the running container: UI-shaped and valid-dict requests pass `inject_papers`; a **list-shaped `paper_store` fails fast** (state→`failed`, clear `ValueError` in logs) instead of hanging |

## Run

```bash
tests/harness/run_harness.sh          # build/run/teardown + both checks
tests/harness/run_harness.sh --keep   # leave the container up for poking
```

The container runs with `API_KEY=localtest` (the Bearer the harness uses) and a
dummy `MISTRAL_API_KEY` (OCR is never called for these shape checks — the
`inject_papers` guard fires before any paper retrieval).

## Why this is sufficient for UI safety

- The UI path provably carries no `paper_store`, so the change cannot alter it.
- When a `paper_store` *is* supplied, a bad (list) shape now fails fast with a
  clear error rather than hanging mid-task — the regression that motivated the fix.
