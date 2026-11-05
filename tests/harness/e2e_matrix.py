"""E2E matrix for the Theorizer A2A server.

Drives a live server over the A2A wire (message/send + tasks/get, the same path
the asta-a2a CLI uses), polls each task to a terminal state, and asserts each
skill emits only the artifact types its step list implies. Cases run in
dependency order so the expensive paper-finding/extraction work is paid once and
reused (run_id + folded theory_store) by later BYO cases. See CASES for the case
list and --tier for the cost tiers.

Usage:
    PYTHONPATH=src .venv/bin/python tests/harness/e2e_matrix.py \\
        --url http://localhost:9100 --api-key "$API_KEY" --tier all

Exit code: 0 iff every selected case passes (skipped cases don't count as passes).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import uuid
from typing import Any, Callable
from urllib import error, request


# Seed data — full-text papers we own, so the BYO chain needs neither PaperFinder nor OCR (deterministic + cheap).

SEED_PAPERS: dict[str, dict] = {
    "seed-icl-transformer": {
        "corpus_id": "13756489",
        "status": "full-text",
        "title": "Attention Is All You Need",
        "paper_markdown": (
            "# Attention Is All You Need\n\n"
            "We propose the Transformer, a model architecture relying entirely on "
            "self-attention to draw global dependencies between input and output, "
            "dispensing with recurrence and convolutions. Multi-head self-attention "
            "lets the model jointly attend to information from different "
            "representation subspaces. Increasing the number of attention heads and "
            "the model dimension increases representational capacity. The Transformer "
            "trains significantly faster than recurrent architectures and reaches "
            "higher BLEU on English-to-German translation.\n"
        ),
    },
    "seed-fewshot": {
        "corpus_id": "218971783",
        "status": "full-text",
        "title": "Language Models are Few-Shot Learners",
        "paper_markdown": (
            "# Language Models are Few-Shot Learners\n\n"
            "We show that scaling up language models greatly improves task-agnostic, "
            "few-shot performance. A 175B-parameter autoregressive Transformer "
            "performs in-context learning: given a natural-language prompt with a few "
            "demonstrations, it adapts to new tasks without gradient updates. "
            "Few-shot accuracy increases with model scale and with the number of "
            "in-context examples. Larger context windows admit more demonstrations, "
            "further improving accuracy.\n"
        ),
    },
}

THEORY_QUERY = (
    "What architectural choices enable transformer language models to "
    "perform few-shot in-context learning?"
)

# Metadata-only seed for resume-extraction; resume has no FIND_PAPERS step so it must hydrate inline (52967399=BERT, OA-hydratable).
RESUME_HYDRATION_CORPUS_ID = "52967399"

# Metadata-only seeds (no markdown) for the server-side hydration path.
HYDRATION_SEED_CORPUS_IDS = ["13756489", "52967399", "218971783"]

TERMINAL_STATES = {"completed", "failed", "canceled", "rejected"}


# A2A wire helpers (urllib only — no third-party deps in the harness)


def post(url: str, api_key: str | None, payload: dict, timeout: int = 60) -> dict:
    req = request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def send_skill(url: str, api_key: str | None, skill: str, data: dict) -> str:
    body = post(url, api_key, {
        "jsonrpc": "2.0", "id": "1", "method": "message/send",
        "params": {"message": {
            "messageId": str(uuid.uuid4()),
            "role": "user",
            "parts": [{"kind": "data", "data": {"kind": skill, "data": data}}],
        }},
    })
    if "result" not in body:
        raise RuntimeError(f"message/send returned no result: {body}")
    return body["result"]["id"]


def get_task(url: str, api_key: str | None, task_id: str) -> dict:
    body = post(url, api_key, {
        "jsonrpc": "2.0", "id": "1", "method": "tasks/get",
        "params": {"id": task_id},
    })
    return body.get("result") or {}


def poll_until_done(url: str, api_key: str | None, task_id: str,
                    timeout_seconds: int, poll_interval: int = 10) -> dict:
    deadline = time.time() + timeout_seconds
    start = time.time()
    last_state = None
    while time.time() < deadline:
        task = get_task(url, api_key, task_id)
        state = (task.get("status") or {}).get("state")
        if state != last_state:
            print(f"    [{int(time.time() - start):>4}s] state={state}", flush=True)
            last_state = state
        if state in TERMINAL_STATES:
            return task
        time.sleep(poll_interval)
    raise TimeoutError(f"Task {task_id} did not reach terminal state within {timeout_seconds}s")


# Artifact / store helpers


# Store section -> count key. These are no longer published as step-progress artifacts;
# they exist only inside the streamed theory_store.
_STORE_COUNT_KEYS = {
    "extraction_schemas": "extraction-schema",
    "extraction_results": "extraction",
    "theories": "theory",
    "theory_evaluations": "novelty",
}


def artifact_type_counts(task: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a in task.get("artifacts") or []:
        t = (a.get("metadata") or {}).get("type") or "unknown"
        counts[t] = counts.get(t, 0) + 1
    store = folded_theory_store(task)
    for section, key in _STORE_COUNT_KEYS.items():
        if store.get(section):
            counts[key] = len(store[section])
    return counts


def _deep_merge(target: dict, patch: dict) -> None:
    for k, v in patch.items():
        if v is None:
            target.pop(k, None)
        elif isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_merge(target[k], v)
        else:
            target[k] = v


def folded_theory_store(task: dict) -> dict:
    """Fold the streaming theory_store artifact (RFC 7396 merge patches)."""
    acc: dict[str, Any] = {}
    for a in task.get("artifacts") or []:
        if (a.get("metadata") or {}).get("type") != "theory_store":
            continue
        for part in a.get("parts") or []:
            if part.get("kind") == "data":
                _deep_merge(acc, part.get("data") or {})
    return acc


def extract_run_id(task: dict) -> str | None:
    for a in task.get("artifacts") or []:
        m = re.match(r"(gen-\d{8}-[0-9a-f]+)-theory-store-raw$", a.get("artifactId") or "")
        if m:
            return m.group(1)
    return None


def task_error_text(task: dict) -> str:
    texts = []
    st_msg = (task.get("status") or {}).get("message") or {}
    for p in (st_msg.get("parts") if isinstance(st_msg, dict) else []) or []:
        if p.get("kind") == "text":
            texts.append(p.get("text", ""))
    for h in task.get("history") or []:
        for p in h.get("parts") or []:
            if p.get("kind") == "text":
                texts.append(p.get("text", ""))
    # Drop the generic "Task accepted" acknowledgement the handler emits on receipt — it's
    # noise that masks the real terminal/failure reason (and looked like the error in reports).
    texts = [t for t in texts if t and t.strip().lower() != "task accepted"]
    return " || ".join(texts)


def seed_paperstore(corpus_ids: list[str]) -> dict:
    return {"paperstore": {
        f"paper-{cid}": {"corpus_id": cid, "status": "metadata-only"}
        for cid in corpus_ids
    }}


def fulltext_paperstore() -> dict:
    return {
        "paperstore": dict(SEED_PAPERS),
        "paperstore_lut": {k: k for k in SEED_PAPERS},
    }


# Reshape a folded store into the BYO input shapes the tasks accept.

def as_schema_input(store: dict) -> dict | None:
    schemas = store.get("extraction_schemas") or {}
    return {"extraction_schemas": schemas} if schemas else None


def as_extraction_results_input(store: dict) -> dict | None:
    schemas = store.get("extraction_schemas") or {}
    results = store.get("extraction_results") or {}
    if not schemas or not results:
        return None
    return {"extraction_schemas": schemas, "extraction_results": results}


def as_theory_store_input(store: dict) -> dict | None:
    if not (store.get("theories") or {}):
        return None
    return {
        "theories": store.get("theories") or {},
        "extraction_schemas": store.get("extraction_schemas") or {},
        "extraction_results": store.get("extraction_results") or {},
        "theory_evaluations": store.get("theory_evaluations") or {},
        "all_ids": store.get("all_ids") or [],
        "next_ids": store.get("next_ids") or {},
    }


# Assertions


def _check_completed(task: dict, label: str) -> tuple[bool, str]:
    state = (task.get("status") or {}).get("state")
    if state != "completed":
        return False, f"{label}: state={state!r} (expected completed); err={task_error_text(task)[:300]!r}"
    return True, ""


def _assert_counts(counts: dict[str, int], expect: dict[str, tuple[str, int]]) -> list[str]:
    """expect: type -> (op, n) where op in {'==','>=','=0'}."""
    fails = []
    for typ, (op, n) in expect.items():
        got = counts.get(typ, 0)
        if op == "==" and got != n:
            fails.append(f"{typ}={got}, expected =={n}")
        elif op == ">=" and got < n:
            fails.append(f"{typ}={got}, expected >={n}")
        elif op == "=0" and got != 0:
            fails.append(f"{typ}={got}, expected 0 (BUG SIGNATURE)")
    return fails


def _gather_referenced_result_ids(components: dict, uuid_index: dict[str, str]) -> list[str]:
    """Extraction-result ids a theory's statements actually cite.

    Mirrors the server-side population (Theorizer1.gather_supporting_evidence_ids): walk every
    evidence item, map each datum-level shorthand uuid (e.g. "e2.0") back to its extraction-result
    id via the rebuilt uuid_index, deduped + order-preserving.
    """
    result_ids, seen = [], set()

    def walk(node):
        if isinstance(node, dict):
            for u in (node.get("uuids") or []):
                rid = uuid_index.get(u)
                if rid and rid not in seen:
                    seen.add(rid)
                    result_ids.append(rid)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(components)
    return result_ids


def supporting_evidence_fails(store: dict) -> list[str]:
    """Per-theory supporting evidence must be the evidence THAT theory cites — not the whole store.

    Guards the fix where every theory used to get the entire training set (plus a stray schema id)
    as supporting_evidence_ids. For each theory asserts, against the folded theory_store:
      * supporting_evidence_ids == the extraction-result ids its own statement UUIDs resolve to,
      * no schema-* (or other non-result) id leaked in,
      * every id is a real extraction_results key.
    """
    theories = store.get("theories") or {}
    results = store.get("extraction_results") or {}
    if not theories:
        return ["folded store has no theories to check"]
    # Rebuild the datum-uuid -> result-id index the same way Struct._index_extraction_result does.
    uuid_index: dict[str, str] = {}
    for rid, r in results.items():
        for item in (r.get("extracted_data") or []):
            if isinstance(item, dict) and item.get("uuid"):
                uuid_index[item["uuid"]] = rid
    fails = []
    total_stored = 0
    for tid, theory in theories.items():
        stored = list(theory.get("supporting_evidence_ids") or [])
        total_stored += len(stored)
        referenced = _gather_referenced_result_ids(theory.get("components") or {}, uuid_index)
        leaked = [x for x in stored if x not in results]
        if leaked:
            fails.append(f"{tid}: non-result id(s) in supporting_evidence_ids: {leaked} (BUG SIGNATURE)")
        if set(stored) != set(referenced):
            fails.append(
                f"{tid}: supporting_evidence_ids={sorted(stored)} != "
                f"statement-referenced {sorted(referenced)} (BUG SIGNATURE)"
            )
    # Floor: per-theory equality is vacuous when both sides empty, so flag the regression where citable uuids exist but no theory resolved any.
    if uuid_index and total_stored == 0:
        fails.append(
            f"all {len(theories)} theories have empty supporting_evidence_ids despite "
            f"{len(uuid_index)} citable datum uuids in the store (BUG SIGNATURE: citations lost)"
        )
    return fails


def folded_paperstore(task: dict) -> dict:
    return folded_theory_store(task).get("paperstore") or {}


def _theory_statements(theory: dict) -> list[dict]:
    """A theory's statements, each unwrapped from its optional 'law' nesting."""
    out = []
    for s in ((theory.get("components") or {}).get("theory_statements") or []):
        if isinstance(s, dict):
            out.append(s["law"] if isinstance(s.get("law"), dict) else s)
    return out


def count_eligible_novelty_statements(store: dict) -> int:
    """Statements novelty WILL evaluate: those the generator pre-classified 'new'
    or 'somewhat-related-to-existing' (EvaluationQualifiedNovelty selects exactly
    these). DEBUG_LIMIT is None in prod, so every eligible statement is assessed."""
    n = 0
    for theory in (store.get("theories") or {}).values():
        for st in _theory_statements(theory):
            cls = (st.get("novelty_evaluation") or {}).get("likely_classification")
            if cls in ("new", "somewhat-related-to-existing"):
                n += 1
    return n


def novelty_coverage_fails(store: dict, novelty_count: int) -> list[str]:
    """G1: novelty must cover EVERY eligible statement. A statement is skipped only
    when its comparison corpus is empty — which never happens once novelty is
    assessed against the full retrieved corpus (any extracted papers). A shortfall
    is the corpus-narrowing regression."""
    eligible = count_eligible_novelty_statements(store)
    if eligible and novelty_count != eligible:
        return [f"novelty assessed {novelty_count}/{eligible} eligible statements "
                f"(BUG SIGNATURE: statements skipped — novelty corpus narrowed)"]
    return []


def content_floor_fails(store: dict, *, schema=False, extraction=False, theory=False) -> list[str]:
    """G5: emitted artifacts must carry real content, not just exist."""
    fails = []
    if schema and not any((s.get("schema") or []) for s in (store.get("extraction_schemas") or {}).values()):
        fails.append("no extraction_schema has any fields (BUG SIGNATURE: empty schema)")
    if extraction and not any((r.get("extracted_data") or []) for r in (store.get("extraction_results") or {}).values()):
        fails.append("no extraction_result has any extracted_data (BUG SIGNATURE: empty extractions)")
    if theory and not any(_theory_statements(t) for t in (store.get("theories") or {}).values()):
        fails.append("no theory has any statements (BUG SIGNATURE: empty theories)")
    return fails


def theory_cites_fails(store: dict) -> list[str]:
    """M5: every theory with statements must cite at least one paper, else the
    theory artifact renders with no literature linkage."""
    fails = []
    for tid, theory in (store.get("theories") or {}).items():
        if _theory_statements(theory) and not (theory.get("supporting_evidence_ids") or []):
            fails.append(f"{tid}: has statements but cites no evidence (BUG SIGNATURE: no literature linkage)")
    return fails


# Context shared across chained cases


class Ctx:
    def __init__(self, url: str, api_key: str | None, timeout: int, paperstore_uri: str | None):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.paperstore_uri = paperstore_uri
        # captured state
        self.schema_input: dict | None = None        # from build-extraction-schema
        self.fe_run_id: str | None = None             # from find-and-extract (BYO dict)
        self.fe_store: dict | None = None
        self.theory_store_input: dict | None = None   # from a form-theory run

    def run(self, skill: str, data: dict, timeout: int | None = None) -> dict:
        task_id = send_skill(self.url, self.api_key, skill, data)
        print(f"    task_id={task_id}")
        return poll_until_done(self.url, self.api_key, task_id, timeout or self.timeout)


# Cases — each (tier, fn). fn(ctx) -> (passed, message).


def case_validation_errors(ctx: Ctx) -> tuple[bool, str]:
    """No LLM. Bad inputs must fail fast with a clear error, never hang."""
    fails = []

    # 1. list-shaped paper_store -> ValueError, state failed.
    t = ctx.run("literature-theory-generation", {
        "theory_query": THEORY_QUERY,
        "paper_store": [{"corpus_id": "13756489", "status": "metadata-only"}],
        "do_qualified_novelty_evaluation": False,
    }, timeout=120)
    state = (t.get("status") or {}).get("state")
    if state != "failed":
        fails.append(f"list paper_store: state={state}, expected failed")

    # 2. form-theory with neither run_id nor extraction_results -> failed.
    t = ctx.run("form-theory", {}, timeout=120)
    state = (t.get("status") or {}).get("state")
    if state != "failed":
        fails.append(f"form-theory(empty): state={state}, expected failed")

    # 3. form-theory with extraction_results but no theory_query -> failed. Self-contained synthetic payload so this runs in every tier, never silently skips.
    synthetic_results = {
        "extraction_schemas": {"s-x": {"schema": [{"name": "f"}], "extraction_query": "q"}},
        "extraction_results": {"r-x": {"paper_id": "p-x", "extraction_schema_id": "s-x", "extracted_data": []}},
    }
    t = ctx.run("form-theory", {"extraction_results": synthetic_results}, timeout=120)
    state = (t.get("status") or {}).get("state")
    if state != "failed":
        fails.append(f"form-theory(results,no query): state={state}, expected failed")

    return (not fails), "; ".join(fails) or "OK"


def case_graceful_empty(ctx: Ctx) -> tuple[bool, str]:
    """No full-text papers to extract from must COMPLETE gracefully (with an explanatory
    message), not fail with a misleading 'timeout'. Forces the path deterministically with an
    empty paper store + search off, so paper_keys is empty and 0 extraction requests submit.
    Cheap (only the schema-build LLM call; pair with a THEORIZER_FAKE_LLM container for $0)."""
    task = ctx.run("literature-theory-generation", {
        "theory_query": THEORY_QUERY,
        "paper_store": {"paperstore": {}, "paperstore_lut": {}},
        "search_additional_papers": False,
    }, timeout=300)
    state = (task.get("status") or {}).get("state")
    if state != "completed":
        return False, f"expected graceful 'completed', got {state!r}; err={task_error_text(task)[:200]!r}"
    text = task_error_text(task).lower()
    if "no theories" not in text and "full text" not in text:
        return False, f"completion message lacks the graceful explanation: {task_error_text(task)[:200]!r}"
    return True, "graceful empty completion with explanatory message"


def case_build_extraction_schema(ctx: Ctx) -> tuple[bool, str]:
    """Schema only. ZERO extraction/theory/novelty. Captures the schema for BYO reuse."""
    task = ctx.run("build-extraction-schema", {"theory_query": THEORY_QUERY}, timeout=600)
    ok, msg = _check_completed(task, "build-extraction-schema")
    if not ok:
        return ok, msg
    counts = artifact_type_counts(task)
    print(f"    artifact counts: {counts}")
    folded = folded_theory_store(task)
    fails = _assert_counts(counts, {
        "extraction-schema": ("==", 1),
        "extraction": ("=0", 0),
        "theory": ("=0", 0),
        "novelty": ("=0", 0),
        "theory_store": ("==", 1),
    })
    fails += content_floor_fails(folded, schema=True)
    ctx.schema_input = as_schema_input(folded)
    if ctx.schema_input:
        print(f"    captured extraction_schema ({len(ctx.schema_input['extraction_schemas'])} schema)")
    return (not fails), "; ".join(fails) or "OK"


def case_find_extract_byo_fulltext(ctx: Ctx) -> tuple[bool, str]:
    """BYO full-text dict paper_store, NO PaperFinder. Schema + extractions, no theory.
    Captures run_id + folded store for the BYO/resume chain."""
    task = ctx.run("find-and-extract", {
        "theory_query": THEORY_QUERY,
        "paper_store": fulltext_paperstore(),
        "search_additional_papers": False,
        "max_papers_to_retrieve": len(SEED_PAPERS),
    })
    ok, msg = _check_completed(task, "find-and-extract(byo-fulltext)")
    if not ok:
        return ok, msg
    counts = artifact_type_counts(task)
    print(f"    artifact counts: {counts}")
    fails = _assert_counts(counts, {
        "extraction-schema": ("==", 1),
        "extraction": (">=", 1),
        "theory": ("=0", 0),
        "novelty": ("=0", 0),
        "theory_store": ("==", 1),
    })
    ctx.fe_run_id = extract_run_id(task)
    ctx.fe_store = folded_theory_store(task)
    fails += content_floor_fails(ctx.fe_store, schema=True, extraction=True)
    # Every seed must have produced an extraction — not just one of them.
    n_results = len(ctx.fe_store.get("extraction_results") or {})
    if n_results < len(SEED_PAPERS):
        fails.append(f"extraction_results={n_results} < {len(SEED_PAPERS)} seeds (a seed produced no extraction)")
    # G7: the BYO full-text seeds must round-trip into the folded store as full-text.
    ps = ctx.fe_store.get("paperstore") or {}
    for sp in SEED_PAPERS.values():
        entry = next((e for e in ps.values() if str(e.get("corpus_id")) == sp["corpus_id"]), None)
        if entry is None:
            fails.append(f"seed {sp['corpus_id']}: absent from folded paperstore")
        elif entry.get("status") != "full-text":
            fails.append(f"seed {sp['corpus_id']}: status={entry.get('status')!r} (expected full-text)")
    print(f"    captured run_id={ctx.fe_run_id}")
    return (not fails), "; ".join(fails) or "OK"


def case_find_extract_byo_schema(ctx: Ctx) -> tuple[bool, str]:
    """BYO extraction_schema -> skips schema construction. Full-text dict, no PaperFinder."""
    # Prefer build-extraction-schema's schema; else reuse find-and-extract's, so this case stands alone in --tier medium.
    schema_input = ctx.schema_input or (as_schema_input(ctx.fe_store) if ctx.fe_store else None)
    if not schema_input:
        return False, "skipped: no extraction_schema available (run build-extraction-schema or find-extract first)"
    task = ctx.run("find-and-extract", {
        "theory_query": THEORY_QUERY,
        "paper_store": fulltext_paperstore(),
        "extraction_schema": schema_input,
        "search_additional_papers": False,
        "max_papers_to_retrieve": len(SEED_PAPERS),
    })
    ok, msg = _check_completed(task, "find-and-extract(byo-schema)")
    if not ok:
        return ok, msg
    counts = artifact_type_counts(task)
    print(f"    artifact counts: {counts}")
    # Injected schema is NOT re-published as a discrete artifact (construction skipped), but must appear in the streaming theory_store.
    fails = _assert_counts(counts, {
        "extraction": (">=", 1),
        "theory": ("=0", 0),
        "novelty": ("=0", 0),
        "theory_store": ("==", 1),
    })
    folded = folded_theory_store(task)
    # Injected schema must be the one that flowed through, not a freshly built one (which would mean construction wasn't skipped).
    injected_ids = set((schema_input.get("extraction_schemas") or {}).keys())
    folded_ids = set((folded.get("extraction_schemas") or {}).keys())
    if not injected_ids:
        fails.append("no injected schema ids to verify")
    elif not injected_ids <= folded_ids:
        fails.append(f"injected schema {injected_ids} not in folded {folded_ids} (BYO schema was rebuilt, not reused)")
    fails += content_floor_fails(folded, extraction=True)
    return (not fails), "; ".join(fails) or "OK"


def case_find_extract_uri(ctx: Ctx) -> tuple[bool, str]:
    """paper_store passed as a file:// (or https://) URI string."""
    if not ctx.paperstore_uri:
        return True, "skipped: no --paperstore-uri provided (URI must be reachable from the server)"
    task = ctx.run("find-and-extract", {
        "theory_query": THEORY_QUERY,
        "paper_store": ctx.paperstore_uri,
        "search_additional_papers": False,
        "max_papers_to_retrieve": len(SEED_PAPERS),
    })
    ok, msg = _check_completed(task, "find-and-extract(uri)")
    if not ok:
        return ok, msg
    counts = artifact_type_counts(task)
    print(f"    artifact counts: {counts}")
    fails = _assert_counts(counts, {
        "extraction-schema": ("==", 1),
        "extraction": (">=", 1),
        "theory": ("=0", 0),
        "novelty": ("=0", 0),
        "theory_store": ("==", 1),
    })
    return (not fails), "; ".join(fails) or "OK"


def case_form_theory_byo_results(ctx: Ctx) -> tuple[bool, str]:
    """BYO extraction_results (+ BYO full-text paper_store) -> theories, no new extraction."""
    if not ctx.fe_store:
        return False, "skipped: no captured extraction results (run find-and-extract first)"
    eri = as_extraction_results_input(ctx.fe_store)
    if not eri:
        return False, "skipped: captured store had no extraction_results to replay"
    task = ctx.run("form-theory", {
        "theory_query": THEORY_QUERY,
        "extraction_results": eri,
        "paper_store": fulltext_paperstore(),
        "generation_temperature": 0.0,
    })
    ok, msg = _check_completed(task, "form-theory(byo-results)")
    if not ok:
        return ok, msg
    counts = artifact_type_counts(task)
    print(f"    artifact counts: {counts}")
    fails = _assert_counts(counts, {
        "theory": (">=", 1),
        "extraction": ("=0", 0),
        "novelty": ("=0", 0),
        "theory_store": ("==", 1),
    })
    folded = folded_theory_store(task)
    for tid, th in (folded.get("theories") or {}).items():
        print(f"    {tid} supporting_evidence_ids={th.get('supporting_evidence_ids')}")
    fails += supporting_evidence_fails(folded)
    fails += content_floor_fails(folded, theory=True)
    fails += theory_cites_fails(folded)
    # G4: published theory artifacts must round-trip 1:1 into the folded store.
    if counts.get("theory", 0) != len(folded.get("theories") or {}):
        fails.append(f"theory artifacts={counts.get('theory', 0)} != folded theories="
                     f"{len(folded.get('theories') or {})} (store/artifact desync)")
    # G6: the BYO extraction_results we injected must appear in the folded store.
    if not (folded.get("extraction_results") or {}):
        fails.append("injected extraction_results absent from folded store (round-trip gap)")
    ctx.theory_store_input = as_theory_store_input(folded)
    if ctx.theory_store_input:
        print(f"    captured theory_store ({len(ctx.theory_store_input['theories'])} theories)")
    return (not fails), "; ".join(fails) or "OK"


def case_form_theory_run_id(ctx: Ctx) -> tuple[bool, str]:
    """Resume a prior run by run_id -> theories, no new extraction."""
    if not ctx.fe_run_id:
        return False, "skipped: no run_id (run find-and-extract first)"
    task = ctx.run("form-theory", {
        "run_id": ctx.fe_run_id,
        "generation_temperature": 0.0,
    })
    ok, msg = _check_completed(task, "form-theory(run_id)")
    if not ok:
        return ok, msg
    counts = artifact_type_counts(task)
    print(f"    artifact counts: {counts}")
    fails = _assert_counts(counts, {
        "theory": (">=", 1),
        "extraction": ("=0", 0),
        "novelty": ("=0", 0),
        "theory_store": ("==", 1),
    })
    folded = folded_theory_store(task)
    for tid, th in (folded.get("theories") or {}).items():
        print(f"    {tid} supporting_evidence_ids={th.get('supporting_evidence_ids')}")
    fails += supporting_evidence_fails(folded)
    fails += content_floor_fails(folded, theory=True)
    fails += theory_cites_fails(folded)
    # Resumed run's evidence must be present, else supporting_evidence_fails passes vacuously (empty results -> floor unarmed).
    if not (folded.get("extraction_results") or {}):
        fails.append("resumed run has no extraction_results in folded store (run_id didn't rehydrate evidence)")
    if counts.get("theory", 0) != len(folded.get("theories") or {}):
        fails.append(f"theory artifacts={counts.get('theory', 0)} != folded theories="
                     f"{len(folded.get('theories') or {})} (store/artifact desync)")
    return (not fails), "; ".join(fails) or "OK"


def case_evaluate_novelty_run_id(ctx: Ctx) -> tuple[bool, str]:
    """Novelty from a run that already has theories. ZERO new theory."""
    if not ctx.fe_run_id:
        return False, "skipped: no run_id (run find-and-extract + form-theory first)"
    task = ctx.run("evaluate-novelty", {"run_id": ctx.fe_run_id})
    ok, msg = _check_completed(task, "evaluate-novelty(run_id)")
    if not ok:
        return ok, msg
    counts = artifact_type_counts(task)
    print(f"    artifact counts: {counts}")
    fails = _assert_counts(counts, {
        "novelty": (">=", 1),
        "theory": ("=0", 0),
        "extraction": ("=0", 0),
        "theory_store": ("==", 1),
    })
    folded = folded_theory_store(task)
    # Resumed run's theories must be present — proves novelty scored the rehydrated run, not an empty/stale store (>=1 alone too weak).
    if not (folded.get("theories") or {}):
        fails.append("resumed run has no theories in folded store (run_id didn't rehydrate)")
    fails += novelty_coverage_fails(folded, counts.get("novelty", 0))
    return (not fails), "; ".join(fails) or "OK"


def case_evaluate_novelty_byo_store(ctx: Ctx) -> tuple[bool, str]:
    """BYO theory_store (+ BYO paper_store) -> novelty, no theory."""
    if not ctx.theory_store_input:
        return False, "skipped: no captured theory_store (run form-theory(byo-results) first)"
    task = ctx.run("evaluate-novelty", {
        "theory_store": ctx.theory_store_input,
        "paper_store": fulltext_paperstore(),
    })
    ok, msg = _check_completed(task, "evaluate-novelty(byo-store)")
    if not ok:
        return ok, msg
    counts = artifact_type_counts(task)
    print(f"    artifact counts: {counts}")
    fails = _assert_counts(counts, {
        "novelty": (">=", 1),
        "theory": ("=0", 0),
        "extraction": ("=0", 0),
        "theory_store": ("==", 1),
    })
    folded = folded_theory_store(task)
    # G6: the injected theory_store must round-trip into the folded store.
    if not (folded.get("theories") or {}):
        fails.append("injected theory_store absent from folded store (round-trip gap)")
    # G1: novelty must cover every eligible statement (full-corpus, no skip).
    fails += novelty_coverage_fails(folded, counts.get("novelty", 0))
    return (not fails), "; ".join(fails) or "OK"


def case_resume_extraction(ctx: Ctx) -> tuple[bool, str]:
    """Add a full-text paper to a prior run -> extractions only, no theory."""
    if not ctx.fe_run_id:
        return False, "skipped: no run_id (run find-and-extract first)"
    task = ctx.run("resume-extraction", {
        "run_id": ctx.fe_run_id,
        # G3: metadata-only seed -> resume (no FIND_PAPERS step) must hydrate it inline.
        "paper_store": {"paperstore": {"seed-resume-meta": {
            "corpus_id": RESUME_HYDRATION_CORPUS_ID, "status": "metadata-only"}}},
    })
    ok, msg = _check_completed(task, "resume-extraction")
    if not ok:
        return ok, msg
    counts = artifact_type_counts(task)
    print(f"    artifact counts: {counts}")
    fails = _assert_counts(counts, {
        "extraction": (">=", 1),
        "theory": ("=0", 0),
        "novelty": ("=0", 0),
        "theory_store": ("==", 1),
    })
    folded = folded_theory_store(task)
    fails += content_floor_fails(folded, extraction=True)
    # G3: the injected metadata-only paper must be hydrated to full-text inline.
    ps = folded.get("paperstore") or {}
    entry = next((e for e in ps.values() if str(e.get("corpus_id")) == RESUME_HYDRATION_CORPUS_ID), None)
    if entry is None:
        fails.append(f"resume seed {RESUME_HYDRATION_CORPUS_ID}: absent from folded paperstore")
    elif entry.get("status") != "full-text":
        fails.append(f"resume seed {RESUME_HYDRATION_CORPUS_ID}: status={entry.get('status')!r} "
                     f"(BUG SIGNATURE: inline hydration didn't run)")
    # Added paper must produce a NEW extraction; extraction>=1 alone is satisfied by prior run's carried-over extractions.
    prior = len((ctx.fe_store or {}).get("extraction_results") or {})
    now = len(folded.get("extraction_results") or {})
    if now <= prior:
        fails.append(f"extraction_results did not grow on resume ({prior} -> {now}); "
                     f"the added paper produced no new extraction")
    return (not fails), "; ".join(fails) or "OK"


def case_find_extract_hydration(ctx: Ctx) -> tuple[bool, str]:
    """metadata-only seeds + PaperFinder -> server-side S2+OCR hydration."""
    MAX = 6  # > len(seeds) so FIND_PAPERS has budget to discover beyond the injected seeds
    task = ctx.run("find-and-extract", {
        "theory_query": THEORY_QUERY,
        "paper_store": seed_paperstore(HYDRATION_SEED_CORPUS_IDS),
        "max_papers_to_retrieve": MAX,
        "search_additional_papers": True,
    })
    ok, msg = _check_completed(task, "find-and-extract(hydration)")
    if not ok:
        return ok, msg
    counts = artifact_type_counts(task)
    print(f"    artifact counts: {counts}")
    fails = _assert_counts(counts, {
        "extraction-schema": ("==", 1),
        "extraction": (">=", 1),
        "theory": ("=0", 0),
        "theory_store": ("==", 1),
    })
    ps = folded_paperstore(task)
    print(f"    folded paperstore: {len(ps)} papers")
    # G7: every metadata-only seed must be hydrated to full-text.
    for cid in HYDRATION_SEED_CORPUS_IDS:
        entry = next((e for e in ps.values() if str(e.get("corpus_id")) == cid), None)
        if entry is None:
            fails.append(f"seed {cid}: absent from folded paperstore")
        elif entry.get("status") != "full-text":
            fails.append(f"seed {cid}: status={entry.get('status')!r} (expected full-text after hydration)")
    # M1: PaperFinder must discover at least one paper beyond the injected seeds.
    seed_set = set(HYDRATION_SEED_CORPUS_IDS)
    if not [e for e in ps.values() if str(e.get("corpus_id")) not in seed_set]:
        fails.append("PaperFinder discovered 0 papers beyond the seeds "
                     "(BUG SIGNATURE: search_additional_papers found nothing)")
    # M4: retrieval must respect max_papers_to_retrieve (cost control).
    if len(ps) > MAX:
        fails.append(f"folded paperstore has {len(ps)} papers > max_papers_to_retrieve={MAX} (cost-control breach)")
    return (not fails), "; ".join(fails) or "OK"


def case_literature_theory_generation(ctx: Ctx) -> tuple[bool, str]:
    """Full pipeline: PaperFinder + extraction + theory (+ novelty)."""
    task = ctx.run("literature-theory-generation", {
        "theory_query": THEORY_QUERY,
        "paper_store": seed_paperstore(HYDRATION_SEED_CORPUS_IDS),
        "max_papers_to_retrieve": 3,
        "search_additional_papers": True,
        "do_qualified_novelty_evaluation": True,
    })
    ok, msg = _check_completed(task, "literature-theory-generation")
    if not ok:
        return ok, msg
    counts = artifact_type_counts(task)
    print(f"    artifact counts: {counts}")
    fails = _assert_counts(counts, {
        "extraction-schema": ("==", 1),
        "extraction": (">=", 1),
        "theory": (">=", 1),
        "novelty": (">=", 1),
        "theory_store": ("==", 1),
    })
    folded = folded_theory_store(task)
    fails += content_floor_fails(folded, schema=True, extraction=True, theory=True)
    fails += supporting_evidence_fails(folded)
    fails += theory_cites_fails(folded)
    # G1: novelty must cover every eligible statement (full-corpus, no skip).
    fails += novelty_coverage_fails(folded, counts.get("novelty", 0))
    # G4: published theory artifacts must round-trip 1:1 into the folded store.
    if counts.get("theory", 0) != len(folded.get("theories") or {}):
        fails.append(f"theory artifacts={counts.get('theory', 0)} != folded theories="
                     f"{len(folded.get('theories') or {})} (store/artifact desync)")
    return (not fails), "; ".join(fails) or "OK"


# (name, tier, fn). Order matters: dependencies first.
CASES: list[tuple[str, str, Callable[[Ctx], tuple[bool, str]]]] = [
    ("build-extraction-schema",        "cheap",     case_build_extraction_schema),
    ("graceful-empty",                 "cheap",     case_graceful_empty),
    ("find-extract-byo-fulltext",      "medium",    case_find_extract_byo_fulltext),
    ("validation-errors",              "cheap",     case_validation_errors),
    ("find-extract-byo-schema",        "medium",    case_find_extract_byo_schema),
    ("find-extract-uri",               "medium",    case_find_extract_uri),
    ("form-theory-byo-results",        "medium",    case_form_theory_byo_results),
    ("form-theory-run-id",             "medium",    case_form_theory_run_id),
    ("evaluate-novelty-run-id",        "medium",    case_evaluate_novelty_run_id),
    ("evaluate-novelty-byo-store",     "medium",    case_evaluate_novelty_byo_store),
    ("resume-extraction",              "medium",    case_resume_extraction),
    ("find-extract-hydration",         "expensive", case_find_extract_hydration),
    ("literature-theory-generation",   "expensive", case_literature_theory_generation),
]

TIERS = ["cheap", "medium", "expensive"]


# --- Query-list mode: replay a list/CSV of real queries as matrix cases -------
# Lets us drive the matrix from a batch of submitted queries (e.g.
# output/theory_queries_last_hour.csv from a bug bash) instead of the fixed CASES,
# while DEDUPING identical inputs so we never burn a run on the same query twice.

def _dedupe_queries(queries: list[str]) -> list[str]:
    """Trim, drop blanks, and keep first occurrence of each distinct query (order-preserving)."""
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        q = (q or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        unique.append(q)
    return unique


def load_queries_csv(path: str, column: str = "theory_query") -> list[str]:
    """Read the `theory_query` column from a CSV (header required)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"CSV {path!r} has no {column!r} column (columns: {reader.fieldnames})")
        return [row.get(column, "") for row in reader]


def _make_query_case(query: str, skill: str, byo_papers: int, paper_kb: int, max_papers: int = 0):
    """Build a matrix case that submits one query and asserts it reaches `completed`.

    byo_papers>0 injects synthetic full-text papers + disables PaperFinder, so the run
    exercises the full schema->extract->theory->novelty pipeline (and the cleanup paths)
    without paying for paper discovery/OCR — pair with a THEORIZER_FAKE_LLM container for a
    $0 end-to-end pass. byo_papers=0 is the real path (PaperFinder + live extraction).
    max_papers>0 caps retrieval (cost/time control) for paper-fetching skills."""
    def case(ctx: Ctx) -> tuple[bool, str]:
        data: dict[str, Any] = {"theory_query": query}
        if max_papers > 0 and skill in ("literature-theory-generation", "find-and-extract"):
            data["max_papers_to_retrieve"] = max_papers
        if byo_papers > 0:
            blob = "This sentence pads the synthetic paper body to a realistic size. " * 16
            body = blob * max(1, (paper_kb * 1024) // len(blob))
            data["paper_store"] = {
                "paperstore": {
                    f"q-paper-{k}": {
                        "corpus_id": f"8{k:06d}", "status": "full-text",
                        "title": f"Synthetic paper {k}", "paper_markdown": f"# Synthetic {k}\n\n{body}",
                    } for k in range(byo_papers)
                },
                "paperstore_lut": {f"q-paper-{k}": f"q-paper-{k}" for k in range(byo_papers)},
            }
            data["search_additional_papers"] = False
        task = ctx.run(skill, data)
        return _check_completed(task, f"{skill}: {query[:50]!r}")
    return case


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:9100")
    p.add_argument("--api-key", default=os.environ.get("API_KEY") or os.environ.get("ASTA_A2A_API_KEY") or "")
    p.add_argument("--timeout", type=int, default=90 * 60,
                   help="Per-case polling timeout in seconds (default 90 min).")
    p.add_argument("--tier", nargs="+", choices=TIERS, default=TIERS,
                   help="Which cost tiers to run. Default: all.")
    p.add_argument("--include", nargs="+", default=None,
                   help="Explicit subset of case names (overrides --tier).")
    p.add_argument("--paperstore-uri", default=os.environ.get("PAPERSTORE_URI") or None,
                   help="file://, https://, or s3:// URI to a wrapped paperstore JSON, "
                        "reachable from the SERVER, for the URI case.")
    p.add_argument("--queries-csv", default=None,
                   help="Run one case per UNIQUE query from this CSV's theory_query column "
                        "(replaces the fixed CASES). Duplicate queries are skipped.")
    p.add_argument("--queries", nargs="+", default=None,
                   help="Run one case per UNIQUE query given inline (replaces the fixed CASES).")
    p.add_argument("--query-skill", default="literature-theory-generation",
                   help="Skill to invoke for --queries-csv / --queries cases.")
    p.add_argument("--query-byo-papers", type=int, default=0,
                   help="Inject N synthetic full-text papers per query + disable PaperFinder "
                        "(free end-to-end path; pair with a THEORIZER_FAKE_LLM container). "
                        "0 = real PaperFinder + live extraction.")
    p.add_argument("--query-paper-kb", type=int, default=50,
                   help="Approx KB of markdown per synthetic paper (with --query-byo-papers).")
    p.add_argument("--query-max-papers", type=int, default=0,
                   help="Cap papers retrieved per query (0 = server default). Bounds cost/time.")
    args = p.parse_args()

    ctx = Ctx(args.url, args.api_key or None, args.timeout, args.paperstore_uri)

    if args.queries_csv or args.queries:
        raw = load_queries_csv(args.queries_csv) if args.queries_csv else list(args.queries)
        unique = _dedupe_queries(raw)
        skipped = len(raw) - len(unique)
        src = args.queries_csv or "inline"
        print(f"Query-list mode: {len(raw)} queries from {src}, {len(unique)} unique "
              f"({skipped} duplicate{'s' if skipped != 1 else ''} skipped) | skill={args.query_skill} "
              f"| byo_papers={args.query_byo_papers}", flush=True)
        selected = [
            (f"q{i:02d}:{q[:40]}", "queries",
             _make_query_case(q, args.query_skill, args.query_byo_papers, args.query_paper_kb, args.query_max_papers))
            for i, q in enumerate(unique)
        ]
    elif args.include:
        selected = [(n, t, f) for (n, t, f) in CASES if n in args.include]
    else:
        selected = [(n, t, f) for (n, t, f) in CASES if t in args.tier]

    print(f"Target: {args.url}  |  tiers={args.tier}  |  {len(selected)} cases", flush=True)

    results: list[tuple[str, str, str]] = []
    for name, tier, fn in selected:
        print(f"\n=== [{tier}] {name} ===", flush=True)
        try:
            ok, msg = fn(ctx)
        except (error.HTTPError, error.URLError, TimeoutError) as e:
            ok, msg = False, f"transport: {e!r}"
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"exception: {e!r}"
        # A case that couldn't run (missing dependency / unconfigured URI) reports SKIP — never a silent PASS — and doesn't fail the suite.
        status = "SKIP" if msg.startswith("skipped") else ("PASS" if ok else "FAIL")
        results.append((name, status, msg))
        print(f"    {status}: {msg}", flush=True)

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    mark = {"PASS": "✓", "SKIP": "–", "FAIL": "✗"}
    for name, status, msg in results:
        print(f"  {mark[status]} {name:<34} {'OK' if status == 'PASS' else msg}")
    passed = sum(s == "PASS" for _, s, _ in results)
    skipped = sum(s == "SKIP" for _, s, _ in results)
    failed = sum(s == "FAIL" for _, s, _ in results)
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed (of {len(results)})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
