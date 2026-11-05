"""Deterministic regression test for the OOM-under-load incident (2026-06-15).

The crash was a Cloud Run OOM, but the *cause* is two deterministic, per-workflow
leaks that need no load to observe:

  1. Every workflow spins up a PaperStore monitor thread that was never stopped.
  2. Finished workflows were retained forever in `workflows_completed`, pinning
     their PaperStore (full paper text / OCR markdown) in memory.

A leak is monotonic: if one workflow leaks a thread/object, a thousand do too and
eventually exceed any memory ceiling. So instead of reproducing the OOM under load,
we assert the *invariant whose violation causes it*: after workflows are retired,
live thread count returns to baseline and the retired workflows are garbage-collected.

Runs in seconds with no LLM/PaperFinder calls. From the repo root:
    .venv/bin/python tests/test_memory_leak.py
or inside the container image:
    docker run --rm -v "$PWD/src:/app/src" -v "$PWD/tests:/app/tests" \
        --entrypoint python asta-theorizer:local tests/test_memory_leak.py
"""

import gc
import os
import sys
import threading
import time
import weakref

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Keep the retained window small so the trim is exercised by a modest workflow count.
os.environ.setdefault("MAX_COMPLETED_RETAINED", "5")

import Theorizer1 as T  # noqa: E402

N_WORKFLOWS = 30
# stop_thread() joins a monitor that sleeps in 1s ticks, so allow a little settle time.
SETTLE_SECONDS = 2.5
# Theorizer construction spins up a fixed set of step-processor/worker threads; allow slack.
THREAD_SLACK = 4


def _active():
    return threading.active_count()


def _settle():
    gc.collect()
    time.sleep(SETTLE_SECONDS)
    gc.collect()


def test_paperstore_thread_is_stopped_on_cleanup():
    """A workflow's PaperStore monitor thread must be gone after cleanup()."""
    base = _active()
    wf = T.TheoryGenerationWorkflowStructure(generation_id="gen-leak-single")
    time.sleep(0.3)
    grew = _active()
    assert grew > base, f"workflow construction should start a monitor thread (base={base}, after={grew})"
    wf.cleanup()
    _settle()
    after = _active()
    assert after <= base + 1, f"thread not reaped after cleanup (base={base}, after={after})"
    return f"single-workflow cleanup: {base} -> {grew} -> {after} threads"


def test_registered_service_is_stopped_on_cleanup():
    """A registered background service (e.g. SchemaExtractionQueue) is stopped too."""
    base = _active()
    wf = T.TheoryGenerationWorkflowStructure(generation_id="gen-leak-svc")
    from SchemaExtractionQueue import SchemaExtractionQueue
    seq = SchemaExtractionQueue(theorystore=wf.theory_store, paperstore=wf.paper_store)
    wf.register_background_service(seq)
    time.sleep(0.3)
    grew = _active()
    assert grew >= base + 2, f"workflow + queue should add >=2 threads (base={base}, after={grew})"
    wf.cleanup()
    _settle()
    after = _active()
    assert after <= base + 1, f"registered service thread not reaped (base={base}, after={after})"
    return f"registered-service cleanup: {base} -> {grew} -> {after} threads"


def test_retire_bounds_memory_and_threads():
    """The core OOM invariant: retiring N workflows through the Theorizer must
    leave thread count at baseline, bound workflows_completed, and let the evicted
    workflows' PaperStores be garbage-collected."""
    th = T.Theorizer(path_in="data/")
    time.sleep(0.5)
    base = _active()

    refs = []  # weakrefs to each workflow's PaperStore — must die once retired+evicted
    for i in range(N_WORKFLOWS):
        wf = T.TheoryGenerationWorkflowStructure(generation_id=f"gen-leak-{i}")
        refs.append(weakref.ref(wf.paper_store))
        th._retire_workflow(wf)
        del wf  # drop our strong ref; only th.workflows_completed may retain it

    peak_completed = len(th.workflows_completed)
    _settle()

    after = _active()
    cap = T.MAX_COMPLETED_RETAINED
    alive_stores = sum(1 for r in refs if r() is not None)

    fails = []
    if len(th.workflows_completed) > cap:
        fails.append(f"workflows_completed not bounded: {len(th.workflows_completed)} > {cap}")
    if after > base + THREAD_SLACK:
        fails.append(f"threads leaked across {N_WORKFLOWS} workflows: base={base}, after={after}")
    # Only the most-recent `cap` workflows are still referenced; the rest must be GC'd.
    if alive_stores > cap:
        fails.append(f"{alive_stores} PaperStores still alive; expected <= {cap} (retained window)")
    if fails:
        raise AssertionError("; ".join(fails))

    th.stop_thread()
    return (f"retired {N_WORKFLOWS} workflows: completed list {peak_completed}->{len(th.workflows_completed)} "
            f"(cap {cap}), threads {base}->{after}, live PaperStores {alive_stores}")


TESTS = [
    test_paperstore_thread_is_stopped_on_cleanup,
    test_registered_service_is_stopped_on_cleanup,
    test_retire_bounds_memory_and_threads,
]


def main():
    failures = 0
    for fn in TESTS:
        try:
            detail = fn()
            print(f"PASS  {fn.__name__}: {detail}", flush=True)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {fn.__name__}: {e!r}", flush=True)
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
