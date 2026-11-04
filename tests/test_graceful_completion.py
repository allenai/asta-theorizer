"""Deterministic unit test for graceful (non-error) terminal completion.

Covers the new `set_completed_state()` primitive added so that legitimate empty results
(e.g. PaperFinder found papers but none had accessible full text) finish as a SUCCESS
terminal with an explanatory message, instead of erroring with a misleading "timeout".
Asserts it's the symmetric counterpart to `set_error_state()` and doesn't regress it.

    .venv/bin/python tests/test_graceful_completion.py
"""

import os
import sys
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import Theorizer1 as T  # noqa: E402
from Theorizer1 import Step  # noqa: E402


def _wf():
    return T.TheoryGenerationWorkflowStructure(generation_id="gen-graceful-test")


def test_set_completed_state_is_graceful_success_terminal():
    wf = _wf()
    try:
        note = "Found 3 papers, but none had accessible full text, so no theories were generated."
        wf.set_completed_state(note)
        assert wf.is_at_terminal_step(), "set_completed_state should reach a terminal step"
        assert not wf.is_in_error_state(), "graceful completion must NOT be an error state"
        assert wf.get_current_step_name() == Step.COMPLETE, \
            f"current step should be COMPLETE, got {wf.get_current_step_name()}"
        assert wf.theory_workflow_structure.get("completion_note") == note, \
            "completion_note should carry the explanatory message"
        return f"completed-state OK (step={wf.get_current_step_name()}, error={wf.is_in_error_state()})"
    finally:
        wf.cleanup()


def test_set_error_state_still_errors():
    """Regression: the existing error terminal must remain an error terminal."""
    wf = _wf()
    try:
        wf.set_error_state("ERROR in `x`: boom")
        assert wf.is_at_terminal_step(), "set_error_state should reach a terminal step"
        assert wf.is_in_error_state(), "set_error_state must be an error state"
        assert wf.get_current_step_name() == Step.ERROR
        assert wf.theory_workflow_structure["errors"][-1].endswith("boom")
        return "error-state OK"
    finally:
        wf.cleanup()


class _StubCtx:
    """Minimal AstaContext stand-in that records what the tracker emits."""
    def __init__(self):
        self.task_id = "task"
        self.context_id = "ctx"
        self.emitted = []      # interim messages via _update_task_status
        self.failed_with = None
        self.failed_code = None
        self.completed_with = None
        self.credits = None

    def _update_task_status(self, state, message=None, **kw):
        self.emitted.append(message)
        return None  # the real one returns a coroutine; tracker passes it to _run_async

    def _run_async(self, _coro):
        return None

    def fail(self, error="", code=None):
        self.failed_with = error
        self.failed_code = code

    def complete(self, message=None):
        self.completed_with = message

    def set_metadata(self, key, value):
        pass


def test_failure_emits_interim_message():
    """Part 2: a workflow that ends in error must emit a user-facing MESSAGE (symmetric with
    the completion message), not just call ctx.fail() (which carries no message body)."""
    from asta.workflow import AstaWorkflow
    wf = _wf()
    wf.set_error_state("ERROR in `extract-from-papers`: boom reason here")
    ctx = _StubCtx()
    tracker = AstaWorkflow(ctx, wf, completion_msg_fn=lambda w, _c: "done")
    try:
        tracker._fail()
        assert ctx.emitted, "failure should emit an interim message before ctx.fail()"
        dumped = str(ctx.emitted[-1].model_dump() if hasattr(ctx.emitted[-1], "model_dump") else ctx.emitted[-1])
        assert "boom reason here" in dumped, f"interim message missing the reason: {dumped[:200]}"
        assert ctx.failed_with and "boom reason here" in ctx.failed_with, "ctx.fail should still carry the reason"
        return "failure emits interim message + ctx.fail"
    finally:
        wf.cleanup()


class _StubStore:
    """Streaming theory_store stand-in whose finalize blows up the way a timed-out
    _run_async does (nora thread 726c7133, 2026-09-18)."""
    artifact_id = "gen-guard-test-theory-store-raw"

    def __init__(self, exc):
        self.exc = exc

    def finalize(self, _data=None):
        raise self.exc


def test_finalize_failure_still_completes_with_artifact_tag():
    """A failed terminal artifact write must not overturn a success already decided from
    workflow state: the user still gets the completion message AND the <artifact> tag,
    which is the webapp's only trigger for opening the theory store."""
    from asta.workflow import AstaWorkflow
    wf = _wf()
    wf.set_steps([Step.INIT, Step.COMPLETE])
    ctx = _StubCtx()
    tracker = AstaWorkflow(ctx, wf, completion_msg_fn=lambda w, _c: "Generated 4 theories:")
    tracker.raw_store = _StubStore(TimeoutError())
    try:
        tracker._succeed(wf)
        assert ctx.failed_with is None, f"must not fail the run: {ctx.failed_with!r}"
        assert ctx.completed_with, "completion message should still be sent"
        assert "Generated 4 theories:" in ctx.completed_with
        assert '<artifact id="gen-guard-test-theory-store-raw">' in ctx.completed_with, \
            f"completion message lost the artifact tag: {ctx.completed_with!r}"
        return "finalize timeout still completes with tag"
    finally:
        wf.cleanup()


class _StubCredits:
    def __init__(self, fail=False):
        self.fail = fail

    def has_credits_backend(self):
        return True

    def consume(self, *_a, **_kw):
        if self.fail:
            raise RuntimeError("402")
        return None


def _assert_msg_fn_called_with_credits(credits, expected):
    """Run _succeed with `credits` as the account; the message fn must be called once
    with (workflow, expected)."""
    from asta.workflow import AstaWorkflow
    wf = _wf()
    wf.set_steps([Step.INIT, Step.COMPLETE])
    ctx = _StubCtx()
    ctx.credits = credits
    msg_fn = Mock(return_value="msg")
    tracker = AstaWorkflow(ctx, wf, completion_msg_fn=msg_fn)
    try:
        tracker._succeed(wf)
        msg_fn.assert_called_once_with(wf, expected)
    finally:
        wf.cleanup()


def test_message_fn_receives_credits_charged():
    # Paperless run prices at BASE_CREDITS.
    from asta.credits import BASE_CREDITS
    _assert_msg_fn_called_with_credits(_StubCredits(), BASE_CREDITS)
    for credits in (_StubCredits(fail=True), None):
        _assert_msg_fn_called_with_credits(credits, None)
    return "message fn gets the charged amount, or None"


def test_finalize_failure_does_not_mask_a_hard_error():
    """The guard is scoped to the terminal write only -- a workflow that genuinely errored
    still routes through _fail()."""
    from asta.workflow import AstaWorkflow
    wf = _wf()
    wf.set_error_state("ERROR in `extract-from-papers`: boom")
    ctx = _StubCtx()
    tracker = AstaWorkflow(ctx, wf, completion_msg_fn=lambda w, _c: "done")
    tracker.raw_store = _StubStore(TimeoutError())
    try:
        tracker._terminate(wf)
        assert ctx.completed_with is None, "an errored workflow must not report completion"
        assert ctx.failed_with and "boom" in ctx.failed_with
        return "hard error still fails"
    finally:
        wf.cleanup()


TESTS = [
    test_set_completed_state_is_graceful_success_terminal,
    test_set_error_state_still_errors,
    test_failure_emits_interim_message,
    test_finalize_failure_still_completes_with_artifact_tag,
    test_finalize_failure_does_not_mask_a_hard_error,
    test_message_fn_receives_credits_charged,
]


def main():
    failures = 0
    for fn in TESTS:
        try:
            print(f"PASS  {fn.__name__}: {fn()}", flush=True)
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {fn.__name__}: {e!r}", flush=True)
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
