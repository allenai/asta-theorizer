"""Empty PaperFinder results carry a code through the real SDK failure message."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from a2a.types import TaskState
from asta_agent import AstaContext

import Theorizer1 as T
from asta.workflow import AstaWorkflow


@pytest.fixture
def workflow():
    wf = T.TheoryGenerationWorkflowStructure(generation_id="gen-error-code-test")
    try:
        yield wf
    finally:
        wf.cleanup()


def _failure_message(workflow):
    # Exercise the installed SDK's fail -> async_fail serialization, capturing
    # exactly the message that the webhook reads rather than a stub's code field.
    # Private setup mirrors asta-agent==1.0.20, pinned in requirements.txt.
    ctx = AstaContext.__new__(AstaContext)
    ctx.task_id = "task"
    ctx.context_id = "context"
    ctx._current_step = None
    ctx._async_finish_all_children = AsyncMock()
    ctx._update_task_status = AsyncMock()
    ctx._run_async = asyncio.run
    AstaWorkflow(ctx, workflow, completion_msg_fn=lambda *_: "done")._fail()
    call = ctx._update_task_status.call_args
    assert call.args == (TaskState.failed,)
    assert call.kwargs["final"] is True
    return call.kwargs["message"]


def test_no_papers_has_error_code_in_failure_message(workflow):
    requests = Mock()
    requests.get_work.return_value = {"result": {"doc_collection": {"documents": []}}}
    T.make_paperfinder_request_executor(workflow, requests)

    assert workflow.is_in_error_state()
    message = _failure_message(workflow)
    assert message.metadata == {"error_code": "Papers_not_found"}
    assert message.parts[0].root.text == "ERROR: PaperFinder returned no papers for this query."


@pytest.mark.parametrize("response", [
    {}, [], "invalid",
    {"result": None}, {"result": []}, {"result": "invalid"}, {"result": {}},
    {"result": {"doc_collection": None}},
    {"result": {"doc_collection": []}},
    {"result": {"doc_collection": "invalid"}},
    {"result": {"doc_collection": {}}},
    *[
        {"result": {"doc_collection": {"documents": documents}}}
        for documents in (None, {}, "", "invalid", 0, False)
    ],
])
def test_malformed_paperfinder_response_is_not_papers_not_found(workflow, response):
    requests = Mock()
    requests.get_work.return_value = response
    T.make_paperfinder_request_executor(workflow, requests)

    assert workflow.is_in_error_state()
    message = _failure_message(workflow)
    assert not message.metadata
    assert "Invalid response from PaperFinder" in message.parts[0].root.text


def test_papers_excluded_by_cutoff_is_not_papers_not_found(workflow):
    documents = [{"corpus_id": "123", "publication_date": "2020-01-01"}]
    workflow.theory_workflow_structure["knowledge_cutoff_year"] = 2019
    workflow.theory_workflow_structure["knowledge_cutoff_month"] = 1
    requests = Mock()
    requests.get_work.return_value = {"result": {"doc_collection": {"documents": documents}}}
    T.make_paperfinder_request_executor(workflow, requests)

    assert workflow.is_in_error_state()
    message = _failure_message(workflow)
    assert not message.metadata
    assert "after applying the knowledge cutoff" in message.parts[0].root.text
    assert workflow.theory_workflow_structure["papers_from_paperfinder_unfiltered"] == documents
    assert workflow.theory_workflow_structure["papers_from_paperfinder"] == []


def test_paperfinder_failure_is_not_papers_not_found(workflow):
    requests = Mock()
    requests.get_work.return_value = {"error": "PaperFinder service unavailable"}
    T.make_paperfinder_request_executor(workflow, requests)

    message = _failure_message(workflow)
    assert not message.metadata
    assert "service unavailable" in message.parts[0].root.text


def test_paperfinder_timeout_is_not_papers_not_found(workflow, monkeypatch):
    monkeypatch.setattr(T, "PAPERFINDER_TOTAL_WAIT_SECONDS", 0)
    requests = Mock()
    requests.get_work.return_value = None
    T.make_paperfinder_request_executor(workflow, requests)

    message = _failure_message(workflow)
    assert not message.metadata
    assert "Timeout reached" in message.parts[0].root.text


def test_uncoded_error_clears_previous_code(workflow):
    workflow.set_error_state("No papers", error_code="Papers_not_found")
    workflow.set_error_state("Different error")
    message = _failure_message(workflow)
    assert not message.metadata
    assert message.parts[0].root.text == "Different error"


def test_new_workflow_has_no_error_code(workflow):
    assert workflow.theory_workflow_structure["error_code"] is None


@pytest.mark.parametrize("state_code", [{}, {"error_code": "Papers_not_found"}])
def test_persisted_code_survives_load_and_is_cleared_for_new_run(
    workflow, monkeypatch, state_code,
):
    workflow.set_error_state("No papers", error_code="Papers_not_found")
    state = dict(workflow.theory_workflow_structure)
    state.pop("error_code")
    state.update(state_code)
    store = Mock()
    store.get_workflow.return_value = state
    store.get_paperstore.return_value = None
    store.get_theorystore.return_value = None
    monkeypatch.setattr(T.Persistence, "store", lambda: store)

    restored = T.TheoryGenerationWorkflowStructure.from_persistence(state["generation_id"])
    try:
        assert restored.theory_workflow_structure["error_code"] == state_code.get("error_code")
        restored.set_steps([T.Step.INIT, T.Step.COMPLETE])
        assert restored.theory_workflow_structure["error_code"] is None
        restored.set_error_state("Different error after resume")
        message = _failure_message(restored)
        assert not message.metadata
        assert message.parts[0].root.text == "Different error after resume"
    finally:
        restored.cleanup()
