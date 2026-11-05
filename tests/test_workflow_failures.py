"""Regression coverage for worker failures and follow-on result accounting."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest

import EvaluationQualifiedNovelty as novelty
import Theorizer1 as T
from asta.context import inject_papers
from asta.schemas import PaperStore


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    paper_store = Mock(paperstore={}, paperstore_lut={})
    monkeypatch.setattr(T, "PaperStore", Mock(return_value=paper_store))
    monkeypatch.setattr(T, "DEBUG_OUTPUT_DIR", str(tmp_path))
    wf = T.TheoryGenerationWorkflowStructure(generation_id="gen-failure-test")
    wf.set_steps([T.Step.EXTRACT_FROM_PAPERS, T.Step.COMPLETE])
    yield wf
    wf.cleanup()


def test_processor_exception_reaches_terminal_error(workflow):
    failed = threading.Event()
    workflow.on_step_change = lambda wf: failed.set() if wf.is_in_error_state() else None
    processor = T.StepProcessor("test-step", max_simultaneous_work=1)
    future = Future()
    future.set_exception(RuntimeError("worker failed"))
    processor.submit_callable = lambda wf: future
    processor.start_thread()
    try:
        processor.add_workflow_to_queue(workflow)
        assert failed.wait(3), "failed worker did not notify the workflow"
        assert workflow.is_current_step_completed()
        assert "worker failed" in workflow.theory_workflow_structure["errors"][-1]
    finally:
        processor.stop_thread()


@pytest.mark.parametrize("executor,dependency", [
    (T.extract_from_papers_executor, "SchemaExtractionQueue"),
    (T.make_theory_from_literature_executor, "consolodate_results_with_subsampling"),
])
def test_direct_executor_exception_fails_workflow(workflow, monkeypatch, executor, dependency):
    workflow.theory_workflow_structure["completed_extraction_results"] = [{"evidence": "data"}]
    workflow.theory_workflow_structure["extraction_schema_id"] = "schema-1"
    if dependency == "SchemaExtractionQueue":
        queue = Mock()
        queue.submit_extraction_request.side_effect = RuntimeError("dependency failed")
        monkeypatch.setattr(T, dependency, Mock(return_value=queue))
    else:
        monkeypatch.setattr(T, dependency, Mock(side_effect=RuntimeError("dependency failed")))
    executor(workflow)
    assert workflow.is_in_error_state()
    assert "dependency failed" in workflow.theory_workflow_structure["errors"][-1]


@pytest.mark.parametrize("rate,expected", [(None, "numeric rate"), (1.0, "No theory result")])
def test_missing_theory_outputs_fail_promptly(workflow, monkeypatch, rate, expected):
    workflow.theory_workflow_structure["completed_extraction_results"] = [{"evidence": "data"}]
    monkeypatch.setattr(T, "consolodate_results_with_subsampling", Mock(return_value={
        "data_subsampled": [{"evidence": "data"}], "subsampling_rate": rate,
    }))
    monkeypatch.setattr(T, "build_theory_from_results_single_theory_reflection3", Mock(return_value=None))
    T.make_theory_from_literature_executor(workflow)
    assert workflow.is_in_error_state()
    assert expected in workflow.theory_workflow_structure["errors"][-1]


@pytest.mark.parametrize("paper_result", ["paper-2", {"error": "conversion failed"}])
@pytest.mark.parametrize("resumed", [False, True])
def test_follow_on_results_and_cost_use_result_ids(workflow, monkeypatch, paper_result, resumed):
    state = workflow.theory_workflow_structure
    state.update(extraction_schema_id="schema-1", paper_keys=["paper-1"], max_papers_to_retrieve=3 if resumed else 2)
    initial = SimpleNamespace(
        extracted_data=[{"evidence": "initial"}], cost=2.0,
        potentially_relevant_new_papers=[{"paper_title": "A follow-on paper title", "rating": 2}],
    )
    follow_on = SimpleNamespace(extracted_data=[{"evidence": "follow-on"}], cost=3.0)
    results = {"result-1": initial, "result-2": follow_on}
    if resumed:
        results["prior-result"] = SimpleNamespace(
            extracted_data=[{"evidence": "prior"}], cost=4.0, potentially_relevant_new_papers=[],
        )
        state["completed_extraction_result_ids"] = ["prior-result"]
        state["completed_extraction_results"] = [{"evidence": "prior"}]
        state["costs"]["paper_extraction"] = 4.0
    workflow.theory_store = SimpleNamespace(
        paperstore=workflow.paper_store,
        get_extraction_result=lambda id: results[id],
    )
    queue = Mock()
    queue.submit_extraction_request.side_effect = [["request-1"], ["request-2"]]
    queue.is_request_complete.side_effect = lambda request_id: {
        "request-1": "result-1", "request-2": "result-2",
    }[request_id]
    monkeypatch.setattr(T, "SchemaExtractionQueue", Mock(return_value=queue))
    # All requests are terminal on the first poll, including conversion failures.
    monkeypatch.setattr(T.time, "sleep", Mock(side_effect=AssertionError("unnecessary wait")))
    workflow.paper_store.submit_paper.return_value = "paper-request-2"
    workflow.paper_store.is_paper_request_complete.return_value = paper_result
    valid = ["paper-2"] if isinstance(paper_result, str) else []
    workflow.paper_store.get_num_papers_with_full_text.return_value = (len(valid), valid)
    T.extract_from_papers_executor(workflow)
    assert not workflow.is_in_error_state(), state["errors"]
    assert workflow.is_current_step_completed()
    expected_ids = (["prior-result"] if resumed else []) + ["result-1"] + (["result-2"] if valid else [])
    assert state["completed_extraction_result_ids"] == expected_ids
    assert state["costs"]["paper_extraction"] == (5.0 if valid else 2.0) + (4.0 if resumed else 0.0)
    assert state["follow_on_paper_ids"] == valid


@pytest.mark.parametrize("uri", [
    "file:///seed/paperstore.json", "s3://example/paperstore.json",
    "http://127.0.0.1/paperstore.json", "https://example.com/paperstore.json",
])
def test_paper_store_uri_rejected_before_merge(workflow, uri):
    with pytest.raises(ValueError, match="URI inputs are not supported"):
        inject_papers(workflow, uri)
    workflow.paper_store.merge_from_dict.assert_not_called()


def test_inline_paper_store_still_merges(workflow):
    inject_papers(workflow, PaperStore(paperstore={}))
    workflow.paper_store.merge_from_dict.assert_called_once_with({"paperstore": {}, "paperstore_lut": {}})


@pytest.mark.parametrize("theory_id,index", [("missing", 0), ("theory-1", 3)])
def test_novelty_lookup_miss_keeps_store_valid(monkeypatch, theory_id, index):
    store = {
        "theories": {"theory-1": {"components": {"theory_statements": [{"law": {}}]}}},
        "extraction_schemas": {}, "extraction_results": {},
    }
    monkeypatch.setattr(novelty, "generate_lut_theory_query_to_paper_ids", Mock(return_value={}))
    monkeypatch.setattr(novelty, "load_theory_statements_to_evaluate_from_theorystore_from_dict", Mock(return_value=[{}]))
    monkeypatch.setattr(novelty, "evaluate_novelty_theory_statement_original_papers", Mock(return_value={
        "theory_id": theory_id, "statement_index": index,
        "final_aggregated_novelty_evaluation": {}, "evidence_for_dimension": {},
    }))
    callback = Mock()
    result = novelty.do_qualified_novelty_evaluation_persistence(store, Mock(), "fake", "fake", on_statement_evaluated=callback)
    assert result["theorystore_data"] is store
    assert isinstance(result["theorystore_data"], dict)
    callback.assert_not_called()
    # A later valid evaluation still succeeds after a lookup miss.
    assert novelty.add_qualified_novelty_result_to_theorystore_dict(store, "theory-1", 0, {}, {}) is store
