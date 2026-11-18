# TestResume.py
# Running a stage again over the same inputs reuses what was done, and changing an input that
# changes the result does not.
#
#   - Extraction run twice over the same papers leaves one result per paper in the TheoryStore,
#     whether the second run resumes from the checkpoint, starts a new session, or calls the model
#     again and gets the same answer.
#   - A changed paper text, schema id, temperature, max_tokens or use_reflection is not served a
#     cached result.  The same holds for variable discovery, paper import and the type sweep.
#
# The model is replaced by a fake that counts its calls.  Runs offline: no keys, no network.
#
#     python tests/TestResume.py

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


QUERY = "What sets cortical gain?"
SCHEMA = [{"name": "finding", "type": "str", "description": "The main finding"}]


def some_papers():
    return [
        {"paper_id": "paper-a", "title": "A", "markdown": "# A\n\nVIP activation raised gain."},
        {"paper_id": "paper-b", "title": "B", "markdown": "# B\n\nSST silencing raised gain."},
        {"paper_id": "paper-c", "title": "C", "markdown": "# C\n\nLocomotion raised gain."},
    ]


class FakeModel:
    # Stands in for the per-paper engine calls, and counts them
    def __init__(self):
        self.calls = 0

    def extract(self, extraction_query, extraction_schema, paper_text, **kwargs):
        self.calls += 1
        return {"response": {"results": [{"name_short": "finding", "finding": paper_text.splitlines()[-1]}]}, "total_cost": 0.0}

    def discover(self, theory_query_information, paper_text, **kwargs):
        self.calls += 1
        return {"response": {"proposed_slots": [{"slot_name": "gain", "slot_description": paper_text[:20]}]}, "total_cost": 0.0}


class fake_engine:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.saved = (th.Legacy.extract_entities_from_paper, th.Legacy.discover_variables_from_paper)
        th.Legacy.extract_entities_from_paper = self.model.extract
        th.Legacy.discover_variables_from_paper = self.model.discover
        return self.model

    def __exit__(self, *exc):
        th.Legacy.extract_entities_from_paper, th.Legacy.discover_variables_from_paper = self.saved
        return False


def open_project(directory):
    th.Config.init(project_dir=os.path.join(directory, "run"), verbose=False)


def extract(papers, **kwargs):
    return th.Pipeline.extract_from_papers(papers, QUERY, SCHEMA, model_str="fake-model", max_workers=2, show_progress=False, **kwargs)


def results_per_paper(store):
    counts = {}
    for result in store.extraction_results.values():
        counts[result.paper_id] = counts.get(result.paper_id, 0) + 1
    return counts


#
#   Tests: no duplicates on a rerun
#

def test_two_runs_leave_one_result_per_paper(directory):
    open_project(directory)
    with fake_engine(FakeModel()) as model:
        extract(some_papers())
        second = extract(some_papers())
    store = th.Project.get_theory_store()
    assert (results_per_paper(store) == {"paper-a": 1, "paper-b": 1, "paper-c": 1}), "a rerun added results again: " + str(results_per_paper(store))
    assert (model.calls == 3), "the rerun called the model again instead of resuming: " + str(model.calls) + " calls"
    records = th.Pipeline.flatten_records(second, save=False)
    assert (len(records) == 3) and all(record.get("uuid") for record in records), "the rerun's records are not the stored, citable ones: " + str(records)


def test_a_new_session_leaves_one_result_per_paper(directory):
    open_project(directory)
    with fake_engine(FakeModel()):
        extract(some_papers())
        open_project(directory)          # a new session: the store is reloaded from disk
        extract(some_papers())
    store = th.Project.get_theory_store()
    assert (results_per_paper(store) == {"paper-a": 1, "paper-b": 1, "paper-c": 1}), "a new session added results again: " + str(results_per_paper(store))


def test_the_same_answer_from_the_model_is_not_added_twice(directory):
    open_project(directory)
    with fake_engine(FakeModel()) as model:
        extract(some_papers())
        shutil.rmtree(os.path.join(directory, "run", "checkpoints"))
        extract(some_papers())
    assert (model.calls == 6), "test premise: with no checkpoint the model is called again"
    store = th.Project.get_theory_store()
    assert (results_per_paper(store) == {"paper-a": 1, "paper-b": 1, "paper-c": 1}), "the same records were added twice: " + str(results_per_paper(store))


def test_a_different_schema_id_is_a_different_result(directory):
    open_project(directory)
    with fake_engine(FakeModel()):
        extract(some_papers(), extraction_schema_id="schema-1")
        extract(some_papers(), extraction_schema_id="schema-2")
    store = th.Project.get_theory_store()
    assert (results_per_paper(store) == {"paper-a": 2, "paper-b": 2, "paper-c": 2}), "results for a second schema were merged into the first: " + str(results_per_paper(store))


#
#   Tests: the checkpoint is keyed to everything that changes a result
#

def test_a_changed_paper_text_is_extracted_again(directory):
    open_project(directory)
    with fake_engine(FakeModel()) as model:
        extract(some_papers())
        changed = some_papers()
        changed[0]["markdown"] = "# A\n\nVIP activation lowered gain."
        packed = extract(changed)
    assert (model.calls == 4), "the changed paper was served its old result (" + str(model.calls) + " calls)"
    finding = [result for result in packed["results"] if (result["paper_id"] == "paper-a")][0]["records"][0]["finding"]
    assert (finding == "VIP activation lowered gain."), "paper-a came back with the old text's record: " + str(finding)


def test_a_changed_schema_id_or_setting_is_extracted_again(directory):
    open_project(directory)
    with fake_engine(FakeModel()) as model:
        extract(some_papers())
        expected = 3
        for kwargs in ({"extraction_schema_id": "schema-2"}, {"temperature": 0.5}, {"max_tokens": 1000}, {"use_reflection": True}):
            extract(some_papers(), **kwargs)
            expected += 3
            assert (model.calls == expected), str(kwargs) + " reused a cached result (" + str(model.calls) + " calls, expected " + str(expected) + ")"


def test_discovery_is_keyed_to_its_inputs(directory):
    open_project(directory)
    info = {"theory_request_normalized": QUERY}
    with fake_engine(FakeModel()) as model:
        th.Pipeline.discover_variables(some_papers(), info, model_str="fake-model", show_progress=False)
        th.Pipeline.discover_variables(some_papers(), info, model_str="fake-model", show_progress=False)
        assert (model.calls == 3), "an identical rerun was not resumed"
        th.Pipeline.discover_variables(some_papers(), info, model_str="fake-model", additional_contextual_info="focus on mice", show_progress=False)
        assert (model.calls == 6), "additional_contextual_info reused a cached result"
        th.Pipeline.discover_variables(some_papers(), info, model_str="fake-model", use_reflection=True, show_progress=False)
        assert (model.calls == 9), "use_reflection reused a cached result"
        changed = some_papers()
        changed[1]["markdown"] = "# B\n\nSomething else entirely."
        th.Pipeline.discover_variables(changed, info, model_str="fake-model", show_progress=False)
        assert (model.calls == 10), "a changed paper text reused a cached result (" + str(model.calls) + " calls)"


def test_import_is_keyed_to_file_content_and_the_effective_model(directory):
    open_project(directory)
    corpus = os.path.join(directory, "corpus")
    os.makedirs(corpus)
    # Two names that differ only past the characters an id keeps, so they truncate alike
    stem = "Deming1992 - epidemic in the Gambia following control as endemic"
    second_name = stem + " II efficacy to tOPV.md"
    first_name = stem + " I descriptive findings.md"
    with open(os.path.join(corpus, second_name), "w") as f:
        f.write("# Paper II\n\nOriginal text.\n")

    def import_ids():
        papers = th.PaperImport.import_papers(corpus, read_metadata=False, show_progress=False)["papers"]
        return {os.path.basename(paper["source_file"]): paper for paper in papers}

    before = import_ids()

    # A file added beside it, whose name sorts first and truncates to the same id
    with open(os.path.join(corpus, first_name), "w") as f:
        f.write("# Paper I\n\nOther text.\n")
    after = import_ids()
    assert (after[second_name]["paper_id"] == before[second_name]["paper_id"]), "adding a file changed another file's id: " + before[second_name]["paper_id"] + " -> " + after[second_name]["paper_id"]
    assert (after[first_name]["paper_id"] != after[second_name]["paper_id"]), "two files share an id"
    assert (after[first_name]["title"] == "Paper I") and (after[second_name]["title"] == "Paper II"), "a file was served another file's import"

    # An edited file is read again, not served its old import
    with open(os.path.join(corpus, second_name), "w") as f:
        f.write("# Paper II, revised\n\nRevised text.\n")
    edited = import_ids()
    assert (edited[second_name]["title"] == "Paper II, revised"), "the edited file was served its old import: " + str(edited[second_name]["title"])

    # The checkpoint names the model actually used, not None
    path = th.Project.checkpoint_path("paper_import", th.PaperImport.import_signature(True, None, None))
    assert (path == th.Project.checkpoint_path("paper_import", th.PaperImport.import_signature(True, th.Config.DEFAULT_MODEL_EXTRACTION, None))), "a defaulted model and the same model named outright are keyed differently"
    assert (path != th.Project.checkpoint_path("paper_import", th.PaperImport.import_signature(True, "another-model", None))), "a different metadata model shares the checkpoint"


def test_the_type_sweep_is_keyed_to_its_evidence_and_settings(directory):
    open_project(directory)
    calls = []

    def fake_build_theory_with_types(theory_query, records, **kwargs):
        calls.append(1)
        return {"theory_response": {"theories_general": [], "theories_specific": []}, "total_cost": 0.0}

    real = th.Legacy.build_theory_with_types
    th.Legacy.build_theory_with_types = fake_build_theory_with_types
    try:
        def sweep(evidence, **kwargs):
            th.Pipeline.generate_theories_over_types([], QUERY, output_types=["ASSOCIATION"], inference_types=[], evidence=evidence, show_progress=False, **kwargs)

        sweep([{"finding": "x"}])
        sweep([{"finding": "x"}])
        assert (len(calls) == 1), "an identical sweep was not resumed"
        sweep([{"finding": "y"}])
        assert (len(calls) == 2), "different evidence of the same size reused a cached run"
        for kwargs in ({"temperature": 0.2}, {"max_tokens": 1000}, {"mission_statement": "mice only"}):
            before = len(calls)
            sweep([{"finding": "x"}], **kwargs)
            assert (len(calls) == before + 1), str(kwargs) + " reused a cached run"
    finally:
        th.Legacy.build_theory_with_types = real


def main():
    print("TestResume.py")
    tests = [
        test_two_runs_leave_one_result_per_paper,
        test_a_new_session_leaves_one_result_per_paper,
        test_the_same_answer_from_the_model_is_not_added_twice,
        test_a_different_schema_id_is_a_different_result,
        test_a_changed_paper_text_is_extracted_again,
        test_a_changed_schema_id_or_setting_is_extracted_again,
        test_discovery_is_keyed_to_its_inputs,
        test_import_is_keyed_to_file_content_and_the_effective_model,
        test_the_type_sweep_is_keyed_to_its_evidence_and_settings,
    ]
    only = sys.argv[1:]
    run = 0
    for test in tests:
        if (len(only) > 0) and (not any(name in test.__name__ for name in only)):
            continue
        directory = tempfile.mkdtemp()
        try:
            test(directory)
            print("  " + test.__name__ + ": OK")
            run += 1
        finally:
            th.Project.close()
            shutil.rmtree(directory, ignore_errors=True)

    print("TestResume.py: " + str(run) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
