# TestNoveltyExport.py
# Qualified-novelty results reach the HTML export.  Evaluate.novelty() writes each evaluated law's
# full result (verdict and per-paper evaluations) into the project, and Export.export_html() picks up
# the newest run: the law's theory page shows its verdict and links to a page with the evidence.
#
# Runs offline.  The engine's evaluation is replaced by a fake that returns a fixed result; no API
# keys, no LLM calls, no cost.
#
#     python tests/TestNoveltyExport.py

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th

import EvaluationQualifiedNovelty


LAW_1 = "VIP activity enables tone-evoked plasticity."
LAW_2 = "SST inhibition lowers it."

VERDICT = {
    "phenomenon_effect": {"degree_of_novelty": "explicit_established", "what_is_known": "Known.", "what_introduced": "-", "what_novel": "-"},
    "explanatory_mechanistic": {"degree_of_novelty": "derivable_unstated", "what_is_known": "-", "what_introduced": "A mechanism.", "what_novel": "The link."},
    "empirical_synthesis_meta_regulariry": None,
}


def a_project(directory):
    th.Config.init(project_dir=directory, verbose=False)
    store = th.Project.get_theory_store()
    store.add_extraction_schema(th.Legacy.ExtractionQuerySchema(schema=[{"name": "finding", "type": "str", "description": "x"}],
                                extraction_query="q", supporting_theory_ids=[], id="schema-1", model_str="written"))
    result_id = store.add_extraction_result(th.Legacy.ExtractionQueryResult(paper_id="paper-a", extraction_schema_id="schema-1",
                                            extracted_data=[{"name_short": "F", "name_full": "F", "brief_description": "f", "finding": "x"}],
                                            potentially_relevant_new_papers=[], model_str="written"))
    uuid = store.extraction_results[result_id].extracted_data[0]["uuid"]
    th.Project.save_theory_store()
    laws = [{"law": {"statement_name": "Plasticity", "theory_statement": LAW_1, "supporting_evidence": [{"text": "t", "uuids": [uuid]}]}},
            {"law": {"statement_name": "Inhibition", "theory_statement": LAW_2, "supporting_evidence": [{"text": "t", "uuids": [uuid]}]}}]
    stored = th.Pipeline.to_theorystore([{"theory_name": "VIP disinhibition", "theory_description": "d", "theory_statements": laws}],
                                        "How do VIP interneurons control plasticity?", model_str="written", method="written")
    return stored, next(iter(stored["theories"]))


class FakeNovelty:
    # Stands in for the engine's evaluation: one law evaluated, against one paper
    def __init__(self, theory_id):
        self.theory_id = theory_id
        self.paper_ids = None

    def __call__(self, theorystore_data, paperstore, model_str_extraction, model_str_decision, num_workers=5, DEBUG_LIMIT=0, paper_ids=None):
        self.paper_ids = paper_ids
        statement = {"theory_id": self.theory_id, "statement_index": 0, "theory_query": "q", "num_papers_evaluated": 1,
                     "theory_statement_law": {"theory_statement": LAW_1, "domain_scope": "mouse A1"},
                     "final_aggregated_novelty_evaluation": VERDICT, "evidence_for_dimension": {},
                     "all_novelty_evaluations": [{"paper_id": "paper-a", "novelty_evaluation": {"best_response": {"novelty_evaluation": {
                         "paper_title": "A paper about VIP cells", "phenomenon_effect": {"degree_of_novelty": "explicit_established",
                                                                                           "what_is_known": "Reported here."}}}}}],
                     "total_cost": 0.5}
        theorystore_data = EvaluationQualifiedNovelty.add_qualified_novelty_result_to_theorystore_dict(
            theorystore_data, self.theory_id, 0, VERDICT, {})
        return {"total_cost": 0.5, "theorystore_data": theorystore_data, "statement_results": [statement]}


def run_novelty(directory):
    stored, theory_id = a_project(directory)
    saved = EvaluationQualifiedNovelty.do_qualified_novelty_evaluation_persistence
    EvaluationQualifiedNovelty.do_qualified_novelty_evaluation_persistence = FakeNovelty(theory_id)
    try:
        packed = th.Evaluate.novelty(stored, paperstore=th.Project.get_paper_store())
    finally:
        EvaluationQualifiedNovelty.do_qualified_novelty_evaluation_persistence = saved
    return packed, theory_id


def test_novelty_writes_each_law_into_the_project(directory):
    packed, theory_id = run_novelty(directory)
    assert (packed["novelty_dir"] is not None) and (packed["novelty_dir"].startswith(directory)), packed["novelty_dir"]
    statements = os.path.join(packed["novelty_dir"], "statements")
    files = os.listdir(statements)
    assert (files == ["qn-" + theory_id + "-statement-0.json"]), files
    with open(os.path.join(statements, files[0])) as f:
        data = json.load(f)
    assert (data["all_novelty_evaluations"]) and (data["final_aggregated_novelty_evaluation"] == VERDICT), data
    with open(os.path.join(packed["novelty_dir"], "theorystore.json")) as f:
        evaluated = json.load(f)
    law = evaluated["theories"][theory_id]["components"]["theory_statements"][0]["law"]
    assert ("qualified_novelty_evaluation" in law), "the evaluated store should carry the verdict on the law"


def test_the_export_shows_the_verdict_and_links_the_evidence(directory):
    packed, theory_id = run_novelty(directory)
    index = th.Export.export_html(show_progress=False)
    out = os.path.dirname(index)

    with open(os.path.join(out, "theories", theory_id + ".html")) as f:
        page = f.read()
    assert ("derivable_unstated x1, explicit_established x1, no evidence x1" in page), "the theory page should summarize the verdict"
    assert ("qualifiednovelty/qn-" in page), "the theory page should link the novelty page"
    assert ("Not evaluated." in page), "the law that was not evaluated should say so"
    assert ("randomly selected subset of 100 laws" not in page), "the old paper-specific note should be gone"

    pages = os.listdir(os.path.join(out, "qualifiednovelty"))
    assert (len(pages) == 1), pages
    with open(os.path.join(out, "qualifiednovelty", pages[0])) as f:
        novelty_page = f.read()
    assert ("A paper about VIP cells" in novelty_page) and ("Reported here." in novelty_page), "the per-paper evidence should be on the novelty page"


def test_a_project_without_novelty_exports_as_before(directory):
    a_project(directory)
    index = th.Export.export_html(show_progress=False)
    assert os.path.exists(index)
    assert (not os.path.exists(os.path.join(os.path.dirname(index), "qualifiednovelty"))) or (len(os.listdir(os.path.join(os.path.dirname(index), "qualifiednovelty"))) == 0)


def test_novelty_compares_against_every_paper_in_the_paperstore(directory):
    # Every paper with full text, not only the ones the laws cite (paper-b is cited by nothing)
    stored, theory_id = a_project(directory)
    paper_store = th.Project.get_paper_store()
    paper_store.merge_from_dict({"paperstore": {"paper-a": {"title": "A", "paper_markdown": "# A"},
                                                "paper-b": {"title": "B", "paper_markdown": "# B"},
                                                "paper-c": {"title": "C, metadata only", "paper_markdown": None}}})
    fake = FakeNovelty(theory_id)
    saved = EvaluationQualifiedNovelty.do_qualified_novelty_evaluation_persistence
    EvaluationQualifiedNovelty.do_qualified_novelty_evaluation_persistence = fake
    try:
        th.Evaluate.novelty(stored, paperstore=paper_store)
    finally:
        EvaluationQualifiedNovelty.do_qualified_novelty_evaluation_persistence = saved
    assert (sorted(fake.paper_ids or []) == ["paper-a", "paper-b"]), fake.paper_ids


def test_an_explicit_novelty_directory_is_used(directory):
    packed, theory_id = run_novelty(directory)
    statements = os.path.join(packed["novelty_dir"], "statements")
    moved = os.path.join(directory, "elsewhere")
    shutil.move(statements, moved)
    index = th.Export.export_html(novelty=moved, show_progress=False)
    with open(os.path.join(os.path.dirname(index), "theories", theory_id + ".html")) as f:
        assert ("qualifiednovelty/qn-" in f.read())


def main():
    print("TestNoveltyExport.py")
    tests = [
        test_novelty_writes_each_law_into_the_project,
        test_the_export_shows_the_verdict_and_links_the_evidence,
        test_a_project_without_novelty_exports_as_before,
        test_an_explicit_novelty_directory_is_used,
        test_novelty_compares_against_every_paper_in_the_paperstore,
    ]
    for test in tests:
        directory = tempfile.mkdtemp()
        try:
            test(directory)
            print("  " + test.__name__ + ": OK")
        finally:
            th.Project.close()
            shutil.rmtree(directory, ignore_errors=True)
    print("TestNoveltyExport.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
