# TestPipeline.py
# Tests for the two ways an extraction schema gets improved, which are easy to confuse and were
# confused in an earlier version of this library.
#
#   variable DISCOVERY  runs per paper, BEFORE any extraction, and proposes `proposed_slots`.
#                       Those go to make_schemas_from_variable_discovery(), which asks a model to
#                       design whole NEW schemas.
#   variable REPORT     runs over records, AFTER extraction, and produces `suggestions`, each with
#                       `suggested_slots_and_extraction_instructions`.  Those go to
#                       extend_schema_with_suggestions(), which APPENDS them to the schema you
#                       already have, with no LLM call.
#
# Feeding one route's output to the other's function produces plausible-looking nonsense rather than
# an error, which is why these are asserted here.  The fixtures below are the real shapes, copied
# from actual runs.
#
# Runs offline.  No API keys, no LLM calls, no cost.
#
#     python tests/TestPipeline.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


# The real shape returned by discover_variables() -- one entry per paper, proposals under
# result["proposed"]["proposed_slots"].
DISCOVERY = {
    "results": [
        {"paper_id": "paper-a", "paper_title": "Paper A", "proposed": {"proposed_slots": [
            {"slot_name": "interneuron_marker_expression", "slot_description": "...", "impact_rating": 7},
            {"slot_name": "circuit_regime", "slot_description": "...", "impact_rating": 6},
        ]}},
        {"paper_id": "paper-b", "paper_title": "Paper B", "proposed": {"proposed_slots": [
            {"slot_name": "gain_measure", "slot_description": "...", "impact_rating": 8},
        ]}},
        # A paper whose proposal did not come back in the expected shape -- these are dropped
        {"paper_id": "paper-c", "paper_title": "Paper C", "proposed": {"something_else": "..."}},
        {"paper_id": "paper-d", "paper_title": "Paper D", "proposed": None},
    ]
}

# The real shape returned by make_variable_report()["output"]["suggestions"]
SUGGESTIONS = [
    {"suggestion_short_name": "Define sign/target of VIP modulation",
     "suggested_slots_and_extraction_instructions": [
         {"slot_name": "vip_modulation_sign", "slot_description_and_extraction_instructions": "Extract whether VIP activation increases or decreases the response."},
         {"slot_name": "vip_modulation_target", "slot_description_and_extraction_instructions": "Extract which cell type is modulated."},
     ]},
    {"suggestion_short_name": "Formalize gain as a transform",
     "suggested_slots_and_extraction_instructions": [
         {"slot_name": "gain_definition", "slot_description_and_extraction_instructions": "Extract how gain was defined."},
     ]},
    {"suggestion_short_name": "Circuit regime labels",
     "suggested_slots_and_extraction_instructions": [
         {"slot_name": "circuit_regime", "slot_description_and_extraction_instructions": "Extract which interneuron class dominates."},
     ]},
    {"suggestion_short_name": "A fourth suggestion, beyond the default cap",
     "suggested_slots_and_extraction_instructions": [
         {"slot_name": "should_not_appear", "slot_description_and_extraction_instructions": "..."},
     ]},
]

SCHEMA = [
    {"name": "finding", "type": "str", "description": "The finding"},
    {"name": "cell_type", "type": "str", "description": "The cell type"},
]


#
#   Tests: the discovery route
#

def test_collect_proposed_slots_reads_the_real_shape():
    collected = th.Pipeline.collect_proposed_slots(DISCOVERY)

    assert (len(collected) == 2), "expected 2 papers with usable proposals, got " + str(len(collected))
    assert ("paper-a" in collected), "paper-a's proposals were lost"
    assert (len(collected["paper-a"]) == 2), "paper-a should have 2 proposed slots"
    assert ("paper-c" not in collected), "a paper with no proposed_slots was kept"
    assert ("paper-d" not in collected), "a paper with a null proposal was kept"
    print("  test_collect_proposed_slots_reads_the_real_shape: OK")


def test_collect_proposed_slots_passes_through_other_shapes():
    # Already in the {paper id: slots} form the src/ function wants
    ready = {"paper-a": [{"slot_name": "x"}]}
    assert (th.Pipeline.collect_proposed_slots(ready) == ready), "an already-correct dict was altered"

    assert (th.Pipeline.collect_proposed_slots([]) == {"proposals": []}), "a list was not wrapped"
    assert (th.Pipeline.collect_proposed_slots(None) == {}), "an unusable input did not give an empty dict"
    print("  test_collect_proposed_slots_passes_through_other_shapes: OK")


def test_schemas_from_discovery_refuses_empty_input():
    # Better to say so than to spend a reasoning call on nothing
    try:
        th.Pipeline.make_schemas_from_variable_discovery({"results": []}, {"theory_query": "x"})
        raise AssertionError("an empty discovery did not raise")
    except ValueError as e:
        assert ("proposed_slots" in str(e)), "the error does not say what was missing: " + str(e)
    print("  test_schemas_from_discovery_refuses_empty_input: OK")


#
#   Tests: the report route
#

def test_extend_schema_appends_the_suggested_slots():
    extended = th.Pipeline.extend_schema_with_suggestions(SCHEMA, SUGGESTIONS, max_suggestions=3, show_progress=False)

    names = []
    for slot in extended:
        names.append(slot["name"])

    # The original slots must survive, unchanged and in order -- that is the whole point of this
    # route rather than generating a new schema
    assert (names[0] == "finding"), "the original schema was reordered"
    assert (names[1] == "cell_type"), "the original schema was reordered"

    assert ("vip_modulation_sign" in names), "a suggested slot was not added"
    assert ("gain_definition" in names), "a suggested slot was not added"
    assert (len(extended) == 2 + 4), "expected 2 original + 4 added, got " + str(len(extended))
    print("  test_extend_schema_appends_the_suggested_slots: OK")


def test_extend_schema_respects_max_suggestions():
    # The paper's pipeline used 3; the fourth suggestion must not appear
    extended = th.Pipeline.extend_schema_with_suggestions(SCHEMA, SUGGESTIONS, max_suggestions=3, show_progress=False)
    names = []
    for slot in extended:
        names.append(slot["name"])
    assert ("should_not_appear" not in names), "max_suggestions was not honoured"

    everything = th.Pipeline.extend_schema_with_suggestions(SCHEMA, SUGGESTIONS, max_suggestions=None, show_progress=False)
    names_all = []
    for slot in everything:
        names_all.append(slot["name"])
    assert ("should_not_appear" in names_all), "max_suggestions=None still capped the suggestions"
    print("  test_extend_schema_respects_max_suggestions: OK")


def test_extend_schema_does_not_duplicate_existing_slots():
    schema = SCHEMA + [{"name": "circuit_regime", "type": "str", "description": "already here"}]
    extended = th.Pipeline.extend_schema_with_suggestions(schema, SUGGESTIONS, max_suggestions=3, show_progress=False)

    count = 0
    for slot in extended:
        if (slot["name"] == "circuit_regime"):
            count += 1
    assert (count == 1), "a slot that already existed was added again (" + str(count) + " copies)"

    # ...and the original description must win, not the suggestion's
    for slot in extended:
        if (slot["name"] == "circuit_regime"):
            assert (slot["description"] == "already here"), "an existing slot's description was overwritten"
    print("  test_extend_schema_does_not_duplicate_existing_slots: OK")


def test_extend_schema_does_not_mutate_the_original():
    original_length = len(SCHEMA)
    th.Pipeline.extend_schema_with_suggestions(SCHEMA, SUGGESTIONS, show_progress=False)
    assert (len(SCHEMA) == original_length), "the schema passed in was modified in place"
    print("  test_extend_schema_does_not_mutate_the_original: OK")


def test_extend_schema_survives_suggestions_with_nothing_in_them():
    # A report whose suggestions carry no slot instructions must not silently look like it worked
    empty = [{"suggestion_short_name": "no slots here"}]
    extended = th.Pipeline.extend_schema_with_suggestions(SCHEMA, empty, show_progress=False)
    assert (len(extended) == len(SCHEMA)), "slots appeared from a suggestion with none in it"

    assert (th.Pipeline.extend_schema_with_suggestions(SCHEMA, None, show_progress=False) is not None), "None suggestions raised"
    assert (len(th.Pipeline.extend_schema_with_suggestions(None, SUGGESTIONS, show_progress=False)) == 4), "an empty starting schema did not just take the suggestions"
    print("  test_extend_schema_survives_suggestions_with_nothing_in_them: OK")


def test_added_slots_have_the_shape_extraction_expects():
    # extract_from_papers() reads name/type/description from every slot
    extended = th.Pipeline.extend_schema_with_suggestions(SCHEMA, SUGGESTIONS, show_progress=False)
    for slot in extended:
        assert ("name" in slot), "a slot has no name"
        assert ("type" in slot), "a slot has no type"
        assert ("description" in slot), "a slot has no description"
        assert (isinstance(slot["description"], str)) and (len(slot["description"]) > 0), "a slot has an empty description"
    print("  test_added_slots_have_the_shape_extraction_expects: OK")


#
#   Runner
#

def test_a_theory_cites_the_extraction_results_its_statements_came_from():
    # Supporting evidence was hardcoded to [] once, and every theory rendered with none.
    class Result:
        def __init__(self, data):
            self.extracted_data = data

    class Store:
        extraction_results = {
            "extraction-result-3": Result([{"uuid": "e3.0"}, {"uuid": "e3.1"}]),
            "extraction-result-9": Result([{"uuid": "e9.0"}]),
        }

    theory = {"statements": [
        {"evidence": [{"text": "a", "uuids": ["e3.1", "e9.0"]}]},
        {"evidence": [{"text": "b", "uuids": ["e3.0", "unknown"]}]},
    ]}

    ids = th.Pipeline._supporting_evidence_ids(theory, Store())
    # Deduped and in the order the theory cites them; an unknown uuid is ignored, not fatal.
    assert ids == ["extraction-result-3", "extraction-result-9"], ids


def test_a_theory_that_cites_nothing_has_no_evidence():
    class Store:
        extraction_results = {}

    assert th.Pipeline._supporting_evidence_ids({"statements": []}, Store()) == []


def test_a_schema_an_extraction_result_names_is_always_in_the_store():
    # buildEvidence in the reader gives up when a result names a schema the store does not carry,
    # and the theory renders with no evidence at all.
    class Result:
        def __init__(self):
            self.extracted_data = [{"uuid": "e1.0"}]
            self.extraction_schema_id = "schema-1"
        def to_dict(self):
            return {"id": "extraction-result-1", "extraction_schema_id": "schema-1",
                    "extracted_data": self.extracted_data}

    class Store:
        extraction_results = {"extraction-result-1": Result()}
        extraction_schemas = {}
        def add_theory(self, theory):
            return "theory-1"

    store = Store()
    out = th.Pipeline.to_theorystore(
        [{"theory_name": "T", "theory_description": "d"}],
        "a question", extraction_schema=[{"name": "x"}], theory_store=store)

    assert "schema-1" in out["extraction_schemas"], out["extraction_schemas"]


def test_an_exported_store_carries_the_schema_without_being_handed_it():
    # extract_from_papers() puts its schema in the store, so to_theorystore() exports it even when
    # the caller does not pass extraction_schema=
    th.Config.init(project_dir=None, verbose=False)
    real = th.Legacy.extract_entities_from_paper
    th.Legacy.extract_entities_from_paper = lambda query, schema, text, **kwargs: {"response": {"results": [{"name_short": "x", "finding": "y"}]}, "total_cost": 0.0}
    try:
        store = th.Legacy.TheoryStore()
        schema = [{"name": "finding", "type": "str", "description": "d"}]
        th.Pipeline.extract_from_papers([{"paper_id": "paper-a", "markdown": "text"}], "q", schema, model_str="fake", theory_store=store,
                                        extraction_schema_id="schema-7", show_progress=False)
        out = th.Pipeline.to_theorystore([{"theory_name": "T", "theory_description": "d"}], "q", theory_store=store)
    finally:
        th.Legacy.extract_entities_from_paper = real
    assert ("schema-7" in out["extraction_schemas"]), "the exported store has no schema for its results: " + str(list(out["extraction_schemas"]))
    assert (out["extraction_schemas"]["schema-7"]["schema"] == schema), "the exported schema is not the one extracted with"
    print("  test_an_exported_store_carries_the_schema_without_being_handed_it: OK")


def test_fetch_papers_needs_only_a_corpus_id():
    class FakePaperStore:
        def __init__(self):
            self.submitted = []
            self.paperstore = {}
        def start_thread(self):
            pass
        def submit_paper(self, corpus_id=None, title=None):
            self.submitted.append((corpus_id, title))
            return "request-" + str(len(self.submitted))
        def is_paper_request_complete(self, request_id):
            return "key-" + request_id
        def get_paper_by_key(self, key):
            return {"title": "A title from Semantic Scholar", "corpus_id": "1"}
        def get_paper_text_by_key(self, key):
            return "# A paper\n\nfull text"

    th.Config.init(project_dir=None, verbose=False)
    paper_store = FakePaperStore()
    out = th.Pipeline.fetch_papers([{"corpus_id": "278481047"}, {"corpus_id": "278481047"}, {"title": "no id"}], paper_store=paper_store, show_progress=False, poll_seconds=0)
    assert (paper_store.submitted == [("278481047", None)]), "a candidate with only a corpus id was not fetched once: " + str(paper_store.submitted)
    assert (out["num_with_full_text"] == 1), "the fetched paper did not come back"
    print("  test_fetch_papers_needs_only_a_corpus_id: OK")


#
#   Tests: the type sweep refuses names it does not know
#

def sweep_with_a_fake_generator(**kwargs):
    # Run generate_theories_over_types() with the generator replaced, so nothing reaches a model.
    # Returns (the packed output or the error raised, the requests the generator was given).
    calls = []

    def fake_build_theory_with_types(theory_query, records, **build_kwargs):
        calls.append((build_kwargs.get("output_type"), build_kwargs.get("inference_type")))
        return {"theory_response": {"theories_general": [], "theories_specific": []}, "total_cost": 0.0}

    th.Config.init(project_dir=None, verbose=False)
    real = th.Legacy.build_theory_with_types
    th.Legacy.build_theory_with_types = fake_build_theory_with_types
    try:
        try:
            result = th.Pipeline.generate_theories_over_types([], "a question", evidence=[{"finding": "x"}], show_progress=False, **kwargs)
        except ValueError as e:
            result = e
    finally:
        th.Legacy.build_theory_with_types = real
    return result, calls


def test_sweep_refuses_an_unknown_output_type():
    # A test run spent $0.55 on "CONFLICTING EVIDENCE", which is not one of the output types.
    result, calls = sweep_with_a_fake_generator(output_types=["ASSOCIATION", "CONFLICTING EVIDENCE"], inference_types=[])
    assert isinstance(result, ValueError), "an unknown output type was not refused: " + str(result)
    assert ("CONFLICTING EVIDENCE" in str(result)), "the error does not name the bad type: " + str(result)
    assert ("MECHANISTIC EXPLANATION" in str(result)), "the error does not list the valid output types: " + str(result)
    assert (calls == []), "the generator was called before the bad name was refused: " + str(calls)
    print("  test_sweep_refuses_an_unknown_output_type: OK")


def test_sweep_refuses_an_unknown_inference_type():
    result, calls = sweep_with_a_fake_generator(output_types=[], inference_types=["GUESSING"])
    assert isinstance(result, ValueError), "an unknown inference type was not refused: " + str(result)
    assert ("GUESSING" in str(result)), "the error does not name the bad type: " + str(result)
    assert ("TRIANGULATION" in str(result)), "the error does not list the valid inference types: " + str(result)
    assert (calls == []), "the generator was called before the bad name was refused: " + str(calls)
    print("  test_sweep_refuses_an_unknown_inference_type: OK")


def test_sweep_still_runs_known_types_and_empty_means_none():
    result, calls = sweep_with_a_fake_generator(output_types=["ASSOCIATION"], inference_types=[])
    assert (not isinstance(result, ValueError)), "a known type was refused: " + str(result)
    assert (calls == [("ASSOCIATION", None)]), "expected one output-type run and no inference-type runs, got " + str(calls)
    print("  test_sweep_still_runs_known_types_and_empty_means_none: OK")


def main():
    print("TestPipeline.py")
    tests = [
        test_collect_proposed_slots_reads_the_real_shape,
        test_collect_proposed_slots_passes_through_other_shapes,
        test_schemas_from_discovery_refuses_empty_input,
        test_extend_schema_appends_the_suggested_slots,
        test_extend_schema_respects_max_suggestions,
        test_extend_schema_does_not_duplicate_existing_slots,
        test_extend_schema_does_not_mutate_the_original,
        test_extend_schema_survives_suggestions_with_nothing_in_them,
        test_added_slots_have_the_shape_extraction_expects,
        test_a_theory_cites_the_extraction_results_its_statements_came_from,
        test_a_theory_that_cites_nothing_has_no_evidence,
        test_a_schema_an_extraction_result_names_is_always_in_the_store,
        test_sweep_refuses_an_unknown_output_type,
        test_sweep_refuses_an_unknown_inference_type,
        test_sweep_still_runs_known_types_and_empty_means_none,
        test_an_exported_store_carries_the_schema_without_being_handed_it,
        test_fetch_papers_needs_only_a_corpus_id,
    ]

    for test in tests:
        test()

    print("TestPipeline.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
