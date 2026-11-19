# TestEvidenceChain.py
# Tests that the evidence chain survives: extracted records carry uuids, and a theory built on them
# can cite those uuids back to the paper the evidence came from.
#
# THIS IS THE TEST THAT WAS MISSING.  The library once returned records with no uuid at all, which
# meant every generated theory came back with empty evidence lists and no way to trace a law to a
# paper -- and the whole rest of the suite passed, because it only ever checked the plumbing around
# the evidence and never the evidence itself.
#
# Runs offline.  No API keys, no LLM calls, no cost.
#
#     python tests/TestEvidenceChain.py

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


def make_extraction_batch():
    # The shape extract_from_papers() produces: one entry per paper, each with its records
    return [
        {"paper_id": "paper-a", "paper_title": "Paper A", "cost": 0.01, "suggested_papers": [],
         "records": [{"name_short": "finding one", "plasticity_effect": "increase"},
                     {"name_short": "finding two", "plasticity_effect": "decrease"}]},
        {"paper_id": "paper-b", "paper_title": "Paper B", "cost": 0.01, "suggested_papers": [],
         "records": [{"name_short": "finding three", "plasticity_effect": "no change"}]},
    ]


def stamp(batch):
    # What extract_from_papers() does after the parallel pass: put each paper's records into an
    # ExtractionQueryResult and add it to a TheoryStore, which assigns the ids and the uuids.
    store = th.Legacy.TheoryStore()
    for result in batch:
        extraction_result = th.Legacy.ExtractionQueryResult(
            paper_id=result["paper_id"], extraction_schema_id="schema-1",
            extracted_data=result["records"], potentially_relevant_new_papers=[],
            cost=result["cost"], model_str="test-model")
        result["extraction_result_id"] = store.add_extraction_result(extraction_result)
    return store


#
#   Tests
#

def test_every_record_gets_a_uuid(directory):
    batch = make_extraction_batch()
    stamp(batch)

    missing = []
    for result in batch:
        for record in result["records"]:
            if (not record.get("uuid")):
                missing.append(record.get("name_short"))

    assert (len(missing) == 0), "records with no uuid: " + str(missing) + " -- theories built on these cannot cite evidence"
    print("  test_every_record_gets_a_uuid: OK")


def test_uuids_are_unique(directory):
    batch = make_extraction_batch()
    stamp(batch)

    uuids = []
    for result in batch:
        for record in result["records"]:
            uuids.append(record["uuid"])

    assert (len(uuids) == len(set(uuids))), "duplicate uuids: " + str(uuids) + " -- evidence would link to the wrong record"
    print("  test_uuids_are_unique: OK (" + ", ".join(uuids) + ")")


def test_a_uuid_traces_back_to_its_paper(directory):
    # The point of the chain: given a uuid cited by a law, find the paper it came from.
    batch = make_extraction_batch()
    store = stamp(batch)

    by_uuid = {}
    for result_id, extraction_result in store.extraction_results.items():
        for record in extraction_result.extracted_data:
            by_uuid[record["uuid"]] = extraction_result.paper_id

    a_uuid = batch[1]["records"][0]["uuid"]
    assert (a_uuid in by_uuid), "a record's uuid is not findable in the store"
    assert (by_uuid[a_uuid] == "paper-b"), "uuid traced to the wrong paper: " + str(by_uuid[a_uuid])
    print("  test_a_uuid_traces_back_to_its_paper: OK")


def test_flatten_reads_from_the_store_and_keeps_the_uuid(directory):
    # flatten_records() sits between extraction and theory generation. It must read the records back
    # out of the TheoryStore -- as the paper's pipeline did with
    # get_extraction_result(id).extracted_data -- because the store is what stamped the uuids.
    batch = make_extraction_batch()
    store = stamp(batch)
    records = th.Pipeline.flatten_records({"results": batch, "theory_store": store})

    assert (len(records) == 3), "expected 3 flattened records, got " + str(len(records))
    for record in records:
        assert (record.get("uuid")), "flatten_records() dropped the uuid from " + str(record.get("name_short"))

    # Provenance is the uuid resolving through the store, NOT an extra key injected onto the record.
    # The paper's pipeline never put paper_id or paper_title on a record, so neither may this.
    for record in records:
        assert ("paper_id" not in record), "flatten_records() injected paper_id onto a record"
        assert ("paper_title" not in record), "flatten_records() injected paper_title onto a record"

    by_uuid = {}
    for extraction_result in store.extraction_results.values():
        for stored in extraction_result.extracted_data:
            by_uuid[stored["uuid"]] = extraction_result.paper_id
    for record in records:
        assert (record["uuid"] in by_uuid), "a flattened record's uuid does not resolve in the store"
    print("  test_flatten_reads_from_the_store_and_keeps_the_uuid: OK")


def test_theorystore_export_carries_the_evidence(directory):
    # to_theorystore() must include the extraction results, or the uuids a law cites have nothing to
    # point at and the exported pages show evidence with no links.
    batch = make_extraction_batch()
    store = stamp(batch)

    theory = {"theory_name": "A theory", "theory_description": "...",
              "theory_statements": [{"law": {"statement_name": "A law",
                                             "supporting_evidence": [{"text": "because of this",
                                                                      "uuids": [batch[0]["records"][0]["uuid"]]}]}}],
              "requested_inference_type": "TRIANGULATION", "requested_output_type": None}

    exported = th.Pipeline.to_theorystore([theory], "a query", model_str="test-model",
                                          extraction_schema=[{"name": "plasticity_effect"}], theory_store=store)

    assert (len(exported["extraction_results"]) == 2), "the extraction results did not make it into the store"

    # every uuid cited by the theory must resolve to a record in the store
    known = set()
    for result in exported["extraction_results"].values():
        for record in result.get("extracted_data") or []:
            known.add(record.get("uuid"))

    cited = []
    the_theory = list(exported["theories"].values())[0]
    for statement in the_theory["components"]["theory_statements"]:
        law = statement.get("law") or statement
        for evidence in law.get("supporting_evidence") or []:
            cited.extend(evidence.get("uuids") or [])

    assert (len(cited) > 0), "the theory cited no evidence at all"
    unresolved = [u for u in cited if u not in known]
    assert (len(unresolved) == 0), "cited uuids that resolve to nothing: " + str(unresolved)
    print("  test_theorystore_export_carries_the_evidence: OK")


def test_records_without_uuids_are_detectable(directory):
    # The failure that went unnoticed: records that never got stamped. A caller must be able to see
    # it rather than discovering it in an exported page weeks later.
    batch = make_extraction_batch()          # deliberately NOT stamped
    records = th.Pipeline.flatten_records({"results": batch})

    without = [r for r in records if not r.get("uuid")]
    assert (len(without) == len(records)), "unstamped records unexpectedly had uuids"
    print("  test_records_without_uuids_are_detectable: OK")



def test_extract_from_papers_itself_stamps_uuids(directory):
    # The tests above exercise the stamping mechanism. This one checks that extract_from_papers()
    # ACTUALLY CALLS IT -- which is the thing that was broken. The per-paper LLM call is stubbed, so
    # this stays offline and free.
    #
    # Without this test, extract_from_papers() could stop stamping and everything above would still
    # pass, which is precisely how the original bug survived a green suite.
    original = th.Legacy.extract_entities_from_paper
    th.Project.close()

    def fake_extract(extraction_query, extraction_schema, paper_text, **kwargs):
        return {"response": {"results": [{"name_short": "a finding", "plasticity_effect": "increase"},
                                         {"name_short": "another finding", "plasticity_effect": "decrease"}],
                             "follow_on_papers": []},
                "total_cost": 0.0}

    th.Legacy.extract_entities_from_paper = fake_extract
    try:
        papers = [{"paper_id": "paper-x", "title": "Paper X", "markdown": "text"},
                  {"paper_id": "paper-y", "title": "Paper Y", "markdown": "text"}]
        # A store of its own, so this asserts on this call's results rather than on whatever the
        # session store has accumulated from earlier tests in the same process.
        packed = th.Pipeline.extract_from_papers(papers, "a query", [{"name": "plasticity_effect"}],
                                                 max_workers=2, show_progress=False,
                                                 theory_store=th.Legacy.TheoryStore())
    finally:
        th.Legacy.extract_entities_from_paper = original

    records = th.Pipeline.flatten_records(packed)
    assert (len(records) == 4), "expected 4 records, got " + str(len(records))

    missing = [r.get("name_short") for r in records if not r.get("uuid")]
    assert (len(missing) == 0), "extract_from_papers() did not stamp uuids on: " + str(missing)

    assert (packed.get("num_records_with_uuid") == 4), "the uuid count was not reported correctly"
    assert (packed.get("theory_store") is not None), "extract_from_papers() returned no TheoryStore"
    assert (len(packed["theory_store"].extraction_results) == 2), "expected one extraction result per paper"
    print("  test_extract_from_papers_itself_stamps_uuids: OK")



def test_theory_ids_are_unique_within_a_store(directory):
    # Ids must come from TheoryStore.add_theory() -> generate_id(), not from a scheme invented in
    # the library. An earlier version numbered them per call, so two sweeps both produced
    # "theory-0001" and merging their stores silently dropped one of the two theories.
    store = th.Legacy.TheoryStore()
    ids = []
    for batch in range(5):
        out = th.Pipeline.to_theorystore(
            [{"theory_name": "T%d-%d" % (batch, i), "theory_statements": []} for i in range(4)],
            "a query", theory_store=store)
        ids.extend(out["theories"].keys())

    assert (len(ids) == 20), "expected 20 theories, got " + str(len(ids))
    assert (len(set(ids)) == 20), "theory ids collided across runs: " + str(sorted(ids))
    assert (len(store.theories) == 20), "the store lost theories: " + str(len(store.theories))
    print("  test_theory_ids_are_unique_within_a_store: OK (" + str(len(ids)) + " distinct)")


def test_evidence_and_theory_ids_share_one_namespace_without_clashing(directory):
    store = th.Legacy.TheoryStore()
    for i in range(5):
        result = th.Legacy.ExtractionQueryResult(paper_id="p%d" % i, extraction_schema_id="schema-1",
                                                 extracted_data=[{"n": "a"}, {"n": "b"}],
                                                 potentially_relevant_new_papers=[], cost=0.0)
        store.add_extraction_result(result)
    th.Pipeline.to_theorystore([{"theory_name": "T", "theory_statements": []}], "q", theory_store=store)

    assert (len(store.all_ids) == len(set(store.all_ids))), "the store's id set contains duplicates"
    assert (len(store.extraction_results) == 5), "extraction results were overwritten"
    assert (len(store.theories) == 1), "theories were overwritten"
    print("  test_evidence_and_theory_ids_share_one_namespace_without_clashing: OK")


def test_the_generator_name_is_saved_with_the_theory(directory):
    # The workspace pages and the exporter read components["method"] from the saved store
    store = th.Legacy.TheoryStore()
    th.Pipeline.to_theorystore([{"theory_name": "T", "theory_statements": []}], "q", method="written", theory_store=store)
    th.Pipeline.to_theorystore([{"theory_name": "S", "theory_statements": [], "requested_output_type": "NULL RESULT"}], "q", theory_store=store)
    methods = sorted(theory.components.get("method") for theory in store.theories.values())
    assert (methods == ["NULL RESULT", "written"]), "methods not saved: " + str(methods)
    print("  test_the_generator_name_is_saved_with_the_theory: OK")


def test_nothing_is_overwritten_by_repeated_adds(directory):
    # Adding to the same store repeatedly must accumulate, never replace.
    store = th.Legacy.TheoryStore()
    for i in range(10):
        result = th.Legacy.ExtractionQueryResult(paper_id="p%d" % i, extraction_schema_id="schema-1",
                                                 extracted_data=[{"n": str(i)}],
                                                 potentially_relevant_new_papers=[], cost=0.0)
        store.add_extraction_result(result)

    assert (len(store.extraction_results) == 10), "extraction results were overwritten: " + str(len(store.extraction_results))
    uuids = [d["uuid"] for r in store.extraction_results.values() for d in r.extracted_data]
    assert (len(uuids) == len(set(uuids)) == 10), "uuids collided: " + str(uuids)
    print("  test_nothing_is_overwritten_by_repeated_adds: OK")


#
#   Runner
#

def main():
    print("TestEvidenceChain.py")
    tests = [
        test_every_record_gets_a_uuid,
        test_uuids_are_unique,
        test_a_uuid_traces_back_to_its_paper,
        test_flatten_reads_from_the_store_and_keeps_the_uuid,
        test_theorystore_export_carries_the_evidence,
        test_records_without_uuids_are_detectable,
        test_extract_from_papers_itself_stamps_uuids,
        test_theory_ids_are_unique_within_a_store,
        test_evidence_and_theory_ids_share_one_namespace_without_clashing,
        test_nothing_is_overwritten_by_repeated_adds,
        test_the_generator_name_is_saved_with_the_theory,
    ]
    for test in tests:
        directory = tempfile.mkdtemp()
        try:
            test(directory)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    print("TestEvidenceChain.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
