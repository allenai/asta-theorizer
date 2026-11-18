# TestNoveltyStore.py
# A qualified-novelty result for a theory or statement the store does not hold is skipped with a
# warning, and the store is left intact -- in memory, and on disk.
#
# Runs offline.  No API keys, no LLM calls, no network.
#
#     python tests/TestNoveltyStore.py

import copy
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th          # noqa: F401  (puts the engine on the path)

import EvaluationQualifiedNovelty as novelty


def a_store():
    return {
        "theories": {
            "theory-1": {"id": "theory-1", "components": {"theory_statements": [{"law": {"if": "a", "then": "b"}}]}},
        },
        "extraction_schemas": {},
        "extraction_results": {},
    }


VERDICT = {"novelty": "somewhat novel"}
EVIDENCE = {"dimension": ["evidence"]}


def test_a_good_result_is_added(directory):
    store = novelty.add_qualified_novelty_result_to_theorystore_dict(a_store(), "theory-1", 0, VERDICT, EVIDENCE)
    assert isinstance(store, dict), "the store came back as " + repr(store)
    law = store["theories"]["theory-1"]["components"]["theory_statements"][0]["law"]
    assert (law["qualified_novelty_evaluation"]["final_aggregated_novelty_evaluation"] == VERDICT), "the verdict was not added"
    assert ("qualified-novelty-theory-1-statement-0" in store["qualified_novelty_evaluations"]), "the evidence was not added"


def test_an_unknown_theory_leaves_the_store_intact(directory):
    original = a_store()
    store = novelty.add_qualified_novelty_result_to_theorystore_dict(copy.deepcopy(original), "theory-99", 0, VERDICT, EVIDENCE)
    assert isinstance(store, dict), "an unknown theory id returned " + repr(store) + " instead of the store"
    assert (store["theories"] == original["theories"]), "the store's theories changed"


def test_a_bad_statement_index_leaves_the_store_intact(directory):
    original = a_store()
    for index in (5, -1):
        store = novelty.add_qualified_novelty_result_to_theorystore_dict(copy.deepcopy(original), "theory-1", index, VERDICT, EVIDENCE)
        assert isinstance(store, dict), "statement index " + str(index) + " returned " + repr(store) + " instead of the store"
        assert (store["theories"] == original["theories"]), "the store's theories changed"


def test_the_file_on_disk_is_not_overwritten(directory):
    path = os.path.join(directory, "theorystore.json")
    with open(path, "w") as f:
        json.dump(a_store(), f)

    novelty.add_qualified_novelty_result_to_theorystore(path, "theory-99", 0, VERDICT, EVIDENCE)

    with open(path) as f:
        saved = json.load(f)
    assert isinstance(saved, dict) and ("theory-1" in saved.get("theories", {})), "the store on disk was overwritten with " + repr(saved)[:80]


def main():
    print("TestNoveltyStore.py")
    tests = [
        test_a_good_result_is_added,
        test_an_unknown_theory_leaves_the_store_intact,
        test_a_bad_statement_index_leaves_the_store_intact,
        test_the_file_on_disk_is_not_overwritten,
    ]
    for test in tests:
        directory = tempfile.mkdtemp()
        try:
            test(directory)
            print("  " + test.__name__ + ": OK")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    print("TestNoveltyStore.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
