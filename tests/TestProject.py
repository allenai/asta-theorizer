# TestProject.py
# Tests for theorizer/Project.py -- saving stage output, numbering, and getting it back.
#
# Runs offline.  No API keys, no LLM calls, no cost.
#
#     python tests/TestProject.py

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


#
#   Tests
#

def test_save_and_load(directory):
    # The basic promise: what a stage returned is what comes back.
    th.Project.use(directory)

    packed = {"output": {"suggestions": [1, 2, 3]}, "model": "gpt-5.2", "total_cost": 0.5}
    path = th.Project.save("variable_report", packed)
    assert (path is not None), "save() returned nothing"
    assert (os.path.isfile(path)), "save() did not write a file"

    reloaded = th.Project.load("variable_report")
    assert (reloaded == packed), "what came back is not what went in"
    print("  test_save_and_load: OK")


def test_nothing_is_overwritten(directory):
    # Extracting three times must leave three files.  A pilot run silently destroying the full run
    # would be the worst possible failure here.
    th.Project.use(directory)

    th.Project.save("extraction", {"results": [1]})
    th.Project.save("extraction", {"results": [1, 2]})
    th.Project.save("extraction", {"results": [1, 2, 3]})

    items = th.Project.contents()
    assert (len(items) == 3), "expected 3 files, found " + str(len(items))

    indices = []
    for item in items:
        indices.append(item["index"])
    assert (indices == [1, 2, 3]), "files were not numbered in order: " + str(indices)
    print("  test_nothing_is_overwritten: OK")


def test_load_by_index_and_by_name(directory):
    # Both ways of asking for something must give the same thing back.
    th.Project.use(directory)

    th.Project.save("plan", {"output": {"short_name": "first"}})
    th.Project.save("records", [{"a": 1}, {"a": 2}])

    by_name = th.Project.load("records")
    by_index = th.Project.load(2)
    assert (by_name == by_index), "load-by-name and load-by-index disagree"
    assert (len(by_name) == 2), "a saved list did not come back as a list of the same length"
    print("  test_load_by_index_and_by_name: OK")


def test_missing_and_ambiguous_names_raise(directory):
    # A wrong name must say so, and must list what is actually there.  Guessing would be worse.
    th.Project.use(directory)
    th.Project.save("extraction-v1", {"results": []})
    th.Project.save("extraction-v2", {"results": []})

    try:
        th.Project.load("no-such-thing")
        raise AssertionError("loading a name that does not exist did not raise")
    except FileNotFoundError as e:
        assert ("extraction-v1" in str(e)), "the error does not say what IS available: " + str(e)

    try:
        th.Project.load("extraction")
        raise AssertionError("an ambiguous name did not raise")
    except ValueError as e:
        assert ("more than one" in str(e)), "unexpected error for an ambiguous name: " + str(e)

    print("  test_missing_and_ambiguous_names_raise: OK")


def test_listing_reads_the_directory(directory):
    # contents() must reflect the directory, not a remembered index -- so that deleting a file with
    # rm, or copying one in, is immediately visible.
    th.Project.use(directory)
    th.Project.save("one", {"total_cost": 1.0})
    th.Project.save("two", {"total_cost": 2.0})

    items = th.Project.contents()
    assert (len(items) == 2), "expected 2 items"

    os.remove(items[0]["path"])
    after = th.Project.contents()
    assert (len(after) == 1), "deleting a file by hand was not noticed"
    assert (after[0]["name"] == "two"), "the wrong file survived"

    # The next save must not reuse the number of the file that was deleted
    th.Project.save("three", {"total_cost": 3.0})
    indices = []
    for item in th.Project.contents():
        indices.append(item["index"])
    assert (3 in indices), "numbering restarted after a deletion, which would overwrite history"
    print("  test_listing_reads_the_directory: OK")


def test_checkpoint_path_depends_on_the_work(directory):
    # Two different schemas must not share a checkpoint, or a re-run after a schema change would
    # silently reuse results extracted with the old one.
    th.Project.use(directory)

    first = th.Project.checkpoint_path("extraction", ["query", "SCHEMA A", "gpt-5-mini"])
    same = th.Project.checkpoint_path("extraction", ["query", "SCHEMA A", "gpt-5-mini"])
    different = th.Project.checkpoint_path("extraction", ["query", "SCHEMA B", "gpt-5-mini"])

    assert (first == same), "the same work produced two different checkpoint files"
    assert (first != different), "a changed schema reused the old checkpoint file"
    print("  test_checkpoint_path_depends_on_the_work: OK")


def test_saving_off(directory):
    # With no project open, the stages must still work -- they just stop writing files.
    th.Project.close()
    assert (th.Project.is_open() is False), "close() did not close the project"
    assert (th.Project.save("anything", {"a": 1}) is None), "save() wrote a file with no project open"
    assert (th.Project.contents() == []), "contents() returned something with no project open"
    print("  test_saving_off: OK")


#
#   Runner
#

def main():
    print("TestProject.py")
    tests = [
        test_save_and_load,
        test_nothing_is_overwritten,
        test_load_by_index_and_by_name,
        test_missing_and_ambiguous_names_raise,
        test_listing_reads_the_directory,
        test_checkpoint_path_depends_on_the_work,
        test_saving_off,
    ]

    for test in tests:
        # Each test gets an empty directory of its own, so they cannot affect each other
        directory = tempfile.mkdtemp()
        try:
            test(directory)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    print("TestProject.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
