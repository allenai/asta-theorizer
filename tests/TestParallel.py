# TestParallel.py
# Tests for theorizer/Parallel.py -- running per-paper work concurrently, surviving failures, and
# resuming from a checkpoint after a crash.
#
# Runs offline.  No API keys, no LLM calls, no cost.
#
#     python tests/TestParallel.py

import os
import shutil
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


def make_papers(count:int):
    papers = []
    for i in range(count):
        papers.append({"paper_id": "paper-" + str(i).zfill(2), "title": "Paper " + str(i)})
    return papers


#
#   Tests
#

def test_results_come_back_in_order(directory):
    # Work finishes out of order under concurrency; the results must not.
    papers = make_papers(20)

    def do_one(paper):
        return {"paper_id": paper["paper_id"]}

    batch = th.Parallel.run_over_items(do_one, papers, max_workers=8, show_progress=False)
    assert (len(batch["results"]) == 20), "expected 20 results"

    returned = []
    for result in batch["results"]:
        returned.append(result["paper_id"])

    expected = []
    for paper in papers:
        expected.append(paper["paper_id"])

    assert (returned == expected), "results came back out of order"
    print("  test_results_come_back_in_order: OK")


def test_one_failure_does_not_stop_the_others(directory):
    # A corpus always has a few bad papers.  997 of 1000 is a success, not a failed run.
    papers = make_papers(10)

    def do_one(paper):
        if (paper["paper_id"] in ("paper-03", "paper-07")):
            raise RuntimeError("bad paper: " + paper["paper_id"])
        return {"paper_id": paper["paper_id"]}

    batch = th.Parallel.run_over_items(do_one, papers, max_workers=4, show_progress=False)
    assert (len(batch["results"]) == 8), "expected 8 successes, got " + str(len(batch["results"]))
    assert (len(batch["failures"]) == 2), "expected 2 failures, got " + str(len(batch["failures"]))

    failed_ids = []
    for failure in batch["failures"]:
        failed_ids.append(failure["item"]["paper_id"])
    assert (sorted(failed_ids) == ["paper-03", "paper-07"]), "the wrong items were recorded as failed"
    assert ("bad paper" in batch["failures"][0]["error"]), "the failure did not record the error"
    print("  test_one_failure_does_not_stop_the_others: OK")


def test_checkpoint_is_written_during_the_run(directory):
    # The point of the checkpoint is that it exists BEFORE the run finishes.  If it were only
    # written at the end it would be useless for the crash it is meant to survive.
    papers = make_papers(6)
    checkpoint = os.path.join(directory, "checkpoint.jsonl")

    seen_during_run = {"lines": 0}
    gate = threading.Event()

    def do_one(paper):
        if (paper["paper_id"] == "paper-05"):
            # Hold the last paper until we have looked at the file
            gate.wait(timeout=10)
        return {"paper_id": paper["paper_id"]}

    def look_at_the_file():
        # Wait until some results have landed, then count the lines while the run is still going
        for attempt in range(100):
            if (os.path.isfile(checkpoint)):
                with open(checkpoint) as f:
                    lines = len(f.readlines())
                if (lines >= 3):
                    seen_during_run["lines"] = lines
                    break
            threading.Event().wait(0.05)
        gate.set()

    watcher = threading.Thread(target=look_at_the_file)
    watcher.start()
    th.Parallel.run_over_items(do_one, papers, max_workers=2, show_progress=False,
                              checkpoint_path=checkpoint, item_key=lambda paper: paper["paper_id"])
    watcher.join()

    assert (seen_during_run["lines"] >= 3), "the checkpoint was empty while the run was still going"
    print("  test_checkpoint_is_written_during_the_run: OK")


def test_resume_skips_what_is_already_done(directory):
    # The crash-and-restart case: the second run must not redo (or re-pay for) finished work.
    papers = make_papers(10)
    checkpoint = os.path.join(directory, "checkpoint.jsonl")
    calls = {"count": 0}
    lock = threading.Lock()

    def do_one(paper):
        with lock:
            calls["count"] += 1
        if (paper["paper_id"] == "paper-06") and (calls["crash"] is True):
            raise RuntimeError("simulated crash")
        return {"paper_id": paper["paper_id"]}

    calls["crash"] = True
    first = th.Parallel.run_over_items(do_one, papers, max_workers=3, show_progress=False,
                                       checkpoint_path=checkpoint, item_key=lambda paper: paper["paper_id"])
    assert (len(first["results"]) == 9), "expected 9 to succeed on the first pass"
    calls_after_first = calls["count"]
    assert (calls_after_first == 10), "expected 10 calls on the first pass"

    # Second run: everything but the failed paper should be resumed, not recomputed
    calls["crash"] = False
    second = th.Parallel.run_over_items(do_one, papers, max_workers=3, show_progress=False,
                                        checkpoint_path=checkpoint, item_key=lambda paper: paper["paper_id"])
    assert (len(second["results"]) == 10), "expected all 10 on the second pass"
    assert (second["num_resumed"] == 9), "expected 9 resumed, got " + str(second["num_resumed"])
    assert (calls["count"] == calls_after_first + 1), "the second run recomputed work it should have resumed"
    print("  test_resume_skips_what_is_already_done: OK")


def test_a_truncated_checkpoint_does_not_break_the_load(directory):
    # A process killed mid-write leaves half a line.  That must cost one item, not the whole file.
    checkpoint = os.path.join(directory, "checkpoint.jsonl")
    with open(checkpoint, "w") as f:
        f.write('{"key": "paper-00", "result": {"paper_id": "paper-00"}}\n')
        f.write('{"key": "paper-01", "result": {"paper_id": "paper-01"}}\n')
        f.write('{"key": "paper-02", "resu')          # killed here

    done = th.Parallel.load_checkpoint(checkpoint)
    assert (len(done) == 2), "expected the 2 complete lines, got " + str(len(done))
    assert ("paper-00" in done), "a complete line was lost"
    print("  test_a_truncated_checkpoint_does_not_break_the_load: OK")


#
#   Runner
#

def main():
    print("TestParallel.py")
    tests = [
        test_results_come_back_in_order,
        test_one_failure_does_not_stop_the_others,
        test_checkpoint_is_written_during_the_run,
        test_resume_skips_what_is_already_done,
        test_a_truncated_checkpoint_does_not_break_the_load,
    ]

    for test in tests:
        directory = tempfile.mkdtemp()
        try:
            test(directory)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    print("TestParallel.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
