# RunAllExamples.py
# Run every example end to end, small and cheap, to check they all still work.
#
# Unlike tests/RunAllTests.py this DOES cost money -- every example makes real LLM calls.  It is
# kept deliberately small: a handful of papers and whatever model you name.
#
#     python examples/RunAllExamples.py /path/to/papers gpt-6-luna
#     python examples/RunAllExamples.py /path/to/papers gpt-6-luna --skip 05

import os
import subprocess
import sys
import time

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXAMPLES_DIR)


# Each entry: script, the arguments it takes, and roughly what it costs to run at this size.
EXAMPLES = [
    {"script": "01_FirstRun.py", "args": "papers+model", "note": "full flow: plan -> papers -> extract -> theories"},
    {"script": "02_BringYourOwnPapers.py", "args": "papers+model", "note": "importing local files, metadata, Semantic Scholar"},
    {"script": "03_IteratingOnASchema.py", "args": "papers+model", "note": "pilot -> variable report -> new schemas -> extract again"},
    {"script": "06_VariableDiscovery.py", "args": "papers+model", "note": "per-paper variable discovery -> schemas"},
    {"script": "05_SymbolicLaws.py", "args": "papers+model", "note": "symbolic schema -> typed table -> solver"},
    {"script": "04_PickingUpLater.py", "args": "project", "note": "reopening example-01-run and continuing"},
    {"script": "07_TheoriesAndModels.py", "args": "project+model", "note": "accuracy vs novelty, models per stage"},
]


def build_command(example, papers_path, model):
    script = os.path.join(EXAMPLES_DIR, example["script"])
    if (example["args"] == "papers"):
        return [sys.executable, script, papers_path]
    if (example["args"] == "papers+model"):
        return [sys.executable, script, papers_path, model]
    if (example["args"] == "project"):
        return [sys.executable, script, "example-01-run"]
    if (example["args"] == "project+model"):
        return [sys.executable, script, "example-01-run", model]
    return [sys.executable, script]


def main():
    if (len(sys.argv) < 2):
        print("Usage: python examples/RunAllExamples.py /path/to/papers [model] [--skip 05,07]")
        return 1

    papers_path = sys.argv[1]
    model = sys.argv[2] if (len(sys.argv) > 2) and (not sys.argv[2].startswith("--")) else "gpt-6-luna"

    skip = []
    if ("--skip" in sys.argv):
        skip = sys.argv[sys.argv.index("--skip") + 1].split(",")

    print("RunAllExamples.py: papers=" + str(papers_path) + ", model=" + str(model))
    print("")

    results = []
    for example in EXAMPLES:
        number = example["script"].split("_")[0]
        if (number in skip):
            print("  SKIPPED  " + example["script"])
            results.append({"script": example["script"], "status": "skipped", "seconds": 0})
            continue

        print("=" * 78)
        print("RUNNING  " + example["script"] + "   -- " + example["note"])
        print("=" * 78, flush=True)

        started = time.time()
        # Run from the repo root so the example project directories land in one predictable place
        outcome = subprocess.run(build_command(example, papers_path, model), cwd=REPO_ROOT)
        elapsed = round(time.time() - started, 1)

        results.append({
            "script": example["script"],
            "status": "ok" if (outcome.returncode == 0) else "FAILED",
            "seconds": elapsed,
        })
        print("")

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    num_failed = 0
    for result in results:
        print("  " + str(result["status"]).ljust(9) + result["script"].ljust(32) + str(result["seconds"]) + "s")
        if (result["status"] == "FAILED"):
            num_failed += 1

    if (num_failed > 0):
        print("")
        print("RunAllExamples.py: " + str(num_failed) + " example(s) FAILED")
        return 1

    print("")
    print("RunAllExamples.py: all examples ran")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
