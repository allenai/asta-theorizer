# TestNoPaperSearch.py
# This build has no paper search: papers arrive from files, or by corpus id.  These check that the
# library offers no paper search, that nothing it imports can reach the old search client, and that
# the wheel ships only the engine modules the library imports.
#
# Runs offline.  No API keys, no LLM calls, no network.
#
#     python tests/TestNoPaperSearch.py

import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# What a call to the old search client looks like in code: the function, its module, its settings
SEARCH_CLIENT_CALLS = re.compile(r"get_paperfinder_results|PaperFinderRequests|PAPER_FINDER_BASE_URL|THEORIZER_PAPERFINDER_URL")


def shipped_engine_modules():
    # The engine files pyproject.toml copies into the wheel, by source path
    with open(os.path.join(REPO_ROOT, "pyproject.toml")) as f:
        text = f.read()
    return sorted(set(re.findall(r'^"(src/[A-Za-z0-9_]+\.py)" = ', text, flags=re.MULTILINE)))


def loaded_engine_modules():
    # Every engine module the library has imported once all of it has been imported
    import theorizer.Evaluate                                   # noqa: F401
    import EvaluationCommon, EvaluationLLMAsAJudge, EvaluationQualifiedNovelty, EvaluationSurprisal   # noqa: E401,F401
    import ExportTheoryStoreAsHTML_NoPaperstore, ExportTheoryStoreAsHTML_NoPaperstore_attribution     # noqa: E401,F401
    src_dir = os.path.realpath(th.Legacy.SRC_DIR)
    out = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if (path is not None) and (os.path.realpath(os.path.dirname(path)) == src_dir):
            out.add(os.path.basename(path)[:-3])
    return out


#
#   Tests: the library offers no paper search
#

def test_the_library_has_no_paper_search(directory):
    for name in ("search_papers", "find_papers", "_paperfinder_candidates"):
        assert (not hasattr(th.Pipeline, name)), "Pipeline." + name + " is still there"
    assert (not hasattr(th.Legacy, "get_paperfinder_results")), "Legacy still binds the search client"
    assert (not hasattr(th.Evaluate, "predictive")), "Evaluate.predictive (paper search) is still there"
    assert (not hasattr(th.Legacy.SemanticScholar, "get_paperfinder_results")) and (not hasattr(sys.modules["SemanticScholar"], "get_paperfinder_results")), "the engine still has the search client"


def test_nothing_the_library_imports_can_search_for_papers(directory):
    loaded = loaded_engine_modules()
    assert ("PaperFinderRequests" not in loaded) and ("Theorizer" not in loaded), "a search-client module was imported: " + str(sorted(loaded))
    for name in sorted(loaded):
        with open(os.path.join(th.Legacy.SRC_DIR, name + ".py")) as f:
            hits = SEARCH_CLIENT_CALLS.findall(f.read())
        assert (len(hits) == 0), name + ".py still refers to the search client: " + str(sorted(set(hits)))


def test_the_wheel_ships_exactly_the_engine_modules_the_library_imports(directory):
    shipped = {os.path.basename(path)[:-3] for path in shipped_engine_modules()}
    loaded = loaded_engine_modules()
    assert (loaded <= shipped), "imported but not shipped (the installed library would fail): " + str(sorted(loaded - shipped))
    assert (shipped <= loaded), "shipped but never imported (dead weight?): " + str(sorted(shipped - loaded))
    for path in shipped_engine_modules():
        assert (os.path.isfile(os.path.join(REPO_ROOT, path))), path + " is listed in pyproject.toml but does not exist"


def main():
    print("TestNoPaperSearch.py")
    tests = [
        test_the_library_has_no_paper_search,
        test_nothing_the_library_imports_can_search_for_papers,
        test_the_wheel_ships_exactly_the_engine_modules_the_library_imports,
    ]
    for test in tests:
        directory = tempfile.mkdtemp()
        try:
            test(directory)
            print("  " + test.__name__ + ": OK")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    print("TestNoPaperSearch.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
