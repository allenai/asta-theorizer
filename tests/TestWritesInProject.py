# TestWritesInProject.py
# A run in a project writes nothing outside it: not in the working directory, not beside the
# library's own code (site-packages, in an installed wheel), not in the folder the papers came from.
# The engine's caches -- OCR, the file backend's data/, Semantic Scholar responses -- and its prompt
# dumps all land in the project directory.
#
# The model is replaced by fakes.  Runs offline: no keys, no network.
#
#     python tests/TestWritesInProject.py

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th

import Persistence
import SemanticScholar


def snapshot(root:str):
    # {path: (size, mtime)} for every file under root, leaving out bytecode caches and VCS/venv dirs
    found = {}
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [name for name in subdirectories if (name not in ("__pycache__", ".git", ".venv"))]
        for filename in filenames:
            path = os.path.join(directory, filename)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            found[path] = (stat.st_size, stat.st_mtime_ns)
    return found


def changes(before:dict, after:dict):
    return sorted(path for path in set(before) | set(after) if (before.get(path) != after.get(path)))


def fake_extract(extraction_query, extraction_schema, paper_text, **kwargs):
    return {"response": {"results": [{"name_short": "finding", "finding": paper_text[-30:]}]}, "total_cost": 0.0}


def fake_discover(theory_query_information, paper_text, **kwargs):
    return {"response": {"proposed_slots": [{"slot_name": "gain", "slot_description": "..."}]}, "total_cost": 0.0}


def test_a_run_writes_only_inside_its_project(directory):
    cwd = os.path.join(directory, "cwd")
    project = os.path.join(directory, "project")
    corpus = os.path.join(directory, "corpus")
    for path in (cwd, corpus):
        os.makedirs(path)
    for name in ("a", "b"):
        with open(os.path.join(corpus, name + ".md"), "w") as f:
            f.write("# Paper " + name + "\n\nVIP activation raised gain in paper " + name + ".\n")

    package_dirs = sorted({os.path.realpath(th.Legacy.PACKAGE_DIR), os.path.realpath(th.Legacy.SRC_DIR), os.path.realpath(th.Legacy.REPO_ROOT)})
    before = {root: snapshot(root) for root in package_dirs + [corpus]}

    saved_cwd = os.getcwd()
    extraction_utils = th.Legacy.ExtractionUtils
    saved = (th.Legacy.extract_entities_from_paper, th.Legacy.discover_variables_from_paper, extraction_utils.completion)

    class Captured(Exception):
        pass

    def fake_completion(**kwargs):
        raise Captured()

    os.chdir(cwd)
    th.Legacy.extract_entities_from_paper = fake_extract
    th.Legacy.discover_variables_from_paper = fake_discover
    extraction_utils.completion = fake_completion
    try:
        th.Config.init(project_dir=project, verbose=False)

        papers = th.PaperImport.import_papers(corpus, read_metadata=False, show_progress=False)["papers"]
        th.Pipeline.discover_variables(papers, {"theory_request_normalized": "q"}, model_str="fake-model", show_progress=False)
        extraction = th.Pipeline.extract_from_papers(papers, "q", [{"name": "finding", "type": "str", "description": "d"}], model_str="fake-model", show_progress=False)
        th.Pipeline.flatten_records(extraction)
        th.Pipeline.to_theorystore([{"theory_name": "T", "theory_description": "d"}], "q", model_str="fake-model")

        # The engine's own writes: a prompt dump, the OCR cache, the S2 response cache, the PaperStore
        try:
            extraction_utils._getLLMResponseJSONMaybeCached("a prompt", "gpt-6-luna", 0.0, 100, False)
        except Captured:
            pass
        Persistence.store().save_ocr_cache("https://example.org/paper.pdf", {"pages": []}, "# A paper")
        client = SemanticScholar.SemanticScholar(api_key=None)
        client.save_cache("title", "a paper", 1, data_to_store={"data": []}, force_save=True)
        th.Project.get_paper_store()
        th.Project.save_paper_store()
    finally:
        th.Legacy.extract_entities_from_paper, th.Legacy.discover_variables_from_paper, extraction_utils.completion = saved
        os.chdir(saved_cwd)
        th.Project.close()

    assert (os.listdir(cwd) == []), "files were written to the working directory: " + str(os.listdir(cwd))
    for root, files in before.items():
        changed = changes(files, snapshot(root))
        assert (len(changed) == 0), "files were written outside the project, under " + root + ": " + str(changed[:5])

    written = set(os.listdir(project))
    for expected in ("ocr-cache", "s2_cache.json", "prompts", "theorystore.json"):
        assert (expected in written), expected + " is not in the project: " + str(sorted(written))


def test_with_no_project_the_engine_uses_the_working_directory(directory):
    th.Config.init(project_dir=None, verbose=False)
    saved_cwd = os.getcwd()
    os.chdir(directory)
    try:
        Persistence.store().save_ocr_cache("https://example.org/paper.pdf", {"pages": []}, "# A paper")
    finally:
        os.chdir(saved_cwd)
        Persistence.reset()
    assert ("ocr-cache" in os.listdir(directory)), "with no project the OCR cache did not go in the working directory: " + str(os.listdir(directory))
    assert (SemanticScholar.FILENAME_S2_CACHE == "s2_cache.json"), "with no project the S2 cache is not in the working directory"


def main():
    print("TestWritesInProject.py")
    tests = [
        test_a_run_writes_only_inside_its_project,
        test_with_no_project_the_engine_uses_the_working_directory,
    ]
    for test in tests:
        directory = tempfile.mkdtemp()
        try:
            test(directory)
            print("  " + test.__name__ + ": OK")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    print("TestWritesInProject.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
