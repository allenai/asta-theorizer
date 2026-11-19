# TestPaperImport.py
# Tests for theorizer/PaperImport.py -- finding files, paper ids, and saving/reloading a corpus.
#
# The metadata reading is one LLM call per paper, so it is not tested here.  Everything around it is.
#
#     python tests/TestPaperImport.py

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


def make_corpus(directory):
    # A small directory of files, including one that should not be picked up
    os.makedirs(os.path.join(directory, "subdir"), exist_ok=True)
    written = []
    for name in ["a.md", "b.markdown", "c.txt", "subdir/d.md"]:
        path = os.path.join(directory, name)
        with open(path, "w") as f:
            f.write("# Title of " + name + "\n\nSome body text.\n")
        written.append(path)

    with open(os.path.join(directory, "notes.docx"), "w") as f:
        f.write("should be ignored")

    return written


#
#   Tests
#

def test_find_files_accepts_a_directory(directory):
    make_corpus(directory)
    found = th.PaperImport.find_files(directory)
    assert (len(found) == 4), "expected 4 papers (recursing, ignoring .docx), got " + str(len(found))

    for path in found:
        assert (not path.endswith(".docx")), "picked up a file type it should not have"
    print("  test_find_files_accepts_a_directory: OK")


def test_find_files_accepts_globs_files_and_lists(directory):
    written = make_corpus(directory)

    one = th.PaperImport.find_files(written[0])
    assert (len(one) == 1), "a single file path did not return one file"

    globbed = th.PaperImport.find_files(os.path.join(directory, "*.md"))
    assert (len(globbed) == 1), "a glob did not return the expected file"

    listed = th.PaperImport.find_files([written[0], written[1]])
    assert (len(listed) == 2), "a list of files did not return both"

    # Duplicates must collapse, or papers get extracted (and paid for) twice
    duplicated = th.PaperImport.find_files([written[0], written[0], directory])
    ids = set(duplicated)
    assert (len(duplicated) == len(ids)), "the same file was returned more than once"
    print("  test_find_files_accepts_globs_files_and_lists: OK")


def test_missing_path_returns_nothing(directory):
    found = th.PaperImport.find_files(os.path.join(directory, "does-not-exist"))
    assert (found == []), "a path that does not exist returned something"
    print("  test_missing_path_returns_nothing: OK")


def test_paper_ids_are_stable_and_readable(directory):
    first = th.PaperImport.make_paper_id("/some/where/Cortical Circuit (2013).md")
    second = th.PaperImport.make_paper_id("/some/where/Cortical Circuit (2013).md")

    assert (first == second), "the same filename produced two different ids"
    assert (first.startswith("paper-")), "an id does not start with 'paper-': " + first
    assert (" " not in first), "an id contains a space: " + first
    assert (first != th.PaperImport.make_paper_id("/some/where/Another Paper.md")), "two different files share an id"
    print("  test_paper_ids_are_stable_and_readable: OK")


def test_paper_ids_are_unique_within_a_batch(directory):
    # These two differ only past the characters make_paper_id() keeps of the name
    first = os.path.join(directory, "Deming1992 - epidemic in the Gambia following control as endemic I descriptive findings.md")
    second = os.path.join(directory, "Deming1992 - epidemic in the Gambia following control as endemic II efficacy to tOPV.md")
    third = os.path.join(directory, "Another Paper.md")
    for path, text in ((first, "first"), (second, "second"), (third, "third")):
        with open(path, "w") as f:
            f.write(text)

    ids = th.PaperImport.make_paper_ids([first, second, third])
    assert (len(set(ids.values())) == 3), "two files in one batch share an id: " + str(ids)
    assert (ids == th.PaperImport.make_paper_ids([first, second, third])), "the same batch got different ids"

    # An id depends on the file alone: importing it with other files, or in another order, or
    # adding a file beside it, does not change it
    assert (th.PaperImport.make_paper_ids([second])[second] == ids[second]), "a file's id depends on the other files in the batch"
    assert (th.PaperImport.make_paper_ids([third, second, first]) == ids), "a file's id depends on the order of the batch"
    print("  test_paper_ids_are_unique_within_a_batch: OK")


def test_save_and_load_papers(directory):
    papers = [
        {"paper_id": "paper-a", "title": "First", "markdown": "text one"},
        {"paper_id": "paper-b", "title": "Second", "markdown": "text two"},
    ]
    path = os.path.join(directory, "papers.jsonl")
    th.PaperImport.save_papers(papers, path)

    reloaded = th.PaperImport.load_papers(path)
    assert (reloaded == papers), "papers did not survive the round trip"

    # One JSON object per line, so a big corpus can be streamed and grepped
    with open(path) as f:
        lines = f.readlines()
    assert (len(lines) == 2), "expected one line per paper"
    print("  test_save_and_load_papers: OK")


def test_default_extensions_are_what_the_warning_names(directory):
    # The empty-corpus warning tells the user which extensions were looked for.  If that list ever
    # drifts from the one find_files() actually uses, the warning becomes misleading.
    assert (".md" in th.PaperImport.DEFAULT_EXTENSIONS), "markdown is not in the default extensions"
    assert (".pdf" in th.PaperImport.DEFAULT_EXTENSIONS), "pdf is not in the default extensions"

    make_corpus(directory)
    only_pdf = th.PaperImport.find_files(directory, extensions=[".pdf"])
    assert (only_pdf == []), "asking for only .pdf returned other file types"
    print("  test_default_extensions_are_what_the_warning_names: OK")


def test_a_markdown_heading_is_the_title_without_metadata(directory):
    # With read_metadata=False there is no LLM call to read a title, but a markdown file that opens
    # with "# Title" has already said what it is called.  A file with no such heading keeps its
    # filename, and only that one counts as a filename fallback.
    with open(os.path.join(directory, "with-heading.md"), "w") as f:
        f.write("Preprint notice\n\n# Plasticity Control by VIP Interneurons\n\n## Methods\n\nBody text.\n")
    with open(os.path.join(directory, "no-heading.md"), "w") as f:
        f.write("## Only a second-level heading\n\nBody text.\n")

    th.Config.init(project_dir=None, verbose=False)
    papers = th.PaperImport.import_papers(directory, read_metadata=False, show_progress=False)["papers"]
    by_file = {os.path.basename(paper["source_file"]): paper for paper in papers}

    titled = by_file["with-heading.md"]
    assert (titled["title"] == "Plasticity Control by VIP Interneurons"), "the # heading was not used as the title: " + str(titled["title"])
    assert (not titled.get("title_is_filename_fallback")), "a heading title was counted as a filename fallback"

    untitled = by_file["no-heading.md"]
    assert (untitled["title"] == "no-heading"), "a file with no # heading should keep its filename: " + str(untitled["title"])
    assert (untitled.get("title_is_filename_fallback") is True), "the filename title was not marked as a fallback"
    print("  test_a_markdown_heading_is_the_title_without_metadata: OK")


#
#   Runner
#

def main():
    print("TestPaperImport.py")
    tests = [
        test_find_files_accepts_a_directory,
        test_find_files_accepts_globs_files_and_lists,
        test_missing_path_returns_nothing,
        test_paper_ids_are_stable_and_readable,
        test_paper_ids_are_unique_within_a_batch,
        test_save_and_load_papers,
        test_default_extensions_are_what_the_warning_names,
        test_a_markdown_heading_is_the_title_without_metadata,
    ]

    for test in tests:
        directory = tempfile.mkdtemp()
        try:
            test(directory)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    print("TestPaperImport.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
