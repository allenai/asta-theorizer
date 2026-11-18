# TestPdfRoute.py
# PDFs are read only through Mistral OCR.  An imported PDF without ocr="mistral" is refused with a
# message that says what to do, and a fetched open-access PDF without a Mistral key is left
# metadata-only (with a log line) rather than converted locally or crashing.
#
# Runs offline.  No API keys, no LLM calls, no network.
#
#     python tests/TestPdfRoute.py

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th

import PaperStore as ps


class without_env:
    # Remove one environment variable for the length of a block
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.saved = os.environ.pop(self.name, None)
        return self

    def __exit__(self, *exc):
        if (self.saved is not None):
            os.environ[self.name] = self.saved
        return False


def a_pdf_file(directory):
    path = os.path.join(directory, "paper.pdf")
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n% not a real PDF; nothing reads it\n")
    return path


def bare_paperstore(ocr_store=None, logged=None):
    store = ps.PaperStore.__new__(ps.PaperStore)
    store.generation_id = None
    store._ocr_store = ocr_store
    store.log_and_print = (lambda message: logged.append(message)) if (logged is not None) else (lambda message: None)
    store.populate_paper_pdf_urls = lambda paper_data: ["https://example.org/paper.pdf"]
    return store


#
#   Tests: importing a PDF
#

def test_a_pdf_without_ocr_is_refused_with_a_clear_error(directory):
    a_pdf_file(directory)
    try:
        th.PaperImport.import_papers(directory, read_metadata=False, show_progress=False)
    except ValueError as e:
        message = str(e)
        assert ("convert PDFs to markdown first" in message) and ("ocr=\"mistral\"" in message), "the error does not say what to do: " + message
    else:
        raise AssertionError("a PDF was imported without ocr=\"mistral\"")


def test_an_unknown_ocr_is_refused(directory):
    a_pdf_file(directory)
    for ocr in ("pymupdf", "local"):
        try:
            th.PaperImport.import_papers(directory, read_metadata=False, show_progress=False, ocr=ocr)
        except ValueError as e:
            assert ("convert PDFs to markdown first" in str(e)), str(e)
        else:
            raise AssertionError("ocr=" + ocr + " was accepted")


def test_mistral_ocr_without_a_key_is_refused_before_any_work(directory):
    a_pdf_file(directory)
    with without_env("MISTRAL_API_KEY"):
        try:
            th.PaperImport.import_papers(directory, read_metadata=False, show_progress=False, ocr="mistral")
        except ValueError as e:
            assert ("MISTRAL_API_KEY" in str(e)), str(e)
        else:
            raise AssertionError("ocr=\"mistral\" ran with no key")


def test_read_pdf_text_refuses_without_mistral(directory):
    path = a_pdf_file(directory)
    try:
        th.PaperImport.read_paper_text(path)
    except ValueError as e:
        assert ("convert PDFs to markdown first" in str(e)), str(e)
    else:
        raise AssertionError("read_paper_text() read a PDF without ocr=\"mistral\"")


def test_markdown_still_imports_with_no_ocr(directory):
    with open(os.path.join(directory, "a.md"), "w") as f:
        f.write("# A paper\n\nBody.\n")
    packed = th.PaperImport.import_papers(directory, read_metadata=False, show_progress=False)
    assert (packed["num_papers"] == 1), "a markdown file did not import"


#
#   Tests: a fetched open-access PDF
#

def test_a_fetched_pdf_without_a_key_is_not_converted_and_says_so(directory):
    def no_download(*args, **kwargs):
        raise AssertionError("downloaded a PDF with no way to convert it")

    saved_get = ps.requests.get
    ps.requests.get = no_download
    logged = []
    try:
        with without_env("MISTRAL_API_KEY"):
            result = bare_paperstore(logged=logged)._get_fulltext_from_legacy_path(corpus_id=1, paper_title="t", paper_data={})
    finally:
        ps.requests.get = saved_get

    assert (not result.found_fulltext()), "a paper got full text with no converter"
    assert any(("MISTRAL_API_KEY" in line) and ("not converted" in line) for line in logged), "no log line said why: " + str(logged)


def test_a_web_page_returned_by_ocr_is_a_miss(directory):
    class FakeOCR:
        def process_pdf(self, pdf_url, generation_id=None):
            return "<!DOCTYPE html>\n<html><head><title>Redirecting</title></head><body>" + ("x" * 500) + "</body></html>"

    saved = os.environ.get("MISTRAL_API_KEY")
    os.environ["MISTRAL_API_KEY"] = "test-key"
    try:
        result = bare_paperstore(ocr_store=FakeOCR())._get_fulltext_from_legacy_path(corpus_id=1, paper_title="t", paper_data={})
    finally:
        if (saved is None):
            os.environ.pop("MISTRAL_API_KEY", None)
        else:
            os.environ["MISTRAL_API_KEY"] = saved
    assert (not result.found_fulltext()), "an HTML page was taken as a paper"


def test_a_fetched_pdf_with_a_key_goes_to_ocr(directory):
    class FakeOCR:
        def process_pdf(self, pdf_url, generation_id=None):
            return "# A paper\n\n" + ("Sea ice albedo feedback accelerates summer loss. " * 6)

    saved = os.environ.get("MISTRAL_API_KEY")
    os.environ["MISTRAL_API_KEY"] = "test-key"
    try:
        result = bare_paperstore(ocr_store=FakeOCR())._get_fulltext_from_legacy_path(corpus_id=1, paper_title="t", paper_data={})
    finally:
        if (saved is None):
            os.environ.pop("MISTRAL_API_KEY", None)
        else:
            os.environ["MISTRAL_API_KEY"] = saved
    assert result.found_fulltext() and ("albedo feedback" in result.fulltext), "OCR text was not used"


#
#   Runner
#

def main():
    print("TestPdfRoute.py")
    th.Config.init(project_dir=None, verbose=False)
    tests = [
        test_a_pdf_without_ocr_is_refused_with_a_clear_error,
        test_an_unknown_ocr_is_refused,
        test_mistral_ocr_without_a_key_is_refused_before_any_work,
        test_read_pdf_text_refuses_without_mistral,
        test_markdown_still_imports_with_no_ocr,
        test_a_fetched_pdf_without_a_key_is_not_converted_and_says_so,
        test_a_web_page_returned_by_ocr_is_a_miss,
        test_a_fetched_pdf_with_a_key_goes_to_ocr,
    ]
    for test in tests:
        directory = tempfile.mkdtemp()
        try:
            test(directory)
            print("  " + test.__name__ + ": OK")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    print("TestPdfRoute.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
