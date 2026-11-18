# TestPdfUrls.py
# The PDF URLs a fetched paper is tried at.  Mistral OCR refuses any URL that is not https (a 422,
# before it downloads anything), and Semantic Scholar gives arXiv's open-access links as
# http://arxiv.org/pdf/..., so every arXiv paper used to spend its first attempt on a certain failure.
#
# Runs offline.  No API keys, no network.
#
#     python tests/TestPdfUrls.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer  # noqa: F401  (puts the engine on the path)

import PaperStore as ps


def bare_paperstore():
    store = ps.PaperStore.__new__(ps.PaperStore)
    store.add_log_entry = lambda message: None
    return store


def s2_record(open_access_url=None, arxiv=None, acl=None):
    record = {"title": "A paper with a long enough title", "externalIds": {}}
    if (open_access_url is not None):
        record["openAccessPdf"] = {"url": open_access_url}
    if (arxiv is not None):
        record["externalIds"]["ArXiv"] = arxiv
    if (acl is not None):
        record["externalIds"]["ACL"] = acl
    return record


def test_an_http_arxiv_link_is_upgraded_to_https():
    urls = bare_paperstore().populate_paper_pdf_urls(s2_record("http://arxiv.org/pdf/2201.11903", arxiv="2201.11903"))
    assert (urls == ["https://arxiv.org/pdf/2201.11903", "https://arxiv.org/pdf/2201.11903.pdf"]), urls
    assert all(url.startswith("https://") for url in urls), urls


def test_upgrading_does_not_leave_a_duplicate():
    urls = bare_paperstore().populate_paper_pdf_urls(s2_record("http://arxiv.org/pdf/2201.11903.pdf", arxiv="2201.11903"))
    assert (urls == ["https://arxiv.org/pdf/2201.11903.pdf"]), urls


def test_https_links_and_their_order_are_unchanged():
    urls = bare_paperstore().populate_paper_pdf_urls(s2_record("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1/pdf", acl="2020.acl-main.1"))
    assert (urls == ["https://aclanthology.org/2020.acl-main.1.pdf", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1/pdf"]), urls


def test_any_http_link_is_upgraded():
    urls = bare_paperstore().populate_paper_pdf_urls(s2_record("HTTP://example.org/paper.pdf"))
    assert (urls == ["https://example.org/paper.pdf"]), urls


def main():
    print("TestPdfUrls.py")
    tests = [
        test_an_http_arxiv_link_is_upgraded_to_https,
        test_upgrading_does_not_leave_a_duplicate,
        test_https_links_and_their_order_are_unchanged,
        test_any_http_link_is_upgraded,
    ]
    for test in tests:
        test()
        print("  " + test.__name__ + ": OK")
    print("TestPdfUrls.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
