# 02_BringYourOwnPapers.py
# Loading papers you already have -- markdown or text -- with their metadata.  PDFs need
# ocr="mistral" (Mistral OCR, needs MISTRAL_API_KEY); otherwise convert them to markdown first.
#
# A file on disk has no title, authors or abstract, unlike a paper fetched through PaperStore.  An
# extraction record that says only "paper-doi101234jexample201900001" does not tell you where the
# evidence came from, so every imported paper gets metadata read out of the document, and can
# optionally be matched against Semantic Scholar for the things a local file cannot carry.
#
#     python examples/02_BringYourOwnPapers.py /path/to/papers [model]
#
# Cost: about $0.003 per paper for the metadata (measured on a small model), so roughly $3 for 1000.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


def main():
    if (len(sys.argv) < 2):
        print("Usage: python examples/02_BringYourOwnPapers.py /path/to/papers")
        return 1

    papers_path = sys.argv[1]
    model = sys.argv[2] if (len(sys.argv) > 2) else None
    th.Config.init(project_dir="example-02-run", model_primary=model, model_extraction=model)

    # find_files() takes a directory, a glob, a single file, or a list of any of those
    files = th.PaperImport.find_files(papers_path)
    print("Found " + str(len(files)) + " file(s); importing the first 5")

    # read_metadata reads title/authors/year/abstract/venue from the document with one LLM call.
    # The title is read by a model rather than by pattern matching on purpose: journal banners,
    # "OPEN ACCESS" notices and preprint watermarks all sit at the top of a converted PDF, and a
    # rule-based version produced titles like "Abstract" and "Neuron" for 88 papers out of 934.
    imported = th.PaperImport.import_papers(files[:5], read_metadata=True)
    papers = imported["papers"]

    for paper in papers:
        print("")
        print("  paper_id: " + str(paper.get("paper_id")))
        print("  title:    " + str(paper.get("title")))
        print("  year:     " + str(paper.get("year")) + "    venue: " + str(paper.get("venue")))
        print("  authors:  " + ", ".join(paper.get("authors") or [])[:70])
        if (paper.get("title_is_filename_fallback")):
            print("  NOTE: no title in the document, so the filename was used")

    print("")
    print("Failures: " + str(len(imported["failures"])))
    for failure in imported["failures"]:
        print("  " + str(failure.get("item")) + ": " + str(failure.get("error"))[:80])

    # Optionally fill in what a local file cannot carry: corpus id, a link, the canonical author
    # list.  This is a separate, narrower pass because Semantic Scholar is rate-limited and the LLM
    # calls are not.  Set S2_RATE_LIMIT (seconds between requests) to go faster if you have a key.
    print("")
    print("Looking the same papers up on Semantic Scholar...")
    completed = th.PaperImport.import_papers(files[:5], lookup_semantic_scholar=True)["papers"]
    for paper in completed:
        matched = paper.get("semantic_scholar_matched")
        print("  " + str(paper.get("title"))[:45].ljust(47) + ("matched: " + str(paper.get("url")) if (matched) else "no match (" + str(paper.get("semantic_scholar_reason"))[:40] + ")"))

    # Save the corpus so it does not have to be re-read or re-paid for
    th.PaperImport.save_papers(completed, "example-02-run/papers.jsonl")
    print("")
    print("Reload later with: papers = th.PaperImport.load_papers('example-02-run/papers.jsonl')")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
