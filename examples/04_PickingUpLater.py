# 04_PickingUpLater.py
# Coming back to work you already paid for, in a new session, without re-running anything.
#
# Run 01 or 03 first to make a project directory, then point this at it:
#
#     python examples/04_PickingUpLater.py example-03-run
#
# Nothing here calls an LLM except the last step, which is optional.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


def main():
    if (len(sys.argv) < 2):
        print("Usage: python examples/04_PickingUpLater.py <project directory>")
        return 1

    th.Config.init(project_dir=sys.argv[1])

    #
    #   What is in here?
    #
    # show() works by reading the directory, so it is accurate even if you have moved, copied or
    # deleted files with ordinary tools since the last run.
    items = th.Project.show()
    if (len(items) == 0):
        print("Nothing saved in " + str(sys.argv[1]) + " yet -- run examples/01_FirstRun.py first.")
        return 1

    #
    #   Load something back, by name or by number
    #
    print("")
    records = None
    for item in items:
        if ("record" in item["name"]):
            records = th.Project.load(item["index"])
            print("Loaded " + str(len(records)) + " record(s) from '" + item["name"] + "'")
            break

    if (records is None):
        print("No saved records in this project; load anything with th.Project.load(<name or number>)")
        return 0

    #
    #   Look at the evidence before deciding what to do with it
    #
    # A record does not carry its paper; its uuid does.  The store made the uuid from its extraction
    # result's id (extraction-result-3 -> e3.0, e3.1, ...), and that result holds the paper_id.  The
    # title comes from the project's PaperStore, which fetch_papers() and import_papers() fill.
    store = th.Project.get_theory_store()
    paper_store = th.Project.get_paper_store()
    papers_seen = set()
    for record in records:
        result = store.extraction_results.get(str(record.get("uuid")).split(".")[0].replace("e", "extraction-result-", 1))
        paper_id = getattr(result, "paper_id", None)
        paper = paper_store.get_paper_by_key(paper_id) if (paper_id is not None) else None
        papers_seen.add(str((paper or {}).get("title") or paper_id))

    print("Those records came from " + str(len(papers_seen)) + " paper(s):")
    for title in sorted(papers_seen):
        print("  - " + str(title)[:70])

    #
    #   Carry on from here.  This is a new stage in the same project, so it saves alongside the rest.
    #
    print("")
    print("Anything you do now appends to the same project.  For example:")
    print("")
    print("    plan  = th.Project.load('plan')")
    print("    query = plan['output']")
    print("    more  = th.Pipeline.generate_theories(records, query['theory_request_normalized'],")
    print("                                          objective='novelty', save_as='theories-novelty')")
    # Theories reach theorystore.json (what reloads and the HTML export read) only through
    # to_theorystore(), so the printed next step includes it.
    print("    found = more['theory_response']['theories_general'] + more['theory_response']['theories_specific']")
    print("    th.Pipeline.to_theorystore(found, query['theory_request_normalized'], model_str=more['model'], method='novelty')")
    print("")
    print("The earlier files are untouched -- new work is numbered after them, never over them.")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
