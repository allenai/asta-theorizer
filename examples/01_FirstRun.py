# 01_FirstRun.py
# The smallest complete run: a request in, theories out, over a handful of papers.
#
# Start here.  It is deliberately small -- six papers and one schema -- because the point is to see
# the shape of a run, not to produce a good theory.  Measured cost: about $0.45.
#
#     python examples/01_FirstRun.py /path/to/papers [model]
#
# Every stage saves itself into ./example-01-run/, so nothing here has to be re-run (or re-paid for)
# if you want to poke at the results afterwards.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


# The papers must be about this request, or the extraction finds nothing to record.
# For papers on another topic, change it to a request that fits them.
REQUEST = "Build a theory of how VIP interneurons control plasticity in auditory cortex"


def main():
    if (len(sys.argv) < 2):
        print("Usage: python examples/01_FirstRun.py /path/to/papers")
        print("  (papers on VIP interneurons and auditory-cortex plasticity; for other papers, edit REQUEST at the top of this file)")
        return 1

    papers_path = sys.argv[1]
    model = sys.argv[2] if (len(sys.argv) > 2) else None

    # Loads API keys, sends the chatty per-call output to a log, opens a directory to save into.
    # model_primary/model_extraction set the default model for every stage; any single stage can
    # still be given its own model_str.
    th.Config.init(project_dir="example-01-run", model_primary=model, model_extraction=model)

    # 1. Turn the request into a search query and a first extraction schema
    plan = th.Pipeline.plan_theory_query(REQUEST)
    query = plan["output"]
    print("Planned: " + str(len(query["extraction_schema"])) + " slots to extract, short_name=" + str(query["short_name"]))

    # 2. Load the papers.  Six is enough to see whether the schema is sensible.
    papers = th.PaperImport.import_papers(th.PaperImport.find_files(papers_path)[:6])["papers"]
    if (len(papers) == 0):
        print("No papers found at " + str(papers_path))
        return 1

    # 3. Extract, one call per paper, in parallel
    extraction = th.Pipeline.extract_from_papers(papers, query["extraction_query"], query["extraction_schema"])

    # 4. Flatten to one record per finding.  Each carries a uuid, which leads back (through the
    #    store's extraction result) to the paper it came from.
    records = th.Pipeline.flatten_records(extraction)
    if (len(records) == 0):
        print("Nothing was extracted -- look at the schema in " + str(plan.get("output", {}).get("short_name")))
        return 1

    # 5. Build theories from the evidence
    theories = th.Pipeline.generate_theories(records, query["theory_request_normalized"])

    response = theories["theory_response"] or {}
    found = (response.get("theories_general") or []) + (response.get("theories_specific") or [])

    # 6. Save them to the project's theorystore.  generate_theories() saves its own output file, but a
    #    theory reaches theorystore.json -- what reloads and the HTML export read -- only through
    #    to_theorystore(), which also records the model and generator that wrote it.
    th.Pipeline.to_theorystore(found, query["theory_request_normalized"], model_str=theories["model"], method="accuracy")

    for theory in found:
        print("  - " + str(theory.get("theory_name")))

    print("")
    th.Project.show()
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
