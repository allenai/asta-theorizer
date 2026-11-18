# 03_IteratingOnASchema.py
# The workflow the library exists for: extract a little, look at it, improve the schema, extract
# again -- and only then run it over everything.
#
# The first schema a model proposes is a guess made before seeing any of the papers.  Running it
# over 900 papers to find that out is an expensive way to learn it.  So: pilot on two papers, read
# the records, ask what is missing, extract again, compare.
#
#     python examples/03_IteratingOnASchema.py /path/to/papers [model]
#
# Cost: about $0.35 -- the pilots are small, the reasoning calls are not.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


REQUEST = "Build a theory of how VIP interneurons control gain in visual cortex"
PILOT_PAPERS = 2


def show_records(records:list, label:str):
    print("")
    print("--- " + label + ": " + str(len(records)) + " record(s) ---")
    for record in records[:3]:
        print("")
        for key in sorted(record.keys()):
            if (key in ("paper_id", "paper_title")):
                continue
            value = str(record[key])
            if (len(value) > 90):
                value = value[:90] + "..."
            print("  " + key.ljust(28) + value)


def main():
    if (len(sys.argv) < 2):
        print("Usage: python examples/03_IteratingOnASchema.py /path/to/papers")
        return 1

    model = sys.argv[2] if (len(sys.argv) > 2) else None
    th.Config.init(project_dir="example-03-run", model_primary=model, model_extraction=model)
    files = th.PaperImport.find_files(sys.argv[1])
    papers = th.PaperImport.import_papers(files[:PILOT_PAPERS])["papers"]

    #
    #   Round 1: the schema the planner guessed, on two papers
    #
    plan = th.Pipeline.plan_theory_query(REQUEST)
    query = plan["output"]

    first = th.Pipeline.extract_from_papers(papers, query["extraction_query"], query["extraction_schema"],
                                            save_as="extraction-pilot-v1")
    records = th.Pipeline.flatten_records(first, save_as="records-pilot-v1")
    show_records(records, "round 1, the planner's schema")

    #
    #   Ask what the schema missed
    #
    # This reads across everything extracted so far and reports variables that are missing, implicit,
    # or worth splitting out.  It is one call, and it is the expensive part of this script.
    report = th.Pipeline.make_variable_report(records, query)
    suggestions = (report["output"] or {}).get("suggestions") or []

    print("")
    print("--- what the first schema missed ---")
    for suggestion in suggestions:
        print("  - " + str(suggestion.get("suggestion_short_name")))

    #
    #   Round 2: extend the schema with what the report suggested
    #
    # This is the variable-REPORT route, and it is not the same as the variable-DISCOVERY route in
    # 06_VariableDiscovery.py.  The report was written after looking at the records you already have,
    # so its suggestions are added to the schema you already have -- the columns you have keep their
    # meaning, and the two rounds stay comparable.  There is no LLM call here: the suggestions
    # already carry the slot names and instructions.
    extended_schema = th.Pipeline.extend_schema_with_suggestions(query["extraction_schema"], suggestions, max_suggestions=3)

    print("")
    print("--- schema grew from " + str(len(query["extraction_schema"])) + " to " + str(len(extended_schema)) + " slots ---")
    for slot in extended_schema[len(query["extraction_schema"]):]:
        print("  + " + str(slot.get("name")))

    second = th.Pipeline.extract_from_papers(papers, query["extraction_query"], extended_schema,
                                             save_as="extraction-pilot-v2")
    records_v2 = th.Pipeline.flatten_records(second, save_as="records-pilot-v2")
    show_records(records_v2, "round 2, the extended schema")

    #
    #   Both rounds are on disk, side by side, to compare
    #
    print("")
    th.Project.show()
    print("")
    print("Happy with v2?  Run it over the whole corpus -- the same call with more papers:")
    print("")
    print("    papers = th.PaperImport.import_papers('" + str(sys.argv[1]) + "')['papers']")
    print("    ext = th.Pipeline.extract_from_papers(papers, chosen['extraction_query'],")
    print("                                          chosen['extraction_schema'],")
    print("                                          max_workers=25, save_as='extraction-full')")
    print("")
    print("It checkpoints per paper, so a crash partway through does not lose what it already did.")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
