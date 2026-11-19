# 06_VariableDiscovery.py
# Asking the papers what is worth extracting, BEFORE committing to a schema.
#
# There are two ways to find out that a schema is missing something, and they happen at different
# points:
#
#   discover_variables()   reads each paper and proposes what could be extracted, before any
#                          extraction has happened.  Per paper, so it sees what is actually in them.
#   make_variable_report() reads the records already extracted and reports what is missing from
#                          them.  One call, after the fact.  This is what 03_IteratingOnASchema uses.
#
# The paper's pipeline ran BOTH, in this order: discover variables -> extract -> variable report ->
# new schemas -> extract again.  This example shows the first half, which nothing else covers.
#
#     python examples/06_VariableDiscovery.py /path/to/papers [model]
#
# Cost: about $0.25 with a cheap model, more with a reasoning model.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


# The papers must be about this request, or the extraction finds nothing to record.
# For papers on another topic, change it to a request that fits them.
REQUEST = "Build a theory of how VIP interneurons control plasticity in auditory cortex"
NUM_PAPERS = 5


def main():
    if (len(sys.argv) < 2):
        print("Usage: python examples/06_VariableDiscovery.py /path/to/papers [model]")
        print("  (papers on VIP interneurons and auditory-cortex plasticity; for other papers, edit REQUEST at the top of this file)")
        return 1

    model = sys.argv[2] if (len(sys.argv) > 2) else None
    th.Config.init(project_dir="example-06-run", model_primary=model, model_extraction=model)

    papers = th.PaperImport.import_papers(th.PaperImport.find_files(sys.argv[1])[:NUM_PAPERS])["papers"]
    if (len(papers) == 0):
        print("No papers found at " + str(sys.argv[1]))
        return 1

    plan = th.Pipeline.plan_theory_query(REQUEST)
    query = plan["output"]

    #
    #   Ask each paper what is worth extracting
    #
    # This is one LLM call per paper, in parallel, and it happens before any extraction schema has
    # been committed to.  Each paper answers from what is actually in it, so between them they cover
    # variables the planner could not have guessed from the request alone.
    discovery = th.Pipeline.discover_variables(papers, query, save_as="variable-discovery")

    print("")
    print("--- what each paper proposed ---")
    for result in discovery["results"]:
        proposed = result.get("proposed")
        print("")
        print("  " + str(result.get("paper_title"))[:70])
        if (isinstance(proposed, dict)):
            for slot in (proposed.get("proposed_slots") or [])[:3]:
                print("    - " + str(slot.get("slot_name")) + "  (impact " + str(slot.get("impact_rating")) + ")")

    #
    #   Turn the proposals into schemas
    #
    # make_schemas_from_variable_discovery() takes the packed dict straight from discover_variables()
    # and returns whole new self-contained extraction schemas -- each with its own extraction query
    # and its own complete slot list.  The model decides how many.
    slots = th.Pipeline.collect_proposed_slots(discovery)
    num_slots = 0
    for paper_slots in slots.values():
        num_slots += len(paper_slots)

    if (num_slots == 0):
        print("")
        print("The discovery step returned no proposed_slots to build schemas from.")
        print("Look at discovery['results'][0]['proposed'] to see what came back.")
        th.Project.show()
        return 0

    print("")
    print("--- " + str(num_slots) + " proposed slot(s) across " + str(len(slots)) + " paper(s) ---")

    schemas = th.Pipeline.make_schemas_from_variable_discovery(discovery, query, save_as="schemas-from-discovery")
    proposed_schemas = (schemas["output"] or {}).get("extraction_schemas") or []

    for schema in proposed_schemas:
        print("  - " + str(schema.get("name", schema.get("short_name"))) + ": " + str(len(schema.get("extraction_schema") or [])) + " slots")

    #
    #   Extract with the first of them
    #
    if (len(proposed_schemas) > 0):
        chosen = proposed_schemas[0]
        # Its own schema id, so the store keeps these records apart from the first schema's
        extraction = th.Pipeline.extract_from_papers(papers, chosen["extraction_query"], chosen["extraction_schema"],
                                                     extraction_schema_id="schema-2", save_as="extraction-from-discovery")
        records = th.Pipeline.flatten_records(extraction, save_as="records-from-discovery")
        print("")
        print("Extracted " + str(len(records)) + " record(s) using a schema built from what the papers themselves proposed.")

    print("")
    th.Project.show()
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
