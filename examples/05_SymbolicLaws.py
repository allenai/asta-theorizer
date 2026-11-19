# 05_SymbolicLaws.py
# The symbolic path: turn extracted evidence into a typed table, and run the law-discovery suite
# over it to find quantitative relationships between the variables.
#
# This differs from generate_theories() in kind, not degree.  Theory generation asks a model to
# write theories in prose from the evidence.  This fits actual functional forms, trees, correlations
# and subgroup rules to a table of numbers, and reports what holds with what support.
#
#     python examples/05_SymbolicLaws.py /path/to/papers [model]
#
# Cost: about $0.60 -- the symbolic schema and the extraction against it are both real work.  The
# solver itself is free (it runs locally).
#
# There are four steps, and only the last two are new; the first two are the ordinary pipeline:
#
#   1. extract normally, to see what is in the papers
#   2. make a SYMBOLIC schema from those records -- typed, with units and closed vocabularies
#   3. extract again against it (src/ applies the symbolic extraction rules automatically)
#   4. build the table and run the solver

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


# The papers must be about this request, or the extraction finds nothing to record.
# For papers on another topic, change it to a request that fits them.
REQUEST = "Build a theory of how VIP interneurons control plasticity in auditory cortex"
PILOT_PAPERS = 6


def main():
    if (len(sys.argv) < 2):
        print("Usage: python examples/05_SymbolicLaws.py /path/to/papers")
        print("  (papers on VIP interneurons and auditory-cortex plasticity; for other papers, edit REQUEST at the top of this file)")
        return 1

    model = sys.argv[2] if (len(sys.argv) > 2) else None
    th.Config.init(project_dir="example-05-run", model_primary=model, model_extraction=model)
    papers = th.PaperImport.import_papers(th.PaperImport.find_files(sys.argv[1])[:PILOT_PAPERS])["papers"]
    if (len(papers) == 0):
        print("No papers found at " + str(sys.argv[1]))
        return 1

    #
    #   1. An ordinary extraction first
    #
    # The symbolic schema is derived from records, not from the request, so there has to be
    # something extracted before there is anything to derive it from.
    plan = th.Pipeline.plan_theory_query(REQUEST)
    query = plan["output"]

    first = th.Pipeline.extract_from_papers(papers, query["extraction_query"], query["extraction_schema"],
                                            save_as="extraction-exploratory")
    records = th.Pipeline.flatten_records(first, save_as="records-exploratory")
    if (len(records) == 0):
        print("Nothing extracted, so there is nothing to build a symbolic schema from.")
        return 1

    #
    #   2. A symbolic schema: declared types, units, closed vocabularies
    #
    symbolic = th.Pipeline.make_symbolic_schema(records, query, save_as="symbolic-schema")
    variables = symbolic["variables"]

    print("")
    print("--- symbolic schema: " + str(len(variables)) + " variable(s) ---")
    for variable in variables:
        detail = str(variable.get("type"))
        if (variable.get("unit")):
            detail += ", " + str(variable["unit"])
        if (variable.get("categorical_values")):
            detail += ", one of: " + ", ".join([str(v) for v in variable["categorical_values"]])[:50]
        print("  " + str(variable.get("name")).ljust(30) + "(" + detail + ")")

    #
    #   3. Extract again, against the symbolic schema
    #
    # This is the ordinary extraction call.  SchemaExtractionQueue in src/ notices the schema is
    # symbolic (by its inclusion_criteria) and adds the symbolic rules to the prompt itself -- the
    # types must be real numbers and booleans, categories must be exactly as declared, and an
    # unknown value must be left null rather than guessed.
    symbolic_query = symbolic["extraction_query"] or query["extraction_query"]
    second = th.Pipeline.extract_from_papers(papers, symbolic_query, variables, save_as="extraction-symbolic")
    typed_records = th.Pipeline.flatten_records(second, save_as="records-symbolic")

    #
    #   4. Build the table, then run the solver over it
    #
    # build_table() coerces each value to its declared type and DROPS anything that does not match,
    # rather than repairing it -- a guessed number becomes a confident wrong law.  Read the coverage
    # warnings it prints: they are how you find out the extraction is not following the schema.
    table = th.Symbolic.build_table(typed_records, symbolic, save_as="symbolic-table")

    print("")
    print("--- coverage (how much of each variable was actually usable) ---")
    for entry in table["coverage"]:
        print("  " + entry["variable"].ljust(30) + str(entry["filled"]).rjust(4) + " filled, " + str(entry["dropped"]).rjust(3) + " dropped")

    if (table["num_rows"] < 5):
        print("")
        print("Only " + str(table["num_rows"]) + " usable row(s) -- too few for the solver to say anything meaningful.")
        print("Extract over more papers before reading anything into a result.")
        return 0

    # SLOW_TOOLS are the two genetic-programming solvers; they are the difference between seconds
    # and many minutes.  Drop the skip argument once you want everything.
    result = th.Symbolic.run_solver(table, skip=th.Symbolic.SLOW_TOOLS, timeout=120, save_as="symbolic-report")

    print("")
    print("--- strongest patterns found ---")
    for highlight in result["highlights"][:10]:
        print("  " + str(highlight.get("description") or highlight.get("pattern"))[:100])

    th.Symbolic.save_report_markdown(result, "example-05-run/symbolic-report.md")

    print("")
    th.Project.show()
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
