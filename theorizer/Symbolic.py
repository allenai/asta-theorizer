# Symbolic.py
# Turning extracted records into a table a solver can work on, running the law-discovery suite over
# it, and reading the report.
#
# WHAT IS AND IS NOT HERE
# Making a symbolic schema is Pipeline.make_symbolic_schema(), which runs the function in src/.
# Extracting against one is Pipeline.extract_from_papers() -- SchemaExtractionQueue already detects a
# symbolic schema (by its inclusion_criteria field) and adds the symbolic extraction rules to the
# prompt itself.  Neither needs anything here.
#
# What was missing is the last three steps: coercing extracted values to the types the schema
# declares, handing the table to symbolic-solver, and reading what comes back.
#
# ON COERCION
# A value that does not match its declared type is DROPPED, and counted, rather than repaired.  The
# solver is only as good as the table it is given, and a guessed value is worse than a missing one:
# a wrong number becomes a wrong law, whereas a missing one just narrows the evidence.  So "5 mV" in
# a float column is dropped (units belong in the schema's `unit`, not in the value), and a
# categorical value outside the declared vocabulary is dropped.  build_table() reports every drop so
# that a schema which is not being followed is visible rather than silent.

from __future__ import annotations

from . import Config
from . import Legacy
from . import Project


#
#   Reading the schema
#

def as_variable_list(schema):
    # The model returns the symbolic schema either as a list of variable dicts, or as a dict keyed
    # by variable name.  Both happen in practice; normalise to a list, and drop anything that is not
    # a variable dict rather than letting it reach the coercion code as a bare string.
    variables = []
    if (isinstance(schema, dict)):
        for name, spec in schema.items():
            if (isinstance(spec, dict)):
                entry = dict(spec)
                entry.setdefault("name", name)
                variables.append(entry)
    elif (isinstance(schema, list)):
        for spec in schema:
            if (isinstance(spec, dict)):
                variables.append(spec)

    return variables


def get_variables(symbolic_schema):
    # The list of variables.  Pipeline.make_symbolic_schema() already unwrapped the prompt's
    # envelope, so packed["variables"] is the plain list -- this just accepts whichever piece the
    # caller happens to be holding: the packed dict, the schema object, or the list itself.
    if (isinstance(symbolic_schema, list)):
        return as_variable_list(symbolic_schema)

    if (isinstance(symbolic_schema, dict)):
        if (isinstance(symbolic_schema.get("variables"), list)):
            return as_variable_list(symbolic_schema["variables"])
        if (isinstance(symbolic_schema.get("extraction_schema"), list)):
            return as_variable_list(symbolic_schema["extraction_schema"])

    return []


def get_extraction_query(symbolic_schema):
    # The schema's own extraction query, set alongside the variables by make_symbolic_schema().
    if (isinstance(symbolic_schema, dict)):
        query = symbolic_schema.get("extraction_query")
        if (isinstance(query, str)) and (len(query.strip()) > 0):
            return query

    return None


def is_symbolic_schema(symbolic_schema):
    # A symbolic variable is one carrying inclusion_criteria -- the same test src/ uses to decide
    # whether to add the symbolic rules to the extraction prompt.
    for variable in get_variables(symbolic_schema):
        if (isinstance(variable, dict)) and ("inclusion_criteria" in variable):
            return True

    return False


#
#   Coercing values to their declared types
#

def coerce_value(value, variable:dict):
    # Returns (value, reason).  reason is None when the value was accepted; otherwise the value is
    # dropped and reason says why, so build_table() can report it.
    if (value is None):
        return None, "missing"

    declared_type = str(variable.get("type", "")).strip().lower()

    if (isinstance(value, str)) and (len(value.strip()) == 0):
        return None, "missing"

    if (declared_type in ("int", "integer")):
        # A bool is an int in Python, which is not what an int column means here
        if (isinstance(value, bool)):
            return None, "bool in an int column"
        try:
            return int(str(value).strip()), None
        except (TypeError, ValueError):
            return None, "not an int: " + str(value)[:40]

    if (declared_type in ("float", "number", "numeric")):
        if (isinstance(value, bool)):
            return None, "bool in a float column"
        try:
            return float(str(value).strip()), None
        except (TypeError, ValueError):
            # Usually a number with its unit stuck to it, which the schema says not to do
            return None, "not a number: " + str(value)[:40]

    if (declared_type in ("bool", "boolean")):
        if (isinstance(value, bool)):
            return value, None
        # The only normalisation done anywhere here: an exact match against a two-value vocabulary
        text = str(value).strip().lower()
        if (text == "true"):
            return True, None
        if (text == "false"):
            return False, None
        return None, "not a bool: " + str(value)[:40]

    if (declared_type in ("categorical", "category", "str", "string")):
        allowed = variable.get("categorical_values")
        if (not allowed):
            # Not a closed vocabulary, so there is nothing to check it against
            return str(value).strip(), None

        # Exact membership only.  Nothing here tries to work out which category an unexpected value
        # "really" means -- that guess is exactly the kind of thing that produces a confident wrong
        # answer downstream.
        text = str(value).strip()
        for candidate in allowed:
            if (text == str(candidate)):
                return str(candidate), None
        for candidate in allowed:
            if (text.lower() == str(candidate).strip().lower()):
                return str(candidate), None

        return None, "outside the declared vocabulary: " + str(value)[:40]

    return value, None


#
#   Building the table
#

def paper_id_from_uuid(uuid):
    # A record carries its uuid, not its paper.  The store made the uuid from its result's id
    # (extraction-result-3 -> e3.0, e3.1, ...), and that result holds the paper_id.
    if (uuid is None):
        return None
    result = Project.get_theory_store().extraction_results.get(str(uuid).split(".")[0].replace("e", "extraction-result-", 1))
    return getattr(result, "paper_id", None)


def build_table(records:list, symbolic_schema, include_paper_id:bool=False, show_progress:bool=True, save_as:str=None):
    # Turn extracted records into rows of typed values.
    #
    # One row per record.  Only the schema's own variables become columns -- paper titles and free
    # text are left out, because the solver infers its own schema from the table and a
    # high-cardinality text column turns into a pile of meaningless "patterns".
    #
    # Returns a packed dict.  packed["rows"] is the table; packed["coverage"] says, per variable,
    # how many records actually supplied a usable value, which is the number to look at before
    # believing anything the solver says.
    variables = get_variables(symbolic_schema)
    if (len(variables) == 0):
        raise ValueError("Symbolic.build_table(): The schema has no variables in it.  Pass the packed dict from Pipeline.make_symbolic_schema(), or its list of variables.")

    names = []
    for variable in variables:
        name = variable.get("name")
        if (name is not None) and (len(str(name).strip()) > 0):
            names.append(str(name))

    rows = []
    provenance = []
    filled = {}
    dropped = {}
    drop_examples = {}
    for name in names:
        filled[name] = 0
        dropped[name] = 0
        drop_examples[name] = []

    for record in records:
        if (not isinstance(record, dict)):
            continue

        row = {}
        for variable in variables:
            name = str(variable.get("name", ""))
            if (name not in filled):
                continue

            value, reason = coerce_value(record.get(name), variable)
            if (value is None):
                row[name] = None
                if (reason is not None) and (reason != "missing"):
                    dropped[name] += 1
                    if (len(drop_examples[name]) < 3):
                        drop_examples[name].append(reason)
            else:
                row[name] = value
                filled[name] += 1

        if (include_paper_id):
            row["paper_id"] = record.get("paper_id") or paper_id_from_uuid(record.get("uuid"))

        # A row with nothing in it is noise in the table
        has_a_value = False
        for name in names:
            if (row.get(name) is not None):
                has_a_value = True
        if (not has_a_value):
            continue

        rows.append(row)
        provenance.append({"paper_id": record.get("paper_id"), "paper_title": record.get("paper_title")})

    coverage = []
    for name in names:
        coverage.append({
            "variable": name,
            "filled": filled[name],
            "dropped": dropped[name],
            "fill_rate": round(filled[name] / len(records), 3) if (len(records) > 0) else 0.0,
            "drop_reasons": drop_examples[name],
        })

    if (show_progress):
        print("Symbolic.build_table(): " + str(len(rows)) + " row(s) from " + str(len(records)) + " record(s), " + str(len(names)) + " column(s)", flush=True)

        empty = []
        for entry in coverage:
            if (entry["filled"] == 0):
                empty.append(entry["variable"])
        if (len(empty) > 0):
            print("Symbolic.build_table(): WARNING: " + str(len(empty)) + " variable(s) have no usable values at all: " + ", ".join(empty[:6]), flush=True)

        for entry in coverage:
            if (entry["dropped"] > 0):
                print("Symbolic.build_table(): WARNING: " + entry["variable"] + " -- " + str(entry["dropped"]) + " value(s) did not match the declared type (e.g. " + str(entry["drop_reasons"][0]) + ")", flush=True)

    # Pack output
    packed = {
        "rows": rows,
        "provenance": provenance,
        "columns": names,
        "coverage": coverage,
        "num_records": len(records),
        "num_rows": len(rows),
    }

    if (Project.is_open()):
        saved_path = Project.save(save_as if (save_as is not None) else "symbolic_table", packed)
        if (show_progress):
            print("  saved: " + str(saved_path), flush=True)

    return packed


#
#   Running the solver
#

# The two slowest tools by a wide margin.  Skipping them is the difference between a run that takes
# seconds and one that takes many minutes, and they are the ones most likely not to be installed.
SLOW_TOOLS = ["pysr_symbolic", "gplearn_symbolic"]


def run_solver(table, skip:list=None, timeout:float=None, parallel:bool=False, quiet:bool=False, cfg:dict=None, show_progress:bool=True, save_as:str=None):
    # Run the law-discovery suite over a table and return its report.
    #
    # table     the packed dict from build_table(), or a plain list of row dicts
    # skip      tool names to leave out (Symbolic.SLOW_TOOLS is the usual choice)
    # timeout   per-tool wall-clock limit in seconds; a tool that runs over is listed as skipped
    #
    # quiet=False (the default) lets the suite print its own per-tool progress.  The suite can run
    # for minutes; watching which tool it is on is how you tell it is working rather than stuck.
    #
    # Every tool runs independently: one that is not installed, or errors, or times out, is reported
    # rather than failing the run.  packed["report"]["highlights"] is the ranked cross-tool summary.
    rows = table
    if (isinstance(table, dict)):
        rows = table.get("rows") or []

    if (len(rows) == 0):
        raise ValueError("Symbolic.run_solver(): The table is empty.  Check build_table()'s coverage report -- if every variable came back empty, the extraction did not follow the symbolic schema.")

    solver_cfg = {"parallel": parallel}
    if (skip is not None) and (len(skip) > 0):
        solver_cfg["skip_tools"] = list(skip)
    if (timeout is not None):
        solver_cfg["tool_timeout"] = float(timeout)
    if (cfg is not None):
        solver_cfg.update(cfg)

    if (show_progress):
        print("Symbolic.run_solver(): Starting on " + str(len(rows)) + " row(s)" + (" (skipping " + ", ".join(skip) + ")" if (skip) else "") + " -- the slower tools can take several minutes", flush=True)

    solver = Legacy.load_symbolic_solver()
    report = solver.run_suite(list(rows), cfg=solver_cfg, quiet=quiet)

    highlights = report.get("highlights") or []
    if (show_progress):
        print("Symbolic.run_solver(): " + str(len(highlights)) + " highlight(s) from " + str(len(report.get("tools_run") or [])) + " tool(s), " + str(len(report.get("tools_skipped") or [])) + " skipped", flush=True)

    # Pack output
    packed = {
        "report": report,
        "num_rows": len(rows),
        "highlights": highlights,
        "tools_run": report.get("tools_run") or [],
        "tools_skipped": report.get("tools_skipped") or [],
    }

    if (Project.is_open()):
        saved_path = Project.save(save_as if (save_as is not None) else "symbolic_report", packed)
        if (show_progress):
            print("  saved: " + str(saved_path), flush=True)

    return packed


def discover_laws(records:list, symbolic_schema, skip:list=None, timeout:float=None, quiet:bool=False, show_progress:bool=True, save_as:str=None):
    # build_table() then run_solver(), for when you do not need to look at the table in between.
    table = build_table(records, symbolic_schema, show_progress=show_progress,
                        save_as=(save_as + "-table") if (save_as is not None) else None)
    return run_solver(table, skip=skip, timeout=timeout, quiet=quiet, show_progress=show_progress,
                      save_as=(save_as + "-report") if (save_as is not None) else None)


#
#   Reading the report
#

def report_to_markdown(report, max_per_tool:int=12):
    # A readable version of the report.  Pass either the packed dict from run_solver() or the raw
    # report inside it.
    if (isinstance(report, dict)) and ("report" in report):
        report = report["report"]

    lines = []
    lines.append("# Symbolic solver report")
    lines.append("")

    highlights = report.get("highlights") or []
    lines.append("## Highlights (" + str(len(highlights)) + ")")
    lines.append("")
    if (len(highlights) == 0):
        lines.append("Nothing reached the reporting threshold.  With a small or sparse table that is the")
        lines.append("expected outcome, and is worth more than a spurious law -- check the coverage report")
        lines.append("from build_table() before reading anything into it.")
    else:
        for highlight in highlights:
            description = highlight.get("description") or highlight.get("pattern") or str(highlight)
            lines.append("- " + str(description))
            support = highlight.get("support")
            score = highlight.get("score")
            detail = []
            if (support is not None):
                detail.append("support " + str(support))
            if (score is not None):
                detail.append("score " + str(round(score, 3) if (isinstance(score, float)) else score))
            if (highlight.get("tool")):
                detail.append("via " + str(highlight["tool"]))
            if (len(detail) > 0):
                lines.append("  (" + ", ".join(detail) + ")")
    lines.append("")

    lines.append("## By tool")
    lines.append("")
    results_by_tool = report.get("results_by_tool") or {}
    for tool in sorted(results_by_tool.keys()):
        patterns = (results_by_tool[tool] or {}).get("patterns") or []
        lines.append("### " + str(tool) + " (" + str(len(patterns)) + ")")
        for pattern in patterns[:max_per_tool]:
            description = pattern.get("description") or pattern.get("pattern") or str(pattern)
            lines.append("- " + str(description))
        if (len(patterns) > max_per_tool):
            lines.append("- ...and " + str(len(patterns) - max_per_tool) + " more")
        lines.append("")

    skipped = report.get("tools_skipped") or []
    if (len(skipped) > 0):
        lines.append("## Not run")
        lines.append("")
        for entry in skipped:
            if (isinstance(entry, dict)):
                lines.append("- " + str(entry.get("tool")) + ": " + str(entry.get("reason")))
            else:
                lines.append("- " + str(entry))
        lines.append("")

    return "\n".join(lines)


def save_report_markdown(report, filename:str):
    text = report_to_markdown(report)
    with open(filename, "w") as f:
        f.write(text)

    print("Symbolic.save_report_markdown(): Wrote " + str(filename), flush=True)
    return filename
