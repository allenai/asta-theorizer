# TestSymbolic.py
# Tests for theorizer/Symbolic.py -- coercing extracted values to their declared types, building the
# table, and running the solver over it.
#
# The coercion tests are the ones that matter.  A value that quietly gets repaired into the wrong
# number becomes a confident wrong law, which is worse than no law at all.
#
# Runs offline.  No API keys, no LLM calls, no cost.
#
#     python tests/TestSymbolic.py

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


SCHEMA = {"variables": [
    {"name": "vip_activation", "type": "float", "unit": "normalized", "inclusion_criteria": "..."},
    {"name": "baseline_rate", "type": "float", "unit": "Hz", "inclusion_criteria": "..."},
    {"name": "plasticity_change", "type": "float", "unit": "ratio", "inclusion_criteria": "..."},
    {"name": "n_cells", "type": "int", "inclusion_criteria": "..."},
    {"name": "was_awake", "type": "bool", "inclusion_criteria": "..."},
    {"name": "behavioral_state", "type": "categorical", "categorical_values": ["quiet", "rewarded"], "inclusion_criteria": "..."},
]}


def make_records(count:int=60):
    # A table with a law planted in it: plasticity_change = 2 * vip_activation + 1
    records = []
    for i in range(count):
        activation = i / float(count)
        records.append({
            "paper_id": "paper-" + str(i).zfill(2),
            "paper_title": "Paper " + str(i),
            "vip_activation": activation,
            "baseline_rate": 5.0 + (i % 7),
            "plasticity_change": (2.0 * activation) + 1.0,
            "n_cells": 10 + i,
            "was_awake": (i % 2 == 0),
            "behavioral_state": "rewarded" if (i % 2 == 0) else "quiet",
        })
    return records


#
#   Tests: reading the schema
#

def test_schema_can_be_given_in_any_shape(directory):
    variables = SCHEMA["variables"]

    assert (len(th.Symbolic.get_variables(SCHEMA)) == len(variables)), "the packed dict form was not read"
    assert (len(th.Symbolic.get_variables(SCHEMA["variables"])) == len(variables)), "the output form was not read"
    assert (len(th.Symbolic.get_variables(variables)) == len(variables)), "the plain list form was not read"
    print("  test_schema_can_be_given_in_any_shape: OK")


def test_symbolic_schemas_are_recognised(directory):
    # inclusion_criteria is the same marker src/SchemaExtractionQueue.py uses
    assert (th.Symbolic.is_symbolic_schema(SCHEMA) is True), "a symbolic schema was not recognised"

    ordinary = [{"name": "finding", "type": "str", "description": "a finding"}]
    assert (th.Symbolic.is_symbolic_schema(ordinary) is False), "an ordinary schema was called symbolic"
    print("  test_symbolic_schemas_are_recognised: OK")


#
#   Tests: coercion
#

def test_good_values_are_kept(directory):
    checks = [
        ({"type": "float"}, 1.5, 1.5),
        ({"type": "float"}, "1.5", 1.5),
        ({"type": "int"}, 42, 42),
        ({"type": "int"}, "42", 42),
        ({"type": "bool"}, True, True),
        ({"type": "bool"}, "false", False),
        ({"type": "categorical", "categorical_values": ["quiet", "rewarded"]}, "quiet", "quiet"),
        ({"type": "categorical", "categorical_values": ["quiet", "rewarded"]}, "QUIET", "quiet"),
    ]
    for variable, given, expected in checks:
        value, reason = th.Symbolic.coerce_value(given, variable)
        assert (value == expected), str(given) + " became " + str(value) + ", expected " + str(expected)
        assert (reason is None), str(given) + " was rejected: " + str(reason)
    print("  test_good_values_are_kept: OK")


def test_bad_values_are_dropped_not_repaired(directory):
    # This is the important one.  A number with its unit stuck on, or a category the schema does not
    # define, must be dropped -- nothing here may guess what was "really" meant.
    checks = [
        ({"type": "float"}, "0.5 mV"),
        ({"type": "float"}, "about 5"),
        ({"type": "float"}, "n/a"),
        ({"type": "int"}, "twelve"),
        ({"type": "int"}, True),
        ({"type": "bool"}, "yes"),
        ({"type": "bool"}, 1),
        ({"type": "categorical", "categorical_values": ["quiet", "rewarded"]}, "asleep"),
        ({"type": "categorical", "categorical_values": ["quiet", "rewarded"]}, "quiet-ish"),
    ]
    for variable, given in checks:
        value, reason = th.Symbolic.coerce_value(given, variable)
        assert (value is None), str(given) + " was accepted as " + str(value) + " -- it should have been dropped"
        assert (reason is not None), str(given) + " was dropped without saying why"
    print("  test_bad_values_are_dropped_not_repaired: OK")


def test_missing_values_are_missing_not_errors(directory):
    for given in [None, "", "   "]:
        value, reason = th.Symbolic.coerce_value(given, {"type": "float"})
        assert (value is None), "a missing value became " + str(value)
        assert (reason == "missing"), "a missing value was reported as '" + str(reason) + "'"
    print("  test_missing_values_are_missing_not_errors: OK")


#
#   Tests: the table
#

def test_build_table_reports_coverage_and_drops(directory):
    records = make_records(20)
    records.append({"paper_id": "bad-1", "vip_activation": "0.5 mV", "behavioral_state": "asleep", "plasticity_change": 3.0})

    table = th.Symbolic.build_table(records, SCHEMA, show_progress=False)
    assert (table["num_rows"] == 21), "expected 21 rows, got " + str(table["num_rows"])
    assert (len(table["columns"]) == 6), "expected 6 columns"

    by_name = {}
    for entry in table["coverage"]:
        by_name[entry["variable"]] = entry

    assert (by_name["vip_activation"]["filled"] == 20), "a good column lost values"
    assert (by_name["vip_activation"]["dropped"] == 1), "the bad float was not counted as dropped"
    assert (by_name["behavioral_state"]["dropped"] == 1), "the bad category was not counted as dropped"
    assert (len(by_name["vip_activation"]["drop_reasons"]) > 0), "no reason was recorded for the drop"
    print("  test_build_table_reports_coverage_and_drops: OK")


def test_free_text_columns_stay_out_of_the_table(directory):
    # The solver infers its own schema from the table; a paper title column would become a
    # high-cardinality categorical and produce meaningless "patterns".
    records = make_records(5)
    table = th.Symbolic.build_table(records, SCHEMA, show_progress=False)

    for row in table["rows"]:
        assert ("paper_title" not in row), "paper_title leaked into the table"
        assert ("paper_id" not in row), "paper_id leaked into the table by default"

    # ...but provenance is kept alongside, so a row can still be traced back
    assert (len(table["provenance"]) == len(table["rows"])), "provenance does not line up with the rows"
    assert (table["provenance"][0]["paper_title"] is not None), "provenance lost the paper title"
    print("  test_free_text_columns_stay_out_of_the_table: OK")


def test_empty_rows_are_left_out(directory):
    records = make_records(3)
    records.append({"paper_id": "empty-1", "paper_title": "Nothing useful"})

    table = th.Symbolic.build_table(records, SCHEMA, show_progress=False)
    assert (table["num_rows"] == 3), "a row with no usable values was kept"
    print("  test_empty_rows_are_left_out: OK")


def test_a_schema_with_no_variables_raises(directory):
    try:
        th.Symbolic.build_table(make_records(3), {"variables": []}, show_progress=False)
        raise AssertionError("an empty schema did not raise")
    except ValueError as e:
        assert ("no variables" in str(e)), "unexpected error: " + str(e)
    print("  test_a_schema_with_no_variables_raises: OK")


def test_an_empty_table_raises_with_a_useful_message(directory):
    try:
        th.Symbolic.run_solver({"rows": []}, show_progress=False)
        raise AssertionError("an empty table did not raise")
    except ValueError as e:
        assert ("coverage" in str(e)), "the error does not point at the coverage report: " + str(e)
    print("  test_an_empty_table_raises_with_a_useful_message: OK")


#
#   Tests: the solver
#

def test_the_solver_finds_a_planted_law(directory):
    # End to end over the real solver: plasticity_change = 2 * vip_activation + 1 must come back.
    th.Project.close()
    records = make_records(60)
    result = th.Symbolic.discover_laws(records, SCHEMA, skip=th.Symbolic.SLOW_TOOLS, timeout=60, show_progress=False)

    found = False
    for highlight in result["highlights"]:
        text = str(highlight.get("description") or highlight.get("pattern") or highlight)
        if ("plasticity_change" in text) and ("vip_activation" in text):
            found = True

    assert (found is True), "the solver did not recover the planted law; highlights were: " + str(result["highlights"][:3])
    assert (len(result["tools_run"]) > 0), "no tools ran at all"
    print("  test_the_solver_finds_a_planted_law: OK (" + str(len(result["tools_run"])) + " tools ran)")


def test_report_to_markdown_is_readable(directory):
    th.Project.close()
    result = th.Symbolic.discover_laws(make_records(40), SCHEMA, skip=th.Symbolic.SLOW_TOOLS, timeout=60, show_progress=False)

    markdown = th.Symbolic.report_to_markdown(result)
    assert ("# Symbolic solver report" in markdown), "the report has no heading"
    assert ("## Highlights" in markdown), "the report has no highlights section"
    assert ("plasticity_change" in markdown), "the report does not mention the variables"

    # It must also accept the raw report, not just the packed dict
    also = th.Symbolic.report_to_markdown(result["report"])
    assert ("# Symbolic solver report" in also), "the raw report form was not accepted"
    print("  test_report_to_markdown_is_readable: OK")


#
#   Runner
#

def main():
    print("TestSymbolic.py")
    tests = [
        test_schema_can_be_given_in_any_shape,
        test_symbolic_schemas_are_recognised,
        test_good_values_are_kept,
        test_bad_values_are_dropped_not_repaired,
        test_missing_values_are_missing_not_errors,
        test_build_table_reports_coverage_and_drops,
        test_free_text_columns_stay_out_of_the_table,
        test_empty_rows_are_left_out,
        test_a_schema_with_no_variables_raises,
        test_an_empty_table_raises_with_a_useful_message,
        test_the_solver_finds_a_planted_law,
        test_report_to_markdown_is_readable,
    ]

    for test in tests:
        directory = tempfile.mkdtemp()
        try:
            test(directory)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    print("TestSymbolic.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
