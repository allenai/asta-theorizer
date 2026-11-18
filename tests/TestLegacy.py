# TestLegacy.py
# Tests for theorizer/Legacy.py -- that every name the library binds to actually exists in src/.
#
# This is the test that matters most for maintenance.  Legacy.py is a list of pointers into
# human-authored code that will keep changing; if a function there is renamed or its arguments
# change, the library breaks somewhere deep inside a paid run.  This catches it in a second, offline.
#
#     python tests/TestLegacy.py

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


# Every name the library depends on, and the arguments the wrappers pass by keyword.  If a wrapper
# passes an argument that no longer exists, that is a break, and it is listed here.
EXPECTED_CALLABLES = {
    "plan_query_and_schema": ["max_tokens", "temperature", "use_reflection", "mission_statement"],
    "generate_schemas_from_variable_discovery": ["max_tokens", "temperature", "use_reflection"],
    "generate_symbolic_schema": ["max_tokens", "temperature", "use_reflection"],
    "generate_variable_report": ["max_tokens", "temperature", "use_reflection"],
    "subsample_records": ["extracted_results_in", "max_tokens"],
    "build_theory_accuracy_focused": ["mission_statement", "model_str", "max_tokens", "temperature", "use_reflection", "max_laws_per_theory", "additional_suggestions"],
    "build_theory_novelty_focused": ["model_str", "max_tokens", "temperature", "use_reflection", "max_laws_per_theory"],
    "extract_entities_from_paper": ["model_str", "max_tokens", "temperature", "use_reflection", "max_generation_time_seconds"],
    "discover_variables_from_paper": ["additional_contextual_info", "model_str", "max_tokens", "temperature", "use_reflection", "max_generation_time_seconds"],
    "count_tokens": [],
    "get_llm_response_json": [],
    "load_s2_api_key": [],
}

EXPECTED_CLASSES = [
    "TheoryStore", "Theory", "ExtractionQuerySchema", "ExtractionQueryResult", "TheoryEvaluation",
    "PaperStore", "PaperStatus", "MistralOCRStore", "SemanticScholar", "Throttle",
    "SchemaExtractionQueue", "VariableDiscoveryQueue", "FilePersistence", "PersistenceBackend",
]


#
#   Tests
#

def test_every_bound_name_exists():
    missing = []
    for name in list(EXPECTED_CALLABLES.keys()) + EXPECTED_CLASSES:
        if (not hasattr(th.Legacy, name)):
            missing.append(name)

    assert (len(missing) == 0), "Legacy.py binds names that do not exist: " + ", ".join(missing)
    print("  test_every_bound_name_exists: OK (" + str(len(EXPECTED_CALLABLES) + len(EXPECTED_CLASSES)) + " names)")


def test_bound_names_are_the_right_kind_of_thing():
    for name in EXPECTED_CALLABLES:
        value = getattr(th.Legacy, name)
        assert (callable(value)), name + " is bound to something that is not callable"

    for name in EXPECTED_CLASSES:
        value = getattr(th.Legacy, name)
        assert (inspect.isclass(value)), name + " is bound to something that is not a class"

    print("  test_bound_names_are_the_right_kind_of_thing: OK")


def test_keyword_arguments_the_wrappers_pass_still_exist():
    # The failure this catches: a src/ function is renamed or loses a parameter, and a stage that
    # passes it by keyword dies partway through a run that has already cost money.
    problems = []
    for name, keywords in EXPECTED_CALLABLES.items():
        if (len(keywords) == 0):
            continue

        function = getattr(th.Legacy, name)
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            continue

        # A function taking **kwargs accepts anything, so there is nothing to check
        accepts_anything = False
        for parameter in signature.parameters.values():
            if (parameter.kind == inspect.Parameter.VAR_KEYWORD):
                accepts_anything = True
        if (accepts_anything):
            continue

        for keyword in keywords:
            if (keyword not in signature.parameters):
                problems.append(name + "() no longer takes '" + keyword + "'")

    assert (len(problems) == 0), "the library passes arguments that src/ no longer accepts:\n    " + "\n    ".join(problems)
    print("  test_keyword_arguments_the_wrappers_pass_still_exist: OK")


def test_verbose_flag_is_present_in_extraction_utils():
    # Config.set_quiet() works by setting these.  If they are ever removed from src/, quiet mode
    # silently stops working rather than failing, so it is worth asserting.
    assert (hasattr(th.Legacy.ExtractionUtils, "VERBOSE")), "ExtractionUtils has no VERBOSE flag"
    assert (hasattr(th.Legacy.ExtractionUtils, "VERBOSE_LOG_FILENAME")), "ExtractionUtils has no VERBOSE_LOG_FILENAME"
    assert (hasattr(th.Legacy.ExtractionUtils, "printVerbose")), "ExtractionUtils has no printVerbose()"

    original = th.Legacy.ExtractionUtils.VERBOSE
    try:
        th.Config.set_quiet(True)
        assert (th.Legacy.ExtractionUtils.VERBOSE is False), "set_quiet(True) did not turn VERBOSE off"
        th.Config.set_quiet(False)
        assert (th.Legacy.ExtractionUtils.VERBOSE is True), "set_quiet(False) did not turn VERBOSE on"
    finally:
        th.Legacy.ExtractionUtils.VERBOSE = original

    print("  test_verbose_flag_is_present_in_extraction_utils: OK")


def test_every_pipeline_stage_is_callable():
    stages = ["plan_theory_query", "extract_from_papers", "flatten_records", "discover_variables",
              "make_variable_report", "make_schemas_from_variable_discovery", "extend_schema_with_suggestions",
              "collect_proposed_slots", "make_symbolic_schema", "fetch_papers",
              "subsample", "generate_theories"]

    for name in stages:
        assert (hasattr(th.Pipeline, name)), "Pipeline has no " + name + "()"
        assert (callable(getattr(th.Pipeline, name))), name + " is not callable"

    print("  test_every_pipeline_stage_is_callable: OK (" + str(len(stages)) + " stages)")


#
#   Runner
#

def main():
    print("TestLegacy.py")
    tests = [
        test_every_bound_name_exists,
        test_bound_names_are_the_right_kind_of_thing,
        test_keyword_arguments_the_wrappers_pass_still_exist,
        test_verbose_flag_is_present_in_extraction_utils,
        test_every_pipeline_stage_is_callable,
    ]

    for test in tests:
        test()

    print("TestLegacy.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
