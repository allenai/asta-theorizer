# TestLLMTimeout.py
# Tests for the per-call time limit on the per-paper stages (extraction, variable discovery), and for
# what happens to a paper whose call gives up.
#
# Extraction used to call the engine with no limit of its own, so every paper got the engine's 300
# seconds whatever the library or the environment said; MAX_GENERATION_TIME_SECONDS did nothing.
# A slow model's answer then arrived after the call had been given up, and was thrown away -- and the
# paper was saved as one with no findings, indistinguishable from a paper that had none, with
# nothing in the extraction's failures.
#
# Runs offline.  The model call is replaced by a fake (ExtractionUtils._getLLMResponseJSON, which
# getLLMResponseJSON looks up by name each time it runs): no API keys, no LLM calls, no cost.
#
#     python tests/TestLLMTimeout.py

import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


ENV_KEYS = ["MAX_GENERATION_TIME_SECONDS", "LITELLM_BASE_URL", "LITELLM_USER_KEY"]

SCHEMA = [{"name": "finding", "type": "str", "description": "The finding"}]

PAPERS = [
    {"paper_id": "paper-fast", "title": "A paper the model answers at once", "markdown": "Fast paper text."},
    {"paper_id": "paper-slow", "title": "A paper the model never answers in time", "markdown": "Slow paper text."},
]

ANSWER = ({"results": [{"name_short": "F1", "brief_description": "A finding", "finding": "x"}], "follow_on_papers": []}, "```json\n{}\n```", 0.01)


#
#   Fakes
#

class FakeTime:
    # Stands in for the time module inside ExtractionUtils, so the retry backoff (0, 15, 30 s) does
    # not slow the tests down.  Everything else is the real module.
    def __getattr__(self, name):
        return getattr(time, name)

    def sleep(self, seconds):
        return None


class FixedAnswer:
    # Gives the same answer to every prompt
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.answer


class FakeModel:
    # Answers every prompt at once, except prompts containing `slow_marker`, which it holds for
    # `slow_seconds` -- longer than the limit under test.  Records every call.
    def __init__(self, slow_marker:str=None, slow_seconds:float=0.0):
        self.slow_marker = slow_marker
        self.slow_seconds = slow_seconds
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, promptStr, model, temperature=0, maxTokens=0, jsonOut=True, *rest):
        with self.lock:
            self.calls.append({"model": model, "slow": (self.slow_marker is not None) and (self.slow_marker in promptStr)})
        if (self.slow_marker is not None) and (self.slow_marker in promptStr):
            threading.Event().wait(self.slow_seconds)
        return ANSWER


class Patched:
    # Install a fake model (and the fake clock) in the engine for the length of a with-block.
    def __init__(self, fake_model):
        self.fake_model = fake_model

    def __enter__(self):
        utils = th.Legacy.ExtractionUtils
        self.saved = (utils._getLLMResponseJSON, utils.time)
        utils._getLLMResponseJSON = self.fake_model
        utils.time = FakeTime()
        return self.fake_model

    def __exit__(self, *exc):
        utils = th.Legacy.ExtractionUtils
        utils._getLLMResponseJSON, utils.time = self.saved
        return False


def recorded_limits():
    # Swap _llm_call_with_retries for one that records the limit each call was given, then runs the
    # real one.  Returns (the list it fills, a function that restores the original).
    utils = th.Legacy.ExtractionUtils
    original = utils._llm_call_with_retries
    limits = []

    def recording(llm_fn, args, fail_return, max_generation_time_seconds=None):
        limits.append(max_generation_time_seconds)
        return original(llm_fn, args, fail_return, max_generation_time_seconds=max_generation_time_seconds)

    utils._llm_call_with_retries = recording

    def restore():
        utils._llm_call_with_retries = original

    return limits, restore


def engine_function(name:str):
    # The engine's own version of a function the library replaces, read from its source file
    namespace = {}
    import ast
    with open(th.Legacy.ExtractionUtils.__file__) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)) and (node.name == name):
            code = compile(ast.Module(body=[node], type_ignores=[]), "<engine>", "exec")
            namespace.update({"os": os, "DEFAULT_GENERATION_TIME_SECONDS": th.Legacy.ExtractionUtils.DEFAULT_GENERATION_TIME_SECONDS})
            exec(code, namespace)
    return namespace[name]


def open_project(directory:str):
    th.Config.init(project_dir=directory, verbose=False)


#
#   Tests: the limit reaches the call
#

def test_the_environment_variable_sets_the_library_limit(directory):
    os.environ.pop("MAX_GENERATION_TIME_SECONDS", None)
    assert (th.Config.get_llm_timeout_seconds() == th.Config.DEFAULT_LLM_TIMEOUT_SECONDS), "without the variable the library default should apply"

    # Read at call time, so setting it after import still takes effect
    os.environ["MAX_GENERATION_TIME_SECONDS"] = "2700"
    assert (th.Config.get_llm_timeout_seconds() == 2700), "MAX_GENERATION_TIME_SECONDS was not read: " + str(th.Config.get_llm_timeout_seconds())

    os.environ["MAX_GENERATION_TIME_SECONDS"] = "not a number"
    assert (th.Config.get_llm_timeout_seconds() == th.Config.DEFAULT_LLM_TIMEOUT_SECONDS), "a bad value should fall back to the default"


def test_the_engine_default_follows_the_environment_variable(directory):
    # A src/ caller on its own (the hosted service's queues) keeps 300 seconds unless the variable is
    # set -- it used to keep 300 even then.  Exercised through the engine's own function, which the
    # library replaces (next test).
    os.environ.pop("MAX_GENERATION_TIME_SECONDS", None)
    engine_default = engine_function("default_generation_time_seconds")
    assert (engine_default() == 300), "the engine's own default changed: " + str(engine_default())
    os.environ["MAX_GENERATION_TIME_SECONDS"] = "900"
    assert (engine_default() == 900), "the engine did not read MAX_GENERATION_TIME_SECONDS"


def test_a_stage_with_no_limit_of_its_own_gets_the_library_limit(directory):
    # Planning, reports and schemas call the engine with no limit; inside the library they get the
    # library's, not the engine's 300
    utils = th.Legacy.ExtractionUtils
    os.environ.pop("MAX_GENERATION_TIME_SECONDS", None)
    assert (utils.default_generation_time_seconds() == th.Config.DEFAULT_LLM_TIMEOUT_SECONDS), utils.default_generation_time_seconds()
    os.environ["MAX_GENERATION_TIME_SECONDS"] = "2700"
    assert (utils.default_generation_time_seconds() == 2700), utils.default_generation_time_seconds()

    limits, restore = recorded_limits()
    try:
        with Patched(FakeModel()):
            utils.getLLMResponseJSON("a planning prompt", "fake-model", jsonOut=False)
    finally:
        restore()
    seen = limits
    assert (seen == [None]), "a call with no limit of its own should be left to the default: " + str(seen)


def test_extraction_passes_the_library_limit_not_300(directory):
    os.environ.pop("MAX_GENERATION_TIME_SECONDS", None)
    open_project(directory)
    limits, restore = recorded_limits()
    try:
        with Patched(FakeModel()):
            th.Pipeline.extract_from_papers(PAPERS, "what was found", SCHEMA, model_str="fake-model", max_workers=2, show_progress=False)
    finally:
        restore()

    assert (len(limits) == 2), "expected one call per paper, got " + str(limits)
    assert (all(limit == th.Config.DEFAULT_LLM_TIMEOUT_SECONDS for limit in limits)), "extraction did not get the library limit: " + str(limits)


def test_an_explicit_limit_beats_the_environment(directory):
    os.environ["MAX_GENERATION_TIME_SECONDS"] = "2700"
    open_project(directory)
    limits, restore = recorded_limits()
    try:
        with Patched(FakeModel()):
            packed = th.Pipeline.extract_from_papers(PAPERS, "what was found", SCHEMA, model_str="fake-model", max_workers=2, show_progress=False, llm_timeout_seconds=42)
            th.Pipeline.discover_variables(PAPERS, {"theory_query": "q"}, model_str="fake-model", max_workers=2, show_progress=False, llm_timeout_seconds=43)
    finally:
        restore()

    assert (limits.count(42) == 2) and (limits.count(43) == 2), "the explicit limits did not reach the calls: " + str(limits)
    assert (packed["llm_timeout_seconds"] == 42), "the saved extraction should say which limit it ran with"


def test_a_slow_answer_inside_the_limit_is_kept(directory):
    # The defect in one line: an answer that takes longer than 300 seconds but less than the limit
    # must be used, not dropped.  Scaled down: the slow paper takes 1.5 s against a 3 s limit.
    open_project(directory)
    with Patched(FakeModel(slow_marker="Slow paper text", slow_seconds=1.5)):
        packed = th.Pipeline.extract_from_papers(PAPERS, "what was found", SCHEMA, model_str="fake-model", max_workers=2, show_progress=False, llm_timeout_seconds=3)

    assert (packed["num_papers_with_records"] == 2), "the slow paper's answer was dropped: " + str(packed["num_papers_with_records"])


#
#   Tests: a paper whose call gives up is reported, not saved as empty
#

def test_a_timed_out_paper_is_a_failure_not_a_paper_with_no_findings(directory):
    open_project(directory)
    with Patched(FakeModel(slow_marker="Slow paper text", slow_seconds=2.5)):
        packed = th.Pipeline.extract_from_papers(PAPERS, "what was found", SCHEMA, model_str="fake-model", max_workers=2, show_progress=False, llm_timeout_seconds=1)

    ids = [result["paper_id"] for result in packed["results"]]
    assert (ids == ["paper-fast"]), "the timed-out paper should not be among the results: " + str(ids)
    assert (len(packed["failures"]) == 1), "the timed-out paper should be a failure: " + str(packed["failures"])
    failure = packed["failures"][0]
    assert (failure["paper_id"] == "paper-slow"), failure
    assert (failure["reason"] == "timeout"), "the failure should say it timed out: " + str(failure)
    assert ("1 s" in failure["error"]), "the failure should say what the limit was: " + str(failure["error"])
    assert ("markdown" not in str(failure)), "a failure should not carry the paper's full text"
    assert (packed["num_failures_by_reason"] == {"timeout": 1}), packed["num_failures_by_reason"]

    # The saved stage says so too
    import json
    saved = [name for name in os.listdir(directory) if name.endswith("-extraction.json")]
    with open(os.path.join(directory, saved[0])) as f:
        on_disk = json.load(f)
    assert (on_disk["failures"][0]["reason"] == "timeout"), "the saved extraction does not record the timeout"


def test_a_rerun_tries_the_timed_out_paper_again(directory):
    # The checkpoint used to record the empty result as done, so a rerun never retried the paper
    open_project(directory)
    with Patched(FakeModel(slow_marker="Slow paper text", slow_seconds=2.5)):
        th.Pipeline.extract_from_papers(PAPERS, "what was found", SCHEMA, model_str="fake-model", max_workers=2, show_progress=False, llm_timeout_seconds=1)

    with Patched(FakeModel()) as fake:
        packed = th.Pipeline.extract_from_papers(PAPERS, "what was found", SCHEMA, model_str="fake-model", max_workers=2, show_progress=False, llm_timeout_seconds=1)

    assert (len(fake.calls) == 1), "the rerun should call the model for the failed paper only: " + str(fake.calls)
    assert (packed["num_papers_with_records"] == 2) and (len(packed["failures"]) == 0), "the rerun should finish both papers"


def test_an_unreadable_answer_is_a_failure(directory):
    open_project(directory)
    with Patched(FixedAnswer((None, "no json here", 0.02))):
        packed = th.Pipeline.extract_from_papers(PAPERS[:1], "what was found", SCHEMA, model_str="fake-model", max_workers=1, show_progress=False)

    assert (len(packed["failures"]) == 1) and (packed["failures"][0]["reason"] == "unreadable_answer"), packed["failures"]


def test_a_paper_with_nothing_to_extract_is_not_a_failure(directory):
    # The model's off-ramp: it read the paper and found nothing relevant
    open_project(directory)
    with Patched(FixedAnswer(({"results": [], "follow_on_papers": []}, "```json\n{}\n```", 0.01))):
        packed = th.Pipeline.extract_from_papers(PAPERS[:1], "what was found", SCHEMA, model_str="fake-model", max_workers=1, show_progress=False)

    assert (len(packed["failures"]) == 0), "an answer with no findings is not a failure: " + str(packed["failures"])
    assert (len(packed["results"]) == 1) and (packed["num_papers_with_records"] == 0), packed


def test_a_model_error_is_a_failure_with_its_message(directory):
    open_project(directory)

    def broken(*args):
        raise RuntimeError("BadRequestError: model not served")

    with Patched(broken):
        packed = th.Pipeline.extract_from_papers(PAPERS[:1], "what was found", SCHEMA, model_str="fake-model", max_workers=1, show_progress=False)

    failure = packed["failures"][0]
    assert (failure["reason"] == "error") and ("model not served" in failure["error"]), failure


def test_discovery_reports_a_timeout_too(directory):
    open_project(directory)
    with Patched(FakeModel(slow_marker="Slow paper text", slow_seconds=2.5)):
        packed = th.Pipeline.discover_variables(PAPERS, {"theory_query": "q"}, model_str="fake-model", max_workers=2, show_progress=False, llm_timeout_seconds=1)

    assert ([r["paper_id"] for r in packed["results"]] == ["paper-fast"]), packed["results"]
    assert (packed["failures"][0]["reason"] == "timeout"), packed["failures"]


def main():
    print("TestLLMTimeout.py")
    tests = [
        test_the_environment_variable_sets_the_library_limit,
        test_the_engine_default_follows_the_environment_variable,
        test_a_stage_with_no_limit_of_its_own_gets_the_library_limit,
        test_extraction_passes_the_library_limit_not_300,
        test_an_explicit_limit_beats_the_environment,
        test_a_slow_answer_inside_the_limit_is_kept,
        test_a_timed_out_paper_is_a_failure_not_a_paper_with_no_findings,
        test_a_rerun_tries_the_timed_out_paper_again,
        test_an_unreadable_answer_is_a_failure,
        test_a_paper_with_nothing_to_extract_is_not_a_failure,
        test_a_model_error_is_a_failure_with_its_message,
        test_discovery_reports_a_timeout_too,
    ]

    saved_env = {key: os.environ.get(key) for key in ENV_KEYS}
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    try:
        for test in tests:
            directory = tempfile.mkdtemp()
            try:
                test(directory)
                print("  " + test.__name__ + ": OK")
            finally:
                shutil.rmtree(directory, ignore_errors=True)
    finally:
        for key, value in saved_env.items():
            if (value is None):
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("TestLLMTimeout.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
