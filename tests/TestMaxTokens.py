# TestMaxTokens.py
# The completion budget for the per-paper stages (extraction, variable discovery), and what a paper
# whose answer was cut off at that budget is reported as.
#
# At reasoning effort high a reasoning model counts its thinking against max_tokens.  At the old
# default of 32000 it sometimes spent all of it thinking and wrote nothing (8 of 300 calls on a
# 100-paper run), and the paper was reported as "unreadable_answer" -- "the model answered, but no
# JSON could be read" -- which points at the parser, not at the budget.  Now:
#   - the default is Config.DEFAULT_MAX_TOKENS_PER_PAPER (64000), capped at the model's own limit
#   - an answer cut off at max_tokens is reason "max_tokens", and says to raise it
#
# Runs offline.  litellm's completion() is replaced by a fake inside the engine, so the engine's
# own handling of the response runs; no API keys, no LLM calls, no cost.
#
#     python tests/TestMaxTokens.py

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th

from litellm import ModelResponse


SCHEMA = [{"name": "finding", "type": "str", "description": "The finding"}]
PAPER = {"paper_id": "paper-dense", "title": "A dense paper", "markdown": "Dense paper text."}
GOOD = '```json\n{"results": [{"name_short": "F1", "name_full": "Finding one", "brief_description": "A finding", "finding": "x"}], "follow_on_papers": []}\n```'


class FakeTime:
    def __getattr__(self, name):
        return getattr(time, name)

    def sleep(self, seconds):
        return None


def response(content, finish_reason, completion_tokens=100, reasoning_tokens=0):
    r = ModelResponse(choices=[{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
                      usage={"prompt_tokens": 50, "completion_tokens": completion_tokens, "total_tokens": 50 + completion_tokens,
                             "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}})
    r._hidden_params["response_cost"] = 0.001
    return r


class FakeCompletion:
    # Stands in for litellm.completion inside the engine; records the budget each call asked for
    def __init__(self, make):
        self.make = make
        self.budgets = []

    def __call__(self, **kwargs):
        self.budgets.append(kwargs.get("max_completion_tokens", kwargs.get("max_tokens")))
        return self.make(kwargs)


class Patched:
    def __init__(self, fake):
        self.fake = fake

    def __enter__(self):
        utils = th.Legacy.ExtractionUtils
        self.saved = (utils.completion, utils.time)
        utils.completion = self.fake
        utils.time = FakeTime()
        return self.fake

    def __exit__(self, *exc):
        utils = th.Legacy.ExtractionUtils
        utils.completion, utils.time = self.saved
        return False


def open_project(directory):
    th.Config.init(project_dir=directory, verbose=False)


#
#   Tests: the reason
#

def test_an_answer_spent_entirely_on_reasoning_is_max_tokens(directory):
    open_project(directory)
    with Patched(FakeCompletion(lambda kw: response("", "length", 64000, 64000))):
        packed = th.Pipeline.extract_from_papers([PAPER], "what was found", SCHEMA, max_workers=1, show_progress=False)

    failure = packed["failures"][0]
    assert (failure["reason"] == "max_tokens"), failure
    assert ("max_tokens=64000" in failure["error"]) and ("64000 of them reasoning" in failure["error"]), failure["error"]
    assert ("Raise max_tokens" in failure["error"]), failure["error"]
    assert (packed["num_failures_by_reason"] == {"max_tokens": 1}), packed["num_failures_by_reason"]


def test_an_answer_cut_off_mid_json_is_max_tokens(directory):
    open_project(directory)
    with Patched(FakeCompletion(lambda kw: response('```json\n{"results": [{"name_short": "F1", "brief', "length", 64000, 30000))):
        packed = th.Pipeline.extract_from_papers([PAPER], "what was found", SCHEMA, max_workers=1, show_progress=False)

    assert (packed["failures"][0]["reason"] == "max_tokens"), packed["failures"]


def test_a_finished_answer_with_no_json_is_still_unreadable(directory):
    open_project(directory)
    with Patched(FakeCompletion(lambda kw: response("I could not find anything to put in JSON.", "stop"))):
        packed = th.Pipeline.extract_from_papers([PAPER], "what was found", SCHEMA, max_workers=1, show_progress=False)

    assert (packed["failures"][0]["reason"] == "unreadable_answer"), packed["failures"]


def test_a_truncated_answer_that_still_parses_is_not_a_failure(directory):
    # finish_reason 'length' with a complete JSON block: the answer is usable, so it is used
    open_project(directory)
    with Patched(FakeCompletion(lambda kw: response(GOOD, "length", 64000, 1000))):
        packed = th.Pipeline.extract_from_papers([PAPER], "what was found", SCHEMA, max_workers=1, show_progress=False)

    assert (len(packed["failures"]) == 0) and (packed["num_papers_with_records"] == 1), packed["failures"]


def test_discovery_reports_max_tokens_too(directory):
    open_project(directory)
    with Patched(FakeCompletion(lambda kw: response("", "length", 64000, 64000))):
        packed = th.Pipeline.discover_variables([PAPER], {"theory_query": "q"}, max_workers=1, show_progress=False)

    assert (packed["failures"][0]["reason"] == "max_tokens"), packed["failures"]


#
#   Tests: the budget
#

def test_the_default_budget_is_64000(directory):
    open_project(directory)
    with Patched(FakeCompletion(lambda kw: response(GOOD, "stop"))) as fake:
        th.Pipeline.extract_from_papers([PAPER], "what was found", SCHEMA, max_workers=1, show_progress=False)
        th.Pipeline.discover_variables([PAPER], {"theory_query": "q"}, max_workers=1, show_progress=False)

    assert (th.Config.DEFAULT_MAX_TOKENS_PER_PAPER == 64000)
    assert (fake.budgets) and all(b == 64000 for b in fake.budgets), fake.budgets


def test_the_budget_is_capped_at_the_models_own_limit(directory):
    open_project(directory)
    assert (th.Config.cap_output_tokens("gpt-4.1", 64000, show=False) == 32768)
    assert (th.Config.cap_output_tokens("gpt-4o", 64000, show=False) == 16384)
    assert (th.Config.cap_output_tokens("openai/gpt-4.1", 64000, show=False) == 32768)
    assert (th.Config.cap_output_tokens("gpt-4.1", 8000, show=False) == 8000)
    assert (th.Config.cap_output_tokens("a-model-litellm-does-not-know", 64000, show=False) == 64000)

    with Patched(FakeCompletion(lambda kw: response(GOOD, "stop"))) as fake:
        th.Pipeline.extract_from_papers([PAPER], "what was found", SCHEMA, model_str="gpt-4.1", max_workers=1, show_progress=False)
    assert (fake.budgets == [32768]), fake.budgets


def test_an_explicit_budget_is_used(directory):
    open_project(directory)
    with Patched(FakeCompletion(lambda kw: response(GOOD, "stop"))) as fake:
        th.Pipeline.extract_from_papers([PAPER], "what was found", SCHEMA, max_tokens=20000, max_workers=1, show_progress=False)
    assert (fake.budgets == [20000]), fake.budgets


def main():
    print("TestMaxTokens.py")
    tests = [
        test_an_answer_spent_entirely_on_reasoning_is_max_tokens,
        test_an_answer_cut_off_mid_json_is_max_tokens,
        test_a_finished_answer_with_no_json_is_still_unreadable,
        test_a_truncated_answer_that_still_parses_is_not_a_failure,
        test_discovery_reports_max_tokens_too,
        test_the_default_budget_is_64000,
        test_the_budget_is_capped_at_the_models_own_limit,
        test_an_explicit_budget_is_used,
    ]
    saved = {k: os.environ.pop(k, None) for k in ("LITELLM_BASE_URL", "LITELLM_USER_KEY", "OPENAI_API_KEY")}
    os.environ["OPENAI_API_KEY"] = "sk-test-not-a-real-key"
    try:
        for test in tests:
            directory = tempfile.mkdtemp()
            try:
                test(directory)
                print("  " + test.__name__ + ": OK")
            finally:
                shutil.rmtree(directory, ignore_errors=True)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if (v is not None):
                os.environ[k] = v

    print("TestMaxTokens.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
