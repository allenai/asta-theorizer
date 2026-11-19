# TestGenerationTimeout.py
# Tests for the time limit on theory generation and the other once-per-run stages, and for what a
# stage records when one of its model calls gives up.
#
# Theory generation named a fixed limit for each call (15 minutes for the default and attribution
# generators, 8 for novelty), which no setting reached.  A 169k-token generation prompt on a slow
# model ran past the 15 minutes, was abandoned (still billed) and retried, and all anyone saw was
# the engine's "LLM Generation timed out" inside the retry loop.  Now every such limit follows the
# library's setting -- MAX_GENERATION_TIME_SECONDS, or llm_timeout_seconds= for one
# stage -- and is never shorter than the engine's own value unless a setting asks for it.
#
# Runs offline.  The model is the sleeping fake from TestLLMTimeout.py: no API keys, no cost.
#
#     python tests/TestGenerationTimeout.py

import json
import os
import re
import shutil
import sys
import tempfile

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(TESTS_DIR))
sys.path.insert(0, TESTS_DIR)

import theorizer as th
from TestLLMTimeout import FakeModel, Patched, engine_function


ENV_KEYS = ["MAX_GENERATION_TIME_SECONDS", "LITELLM_BASE_URL", "LITELLM_USER_KEY"]

RECORDS = [{"uuid": "e1.0", "paper_id": "paper-a", "name_short": "F1", "brief_description": "VIP activity enables plasticity"}]


#
#   Tests: the limit
#

def test_a_fixed_engine_limit_follows_the_setting(directory):
    utils = th.Legacy.ExtractionUtils
    os.environ.pop("MAX_GENERATION_TIME_SECONDS", None)
    # Unset: the longer of the engine's value and the library default -- never shorter than before
    assert (utils.generation_time_seconds(900) == max(900, th.Config.DEFAULT_LLM_TIMEOUT_SECONDS)), utils.generation_time_seconds(900)
    assert (utils.generation_time_seconds(480) == max(480, th.Config.DEFAULT_LLM_TIMEOUT_SECONDS)), utils.generation_time_seconds(480)

    os.environ["MAX_GENERATION_TIME_SECONDS"] = "2700"
    assert (utils.generation_time_seconds(900) == 2700) and (utils.generation_time_seconds(480) == 2700), "the variable should set every generation call"

    with th.Config.llm_timeout(42):
        assert (utils.generation_time_seconds(900) == 42), "the stage's setting should beat the variable"
    assert (utils.generation_time_seconds(900) == 2700), "the stage's setting should end with the stage"

    os.environ["MAX_GENERATION_TIME_SECONDS"] = "300"
    assert (utils.generation_time_seconds(900) == 300), "a setting shorter than the engine's value is still the person's choice"


def test_the_engine_on_its_own_keeps_its_fixed_values(directory):
    # The hosted service imports src/ without the library: its limits change only if it sets the variable
    engine = engine_function("_engine_generation_time_seconds")
    os.environ.pop("MAX_GENERATION_TIME_SECONDS", None)
    assert (engine(900) == 900) and (engine(480) == 480), "the engine's own values changed"
    os.environ["MAX_GENERATION_TIME_SECONDS"] = "2400"
    assert (engine(900) == 2400), "the engine should honour the variable"


def test_no_engine_call_keeps_a_fixed_limit(directory):
    # Every limit theory generation and the evaluations name goes through generation_time_seconds()
    src = th.Legacy.SRC_DIR
    with open(os.path.join(src, "TheorizerProcessing.py")) as f:
        processing = f.read()
    constants = re.findall(r"MAX_GENERATION_TIME_SECS = (.*)", processing)
    assert (len(constants) == 8), "expected 8 per-call limits in TheorizerProcessing.py, found " + str(len(constants))
    for value in constants:
        assert (value.startswith("generation_time_seconds(")), "a fixed limit is left: " + value
        assert ("8 minutes" not in value) or ("60 * 8" in value), "a stale comment: " + value
    for name in ("TheorizerProcessing", "EvaluationCommon", "EvaluationLLMAsAJudge", "EvaluationQualifiedNovelty", "EvaluationSurprisal"):
        with open(os.path.join(src, name + ".py")) as f:
            literal = re.findall(r"max_generation_time_seconds=\d+", f.read())
        assert (len(literal) == 0), name + ".py passes a fixed limit: " + str(literal)


def test_a_generation_stage_passes_its_setting_to_the_call(directory):
    th.Config.init(project_dir=directory, verbose=False)
    utils = th.Legacy.ExtractionUtils
    original = utils._llm_call_with_retries
    limits = []

    def recording(llm_fn, args, fail_return, max_generation_time_seconds=None):
        limits.append(max_generation_time_seconds)
        return fail_return

    utils._llm_call_with_retries = recording
    try:
        with Patched(FakeModel()):
            packed = th.Pipeline.generate_theories(RECORDS, "how VIP interneurons set plasticity", model_str="fake-model", use_reflection=False, llm_timeout_seconds=1234)
    finally:
        utils._llm_call_with_retries = original

    assert (limits[:1] == [1234]), "the generation call did not get the stage's limit: " + str(limits)
    assert (packed["llm_timeout_seconds"] == 1234), "the stage should record its limit"


#
#   Tests: a call that gives up is reported
#

def test_a_generation_call_that_gives_up_is_reported(directory):
    th.Config.init(project_dir=directory, verbose=False)
    with Patched(FakeModel(slow_marker="", slow_seconds=2.5)):
        packed = th.Pipeline.generate_theories(records=RECORDS, theory_query="how VIP interneurons set plasticity", model_str="fake-model", use_reflection=False, llm_timeout_seconds=1)

    failures = packed["call_failures"]
    assert (len(failures) == 1), failures
    failure = failures[0]
    assert (failure["reason"] == "timeout") and (failure["max_generation_time_seconds"] == 1), failure
    assert (failure["caller"] == "build_theory_from_results_single_theory_reflection4"), "which call gave up: " + str(failure["caller"])
    assert (failure["model"] == "fake-model") and (failure["prompt_tokens"] > 0), failure

    saved = [name for name in os.listdir(directory) if name.endswith("-theories.json")]
    with open(os.path.join(directory, saved[0])) as f:
        on_disk = json.load(f)
    assert (on_disk["llm_timeout_seconds"] == 1) and (on_disk["call_failures"][0]["reason"] == "timeout"), "the saved stage should record the limit and the call"


def main():
    print("TestGenerationTimeout.py")
    tests = [
        test_a_fixed_engine_limit_follows_the_setting,
        test_the_engine_on_its_own_keeps_its_fixed_values,
        test_no_engine_call_keeps_a_fixed_limit,
        test_a_generation_stage_passes_its_setting_to_the_call,
        test_a_generation_call_that_gives_up_is_reported,
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
                os.environ.pop("MAX_GENERATION_TIME_SECONDS", None)
    finally:
        for key, value in saved_env.items():
            if (value is None):
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("TestGenerationTimeout.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
