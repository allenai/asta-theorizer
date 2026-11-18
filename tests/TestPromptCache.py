# TestPromptCache.py
# Tests for the prompt-cache markers in src/ExtractionUtils.py -- which provider gets which.
#
# A generation pass re-sends the whole evidence prefix once per theory and again per reflection, so
# the prefix has to be marked as cacheable or it is paid for in full every time.  The two providers
# read different markers, and sending the wrong one is not a loud failure: OpenAI's marker is
# ignored by Anthropic (silently uncached, 5-10x the price on a large run), while OpenAI's
# prompt_cache_options rode in extra_body, which Anthropic rejects outright with a 400 that killed
# every theory generation on a Claude model.
#
#     python tests/TestPromptCache.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th                                   # noqa: F401  (sets up the engine import path)
import ExtractionUtils as extraction_utils


PROMPT = "EVIDENCE PREFIX, shared by every call in the pass.\n\nTHE PART THAT VARIES"
PREFIX = "EVIDENCE PREFIX, shared by every call in the pass.\n\n"


def captured_call(model:str):
    # Run _getLLMResponseJSONMaybeCached far enough to see the request it would send, without a
    # network call: litellm.completion is replaced by one that records its arguments and raises.
    seen = {}

    class Captured(Exception):
        pass

    def fake_completion(**kwargs):
        seen.update(kwargs)
        raise Captured()

    real_completion = extraction_utils.completion
    cwd = os.getcwd()
    extraction_utils.completion = fake_completion
    try:
        os.chdir(os.environ["TEST_PROMPT_CACHE_DIR"])      # the function dumps prompts/ beside cwd
        try:
            extraction_utils._getLLMResponseJSONMaybeCached(PROMPT, model, 0.0, 1000, False, PREFIX, "test-key")
        except Captured:
            pass
    finally:
        extraction_utils.completion = real_completion
        os.chdir(cwd)

    return seen


def prefix_block(seen:dict):
    blocks = seen["messages"][0]["content"]
    assert isinstance(blocks, list), "the prompt should be split into blocks when a prefix is given"
    assert len(blocks) == 2, "expected a prefix block and a suffix block, got " + str(len(blocks))
    assert blocks[0]["text"] == PREFIX, "the first block should be exactly the cache prefix"
    return blocks[0]


def test_anthropic_gets_cache_control():
    seen = captured_call("claude-opus-5-5")
    block = prefix_block(seen)
    assert block.get("cache_control") == {"type": "ephemeral"}, "Anthropic needs cache_control on the prefix block, got " + str(block)
    assert "prompt_cache_breakpoint" not in block, "OpenAI's marker buys nothing on Anthropic"
    print("  test_anthropic_gets_cache_control: PASSED")


def test_anthropic_is_sent_no_extra_body():
    # The 400 that killed every Claude generation: "extra_body: Extra inputs are not permitted".
    seen = captured_call("claude-opus-5-5")
    assert "extra_body" not in seen, "Anthropic rejects the whole request when extra_body is present"
    assert "prompt_cache_key" not in seen, "prompt_cache_key is an OpenAI argument"
    print("  test_anthropic_is_sent_no_extra_body: PASSED")


def test_openai_keeps_its_breakpoint_and_extra_body():
    seen = captured_call("gpt-6-sol")
    block = prefix_block(seen)
    assert block.get("prompt_cache_breakpoint") == {"mode": "explicit"}, "OpenAI's prefix marker went missing, got " + str(block)
    assert "cache_control" not in block, "cache_control is Anthropic's marker"
    assert seen.get("extra_body", {}).get("prompt_cache_options") == {"mode": "explicit"}, "OpenAI's caching rides in extra_body"
    assert seen.get("prompt_cache_key") == "test-key"
    print("  test_openai_keeps_its_breakpoint_and_extra_body: PASSED")


def test_no_prefix_means_one_plain_block():
    seen = {}

    class Captured(Exception):
        pass

    def fake_completion(**kwargs):
        seen.update(kwargs)
        raise Captured()

    real_completion = extraction_utils.completion
    cwd = os.getcwd()
    extraction_utils.completion = fake_completion
    try:
        os.chdir(os.environ["TEST_PROMPT_CACHE_DIR"])
        try:
            extraction_utils._getLLMResponseJSONMaybeCached(PROMPT, "claude-opus-5-5", 0.0, 1000, False, None, None)
        except Captured:
            pass
    finally:
        extraction_utils.completion = real_completion
        os.chdir(cwd)

    blocks = seen["messages"][0]["content"]
    assert len(blocks) == 1, "with no prefix there is nothing to cache, so one block"
    assert "cache_control" not in blocks[0]
    assert "extra_body" not in seen
    print("  test_no_prefix_means_one_plain_block: PASSED")


def main():
    import tempfile

    tests = [
        test_anthropic_gets_cache_control,
        test_anthropic_is_sent_no_extra_body,
        test_openai_keeps_its_breakpoint_and_extra_body,
        test_no_prefix_means_one_plain_block,
    ]

    with tempfile.TemporaryDirectory() as scratch:
        os.environ["TEST_PROMPT_CACHE_DIR"] = scratch
        try:
            for test in tests:
                test()
        finally:
            os.environ.pop("TEST_PROMPT_CACHE_DIR", None)

    print("TestPromptCache.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
