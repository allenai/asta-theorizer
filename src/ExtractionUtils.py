# ExtractionUtils.py
# This is a set of wrappers for LiteLLM to assist with LLM interactions, including cost tracking, retries, timeouts, json parsing, etc.  Wrappers are somewhat rough-and-ready, cost and token tracking may be approximate.

import os
import json
import time
import traceback
import random
import threading

from litellm import completion
from litellm import embedding

import tiktoken



# Cost Estimates. Note that these are just estimates and may not be accurate -- you should ideally have other methods (like a hard limit on your account) to help limit unexpected costs.
TOTAL_LLM_COST = 0.0            # Running (estimated) cost of all LLM queries so far. This is a global variable that is updated with each query.

DEFAULT_MAX_TOKENS = 32000

# Full-response debug dump.  The per-query status lines (model, running cost, prompt tokens, completed cost) ALWAYS print -- they're how you can see a run progressing.  This flag covers only the full ModelResponse dump below, which is many lines per call; when VERBOSE is False it's written to VERBOSE_LOG_FILENAME instead, so nothing is lost.
VERBOSE = True
VERBOSE_LOG_FILENAME = "theorizer-verbose.log"
PROMPTS_DIRECTORY = "prompts"            # where every prompt and response is dumped (one file each)
VERBOSE_LOCK = threading.Lock()         # Status messages come from many worker threads at once


#
#   Helper: Full-response debug dump
#

# Print the debug dump, either to the console (VERBOSE) or to the verbose log file (not VERBOSE).
def printVerbose(strIn):
    if (VERBOSE):
        print(strIn)
        return

    with VERBOSE_LOCK:
        try:
            with open(VERBOSE_LOG_FILENAME, "a") as f:
                f.write(str(strIn) + "\n")
        except Exception as e:
            print("printVerbose(): ERROR: Could not write to the verbose log (" + str(e) + ")")


#
#   Helper: Counting tokens
#
# Use TikToken to measure the number of tokens in an input string
tiktokenEncoder = tiktoken.encoding_for_model("gpt-4")
def countTokens(inputStr:str):
    tokens = tiktokenEncoder.encode(inputStr, disallowed_special=())
    return len(tokens)


def tokenize(inputStr:str):
    tokens = tiktokenEncoder.encode(inputStr, disallowed_special=())
    # Convert each token to it's respective string, keeping the output as a list
    tokens = [tiktokenEncoder.decode([token]) for token in tokens]
    return tokens


def truncate_to_model_context(text:str, model_str:str, headroom:int=35000) -> str:
    """Truncate text to fit model_str's input window, reserving headroom for prompt + completion."""
    import litellm
    # litellm.get_max_tokens() reports the model's maximum OUTPUT length, not its input window.  Using
    # it here truncated to a fraction of what actually fits (93k rather than 272k on gpt-5-mini, or
    # 1.05M on gpt-5.x), and went NEGATIVE for any model whose output cap is below the headroom --
    # gpt-4o-mini gave 16384-35000, which silently emptied every paper shorter than 18,616 tokens.
    # max_input_tokens is the correct field; get_max_tokens is only a fallback when it is missing.
    # Note get_max_tokens() raises for a model it does not know, so the fallback has to be guarded --
    # otherwise an unmapped model is an exception here rather than a sensible default.
    limit = (litellm.model_cost.get(model_str) or {}).get("max_input_tokens")
    if (limit is None):
        try:
            limit = litellm.get_max_tokens(model_str) or 200000
        except Exception:
            print("truncate_to_model_context(): WARNING: no context length known for " + str(model_str) + ", assuming 200000 tokens.", flush=True)
            limit = 200000
    max_tokens = max(limit - headroom, 1000)
    if countTokens(text) <= max_tokens:
        return text
    return tiktokenEncoder.decode(tiktokenEncoder.encode(text, disallowed_special=())[:max_tokens])


#
#   Helper: Load the API keys
#

def loadAPIKeys():
    # Fills in keys from `api_keys.donotcommit.json` in the working directory.  A key already in
    # the environment wins; the file only fills in the ones that are missing.
    print("Loading API keys...")

    API_KEY_FILE = "api_keys.donotcommit.json"
    KEY_ENV_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
                    "openrouter": "OPENROUTER_API_KEY", "togetherai": "TOGETHERAI_API_KEY", "mistral": "MISTRAL_API_KEY"}
    try:
        with open(API_KEY_FILE, 'r') as fileIn:
            apiKeys = json.load(fileIn)
    except Exception as e:
        print("Note: no readable API key file (`api_keys.donotcommit.json`) here; using the keys in the environment.")
        return

    for name, env_var in KEY_ENV_VARS.items():
        if (apiKeys.get(name)) and (not os.environ.get(env_var)):
            os.environ[env_var] = apiKeys[name]
    if (os.environ.get("MISTRAL_API_KEY")):
        print("Mistral API key: present")


#
#   Helper: Get the total cost of all LLM queries
#
def getTotalLLMCost():
    global TOTAL_LLM_COST
    return TOTAL_LLM_COST



#
#   Helper: recovering a response whose JSON does not parse
#
# Three tiers, cheapest first.  Only used when a caller asks for it (see repairJSON on
# getLLMResponseJSON) -- a malformed response is normally better surfaced than papered over.
#
# Measured on three real failures from a sweep: a missing comma, an orphaned string inside a triple,
# and a nesting confusion.  The library fixed the first two with 99.4% of the content preserved, but
# on the third it returned a LIST instead of the expected object and raised nothing -- a silently
# wrong answer.  That is why every repair is shape-checked and content-checked before it is accepted.

def jsonRepairWithLibrary(jsonStr:str):
    # Tier 2: json_repair.  Free and fast, but it will happily return a wrong-shaped object rather
    # than fail, so the caller must validate what comes back.
    try:
        import json_repair
        return json_repair.loads(jsonStr)
    except Exception as e:
        print("jsonRepairWithLibrary(): failed (" + str(e)[:120] + ")")
        return None


def jsonRepairCollectStrings(obj, out:list):
    if (isinstance(obj, dict)):
        for k, v in obj.items():
            out.append(str(k))
            jsonRepairCollectStrings(v, out)
    elif (isinstance(obj, list)):
        for v in obj:
            jsonRepairCollectStrings(v, out)
    elif (isinstance(obj, str)):
        out.append(obj)
    return out


def jsonRepairLooksSane(repaired, originalStr:str, minRetained:float=0.90):
    # A repair is only acceptable if it (a) is an object, not some other shape the repairer invented,
    # and (b) does not invent text that was not in the original.  (b) is the important one for the
    # LLM tier: asking a model to reproduce 100k characters unchanged is itself an opportunity to
    # quietly rewrite something.
    if (not isinstance(repaired, dict)):
        print("jsonRepairLooksSane(): rejected -- repair produced a " + type(repaired).__name__ + ", not an object.")
        return False

    strings = jsonRepairCollectStrings(repaired, [])
    invented = [x for x in strings if (len(x) > 40) and (x not in originalStr)]
    if (len(invented) > 0):
        print("jsonRepairLooksSane(): rejected -- repair introduced " + str(len(invented)) + " string(s) absent from the original, e.g. " + repr(invented[0][:90]))
        return False

    normOriginal = " ".join(originalStr.split())
    normRepaired = " ".join(json.dumps(repaired).split())
    retained = (len(normRepaired) / len(normOriginal)) if (len(normOriginal) > 0) else 0.0
    if (retained < minRetained):
        print("jsonRepairLooksSane(): rejected -- only " + str(round(retained * 100)) + "% of the content survived the repair.")
        return False

    print("jsonRepairLooksSane(): accepted (" + str(round(retained * 100)) + "% of content retained, no invented text).")
    return True


# The reasoning effort sent with every call to a reasoning model (the gpt-5/gpt-6/o-series/claude
# branch below).  It was a fixed "high" for every stage; a caller can now replace
# default_reasoning_effort(model) -- the theorizer library does, per model -- and None sends no
# reasoning_effort at all, leaving the model at its own default.
DEFAULT_REASONING_EFFORT = "high"

def default_reasoning_effort(model:str):
    return DEFAULT_REASONING_EFFORT


# The model the third repair pass asks, when the caller names none.  A caller that sets its models
# (the theorizer library does) replaces default_repair_model() so the repair goes to one of them --
# a fixed name here fails behind any proxy or key that does not serve it.
DEFAULT_REPAIR_MODEL = "gpt-6-luna"

def default_repair_model():
    return DEFAULT_REPAIR_MODEL


def jsonRepairWithLLM(jsonStr:str, errorStr:str, model:str):
    # Tier 3: hand the broken JSON and the parser's own complaint to a cheap model and ask for the
    # same document with the syntax fixed and nothing else changed.  Used only when the library
    # tier fails or is rejected, which in practice is rare.
    print("jsonRepairWithLLM(): asking " + str(model) + " to repair the JSON (" + str(len(jsonStr)) + " chars)...")
    prompt = ""
    prompt += "The following JSON document is malformed and will not parse.\n"
    prompt += "The Python json module reports: " + str(errorStr) + "\n"
    prompt += "\n"
    prompt += "Return the SAME document with only the syntax error(s) corrected.\n"
    prompt += "Do not add, remove, reword, summarise, truncate or reorder any content. Every string, number, key and list element must be preserved exactly as written.\n"
    prompt += "If a fragment of text has lost its key, attach it to the field it clearly belongs to rather than deleting it.\n"
    prompt += "Return only the corrected JSON, between a single set of triple backticks.\n"
    prompt += "\n"
    prompt += "```\n" + jsonStr + "\n```\n"
    try:
        repairedJSON, repairedText, repairCost = getLLMResponseJSON(prompt, model, temperature=0.0, maxTokens=64000, jsonOut=False)
        return repairedJSON, repairCost
    except Exception as e:
        print("jsonRepairWithLLM(): failed (" + str(e)[:150] + ")")
        return None, 0.0


def jsonRepairResponse(jsonStr:str, errorStr:str, model:str):
    # Tier 2 then tier 3, each validated before being accepted.  Returns (parsedOrNone, extraCost).
    repaired = jsonRepairWithLibrary(jsonStr)
    if (repaired is not None) and (jsonRepairLooksSane(repaired, jsonStr)):
        print("jsonRepairResponse(): recovered with json_repair.")
        return repaired, 0.0

    print("jsonRepairResponse(): library repair unusable, falling back to an LLM repair pass.")
    repaired, repairCost = jsonRepairWithLLM(jsonStr, errorStr, model)
    if (repaired is not None) and (jsonRepairLooksSane(repaired, jsonStr)):
        print("jsonRepairResponse(): recovered with an LLM repair pass ($" + str(round(repairCost, 4)) + ").")
        return repaired, repairCost

    print("jsonRepairResponse(): ERROR: could not recover this response.")
    return None, repairCost if (repaired is not None) else 0.0


#
#   Helper: Get a response from an LLM model
#


# # Get a response from an LLM model using litellm
# def getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True, max_generation_time_seconds:int=300):
#     MAX_RETRIES = 5
#     MAX_GENERATION_TIME_SECONDS = 60 * 5        # Maximum of 5 minutes per generation (guard against long hangs)
#     if (max_generation_time_seconds != None):
#         MAX_GENERATION_TIME_SECONDS = max_generation_time_seconds

#     count_too_long_errors = 0

#     for retryIdx in range(MAX_RETRIES):
#         try:
#             # Use timeout
#             responseJSON, responseText, cost = func_timeout(MAX_GENERATION_TIME_SECONDS, _getLLMResponseJSON, args=(promptStr, model, temperature, maxTokens, jsonOut))
#             return responseJSON, responseText, cost

#         # timeout
#         except FunctionTimedOut:
#             errorInfo = "Time: " + str(time.strftime("%Y%m%d-%H%M%S")) + "  count_too_long_errors: " + str(count_too_long_errors) + "  model: " + model + "  prompt length: " + str(len(promptStr))
#             print("ERROR: LLM Generation timed out. " + str(errorInfo))
#             count_too_long_errors += 1
#             if (count_too_long_errors >= 3):
#                 print("ERROR: LLM Generation time out: Too many timeouts. Exiting.")
#                 #exit(1)
#                 return None, "", 0

#         # Keyboard exception
#         except KeyboardInterrupt:
#             print("Generation interrupted by user. Exiting.")
#             exit(1)
#         except Exception as e:
#             print("ERROR: Could not get LLM response. Retrying... ")
#             print("ERROR MESSAGE:")
#             print(e)
#             print(traceback.format_exc())

#             # Check for some known kinds of errors
#             errorStr = str(e)
#             if ("prompt is too long" in errorStr):
#                 print("ERROR: Prompt is too long. Exiting.")
#                 return None, "", 0

#         # Short delay before retrying
#         print("Delaying for a few seconds before retrying...")
#         print("Attempt " + str(retryIdx) + " of " + str(MAX_RETRIES))
#         time.sleep(retryIdx * 15)

#     # If we reach here, something terrible happened
#     print("ERROR: Could not get LLM response. Exiting.")
#     #exit(1)
#     return None, "", 0

import time
import traceback
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


# Seconds before an LLM call that was given no limit of its own is abandoned and retried.  The
# MAX_GENERATION_TIME_SECONDS environment variable sets it, read at call time (it used to be read as a
# default argument when this module was imported, and every caller then passed 300 over it, so
# setting it did nothing).  A call that passes max_generation_time_seconds keeps its own limit.
DEFAULT_GENERATION_TIME_SECONDS = 300

def default_generation_time_seconds():
    value = os.environ.get("MAX_GENERATION_TIME_SECONDS")
    if (value is None) or (len(str(value).strip()) == 0):
        return DEFAULT_GENERATION_TIME_SECONDS
    try:
        return int(float(value))
    except ValueError:
        print("ExtractionUtils.default_generation_time_seconds(): WARNING: MAX_GENERATION_TIME_SECONDS is not a number (" + str(value) + "), using " + str(DEFAULT_GENERATION_TIME_SECONDS), flush=True)
        return DEFAULT_GENERATION_TIME_SECONDS


# Why the last call made on this thread gave up, or None if it returned an answer.  A failed call
# returns the same empty value as a call whose answer could not be parsed, so a caller that needs to
# tell "timed out" from "answered with nothing" asks here.  Per thread, because the per-paper stages
# run one paper per thread.
_last_call = threading.local()

def get_last_call_failure():
    # {"reason": "timeout" | "prompt_too_long" | "max_tokens" | "interpreter_shutdown" | "error", "message": str,
    #  "attempts": int, "max_generation_time_seconds": int}, or None
    return getattr(_last_call, "failure", None)

# Every call that gave up in this process, in order, from any thread: what a command reports after a
# stage (a generation call's reflection runs on worker threads, where the per-thread record above
# is out of the caller's sight).  Each entry: the reason and message as above, plus the model, the
# prompt's size and the engine function that made the call.
CALL_FAILURES = []
_call_failures_lock = threading.Lock()

def _calling_function():
    # The first frame outside this module: the engine function that asked for the call.  A call
    # submitted to a worker thread has no such frame.
    for frame in reversed(traceback.extract_stack()[:-1]):
        if (os.path.basename(frame.filename) != os.path.basename(__file__)) and ("concurrent" not in frame.filename) and ("threading" not in frame.filename):
            return frame.name
    return None

def _set_last_call_failure(reason:str, message:str, attempts:int, max_generation_time_seconds, args=None):
    _last_call.failure = {"reason": reason, "message": message, "attempts": attempts, "max_generation_time_seconds": max_generation_time_seconds}
    prompt = args[0] if (args is not None) and (len(args) > 0) and isinstance(args[0], str) else ""
    try:
        prompt_tokens = countTokens(prompt)
    except Exception:
        prompt_tokens = None
    entry = dict(_last_call.failure)
    entry.update({"model": str(args[1]) if (args is not None) and (len(args) > 1) else None, "prompt_characters": len(prompt),
                  "prompt_tokens": prompt_tokens, "caller": _calling_function(), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    with _call_failures_lock:
        CALL_FAILURES.append(entry)


# The limit for a call whose caller names a fixed one of its own (theory generation: 15 or 8
# minutes; the evaluations: 300 s).  Those callers write generation_time_seconds(their value).  On
# its own the engine keeps their value unless MAX_GENERATION_TIME_SECONDS is set; a caller that
# sets its own policy (the theorizer library does) replaces _generation_time_resolver.  The function
# reads the resolver at call time, so modules that did `from ExtractionUtils import *` see it too.
def _engine_generation_time_seconds(fixed_seconds:int):
    value = os.environ.get("MAX_GENERATION_TIME_SECONDS")
    if (value is None) or (len(str(value).strip()) == 0):
        return fixed_seconds
    try:
        return int(float(value))
    except ValueError:
        return fixed_seconds

_generation_time_resolver = _engine_generation_time_seconds

def generation_time_seconds(fixed_seconds:int):
    return _generation_time_resolver(fixed_seconds)


# Set by the call itself (on the worker thread that makes it) when the model stopped at max_tokens:
# a message saying how much of the budget went where.  Read back by _run_noting_truncation() on the
# same thread, and handed to the caller's thread with the result.
_call_note = threading.local()

def _note_truncation(model, max_tokens, response):
    completion_tokens = reasoning_tokens = None
    try:
        usage = response["usage"]
        completion_tokens = usage.get("completion_tokens")
        details = usage.get("completion_tokens_details")
        if (details is not None):
            reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else getattr(details, "reasoning_tokens", None)
    except Exception:
        pass
    spent = "" if (completion_tokens is None) else (" -- " + str(completion_tokens) + " completion tokens" + ("" if (reasoning_tokens is None) else (", " + str(reasoning_tokens) + " of them reasoning")))
    _call_note.truncation = ("the model stopped at max_tokens=" + str(max_tokens) + spent + " before finishing its answer (model " + str(model)
                             + "). Raise max_tokens= on this stage.")

def _run_noting_truncation(llm_fn, args):
    _call_note.truncation = None
    result = llm_fn(*args)
    return result, getattr(_call_note, "truncation", None)


# Shared retry/timeout logic for LLM calls.
def _llm_call_with_retries(llm_fn, args, fail_return, max_generation_time_seconds=None):
    MAX_RETRIES = 5
    if (max_generation_time_seconds is None):
        max_generation_time_seconds = default_generation_time_seconds()
    MAX_GENERATION_TIME_SECONDS = max_generation_time_seconds
    _last_call.failure = None

    count_too_long_errors = 0
    lastErrorStr = ""

    model_slug = "unknown"

    for retryIdx in range(MAX_RETRIES):
        try:
            # e.g. llm_gpt-5-mini_0, llm_claude-3-7-sonnet-20250219_0
            model_slug = str(args[1]).replace("/", "_") if len(args) > 1 else str(args[1])

            # Uses ThreadPoolExecutor for thread-safe timeouts (not func_timeout which corrupts global state).
            ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"llm_{model_slug}")

            try:
                fut = ex.submit(_run_noting_truncation, llm_fn, args)
                result, truncation = fut.result(timeout=MAX_GENERATION_TIME_SECONDS)
                # An answer cut off at max_tokens comes back exactly like one the model wrote but
                # that held no JSON.  Only the call knows the difference, so say which it was.
                if (truncation is not None) and isinstance(result, tuple) and (len(result) > 0) and (result[0] is None):
                    _set_last_call_failure("max_tokens", truncation, retryIdx + 1, MAX_GENERATION_TIME_SECONDS, args)
                return result
            finally:
                # let the calling thread move on; worker thread still runs until HTTP returns
                ex.shutdown(wait=False, cancel_futures=True)

        except FuturesTimeoutError:
            errorInfo = (
                "Time: " + str(time.strftime("%Y%m%d-%H%M%S"))
                + "  count_too_long_errors: " + str(count_too_long_errors)
                + "  model: " + model_slug
                + "  prompt length: " + str(len(args[0]))
            )
            print("ERROR: LLM Generation timed out. " + str(errorInfo))
            count_too_long_errors += 1

            if count_too_long_errors >= 3:
                print("ERROR: LLM Generation time out: Too many timeouts. Exiting.")
                print("ERROR: 3 calls ran past " + str(MAX_GENERATION_TIME_SECONDS) + "s each and were abandoned."
                      + " A prompt this size (" + str(len(args[0])) + " characters) may simply need longer: raise"
                      + " MAX_GENERATION_TIME_SECONDS, or hand the call less evidence with max_data_tokens=.")
                _set_last_call_failure("timeout", "timed out " + str(count_too_long_errors) + " times (" + str(MAX_GENERATION_TIME_SECONDS) + " s each)", retryIdx + 1, MAX_GENERATION_TIME_SECONDS, args)
                return fail_return

        except KeyboardInterrupt:
            print("Generation interrupted by user. Exiting.")
            raise

        except Exception as e:
            print("ERROR: Could not get LLM response. Retrying... ")
            print("ERROR MESSAGE:")
            print(e)
            print(traceback.format_exc())

            errorStr = str(e)
            lastErrorStr = type(e).__name__ + ": " + errorStr
            if "prompt is too long" in errorStr or "ContextWindowExceededError" in errorStr:
                print("ERROR: Prompt is too long / context window exceeded. Exiting.")
                _set_last_call_failure("prompt_too_long", lastErrorStr[:500], retryIdx + 1, MAX_GENERATION_TIME_SECONDS, args)
                return fail_return

            if "interpreter shutdown" in errorStr:
                print("ERROR: Interpreter shutdown detected. Cannot retry. Exiting.")
                _set_last_call_failure("interpreter_shutdown", lastErrorStr[:500], retryIdx + 1, MAX_GENERATION_TIME_SECONDS, args)
                return fail_return

            if "LLM refused this prompt" in errorStr:
                print("ERROR: The prompt was refused. Retrying cannot change that. Exiting.")
                raise

        # Backoff before retry
        print("Delaying for a few seconds before retrying...")
        print("Attempt " + str(retryIdx) + " of " + str(MAX_RETRIES))
        time.sleep(retryIdx * 15)

    print("ERROR: Could not get LLM response. Exiting.")
    message = "no answer after " + str(MAX_RETRIES) + " attempts"
    if (count_too_long_errors > 0):
        message += " (" + str(count_too_long_errors) + " timed out after " + str(MAX_GENERATION_TIME_SECONDS) + " s)"
    if (len(lastErrorStr) > 0):
        message += "; last error: " + lastErrorStr[:400]
    _set_last_call_failure("timeout" if (count_too_long_errors > 0) and (len(lastErrorStr) == 0) else "error", message, MAX_RETRIES, MAX_GENERATION_TIME_SECONDS, args)
    return fail_return


def getLLMResponseJSON(promptStr, model, temperature=0, maxTokens=DEFAULT_MAX_TOKENS, jsonOut=True, max_generation_time_seconds=None, cachePrefixStr=None, cacheKey=None, retry_at_lower_effort_if_empty=False, repairJSON=False, repairModel=None):
    # cachePrefixStr/cacheKey turn on explicit prompt caching, and
    # retry_at_lower_effort_if_empty turns on the empty-response retry (see _getLLMResponseJSON).
    # All three default to off, so the call behaves exactly as it always has.
    return _llm_call_with_retries(
        _getLLMResponseJSON,
        (promptStr, model, temperature, maxTokens, jsonOut, cachePrefixStr, cacheKey, retry_at_lower_effort_if_empty, repairJSON, repairModel),
        fail_return=(None, "", 0),
        max_generation_time_seconds=max_generation_time_seconds,
    )


def getLLMResponseJSONWithMetadata(promptStr, model, temperature=0, maxTokens=DEFAULT_MAX_TOKENS, jsonOut=True, max_generation_time_seconds=None):
    return _llm_call_with_retries(
        _getLLMResponseJSONWithMetadata,
        (promptStr, model, temperature, maxTokens, jsonOut),
        fail_return=(None, "", 0, {}),
        max_generation_time_seconds=max_generation_time_seconds,
    )




def _getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True, cachePrefixStr:str=None, cacheKey:str=None, retry_at_lower_effort_if_empty:bool=False, repairJSON:bool=False, repairModel:str=None):
    # Some models reject the prompt_cache_options parameter (gpt-5-mini: 400 "prompt_cache_options is
    # not supported on this model").  If the cached request is rejected, make it once more without
    # caching rather than failing every retry the same way.
    if (cachePrefixStr is None):
        return _getLLMResponseJSONMaybeCached(promptStr, model, temperature, maxTokens, jsonOut, None, None, retry_at_lower_effort_if_empty, repairJSON, repairModel)
    try:
        return _getLLMResponseJSONMaybeCached(promptStr, model, temperature, maxTokens, jsonOut, cachePrefixStr, cacheKey, retry_at_lower_effort_if_empty, repairJSON, repairModel)
    except Exception as e:
        message = str(e)
        if ("prompt_cache" not in message) and ("cache" not in message.lower() or "not supported" not in message.lower()):
            raise
        print("getLLMResponseJSON(): " + str(model) + " rejected the prompt-cache options; calling again without caching", flush=True)
        return _getLLMResponseJSONMaybeCached(promptStr, model, temperature, maxTokens, jsonOut, None, None, retry_at_lower_effort_if_empty, repairJSON, repairModel)


def _getLLMResponseJSONMaybeCached(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True, cachePrefixStr:str=None, cacheKey:str=None, retry_at_lower_effort_if_empty:bool=False, repairJSON:bool=False, repairModel:str=None):
    # Explicit prompt caching (optional).  If cachePrefixStr is given it must be a prefix of
    # promptStr, and marks the part that is identical across a group of calls -- normally everything
    # up to and including the evidence.  That part is written to the cache once and read back by the
    # later calls in the group, which on these models is far cheaper than re-sending it.
    #
    # Two things are required, and both are easy to get wrong:
    #   - a breakpoint on the content block, saying where the cacheable prefix ends
    #   - a cacheKey, which routes calls in the same group to the same machine so the read can hit
    # Without a breakpoint these models cache on the whole prompt, so a shared prefix with a
    # different tail never hits.  There is no point caching a call that nothing reads back: a cache
    # write costs 1.25x a normal input token, so only turn this on for a group of calls.
    global TOTAL_LLM_COST
    print("Querying LLM model (" + str(model) + ")... ")

    # Note the running cost of all LLM queries
    print("(Running cost of all LLM generations so far: " + str(round(TOTAL_LLM_COST, 2)) + ")")

    # Measure the number of tokens in the prompt
    print("Prompt tokens: " + str(countTokens(promptStr)))

    contentBlocks = [{"type": "text", "text": promptStr}]
    if (cachePrefixStr is not None):
        if (not promptStr.startswith(cachePrefixStr)):
            print("getLLMResponseJSON(): WARNING: cachePrefixStr is not a prefix of the prompt -- caching disabled for this call.", flush=True)
            cachePrefixStr = None
        else:
            # Both providers cache a prefix block, and differ only in how the block is marked:
            # OpenAI reads prompt_cache_breakpoint, Anthropic reads cache_control.  Sending
            # OpenAI's marker to Anthropic is harmless (it is ignored), but it buys nothing, and a
            # generation pass re-sends this whole prefix once per theory and again per reflection,
            # so an unmarked prefix is paid for in full every time -- 5 to 10x the cached price on
            # a large run.  Anthropic's cache entry lives 5 minutes and needs 1024 tokens; an
            # evidence prefix clears that by three orders of magnitude.
            if ("claude" in model):
                prefixBlock = {"type": "text", "text": cachePrefixStr, "cache_control": {"type": "ephemeral"}}
            else:
                prefixBlock = {"type": "text", "text": cachePrefixStr, "prompt_cache_breakpoint": {"mode": "explicit"}}
            contentBlocks = [
                prefixBlock,
                {"type": "text", "text": promptStr[len(cachePrefixStr):]},
            ]
            print("Prompt cache: prefix " + str(countTokens(cachePrefixStr)) + " tokens, key '" + str(cacheKey) + "'", flush=True)

    messages=[
        {"role": "user",
         "content": contentBlocks
        }
    ]

    # Dump the whole thing to a file called (prompt-debug.txt)
    # Check 'prompts' directory exists, and make it otherwise
    if not os.path.exists(PROMPTS_DIRECTORY):
        os.makedirs(PROMPTS_DIRECTORY)
    print("Writing prompt to prompt-debug.txt...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Also add a random 4 digit number to the filename
    timestamp += "-" + str(random.randint(1000, 9999))
    filenameOut = os.path.join(PROMPTS_DIRECTORY, "prompt-debug." + timestamp + ".txt")
    with open(filenameOut, "w") as f:
        f.write(promptStr)

    import litellm
    litellm.drop_params = True

    extra_params = {}
    if (cachePrefixStr is not None) and ("claude" not in model):
        # prompt_cache_key goes as a normal argument; prompt_cache_options has to ride in extra_body
        # because litellm has no named parameter for it and drop_params would discard it.
        #
        # OpenAI only.  Anthropic rejects the whole request with "extra_body: Extra inputs are not
        # permitted" (a 400, not a dropped parameter), which killed every theory generation on a
        # Claude model while per-paper calls -- which pass no cache prefix -- went through fine.
        # Anthropic caches through cache_control blocks on the content itself, so wiring that up is
        # a separate job; until then a Claude run simply pays for the prefix each time.
        extra_params["prompt_cache_key"] = cacheKey
        extra_params["extra_body"] = {"prompt_cache_options": {"mode": "explicit"}, "prompt_cache_key": cacheKey}
    if ("gpt-oss-120b" in model):
        extra_params["provider"] = {"only": ["cerebras"], "allow_fallbacks": False}

    extra_headers = None
    if (model == "claude-3-5-sonnet-20240620") and (maxTokens > 4096):
        extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}

    response = None
    reasoning_effort = None
    # New (special handling for different models)
    if (model == "deepseek/deepseek-reasoner"):
        response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers, **extra_params)

    elif ("o3-mini" in model) or ("o4-mini" in model) or ("gpt-4.1" in model) or ("gpt-5" in model) or ("gpt-6" in model) or ("claude-4" in model) or ("claude" in model):
        # "high" unless the caller has set it per model (see default_reasoning_effort()); None sends none
        reasoning_effort = default_reasoning_effort(model)

        if (jsonOut):
            if (reasoning_effort == None):
                response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers, **extra_params)
            else:
                response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers, reasoning_effort=reasoning_effort, **extra_params)
        else:
            if (reasoning_effort == None):
                response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, extra_headers=extra_headers, **extra_params)
            else:
                response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, extra_headers=extra_headers, reasoning_effort=reasoning_effort, **extra_params)

    else:
        if (jsonOut):
            response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers, **extra_params)
        else:
            response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, extra_headers=extra_headers, **extra_params)


    print(response._hidden_params["response_cost"])

    # Get the response text

    # DEBUG: PRINT FULL RESPONSE
    printVerbose("FULL RESPONSE:")
    printVerbose(response)

    responseText = response["choices"][0]["message"]["content"]

    # A provider that refuses the prompt returns no content at all, with finish_reason
    # 'content_filter' (litellm's name for Anthropic's 'refusal').  A retry sends the same text to
    # the same classifier, so it is billed MAX_RETRIES times to fail identically; raise instead, and
    # say which prompt and model, because the rest of the stack cannot tell this from a dead key.
    finish_reason = response["choices"][0].get("finish_reason")
    if (responseText is None) and (finish_reason in ("content_filter", "refusal")):
        raise ValueError("LLM refused this prompt (finish_reason " + str(finish_reason) + ", model " + str(model)
                         + "): the provider's safety classifier returned no content. The same prompt will be refused"
                         + " again, so this is not retried. Rephrase the request, or run it on another model.")

    # If a reasoning model spends its entire completion budget on internal reasoning it returns an
    # EMPTY message with finish_reason 'length'.  There is no JSON to find and nothing in a code
    # block to salvage -- the model never began writing the answer.  (Measured: 27 of 273 calls in
    # one sweep, all with completion_tokens == reasoning_tokens, which silently produced runs with
    # zero theories.)  Retry once at a lower reasoning effort, which leaves more of the budget for
    # the answer itself.  The discarded call is still billed, so its cost is carried forward below.
    # Off by default: only callers that know an empty response is worth paying to retry should set
    # retry_at_lower_effort_if_empty (theory generation does; nothing else should need it).
    discardedCost = 0.0
    if (retry_at_lower_effort_if_empty) and ((responseText is None) or (len(str(responseText).strip()) == 0)):
        finishReason = None
        try:
            finishReason = response["choices"][0]["finish_reason"]
        except Exception:
            pass
        if (finishReason == "length"):
            retry_effort = reasoning_effort if (reasoning_effort in ("minimal", "low")) else "medium"
            print("WARNING: Empty response with finish_reason='length' -- the whole completion budget went to reasoning. Retrying once at reasoning_effort='" + retry_effort + "'.", flush=True)
            try:
                discardedCost = response._hidden_params.get("response_cost") or 0.0
            except Exception:
                discardedCost = 0.0
            try:
                if (jsonOut):
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers, reasoning_effort=retry_effort, **extra_params)
                else:
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, extra_headers=extra_headers, reasoning_effort=retry_effort, **extra_params)
                responseText = response["choices"][0]["message"]["content"]
                gotText = (responseText is not None) and (len(str(responseText).strip()) > 0)
                print("Retry at reasoning_effort='" + retry_effort + "': " + ("recovered " + str(len(str(responseText))) + " character(s)" if gotText else "still empty"), flush=True)
            except Exception as e:
                print("Retry at reasoning_effort='" + retry_effort + "' failed (" + str(e)[:150] + ")", flush=True)

    # Did the answer that will be parsed stop at max_tokens?  If no JSON comes out of it, that is
    # why, and the caller is told so (see _run_noting_truncation).
    try:
        if (response["choices"][0].get("finish_reason") == "length"):
            _note_truncation(model, maxTokens, response)
    except Exception:
        pass

    cost = 0
    if ("response_cost" in response._hidden_params) and (response._hidden_params["response_cost"] != None):
        cost = response._hidden_params["response_cost"]
        TOTAL_LLM_COST += cost
    else:
        # For models without cost information, try to estimate the cost based on the number of tokens
        prompt_tokens = response["usage"].get("prompt_tokens", 0)
        completion_tokens = response["usage"].get("completion_tokens", 0)
        reasoning_tokens = 0
        if ("completion_tokens_details" in response["usage"]):
            reasoning_tokens = None
            try:
                reasoning_tokens = response["usage"]["completion_tokens_details"].get("reasoning_tokens", 0)
            except Exception as e:
                pass
            if (reasoning_tokens == None):
                # Try to parse from the new CompletionTokensDetailsWrapper structure
                completion_details = response["usage"]["completion_tokens_details"]
                if (completion_details != None):
                    reasoning_tokens = completion_details.reasoning_tokens
            if (reasoning_tokens == None):
                reasoning_tokens = 0
        print("Reasoning tokens: " + str(reasoning_tokens))
        completion_tokens += reasoning_tokens   # Usually reasoning tokens count as completion tokens

        # Calculate the cost
        cost_prompt_tokens_per_million = 0.0
        cost_completion_tokens_per_million = 0.0
        costs_dict = {
            "default": {
                "prompt_tokens_per_million": 5.00,              # If we don't know the model, use some (nominally) high cost estimate, though this is not accurate
                "completion_tokens_per_million": 20.00
            },
            "deepseek/deepseek-reasoner": {
                "prompt_tokens_per_million": 0.55,
                "completion_tokens_per_million": 2.20
            },
            "o3-mini-2025-01-31": {
                "prompt_tokens_per_million": 1.10,
                "completion_tokens_per_million": 4.40
            },
            "o3-mini": {
                "prompt_tokens_per_million": 1.10,
                "completion_tokens_per_million": 4.40
            },
            "openai/o3-mini-2025-01-31": {
                "prompt_tokens_per_million": 1.10,
                "completion_tokens_per_million": 4.40
            },
            "openai/gpt-5": {
                "prompt_tokens_per_million": 1.25,
                "completion_tokens_per_million": 10.0
            },
            "openai/gpt-5-mini": {
                "prompt_tokens_per_million": 0.25,
                "completion_tokens_per_million": 2.00
            },
            "openai/gpt-5-nano": {
                "prompt_tokens_per_million": 0.05,
                "completion_tokens_per_million": 0.50
            },
            "gpt-4.1-2025-04-14": {
                "prompt_tokens_per_million": 2.00,
                "completion_tokens_per_million": 8.00
            }
        }
        if (model in costs_dict):
            cost_prompt_tokens_per_million = costs_dict[model]["prompt_tokens_per_million"]
            cost_completion_tokens_per_million = costs_dict[model]["completion_tokens_per_million"]
        else:
            cost_prompt_tokens_per_million = costs_dict["default"]["prompt_tokens_per_million"]
            cost_completion_tokens_per_million = costs_dict["default"]["completion_tokens_per_million"]
            print("WARNING: Model costs not found. Using default (high) costs of " + str(cost_prompt_tokens_per_million) + " and " + str(cost_completion_tokens_per_million) + " per million tokens.")

        # Calculate the cost
        cost = (prompt_tokens * cost_prompt_tokens_per_million / 1000000) + (completion_tokens * cost_completion_tokens_per_million / 1000000)

    # The abandoned empty call above was still billed; fold it in so totals stay honest.
    if (discardedCost > 0):
        cost += discardedCost
        TOTAL_LLM_COST += discardedCost

    # Report what the cache actually did.  A run that believes it is caching but reports cached=0
    # every time is paying the write premium for nothing, and that is invisible without this line.
    try:
        usageDetails = getattr(response.usage, "prompt_tokens_details", None)
        cachedTokens = (getattr(usageDetails, "cached_tokens", 0) or 0) if (usageDetails is not None) else 0
        writtenTokens = (getattr(usageDetails, "cache_write_tokens", 0) or 0) if (usageDetails is not None) else 0
        if (cachedTokens > 0) or (writtenTokens > 0) or (cachePrefixStr is not None):
            print("Prompt cache: read " + str(cachedTokens) + " token(s), wrote " + str(writtenTokens) + " token(s)", flush=True)
    except Exception as e:
        print("Prompt cache: could not read cache token counts (" + str(e) + ")", flush=True)

    print("Completed.  Cost: " + str(round(cost, 2)) + "  (Total Cost: " + str(round(TOTAL_LLM_COST, 2)) + ")")

    # Also dump the response (timestamped)
    filenameOut = os.path.join(PROMPTS_DIRECTORY, "prompt-debug." + timestamp + ".response.txt")
    with open(filenameOut, "w") as f:
        f.write(responseText or "")     # an empty response is an outcome to record, not a reason to crash

    responseOutJSON = None
    # Convert the response text to JSON
    try:
        responseOutJSON = json.loads(responseText)
    # Keyboard exception
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        # Usual when the JSON comes in a code block, which the second pass handles; say so only in the
        # verbose log.  A response that the later passes cannot read either still prints an ERROR.
        printVerbose("WARNING: Could not convert response to JSON on first pass. ")

    # Second pass for converting to JSON
    blockStr = None
    blockError = None
    if (responseOutJSON == None):
        # Try to find the last JSON block in the response, which starts with "```"
        lines = responseText.split("\n")
        startIdx = 0
        endIdx = 0
        for idx, line in enumerate(lines):
            if (line.startswith("```")):
                startIdx = endIdx
                endIdx = idx

        if (startIdx >= 0) and (endIdx > 0):
            # Exclude the start and end line
            linesJSON = lines[startIdx+1:endIdx]
            # Join the lines
            linesJSONStr = "\n".join(linesJSON)
            blockStr = linesJSONStr
            # Try to convert to JSON
            try:
                responseOutJSON = json.loads(linesJSONStr)
            except Exception as e:
                blockError = str(e)
                print("ERROR: Could not convert response to JSON on second pass. ")
        else:
            print("ERROR: Could not find JSON block in response. ")

    # Third pass: repair, if the caller asked for it.  A well-formed code block whose contents are
    # malformed is recoverable; an empty response is not, so there is nothing to do in that case.
    if (responseOutJSON == None) and (repairJSON) and (blockStr is not None) and (len(blockStr.strip()) > 0):
        repaired, repairCost = jsonRepairResponse(blockStr, blockError, repairModel if (repairModel is not None) else default_repair_model())
        if (repaired is not None):
            responseOutJSON = repaired
            cost += repairCost
            TOTAL_LLM_COST += repairCost

    # Return
    return responseOutJSON, responseText, cost


def _getLLMResponseJSONWithMetadata(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True, manualMessages:list=None):
    global TOTAL_LLM_COST
    print("Querying LLM model (" + str(model) + ")... ")

    # Note the running cost of all LLM queries
    print("(Running cost of all LLM generations so far: " + str(round(TOTAL_LLM_COST, 2)) + ")")

    # Measure the number of tokens in the prompt
    if (promptStr != None):
        print("Prompt tokens: " + str(countTokens(promptStr)))

    messages = []
    if manualMessages != None:
        messages = manualMessages
    else:
        messages=[
            {"role": "user",
            "content": [
                {
                    "type": "text",
                    "text": promptStr
                },
            ]
            }
        ]

    # Dump the whole thing to a file called (prompt-debug.txt)
    # Check 'prompts' directory exists, and make it otherwise
    if not os.path.exists(PROMPTS_DIRECTORY):
        os.makedirs(PROMPTS_DIRECTORY)
    print("Writing prompt to prompt-debug.txt...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Also add a random 4 digit number to the filename
    timestamp += "-" + str(random.randint(1000, 9999))
    if (manualMessages != None):
        filenameOut = os.path.join(PROMPTS_DIRECTORY, "prompt-debug." + timestamp + ".manual.json")
        with open(filenameOut, "w") as f:
            json.dump(messages, f, indent=4)
        # But also save a text version
        outStr = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            outStr += "-" * 80 + "\n"
            outStr += "Role: " + role + "\n"
            outStr += "-" * 80 + "\n"
            outStr += content + "\n"

        filenameOut = os.path.join(PROMPTS_DIRECTORY, "prompt-debug." + timestamp + ".manual.txt")
        with open(filenameOut, "w") as f:
            f.write(outStr)


    else:
        filenameOut = os.path.join(PROMPTS_DIRECTORY, "prompt-debug." + timestamp + ".txt")
        with open(filenameOut, "w") as f:
            f.write(promptStr)

    #litellm.drop_params=True
    cost = 0
    if (promptStr != None):
        # Set litellm to drop params
        import litellm
        litellm.drop_params = True

        extra_params = {}
        if ("gpt-oss-120b" in model):
            extra_params["provider"] = {"only": ["cerebras"], "allow_fallbacks": False}

        extra_headers = None
        if (model == "claude-3-5-sonnet-20240620") and (maxTokens > 4096):
            extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}

        response = None


        # New (special handling for different models)
        if (model == "deepseek/deepseek-reasoner"):
            response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers, **extra_params)

        elif ("o3-mini" in model) or ("o4-mini" in model) or ("gpt-5" in model) or ("gpt-6" in model) or ("claude" in model):
            reasoning_effort = default_reasoning_effort(model)

            if (jsonOut):
                if (reasoning_effort == None):
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers, **extra_params)
                else:
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers, reasoning_effort=reasoning_effort, **extra_params)
            else:
                if (reasoning_effort == None):
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, extra_headers=extra_headers, **extra_params)
                else:
                    response = completion(model=model, messages=messages, max_completion_tokens=maxTokens, extra_headers=extra_headers, reasoning_effort=reasoning_effort, **extra_params)

        elif ("claude" in model):
            #print("** SPECIAL HANDLING FOR CLAUDE **")
            # Do not use special JSON model for claude -- it seems to do better at adhering to the schema without it
            response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, extra_headers=extra_headers, **extra_params)

        else:
            if (jsonOut):
                response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, response_format={"type": "json_object"}, extra_headers=extra_headers, **extra_params)
            else:
                response = completion(model=model, messages=messages, temperature=temperature, max_tokens=maxTokens, extra_headers=extra_headers, **extra_params)


        print(response._hidden_params["response_cost"])

        # Get the response text

        # DEBUG: PRINT FULL RESPONSE
        printVerbose("FULL RESPONSE:")
        printVerbose(response)
        metadata = {}

        responseText = response["choices"][0]["message"]["content"]

        if ("response_cost" in response._hidden_params) and (response._hidden_params["response_cost"] != None):
            cost = response._hidden_params["response_cost"]
            print("Found cost in response: " + str(cost))
            TOTAL_LLM_COST += cost
        else:
            # Try to calculate the cost based on the number of tokens
            prompt_tokens = response["usage"].get("prompt_tokens", 0)
            completion_tokens = response["usage"].get("completion_tokens", 0)
            reasoning_tokens = 0
            if ("completion_tokens_details" in response["usage"]):
                reasoning_tokens = None
                try:
                    reasoning_tokens = response["usage"]["completion_tokens_details"].get("reasoning_tokens", 0)
                except Exception as e:
                    pass
                if (reasoning_tokens == None):
                    # Try to parse from the new CompletionTokensDetailsWrapper structure
                    completion_details = response["usage"]["completion_tokens_details"]
                    if (completion_details != None):
                        reasoning_tokens = completion_details.reasoning_tokens
                if (reasoning_tokens == None):
                    reasoning_tokens = 0
            print("Reasoning tokens: " + str(reasoning_tokens))
            completion_tokens += reasoning_tokens   # Usually reasoning tokens count as completion tokens

            # Calculate the cost
            cost_prompt_tokens_per_million = 0.0
            cost_completion_tokens_per_million = 0.0
            costs_dict = {
                "default": {
                    "prompt_tokens_per_million": 5.00,              # Use some high cost estimate
                    "completion_tokens_per_million": 20.00
                },
                "deepseek/deepseek-reasoner": {
                    "prompt_tokens_per_million": 0.55,
                    "completion_tokens_per_million": 2.20
                },
                "o3-mini-2025-01-31": {
                    "prompt_tokens_per_million": 1.10,
                    "completion_tokens_per_million": 4.40
                },
                "o3-mini": {
                    "prompt_tokens_per_million": 1.10,
                    "completion_tokens_per_million": 4.40
                },
                "openai/o3-mini-2025-01-31": {
                    "prompt_tokens_per_million": 1.10,
                    "completion_tokens_per_million": 4.40
                },
                "ft:gpt-4o-mini-2024-07-18": {
                    "prompt_tokens_per_million": 0.30,
                    "completion_tokens_per_million": 3.00
                },
                "openai/gpt-5": {
                    "prompt_tokens_per_million": 1.25,
                    "completion_tokens_per_million": 10.0
                },
                "openai/gpt-5-mini": {
                    "prompt_tokens_per_million": 0.25,
                    "completion_tokens_per_million": 2.00
                },
                "openai/gpt-5-nano": {
                    "prompt_tokens_per_million": 0.05,
                    "completion_tokens_per_million": 0.50
                }
            }

            # Get keys in the costs_dict
            cost_model_keys = costs_dict.keys()
            if (model in costs_dict):
                cost_prompt_tokens_per_million = costs_dict[model]["prompt_tokens_per_million"]
                cost_completion_tokens_per_million = costs_dict[model]["completion_tokens_per_million"]
            else:
                # Check if it's a fine-tuned model
                if (model.startswith("ft:")):
                    # Fine tuned models have unique names, but might start with one of the same prefixes
                    # Check if the model name is in the costs_dict
                    found = False
                    for key in cost_model_keys:
                        if (model.startswith(key)):
                            cost_prompt_tokens_per_million = costs_dict[key]["prompt_tokens_per_million"]
                            cost_completion_tokens_per_million = costs_dict[key]["completion_tokens_per_million"]
                            found = True
                            print("INFO: Found fine-tuned model in costs_dict. Using costs of " + str(cost_prompt_tokens_per_million) + " and " + str(cost_completion_tokens_per_million) + " per million tokens.")
                            break
                    if (not found):
                        cost_prompt_tokens_per_million = costs_dict["default"]["prompt_tokens_per_million"]
                        cost_completion_tokens_per_million = costs_dict["default"]["completion_tokens_per_million"]
                        print("WARNING: Model costs not found. Using default (high) costs of " + str(cost_prompt_tokens_per_million) + " and " + str(cost_completion_tokens_per_million) + " per million tokens.")
                else:
                    cost_prompt_tokens_per_million = costs_dict["default"]["prompt_tokens_per_million"]
                    cost_completion_tokens_per_million = costs_dict["default"]["completion_tokens_per_million"]
                    print("WARNING: Model costs not found. Using default (high) costs of " + str(cost_prompt_tokens_per_million) + " and " + str(cost_completion_tokens_per_million) + " per million tokens.")

            # Calculate the cost
            cost = (prompt_tokens * cost_prompt_tokens_per_million / 1000000) + (completion_tokens * cost_completion_tokens_per_million / 1000000)

        print("Completed.  Cost: " + str(round(cost, 2)) + "  (Total Cost: " + str(round(TOTAL_LLM_COST, 2)) + ")")


    # Metadata
    try:
        prompt_tokens = response["usage"].get("prompt_tokens", 0)
        completion_tokens = response["usage"].get("completion_tokens", 0)
        reasoning_tokens = 0
        if ("completion_tokens_details" in response["usage"]) and (response["usage"]["completion_tokens_details"] != None):
            reasoning_tokens = None
            try:
                reasoning_tokens = response["usage"]["completion_tokens_details"].get("reasoning_tokens", 0)
            except Exception as e:
                pass
            if (reasoning_tokens == None):
                # Try to parse from the new CompletionTokensDetailsWrapper structure
                completion_details = response["usage"]["completion_tokens_details"]
                if (completion_details != None):
                    reasoning_tokens = completion_details.reasoning_tokens
            if (reasoning_tokens == None):
                reasoning_tokens = 0
        print("Reasoning tokens: " + str(reasoning_tokens))

        metadata['tokens_prompt'] = prompt_tokens
        metadata['tokens_completion'] = completion_tokens
        metadata['tokens_reasoning'] = reasoning_tokens
        metadata['tokens_total'] = prompt_tokens + completion_tokens + reasoning_tokens
        metadata['cost'] = cost
        metadata['model'] = model
        metadata['temperature'] = temperature
        metadata['max_tokens'] = maxTokens
    except Exception as e:
        print("ERROR: Could not extract metadata from response. ")
        print(e)
        import traceback
        print(traceback.format_exc())


    # Also dump the response (timestamped)
    filenameOut = os.path.join(PROMPTS_DIRECTORY, "prompt-debug." + timestamp + ".response.txt")
    with open(filenameOut, "w") as f:
        f.write(responseText or "")     # an empty response is an outcome to record, not a reason to crash

    responseOutJSON = None
    # Convert the response text to JSON
    try:
        responseOutJSON = json.loads(responseText)
    # Keyboard exception
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        # Usual when the JSON comes in a code block, which the second pass handles; say so only in the
        # verbose log.  A response that the later passes cannot read either still prints an ERROR.
        printVerbose("WARNING: Could not convert response to JSON on first pass. ")

    # Second pass for converting to JSON
    if (responseOutJSON == None):
        # Try to find the last JSON block in the response, which starts with "```"
        lines = responseText.split("\n")
        startIdx = 0
        endIdx = 0
        for idx, line in enumerate(lines):
            if (line.startswith("```")):
                startIdx = endIdx
                endIdx = idx

        if (startIdx >= 0) and (endIdx > 0):
            # Exclude the start and end line
            linesJSON = lines[startIdx+1:endIdx]
            # Join the lines
            linesJSONStr = "\n".join(linesJSON)
            # Try to convert to JSON
            try:
                responseOutJSON = json.loads(linesJSONStr)
            except Exception as e:
                print("ERROR: Could not convert response to JSON on second pass. ")
        else:
            print("ERROR: Could not find JSON block in response. ")

    # Return
    return responseOutJSON, responseText, cost, metadata


# OpenAI Embeddings
def getEmbedding(textStr:str, model:str = "text-embedding-3-small"):
    # Get the embedding from the model
    response = embedding(
        model=model,
        input=textStr,
    )

    try:
        vectorOut = response["data"][0]["embedding"]
        return vectorOut
    except Exception as e:
        print("ERROR: Could not extract embedding from response. ")
        return None


# Accepts (and returns) a list of strings
def getEmbeddingsList(text_list:list, model="text-embedding-3-small"):
    if isinstance(text_list, str):
        text_list = [text_list]

    response = embedding(
        model=model,
        input=text_list,
    )

    try:
        embeddings = [item["embedding"] for item in response["data"]]
        return embeddings

    except Exception as e:
        print("ERROR extracting embeddings:", e)
        return None


def cosineSimilarity(vec1, vec2):
    from numpy import dot
    from numpy.linalg import norm
    return dot(vec1, vec2)/(norm(vec1)*norm(vec2))




# Helper for parsing codeblocks
# Finds all the codeblocks in a string, and returns them as a list of lists of strings.
# Very useful when providing a format prompt to an LLM, as you can ask it to provide specific structured responses within a codeblock, then extract these.
# e.g. "Please respond in JSON format, as a dictionary with a single key, `answer', which is a number. Place your response between codeblocks (```)"
# Expected input_str:
# ```
# {
#    "answer": 42
# }
# ```
# Returns: [["{", "\"answer\": 42", "}"]]
# Will handle multiple codeblocks in the input string.
# NOTE: This function is used in the LLM proxy code, and is critical for extracting structured data from LLM responses.
def find_codeblocks(input_str):
    # Find all codeblocks in the input string
    codeblocks = []
    lines = input_str.split("\n")
    current_codeblock = []
    active = False

    for idx, line in enumerate(lines):
        if line.startswith("```"):
            if (active == True):
                # Finish off the current codeblock
                codeblocks.append(current_codeblock)
                current_codeblock = []
                active = False
            else:
                # Start a new codeblock
                active = True
        else:
            # If we're currently in the middle of a codeblock, add the line to the current codeblock (we never want the ``` to be included in the codeblock)
            if (active == True):
                current_codeblock.append(line)

    # For each codeblock, make it a flat string instead of a list of lines
    for idx, codeblock in enumerate(codeblocks):
        codeblocks[idx] = "\n".join(codeblock)

    return codeblocks



# Quick test
if __name__ == "__main__":
    # Set keys
    loadAPIKeys()

    # # Test of deepseek
    #modelStr = "openrouter/openai/gpt-oss-120b"
    # #modelStr = "claude-3-7-sonnet-20250219"
    # #modelStr = "claude-3-5-sonnet-20241022"
    # #modelStr = "o1-mini"
    # #modelStr = "deepseek/deepseek-chat"
    # #modelStr = "deepseek/deepseek-reasoner"
    # #modelStr = "openai/o3-mini-2025-01-31"
    modelStr = "gpt-5-mini"

    # #promptStr = "What is the capital of France? Respond in JSON."
    promptStr = "Write a 1000 word story about penguins. Respond in JSON, as a dictionary, with the key {\"story\": \"...\"}."
    # #responseJSON, responseText, cost = getLLMResponseJSON(promptStr, modelStr, temperature=0.0, maxTokens=DEFAULT_MAX_TOKENS, jsonOut=True)
    responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr, modelStr, temperature=0.0, maxTokens=DEFAULT_MAX_TOKENS, jsonOut=False)

    print("RESPONSE (JSON):")
    print(json.dumps(responseJSON, indent=4))
    print("RESPONSE:")
    print(responseText)
    print("COST: " + str(cost))
    print("Metadata:")
    print(json.dumps(metadata, indent=4))


    # # Embeddings test
    # # Example
    # texts = [
    #     "Graph neural networks for materials discovery.",
    #     "LLMs can reason about scientific hypotheses."
    # ]

    # vectors = getEmbeddingsList(texts)
    # print(len(vectors), "embeddings returned.")
    # print(len(vectors[0]), "dimensions per vector.")
