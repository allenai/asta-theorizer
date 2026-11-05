# ExtractionUtils.py
# This is a set of wrappers for LiteLLM to assist with LLM interactions, including cost tracking, retries, timeouts, json parsing, etc.  Wrappers are somewhat rough-and-ready, cost and token tracking may be approximate.

import os
import json
import time
import traceback
import random

from litellm import completion
from litellm import embedding

import tiktoken



# Cost Estimates. Note that these are just estimates and may not be accurate -- you should ideally have other methods (like a hard limit on your account) to help limit unexpected costs.
TOTAL_LLM_COST = 0.0            # Running (estimated) cost of all LLM queries so far. This is a global variable that is updated with each query.

DEFAULT_MAX_TOKENS = 8000


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
    max_tokens = (litellm.get_max_tokens(model_str) or 200000) - headroom
    if countTokens(text) <= max_tokens:
        return text
    return tiktokenEncoder.decode(tiktokenEncoder.encode(text, disallowed_special=())[:max_tokens])


#
#   Helper: Load the API keys
#

def loadAPIKeys():
    print("Loading API keys...")


    API_KEY_FILE = "api_keys.donotcommit.json"
    try:
        with open(API_KEY_FILE, 'r') as fileIn:
            apiKeys = json.load(fileIn)
            # Set the OpenAI environment variable
            if ("openai" in apiKeys):
                os.environ["OPENAI_API_KEY"] = apiKeys["openai"]
            # Set the Anthropic environment variable
            if ("anthropic" in apiKeys):
                os.environ["ANTHROPIC_API_KEY"] = apiKeys["anthropic"]
            # Set the deepseek environment variable
            if ("deepseek" in apiKeys):
                os.environ["DEEPSEEK_API_KEY"] = apiKeys["deepseek"]
            if ("openrouter" in apiKeys):
                os.environ["OPENROUTER_API_KEY"] = apiKeys["openrouter"]
            # Together.ai
            if ("togetherai" in apiKeys):
                os.environ["TOGETHERAI_API_KEY"] = apiKeys["togetherai"]
            # Mistral
            if ("mistral" in apiKeys):
                os.environ["MISTRAL_API_KEY"] = apiKeys["mistral"]


    except Exception as e:
        print("ERROR: Was unable to load API key file (`api_keys.donotcommit.json`).")
        print("Continuing with the assumption that these are already set in the environment.")


#
#   Helper: Get the total cost of all LLM queries
#
def getTotalLLMCost():
    global TOTAL_LLM_COST
    return TOTAL_LLM_COST


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


# Shared retry/timeout logic for LLM calls.
def _llm_call_with_retries(llm_fn, args, fail_return, max_generation_time_seconds=int(os.environ.get("MAX_GENERATION_TIME_SECONDS", 300))):
    MAX_RETRIES = 5
    MAX_GENERATION_TIME_SECONDS = max_generation_time_seconds

    count_too_long_errors = 0

    model_slug = "unknown"

    for retryIdx in range(MAX_RETRIES):
        try:
            # e.g. llm_gpt-5-mini_0, llm_claude-3-7-sonnet-20250219_0
            model_slug = str(args[1]).replace("/", "_") if len(args) > 1 else str(args[1])

            # Uses ThreadPoolExecutor for thread-safe timeouts (not func_timeout which corrupts global state).
            ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"llm_{model_slug}")

            try:
                fut = ex.submit(llm_fn, *args)
                return fut.result(timeout=MAX_GENERATION_TIME_SECONDS)
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
            if "prompt is too long" in errorStr or "ContextWindowExceededError" in errorStr:
                print("ERROR: Prompt is too long / context window exceeded. Exiting.")
                return fail_return

            if "interpreter shutdown" in errorStr:
                print("ERROR: Interpreter shutdown detected. Cannot retry. Exiting.")
                return fail_return

        # Backoff before retry
        print("Delaying for a few seconds before retrying...")
        print("Attempt " + str(retryIdx) + " of " + str(MAX_RETRIES))
        time.sleep(retryIdx * 15)

    print("ERROR: Could not get LLM response. Exiting.")
    return fail_return


# --- Test-only fake LLM seam -------------------------------------------------
# When THEORIZER_FAKE_LLM is set, every LLM call returns a canned, shape-valid
# response for $0 instead of hitting a provider. This exists so the load/stability
# harness can run the REAL server end-to-end — real A2A handling, queue, PaperStore
# threads, retention, concurrency, and resident paper memory — at high concurrency
# without spending tokens. The model's *content* doesn't affect whether N concurrent
# requests fit in memory, so faking it preserves the only thing the scaling test
# cares about. No effect at all when the env var is unset (the default).
def _fake_llm_response(promptStr, with_metadata=False):
    delay = float(os.environ.get("THEORIZER_FAKE_LLM_DELAY_S", "0") or 0)
    if delay:
        time.sleep(delay)  # simulate real per-call latency so concurrency is sustained, not a t=0 spike
    # NOTE: every prompt opens with "You are ScientistGPT", so route on markers unique
    # to each step, checked most-specific first.
    p = promptStr or ""
    if "# Paper Text" in p:
        # Per-paper extraction: SchemaExtractionQueue reads response["results"] into extracted_data.
        obj = {
            "results": [{"key_finding": "fake finding (load test)",
                         "mechanism": "fake mechanism (load test)"}],
            "follow_on_papers": [],
        }
    elif "paper-search query and extraction schema" in p:
        # Step-2 schema build: must yield string query/extraction_query + a list schema.
        obj = {
            "theory_request_normalized": "Build a theory based on the following results.",
            "short_name": "fake-load-test",
            "paper_search_query": "fake load-test query",
            "extraction_query": "Extract the key finding and its mechanism.",
            "extraction_schema": [
                {"name": "key_finding", "type": "str", "description": "The key finding."},
                {"name": "mechanism", "type": "str", "description": "The proposed mechanism."},
            ],
        }
    else:
        # Theory formation (and anything else): empty theory sets -> pipeline completes
        # cleanly with "No theories generated" without needing a shape-perfect fixture.
        obj = {"theories_general": [], "theories_specific": []}
    text = json.dumps(obj)
    return (obj, text, 0.0, {"fake": True}) if with_metadata else (obj, text, 0.0)


def _fake_llm_enabled():
    return os.environ.get("THEORIZER_FAKE_LLM", "").strip().lower() in ("1", "true", "yes")


def getLLMResponseJSON(promptStr, model, temperature=0, maxTokens=DEFAULT_MAX_TOKENS, jsonOut=True, max_generation_time_seconds=300):
    if _fake_llm_enabled():
        return _fake_llm_response(promptStr, with_metadata=False)
    return _llm_call_with_retries(
        _getLLMResponseJSON,
        (promptStr, model, temperature, maxTokens, jsonOut),
        fail_return=(None, "", 0),
        max_generation_time_seconds=max_generation_time_seconds,
    )


def getLLMResponseJSONWithMetadata(promptStr, model, temperature=0, maxTokens=DEFAULT_MAX_TOKENS, jsonOut=True, max_generation_time_seconds=300):
    if _fake_llm_enabled():
        return _fake_llm_response(promptStr, with_metadata=True)
    return _llm_call_with_retries(
        _getLLMResponseJSONWithMetadata,
        (promptStr, model, temperature, maxTokens, jsonOut),
        fail_return=(None, "", 0, {}),
        max_generation_time_seconds=max_generation_time_seconds,
    )




def _getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True):
    global TOTAL_LLM_COST
    print("Querying LLM model (" + str(model) + ")... ")

    # Note the running cost of all LLM queries
    print("(Running cost of all LLM generations so far: " + str(round(TOTAL_LLM_COST, 2)) + ")")

    # Measure the number of tokens in the prompt
    print("Prompt tokens: " + str(countTokens(promptStr)))

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
    if not os.path.exists("prompts"):
        os.makedirs("prompts")
    print("Writing prompt to prompt-debug.txt...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Also add a random 4 digit number to the filename
    timestamp += "-" + str(random.randint(1000, 9999))
    filenameOut = "prompts/prompt-debug." + timestamp + ".txt"
    with open(filenameOut, "w") as f:
        f.write(promptStr)

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

    elif ("o3-mini" in model) or ("o4-mini" in model) or ("gpt-4.1" in model) or ("gpt-5" in model) or ("claude-4" in model) or ("claude" in model):
        #reasoning_effort = None
        #reasoning_effort = "low"
        #reasoning_effort = "medium"
        reasoning_effort = "high"

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
    print("FULL RESPONSE:")
    print(response)

    responseText = response["choices"][0]["message"]["content"]
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

    print("Completed.  Cost: " + str(round(cost, 2)) + "  (Total Cost: " + str(round(TOTAL_LLM_COST, 2)) + ")")

    # Also dump the response (timestamped)
    filenameOut = "prompts/prompt-debug." + timestamp + ".response.txt"
    with open(filenameOut, "w") as f:
        f.write(responseText)

    responseOutJSON = None
    # Convert the response text to JSON
    try:
        responseOutJSON = json.loads(responseText)
    # Keyboard exception
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        print("WARNING: Could not convert response to JSON on first pass. ")

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
    if not os.path.exists("prompts"):
        os.makedirs("prompts")
    print("Writing prompt to prompt-debug.txt...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Also add a random 4 digit number to the filename
    timestamp += "-" + str(random.randint(1000, 9999))
    if (manualMessages != None):
        filenameOut = "prompts/prompt-debug." + timestamp + ".manual.json"
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

        filenameOut = "prompts/prompt-debug." + timestamp + ".manual.txt"
        with open(filenameOut, "w") as f:
            f.write(outStr)


    else:
        filenameOut = "prompts/prompt-debug." + timestamp + ".txt"
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

        elif ("o3-mini" in model) or ("o4-mini" in model) or ("gpt-5" in model) or ("claude" in model):
            #reasoning_effort = None
            #reasoning_effort = "low"
            #reasoning_effort = "medium"
            reasoning_effort = "high"

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
        print("FULL RESPONSE:")
        print(response)
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
    filenameOut = "prompts/prompt-debug." + timestamp + ".response.txt"
    with open(filenameOut, "w") as f:
        f.write(responseText)

    responseOutJSON = None
    # Convert the response text to JSON
    try:
        responseOutJSON = json.loads(responseText)
    # Keyboard exception
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        print("WARNING: Could not convert response to JSON on first pass. ")

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
