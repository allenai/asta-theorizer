# TheorizerProcessing.py
# This file has many of the LLM prompt-based functions for the various stages of the theory generation process.
import os
import json
import time
import requests

import threading

from Struct import *
from ExtractionUtils import *


#
#   Converting a theory request (e.g. make me a theory about X) into a query and schema for extraction on papers.
#
def convert_theory_request_to_query_and_schema(theory_request:str, model_str:str, max_tokens:int=8000, temperature:float=0.0, use_reflection:bool=False):
    def mkPrompt(theory_request:str, reflection:str=None):
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"

        prompt += "# Task\n"
        prompt += "You are part of a system that is designed to infer scientific theories on specific topics, based on evidence gathered from scientific papers.\n"
        prompt += "The steps in the process are:\n"
        prompt += "1. Collect the theory request from the user (e.g. 'make me a theory about X')\n"
        prompt += "*2. Convert the theory request into a paper-search query and extraction schema for extracting entities/supporting information from scientific papers.\n"
        prompt += "3. Use the query and schema to extract entities from scientific papers.\n"
        prompt += "4. Use the extracted entities to build a theory about the topic.\n"
        prompt += "\n"
        prompt += "We are currently on step 2.\n"
        prompt += "You will be given a theory request, and your task is to convert it into an paper search query and extraction schema for extracting entities from scientific papers that would provide highly relevant evidence to support building theories in this domain.\n"
        prompt += "- The `paper search query` is used to find papers that are relevant to the theory request, using a paper search tool over Semantic Scholar.\n"
        prompt += "- The `extraction schema` is used by an extraction tool to extract relevant entities/information from the full text of each paper found by the query.\n"
        prompt += "\n"

        prompt += "# Theory Request\n"
        prompt += "Here is the theory request:\n"
        prompt += "```\n"
        prompt += theory_request + "\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Thinking about the goals of a good paper query and schema. "
        prompt += "The goal of the `paper search query` is to find papers that contain evidence that is relevant to the theory request.\n"
        prompt += "The goal of the `extraction schema` is to broadly extract entities/relevant information that can be used to build theories about the topic.\n"
        prompt += "\n"
        prompt += "Things to consider:\n"
        prompt += "- The query should be broad enough to capture a wide range of papers on this specific topic, but specific enough to avoid irrelevant or loosely relevant papers.\n"
        prompt += "- The schema should capture easily-extracted evidence that is relevant to the theory request.  A theory needs to explain a wide variety of evidence, so it's important to consider not just evidence that might support a theory, but also extract evidence/counterpoints that might provide challenges to creating an all-encompassing theory.\n"
        prompt += "\n"
        prompt += "## Examples\n"
        prompt += "Here are some examples of theory requests, and the corresponding paper search queries and extraction schemas.  These are illustrative only, and may have issues/challenges -- you can likely do better if you think about it:\n"
        prompt += "Here is a non-exhaustive list of some of the aspects in the extraction schema examples that help make them useful for the theory extraction task:\n"
        prompt += "- Requests to extract specific types of entities (e.g. model names, agent names, benchmark names, etc.)\n"
        prompt += "- Requests to extract specific types of information or properties about the entities (e.g. descriptions, performance, categories/types, aspects such as model size or other properties that might be relevant, etc.)\n"
        prompt += "- Checks for binary conditions that are relevant to the criteria (e.g. does the agent use memory, does the paper report performance with and without memory, etc.)\n"
        prompt += "- Requests for counter-examples or comparisons (e.g. not only performance with memory, but also performance without memory, so these conditions can be compared)\n"
        prompt += "- Requests for results of studies, summarized in brief/punchy and information-dense ways (remember, this extraction will be used on many papers, so the results should be concise and information-dense so the important aspects can easily stand out)\n"
        prompt += "\n"
        prompt += "Example 1:\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += "    \"theory_request_original\": \"A theory of how agents can use memory to play text games\",\n"
        prompt += "    \"theory_request_normalized\": \"Build a theory of how agents can use memory to play text games, based on the following results.\",\n"
        prompt += "    \"short_name\": \"agent-memory-text-games\",\n"
        prompt += "    \"paper_search_query\": \"Extract any mentions of agents playing specific text games, and whether or not the agents use a memory.\",\n"
        prompt += "    \"extraction_query\": \"Extract any mentions of agents playing specific text games, and whether or not the agents use a memory.\",\n"
        prompt += "    \"extraction_schema\": [\n"
        prompt += "        {\"name\": \"agent_name\", \"type\": \"str\", \"description\": \"The name of the agent that plays the game.\"},\n"
        prompt += "        {\"name\": \"agent_description\", \"type\": \"str\", \"description\": \"A brief description of the agent that plays the game.\"},\n"
        prompt += "        {\"name\": \"benchmark_name\", \"type\": \"str\", \"description\": \"The name of the text game benchmark that the agent plays\"},\n"
        prompt += "        {\"name\": \"agent_memory\", \"type\": \"bool\", \"description\": \"Does the agent use a form of memory that learns from the task? (true, false, or null for no information)\"},\n"
        prompt += "        {\"name\": \"agent_memory_type\", \"type\": \"str\", \"description\": \"If the agent uses a memory, describe the type of memory, and what the representations in the memory are.\"},\n"
        prompt += "        {\"name\": \"performance_with_memory\", \"type\": \"str\", \"description\": \"If the agent uses a memory, what is the performance of the agent on the benchmark when it uses that memory? (numerical, include units. null if no information)\"},\n"
        prompt += "        {\"name\": \"performance_without_memory\", \"type\": \"str\", \"description\": \"What is the performance of the vanilla (i.e. no-memory) version of the agent on the benchmark? (numerical, include units. null if no information)\"},\n"
        prompt += "        {\"name\": \"has_performance_with_without_memory\", \"type\": \"bool\", \"description\": \"Does the paper report performance on this agent both with and without memory? (true, false, or null for no information)\"}\n"
        prompt += "    ]\n"
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"
        prompt += "Example 2:\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += "    \"theory_request_original\": \"I'd like a theory of how agents can process/perform different kinds of arithmetic/mathematics\",\n"
        prompt += "    \"theory_request_normalized\": \"Build a theory of how agents can process/perform different kinds of arithmetic/mathematics, based on the following results.\",\n"
        prompt += "    \"short_name\": \"agents-perform-arithmetics-mathematics\",\n"
        prompt += "    \"paper_search_query\": \"Extract any mentions of the performance of LLMs on artithmetic problems.\",\n"
        prompt += "    \"extraction_query\": \"Extract any mentions of the performance of LLMs on artithmetic problems.\",\n"
        prompt += "    \"extraction_schema\": [\n"
        prompt += "        {\"name\": \"model_name\", \"type\": \"str\", \"description\": \"The name of the LLM model that is being evaluated.\"},\n"
        prompt += "        {\"name\": \"model_description\", \"type\": \"str\", \"description\": \"A brief description of the LLM model.\"},\n"
        prompt += "        {\"name\": \"model_size\", \"type\": \"str\", \"description\": \"How large was the model, in parameters? (e.g. 1B, 2B, 7B, 13B, 70B, etc.)\"},\n"
        prompt += "        {\"name\": \"math_problems\", \"type\": \"str\", \"description\": \"What kind of math problems were used to evaluate the model? Be specific.\"},\n"
        prompt += "        {\"name\": \"performance\", \"type\": \"str\", \"description\": \"What was the performance of the model on the math problems? Ideally this should be as detailed as possible, and broken down by problem type. (numerical, include units. null if no information)\"}\n"
        prompt += "    ]\n"
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"
        prompt += "Example 3:\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += "    \"theory_request_original\": \"Please investigate a theory for how probing language models works\",\n"
        prompt += "    \"theory_request_normalized\": \"Build a theory of how probing works in large language models.\",\n"
        prompt += "    \"short_name\": \"probing-large-language-models\",\n"
        prompt += "    \"paper_search_query\": \"Extract any mentions of different types of probes for probing large language models (LLMs).\",\n"
        prompt += "    \"extraction_query\": \"Extract any mentions of different types of probes for probing large language models (LLMs).\",\n"
        prompt += "    \"extraction_schema\": [\n"
        prompt += "        {\"name\": \"probe_name\", \"type\": \"str\", \"description\": \"The name of the probe that is being evaluated.\"},\n"
        prompt += "        {\"name\": \"probe_description\", \"type\": \"str\", \"description\": \"A detailed description of the probe.\"},\n"
        prompt += "        {\"name\": \"what_is_probed\", \"type\": \"str\", \"description\": \"What is the probe probing? (e.g. 'the model's ability to perform arithmetic', 'the model's ability to understand natural language', etc.)\"},\n"
        prompt += "        {\"name\": \"results\", \"type\": \"str\", \"description\": \"What are the results of the probe? (e.g. 'the model's performance on arithmetic problems', 'the model's performance on natural language understanding tasks', etc.)\"}\n"
        prompt += "    ]\n"
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"
        prompt += "Example 4:\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += "    \"theory_request_original\": \"Could you create one or more theories of how theory-of-mind works in large language models (LLMs).\",\n"
        prompt += "    \"theory_request_normalized\": \"Build a theory of how theory-of-mind works in large language models (LLMs), based on the following results.\",\n"
        prompt += "    \"short_name\": \"theory-of-mind-llms\",\n"
        prompt += "    \"paper_search_query\": \"Extract any mentions of how LLMs perform theory-of-mind tasks, and what the results are.\",\n"
        prompt += "    \"extraction_query\": \"Extract any mentions of how LLMs perform theory-of-mind tasks, and what the results are.\",\n"
        prompt += "    \"extraction_schema\": [\n"
        prompt += "        {\"name\": \"model_name\", \"type\": \"str\", \"description\": \"The name of the LLM model that is being evaluated.\"},\n"
        prompt += "        {\"name\": \"model_description\", \"type\": \"str\", \"description\": \"A brief description of the LLM model.\"},\n"
        prompt += "        {\"name\": \"model_size\", \"type\": \"str\", \"description\": \"How large was the model, in parameters? (e.g. 1B, 2B, 7B, 13B, 70B, etc.)\"},\n"
        prompt += "        {\"name\": \"task_name\", \"type\": \"str\", \"description\": \"The name of the theory-of-mind task that the model is being evaluated on.\"},\n"
        prompt += "        {\"name\": \"task_description\", \"type\": \"str\", \"description\": \"A brief description of the theory-of-mind task that the model is being evaluated on.\"},\n"
        prompt += "        {\"name\": \"performance\", \"type\": \"str\", \"description\": \"What was the performance of the model on the theory-of-mind task? Ideally this should be as detailed as possible, and broken down by task type. (numerical, include units. null if no information)\"}\n"
        prompt += "    ]\n"
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"

        if (reflection is not None):
            prompt += "\n"
            prompt += "# Reflection\n"
            prompt += "This is a reflection step. Previously, you generated a response (below).  Now, your task is to reflect on that response, and fix any errors, inconsistencies, omissions, or any other issues.\n"
            prompt += "```\n"
            prompt += reflection + "\n"
            prompt += "```\n"
            prompt += "\n"

        prompt += "# Output Format\n"
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
        prompt += "\n"
        prompt += "Your JSON response must be a dictionary, that contains exactly 6 keys: `theory_request_original`, `theory_request_normalized`, `short_name`, `paper_search_query`, `extraction_query`, and `extraction_schema`.  The values for these keys are as follows:\n"
        prompt += "- `theory_request_original`(str): The original theory request that was provided (e.g. 'make me a theory about X')\n"
        prompt += "- `theory_request_normalized`(str): A normalized, clear version of the theory request, that always starts with `Build a theory of...` and ends with `based on the following results.`.\n"
        prompt += "- `short_name`(str): A short name for the theory request, that is a concise, and generally 4 words long, separated by '-'. This will be a prefix appended to filenames to save data for this request.\n"
        prompt += "- `paper_search_query`(str): The query that should be used to search for papers that are relevant to the theory request, using a paper search tool over Semantic Scholar.\n"
        prompt += "- `extraction_query`(str): A natural language request query, that's provided to an extraction prompt (along with the extraction schema below) to outline the overall purpose of the extraction process.  Should start with `Extract any mentions of...`.\n"
        prompt += "- `extraction_schema`(list): A list of dictionaries, where each dictionary contains the following keys:\n"
        prompt += "    - `name`(str): The name of the entity to extract (e.g. 'entity_name')\n"
        prompt += "    - `type`(str): The type of the entity to extract (e.g. 'str', 'int', 'float', etc.)\n"
        prompt += "    - `description`(str): A brief, concise, and helpful description of the entity to extract, often times with examples, that very explicitly spells out what information should be extracted.\n"
        prompt += "\n"

        prompt += "## Special instructions\n"
        prompt += "- You can (and are encouraged to) think and plan before writing your JSON response.  But, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
        prompt += "- Please follow the instructions carefully.\n"
        prompt += "- Do not hallucinate.\n"

        return prompt


    total_cost = 0.0
    responses = []
    responses_raw = []

    # Create the prompt
    prompt = mkPrompt(theory_request)
    # Call the LLM
    responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=0.0, maxTokens=DEFAULT_MAX_TOKENS, jsonOut=False)
    responses.append(responseJSON)
    responses_raw.append(responseText)
    total_cost += cost

    # If we are using reflection, we will call the LLM again with the reflection step
    if (use_reflection):
        # Create the reflection prompt
        reflection_prompt = mkPrompt(theory_request, responseText)
        # Call the LLM again
        responseJSON, responseText, cost = getLLMResponseJSON(reflection_prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False)
        responses.append(responseJSON)
        responses_raw.append(responseText)
        total_cost += cost


    # Get the best response (the last one that's not None)
    best_response = None
    for response in reversed(responses):
        if (response is not None):
            best_response = response
            break

    # Pack output
    packed = {
        "theory_request": theory_request,
        "model": model_str,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "use_reflection": use_reflection,
        "output": best_response,
        "responses": responses,
        "responses_raw": responses_raw,
        "total_cost": total_cost,
    }
    return packed




#
#   Theory Building
#


# Theory Generation: `Accuracy-Focused` Objective
# Build a set of theories from a list of results, with reflection.
# Performs the reflection one theory at a time, to encourage it to fix any issues with the single theory, more fully populate the evidence, etc.
# This one explicitly focuses on trying to abstract new qualitative and quantitative laws from the supporting evidence.
def build_theory_from_results_single_theory_reflection3(query:str, results:list, model_str="gpt-4o-mini", max_tokens=12000, temperature=0.0, use_reflection:bool=False, include_query_in_generation:bool=True):
    version_str = "theorizer-literature-supported-accuracy-focused-12102025"
    def mkPrompt(query:str, results:str, reflection:str=None, include_query_in_generation:bool=True):
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"

        prompt += "# Task\n"
        prompt += "Your task is to construct a theory of now a scientific phenomenon works, based on a collection of evidence that you will be provided.\n"
        prompt += "## Subtask 1: General Theories\n"
        prompt += "In the first subtask, you will be asked to provide broad, high-level theories that explain the phenomenon, based on evidence provided.\n"
        prompt += "## Subtask 2: Specific Theories\n"
        prompt += "In the second subtask, you will be asked to provide more specific, more granular theories, that may only explain a subset of the phenomenon/data provided.\n"
        prompt += "\n"

        prompt += "# Total theories to generate\n"
        prompt += "You should generate 2 general theories, and 2 specific theories (i.e. 4 theories total).\n"
        prompt += "\n"

        prompt += "# Theory Quality\n"
        prompt += "You should focus on generating high-quality laws and theories that are (a) genuinely novel and insightful, (b) impactful, and (c) accurate. The goal is discovery, not rehashing already known phenomena.\n"
        prompt += "\n"

        prompt += "# What is a theory?\n"
        prompt += "A theory (from the perspective of philosophy of science, in the context of Kuhn, or later work by Pat Langley and collaborators) is a collection of statements that together explain a phenomenon, and provide predictive power for describing what will happen with future/unseen observations. In this way they are related to models.\n"
        prompt += "Theories can include both qualitative and quantitative statements/components. For example, an equation (like `force = mass * acceleration`) can be part of a theory, but so can qualitative statements (i.e. `objects that are thrown upwards will eventually fall back down to a planet due to the force of gravity, unless they are expelled at such a speed that they can escape that planet's gravity`).\n"
        prompt += "\n"

        prompt += "# Abstracting Laws/Statements\n"
        prompt += "The theories that you generate should contain (nominally, novel) qualitative and/or quantitative laws, and these laws should be explicitly abstracted from the supporting evidence.\n"
        prompt += "For example, if (in the supporting evidence) you see groups of results (across papers) that show that as certain variables change, others change in specific ways, you should try to abstract these relationships into qualitative or quantitative laws/statements that describe how these variables relate to each other.\n"
        prompt += "These laws/statements should be explicitly supported by the evidence you have been provided.\n"
        prompt += "\n"
        prompt += "## Inferring laws through logical deduction, abduction, and induction\n"
        prompt += "You can also use logical deduction, abduction, and induction to infer new laws/statements that are not explicitly stated in the evidence, but can be logically inferred from it.\n"
        prompt += "Similarly, you can use these reasoning techniques to combine laws/statements that are explicitly stated in the evidence to derive new laws/statements that are not explicitly stated, but follow logically from them.  If you do this, list this explicitly in the 'supporting evidence' for each law made in this way (with the UUIDs as an empty list). For example: 'supporting evidence: This law/statement was logically inferred through deduction/abduction/induction from law/statement 3 (xyz) and law/statement 5 (abc) above'.\n"
        prompt += "\n"

        prompt += "# Format of Laws / Theory Statements\n"
        prompt += "The theory statements that you provide must generally be framed as qualitative or quantitative laws/statements.\n"
        prompt += "Qualitative laws/statements describe relationships between variables or phenomena in a non-numeric way (e.g. X causes Y, or increasing A leads to a decrease in B, or objects with property P tend to exhibit behavior Q under conditions C).\n"
        prompt += "Quantitative laws/statements describe relationships between variables or phenomena in a numeric way (e.g. equations, proportionalities, statistical relationships, etc.).\n"
        prompt += "\n"
        prompt += "## Domain/Scope\n"
        prompt += "Each theory statement must also include a precise description of the domain/scope of the law/statement that it applies to.  For example, Newtonian gravity applies to macroscopic objects at non-relativistic speeds, but not to quantum-scale objects or objects moving near the speed of light.  This domain/scope description should be clear, and information-dense.\n"
        prompt += "Having an appropriate domain/scope is critical for ensuring that the law/statement is properly contextualized, and that its applicability is clear, particularly for evaluating the accuracy and impact of the law/statement. If the domain/scope is too broad, then the law/statement will look less accurate than it actually is, and if it is too narrow, then the law/statement will look less impactful than it actually is.\n"
        prompt += "Laws/statements are evaluated on literature.  You could imagine that if you defined Newtonian Gravity as a theory, then evaluated it on recent papers (but omiting the scope that it only applies to macroscopic objects at non-relativistic speeds), it would look inaccurate -- even though it's one of the most historically successful theories -- because many recent papers discuss relativistic or quantum effects that Newtonian gravity does not account for.  However, if you include the proper scope/domain, then it becomes clear that Newtonian gravity is accurate within its domain/scope, and thus the law/statement is more accurate than it would appear with an inappropriate domain/scope.\n"
        prompt += "Note that 'precise' domains don't mean you need to very precisely define everything (e.g. having to define a particular numerical range for 'non-relativistic speeds', which would be difficult -- the accuracy of Newtonian gravity degrades gradually as speeds approach the speed of light, so there is no hard cutoff), but rather that you should be as clear, faithful, accurate, and information-dense as possible in describing the domain/scope.\n"
        prompt += "\n"
        #prompt += "For ease of interpretation and evaluation, they must be expressed as a form of conditional expression (i.e. IF ... THEN ...), in a format that looks like a triple.\n"
        prompt += "More specifically, each theory statement must be in the following format:\n"
        # Expressed as triples, and in JSON
        prompt += "In JSON, this would look like:\n"
        prompt += "```\n"
        prompt += "{\n"
        #prompt += '    "law_name": "...", # A short name for the law\n'
        prompt += '    "statement_name": "...", # A short name for the statement/law\n'
        prompt += '    "theory_statement": "...", # The full text of the law/statement, in a concise, clear, and information-dense manner.\n'
        prompt += '    "domain_scope": "...", # A precise, clear, and information-dense description of the domain/scope of the law/statement\n'
        prompt += '    "special_cases`(list): A list of any special cases, exceptions, or boundary conditions that apply to the law/statement, if any.\n'
        # prompt += '    "if": [\n'
        # prompt += '        {"subject": "...", "relation": "...", "object": "..."}, # One triple\n'
        # prompt += '        {"subject": "...", "relation": "...", "object": "..."}, # AND, second triple\n'
        # prompt += '        # ... more triples as needed\n'
        # prompt += '    ],\n'
        # prompt += '    "then": [\n'
        # prompt += '        {"subject": "...", "relation": "...", "object": "..."}, # One triple\n'
        # prompt += '        {"subject": "...", "relation": "...", "object": "..."}, # AND, second triple\n'
        # prompt += '        # ... more triples as needed\n'
        # prompt += '    ]\n'
        # Supporting evidence
        prompt += '    "supporting_evidence": [\n'
        prompt += '        {"text": "...", "uuids": [...]}, # One piece of supporting evidence\n'
        prompt += '        {"text": "...", "uuids": [...]}, # Second piece of supporting evidence\n'
        prompt += '        # ... more pieces of supporting evidence as available\n'
        prompt += '    ]\n'
        # Qual/qant
        prompt += '    "qual_or_quant": "..." # Is this law/statement primarily "qualitative", "quantitative", or "mixed"\n'
        # Existing law
        prompt += '    "novelty_evaluation": {\n'
        prompt += '        "what_already_exists": "...", # brief, information-dense, 1-2 sentence explanation of what in this law/statement is already known in previous work",\n'
        prompt += '        "what_is_novel": "...", # brief, information-dense, 1-2 sentence explanation of what SIGNIFICANT aspects of this law/statement are novel",\n'
        prompt += '        "classification_explanation": "..." # A brief, concise, information-dense, 1-2 sentence explanation as to why this classification was made, referencing existing/novel work>"\n'
        prompt += '        "likely_classification": "...", # One of `existing`, `closely-related-to-existing`, `somewhat-related-to-existing`, or `new`\n'
        prompt += '        "references": [...], # A list of relevant references here. Format: Citation, then square brackets very briefly describing how it relates to the law.\n'
        prompt += '    }\n'
        prompt += '}\n'
        prompt += "```\n"
        #prompt += "The purpose of framing the statements/laws in this way is to make it easy to evaluate and compare across theories. The subjects, relations, and objects should be concise, but also information rich and detailed.\n"
        #prompt += "Each triple represents a single assertion about the relationship between a subject and an object, via a relation.\n"
        #prompt += "Multiple triples in the `if` section are combined with AND logic (i.e. all conditions must be met), and multiple triples in the `then` section are also combined with AND logic (i.e. all outcomes occur if the conditions are met).\n"
        prompt += "\n"


        prompt += "# Query\n"
        prompt += "Here is the general area/query that your theories should be centered around:"
        if (include_query_in_generation):
            prompt += "```\n"
            prompt += query + "\n"
            prompt += "```\n"
            prompt += "\n"
            prompt += "Here is a list of specific results that you should use to build your theories:\n"
        # If this isn't set, just give it the raw data and ask it to brainstorm in a data-driven way rather than in a query-driven way
        prompt += "```\n"
        prompt += json.dumps(results, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"


        if (reflection is not None):
            prompt += "\n"
            prompt += "# Reflection\n"
            prompt += "This is a reflection step. Previously, you generated a response (below).  Now, your task is to reflect on that response, and fix any errors, inconsistencies, omissions, or any other issues.\n"
            prompt += "NOTE: This reflection is focused on a single theory that you generated previously. Your task is to fix any issues with that single theory, and ensure it is as complete, accurate, and well-supported as possible.\n"
            prompt += "This includes filling in missing evidence present in the extracted evidence but not included in the theory's list of supporting evidence for each law, conflicting evidence, special cases, etc., as well as fixing any errors, inconsistencies, omissions, or other issues you find.\n"
            prompt += "You should also pay particular attention to ensuring that the domain/scope of each law/statement is accurate and appropriately scoped, as well as clear, detailed, and information-dense.\n"
            prompt += "For example, the list of supporting evidence shouldn't be limited to some small amount (e.g. 5 per law) -- it should be as exhaustive as possible, and include all genuinely relevant evidence provided in the results.\n"
            prompt += "\n"
            prompt += "```\n"
            prompt += reflection + "\n"
            prompt += "```\n"

        prompt += "# Output Format\n"
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
        prompt += "\n"
        prompt += "Your JSON response must be a dictionary, that contains two keys: `theories_general`, and `theories_specific`.  Each is a list of dictionaries, with the following keys:\n"
        prompt += "- `theory_name`(str): A short name for the theory\n"
        prompt += "- `theory_description`(str): A full description of the theory\n"
        #prompt += "- `supporting_evidence`(list): A list of specific supporting evidence for the theory.\n"
        #prompt += "- `theory_statements`(list): A list of specific predictive statements (either qualitative, quantitative, or both) that describe the assertions of the theory.\n"
        prompt += "- `theory_statements`(list): A list of specific laws (either qualitative, quantitative, or both) that describe the assertions of the theory.\n"
        prompt += "- `new_predictions_likely`(list): A list of several example (testable) predictions that the theory makes (the situation, and the expected results) that are not in the observed evidence. For this set of new predictions, the predictions should be fairly likely to work out.\n"
        prompt += "- `new_predictions_unknown`(list): A list of several example (testable) predictions that the theory makes (the situation, and the expected results) that are not in the observed evidence. For this set of new predictions, the predictions should be difficult to tell if they would genuinely work out or not, and have particularly impactful results if they do (or don't) work out.\n"
        prompt += "- `negative_experiments`(list): A list of several (testable) predictions that test the assumptions or statements of the theory, and if false, would call the theory statements into question.\n"
        prompt += "- `unaccounted_for`(list): A list of specific pieces of evidence that are not explained by the theory, if any.\n"
        prompt += "- `conflicting_evidence`(list): A list of specific pieces of evidence that appear to conflict with the theory, if any.\n"
        prompt += "- `special_cases`(list): A list of any special cases, exceptions, or boundary conditions that apply to the theory, if any.\n"
        prompt += "- `existing_theory`(dict): A dictionary that describes whether (to the best of your knowledge, and the information available in the papers) if this is a pre-existing theory, or a new theory. It has the following keys: `likely_classification`:str, which is one of `new`, `somewhat-related-to-existing`, `closely-related-to-existing`, or `existing`, and `references`:list(str), which are lists of references (one per string) to new (or related) theories in the form `First author (year) title [brief note of how related, and name(s) of relevant theories, in square brackets]`. These should help confirm the theory is new, or help confirm it is existing or closely related. It also has: `what_already_exists`(str), `what_is_novel`(str), and `classification_explanation`(str).\n"
        prompt += "\n"
        prompt += "While most fields are lists of strings, for `supporting_evidence` and `unaccounted_for`, the list should be a list of dictionaries, where each dictionary has the following keys:\n"
        prompt += "- `text`(str): A textual description of the evidence\n"
        prompt += "- `uuids`(list): A list of UUIDs that correspond to the specific pieces of evidence in the original results that support this text description. These lists can be as long or short as they need to be to encompass all the correct evidence references.\n"
        prompt += "\n"
        prompt += "Similarly, the `theory_statements` field is a list of dictionaries, where each dictionary has the following key:\n"
        prompt += "- `law`(dict): A dictionary that describes a single qualitative or quantitative law, in the format described above (i.e. with `if` and `then` fields, each containing a list of triples), and specific supporting evidence that supports the development of this law.\n"
        prompt += "\n"
        prompt += "For example (a cartoon example -- your theories should be much more detailed, and have much more supporting evidence, linked back with proper UUIDs):\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += "    \"theories_general\": [\n"
        prompt += '        {\n'
        prompt += '            "theory_name": "Theory of Gravity",\n'
        prompt += '            "theory_description": "A theory that explains the force of gravity as a result of mass attracting other mass.",\n'
        # prompt += '            "supporting_evidence": [\n'
        # prompt += '                {\"text\": "Objects fall towards the Earth when dropped.", \"uuids\": [1, 5]},\n'
        # prompt += '                {\"text\": "The Moon orbits the Earth due to gravitational attraction.", \"uuids\": [2, 3, 8]},\n'
        # prompt += '                {\"text\": "The planets orbit the Sun due to the Sun\'s gravitational pull.", \"uuids\": [6, 10, 15, 18, 21]},\n'
        # prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        # prompt += '            ],\n'
        # New format
        prompt += '            "theory_statements": [\n'
        prompt += '                {\n'
        prompt += '                  "statement_name": "Mutual Attraction of Massive Objects",\n'
        prompt += '                  "theory_statement": "Objects with mass exert a mutual attractive force on each other, proportional to their masses.",\n'
        prompt += '                  "domain_scope": "Applies to macroscopic objects with mass in a classical physics context, excluding relativistic and quantum effects.",\n'
        prompt += '                  "special_cases": [\n'
        prompt += '                      "Does not apply to massless particles like photons.",\n'
        prompt += '                      "Breaks down at quantum scales and near singularities."\n'
        prompt += '                      # ... add more special cases as needed\n'
        prompt += '                  ],\n'
        prompt += '                  "supporting_evidence": [\n'
        prompt += '                      { "text": "Dropped objects fall toward Earth.", "uuids": [...] },\n'
        prompt += '                      { "text": "The Moon remains bound to Earth through mutual attraction.", "uuids": [...] }\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        prompt += '                  "qual_or_quant": "qualitative",\n'
        prompt += '                  "novelty_evaluation": {\n'
        prompt += '                        "what_already_exists": "<brief, information-dense, 1-2 sentence explanation of what in this law is already known in previous work>",\n'
        prompt += '                        "what_is_novel": "<brief, information-dense, 1-2 sentence explanation of what in this law is novel>",\n'
        prompt += '                        "classification_explanation": "<brief, concise, information-dense, 1-2 sentence explanation as to why this classification was made, referencing existing/novel work>",\n'
        prompt += '                        "likely_classification": "<existing|closely-related-to-existing|somewhat-related-to-existing|new>",\n'
        prompt += '                        "references": [\n'
        prompt += '                            # Add relevant references here\n'
        prompt += '                        ]\n'
        prompt += '                  }\n'
        prompt += '                },\n'
        # Next statement/law
        prompt += '                {\n'
        prompt += '                  "statement_name": "Inverse Square Law of Gravitational Force",\n'
        prompt += '                  "theory_statement": "The gravitational force between two masses is inversely proportional to the square of the distance between them.",\n'
        prompt += '                  "domain_scope": "Applies to point masses or spherically symmetric masses in a classical physics context, excluding relativistic and quantum effects.",\n'
        prompt += '                  "special_cases": [\n'
        prompt += '                      "Breaks down at very small scales where quantum effects dominate.",\n'
        prompt += '                      "Requires modification in strong gravitational fields as described by General Relativity."\n'
        prompt += '                      # ... add more special cases as needed\n'
        prompt += '                  ],\n'
        prompt += '                  "supporting_evidence": [\n'
        prompt += '                      { "text": "The motion of planets and moons follows an inverse square relationship.", "uuids": [...] },\n'
        prompt += '                      { "text": "Laboratory experiments measuring gravitational attraction confirm the inverse square law.", "uuids": [...] }\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        prompt += '                  "qual_or_quant": "quantitative",\n'
        prompt += '                  "novelty_evaluation": {\n'
        prompt += '                        "what_already_exists": "<brief, information-dense, 1-2 sentence explanation of what in this law is already known in previous work>",\n'
        prompt += '                        "what_is_novel": "<brief, information-dense, 1-2 sentence explanation of what in this law is novel>",\n'
        prompt += '                        "classification_explanation": "<brief, concise, information-dense, 1-2 sentence explanation as to why this classification was made, referencing existing/novel work>",\n'
        prompt += '                        "likely_classification": "<existing|closely-related-to-existing|somewhat-related-to-existing|new>",\n'
        prompt += '                        "references": [\n'
        prompt += '                            # Add relevant references here\n'
        prompt += '                        ]\n'
        prompt += '                  }\n'
        prompt += '                },\n'
        # Next statement/law
        # This time, do a quantitative law
        prompt += '                {\n'
        prompt += '                  "statement_name": "Gravitational Force Proportional to Product of Masses",\n'
        prompt += '                  "theory_statement": "The gravitational force between two objects is directly proportional to the product of their masses.",\n'
        prompt += '                  "domain_scope": "Applies to macroscopic objects with mass in a classical physics context, excluding relativistic and quantum effects.",\n'
        prompt += '                  "special_cases": [\n'
        prompt += '                      "Does not apply to massless particles like photons.",\n'
        prompt += '                      "Breaks down at quantum scales and near singularities."\n'
        prompt += '                      # ... add more special cases as needed\n'
        prompt += '                  ],\n'
        prompt += '                  "supporting_evidence": [\n'
        prompt += '                      { "text": "Experiments show that increasing the mass of either object increases the gravitational force between them.", "uuids": [...] },\n'
        prompt += '                      { "text": "Observations of planetary motion confirm the relationship between mass and gravitational force.", "uuids": [...] }\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        prompt += '                  "qual_or_quant": "quantitative",\n'
        prompt += '                  "novelty_evaluation": {\n'
        prompt += '                        "what_already_exists": "<brief, information-dense, 1-2 sentence explanation of what in this law is already known in previous work>",\n'
        prompt += '                        "what_is_novel": "<brief, information-dense, 1-2 sentence explanation of what in this law is novel>",\n'
        prompt += '                        "classification_explanation": "<brief, concise, information-dense, 1-2 sentence explanation as to why this classification was made, referencing existing/novel work>",\n'
        prompt += '                        "likely_classification": "<existing|closely-related-to-existing|somewhat-related-to-existing|new>",\n'
        prompt += '                        "references": [\n'
        prompt += '                            # Add relevant references here\n'
        prompt += '                        ]\n'
        prompt += '                  }\n'
        prompt += '                }\n'
        prompt += '                # ... add more laws/statements as needed\n'
        prompt += '            ],\n'
        prompt += '            "new_predictions_likely": [\n'
        prompt += '                "If a large object is placed in space, it will attract smaller objects towards it due to gravity.",\n'
        prompt += '                "If two objects are in space, they will eventually collide if they are on a collision course due to gravitational attraction."\n'
        prompt += '            ],\n'
        prompt += '            "new_predictions_unknown": [\n'
        prompt += '                "If a massive object were to suddenly disappear, the gravitational effects on nearby objects would be unknown.",\n'
        prompt += '                "The behavior of gravity in extreme conditions, such as near black holes, is not fully understood."\n'
        prompt += '                # ... this should be as detailed/exhaustive as possible. Add as many elements as needed.\n'
        prompt += '            ],\n'
        prompt += '            "negative_experiments": [\n'
        prompt += '                "Finding cases where two massive objects do not attract each other despite being in close proximity would call into question the theory of gravity.",\n'
        prompt += '                "Finding cases where the force of gravity does not follow the inverse square law would call into question the theory of gravity."\n'
        prompt += '            ],\n'
        prompt += '            "unaccounted_for": [\n'
        prompt += '                {\"text\": "The exact mechanism of how gravity works at the quantum level is not fully understood.", \"uuids\": [7, 25]},\n'
        prompt += '                {\"text\": "The behavior of gravity in extreme conditions, such as near black holes, is not fully understood.", \"uuids\": [12, 28]}\n'
        prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '            ],\n'
        prompt += '            "conflicting_evidence": [\n'
        prompt += '                {\"text\": "Observations of galaxy rotation curves that do not match predictions based on visible matter challenge the theory of gravity as currently understood.", \"uuids\": [30, 32]},\n'
        prompt += '                {\"text\": "The accelerated expansion of the universe challenges the traditional understanding of gravity.", \"uuids\": [35, 40]}\n'
        prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '            ],\n'
        prompt += '            "special_cases": [\n'
        prompt += '                "In the presence of extremely strong gravitational fields, such as near black holes, the predictions of general relativity may differ from Newtonian gravity.",\n'
        prompt += '                "At very small scales, quantum effects may alter the behavior of gravity."\n'
        prompt += '                # ... add more special cases as needed\n'
        prompt += '            ],\n'
        prompt += '            "existing_theory": { # Whole-theory-related classification\n'
        prompt += '                "what_already_exists": "...",\n'
        prompt += '                "what_is_novel": "...",\n'
        prompt += '                "classification_explanation": "..." # A brief, concise, information-dense, 1-2 sentence explanation as to why this classification was made, specifically referencing existing/novel work\n'
        prompt += '                "likely_classification": "existing",\n'
        prompt += '                "references": [\n'
        prompt += '                    "Newton (1687) Philosophiae Naturalis Principia Mathematica [First comprehensive theory of gravity, Newtonian gravity]",\n'
        prompt += '                    "Einstein (1915) Die Feldgleichungen der Gravitation [General relativity, which describes gravity as a curvature of spacetime rather than a force]",\n'
        prompt += '                    "Hawking (1973) Black hole explosions? [Hawking radiation, which describes how black holes can emit radiation and eventually evaporate, challenging traditional notions of gravity and black holes]"\n'
        prompt += '                ]\n'
        prompt += '        }\n'
        prompt += '    # Add more general theories as needed\n'
        prompt += '    ],\n'
        prompt += '    "theories_specific": [\n'
        prompt += '    # Add theories that describe more specific aspects or sub-phenomena as needed\n'
        prompt += '    ]\n'
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"
        prompt += "# Important Notes\n"
        prompt += "- You must return a JSON response, and it must be valid JSON, or it will not be parsed.\n"
        prompt += "- You are encouraged to think and plan before writing your JSON response, but your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
        prompt += "- Values that are 'none'/'null' in the JSON response should be represented as `null` in the JSON, not as an empty string, string saying \"null\", or any other value.\n"
        prompt += "- Where possible, you should prefer generating new, impactful theories.\n"
        prompt += "- All your information must be accurate. Do not hallucinate.\n"

        return prompt


    total_cost = 0.0

    # Create the prompt
    prompt = mkPrompt(query, results, include_query_in_generation=include_query_in_generation)

    # Call the LLM
    #responseJSON, responseText, cost = def getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True):
    all_responses = []
    MAX_GENERATION_TIME_SECS = 60 * 8    # 8 minutes
    responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=MAX_GENERATION_TIME_SECS)
    all_responses.append(responseJSON)
    total_cost += cost

    # New one-theory-at-a-time reflection approach
    reflected_response = None
    if (use_reflection):
        # First, get the list of theories from the initial response
        theories = []
        if (responseJSON is not None) and ("theories_general" in responseJSON):
            for theory in responseJSON["theories_general"]:
                theory["theory_type_general_specific"] = "general"                      # Adding this tag helps the reflection keep track of which type of theory to categorize it as later
                theories.append(theory)

        if (responseJSON is not None) and ("theories_specific" in responseJSON):
            for theory in responseJSON["theories_specific"]:
                theory["theory_type_general_specific"] = "specific"
                theories.append(theory)

        reflected_theories_general = []
        reflected_theories_specific = []
        reflected_theories_other = []

        # PARALLEL VERSION:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        MAX_THEORY_GENERATION_WORKERS = 5
        with ThreadPoolExecutor(max_workers=MAX_THEORY_GENERATION_WORKERS) as executor:
            future_to_theory_idx = {}
            for idx, theory in enumerate(theories):
                # Create a reflection prompt for this theory
                theory_str = json.dumps(theory, indent=4)
                reflection_prompt = mkPrompt(query, results, reflection=theory_str, include_query_in_generation=include_query_in_generation)

                # Submit the LLM call to the executor
                future = executor.submit(getLLMResponseJSON, reflection_prompt, model_str, temperature, max_tokens, False)
                future_to_theory_idx[future] = (idx, theory)

            # Process the completed futures
            from tqdm import tqdm
            for future in tqdm(as_completed(future_to_theory_idx), total=len(future_to_theory_idx), desc="Reflecting on theories"):
                idx, theory = future_to_theory_idx[future]
                try:
                    responseJSON, responseText, cost = future.result()
                    total_cost += cost

                    # If we got a valid response, extract the theory and add it to the list
                    if (responseJSON is not None):
                        # Try to extract the single theory from the response
                        num_added = 0
                        if ("theories_general" in responseJSON) and (len(responseJSON["theories_general"]) > 0):
                            for theory_candidate in responseJSON["theories_general"]:
                                # Add the idx of the original theory as a field to help with tracking/debugging
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_general.append(theory_candidate)
                                num_added += 1
                        elif ("theories_specific" in responseJSON) and (len(responseJSON["theories_specific"]) > 0):
                            for theory_candidate in responseJSON["theories_specific"]:
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_specific.append(theory_candidate)
                                num_added += 1

                        # If we couldn't extract a theory, just use the original one
                        if (num_added == 0):
                            if (theory["theory_type_general_specific"] == "general"):
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_general.append(theory)
                            elif (theory["theory_type_general_specific"] == "specific"):
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_specific.append(theory)
                            else:
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_other.append(theory)
                    else:
                        # If we didn't get a valid response, just use the original one
                        if (theory["theory_type_general_specific"] == "general"):
                            theory_candidate["reflected_from_theory_index"] = idx
                            reflected_theories_general.append(theory)
                        elif (theory["theory_type_general_specific"] == "specific"):
                            theory_candidate["reflected_from_theory_index"] = idx
                            reflected_theories_specific.append(theory)
                        else:
                            theory_candidate["reflected_from_theory_index"] = idx
                            reflected_theories_other.append(theory)

                except Exception as e:
                    print(f"build_theory_from_results_single_theory_reflection(): ERROR processing theory index {idx}: {e}")
                    # If we hit an error, just use the original theory
                    if (theory["theory_type_general_specific"] == "general"):
                        reflected_theories_general.append(theory)
                    elif (theory["theory_type_general_specific"] == "specific"):
                        reflected_theories_specific.append(theory)
                    else:
                        reflected_theories_other.append(theory)

        # So we don't break the pipeline that expects there to be only two bins (general and specific), we will just merge any "other" theories into the general bin.
        for theory in reflected_theories_other:
            reflected_theories_general.append(theory)

        # Sort by `reflected_from_theory_index` to preserve original order as much as possible.  If the field is missing, put those at the end.
        def get_reflection_index(theory):
            if ("reflected_from_theory_index" in theory):
                return theory["reflected_from_theory_index"]
            else:
                return 1e9    # Large number to push to end
        reflected_theories_general = sorted(reflected_theories_general, key=get_reflection_index)
        reflected_theories_specific = sorted(reflected_theories_specific, key=get_reflection_index)

        # Add 'version' field to each theory
        for theory in reflected_theories_general:
            theory["version"] = version_str
        for theory in reflected_theories_specific:
            theory["version"] = version_str


        # Now, build the final reflected response
        if (len(reflected_theories_general) + len(reflected_theories_specific) > 0):
            reflected_response = {
                "theories_general": reflected_theories_general,
                "theories_specific": reflected_theories_specific
            }
            all_responses.append(reflected_response)

    # Get the best response (the last one that's not None)
    best_response = None
    for response in reversed(all_responses):
        if (response is not None):
            best_response = response
            break

    # Pack output
    packed = {
        "query": query,
        "model": model_str,
        "use_reflection": use_reflection,
        "theory_response": best_response,
        "results": results,
        "raw_responses": all_responses,
        "total_cost": total_cost,
    }

    return packed


# Parametric-only version of the above function
def build_theory_from_results_single_theory_reflection3_llm_baseline_no_evidence(query:str, original_theory_id:str, original_theory_name:str, provide_matched_control_thery_name:bool=False, model_str="gpt-4o-mini", max_tokens=12000, temperature=0.0, use_reflection:bool=False, include_query_in_generation:bool=True):
    #version_str = "built-theory-from-results-single-theory-reflection3-dec10-2025-LLM-BASELINE-NO-EVIDENCE"
    version_str = "theorizer-parametric-only-accuracy-focused-12102025"
    matched_control_theory_name:str=None

    if (provide_matched_control_thery_name == True):
        version_str += "-with-matched-control-theory-name"
        matched_control_theory_name = original_theory_name
    else:
        version_str += "-without-matched-control-theory-name"


    def mkPrompt(query:str, reflection:str=None, include_query_in_generation:bool=True, matched_control_theory_name:str=None) -> str:
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"

        prompt += "# Task\n"
        prompt += "Your task is to construct a theory of now a scientific phenomenon works.\n" #, based on a collection of evidence that you will be provided.\n"
        prompt += "## Subtask 1: General Theories\n"
        prompt += "In the first subtask, you will be asked to provide broad, high-level theories that explain the phenomenon.\n" #, based on evidence provided.\n"
        prompt += "## Subtask 2: Specific Theories\n"
        prompt += "In the second subtask, you will be asked to provide more specific, more granular theories, that may only explain a subset of the phenomenon.\n" #/data provided.\n"
        prompt += "\n"

        if (matched_control_theory_name is not None):
            prompt += "## Specific Task\n"
            prompt += "More specifically, in this run, you are asked to generate theories all with the following name:\n"
            prompt += "```\n"
            prompt += matched_control_theory_name + "\n"
            prompt += "```\n"
            prompt += "\n"
            if (len(matched_control_theory_name) < 2):
                print("WARNING: build_theory_from_results_single_theory_reflection_llm_baseline_no_evidence(): matched_control_theory_name is very short: '" + matched_control_theory_name + "'")


        prompt += "# Total theories to generate\n"
        prompt += "You should generate 2 general theories, and 2 specific theories (i.e. 4 theories total).\n"
        prompt += "\n"

        prompt += "# Theory Quality\n"
        prompt += "You should focus on generating high-quality laws and theories that are (a) genuinely novel and insightful, (b) impactful, and (c) accurate. The goal is discovery, not rehashing already known phenomena.\n"
        prompt += "\n"

        prompt += "# What is a theory?\n"
        prompt += "A theory (from the perspective of philosophy of science, in the context of Kuhn, or later work by Pat Langley and collaborators) is a collection of statements that together explain a phenomenon, and provide predictive power for describing what will happen with future/unseen observations. In this way they are related to models.\n"
        prompt += "Theories can include both qualitative and quantitative statements/components. For example, an equation (like `force = mass * acceleration`) can be part of a theory, but so can qualitative statements (i.e. `objects that are thrown upwards will eventually fall back down to a planet due to the force of gravity, unless they are expelled at such a speed that they can escape that planet's gravity`).\n"
        prompt += "\n"

        prompt += "# Abstracting Laws/Statements\n"
        prompt += "The theories that you generate should contain (nominally, novel) qualitative and/or quantitative laws, and these laws should be explicitly abstracted from the supporting evidence.\n"
        prompt += "For example, if (in the supporting evidence) you see groups of results (across papers) that show that as certain variables change, others change in specific ways, you should try to abstract these relationships into qualitative or quantitative laws/statements that describe how these variables relate to each other.\n"
        prompt += "These laws/statements should be explicitly supported by the evidence you have been provided.\n"
        prompt += "\n"
        prompt += "## Inferring laws through logical deduction, abduction, and induction\n"
        prompt += "You can also use logical deduction, abduction, and induction to infer new laws/statements that are not explicitly stated in the evidence, but can be logically inferred from it.\n"
        prompt += "Similarly, you can use these reasoning techniques to combine laws/statements that are explicitly stated in the evidence to derive new laws/statements that are not explicitly stated, but follow logically from them.  If you do this, list this explicitly in the 'supporting evidence' for each law made in this way (with the UUIDs as an empty list). For example: 'supporting evidence: This law/statement was logically inferred through deduction/abduction/induction from law/statement 3 (xyz) and law/statement 5 (abc) above'.\n"
        prompt += "\n"

        prompt += "# Format of Laws / Theory Statements\n"
        prompt += "The theory statements that you provide must generally be framed as qualitative or quantitative laws/statements.\n"
        prompt += "Qualitative laws/statements describe relationships between variables or phenomena in a non-numeric way (e.g. X causes Y, or increasing A leads to a decrease in B, or objects with property P tend to exhibit behavior Q under conditions C).\n"
        prompt += "Quantitative laws/statements describe relationships between variables or phenomena in a numeric way (e.g. equations, proportionalities, statistical relationships, etc.).\n"
        prompt += "\n"
        prompt += "## Domain/Scope\n"
        prompt += "Each theory statement must also include a precise description of the domain/scope of the law/statement that it applies to.  For example, Newtonian gravity applies to macroscopic objects at non-relativistic speeds, but not to quantum-scale objects or objects moving near the speed of light.  This domain/scope description should be clear, and information-dense.\n"
        prompt += "Having an appropriate domain/scope is critical for ensuring that the law/statement is properly contextualized, and that its applicability is clear, particularly for evaluating the accuracy and impact of the law/statement. If the domain/scope is too broad, then the law/statement will look less accurate than it actually is, and if it is too narrow, then the law/statement will look less impactful than it actually is.\n"
        prompt += "Laws/statements are evaluated on literature.  You could imagine that if you defined Newtonian Gravity as a theory, then evaluated it on recent papers (but omiting the scope that it only applies to macroscopic objects at non-relativistic speeds), it would look inaccurate -- even though it's one of the most historically successful theories -- because many recent papers discuss relativistic or quantum effects that Newtonian gravity does not account for.  However, if you include the proper scope/domain, then it becomes clear that Newtonian gravity is accurate within its domain/scope, and thus the law/statement is more accurate than it would appear with an inappropriate domain/scope.\n"
        prompt += "Note that 'precise' domains don't mean you need to very precisely define everything (e.g. having to define a particular numerical range for 'non-relativistic speeds', which would be difficult -- the accuracy of Newtonian gravity degrades gradually as speeds approach the speed of light, so there is no hard cutoff), but rather that you should be as clear, faithful, accurate, and information-dense as possible in describing the domain/scope.\n"
        prompt += "\n"
        #prompt += "For ease of interpretation and evaluation, they must be expressed as a form of conditional expression (i.e. IF ... THEN ...), in a format that looks like a triple.\n"
        prompt += "More specifically, each theory statement must be in the following format:\n"
        # Expressed as triples, and in JSON
        prompt += "In JSON, this would look like:\n"
        prompt += "```\n"
        prompt += "{\n"
        #prompt += '    "law_name": "...", # A short name for the law\n'
        prompt += '    "statement_name": "...", # A short name for the statement/law\n'
        prompt += '    "theory_statement": "...", # The full text of the law/statement, in a concise, clear, and information-dense manner.\n'
        prompt += '    "domain_scope": "...", # A precise, clear, and information-dense description of the domain/scope of the law/statement\n'
        prompt += '    "special_cases`(list): A list of any special cases, exceptions, or boundary conditions that apply to the law/statement, if any.\n'
        # prompt += '    "if": [\n'
        # prompt += '        {"subject": "...", "relation": "...", "object": "..."}, # One triple\n'
        # prompt += '        {"subject": "...", "relation": "...", "object": "..."}, # AND, second triple\n'
        # prompt += '        # ... more triples as needed\n'
        # prompt += '    ],\n'
        # prompt += '    "then": [\n'
        # prompt += '        {"subject": "...", "relation": "...", "object": "..."}, # One triple\n'
        # prompt += '        {"subject": "...", "relation": "...", "object": "..."}, # AND, second triple\n'
        # prompt += '        # ... more triples as needed\n'
        # prompt += '    ]\n'
        # Supporting evidence
        prompt += '    "supporting_evidence": [\n'
        prompt += '        {"text": "...", "uuids": [...]}, # One piece of supporting evidence\n'
        prompt += '        {"text": "...", "uuids": [...]}, # Second piece of supporting evidence\n'
        prompt += '        # ... more pieces of supporting evidence as available\n'
        prompt += '    ]\n'
        # Qual/qant
        prompt += '    "qual_or_quant": "..." # Is this law/statement primarily "qualitative", "quantitative", or "mixed"\n'
        # Existing law
        prompt += '    "novelty_evaluation": {\n'
        prompt += '        "what_already_exists": "...", # brief, information-dense, 1-2 sentence explanation of what in this law/statement is already known in previous work",\n'
        prompt += '        "what_is_novel": "...", # brief, information-dense, 1-2 sentence explanation of what SIGNIFICANT aspects of this law/statement are novel",\n'
        prompt += '        "classification_explanation": "..." # A brief, concise, information-dense, 1-2 sentence explanation as to why this classification was made, referencing existing/novel work>"\n'
        prompt += '        "likely_classification": "...", # One of `existing`, `closely-related-to-existing`, `somewhat-related-to-existing`, or `new`\n'
        prompt += '        "references": [...], # A list of relevant references here. Format: Citation, then square brackets very briefly describing how it relates to the law.\n'
        prompt += '    }\n'
        prompt += '}\n'
        prompt += "```\n"
        #prompt += "The purpose of framing the statements/laws in this way is to make it easy to evaluate and compare across theories. The subjects, relations, and objects should be concise, but also information rich and detailed.\n"
        #prompt += "Each triple represents a single assertion about the relationship between a subject and an object, via a relation.\n"
        #prompt += "Multiple triples in the `if` section are combined with AND logic (i.e. all conditions must be met), and multiple triples in the `then` section are also combined with AND logic (i.e. all outcomes occur if the conditions are met).\n"
        prompt += "\n"


        prompt += "# Query\n"
        prompt += "Here is the general area/query that your theories should be centered around:"
        if (include_query_in_generation):
            prompt += "```\n"
            prompt += query + "\n"
            prompt += "```\n"
            prompt += "\n"
        else:
            prompt += "No specific query is provided. You can build theories as you see fit, based on the supporting evidence.\n"
            prompt += "\n"

        # If this isn't set, just give it the raw data and ask it to brainstorm in a data-driven way rather than in a query-driven way
        prompt += "For supporting evidence when building your theories, you should use evidence that is already known to you. If you are asked for `uuids` for this evidence, just leave them blank.\n"
        prompt += "\n"
        # prompt += "```\n"
        # prompt += json.dumps(results, indent=4) + "\n"
        # prompt += "```\n"
        # prompt += "\n"


        if (reflection is not None):
            prompt += "\n"
            prompt += "# Reflection\n"
            prompt += "This is a reflection step. Previously, you generated a response (below).  Now, your task is to reflect on that response, and fix any errors, inconsistencies, omissions, or any other issues.\n"
            prompt += "NOTE: This reflection is focused on a single theory that you generated previously. Your task is to fix any issues with that single theory, and ensure it is as complete, accurate, and well-supported as possible.\n"
            prompt += "This includes filling in missing evidence present in the extracted evidence but not included in the theory's list of supporting evidence for each law, conflicting evidence, special cases, etc., as well as fixing any errors, inconsistencies, omissions, or other issues you find.\n"
            prompt += "You should also pay particular attention to ensuring that the domain/scope of each law/statement is accurate and appropriately scoped, as well as clear, detailed, and information-dense.\n"
            prompt += "For example, the list of supporting evidence shouldn't be limited to some small amount (e.g. 5 per law) -- it should be as exhaustive as possible, and include all genuinely relevant evidence provided in the results.\n"
            prompt += "\n"
            prompt += "```\n"
            prompt += reflection + "\n"
            prompt += "```\n"

        prompt += "# Output Format\n"
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
        prompt += "\n"
        prompt += "Your JSON response must be a dictionary, that contains two keys: `theories_general`, and `theories_specific`.  Each is a list of dictionaries, with the following keys:\n"
        prompt += "- `theory_name`(str): A short name for the theory\n"
        prompt += "- `theory_description`(str): A full description of the theory\n"
        #prompt += "- `supporting_evidence`(list): A list of specific supporting evidence for the theory.\n"
        #prompt += "- `theory_statements`(list): A list of specific predictive statements (either qualitative, quantitative, or both) that describe the assertions of the theory.\n"
        prompt += "- `theory_statements`(list): A list of specific laws (either qualitative, quantitative, or both), in the dictionary format described above, that describe the assertions of the theory.\n"
        prompt += "- `new_predictions_likely`(list): A list of several example (testable) predictions that the theory makes (the situation, and the expected results) that are not in the observed evidence. For this set of new predictions, the predictions should be fairly likely to work out.\n"
        prompt += "- `new_predictions_unknown`(list): A list of several example (testable) predictions that the theory makes (the situation, and the expected results) that are not in the observed evidence. For this set of new predictions, the predictions should be difficult to tell if they would genuinely work out or not, and have particularly impactful results if they do (or don't) work out.\n"
        prompt += "- `negative_experiments`(list): A list of several (testable) predictions that test the assumptions or statements of the theory, and if false, would call the theory statements into question.\n"
        prompt += "- `unaccounted_for`(list): A list of specific pieces of evidence that are not explained by the theory, if any.\n"
        prompt += "- `conflicting_evidence`(list): A list of specific pieces of evidence that appear to conflict with the theory, if any.\n"
        prompt += "- `special_cases`(list): A list of any special cases, exceptions, or boundary conditions that apply to the theory, if any.\n"
        prompt += "- `existing_theory`(dict): A dictionary that describes whether (to the best of your knowledge, and the information available in the papers) if this is a pre-existing theory, or a new theory. It has the following keys: `likely_classification`:str, which is one of `new`, `somewhat-related-to-existing`, `closely-related-to-existing`, or `existing`, and `references`:list(str), which are lists of references (one per string) to new (or related) theories in the form `First author (year) title [brief note of how related, and name(s) of relevant theories, in square brackets]`. These should help confirm the theory is new, or help confirm it is existing or closely related. It also has: `what_already_exists`(str), `what_is_novel`(str), and `classification_explanation`(str).\n"
        prompt += "\n"
        prompt += "While most fields are lists of strings, for `supporting_evidence` and `unaccounted_for`, the list should be a list of dictionaries, where each dictionary has the following keys:\n"
        prompt += "- `text`(str): A textual description of the evidence\n"
        prompt += "- `uuids`(list): A list of UUIDs that correspond to the specific pieces of evidence in the original results that support this text description. These lists can be as long or short as they need to be to encompass all the correct evidence references.\n"
        prompt += "\n"
        #prompt += "Similarly, the `theory_statements` field is a list of dictionaries, where each dictionary has the following key:\n"
        #prompt += "- `law`(dict): A dictionary that describes a single qualitative or quantitative law, in the format described above (i.e. with `if` and `then` fields, each containing a list of triples), and specific supporting evidence that supports the development of this law.\n"  # Modified from original to remove the 'if' and 'then' references
        prompt += "\n"
        prompt += "For example (a cartoon example -- your theories should be much more detailed, and have much more supporting evidence, linked back with proper UUIDs):\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += "    \"theories_general\": [\n"
        prompt += '        {\n'
        prompt += '            "theory_name": "Theory of Gravity",\n'
        prompt += '            "theory_description": "A theory that explains the force of gravity as a result of mass attracting other mass.",\n'
        # prompt += '            "supporting_evidence": [\n'
        # prompt += '                {\"text\": "Objects fall towards the Earth when dropped.", \"uuids\": [1, 5]},\n'
        # prompt += '                {\"text\": "The Moon orbits the Earth due to gravitational attraction.", \"uuids\": [2, 3, 8]},\n'
        # prompt += '                {\"text\": "The planets orbit the Sun due to the Sun\'s gravitational pull.", \"uuids\": [6, 10, 15, 18, 21]},\n'
        # prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        # prompt += '            ],\n'
        # New format
        prompt += '            "theory_statements": [\n'
        prompt += '                {\n'
        prompt += '                  "statement_name": "Mutual Attraction of Massive Objects",\n'
        prompt += '                  "theory_statement": "Objects with mass exert a mutual attractive force on each other, proportional to their masses.",\n'
        prompt += '                  "domain_scope": "Applies to macroscopic objects with mass in a classical physics context, excluding relativistic and quantum effects.",\n'
        prompt += '                  "special_cases": [\n'
        prompt += '                      "Does not apply to massless particles like photons.",\n'
        prompt += '                      "Breaks down at quantum scales and near singularities."\n'
        prompt += '                      # ... add more special cases as needed\n'
        prompt += '                  ],\n'
        prompt += '                  "supporting_evidence": [\n'
        prompt += '                      { "text": "Dropped objects fall toward Earth.", "uuids": [...] },\n'
        prompt += '                      { "text": "The Moon remains bound to Earth through mutual attraction.", "uuids": [...] }\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        prompt += '                  "qual_or_quant": "qualitative",\n'
        prompt += '                  "novelty_evaluation": {\n'
        prompt += '                        "what_already_exists": "<brief, information-dense, 1-2 sentence explanation of what in this law is already known in previous work>",\n'
        prompt += '                        "what_is_novel": "<brief, information-dense, 1-2 sentence explanation of what in this law is novel>",\n'
        prompt += '                        "classification_explanation": "<brief, concise, information-dense, 1-2 sentence explanation as to why this classification was made, referencing existing/novel work>",\n'
        prompt += '                        "likely_classification": "<existing|closely-related-to-existing|somewhat-related-to-existing|new>",\n'
        prompt += '                        "references": [\n'
        prompt += '                            # Add relevant references here\n'
        prompt += '                        ]\n'
        prompt += '                  }\n'
        prompt += '                },\n'
        # Next statement/law
        prompt += '                {\n'
        prompt += '                  "statement_name": "Inverse Square Law of Gravitational Force",\n'
        prompt += '                  "theory_statement": "The gravitational force between two masses is inversely proportional to the square of the distance between them.",\n'
        prompt += '                  "domain_scope": "Applies to point masses or spherically symmetric masses in a classical physics context, excluding relativistic and quantum effects.",\n'
        prompt += '                  "special_cases": [\n'
        prompt += '                      "Breaks down at very small scales where quantum effects dominate.",\n'
        prompt += '                      "Requires modification in strong gravitational fields as described by General Relativity."\n'
        prompt += '                      # ... add more special cases as needed\n'
        prompt += '                  ],\n'
        prompt += '                  "supporting_evidence": [\n'
        prompt += '                      { "text": "The motion of planets and moons follows an inverse square relationship.", "uuids": [...] },\n'
        prompt += '                      { "text": "Laboratory experiments measuring gravitational attraction confirm the inverse square law.", "uuids": [...] }\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        prompt += '                  "qual_or_quant": "quantitative",\n'
        prompt += '                  "novelty_evaluation": {\n'
        prompt += '                        "what_already_exists": "<brief, information-dense, 1-2 sentence explanation of what in this law is already known in previous work>",\n'
        prompt += '                        "what_is_novel": "<brief, information-dense, 1-2 sentence explanation of what in this law is novel>",\n'
        prompt += '                        "classification_explanation": "<brief, concise, information-dense, 1-2 sentence explanation as to why this classification was made, referencing existing/novel work>",\n'
        prompt += '                        "likely_classification": "<existing|closely-related-to-existing|somewhat-related-to-existing|new>",\n'
        prompt += '                        "references": [\n'
        prompt += '                            # Add relevant references here\n'
        prompt += '                        ]\n'
        prompt += '                  }\n'
        prompt += '                },\n'
        # Next statement/law
        # This time, do a quantitative law
        prompt += '                {\n'
        prompt += '                  "statement_name": "Gravitational Force Proportional to Product of Masses",\n'
        prompt += '                  "theory_statement": "The gravitational force between two objects is directly proportional to the product of their masses.",\n'
        prompt += '                  "domain_scope": "Applies to macroscopic objects with mass in a classical physics context, excluding relativistic and quantum effects.",\n'
        prompt += '                  "special_cases": [\n'
        prompt += '                      "Does not apply to massless particles like photons.",\n'
        prompt += '                      "Breaks down at quantum scales and near singularities."\n'
        prompt += '                      # ... add more special cases as needed\n'
        prompt += '                  ],\n'
        prompt += '                  "supporting_evidence": [\n'
        prompt += '                      { "text": "Experiments show that increasing the mass of either object increases the gravitational force between them.", "uuids": [...] },\n'
        prompt += '                      { "text": "Observations of planetary motion confirm the relationship between mass and gravitational force.", "uuids": [...] }\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        prompt += '                  "qual_or_quant": "quantitative",\n'
        prompt += '                  "novelty_evaluation": {\n'
        prompt += '                        "what_already_exists": "<brief, information-dense, 1-2 sentence explanation of what in this law is already known in previous work>",\n'
        prompt += '                        "what_is_novel": "<brief, information-dense, 1-2 sentence explanation of what in this law is novel>",\n'
        prompt += '                        "classification_explanation": "<brief, concise, information-dense, 1-2 sentence explanation as to why this classification was made, referencing existing/novel work>",\n'
        prompt += '                        "likely_classification": "<existing|closely-related-to-existing|somewhat-related-to-existing|new>",\n'
        prompt += '                        "references": [\n'
        prompt += '                            # Add relevant references here\n'
        prompt += '                        ]\n'
        prompt += '                  }\n'
        prompt += '                }\n'
        prompt += '                # ... add more laws/statements as needed\n'
        prompt += '            ],\n'
        prompt += '            "new_predictions_likely": [\n'
        prompt += '                "If a large object is placed in space, it will attract smaller objects towards it due to gravity.",\n'
        prompt += '                "If two objects are in space, they will eventually collide if they are on a collision course due to gravitational attraction."\n'
        prompt += '            ],\n'
        prompt += '            "new_predictions_unknown": [\n'
        prompt += '                "If a massive object were to suddenly disappear, the gravitational effects on nearby objects would be unknown.",\n'
        prompt += '                "The behavior of gravity in extreme conditions, such as near black holes, is not fully understood."\n'
        prompt += '                # ... this should be as detailed/exhaustive as possible. Add as many elements as needed.\n'
        prompt += '            ],\n'
        prompt += '            "negative_experiments": [\n'
        prompt += '                "Finding cases where two massive objects do not attract each other despite being in close proximity would call into question the theory of gravity.",\n'
        prompt += '                "Finding cases where the force of gravity does not follow the inverse square law would call into question the theory of gravity."\n'
        prompt += '            ],\n'
        prompt += '            "unaccounted_for": [\n'
        prompt += '                {\"text\": "The exact mechanism of how gravity works at the quantum level is not fully understood.", \"uuids\": [7, 25]},\n'
        prompt += '                {\"text\": "The behavior of gravity in extreme conditions, such as near black holes, is not fully understood.", \"uuids\": [12, 28]}\n'
        prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '            ],\n'
        prompt += '            "conflicting_evidence": [\n'
        prompt += '                {\"text\": "Observations of galaxy rotation curves that do not match predictions based on visible matter challenge the theory of gravity as currently understood.", \"uuids\": [30, 32]},\n'
        prompt += '                {\"text\": "The accelerated expansion of the universe challenges the traditional understanding of gravity.", \"uuids\": [35, 40]}\n'
        prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '            ],\n'
        prompt += '            "special_cases": [\n'
        prompt += '                "In the presence of extremely strong gravitational fields, such as near black holes, the predictions of general relativity may differ from Newtonian gravity.",\n'
        prompt += '                "At very small scales, quantum effects may alter the behavior of gravity."\n'
        prompt += '                # ... add more special cases as needed\n'
        prompt += '            ],\n'
        prompt += '            "existing_theory": { # Whole-theory-related classification\n'
        prompt += '                "what_already_exists": "...",\n'
        prompt += '                "what_is_novel": "...",\n'
        prompt += '                "classification_explanation": "..." # A brief, concise, information-dense, 1-2 sentence explanation as to why this classification was made, specifically referencing existing/novel work\n'
        prompt += '                "likely_classification": "existing",\n'
        prompt += '                "references": [\n'
        prompt += '                    "Newton (1687) Philosophiae Naturalis Principia Mathematica [First comprehensive theory of gravity, Newtonian gravity]",\n'
        prompt += '                    "Einstein (1915) Die Feldgleichungen der Gravitation [General relativity, which describes gravity as a curvature of spacetime rather than a force]",\n'
        prompt += '                    "Hawking (1973) Black hole explosions? [Hawking radiation, which describes how black holes can emit radiation and eventually evaporate, challenging traditional notions of gravity and black holes]"\n'
        prompt += '                ]\n'
        prompt += '        }\n'
        prompt += '    # Add more general theories as needed\n'
        prompt += '    ],\n'
        prompt += '    "theories_specific": [\n'
        prompt += '    # Add theories that describe more specific aspects or sub-phenomena as needed\n'
        prompt += '    ]\n'
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"
        prompt += "# Important Notes\n"
        prompt += "- You must return a JSON response, and it must be valid JSON, or it will not be parsed.\n"
        prompt += "- You are encouraged to think and plan before writing your JSON response, but your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
        prompt += "- Values that are 'none'/'null' in the JSON response should be represented as `null` in the JSON, not as an empty string, string saying \"null\", or any other value.\n"
        prompt += "- Where possible, you should prefer generating new, impactful theories.\n"
        prompt += "- All your information must be accurate. Do not hallucinate.\n"

        return prompt


    total_cost = 0.0

    # Create the prompt
    prompt = mkPrompt(query, include_query_in_generation=include_query_in_generation, matched_control_theory_name=matched_control_theory_name)

    # Call the LLM
    #responseJSON, responseText, cost = def getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True):
    all_responses = []
    MAX_GENERATION_TIME_SECS = 60 * 8    # 8 minutes
    responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=MAX_GENERATION_TIME_SECS)
    all_responses.append(responseJSON)
    total_cost += cost

    # New one-theory-at-a-time reflection approach
    reflected_response = None
    if (use_reflection):
        # First, get the list of theories from the initial response
        theories = []
        if (responseJSON is not None) and ("theories_general" in responseJSON):
            for theory in responseJSON["theories_general"]:
                theory["theory_type_general_specific"] = "general"                      # Adding this tag helps the reflection keep track of which type of theory to categorize it as later
                theories.append(theory)

        if (responseJSON is not None) and ("theories_specific" in responseJSON):
            for theory in responseJSON["theories_specific"]:
                theory["theory_type_general_specific"] = "specific"
                theories.append(theory)

        reflected_theories_general = []
        reflected_theories_specific = []
        reflected_theories_other = []

        # PARALLEL VERSION:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        MAX_THEORY_GENERATION_WORKERS = 5
        with ThreadPoolExecutor(max_workers=MAX_THEORY_GENERATION_WORKERS) as executor:
            future_to_theory_idx = {}
            for idx, theory in enumerate(theories):
                # Create a reflection prompt for this theory
                theory_str = json.dumps(theory, indent=4)
                reflection_prompt = mkPrompt(query, reflection=theory_str, include_query_in_generation=include_query_in_generation, matched_control_theory_name=matched_control_theory_name)

                # Submit the LLM call to the executor
                future = executor.submit(getLLMResponseJSON, reflection_prompt, model_str, temperature, max_tokens, False)
                future_to_theory_idx[future] = (idx, theory)

            # Process the completed futures
            from tqdm import tqdm
            for future in tqdm(as_completed(future_to_theory_idx), total=len(future_to_theory_idx), desc="Reflecting on theories"):
                idx, theory = future_to_theory_idx[future]
                try:
                    responseJSON, responseText, cost = future.result()
                    total_cost += cost

                    # If we got a valid response, extract the theory and add it to the list
                    if (responseJSON is not None):
                        # Try to extract the single theory from the response
                        num_added = 0
                        if ("theories_general" in responseJSON) and (len(responseJSON["theories_general"]) > 0):
                            for theory_candidate in responseJSON["theories_general"]:
                                # Add the idx of the original theory as a field to help with tracking/debugging
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_general.append(theory_candidate)
                                num_added += 1
                        elif ("theories_specific" in responseJSON) and (len(responseJSON["theories_specific"]) > 0):
                            for theory_candidate in responseJSON["theories_specific"]:
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_specific.append(theory_candidate)
                                num_added += 1

                        # If we couldn't extract a theory, just use the original one
                        if (num_added == 0):
                            if (theory["theory_type_general_specific"] == "general"):
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_general.append(theory)
                            elif (theory["theory_type_general_specific"] == "specific"):
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_specific.append(theory)
                            else:
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_other.append(theory)
                    else:
                        # If we didn't get a valid response, just use the original one
                        if (theory["theory_type_general_specific"] == "general"):
                            theory_candidate["reflected_from_theory_index"] = idx
                            reflected_theories_general.append(theory)
                        elif (theory["theory_type_general_specific"] == "specific"):
                            theory_candidate["reflected_from_theory_index"] = idx
                            reflected_theories_specific.append(theory)
                        else:
                            theory_candidate["reflected_from_theory_index"] = idx
                            reflected_theories_other.append(theory)

                except Exception as e:
                    print(f"build_theory_from_results_single_theory_reflection(): ERROR processing theory index {idx}: {e}")
                    # If we hit an error, just use the original theory
                    if (theory["theory_type_general_specific"] == "general"):
                        reflected_theories_general.append(theory)
                    elif (theory["theory_type_general_specific"] == "specific"):
                        reflected_theories_specific.append(theory)
                    else:
                        reflected_theories_other.append(theory)

        # So we don't break the pipeline that expects there to be only two bins (general and specific), we will just merge any "other" theories into the general bin.
        for theory in reflected_theories_other:
            reflected_theories_general.append(theory)

        # Sort by `reflected_from_theory_index` to preserve original order as much as possible.  If the field is missing, put those at the end.
        def get_reflection_index(theory):
            if ("reflected_from_theory_index" in theory):
                return theory["reflected_from_theory_index"]
            else:
                return 1e9    # Large number to push to end
        reflected_theories_general = sorted(reflected_theories_general, key=get_reflection_index)
        reflected_theories_specific = sorted(reflected_theories_specific, key=get_reflection_index)


        # Add 'version' field to each theory
        for theory in reflected_theories_general:
            theory["type"] = "general"
            theory["version"] = version_str
            theory["theory_query"] = query
            theory["original_theory_id"] = original_theory_id
            theory["original_theory_name"] = original_theory_name
            theory["provide_matched_control_thery_name"] = provide_matched_control_thery_name
            theory["matched_control_theory_name"] = matched_control_theory_name
            theory["model_str"] = model_str
        for theory in reflected_theories_specific:
            theory["type"] = "specific"
            theory["version"] = version_str
            theory["theory_query"] = query
            theory["original_theory_id"] = original_theory_id
            theory["original_theory_name"] = original_theory_name
            theory["provide_matched_control_thery_name"] = provide_matched_control_thery_name
            theory["matched_control_theory_name"] = matched_control_theory_name
            theory["model_str"] = model_str


        # Now, build the final reflected response
        if (len(reflected_theories_general) + len(reflected_theories_specific) > 0):
            reflected_response = {
                "theories_general": reflected_theories_general,
                "theories_specific": reflected_theories_specific
            }
            all_responses.append(reflected_response)

    else:
        # Also add the 'version' field to each theory in the original response
        if (responseJSON is not None):
            if ("theories_general" in responseJSON):
                for theory in responseJSON["theories_general"]:
                    theory["type"] = "general"
                    theory["version"] = version_str
                    theory["theory_query"] = query
                    theory["original_theory_id"] = original_theory_id
                    theory["original_theory_name"] = original_theory_name
                    theory["provide_matched_control_thery_name"] = provide_matched_control_thery_name
                    theory["matched_control_theory_name"] = matched_control_theory_name
                    theory["model_str"] = model_str
            if ("theories_specific" in responseJSON):
                for theory in responseJSON["theories_specific"]:
                    theory["type"] = "specific"
                    theory["version"] = version_str
                    theory["theory_query"] = query
                    theory["original_theory_id"] = original_theory_id
                    theory["original_theory_name"] = original_theory_name
                    theory["provide_matched_control_thery_name"] = provide_matched_control_thery_name
                    theory["matched_control_theory_name"] = matched_control_theory_name
                    theory["model_str"] = model_str


    # Get the best response (the last one that's not None)
    best_response = None
    for response in reversed(all_responses):
        if (response is not None):
            best_response = response
            break

    # Pack output
    packed = {
        "query": query,
        "model": model_str,
        "use_reflection": use_reflection,
        "theory_response": best_response,
        #"results": results,
        "raw_responses": all_responses,
        "total_cost": total_cost,
    }

    return packed



#
#   Novelty-focused versions
#

def build_theory_from_results_single_theory_reflection4_nonsafebasin(query:str, results:list, model_str="gpt-4o-mini", max_tokens=12000, temperature=1.0, use_reflection:bool=False, include_query_in_generation:bool=True):
    #version_str = "built-theory-from-results-single-theory-reflection4-dec27-2025-NONSAFEBASIN"
    version_str = "theorizer-literature-supported-novelty-focused-12272025"

    # This prompt re-cooked from the accuracy-focused objective to generate less conservative theories that break out of the 'safe basin' of prior knowledge. (with GPT-5.2 help)
    def mkPrompt(query: str, results:str, reflection:str=None, include_query_in_generation:bool=True, matched_control_theory_name:str=None):
        prompt = ""
        prompt += "You are ScientistGPT, an AI scientist.\n"
        prompt += "You must prioritize falsifiable, high-information theories over safe or generic ones.\n"
        prompt += "Scientific integrity here means: precise commitments, explicit scope, and explicit falsifiers — not conservatism.\n"
        prompt += "If evidence is provided, you must ground claims in it. If evidence is not provided, you must clearly label claims as background principles or inferences.\n\n"

        prompt += "# Task\n"
        prompt += "Your task is to construct a theory of how a scientific phenomenon works.\n"
        prompt += "## Subtask 1: General Theories\n"
        prompt += "In the first subtask, you will be asked to provide broad, high-level theories that explain the phenomenon.\n"
        prompt += "## Subtask 2: Specific Theories\n"
        prompt += "In the second subtask, you will be asked to provide more specific, more granular theories, that may only explain a subset of the phenomenon.\n"
        prompt += "\n"

        if matched_control_theory_name is not None:
            prompt += "## Specific Task\n"
            prompt += "More specifically, in this run, you are asked to generate theories all with the following name:\n"
            prompt += "```\n"
            prompt += matched_control_theory_name + "\n"
            prompt += "```\n"
            prompt += "\n"
            if len(matched_control_theory_name) < 2:
                print("WARNING: mkPrompt(): matched_control_theory_name is very short: '" + matched_control_theory_name + "'")

        prompt += "# Total theories to generate\n"
        prompt += "You should generate 2 general theories, and 2 specific theories (i.e. 4 theories total).\n"
        prompt += "\n"

        prompt += "# Theory Quality\n"
        prompt += "You should focus on generating high-quality laws and theories that are (a) genuinely novel and insightful, (b) impactful, and (c) accurate within a clearly specified domain/scope.\n"
        prompt += "The goal is discovery, not rehashing already known phenomena.\n"
        prompt += "\n"

        # ------------------------------------------------------------
        # Basin-breaking additions (STRICT)
        # ------------------------------------------------------------
        prompt += "# Basin-breaking mode (STRICT)\n"
        prompt += "Definition: The 'high-probability basin' is the set of safe, generic, high-prior statements that are broadly true across many domains and time periods.\n"
        prompt += "These basin statements avoid risk by being vague, mechanism-agnostic, and compatible with many outcomes (e.g., 'X helps Y', 'better representations', 'regularization improves generalization').\n"
        prompt += "\n"
        prompt += "Goal: Break out of that basin by producing *high-information*, *falsifiable* statements that make risky, specific commitments.\n"
        prompt += "In this prompt, 'basin-breaking' means:\n"
        prompt += "- You MUST avoid generic high-prior statements.\n"
        prompt += "- You MUST commit to specific, discriminative claims that could easily be wrong.\n"
        prompt += "- You MUST state exactly what empirical outcome would refute each claim.\n"
        prompt += "\n"
        prompt += "## What counts as 'basin' (INVALID)\n"
        prompt += "A theory_statement is INVALID if it has any of the following properties:\n"
        prompt += "1) Generic truth: It would plausibly be supported by many pre-2023 papers in the field.\n"
        prompt += "2) Non-committal: It does not specify a clear boundary condition or failure regime.\n"
        prompt += "3) Non-discriminative: It does not distinguish your theory from a plausible alternative.\n"
        prompt += "4) Non-falsifiable: It does not specify a single concrete experiment and a refuting outcome.\n"
        prompt += "\n"
        prompt += "## Forbidden wording patterns (if you use these, the statement is INVALID)\n"
        prompt += "- Hedging: 'generally', 'often', 'tends to', 'in many cases', 'is associated with'.\n"
        prompt += "- Safe abstractions without discriminators: 'better representations', 'robustness', 'regularization helps', 'scaling improves'.\n"
        prompt += "- Multi-mechanism hedging: presenting multiple mechanisms as equally likely.\n"
        prompt += "\n"
        prompt += "## Required structure for EACH theory_statement (NON-NEGOTIABLE)\n"
        prompt += "Each theory_statement MUST include ALL of the following, explicitly:\n"
        prompt += "A) A FORK: commit to one side of a plausible tradeoff where the other side is also plausible.\n"
        prompt += "   Example fork forms:\n"
        prompt += "   - 'Beyond threshold T, X improves Y' vs 'Beyond threshold T, X degrades Y'\n"
        prompt += "   - 'Mechanism A dominates under condition C' vs 'Mechanism B dominates under condition C'\n"
        prompt += "B) A NUMERIC THRESHOLD or bound (rough is fine, but must be numeric).\n"
        prompt += "C) A NEGATIVE prediction (something will NOT happen / will worsen / will reverse under condition C).\n"
        prompt += "D) A DISCRIMINATING EXPERIMENT (one concrete experiment/ablation).\n"
        prompt += "E) A FALSIFIER: the specific observed outcome that would refute the statement.\n"
        prompt += "\n"
        prompt += "## No hedging rule\n"
        prompt += "Do NOT say 'it depends'. Do NOT provide multiple options. Pick one and commit.\n"
        prompt += "\n"

        prompt += "# What is a theory?\n"
        prompt += "A theory (from the perspective of philosophy of science, in the context of Kuhn, or later work by Pat Langley and collaborators) is a collection of statements that together explain a phenomenon, and provide predictive power for describing what will happen with future/unseen observations.\n"
        prompt += "Theories can include both qualitative and quantitative statements/components.\n"
        prompt += "\n"

        prompt += "# Abstracting Laws/Statements\n"
        prompt += "The theories that you generate should contain (nominally, novel) qualitative and/or quantitative laws, and these laws should be explicitly abstracted from the supporting evidence.\n"
        prompt += "If (in the supporting evidence) you see groups of results that show that as certain variables change, others change in specific ways, abstract these relationships into qualitative or quantitative laws/statements.\n"
        prompt += "These laws/statements should be explicitly supported by the evidence you have been provided.\n"
        prompt += "\n"
        prompt += "## Inferring laws through logical deduction, abduction, and induction\n"
        prompt += "You can also use logical deduction, abduction, and induction to infer new laws/statements that are not explicitly stated in the evidence, but can be logically inferred from it.\n"
        prompt += "If you do this, list this explicitly in the 'supporting evidence' for each law made in this way (with the UUIDs as an empty list).\n"
        prompt += "\n"

        # ------------------------------------------------------------
        # Evidence policy changes (STRICT)
        # ------------------------------------------------------------
        prompt += "# Evidence use policy (STRICT)\n"
        prompt += "If any supporting evidence is provided in the results, you MUST ground each theory_statement primarily in that provided evidence.\n"
        prompt += "- Each theory_statement.supporting_evidence must include at least 2 items that clearly reference the provided evidence.\n"
        prompt += "- If evidence is provided, do NOT rely primarily on general background knowledge.\n"
        prompt += "\n"
        prompt += "If NO supporting evidence is provided, you may rely on general scientific knowledge, but you must explicitly label each supporting_evidence item as either:\n"
        prompt += "- 'background_principle' (widely established), or\n"
        prompt += "- 'inference' (a reasoned conjecture).\n"
        prompt += "In the NO-evidence case, do NOT fabricate citations or pretend a specific study exists.\n"
        prompt += "\n"

        prompt += "# Format of Laws / Theory Statements\n"
        prompt += "The theory statements that you provide must generally be framed as qualitative or quantitative laws/statements.\n"
        prompt += "Qualitative laws/statements describe relationships between variables or phenomena in a non-numeric way.\n"
        prompt += "Quantitative laws/statements describe relationships between variables or phenomena in a numeric way.\n"
        prompt += "\n"

        prompt += "## Domain/Scope\n"
        prompt += "Each theory statement must also include a precise description of the domain/scope of the law/statement that it applies to.\n"
        prompt += "Having an appropriate domain/scope is critical for ensuring that the law/statement is properly contextualized and its applicability is clear.\n"
        prompt += "\n"

        prompt += "More specifically, each theory statement must be in the following format:\n"
        prompt += "In JSON, this would look like:\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += '    "statement_name": "...", # A short name for the statement/law\n'
        prompt += '    "theory_statement": "...", # The full text of the law/statement, concise, clear, information-dense, and MUST include: fork + numeric threshold + negative prediction\n'
        prompt += '    "domain_scope": "...", # Precise, clear, information-dense domain/scope\n'
        prompt += '    "special_cases": ["..."], # Special cases, exceptions, boundary conditions\n'
        prompt += '    "supporting_evidence": [\n'
        prompt += '        {"text": "...", "uuids": [...]},\n'
        prompt += '        {"text": "...", "uuids": [...]}\n'
        prompt += '    ],\n'
        prompt += '    "qual_or_quant": "qualitative|quantitative|mixed",\n'
        prompt += '    "novelty_evaluation": {\n'
        prompt += '        "what_already_exists": "...",\n'
        prompt += '        "what_is_novel": "...",\n'
        prompt += '        "classification_explanation": "...",\n'
        prompt += '        "likely_classification": "existing|closely-related-to-existing|somewhat-related-to-existing|new",\n'
        prompt += '        "references": [...]\n'
        prompt += '    },\n'
        prompt += '    "discriminating_experiment": "...", # One concrete experiment/ablation that would falsify the statement\n'
        prompt += '    "falsifies_if": "..." # A concrete expected outcome that would refute the statement\n'
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Query\n"
        prompt += "Here is the general area/query that your theories should be centered around:\n"
        if include_query_in_generation:
            prompt += "```\n"
            prompt += query + "\n"
            prompt += "```\n"
            prompt += "\n"
        else:
            prompt += "No specific query is provided. You can build theories as you see fit, based on the supporting evidence.\n"
            prompt += "\n"

        # ------------------------------------------------------------
        # NEW: Results section (supporting evidence / literature condition)
        # ------------------------------------------------------------
        prompt += "# Results (supporting evidence)\n"
        prompt += "Below are extracted results / evidence items you may use as supporting evidence.\n"
        prompt += "If this section is empty or null, then you have NO provided evidence and must rely on background_principle and inference (without inventing studies).\n"
        prompt += "\n"
        prompt += "```\n"
        if results is None:
            prompt += "null\n"
        else:
            # If someone passes an empty list/dict/string, preserve that so the model sees it's empty.
            try:
                prompt += json.dumps(results, indent=4, ensure_ascii=False) + "\n"
            except TypeError:
                # Fallback for non-JSON-serializable objects
                prompt += str(results) + "\n"
        prompt += "```\n"
        prompt += "\n"

        # Keep the legacy behavior but now consistent with strict evidence policy:
        prompt += "UUID policy: If an evidence item comes from the provided Results section, include its UUID(s) if available.\n"
        prompt += "If a supporting_evidence item is 'background_principle' or 'inference', leave uuids blank.\n"
        prompt += "\n"

        if reflection is not None:
            prompt += "\n"
            prompt += "# Reflection\n"
            prompt += "This is a reflection step. Previously, you generated a response (below). Now, reflect on that response and fix errors, inconsistencies, omissions, missing evidence links, conflicts, special cases, and scope issues.\n"
            prompt += "Ensure each statement still satisfies Basin-breaking mode: fork + numeric threshold + negative prediction + discriminating experiment + falsifier.\n"
            prompt += "\n"
            prompt += "```\n"
            prompt += reflection + "\n"
            prompt += "```\n"

        prompt += "# Output Format\n"
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).\n"
        prompt += "You may write text to think/plan before writing JSON, but the JSON must be the last thing you write, must be valid JSON, and must be inside a single set of codeblocks (```), or it will not be parsed.\n"
        prompt += "\n"

        prompt += "Your JSON response must be a dictionary with two keys: `theories_general` and `theories_specific`.\n"
        prompt += "Each is a list of dictionaries with the following keys:\n"
        prompt += "- `theory_name`(str): A short name for the theory\n"
        prompt += "- `theory_description`(str): A full description of the theory\n"
        prompt += "- `theory_statements`(list): A list of specific laws (qualitative, quantitative, or mixed) in the required format above\n"
        prompt += "- `new_predictions_likely`(list): Non-trivial, testable predictions likely within the theory's narrow scope; MUST include at least one numeric expectation\n"
        prompt += "- `new_predictions_unknown`(list): High-variance, discriminative predictions that strongly separate your theory from plausible alternatives\n"
        prompt += "- `negative_experiments`(list): MUST include at least one fork test that would decisively refute the theory if it goes the other way\n"
        prompt += "- `unaccounted_for`(list): Evidence not explained by the theory, if any\n"
        prompt += "- `conflicting_evidence`(list): Evidence that conflicts with the theory, if any\n"
        prompt += "- `special_cases`(list): Special cases/exceptions/boundary conditions, if any\n"
        prompt += "- `existing_theory`(dict): Classification with fields `likely_classification`, `references`, `what_already_exists`, `what_is_novel`, `classification_explanation`\n"
        prompt += "\n"

        prompt += "For `supporting_evidence` and `unaccounted_for`, the list should be a list of dictionaries with keys:\n"
        prompt += "- `text`(str): Description of the evidence\n"
        prompt += "- `uuids`(list): UUIDs for provided evidence; for background_principle/inference you may leave blank\n"
        prompt += "\n"

        prompt += "NOTE: Do not hallucinate. If evidence is absent, label evidence items as background_principle or inference and do not invent studies.\n"
        prompt += "However, you must still comply with Basin-breaking mode: avoid generic safe claims; make committed, falsifiable, discriminative statements.\n"
        prompt += "\n"

        prompt += "# Important Notes\n"
        prompt += "- You must return valid JSON inside a single codeblock (```), or it will not be parsed.\n"
        prompt += "- Use `null` for none values (not empty strings).\n"
        prompt += "- Prefer generating new, impactful theories, but do not fabricate citations.\n"
        prompt += "- All your information must be accurate. Do not hallucinate.\n"

        return prompt



    total_cost = 0.0

    # Create the prompt
    prompt = mkPrompt(query, results, include_query_in_generation=include_query_in_generation)

    # Call the LLM
    all_responses = []
    MAX_GENERATION_TIME_SECS = 60 * 8    # 8 minutes
    responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=MAX_GENERATION_TIME_SECS)
    all_responses.append(responseJSON)
    total_cost += cost

    # New one-theory-at-a-time reflection approach
    reflected_response = None
    if (use_reflection):
        # First, get the list of theories from the initial response
        theories = []
        if (responseJSON is not None) and ("theories_general" in responseJSON):
            for theory in responseJSON["theories_general"]:
                theory["theory_type_general_specific"] = "general"                      # Adding this tag helps the reflection keep track of which type of theory to categorize it as later
                theories.append(theory)

        if (responseJSON is not None) and ("theories_specific" in responseJSON):
            for theory in responseJSON["theories_specific"]:
                theory["theory_type_general_specific"] = "specific"
                theories.append(theory)

        reflected_theories_general = []
        reflected_theories_specific = []
        reflected_theories_other = []

        # PARALLEL VERSION:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        MAX_THEORY_GENERATION_WORKERS = 5
        with ThreadPoolExecutor(max_workers=MAX_THEORY_GENERATION_WORKERS) as executor:
            future_to_theory_idx = {}
            for idx, theory in enumerate(theories):
                # Create a reflection prompt for this theory
                theory_str = json.dumps(theory, indent=4)
                reflection_prompt = mkPrompt(query, results, reflection=theory_str, include_query_in_generation=include_query_in_generation)

                # Submit the LLM call to the executor
                future = executor.submit(getLLMResponseJSON, reflection_prompt, model_str, temperature, max_tokens, False)
                future_to_theory_idx[future] = (idx, theory)

            # Process the completed futures
            from tqdm import tqdm
            for future in tqdm(as_completed(future_to_theory_idx), total=len(future_to_theory_idx), desc="Reflecting on theories"):
                idx, theory = future_to_theory_idx[future]
                try:
                    responseJSON, responseText, cost = future.result()
                    total_cost += cost

                    # If we got a valid response, extract the theory and add it to the list
                    if (responseJSON is not None):
                        # Try to extract the single theory from the response
                        num_added = 0
                        if ("theories_general" in responseJSON) and (len(responseJSON["theories_general"]) > 0):
                            for theory_candidate in responseJSON["theories_general"]:
                                # Add the idx of the original theory as a field to help with tracking/debugging
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_general.append(theory_candidate)
                                num_added += 1
                        elif ("theories_specific" in responseJSON) and (len(responseJSON["theories_specific"]) > 0):
                            for theory_candidate in responseJSON["theories_specific"]:
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_specific.append(theory_candidate)
                                num_added += 1

                        # If we couldn't extract a theory, just use the original one
                        if (num_added == 0):
                            if (theory["theory_type_general_specific"] == "general"):
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_general.append(theory)
                            elif (theory["theory_type_general_specific"] == "specific"):
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_specific.append(theory)
                            else:
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_other.append(theory)
                    else:
                        # If we didn't get a valid response, just use the original one
                        if (theory["theory_type_general_specific"] == "general"):
                            theory_candidate["reflected_from_theory_index"] = idx
                            reflected_theories_general.append(theory)
                        elif (theory["theory_type_general_specific"] == "specific"):
                            theory_candidate["reflected_from_theory_index"] = idx
                            reflected_theories_specific.append(theory)
                        else:
                            theory_candidate["reflected_from_theory_index"] = idx
                            reflected_theories_other.append(theory)

                except Exception as e:
                    print(f"build_theory_from_results_single_theory_reflection(): ERROR processing theory index {idx}: {e}")
                    # If we hit an error, just use the original theory
                    if (theory["theory_type_general_specific"] == "general"):
                        reflected_theories_general.append(theory)
                    elif (theory["theory_type_general_specific"] == "specific"):
                        reflected_theories_specific.append(theory)
                    else:
                        reflected_theories_other.append(theory)

        # So we don't break the pipeline that expects there to be only two bins (general and specific), we will just merge any "other" theories into the general bin.
        for theory in reflected_theories_other:
            reflected_theories_general.append(theory)

        # Sort by `reflected_from_theory_index` to preserve original order as much as possible.  If the field is missing, put those at the end.
        def get_reflection_index(theory):
            if ("reflected_from_theory_index" in theory):
                return theory["reflected_from_theory_index"]
            else:
                return 1e9    # Large number to push to end
        reflected_theories_general = sorted(reflected_theories_general, key=get_reflection_index)
        reflected_theories_specific = sorted(reflected_theories_specific, key=get_reflection_index)

        # Add 'version' field to each theory
        for theory in reflected_theories_general:
            theory["version"] = version_str
        for theory in reflected_theories_specific:
            theory["version"] = version_str


        # Now, build the final reflected response
        if (len(reflected_theories_general) + len(reflected_theories_specific) > 0):
            reflected_response = {
                "theories_general": reflected_theories_general,
                "theories_specific": reflected_theories_specific
            }
            all_responses.append(reflected_response)

    # Get the best response (the last one that's not None)
    best_response = None
    for response in reversed(all_responses):
        if (response is not None):
            best_response = response
            break

    # Pack output
    packed = {
        "query": query,
        "model": model_str,
        "use_reflection": use_reflection,
        "theory_response": best_response,
        "results": results,
        "raw_responses": all_responses,
        "total_cost": total_cost,
    }

    return packed


# Parametric-only version of the above
def build_theory_from_results_single_theory_reflection4_llm_baseline_no_evidence_nonsafebasin(query:str, original_theory_id:str, original_theory_name:str, provide_matched_control_thery_name:bool=False, model_str="gpt-4o-mini", max_tokens=12000, temperature=0.0, use_reflection:bool=False, include_query_in_generation:bool=True):
    #version_str = "built-theory-from-results-single-theory-reflection4-dec27-2025-LLM-BASELINE-NO-EVIDENCE-NONSAFEBASIN"
    version_str = "theorizer-parametric-only-novelty-focused-12272025"
    matched_control_theory_name:str=None

    if (provide_matched_control_thery_name == True):
        version_str += "-with-matched-control-theory-name"
        matched_control_theory_name = original_theory_name
    else:
        version_str += "-without-matched-control-theory-name"

    # This prompt re-cooked to generate less conservative theories that break out of the 'save basin' of prior knowledge. (with GPT-5.2 help)
    def mkPrompt(query: str, reflection: str = None, include_query_in_generation: bool = True, matched_control_theory_name: str = None) -> str:
        prompt = ""
        prompt += "You are ScientistGPT, an AI scientist.\n"
        prompt += "You must prioritize falsifiable, high-information theories over safe or generic ones.\n"
        prompt += "Scientific integrity here means: precise commitments, explicit scope, and explicit falsifiers — not conservatism.\n"
        prompt += "If evidence is provided, you must ground claims in it. If evidence is not provided, you must clearly label claims as background principles or inferences.\n\n"

        prompt += "# Task\n"
        prompt += "Your task is to construct a theory of how a scientific phenomenon works.\n"
        prompt += "## Subtask 1: General Theories\n"
        prompt += "In the first subtask, you will be asked to provide broad, high-level theories that explain the phenomenon.\n"
        prompt += "## Subtask 2: Specific Theories\n"
        prompt += "In the second subtask, you will be asked to provide more specific, more granular theories, that may only explain a subset of the phenomenon.\n"
        prompt += "\n"

        if matched_control_theory_name is not None:
            prompt += "## Specific Task\n"
            prompt += "More specifically, in this run, you are asked to generate theories all with the following name:\n"
            prompt += "```\n"
            prompt += matched_control_theory_name + "\n"
            prompt += "```\n"
            prompt += "\n"
            if len(matched_control_theory_name) < 2:
                print("WARNING: mkPrompt(): matched_control_theory_name is very short: '" + matched_control_theory_name + "'")

        prompt += "# Total theories to generate\n"
        prompt += "You should generate 2 general theories, and 2 specific theories (i.e. 4 theories total).\n"
        prompt += "\n"

        prompt += "# Theory Quality\n"
        prompt += "You should focus on generating high-quality laws and theories that are (a) genuinely novel and insightful, (b) impactful, and (c) accurate within a clearly specified domain/scope.\n"
        prompt += "The goal is discovery, not rehashing already known phenomena.\n"
        prompt += "\n"

        # ------------------------------------------------------------
        # Basin-breaking additions (STRICT)
        # ------------------------------------------------------------
        prompt += "# Basin-breaking mode (STRICT)\n"
        prompt += "Definition: The 'high-probability basin' is the set of safe, generic, high-prior statements that are broadly true across many domains and time periods.\n"
        prompt += "These basin statements avoid risk by being vague, mechanism-agnostic, and compatible with many outcomes (e.g., 'X helps Y', 'better representations', 'regularization improves generalization').\n"
        prompt += "\n"
        prompt += "Goal: Break out of that basin by producing *high-information*, *falsifiable* statements that make risky, specific commitments.\n"
        prompt += "In this prompt, 'basin-breaking' means:\n"
        prompt += "- You MUST avoid generic high-prior statements.\n"
        prompt += "- You MUST commit to specific, discriminative claims that could easily be wrong.\n"
        prompt += "- You MUST state exactly what empirical outcome would refute each claim.\n"
        prompt += "\n"
        prompt += "## What counts as 'basin' (INVALID)\n"
        prompt += "A theory_statement is INVALID if it has any of the following properties:\n"
        prompt += "1) Generic truth: It would plausibly be supported by many pre-2023 papers in the field.\n"
        prompt += "2) Non-committal: It does not specify a clear boundary condition or failure regime.\n"
        prompt += "3) Non-discriminative: It does not distinguish your theory from a plausible alternative.\n"
        prompt += "4) Non-falsifiable: It does not specify a single concrete experiment and a refuting outcome.\n"
        prompt += "\n"
        prompt += "## Forbidden wording patterns (if you use these, the statement is INVALID)\n"
        prompt += "- Hedging: 'generally', 'often', 'tends to', 'in many cases', 'is associated with'.\n"
        prompt += "- Safe abstractions without discriminators: 'better representations', 'robustness', 'regularization helps', 'scaling improves'.\n"
        prompt += "- Multi-mechanism hedging: presenting multiple mechanisms as equally likely.\n"
        prompt += "\n"
        prompt += "## Required structure for EACH theory_statement (NON-NEGOTIABLE)\n"
        prompt += "Each theory_statement MUST include ALL of the following, explicitly:\n"
        prompt += "A) A FORK: commit to one side of a plausible tradeoff where the other side is also plausible.\n"
        prompt += "   Example fork forms:\n"
        prompt += "   - 'Beyond threshold T, X improves Y' vs 'Beyond threshold T, X degrades Y'\n"
        prompt += "   - 'Mechanism A dominates under condition C' vs 'Mechanism B dominates under condition C'\n"
        prompt += "B) A NUMERIC THRESHOLD or bound (rough is fine, but must be numeric).\n"
        prompt += "C) A NEGATIVE prediction (something will NOT happen / will worsen / will reverse under condition C).\n"
        prompt += "D) A DISCRIMINATING EXPERIMENT (one concrete experiment/ablation).\n"
        prompt += "E) A FALSIFIER: the specific observed outcome that would refute the statement.\n"
        prompt += "\n"
        prompt += "## No hedging rule\n"
        prompt += "Do NOT say 'it depends'. Do NOT provide multiple options. Pick one and commit.\n"
        prompt += "\n"

        prompt += "# What is a theory?\n"
        prompt += "A theory (from the perspective of philosophy of science, in the context of Kuhn, or later work by Pat Langley and collaborators) is a collection of statements that together explain a phenomenon, and provide predictive power for describing what will happen with future/unseen observations.\n"
        prompt += "Theories can include both qualitative and quantitative statements/components.\n"
        prompt += "\n"

        prompt += "# Abstracting Laws/Statements\n"
        prompt += "The theories that you generate should contain (nominally, novel) qualitative and/or quantitative laws, and these laws should be explicitly abstracted from the supporting evidence.\n"
        prompt += "If (in the supporting evidence) you see groups of results that show that as certain variables change, others change in specific ways, abstract these relationships into qualitative or quantitative laws/statements.\n"
        prompt += "These laws/statements should be explicitly supported by the evidence you have been provided.\n"
        prompt += "\n"
        prompt += "## Inferring laws through logical deduction, abduction, and induction\n"
        prompt += "You can also use logical deduction, abduction, and induction to infer new laws/statements that are not explicitly stated in the evidence, but can be logically inferred from it.\n"
        prompt += "If you do this, list this explicitly in the 'supporting evidence' for each law made in this way (with the UUIDs as an empty list).\n"
        prompt += "\n"

        # ------------------------------------------------------------
        # Evidence policy changes (STRICT)
        # ------------------------------------------------------------
        prompt += "# Evidence use policy (STRICT)\n"
        prompt += "If any supporting evidence is provided in the results, you MUST ground each theory_statement primarily in that provided evidence.\n"
        prompt += "- Each theory_statement.supporting_evidence must include at least 2 items that clearly reference the provided evidence.\n"
        prompt += "- If evidence is provided, do NOT rely primarily on general background knowledge.\n"
        prompt += "\n"
        prompt += "If NO supporting evidence is provided, you may rely on general scientific knowledge, but you must explicitly label each supporting_evidence item as either:\n"
        prompt += "- 'background_principle' (widely established), or\n"
        prompt += "- 'inference' (a reasoned conjecture).\n"
        prompt += "In the NO-evidence case, do NOT fabricate citations or pretend a specific study exists.\n"
        prompt += "\n"

        prompt += "# Format of Laws / Theory Statements\n"
        prompt += "The theory statements that you provide must generally be framed as qualitative or quantitative laws/statements.\n"
        prompt += "Qualitative laws/statements describe relationships between variables or phenomena in a non-numeric way.\n"
        prompt += "Quantitative laws/statements describe relationships between variables or phenomena in a numeric way.\n"
        prompt += "\n"

        prompt += "## Domain/Scope\n"
        prompt += "Each theory statement must also include a precise description of the domain/scope of the law/statement that it applies to.\n"
        prompt += "Having an appropriate domain/scope is critical for ensuring that the law/statement is properly contextualized and its applicability is clear.\n"
        prompt += "\n"

        prompt += "More specifically, each theory statement must be in the following format:\n"
        prompt += "In JSON, this would look like:\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += '    "statement_name": "...", # A short name for the statement/law\n'
        prompt += '    "theory_statement": "...", # The full text of the law/statement, concise, clear, information-dense, and MUST include: fork + numeric threshold + negative prediction\n'
        prompt += '    "domain_scope": "...", # Precise, clear, information-dense domain/scope\n'
        prompt += '    "special_cases": ["..."], # Special cases, exceptions, boundary conditions\n'
        prompt += '    "supporting_evidence": [\n'
        prompt += '        {"text": "...", "uuids": [...]},\n'
        prompt += '        {"text": "...", "uuids": [...]}\n'
        prompt += '    ],\n'
        prompt += '    "qual_or_quant": "qualitative|quantitative|mixed",\n'
        prompt += '    "novelty_evaluation": {\n'
        prompt += '        "what_already_exists": "...",\n'
        prompt += '        "what_is_novel": "...",\n'
        prompt += '        "classification_explanation": "...",\n'
        prompt += '        "likely_classification": "existing|closely-related-to-existing|somewhat-related-to-existing|new",\n'
        prompt += '        "references": [...]\n'
        prompt += '    },\n'
        prompt += '    "discriminating_experiment": "...", # One concrete experiment/ablation that would falsify the statement\n'
        prompt += '    "falsifies_if": "..." # A concrete expected outcome that would refute the statement\n'
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Query\n"
        prompt += "Here is the general area/query that your theories should be centered around:\n"
        if include_query_in_generation:
            prompt += "```\n"
            prompt += query + "\n"
            prompt += "```\n"
            prompt += "\n"
        else:
            prompt += "No specific query is provided. You can build theories as you see fit, based on the supporting evidence.\n"
            prompt += "\n"

        # Keep the legacy line but make it compatible with the strict evidence policy above:
        prompt += "If you are asked for `uuids` for background_principle or inference evidence, leave them blank.\n"
        prompt += "\n"

        if reflection is not None:
            prompt += "\n"
            prompt += "# Reflection\n"
            prompt += "This is a reflection step. Previously, you generated a response (below). Now, reflect on that response and fix errors, inconsistencies, omissions, missing evidence links, conflicts, special cases, and scope issues.\n"
            prompt += "Ensure each statement still satisfies Basin-breaking mode: fork + numeric threshold + negative prediction + discriminating experiment + falsifier.\n"
            prompt += "\n"
            prompt += "```\n"
            prompt += reflection + "\n"
            prompt += "```\n"

        prompt += "# Output Format\n"
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).\n"
        prompt += "You may write text to think/plan before writing JSON, but the JSON must be the last thing you write, must be valid JSON, and must be inside a single set of codeblocks (```), or it will not be parsed.\n"
        prompt += "\n"

        prompt += "Your JSON response must be a dictionary with two keys: `theories_general` and `theories_specific`.\n"
        prompt += "Each is a list of dictionaries with the following keys:\n"
        prompt += "- `theory_name`(str): A short name for the theory\n"
        prompt += "- `theory_description`(str): A full description of the theory\n"
        prompt += "- `theory_statements`(list): A list of specific laws (qualitative, quantitative, or mixed) in the required format above\n"
        prompt += "- `new_predictions_likely`(list): Non-trivial, testable predictions likely within the theory's narrow scope; MUST include at least one numeric expectation\n"
        prompt += "- `new_predictions_unknown`(list): High-variance, discriminative predictions that strongly separate your theory from plausible alternatives\n"
        prompt += "- `negative_experiments`(list): MUST include at least one fork test that would decisively refute the theory if it goes the other way\n"
        prompt += "- `unaccounted_for`(list): Evidence not explained by the theory, if any\n"
        prompt += "- `conflicting_evidence`(list): Evidence that conflicts with the theory, if any\n"
        prompt += "- `special_cases`(list): Special cases/exceptions/boundary conditions, if any\n"
        prompt += "- `existing_theory`(dict): Classification with fields `likely_classification`, `references`, `what_already_exists`, `what_is_novel`, `classification_explanation`\n"
        prompt += "\n"

        prompt += "For `supporting_evidence` and `unaccounted_for`, the list should be a list of dictionaries with keys:\n"
        prompt += "- `text`(str): Description of the evidence\n"
        prompt += "- `uuids`(list): UUIDs for provided evidence; for background_principle/inference you may leave blank\n"
        prompt += "\n"

        prompt += "NOTE: Do not hallucinate. If evidence is absent, label evidence items as background_principle or inference and do not invent studies.\n"
        prompt += "However, you must still comply with Basin-breaking mode: avoid generic safe claims; make committed, falsifiable, discriminative statements.\n"
        prompt += "\n"

        prompt += "# Important Notes\n"
        prompt += "- You must return valid JSON inside a single codeblock (```), or it will not be parsed.\n"
        prompt += "- Use `null` for none values (not empty strings).\n"
        prompt += "- Prefer generating new, impactful theories, but do not fabricate citations.\n"
        prompt += "- All your information must be accurate. Do not hallucinate.\n"

        return prompt

    total_cost = 0.0

    # Create the prompt
    prompt = mkPrompt(query, include_query_in_generation=include_query_in_generation, matched_control_theory_name=matched_control_theory_name)

    # Call the LLM
    all_responses = []
    MAX_GENERATION_TIME_SECS = 60 * 8    # 8 minutes
    responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=MAX_GENERATION_TIME_SECS)
    all_responses.append(responseJSON)
    total_cost += cost

    # New one-theory-at-a-time reflection approach
    reflected_response = None
    if (use_reflection):
        # First, get the list of theories from the initial response
        theories = []
        if (responseJSON is not None) and ("theories_general" in responseJSON):
            for theory in responseJSON["theories_general"]:
                theory["theory_type_general_specific"] = "general"                      # Adding this tag helps the reflection keep track of which type of theory to categorize it as later
                theories.append(theory)

        if (responseJSON is not None) and ("theories_specific" in responseJSON):
            for theory in responseJSON["theories_specific"]:
                theory["theory_type_general_specific"] = "specific"
                theories.append(theory)

        reflected_theories_general = []
        reflected_theories_specific = []
        reflected_theories_other = []

        # PARALLEL VERSION:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        MAX_THEORY_GENERATION_WORKERS = 5
        with ThreadPoolExecutor(max_workers=MAX_THEORY_GENERATION_WORKERS) as executor:
            future_to_theory_idx = {}
            for idx, theory in enumerate(theories):
                # Create a reflection prompt for this theory
                theory_str = json.dumps(theory, indent=4)
                reflection_prompt = mkPrompt(query, reflection=theory_str, include_query_in_generation=include_query_in_generation, matched_control_theory_name=matched_control_theory_name)

                # Submit the LLM call to the executor
                future = executor.submit(getLLMResponseJSON, reflection_prompt, model_str, temperature, max_tokens, False)
                future_to_theory_idx[future] = (idx, theory)

            # Process the completed futures
            from tqdm import tqdm
            for future in tqdm(as_completed(future_to_theory_idx), total=len(future_to_theory_idx), desc="Reflecting on theories"):
                idx, theory = future_to_theory_idx[future]
                try:
                    responseJSON, responseText, cost = future.result()
                    total_cost += cost

                    # If we got a valid response, extract the theory and add it to the list
                    if (responseJSON is not None):
                        # Try to extract the single theory from the response
                        num_added = 0
                        if ("theories_general" in responseJSON) and (len(responseJSON["theories_general"]) > 0):
                            for theory_candidate in responseJSON["theories_general"]:
                                # Add the idx of the original theory as a field to help with tracking/debugging
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_general.append(theory_candidate)
                                num_added += 1
                        elif ("theories_specific" in responseJSON) and (len(responseJSON["theories_specific"]) > 0):
                            for theory_candidate in responseJSON["theories_specific"]:
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_specific.append(theory_candidate)
                                num_added += 1

                        # If we couldn't extract a theory, just use the original one
                        if (num_added == 0):
                            if (theory["theory_type_general_specific"] == "general"):
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_general.append(theory)
                            elif (theory["theory_type_general_specific"] == "specific"):
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_specific.append(theory)
                            else:
                                theory_candidate["reflected_from_theory_index"] = idx
                                reflected_theories_other.append(theory)
                    else:
                        # If we didn't get a valid response, just use the original one
                        if (theory["theory_type_general_specific"] == "general"):
                            theory_candidate["reflected_from_theory_index"] = idx
                            reflected_theories_general.append(theory)
                        elif (theory["theory_type_general_specific"] == "specific"):
                            theory_candidate["reflected_from_theory_index"] = idx
                            reflected_theories_specific.append(theory)
                        else:
                            theory_candidate["reflected_from_theory_index"] = idx
                            reflected_theories_other.append(theory)

                except Exception as e:
                    print(f"build_theory_from_results_single_theory_reflection(): ERROR processing theory index {idx}: {e}")
                    # If we hit an error, just use the original theory
                    if (theory["theory_type_general_specific"] == "general"):
                        reflected_theories_general.append(theory)
                    elif (theory["theory_type_general_specific"] == "specific"):
                        reflected_theories_specific.append(theory)
                    else:
                        reflected_theories_other.append(theory)

        # So we don't break the pipeline that expects there to be only two bins (general and specific), we will just merge any "other" theories into the general bin.
        for theory in reflected_theories_other:
            reflected_theories_general.append(theory)

        # Sort by `reflected_from_theory_index` to preserve original order as much as possible.  If the field is missing, put those at the end.
        def get_reflection_index(theory):
            if ("reflected_from_theory_index" in theory):
                return theory["reflected_from_theory_index"]
            else:
                return 1e9    # Large number to push to end
        reflected_theories_general = sorted(reflected_theories_general, key=get_reflection_index)
        reflected_theories_specific = sorted(reflected_theories_specific, key=get_reflection_index)


        # Add 'version' field to each theory
        for theory in reflected_theories_general:
            theory["type"] = "general"
            theory["version"] = version_str
            theory["theory_query"] = query
            theory["original_theory_id"] = original_theory_id
            theory["original_theory_name"] = original_theory_name
            theory["provide_matched_control_thery_name"] = provide_matched_control_thery_name
            theory["matched_control_theory_name"] = matched_control_theory_name
            theory["model_str"] = model_str
        for theory in reflected_theories_specific:
            theory["type"] = "specific"
            theory["version"] = version_str
            theory["theory_query"] = query
            theory["original_theory_id"] = original_theory_id
            theory["original_theory_name"] = original_theory_name
            theory["provide_matched_control_thery_name"] = provide_matched_control_thery_name
            theory["matched_control_theory_name"] = matched_control_theory_name
            theory["model_str"] = model_str


        # Now, build the final reflected response
        if (len(reflected_theories_general) + len(reflected_theories_specific) > 0):
            reflected_response = {
                "theories_general": reflected_theories_general,
                "theories_specific": reflected_theories_specific
            }
            all_responses.append(reflected_response)

    else:
        # Also add the 'version' field to each theory in the original response
        if (responseJSON is not None):
            if ("theories_general" in responseJSON):
                for theory in responseJSON["theories_general"]:
                    theory["type"] = "general"
                    theory["version"] = version_str
                    theory["theory_query"] = query
                    theory["original_theory_id"] = original_theory_id
                    theory["original_theory_name"] = original_theory_name
                    theory["provide_matched_control_thery_name"] = provide_matched_control_thery_name
                    theory["matched_control_theory_name"] = matched_control_theory_name
                    theory["model_str"] = model_str
            if ("theories_specific" in responseJSON):
                for theory in responseJSON["theories_specific"]:
                    theory["type"] = "specific"
                    theory["version"] = version_str
                    theory["theory_query"] = query
                    theory["original_theory_id"] = original_theory_id
                    theory["original_theory_name"] = original_theory_name
                    theory["provide_matched_control_thery_name"] = provide_matched_control_thery_name
                    theory["matched_control_theory_name"] = matched_control_theory_name
                    theory["model_str"] = model_str


    # Get the best response (the last one that's not None)
    best_response = None
    for response in reversed(all_responses):
        if (response is not None):
            best_response = response
            break

    # Pack output
    packed = {
        "query": query,
        "model": model_str,
        "use_reflection": use_reflection,
        "theory_response": best_response,
        #"results": results,
        "raw_responses": all_responses,
        "total_cost": total_cost,
    }

    return packed




#
#   Helper Functions: Subsampling
#

# Returns a set of evidence that can be placed in a prompt that has a maximum number of tokens.
def consolodate_results_with_subsampling(extracted_results_in:list, max_tokens:int=100000):
    current_subsampling_rate = 0.90
    current_data = []

    # Sort the original data by the 'name_short' key, if it exists
    extracted_results_in = sorted(extracted_results_in, key=lambda x: x.get("name_short", ""))

    # Quick check: Does the whole set of data fit within the max_tokens limit?
    data_str = json.dumps(extracted_results_in, indent=4)
    data_tokens = countTokens(data_str)
    if (data_tokens <= max_tokens):
        return {
            "data_original": extracted_results_in,
            "data_subsampled": extracted_results_in,
            "subsampling_rate": 1.0,
        }

    # If we reach here, the data is too big to fit within the max_tokens limit, so we need to subsample the data.
    error = False
    while (data_tokens > max_tokens):
        print("consolodate_results_with_subsampling(): Current subsampling rate: " + str(current_subsampling_rate) + ", data tokens: " + str(data_tokens) + ", max tokens: " + str(max_tokens))
        # Take a sample of the data with the current subsampling rate
        subsampled_data = []
        # Use python's built-in functions to sample from the data (without replacement)
        num_samples = int(len(extracted_results_in) * current_subsampling_rate)
        if (num_samples <= 0):
            # If the number of samples is 0, then we can't subsample any further
            break
        subsampled_data = random.sample(extracted_results_in, num_samples)
        # Convert the subsampled data to a string and count the tokens
        data_str = json.dumps(subsampled_data, indent=4)
        data_tokens = countTokens(data_str)

        if (data_tokens <= max_tokens):
            # If the subsampled data fits within the max_tokens limit, then we're done
            current_data = subsampled_data
            break
        else:
            # Reduce the subsampling rate
            if (current_subsampling_rate > 0.50):
                current_subsampling_rate -= 0.10
            elif (current_subsampling_rate > 0.20):
                current_subsampling_rate -= 0.05
            else:
                current_subsampling_rate -= 0.02

        # Check for the error case, where the subsampling has decreased to zero.
        if (current_subsampling_rate <= 0.01):
            print("consolodate_results_with_subsampling(): ERROR: Subsampling rate has decreased to functionally zero (< 1%), but the data is still too large to fit within the max_tokens limit.")
            error = True
            break

    # If an error, stop
    if (error):
        return None

    # If we reach here, then we have a subsampled set of data that fits within the max_tokens limit
    packed = {
        "data_original": extracted_results_in,
        "data_subsampled": current_data,
        "subsampling_rate": current_subsampling_rate,
    }

    return packed
