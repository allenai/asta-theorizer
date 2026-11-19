# TheorizerProcessing.py
# This file has many of the LLM prompt-based functions for the various stages of the theory generation process.
import os
import json
import time
import requests

import threading

from Struct import *
from ExtractionUtils import *


# Knowledge and inference types for:
# def build_theory_from_results_single_theory_reflection5_inferencetypes(query:str, results:list, mission_statement:str=None, model_str="gpt-4o-mini", max_tokens=12000, temperature=0.0, use_reflection:bool=True, include_query_in_generation:bool=True, additional_suggestions=None, max_laws_per_theory:int=3, inference_type:str=None, output_type:str=None):

OUTPUT_TYPES = [
    "MECHANISTIC EXPLANATION",
    "COMPARATIVE RANKING",
    "STATE OF THE LITERATURE",
    "CAPABILITY CLAIM",
    "CONTEXT DEPENDENCE",
    "ASSOCIATION",
    "EFFECT OF AN INTERVENTION",
    "RECOMMENDATION",
    "DEFINITION OR TAXONOMY",
    "APPRAISAL OF EVIDENCE QUALITY",
    "TEMPORAL TREND",
    "NULL RESULT"
]

INFERENCE_TYPES = [
    "ENUMERATIVE INDUCTION",
    "TRIANGULATION",
    "META-ANALYTIC POOLING",
    "CONTRAST ARGUMENT (HETEROGENEITY)",
    "CONTROLLED COMPARISON OR ABLATION",
    "TABULATION OR FREQUENCY COUNTING",
    "STATISTICAL INFERENCE",
    "ABDUCTION (BEST EXPLANATION)",
    "DEDUCTION FROM DEFINITIONS OR RULES",
    "PROOF OF CONCEPT",
    "INDUCTION FROM SINGLE CASE",
    "ARGUMENT FROM ANALOGY",
    "PRACTICAL (MEANS-END) REASONING",
    "CRITICAL APPRAISAL OF THE EVIDENCE"
]

#
#   Converting a theory request (e.g. make me a theory about X) into a query and schema for extraction on papers.
#
def convert_theory_request_to_query_and_schema(theory_request:str, model_str:str, max_tokens:int=8000, temperature:float=0.0, use_reflection:bool=False, mission_statement:str=None):
    def mkPrompt(theory_request:str, mission_statement:str, reflection:str=None):
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
        prompt += str(theory_request) + "\n"
        prompt += "```\n"
        prompt += "\n"
        if (mission_statement is not None):
            prompt += "# Mission Statement\n"
            prompt += "Here is the mission statement that provides additional context for the theory request, and should be used to help guide the generation of the query and schema:\n"
            prompt += "```\n"
            prompt += str(mission_statement) + "\n"
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
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output, as these are not valid JSON and will cause a parsing error.\n"
        prompt += "\n"
        prompt += "Your JSON response must be a dictionary, that contains exactly 6 keys: `theory_request_original`, `theory_request_normalized`, `short_name`, `paper_search_query`, `extraction_query`, and `extraction_schema`.  The values for these keys are as follows:\n"
        prompt += "- `theory_request_original`(str): The original theory request that was provided (e.g. 'make me a theory about X')\n"
        #prompt += "- `theory_request_normalized`(str): A normalized, clear version of the theory request, that always starts with `Build a theory of...` and ends with `based on the following results.`.\n"
        prompt += "- `theory_request_normalized`(str): The theory request should always start with `Build theories of/about...` and end with `based on the following results.`.  If the user-supplied theory query does not start with this, then please append these to the user-supplied theory query with minimal edits. This is a cosmetic change -- you should retain *all* the content of the user's theory query request, and just make minor changes to the surface form for the 'build theories about...' and 'based on the following results' modifications, without losing any information from the user theory query. If the users theory query is detailed, keep it detailed and retain all the content, just make these minor modifications/rephrasing.\n"
        prompt += "- `short_name`(str): A short name for the theory request, that is a concise, and generally 4 words long, separated by '-'. This will be a prefix appended to filenames to save data for this request.\n"
        prompt += "- `paper_search_query`(str): The query that should be used to search for papers that are relevant to the theory request, using a paper search tool over Semantic Scholar.\n"
        prompt += "- `extraction_query`(str): A natural language request query, that's provided to an extraction prompt (along with the extraction schema below) to outline the overall purpose of the extraction process.  Should start with `Extract any mentions of...`.\n"
        prompt += "- `extraction_schema`(list): A list of dictionaries, where each dictionary contains the following keys:\n"
        prompt += "    - `name`(str): The name of the entity to extract (e.g. 'entity_name')\n"
        prompt += "    - `type`(str): The type of the entity to extract (e.g. 'str', 'int', 'float', etc.)\n"
        prompt += "    - `description`(str): A brief, concise, and helpful description of the entity to extract, often times with examples, that very explicitly spells out what information should be extracted.\n"
        prompt += "\n"

        prompt += "## Special instructions\n"
        prompt += "- You can (and are encouraged to) think and plan before writing your JSON response.  But, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output.\n"
        prompt += "- Please follow the instructions carefully.\n"
        prompt += "- Do not hallucinate.\n"

        return prompt


    total_cost = 0.0
    responses = []
    responses_raw = []

    # Create the prompt
    prompt = mkPrompt(theory_request, mission_statement, reflection=None)
    # Call the LLM
    responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=0.0, maxTokens=max(max_tokens, DEFAULT_MAX_TOKENS), jsonOut=False)
    responses.append(responseJSON)
    responses_raw.append(responseText)
    total_cost += cost

    # If we are using reflection, we will call the LLM again with the reflection step
    if (use_reflection):
        # Create the reflection prompt
        reflection_prompt = mkPrompt(theory_request, mission_statement, responseText)
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
#   Schema generation from variable discovery
#
def generate_schema_from_variable_discovery(theory_query_information:str, variable_discovery_results:dict, model_str:str, max_tokens:int=8000, temperature:float=0.0, use_reflection:bool=False):
    def mkPrompt(theory_query_information, variable_discovery_results, reflection:str=None):
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"

        prompt += "# Task\n"
        prompt += "You are part of a system that is designed to infer scientific theories on specific topics, based on evidence gathered from scientific papers.\n"
        prompt += "The steps in the process are:\n"
        prompt += "1. Collect the theory request from the user (e.g. 'make me a theory about X')\n"
        prompt += "2. Convert the theory request into a paper-search query.\n"
        prompt += "3. Read the papers to discover potentially relevant variables/information to extract from the papers, to support building high-utility/high-impact and novel theory/law discoveries.\n"
        prompt += "*4. Build extraction schema for extracting entities/supporting information from scientific papers, based on the variable discovery results.\n"
        prompt += "5. Use the query and schema to extract entities from scientific papers.\n"
        prompt += "6. Use the extracted entities to build a theory about the topic.\n"
        prompt += "\n"
        prompt += "We are currently on step 4.\n"
        prompt += "You will be given a theory request, and the results from a variable discovery process, and your task is to convert it into one (or more) extraction schema for extracting entities from scientific papers that would provide highly relevant evidence to support building theories in this domain.\n"
        prompt += "\n"


        prompt += "## High-Level Task Context\n"
        prompt += "The overall task is to formulate new theories/laws based on scientific evidence, in part through a large-scale literature search.\n"
        prompt += "The general process is: (a) extract information from papers using a detailed extraction schema, that can supply information useful for theory building, then (b) use the extracted information across many (dozens/hundreds) of papers to infer patterns and build theories.\n"
        prompt += "The slots in the extraction schema can be high-level/broad, or low-level/specific.\n"
        prompt += "\n"
        prompt += "Here are examples of high-level/open-ended slots:\n"
        prompt += "- `key_factors`: Key factors that contribute to the phenomenon of interest (e.g. Alzheimer's disease development)\n"
        prompt += "- `challenges`: Challenges that make it difficult to solve the problem of interest\n"
        prompt += "- `proposed_approaches`: Proposed approaches to solving the problem of interest\n"
        prompt += "- `research_methods`: A description of how the problem was studied/the research methods used\n"
        prompt += "- `findings`: The key findings/results of the paper\n"
        prompt += "\n"
        prompt += "Here are examples of low-level/specific slots:\n"
        prompt += "- `datasets_used`: A list of datasets used in the paper\n"
        prompt += "- `model_architectures_used`: A list of model architectures used in the paper\n"
        prompt += "- `evaluation_metrics_used`: A list of evaluation metrics used in the paper\n"
        prompt += "- `model_performance_on_question_answering`: The performance of the model(s) used on a specific task (e.g. question answering), if applicable\n"
        prompt += "- `model_performance_after_treatment`: The performance of the model(s) after a specific treatment/intervention (e.g. memory augmentation), if applicable\n"
        prompt += "\n"
        prompt += "## Thinking about slots\n"
        prompt += "The slots in the extraction schema are designed to extract information that is useful for theory building.\n"
        prompt += "Theory building in this way is essentially an exercise in finding patterns in the extracted information across many papers, and then using those patterns to build theories that explain the phenomenon of interest.\n"
        prompt += "Some of these patterns can be sparse (i.e. occuring in only a small number of papers), and some can be dense (i.e. occurring in many papers).\n"
        prompt += "\n"
        prompt += "## Slots and their relationship to variables in theories:\n"
        prompt += "Sometimes slots are specific enough, that they themselves are very close to variables that would appear in a theory.  As a toy example, if the extraction schema contained slots for 'force', 'mass', and 'acceleration', and the papers all described experiments measuring the force, mass, and acceleration of objects, then it'd be likely to discover a direct quantitative relationship between these variables (i.e. force = mass * acceleration) -- though of course this is a toy example because it's already known.\n"
        prompt += "Other times, slots are more high-level and broad, and they might contain a mix of different kinds of information that would need to be further processed/analyzed to discover finer-grained (or even more abstracted) patterns.\n"
        prompt += "Some slots can be filled with quantitative information (e.g. a number), some qualitative information, some categorical information, some prose descriptions, short lists, or other formats -- whatever is needed.\n"
        prompt += "In this way, some theories/laws that are synthesized may be quantitative, qualitative, or a mix of both.\n"
        prompt += "\n"
        prompt += "## Implicit/Hidden Variable Discovery\n"
        prompt += "In some cases, the most important variables to extract may be implicit, hidden, or otherwise non-obvious.\n"
        prompt += "Here is a real (but somewhat toy) example from the Chemistry domain, and an early qualitative law discovery system.\n"
        prompt += "The law that was discovered by the system was: `acids and alkalis react to produce a salt and water`.\n"
        prompt += "You can imagine that if the extraction slots directly included this classification information (i.e. whether a substance was an acid, alkali, salt), then it would be relatively straightforward to discover this law.\n"
        prompt += "It's also important to node that this information could have been included in a variety of ways:\n"
        prompt += "- Possibility: `reactant_1_classification`, `reactant_2_classification`: These are high-level slots, that aren't restricted to being a specific variable type (e.g. acid/alkali/salt), but it might get lucky and fill those classifications in for a subset of papers, and then the broader analysis would discover the pattern (there's a precision/recall tradeoff analog here: being specific makes it easier to discover a specific pattern, but harder to go fishing/abstract/discover somewhat unexpected patterns).\n"
        prompt += "- Possibility: `reactant_1_properties`, `reactant_2_properties`: These are more general slots, that could contain a variety of information about the reactants, but if they contain the right information (e.g. pH level, or other chemical properties), then the broader analysis might discover the pattern (though this is harder than the previous possibility, because the information is less standardized and more variable across papers, so it's harder to discover patterns in it).\n"
        prompt += "- Possibility: `is_reactant_1_an_acid`, `is_reactant_1_an_alkali`, `is_reactant_1_a_salt`: These are very specific slots (and could have been more categorical: e.g. `classification_acid_alkali_salt`), that directly ask for classification based on a specific categorization scheme.\n"
        prompt += "- Possibility: `reactant_1_chemical_formula`, `reactant_2_chemical_formula`: These are more indirect, that ask for chemical formulas, but if the analysis is sophisticated enough, it might be able to infer the classifications from the chemical formulas (though this is much harder than the previous possibilities, because it requires more chemical knowledge and reasoning to notice the pattern/infer the classifications from the chemical formulas).\n"
        prompt += "- Possibility: `reactant_1_chemical_name`, `reactant_2_chemical_name`: These are even harder, as there would be quite a bit of effort required to discover that chemical X is an acid, chemical Y is an alkali, and chemical Z is a salt, based on their names alone (though in some cases it might be possible to infer this from the names, if the names contain indicative information).\n"
        prompt += "One of the more interesting things is that the most important or useful variables might be non-obvious, or classifications that people haven't thought about before.\n"
        prompt += "That is: some discovered patterns might be rules over known classes (e.g. acids+alkalis -> salts, if this rule wasn't already known, but the classes of `acid`, `alkali`, and `salt` were known)\n"
        prompt += "BUT: some discovered patterns might be over new classes that haven't been explicitly recognized before. One of my favorite examples is that a colleague recently discovered (in a natural langugae processing task) that the standard input people were using actually contained two large, completely independent classes of problems that had been unrecognized, and that if you studied them independent, you discovered patterns that were obscured or difficult to see when you didn't know the classes existed.\n"
        prompt += "While that's a more classification-centered, splitting-hypothesis-centered example, there are also classic versions of other examples: such as the idea that tectonic plates are a hidden variable/mechanistic explanation that unifies (rather than split) seeminly disparate phenomena (e.g. earthquakes, volcanoes, mountain formation, continental drift) into a unifying theory, that also allows making better predictions, and asking better research questions.\n"
        prompt += "\n"
        prompt += "In summary:\n"
        prompt += "- Discovering novel rules over known variables can be impactful\n"
        prompt += "- Discovering novel classifications, or ways of looking at the data, that reveal new variables can be impactful\n"
        prompt += "- New variables can be low-level and specific, high-level and abstract, or anywhere in between, and still potentially be useful for theory building\n"
        prompt += "\n"

        prompt += "# Specific Task\n"
        prompt += "You will be provided with:\n"
        prompt += "- `theory_query_information`: This is a dictionary that contains information about the specific theory-building question or topic of interest.\n"
        prompt += "- The results of a variable discovery process, that has been run over a set of papers that are relevant to the theory-building question or topic of interest.  This variable discovery process is designed to find potentially relevant variables/information to extract from the papers, to support building high-utility/high-impact and novel theory/law discoveries.  The variable discovery results are not perfect, and may contain errors, omissions, or other issues, but they should provide a good starting point for thinking about what information to extract from the papers.\n"
        prompt += "\n"
        prompt += "Your task is to examine the theory query, and variable discovery results, and suggest schema slots that you think would be useful for extracting information from a broad corpus of papers in support of the theory building process.\n"
        prompt += "You can suggest up to 4 different extraction schemas, and each schema can have as many as 10-12 slots.\n"
        prompt += "\n"


        prompt += "# Theory Request\n"
        prompt += "Here is the theory request:\n"
        prompt += "```\n"
        if (isinstance(theory_query_information, dict)):
            prompt += json.dumps(theory_query_information, indent=4)
        else:
            prompt += str(theory_query_information) + "\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Variable Discovery Results\n"
        prompt += "Here are the results from the variable discovery process.  Each process was undertaken on a single paper, and then the results (per paper) are shown here.\n"
        prompt += "Some central themes may emerge across many papers, while some subsets of papers may suggest unique or interesting variables/slots that might be indicative of broader patters, where they investigated more generally.\n"
        prompt += "NOTE: The variable discovery results are not perfect, and they were made by a less powerful model, so they may contain errors, omissions, or other issues.  But they should provide a good starting point for thinking about what information to extract from the papers.\n"
        prompt += "```\n"
        if (isinstance(variable_discovery_results, dict) or isinstance(variable_discovery_results, list)):
            prompt += json.dumps(variable_discovery_results, indent=4)
        else:
            prompt += str(variable_discovery_results) + "\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Thinking about the goals of a good extraction schema. "
        prompt += "The goal of the `extraction schema` is to broadly extract entities/relevant information that can be used to build theories about the topic.\n"
        prompt += "\n"
        prompt += "Things to consider:\n"
        prompt += "- The schema should capture evidence that is relevant to the theory request.  A theory needs to explain a wide variety of evidence, so it's important to consider not just evidence that might support a theory, but also extract evidence/counterpoints that might provide challenges to creating an all-encompassing theory.\n"
        prompt += "\n"
        prompt += "## Examples\n"
        prompt += "Here are some examples of theory requests, and the corresponding paper search queries and extraction schemas.  These are illustrative only, and may have issues/challenges -- you can likely do much better if you think about it.\n"
        prompt += "\n"
        prompt += "The description that you choose for each variable will likely be a mix of the different fields in the variable discovery results. The intended audience for the schema descriptions is an extraction language model, that will be given the schema and a paper, and asked to extract any instances of the variables in the schema from the paper. (Each paper might return zero, one, or multiple instances of matches to a schema).\n"
        prompt += "\n"
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
        prompt += "    ],\n"
        prompt += "    \"notes\": \"...\",\n"
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
        prompt += "    ],\n"
        prompt += "    \"notes\": \"...\",\n"
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
        prompt += "    ],\n"
        prompt += "    \"notes\": \"...\",\n"
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
        prompt += "    ],\n"
        prompt += "    \"notes\": \"...\",\n"
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
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output, as these are not valid JSON and will cause a parsing error.\n"
        prompt += "\n"
        prompt += "Your JSON response must be a dictionary with a single key, `extraction_schemas`, that is a list of dictionaries, where each dictionary represents an extraction schema.\n"
        prompt += "The schemas should be in ranked order, so that the most promising schema for impactful theory building is listed first.\n"
        prompt += "Each extraction schema dictionary must contains exactly 6 keys: `theory_request_original`, `theory_request_normalized`, `short_name`, `paper_search_query`, `extraction_query`, and `extraction_schema`.  The values for these keys are as follows:\n"
        prompt += "- `theory_request_original`(str): The original theory request that was provided (e.g. 'make me a theory about X')\n"
        prompt += "- `theory_request_normalized`(str): A normalized, clear version of the theory request, that always starts with `Build a theory of...` and ends with `based on the following results.`.\n"
        prompt += "- `short_name`(str): A short name for the theory request, that is a concise, and generally 4 words long, separated by '-'. This will be a prefix appended to filenames to save data for this request.\n"
        prompt += "- `paper_search_query`(str): The query that should be used to search for papers that are relevant to the theory request, using a paper search tool over Semantic Scholar.\n"
        prompt += "- `extraction_query`(str): A natural language request query, that's provided to an extraction prompt (along with the extraction schema below) to outline the overall purpose of the extraction process.  Should start with `Extract any mentions of...`.\n"
        prompt += "- `extraction_schema`(list): A list of dictionaries, where each dictionary contains the following keys:\n"
        prompt += "    - `name`(str): The name of the entity to extract (e.g. 'entity_name')\n"
        prompt += "    - `type`(str): The type of the entity to extract (e.g. 'str', 'int', 'float', etc.)\n"
        prompt += "    - `description`(str): A brief, concise, and helpful description of the entity to extract, often times with examples, that very explicitly spells out what information should be extracted.\n"
        prompt += "- `notes`(str): A brief, concise, 1-2 sentence note to the user to mention what this schema was designed for, and how it compares/its relative utility compared to any other extraction schema(s) generated.\n"
        prompt += "\n"

        prompt += "## Special instructions\n"
        prompt += "- You can (and are encouraged to) think and plan before writing your JSON response.  But, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output.\n"
        prompt += "- Please follow the instructions carefully.\n"
        prompt += "- Do not hallucinate.\n"

        return prompt


    total_cost = 0.0
    responses = []
    responses_raw = []

    # Create the prompt
    #prompt = mkPrompt(theory_request)
    prompt = mkPrompt(theory_query_information, variable_discovery_results, reflection=None)
    # Call the LLM
    responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=0.0, maxTokens=max(max_tokens, DEFAULT_MAX_TOKENS), jsonOut=False)
    responses.append(responseJSON)
    responses_raw.append(responseText)
    total_cost += cost

    # If we are using reflection, we will call the LLM again with the reflection step
    if (use_reflection):
        # Create the reflection prompt
        reflection_prompt = mkPrompt(theory_query_information, variable_discovery_results, reflection=responseText)
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
        #"theory_request": theory_request,
        "theory_query_information": theory_query_information,
        "model": model_str,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "use_reflection": use_reflection,
        "output": best_response,
        "responses": responses,
        "responses_raw": responses_raw,
        "variable_discovery_results": variable_discovery_results,
        "total_cost": total_cost,
    }
    return packed


#
#   Schema generation from variable discovery
#
def generate_schema_from_extraction_results_for_symbolic_solvers(theory_query_information:str, extracted_records:list, model_str:str, max_tokens:int=16000, temperature:float=0.0, use_reflection:bool=False):
    def mkPrompt(theory_query_information, extracted_records, reflection:str=None):
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"

        prompt += "# Task\n"
        prompt += "You are part of a system that is designed to infer scientific theories on specific topics, based on evidence gathered from scientific papers.\n"
        prompt += "The steps in the process are:\n"
        prompt += "1. Collect the theory request from the user (e.g. 'make me a theory about X')\n"
        prompt += "2. Convert the theory request into a paper-search query.\n"
        prompt += "3. Read the papers to discover potentially relevant variables/information to extract from the papers, to support building high-utility/high-impact and novel theory/law discoveries.\n"
        prompt += "4. Build extraction schema for extracting entities/supporting information from scientific papers, based on the variable discovery results.\n"
        prompt += "5. Use the query and schema to extract entities/content from scientific papers.\n"
        prompt += "*6. Symbolic schema reformulation: Examine the extracted information, and reformulate the extraction schema to be more amenable to using symbolic solvers.\n"
        prompt += "7. Use the query and symbolic schema to extract entities/content from papers.\n"
        prompt += "8. Provide the extracted symbolic content to symbolic solvers, for theory/law/pattern discovery.\n"
        prompt += "\n"
        prompt += "We are currently on step 6.\n"
        prompt += "You will be given a theory request, and the results from extracting content from papers, and your task is to generate a (symbolic-appropriate) extraction schema that would extract content from the papers in a way that a suite of symbolic solvers would be able to use it to discover useful/impactful theories/laws/patterns in this domain.\n"
        prompt += "\n"


        prompt += "## High-Level Task Context\n"
        prompt += "The overall task is to formulate new theories/laws based on scientific evidence, in part through a large-scale literature search.\n"
        prompt += "The general process is: (a) extract information from papers using a detailed extraction schema, that can supply information useful for theory building, then (b) use the extracted information across many (dozens/hundreds) of papers to infer patterns and build theories.\n"
        prompt += "The slots in the extraction schema can be high-level/broad, or low-level/specific.\n"
        prompt += "\n"
        prompt += "## Slots for the symbolic solver case (this case)\n"
        prompt += "The slots in the extraction schema must have content that fills them in one of the following types:\n"
        prompt += "- Numerical: A number, that can be used in a symbolic solver (e.g. 1, 2.5, -3.14, etc.)\n"
        prompt += "- Categorical: A category that is frequently used in the domain (or, that does or may appear frequently in this context whether or not is is currently known in the domain), that can be used in a symbolic solver (e.g. 'acid', 'alkali', 'salt', etc.). The list of possible category values must generally be known before hand, though an 'other' category is possible. The exact categorical string must be filled as the value in this column, so that (across papers), the same categories are used for the same content, and patterns can be discovered.\n"
        prompt += "- Boolean: A THREE-valued field: `true`, `false`, or `null`. `true` and `false` may be used ONLY when the paper explicitly reports evidence one way or the other. If the paper does not address the question at all, the value MUST be `null`, NOT `false`. This is critical and must be stated in the slot's own description: every boolean slot description must end with `(true, false, or null if the paper does not report it)`. A boolean that cannot express 'not reported' is useless, because 'the paper says no' and 'the paper is silent' collapse into the same value and the slot carries no information.\n"
        prompt += "For EVERY slot of EVERY type -- numerical, categorical and boolean -- `null` is always an available value, and is the REQUIRED value when a paper does not report that information. Do not add a catch-all category such as 'not_reported', 'unspecified', 'not_tested' or 'not_distinguished' to a categorical vocabulary just to express absence: absence is `null`. Measured results from an earlier version of this task showed such catch-all values being selected for over 90%% of rows, which destroyed the slot. Reserve the vocabulary for real, substantive distinctions.\n"
        prompt += "All other types of values, particularly arbitrary text/prose, are not suitable for this particular subtask, because they can not be used in a symbolic solver.\n"
        prompt += "It is important to note that a given paper may not have values that speak to all of the slots in the extraction schema, and their value on a particular slot may be missing/None. This is okay and expected.\n"
        prompt += "\n"
        prompt += "## How many slots, and choosing the most informative ones\n"
        prompt += "Return NO MORE THAN 30 slots. Choose the slots you believe will actually be filled across many papers and will vary between them. A slot is only useful to a symbolic solver if it is (a) frequently present, and (b) takes different values in different papers. A slot that is almost always the same value, or almost always missing, contributes nothing and costs extraction accuracy on the slots that matter.\n"
        prompt += "Guidance drawn from measured results of an earlier version of this task:\n"
        prompt += "- Prefer a single NORMALISED quantity (a percentage change, ratio, or index relative to a matched control) over a pair of separate absolute baseline/condition measurements. Papers routinely state 'increased by 40%' but rarely state both the baseline and condition absolute values in a form that can be extracted; paired absolute slots come back almost entirely empty.\n"
        prompt += "- Only propose a numerical slot if papers in this literature typically state that quantity directly and explicitly. Do not propose a numerical slot whose value would have to be computed, inferred, or read off a figure.\n"
        prompt += "- Prefer categorical slots with a well-chosen, mutually exclusive vocabulary; these are reliably filled and are directly usable for rule and subgroup discovery.\n"
        prompt += "- Avoid slots whose vocabulary contains a dominant catch-all value (e.g. 'not_distinguished', 'not_tested') that will be selected for most rows; either sharpen the distinction so it can actually be made, or drop the slot.\n"
        prompt += "\n"

        # prompt += "Here are examples of high-level/open-ended slots:\n"
        # prompt += "- `key_factors`: Key factors that contribute to the phenomenon of interest (e.g. Alzheimer's disease development)\n"
        # prompt += "- `challenges`: Challenges that make it difficult to solve the problem of interest\n"
        # prompt += "- `proposed_approaches`: Proposed approaches to solving the problem of interest\n"
        # prompt += "- `research_methods`: A description of how the problem was studied/the research methods used\n"
        # prompt += "- `findings`: The key findings/results of the paper\n"
        # prompt += "\n"
        # prompt += "Here are examples of low-level/specific slots:\n"
        # prompt += "- `datasets_used`: A list of datasets used in the paper\n"
        # prompt += "- `model_architectures_used`: A list of model architectures used in the paper\n"
        # prompt += "- `evaluation_metrics_used`: A list of evaluation metrics used in the paper\n"
        # prompt += "- `model_performance_on_question_answering`: The performance of the model(s) used on a specific task (e.g. question answering), if applicable\n"
        # prompt += "- `model_performance_after_treatment`: The performance of the model(s) after a specific treatment/intervention (e.g. memory augmentation), if applicable\n"
        # prompt += "\n"
        prompt += "## Thinking about slots\n"
        prompt += "The slots in the extraction schema are designed to extract information that is useful for theory building.\n"
        prompt += "Theory building in this way is essentially an exercise in finding patterns in the extracted information across many papers, and then using those patterns to build theories that explain the phenomenon of interest.\n"
        prompt += "Some of these patterns can be sparse (i.e. occuring in only a small number of papers), and some can be dense (i.e. occurring in many papers).\n"
        prompt += "\n"
        prompt += "## Slots and their relationship to variables in theories:\n"
        prompt += "Slots are explicit variables.  They can be either quantitative (i.e. numbers), or qualitative (e.g. categories). As a toy example, if the extraction schema contained slots for 'force', 'mass', and 'acceleration', and the papers all described experiments measuring the force, mass, and acceleration of objects, then it'd be likely to discover a direct quantitative relationship between these variables (i.e. force = mass * acceleration) -- though of course this is a toy example because it's already known.\n"
        prompt += "Some theories/laws that are synthesized may be quantitative, qualitative, or a mix of both (e.g. 'relationship X appears to hold, but only when categorical variable Y has value Z, etc.).\n"
        prompt += "\n"
        prompt += "## Implicit/Hidden Variable Discovery\n"
        prompt += "In some cases, the most important variables to extract may be implicit, hidden, or otherwise non-obvious.\n"
        prompt += "Here is a real (but somewhat toy) example from the Chemistry domain, and an early qualitative law discovery system.\n"
        prompt += "The law that was discovered by the system was: `acids and alkalis react to produce a salt and water`.\n"
        prompt += "You can imagine that if the extraction slots directly included this classification information (i.e. whether a substance was an acid, alkali, salt), then it would be relatively straightforward to discover this law.\n"
        prompt += "It's also important to node that this information could have been included in a variety of ways:\n"
        prompt += "- Possibility: `reactant_1_classification`, `reactant_2_classification`: These are high-level slots, that aren't restricted to being a specific variable type (e.g. acid/alkali/salt), but it might get lucky and fill those classifications in for a subset of papers, and then the broader analysis would discover the pattern (there's a precision/recall tradeoff analog here: being specific makes it easier to discover a specific pattern, but harder to go fishing/abstract/discover somewhat unexpected patterns).\n"
        prompt += "- Possibility: `reactant_1_property_X`, `reactant_2_property_X`: These are more general slot ideas, that might be instantiated in specific ways (i.e. specific properties for the X), but if they contain the right information (e.g. pH level, or a different chemical property), then the higher-level categorical variable (i.e. low PH = acid, high PH = alkali) might be available to be discovered, though it could be harder.\n"
        prompt += "- Possibility: `is_reactant_1_an_acid`, `is_reactant_1_an_alkali`, `is_reactant_1_a_salt`: These are very specific slots (and the example here is binary -- they could also have been categorical), that directly ask for classification based on a specific categorization scheme, as long as that category set is known before hand, and has relatively straightforward classification criteria.\n"
        prompt += "- Possibility: `reactant_1_chemical_formula`, `reactant_2_chemical_formula`: These arelikely not appropriate here, because they are not categorical variables (i.e. each formula is potentially unique, so no pattern will be possible to find with symbolic solvers).  A case where this might be appropriate is when there are a few high-frequency chemicals that are common in the data, in which case this effectively becomes a categorical variable (and should be expressed/instantiated in the schema as such).\n"
        prompt += "One of the more interesting things is that the most important or useful variables might be non-obvious, or classifications that people haven't thought about before.\n"
        prompt += "That is: some discovered patterns might be rules over known classes (e.g. acids+alkalis -> salts, if this rule wasn't already known, but the classes of `acid`, `alkali`, and `salt` were known)\n"
        prompt += "BUT: some discovered patterns might be over new classes that haven't been explicitly recognized before. One of my favorite examples is that a colleague recently discovered (in a natural langugae processing task) that the standard input people were using actually contained two large, completely independent classes of problems that had been unrecognized, and that if you studied them independent, you discovered patterns that were obscured or difficult to see when you didn't know the classes existed.\n"
        prompt += "While that's a more classification-centered, splitting-hypothesis-centered example, there are also classic versions of other examples: such as the idea that tectonic plates are a hidden variable/mechanistic explanation that unifies (rather than split) seeminly disparate phenomena (e.g. earthquakes, volcanoes, mountain formation, continental drift) into a unifying theory, that also allows making better predictions, and asking better research questions.\n"
        prompt += "\n"
        prompt += "In summary:\n"
        prompt += "- Discovering novel rules over known variables can be impactful\n"
        prompt += "- Discovering novel classifications, or ways of looking at the data, that reveal new variables can be impactful\n"
        prompt += "- New variables can be low-level and specific, high-level and abstract, or anywhere in between, and still potentially be useful for theory building\n"
        prompt += "\n"

        prompt += "# Specific Task\n"
        prompt += "You will be provided with:\n"
        prompt += "- `theory_query_information`: This is a dictionary that contains information about the specific theory-building question or topic of interest.\n"
        prompt += "- The results of using an existing extraction run, and likely comparatively open-ended (and likely not symbolic-intended) extraction schema that has been run over a set of papers that are relevant to the theory-building question or topic of interest.  This is essentially to help you in the variable discovery process, and help find potentially relevant variables/information to extract from the papers, to support building high-utility/high-impact and novel theory/law discoveries.  The variable discovery results are not perfect, and may contain errors, omissions, or other issues, but they should provide a good starting point for thinking about what information to extract from the papers.\n"
        prompt += "\n"
        prompt += "Your task is to examine the theory query, and extraction schema/variable discovery brainstorming results, and suggest schema slots SUITABLE FOR A SUITE OF SYMBOLIC SOLVERS that you think would be useful for extracting information from a broad corpus of papers in support of the theory building process.\n"
        #prompt += "You can suggest up to 4 different extraction schemas, and each schema can have as many as 10-12 slots.\n"
        prompt += "\n"

        prompt += "# More specific, helpful instructions\n"
        prompt += "1) Table: The results can be thought of as a table. The columns are defined by the extraction schema, and the rows are the dictionaries populated from the papers. (Each paper may contribute zero, one, or more than one row, depending on whether there are multiple content matches in the paper).\n"
        prompt == "2) Column Variable Type Exploration: For each column, look at all the values of the row.\n"
        prompt += " - Are these values generally numerical? Or categorical? Or boolean?\n"
        prompt += " - Are there any patterns that appear to emerge from the values in the column? If so, write these down for each column.\n"
        prompt += "3) All same/low entropy: If the values in the column are generally all the same, then there is no variance in the column, so it is unlikely to be useful for theory building. In this case, you might consider if a finer-grained variable (that allows more variance) is appropriate.\n"
        prompt += "4) All different/high entropy: If the values in the column are generally all different kinds of knowledge, then it is unlikely to be useful for symbolic theory building, because there are no clear e.g. categorical patterns and everything is unique. In this case, you might consider if there are higher-level groupings or classifications, to turn these cases into categorical variables where regularities can be observed.\n"
        prompt += "5) Splitting: It is difficult, a priori, to necessarily know how to convert/split a prose description into different symbolic variables, and there may be multiple ways (and you should explore them all, and potentially include them, as long as they are not extremely sparse).\n"
        prompt += " - For example, a prose variable called 'performance' might have had all kinds of different content in it (e.g. '30% accuracy on task X', '90 F1 on task Y', 'showed that A was better than B', etc.).  These can't directly be turned into a numerical or categorical variable, because the grouping is too broad, and simply mixing units (e.g. '30' and '90' in a 'performance' category, even though they're different metrics, is not useful)\n"
        prompt += " - An alternative is to split these into multiple separate variables: For example, being metric focused: 'performance_accuracy', 'performance_f1', etc.\n"
        prompt += " - Another alternative is to split them into multiple separate variables on the task, which might be more meaningful for many scenarios: 'performance_accuracy_on_task_x', 'performance_f1_on_task_y', etc.\n"
        prompt += " - In many cases, relative differences are particularly useful.  These might be split to separate variables, that separate (for example) the baseline/control and experimental conditions: 'performance_with_memory', 'performance_without_memory'.  They could also be specified as 'performance_difference_score', if that is the primary way that they are reported in the papers (i.e. the papers do not report scores in both conditions).\n"
        prompt += " - It's possible that some, all, none of these decomposition styles, or different decomposition styles, may be appropriate for a given problem. What matters is that useful, interesting, inpactful patterns are discoverable by the symbolic solver.\n"
        prompt += "\n"

        prompt += "## Note on Categorical Variables\n"
        prompt += "Categorical variables can come in two main forms: ordered, and unordered.\n"
        prompt += "An unordered categorical variable is one where there is no clear ordering to the variables. For example, [`cat`, `dog`, `wolf`, `fox`] is an unordered categorical variable, because there is no underlying ordering to these.\n"
        prompt += "An ordered example is: [`small`, `medium`, `large`, `extra-large`], which has a clear ordering to the variables.  Another ordered example might be [`low`, `medium`, `high`].\n"
        prompt += "Some apparently unordered categorical variables may actually be ordered. For example, categorical colors (red, green, blue, etc.) are actually ordered, because they have an underlying numerical representation (their wavelength), and could be ordered as follows: [`violet`, `blue`, `green`, `yellow`, `orange`, `red`].  Another example is categorical chemical elements, which couuld be ordered by their atomic number (e.g. [`H`, `He`, `Li`, `Be`, `B`, `C`, `N`, `O`, `F`, `Ne`, ...]).\n"
        prompt += "Why is this important: Categorical variables that are ordered have additional structure that may be able to be exploted by symbolic solvers. When possible, please organize genuinely ordered categorical variables appropriately.\n"
        prompt += "More details: Categorical variables that are marked as being ordered will automatically be converted by some symbolic solvers into integers, to see if they can be used as numeric variables in equations.\n"
        prompt += "For example: {'violet': 1, 'blue': 2, 'green': 3, 'yellow': 4, 'orange': 5, 'red': 6}.\n"
        prompt += "In this way, it's ideal to keep the spacing between categories relatively even, so the integers are meaningful.\n"
        prompt += "Here's a negative example: ['single-atom-sized', 'small', 'building-sized', 'two-building-sized', 'continent-sized', etc] -- there's no clear regular spacing between these categories (e.g. single-atom-sized is likely much smaller than 'small', whatever that means, and the relative difference between 'building-sized' and 'two-building-sized' is way smaller than 'building-sized' and 'continent-sized').  This uneven spacing may be unavoidable in some cases, but if possible and straightforward, please try to maintain reasonable spacing for ordered categorical variables with some clear implicit underlying value (e.g. approximate size in meters).\n"
        prompt += "\n"

        prompt += "# Theory Request\n"
        prompt += "Here is the theory request:\n"
        prompt += "```\n"
        if (isinstance(theory_query_information, dict)):
            prompt += json.dumps(theory_query_information, indent=4)
        else:
            prompt += str(theory_query_information) + "\n"
        prompt += "```\n"
        prompt += "\n"

        prompt += "# Variable Discovery Results\n"
        prompt += "Here are the results from the variable discovery process.  Each process was undertaken on a single paper, and then the results (per paper) are shown here.\n"
        prompt += "Some central themes may emerge across many papers, while some subsets of papers may suggest unique or interesting variables/slots that might be indicative of broader patters, where they investigated more generally.\n"
        prompt += "NOTE: The variable discovery results are not perfect, and they were made by a less powerful model, so they may contain errors, omissions, or other issues.  But they should provide a good starting point for thinking about what information to extract from the papers.\n"
        prompt += "```\n"
        if (isinstance(extracted_records, dict) or isinstance(extracted_records, list)):
            prompt += json.dumps(extracted_records, indent=4)
        else:
            prompt += str(extracted_records) + "\n"
        prompt += "```\n"
        prompt += "\n"

        # prompt += "# Thinking about the goals of a good extraction schema. "
        # prompt += "The goal of the `extraction schema` is to broadly extract entities/relevant information that can be used to build theories about the topic.\n"
        # prompt += "\n"
        # prompt += "Things to consider:\n"
        # prompt += "- The schema should capture evidence that is relevant to the theory request.  A theory needs to explain a wide variety of evidence, so it's important to consider not just evidence that might support a theory, but also extract evidence/counterpoints that might provide challenges to creating an all-encompassing theory.\n"
        # prompt += "\n"
        # prompt += "## Examples\n"
        # prompt += "Here are some examples of theory requests, and the corresponding paper search queries and extraction schemas.  These are illustrative only, and may have issues/challenges -- you can likely do much better if you think about it.\n"
        # prompt += "\n"
        # prompt += "The description that you choose for each variable will likely be a mix of the different fields in the variable discovery results. The intended audience for the schema descriptions is an extraction language model, that will be given the schema and a paper, and asked to extract any instances of the variables in the schema from the paper. (Each paper might return zero, one, or multiple instances of matches to a schema).\n"
        # prompt += "\n"
        # prompt += "Here is a non-exhaustive list of some of the aspects in the extraction schema examples that help make them useful for the theory extraction task:\n"
        # prompt += "- Requests to extract specific types of entities (e.g. model names, agent names, benchmark names, etc.)\n"
        # prompt += "- Requests to extract specific types of information or properties about the entities (e.g. descriptions, performance, categories/types, aspects such as model size or other properties that might be relevant, etc.)\n"
        # prompt += "- Checks for binary conditions that are relevant to the criteria (e.g. does the agent use memory, does the paper report performance with and without memory, etc.)\n"
        # prompt += "- Requests for counter-examples or comparisons (e.g. not only performance with memory, but also performance without memory, so these conditions can be compared)\n"
        # prompt += "- Requests for results of studies, summarized in brief/punchy and information-dense ways (remember, this extraction will be used on many papers, so the results should be concise and information-dense so the important aspects can easily stand out)\n"
        # prompt += "\n"
        # prompt += "Example 1:\n"
        # prompt += "```\n"
        # prompt += "{\n"
        # prompt += "    \"theory_request_original\": \"A theory of how agents can use memory to play text games\",\n"
        # prompt += "    \"theory_request_normalized\": \"Build a theory of how agents can use memory to play text games, based on the following results.\",\n"
        # prompt += "    \"short_name\": \"agent-memory-text-games\",\n"
        # prompt += "    \"paper_search_query\": \"Extract any mentions of agents playing specific text games, and whether or not the agents use a memory.\",\n"
        # prompt += "    \"extraction_query\": \"Extract any mentions of agents playing specific text games, and whether or not the agents use a memory.\",\n"
        # prompt += "    \"extraction_schema\": [\n"
        # prompt += "        {\"name\": \"agent_name\", \"type\": \"str\", \"description\": \"The name of the agent that plays the game.\"},\n"
        # prompt += "        {\"name\": \"agent_description\", \"type\": \"str\", \"description\": \"A brief description of the agent that plays the game.\"},\n"
        # prompt += "        {\"name\": \"benchmark_name\", \"type\": \"str\", \"description\": \"The name of the text game benchmark that the agent plays\"},\n"
        # prompt += "        {\"name\": \"agent_memory\", \"type\": \"bool\", \"description\": \"Does the agent use a form of memory that learns from the task? (true, false, or null for no information)\"},\n"
        # prompt += "        {\"name\": \"agent_memory_type\", \"type\": \"str\", \"description\": \"If the agent uses a memory, describe the type of memory, and what the representations in the memory are.\"},\n"
        # prompt += "        {\"name\": \"performance_with_memory\", \"type\": \"str\", \"description\": \"If the agent uses a memory, what is the performance of the agent on the benchmark when it uses that memory? (numerical, include units. null if no information)\"},\n"
        # prompt += "        {\"name\": \"performance_without_memory\", \"type\": \"str\", \"description\": \"What is the performance of the vanilla (i.e. no-memory) version of the agent on the benchmark? (numerical, include units. null if no information)\"},\n"
        # prompt += "        {\"name\": \"has_performance_with_without_memory\", \"type\": \"bool\", \"description\": \"Does the paper report performance on this agent both with and without memory? (true, false, or null for no information)\"}\n"
        # prompt += "    ],\n"
        # prompt += "    \"notes\": \"...\",\n"
        # prompt += "}\n"
        # prompt += "```\n"
        # prompt += "\n"
        # prompt += "Example 2:\n"
        # prompt += "```\n"
        # prompt += "{\n"
        # prompt += "    \"theory_request_original\": \"I'd like a theory of how agents can process/perform different kinds of arithmetic/mathematics\",\n"
        # prompt += "    \"theory_request_normalized\": \"Build a theory of how agents can process/perform different kinds of arithmetic/mathematics, based on the following results.\",\n"
        # prompt += "    \"short_name\": \"agents-perform-arithmetics-mathematics\",\n"
        # prompt += "    \"paper_search_query\": \"Extract any mentions of the performance of LLMs on artithmetic problems.\",\n"
        # prompt += "    \"extraction_query\": \"Extract any mentions of the performance of LLMs on artithmetic problems.\",\n"
        # prompt += "    \"extraction_schema\": [\n"
        # prompt += "        {\"name\": \"model_name\", \"type\": \"str\", \"description\": \"The name of the LLM model that is being evaluated.\"},\n"
        # prompt += "        {\"name\": \"model_description\", \"type\": \"str\", \"description\": \"A brief description of the LLM model.\"},\n"
        # prompt += "        {\"name\": \"model_size\", \"type\": \"str\", \"description\": \"How large was the model, in parameters? (e.g. 1B, 2B, 7B, 13B, 70B, etc.)\"},\n"
        # prompt += "        {\"name\": \"math_problems\", \"type\": \"str\", \"description\": \"What kind of math problems were used to evaluate the model? Be specific.\"},\n"
        # prompt += "        {\"name\": \"performance\", \"type\": \"str\", \"description\": \"What was the performance of the model on the math problems? Ideally this should be as detailed as possible, and broken down by problem type. (numerical, include units. null if no information)\"}\n"
        # prompt += "    ],\n"
        # prompt += "    \"notes\": \"...\",\n"
        # prompt += "}\n"
        # prompt += "```\n"
        # prompt += "\n"
        # prompt += "Example 3:\n"
        # prompt += "```\n"
        # prompt += "{\n"
        # prompt += "    \"theory_request_original\": \"Please investigate a theory for how probing language models works\",\n"
        # prompt += "    \"theory_request_normalized\": \"Build a theory of how probing works in large language models.\",\n"
        # prompt += "    \"short_name\": \"probing-large-language-models\",\n"
        # prompt += "    \"paper_search_query\": \"Extract any mentions of different types of probes for probing large language models (LLMs).\",\n"
        # prompt += "    \"extraction_query\": \"Extract any mentions of different types of probes for probing large language models (LLMs).\",\n"
        # prompt += "    \"extraction_schema\": [\n"
        # prompt += "        {\"name\": \"probe_name\", \"type\": \"str\", \"description\": \"The name of the probe that is being evaluated.\"},\n"
        # prompt += "        {\"name\": \"probe_description\", \"type\": \"str\", \"description\": \"A detailed description of the probe.\"},\n"
        # prompt += "        {\"name\": \"what_is_probed\", \"type\": \"str\", \"description\": \"What is the probe probing? (e.g. 'the model's ability to perform arithmetic', 'the model's ability to understand natural language', etc.)\"},\n"
        # prompt += "        {\"name\": \"results\", \"type\": \"str\", \"description\": \"What are the results of the probe? (e.g. 'the model's performance on arithmetic problems', 'the model's performance on natural language understanding tasks', etc.)\"}\n"
        # prompt += "    ],\n"
        # prompt += "    \"notes\": \"...\",\n"
        # prompt += "}\n"
        # prompt += "```\n"
        # prompt += "\n"
        # prompt += "Example 4:\n"
        # prompt += "```\n"
        # prompt += "{\n"
        # prompt += "    \"theory_request_original\": \"Could you create one or more theories of how theory-of-mind works in large language models (LLMs).\",\n"
        # prompt += "    \"theory_request_normalized\": \"Build a theory of how theory-of-mind works in large language models (LLMs), based on the following results.\",\n"
        # prompt += "    \"short_name\": \"theory-of-mind-llms\",\n"
        # prompt += "    \"paper_search_query\": \"Extract any mentions of how LLMs perform theory-of-mind tasks, and what the results are.\",\n"
        # prompt += "    \"extraction_query\": \"Extract any mentions of how LLMs perform theory-of-mind tasks, and what the results are.\",\n"
        # prompt += "    \"extraction_schema\": [\n"
        # prompt += "        {\"name\": \"model_name\", \"type\": \"str\", \"description\": \"The name of the LLM model that is being evaluated.\"},\n"
        # prompt += "        {\"name\": \"model_description\", \"type\": \"str\", \"description\": \"A brief description of the LLM model.\"},\n"
        # prompt += "        {\"name\": \"model_size\", \"type\": \"str\", \"description\": \"How large was the model, in parameters? (e.g. 1B, 2B, 7B, 13B, 70B, etc.)\"},\n"
        # prompt += "        {\"name\": \"task_name\", \"type\": \"str\", \"description\": \"The name of the theory-of-mind task that the model is being evaluated on.\"},\n"
        # prompt += "        {\"name\": \"task_description\", \"type\": \"str\", \"description\": \"A brief description of the theory-of-mind task that the model is being evaluated on.\"},\n"
        # prompt += "        {\"name\": \"performance\", \"type\": \"str\", \"description\": \"What was the performance of the model on the theory-of-mind task? Ideally this should be as detailed as possible, and broken down by task type. (numerical, include units. null if no information)\"}\n"
        # prompt += "    ],\n"
        # prompt += "    \"notes\": \"...\",\n"
        # prompt += "}\n"
        # prompt += "```\n"
        # prompt += "\n"

        if (reflection is not None):
            prompt += "\n"
            prompt += "# Reflection\n"
            prompt += "This is a reflection step. Previously, you generated a response (below).  Now, your task is to reflect on that response, and fix any errors, inconsistencies, omissions, or any other issues.\n"
            prompt += "```\n"
            prompt += reflection + "\n"
            prompt += "```\n"
            prompt += "\n"

        prompt += "# Output Format\n"
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output, as these are not valid JSON and will cause a parsing error.\n"
        prompt += "\n"
        prompt += "Your JSON response must be a dictionary with a single key, `extraction_schema`, that has the following keys:\n"
        prompt += "Each extraction schema dictionary must contains exactly 6 keys: `theory_request_original`, `theory_request_normalized`, `short_name`, `paper_search_query`, `extraction_query`, and `extraction_schema`.  The values for these keys are as follows:\n"
        prompt += "- `theory_request_original`(str): The original theory request that was provided (e.g. 'make me a theory about X')\n"
        prompt += "- `theory_request_normalized`(str): A normalized, clear version of the theory request, that always starts with `Build a theory of...` and ends with `based on the following results.`.\n"
        prompt += "- `short_name`(str): A short name for the theory request, that is a concise, and generally 4 words long, separated by '-'. This will be a prefix appended to filenames to save data for this request.\n"
        prompt += "- `paper_search_query`(str): The query that should be used to search for papers that are relevant to the theory request, using a paper search tool over Semantic Scholar.\n"
        prompt += "- `extraction_query`(str): A natural language request query, that's provided to an extraction prompt (along with the extraction schema below) to outline the overall purpose of the extraction process.  Should start with `Extract any mentions of...`.\n"
        prompt += "- `extraction_schema`(list): A list of dictionaries, where each dictionary contains the following keys:\n"
        prompt += "    - `name`(str): The name of the entity to extract (e.g. 'entity_name')\n"
        prompt += "    - `type`(str): The type of the entity to extract (valid types are: 'int', 'float', 'bool', or 'categorical')\n"
        prompt += "    - `inclusion_criteria`(str): A very brief, concise, and helpful description to the extraction model that explicitly spells out what/when information should be extracted. If inclusion criteria are not met, no variable value will be set. Max 1-2 sentences.\n"
        prompt += "    - `exclusion_criteria`(str): A very brief, concise, and helpful description to the extraction model that explicitly spells out if conditions exist where information should NOT be extracted. If exclusion criteria are met, no variable value will be set. Max 1-2 sentences.\n"
        prompt += "    - `description`(str): A brief, concise, and helpful description of the entity to extract, that very explicitly spells out what information should be extracted. Numerical values MUST always explicitly specify the units of the number (e.g. kg, cm, precision@10, F1, etc.).\n"
        prompt += "    - `unit`(str): For all numerical variables, you are required to have an EXPLICIT unit (e.g. kg, cm, precision@10, F1, etc.).  You are NOT allowed to have a dynamic unit (e.g. 'in units specified by the paper') because the values will NOT be comparable, unless there is some *very special case* where the issue is not of normalization (e.g. having to convert numbers), but where there is some clear utility and semantics in the number (like a relative difference), in which case, you MUST also include the phrase '** SPECIAL UNIT **' (**'s required for automatic pattern matching) and a description of how this meets the requirements and will not generate noise, at the end of this string.  If the variable is not numerical, then set this to None.  Similiarly, do NOT explicitly use shortforms -- be VERY descript, since the extraction model is cheap and underpowered. For example, `millimeters (mm)`, not `mm`. `accuracy, expressed as a percentage (%)`, not `accuracy` (which might allow anything from percentages to normalized 0-1 ranges, etc). The more specific you are here, and the better you do here, the better the extraction results will be.\n"
        prompt += "    - `examples`:list: A list of 3-5 examples of what the entity/values look like, to help the extration model. Each example element should be in the native format, e.g. int/float should be numeric, bool should be true/false, categorical should be a string. IF the variable is categorical, then leave this None, since `categorical_values` lists all valid values.\n"
        prompt += "    - `categorical_values`(list): IF (and only if) this variable is categorical, then this is a list of the only valid categorical values that the variable can take on. If the categorical variable is potentially an ORDERED variable, then these should be in the exact canonical order.  If this is not a categorical variable, set this variable to None.\n"
        prompt += "    - `categorical_values_description`(str): IF (and only if) this variable is categorical, then this is a brief description of the categorical values, and what they mean. If this is not a categorical variable, set this variable to None.\n"
        prompt += "    - `categorical_values_ordered`(bool): IF (and only if) this variable is categorical, then this is a boolean that indicates whether the categorical values are ordered or not. If this is not a categorical variable, set this variable to None.\n"
        #prompt += "- `notes`(str): A brief, concise, 1-2 sentence note to the user to mention what this schema was designed for\n"
        prompt += "\n"

        prompt += "## Special instructions\n"
        prompt += "- You can (and are encouraged to) think and plan before writing your JSON response.  But, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output.\n"
        prompt += "- Please follow the instructions carefully.\n"
        prompt += "- Do not hallucinate.\n"

        return prompt


    total_cost = 0.0
    responses = []
    responses_raw = []

    # Create the prompt
    #prompt = mkPrompt(theory_request)
    prompt = mkPrompt(theory_query_information, extracted_records, reflection=None)
    # Call the LLM
    responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=0.0, maxTokens=max(max_tokens, DEFAULT_MAX_TOKENS), jsonOut=False)
    responses.append(responseJSON)
    responses_raw.append(responseText)
    total_cost += cost

    # If we are using reflection, we will call the LLM again with the reflection step
    if (use_reflection):
        # Create the reflection prompt
        reflection_prompt = mkPrompt(theory_query_information, extracted_records, reflection=responseText)
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

    # Try to determine the number of variables/slots in the extraction schema
    num_variables_in_schema = None
    try:
        if (best_response is not None and "extraction_schema" in best_response):
            num_variables_in_schema = len(best_response["extraction_schema"]["extraction_schema"])
    except Exception as e:
        num_variables_in_schema = None

    # Pack output
    packed = {
        #"theory_request": theory_request,
        "theory_query_information": theory_query_information,
        "model": model_str,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "use_reflection": use_reflection,
        "num_variables_in_schema": num_variables_in_schema,
        "output": best_response,
        "responses": responses,
        "responses_raw": responses_raw,
        "extracted_records": extracted_records,
        "total_cost": total_cost,
    }
    return packed



#
#   Implicit Variable Discovery from Extracted Data
#
#
#   Schema generation from variable discovery
#
def generate_variable_discovery_from_data_report(theory_query_information:str, extracted_records:list, model_str:str, max_tokens:int=8000, temperature:float=0.0, use_reflection:bool=False):
    def mkPrompt(theory_query_information, extracted_records, reflection:str=None):
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"

        prompt += "# Task\n"
        prompt += "You are part of a system that is designed to infer scientific theories on specific topics, based on evidence gathered from scientific papers.\n"
        prompt += "The steps in the process are:\n"
        prompt += "1. Collect the theory request from the user (e.g. 'make me a theory about X')\n"
        prompt += "2. Convert the theory request into a paper-search query.\n"
        prompt += "3. Read the papers to discover potentially relevant variables/information to extract from the papers, to support building high-utility/high-impact and novel theory/law discoveries.\n"
        prompt += "4. Build extraction schema for extracting entities/supporting information from scientific papers, based on the variable discovery results.\n"
        prompt += "5. Use the query and schema to extract entities from scientific papers.\n"
        prompt += "*6. Re-examine the data extracted from papers to see if there are implicit, hidden, or other variables not captured from the extraction schema.\n"
        prompt += "7. Re-build / Re-extract data based on discovered variables, to make the data as clean, and easy-to-spot patterns as possible.\n"
        prompt += "6. Use this final set of data to build theories about the topic.\n"
        prompt += "\n"
        prompt += "We are currently on step 6.\n"
        #prompt += "You will be given a theory request, and the results from a variable discovery process, and your task is to convert it into one (or more) extraction schema for extracting entities from scientific papers that would provide highly relevant evidence to support building theories in this domain.\n"
        prompt += "You will be given a theory request, and the results from extracting data from papers based on an initial extraction schema. Your task is to examine the extracted data, and see if there are any implicit, hidden, or other variables that are not captured in the initial extraction schema, that could be captured in a new extraction schema, that would make it easier to discover patterns and build theories from the data.\n"
        prompt += "\n"


        prompt += "## High-Level Task Context\n"
        prompt += "The overall task is to formulate new theories/laws based on scientific evidence, in part through a large-scale literature search.\n"
        prompt += "The general process is: (a) extract information from papers using a detailed extraction schema, that can supply information useful for theory building, then (b) use the extracted information across many (dozens/hundreds) of papers to infer patterns and build theories.\n"
        prompt += "The slots in the extraction schema can be high-level/broad, or low-level/specific.\n"
        prompt += "\n"
        prompt += "Here are examples of high-level/open-ended slots:\n"
        prompt += "- `key_factors`: Key factors that contribute to the phenomenon of interest (e.g. Alzheimer's disease development)\n"
        prompt += "- `challenges`: Challenges that make it difficult to solve the problem of interest\n"
        prompt += "- `proposed_approaches`: Proposed approaches to solving the problem of interest\n"
        prompt += "- `research_methods`: A description of how the problem was studied/the research methods used\n"
        prompt += "- `findings`: The key findings/results of the paper\n"
        prompt += "\n"
        prompt += "Here are examples of low-level/specific slots:\n"
        prompt += "- `datasets_used`: A list of datasets used in the paper\n"
        prompt += "- `model_architectures_used`: A list of model architectures used in the paper\n"
        prompt += "- `evaluation_metrics_used`: A list of evaluation metrics used in the paper\n"
        prompt += "- `model_performance_on_question_answering`: The performance of the model(s) used on a specific task (e.g. question answering), if applicable\n"
        prompt += "- `model_performance_after_treatment`: The performance of the model(s) after a specific treatment/intervention (e.g. memory augmentation), if applicable\n"
        prompt += "\n"
        prompt += "## Thinking about slots\n"
        prompt += "The slots in the extraction schema are designed to extract information that is useful for theory building.\n"
        prompt += "Theory building in this way is essentially an exercise in finding patterns in the extracted information across many papers, and then using those patterns to build theories that explain the phenomenon of interest.\n"
        prompt += "Some of these patterns can be sparse (i.e. occuring in only a small number of papers), and some can be dense (i.e. occurring in many papers).\n"
        prompt += "\n"
        prompt += "## Slots and their relationship to variables in theories:\n"
        prompt += "Sometimes slots are specific enough, that they themselves are very close to variables that would appear in a theory.  As a toy example, if the extraction schema contained slots for 'force', 'mass', and 'acceleration', and the papers all described experiments measuring the force, mass, and acceleration of objects, then it'd be likely to discover a direct quantitative relationship between these variables (i.e. force = mass * acceleration) -- though of course this is a toy example because it's already known.\n"
        prompt += "Other times, slots are more high-level and broad, and they might contain a mix of different kinds of information that would need to be further processed/analyzed to discover finer-grained (or even more abstracted) patterns.\n"
        prompt += "Some slots can be filled with quantitative information (e.g. a number), some qualitative information, some categorical information, some prose descriptions, short lists, or other formats -- whatever is needed.\n"
        prompt += "In this way, some theories/laws that are synthesized may be quantitative, qualitative, or a mix of both.\n"
        prompt += "\n"
        prompt += "## Implicit/Hidden Variable Discovery\n"
        prompt += "In some cases, the most important variables to extract may be implicit, hidden, or otherwise non-obvious.\n"
        prompt += "Here is a real (but somewhat toy) example from the Chemistry domain, and an early qualitative law discovery system.\n"
        prompt += "The law that was discovered by the system was: `acids and alkalis react to produce a salt and water`.\n"
        prompt += "You can imagine that if the extraction slots directly included this classification information (i.e. whether a substance was an acid, alkali, salt), then it would be relatively straightforward to discover this law.\n"
        prompt += "It's also important to node that this information could have been included in a variety of ways:\n"
        prompt += "- Possibility: `reactant_1_classification`, `reactant_2_classification`: These are high-level slots, that aren't restricted to being a specific variable type (e.g. acid/alkali/salt), but it might get lucky and fill those classifications in for a subset of papers, and then the broader analysis would discover the pattern (there's a precision/recall tradeoff analog here: being specific makes it easier to discover a specific pattern, but harder to go fishing/abstract/discover somewhat unexpected patterns).\n"
        prompt += "- Possibility: `reactant_1_properties`, `reactant_2_properties`: These are more general slots, that could contain a variety of information about the reactants, but if they contain the right information (e.g. pH level, or other chemical properties), then the broader analysis might discover the pattern (though this is harder than the previous possibility, because the information is less standardized and more variable across papers, so it's harder to discover patterns in it).\n"
        prompt += "- Possibility: `is_reactant_1_an_acid`, `is_reactant_1_an_alkali`, `is_reactant_1_a_salt`: These are very specific slots (and could have been more categorical: e.g. `classification_acid_alkali_salt`), that directly ask for classification based on a specific categorization scheme.\n"
        prompt += "- Possibility: `reactant_1_chemical_formula`, `reactant_2_chemical_formula`: These are more indirect, that ask for chemical formulas, but if the analysis is sophisticated enough, it might be able to infer the classifications from the chemical formulas (though this is much harder than the previous possibilities, because it requires more chemical knowledge and reasoning to notice the pattern/infer the classifications from the chemical formulas).\n"
        prompt += "- Possibility: `reactant_1_chemical_name`, `reactant_2_chemical_name`: These are even harder, as there would be quite a bit of effort required to discover that chemical X is an acid, chemical Y is an alkali, and chemical Z is a salt, based on their names alone (though in some cases it might be possible to infer this from the names, if the names contain indicative information).\n"
        prompt += "One of the more interesting things is that the most important or useful variables might be non-obvious, or classifications that people haven't thought about before.\n"
        prompt += "That is: some discovered patterns might be rules over known classes (e.g. acids+alkalis -> salts, if this rule wasn't already known, but the classes of `acid`, `alkali`, and `salt` were known)\n"
        prompt += "BUT: some discovered patterns might be over new classes that haven't been explicitly recognized before. One of my favorite examples is that a colleague recently discovered (in a natural langugae processing task) that the standard input people were using actually contained two large, completely independent classes of problems that had been unrecognized, and that if you studied them independent, you discovered patterns that were obscured or difficult to see when you didn't know the classes existed.\n"
        prompt += "While that's a more classification-centered, splitting-hypothesis-centered example, there are also classic versions of other examples: such as the idea that tectonic plates are a hidden variable/mechanistic explanation that unifies (rather than split) seeminly disparate phenomena (e.g. earthquakes, volcanoes, mountain formation, continental drift) into a unifying theory, that also allows making better predictions, and asking better research questions.\n"
        prompt += "\n"
        prompt += "In summary:\n"
        prompt += "- Discovering novel rules over known variables can be impactful\n"
        prompt += "- Discovering novel classifications, or ways of looking at the data, that reveal new variables can be impactful\n"
        prompt += "- New variables can be low-level and specific, high-level and abstract, or anywhere in between, and still potentially be useful for theory building\n"
        prompt += "\n"

        prompt += "# Specific Task\n"
        prompt += "You will be provided with:\n"
        prompt += "- `theory_query_information`: This is a dictionary that contains information about the specific theory-building question or topic of interest.\n"
        #prompt += "- The results of a variable discovery process, that has been run over a set of papers that are relevant to the theory-building question or topic of interest.  This variable discovery process is designed to find potentially relevant variables/information to extract from the papers, to support building high-utility/high-impact and novel theory/law discoveries.  The variable discovery results are not perfect, and may contain errors, omissions, or other issues, but they should provide a good starting point for thinking about what information to extract from the papers.\n"
        prompt += "- The results of extracting data from papers based on an initial extraction schema. This is a list of dictionaries, where each dictionary contains an extraction instance from a paper.  The keys are variables/slots, and the values are the values extracted from the papers. Each paper may have zero, one, or more than one extraction records."
        prompt += "\n"
        #prompt += "Your task is to examine the theory query, and variable discovery results, and suggest schema slots that you think would be useful for extracting information from a broad corpus of papers in support of the theory building process.\n"
        #prompt += "You can suggest up to 4 different extraction schemas, and each schema can have as many as 10-12 slots.\n"
        prompt += "Your task is to examine the theory query, and the extracted data from papers based on an initial extraction schema, and suggest new schema slots, new variables, distilled/higher signal-to-noise slots/variables, or other suggestions that would help sharpen the likelihood of discovering novel, impactful patterns/theories/laws from the data.\n"
        prompt += "\n"


        prompt += "# Theory Request\n"
        prompt += "Here is the theory request:\n"
        prompt += "```\n"
        if (isinstance(theory_query_information, dict)):
            prompt += json.dumps(theory_query_information, indent=4)
        else:
            prompt += str(theory_query_information) + "\n"
        prompt += "```\n"
        prompt += "\n"

        #prompt += "# Variable Discovery Results\n"
        #prompt += "Here are the results from the variable discovery process.  Each process was undertaken on a single paper, and then the results (per paper) are shown here.\n"
        #prompt += "Some central themes may emerge across many papers, while some subsets of papers may suggest unique or interesting variables/slots that might be indicative of broader patters, where they investigated more generally.\n"
        #prompt += "NOTE: The variable discovery results are not perfect, and they were made by a less powerful model, so they may contain errors, omissions, or other issues.  But they should provide a good starting point for thinking about what information to extract from the papers.\n"
        #prompt += "```\n"
        #if (isinstance(variable_discovery_results, dict) or isinstance(variable_discovery_results, list)):
        #    prompt += json.dumps(variable_discovery_results, indent=4)
        #else:
        #    prompt += str(variable_discovery_results) + "\n"
        #prompt += "```\n"
        #prompt += "\n"

        prompt += "# Extracted Data from Papers\n"
        prompt += "Here are the results from extracting data from papers based on an initial extraction schema.\n"
        prompt += "NOTE: The results are not perfect, as they were made by a less powerful model, so they may contain errors, omissions, or other issues.\n"
        prompt += "```\n"
        if (isinstance(extracted_records, dict) or isinstance(extracted_records, list)):
            prompt += json.dumps(extracted_records, indent=4)
        else:
            prompt += str(extracted_records) + "\n"
        prompt += "```\n"
        prompt += "\n"


        # prompt += "# Thinking about the goals of a good schema slots / variables. "
        # prompt += "The goal of the `extraction schema` is to broadly extract entities/relevant information that can be used to build theories about the topic.\n"
        # prompt += "\n"
        # prompt += "Things to consider:\n"
        # prompt += "- The schema should capture evidence that is relevant to the theory request.  A theory needs to explain a wide variety of evidence, so it's important to consider not just evidence that might support a theory, but also extract evidence/counterpoints that might provide challenges to creating an all-encompassing theory.\n"
        # prompt += "\n"
        # prompt += "## Examples\n"
        # prompt += "Here are some examples of theory requests, and the corresponding paper search queries and extraction schemas.  These are illustrative only, and may have issues/challenges -- you can likely do much better if you think about it.\n"
        # prompt += "\n"
        # prompt += "The description that you choose for each variable will likely be a mix of the different fields in the variable discovery results. The intended audience for the schema descriptions is an extraction language model, that will be given the schema and a paper, and asked to extract any instances of the variables in the schema from the paper. (Each paper might return zero, one, or multiple instances of matches to a schema).\n"
        # prompt += "\n"
        # prompt += "Here is a non-exhaustive list of some of the aspects in the extraction schema examples that help make them useful for the theory extraction task:\n"
        # prompt += "- Requests to extract specific types of entities (e.g. model names, agent names, benchmark names, etc.)\n"
        # prompt += "- Requests to extract specific types of information or properties about the entities (e.g. descriptions, performance, categories/types, aspects such as model size or other properties that might be relevant, etc.)\n"
        # prompt += "- Checks for binary conditions that are relevant to the criteria (e.g. does the agent use memory, does the paper report performance with and without memory, etc.)\n"
        # prompt += "- Requests for counter-examples or comparisons (e.g. not only performance with memory, but also performance without memory, so these conditions can be compared)\n"
        # prompt += "- Requests for results of studies, summarized in brief/punchy and information-dense ways (remember, this extraction will be used on many papers, so the results should be concise and information-dense so the important aspects can easily stand out)\n"
        # prompt += "\n"
        # prompt += "Example 1:\n"
        # prompt += "```\n"
        # prompt += "{\n"
        # prompt += "    \"theory_request_original\": \"A theory of how agents can use memory to play text games\",\n"
        # prompt += "    \"theory_request_normalized\": \"Build a theory of how agents can use memory to play text games, based on the following results.\",\n"
        # prompt += "    \"short_name\": \"agent-memory-text-games\",\n"
        # prompt += "    \"paper_search_query\": \"Extract any mentions of agents playing specific text games, and whether or not the agents use a memory.\",\n"
        # prompt += "    \"extraction_query\": \"Extract any mentions of agents playing specific text games, and whether or not the agents use a memory.\",\n"
        # prompt += "    \"extraction_schema\": [\n"
        # prompt += "        {\"name\": \"agent_name\", \"type\": \"str\", \"description\": \"The name of the agent that plays the game.\"},\n"
        # prompt += "        {\"name\": \"agent_description\", \"type\": \"str\", \"description\": \"A brief description of the agent that plays the game.\"},\n"
        # prompt += "        {\"name\": \"benchmark_name\", \"type\": \"str\", \"description\": \"The name of the text game benchmark that the agent plays\"},\n"
        # prompt += "        {\"name\": \"agent_memory\", \"type\": \"bool\", \"description\": \"Does the agent use a form of memory that learns from the task? (true, false, or null for no information)\"},\n"
        # prompt += "        {\"name\": \"agent_memory_type\", \"type\": \"str\", \"description\": \"If the agent uses a memory, describe the type of memory, and what the representations in the memory are.\"},\n"
        # prompt += "        {\"name\": \"performance_with_memory\", \"type\": \"str\", \"description\": \"If the agent uses a memory, what is the performance of the agent on the benchmark when it uses that memory? (numerical, include units. null if no information)\"},\n"
        # prompt += "        {\"name\": \"performance_without_memory\", \"type\": \"str\", \"description\": \"What is the performance of the vanilla (i.e. no-memory) version of the agent on the benchmark? (numerical, include units. null if no information)\"},\n"
        # prompt += "        {\"name\": \"has_performance_with_without_memory\", \"type\": \"bool\", \"description\": \"Does the paper report performance on this agent both with and without memory? (true, false, or null for no information)\"}\n"
        # prompt += "    ],\n"
        # prompt += "    \"notes\": \"...\",\n"
        # prompt += "}\n"
        # prompt += "```\n"
        # prompt += "\n"
        # prompt += "Example 2:\n"
        # prompt += "```\n"
        # prompt += "{\n"
        # prompt += "    \"theory_request_original\": \"I'd like a theory of how agents can process/perform different kinds of arithmetic/mathematics\",\n"
        # prompt += "    \"theory_request_normalized\": \"Build a theory of how agents can process/perform different kinds of arithmetic/mathematics, based on the following results.\",\n"
        # prompt += "    \"short_name\": \"agents-perform-arithmetics-mathematics\",\n"
        # prompt += "    \"paper_search_query\": \"Extract any mentions of the performance of LLMs on artithmetic problems.\",\n"
        # prompt += "    \"extraction_query\": \"Extract any mentions of the performance of LLMs on artithmetic problems.\",\n"
        # prompt += "    \"extraction_schema\": [\n"
        # prompt += "        {\"name\": \"model_name\", \"type\": \"str\", \"description\": \"The name of the LLM model that is being evaluated.\"},\n"
        # prompt += "        {\"name\": \"model_description\", \"type\": \"str\", \"description\": \"A brief description of the LLM model.\"},\n"
        # prompt += "        {\"name\": \"model_size\", \"type\": \"str\", \"description\": \"How large was the model, in parameters? (e.g. 1B, 2B, 7B, 13B, 70B, etc.)\"},\n"
        # prompt += "        {\"name\": \"math_problems\", \"type\": \"str\", \"description\": \"What kind of math problems were used to evaluate the model? Be specific.\"},\n"
        # prompt += "        {\"name\": \"performance\", \"type\": \"str\", \"description\": \"What was the performance of the model on the math problems? Ideally this should be as detailed as possible, and broken down by problem type. (numerical, include units. null if no information)\"}\n"
        # prompt += "    ],\n"
        # prompt += "    \"notes\": \"...\",\n"
        # prompt += "}\n"
        # prompt += "```\n"
        # prompt += "\n"
        # prompt += "Example 3:\n"
        # prompt += "```\n"
        # prompt += "{\n"
        # prompt += "    \"theory_request_original\": \"Please investigate a theory for how probing language models works\",\n"
        # prompt += "    \"theory_request_normalized\": \"Build a theory of how probing works in large language models.\",\n"
        # prompt += "    \"short_name\": \"probing-large-language-models\",\n"
        # prompt += "    \"paper_search_query\": \"Extract any mentions of different types of probes for probing large language models (LLMs).\",\n"
        # prompt += "    \"extraction_query\": \"Extract any mentions of different types of probes for probing large language models (LLMs).\",\n"
        # prompt += "    \"extraction_schema\": [\n"
        # prompt += "        {\"name\": \"probe_name\", \"type\": \"str\", \"description\": \"The name of the probe that is being evaluated.\"},\n"
        # prompt += "        {\"name\": \"probe_description\", \"type\": \"str\", \"description\": \"A detailed description of the probe.\"},\n"
        # prompt += "        {\"name\": \"what_is_probed\", \"type\": \"str\", \"description\": \"What is the probe probing? (e.g. 'the model's ability to perform arithmetic', 'the model's ability to understand natural language', etc.)\"},\n"
        # prompt += "        {\"name\": \"results\", \"type\": \"str\", \"description\": \"What are the results of the probe? (e.g. 'the model's performance on arithmetic problems', 'the model's performance on natural language understanding tasks', etc.)\"}\n"
        # prompt += "    ],\n"
        # prompt += "    \"notes\": \"...\",\n"
        # prompt += "}\n"
        # prompt += "```\n"
        # prompt += "\n"
        # prompt += "Example 4:\n"
        # prompt += "```\n"
        # prompt += "{\n"
        # prompt += "    \"theory_request_original\": \"Could you create one or more theories of how theory-of-mind works in large language models (LLMs).\",\n"
        # prompt += "    \"theory_request_normalized\": \"Build a theory of how theory-of-mind works in large language models (LLMs), based on the following results.\",\n"
        # prompt += "    \"short_name\": \"theory-of-mind-llms\",\n"
        # prompt += "    \"paper_search_query\": \"Extract any mentions of how LLMs perform theory-of-mind tasks, and what the results are.\",\n"
        # prompt += "    \"extraction_query\": \"Extract any mentions of how LLMs perform theory-of-mind tasks, and what the results are.\",\n"
        # prompt += "    \"extraction_schema\": [\n"
        # prompt += "        {\"name\": \"model_name\", \"type\": \"str\", \"description\": \"The name of the LLM model that is being evaluated.\"},\n"
        # prompt += "        {\"name\": \"model_description\", \"type\": \"str\", \"description\": \"A brief description of the LLM model.\"},\n"
        # prompt += "        {\"name\": \"model_size\", \"type\": \"str\", \"description\": \"How large was the model, in parameters? (e.g. 1B, 2B, 7B, 13B, 70B, etc.)\"},\n"
        # prompt += "        {\"name\": \"task_name\", \"type\": \"str\", \"description\": \"The name of the theory-of-mind task that the model is being evaluated on.\"},\n"
        # prompt += "        {\"name\": \"task_description\", \"type\": \"str\", \"description\": \"A brief description of the theory-of-mind task that the model is being evaluated on.\"},\n"
        # prompt += "        {\"name\": \"performance\", \"type\": \"str\", \"description\": \"What was the performance of the model on the theory-of-mind task? Ideally this should be as detailed as possible, and broken down by task type. (numerical, include units. null if no information)\"}\n"
        # prompt += "    ],\n"
        # prompt += "    \"notes\": \"...\",\n"
        # prompt += "}\n"
        # prompt += "```\n"
        # prompt += "\n"

        if (reflection is not None):
            prompt += "\n"
            prompt += "# Reflection\n"
            prompt += "This is a reflection step. Previously, you generated a response (below).  Now, your task is to reflect on that response, and fix any errors, inconsistencies, omissions, or any other issues.\n"
            prompt += "```\n"
            prompt += reflection + "\n"
            prompt += "```\n"
            prompt += "\n"

        prompt += "# Output Format\n"
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output, as these are not valid JSON and will cause a parsing error.\n"
        prompt += "\n"

        prompt += "Your JSON response must be a dictionary with a single key, `suggestions`, that is a list of dictionaries, where each dictionary represents a suggestion for new schema slots, new variables, distilled/higher signal-to-noise slots/variables, or other suggestions that would help sharpen the likelihood of discovering novel, impactful patterns/theories/laws from the data.\n"
        prompt += "The suggestions should be in ranked order, so that the most promising suggestion for impactful theory building is listed first.\n"
        prompt += "Each suggestion dictionary must contain the following keys:\n"
        prompt += "- `suggestion_short_name`(str): A short name for the suggestion (a few words long)\n"
        prompt += "- `suggestion_description`(str): a detailed, information-dense description of the suggestion\n"
        prompt += "- `suggested_slots_and_extraction_instructions`(list of dict):[]: a list of dictionaries, where each dictionary contains a `slot_name`(str), and `slot_description_and_extraction_instructions`(str). `slot_name` should be a snake case name for this extraction slot, and `slot_description_and_extraction_instructions` should be a concise, detailed, information-dense description of what information to extract from the paper for this slot, and how to represent it. Should be no more than 1-2 sentences.\n"
        prompt += "- `potential_utility`(str): a detailed, information-dense description of the potential utility of the suggestion for theory building, and how it might help discover novel, impactful patterns/theories/laws from the data.\n"
        prompt += "- `details`(str): How would the extraction be different than it currently is?  Be specific, and potentially provide what instructions might look like for this new kind of extraction.\n"
        prompt += "- `example`(str): A clear, plain-language, toy exmaple that illustrates this suggestion, the before/after, and how it might be useful for theory building.\n"
        prompt += "- `potential_paired_slots`(list of str):[]: Are there specific other slots/variables that this is likely to be well-paired with to discover new patterns/laws/theories about? (e.g. cartoon example: pairing 'force' with 'mass' and 'acceleration')? Provide the slot names here as a list of strings, where each element is one slot name key, exactly as it appears in the extraction schema/records.\n"
        prompt += "- `potential_patterns`(str): Are there candidate patterns that you already observe between this and other variables, that might be particularly worth exploring, or explored in more detail?"
        prompt += "- `reextract_or_reformulate`(str): Some suggestsions require re-running the extraction process on the paper full-text with a modified extraction schema (`reextract`), while others can clearly use the exising information that's already been extracted and reformulate it into the new information needs (e.g. going from a specific chemical like `HCl` to a category, like `acid`). Please write either `reextract` or `reformulate` here, to choose the process to use). The name must be exact, and is case sensitive.\n"
        prompt += "\n"

        prompt += "## Special instructions\n"
        prompt += "- You can (and are encouraged to) think and plan before writing your JSON response.  But, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output.\n"
        prompt += "- Please follow the instructions carefully.\n"
        prompt += "- Do not hallucinate.\n"

        return prompt


    total_cost = 0.0
    responses = []
    responses_raw = []

    # Create the prompt
    #prompt = mkPrompt(theory_request)
    prompt = mkPrompt(theory_query_information, extracted_records, reflection=None)
    # Call the LLM
    responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=0.0, maxTokens=max(max_tokens, DEFAULT_MAX_TOKENS), jsonOut=False)
    responses.append(responseJSON)
    responses_raw.append(responseText)
    total_cost += cost

    # If we are using reflection, we will call the LLM again with the reflection step
    if (use_reflection):
        # Create the reflection prompt
        reflection_prompt = mkPrompt(theory_query_information, extracted_records, reflection=responseText)
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

    suggestions = None
    if (best_response is not None) and ("suggestions" in best_response):
        suggestions = best_response["suggestions"]

    # Pack output
    packed = {
        #"theory_request": theory_request,
        "theory_query_information": theory_query_information,
        "model": model_str,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "use_reflection": use_reflection,
        "output": best_response,
        "responses": responses,
        "responses_raw": responses_raw,
        "suggestions": suggestions,
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
def build_theory_from_results_single_theory_reflection3(query:str, results:list, mission_statement:str=None, model_str="gpt-4o-mini", max_tokens=12000, temperature=0.0, use_reflection:bool=False, include_query_in_generation:bool=True, additional_suggestions=None, max_laws_per_theory:int=3):
    version_str = "theorizer-literature-supported-accuracy-focused-12102025"
    def mkPrompt(query:str, results:str, reflection:str=None, include_query_in_generation:bool=True, mission_statement:str=None, additional_suggestions=None, max_laws_per_theory=None):
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"

        prompt += "# Task\n"
        prompt += "Your task is to construct a theory of how a scientific phenomenon works, based on a collection of evidence that you will be provided.\n"
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
            if (mission_statement is not None):
                prompt += "Here is a mission statement that provides more detail about the goals of the theories you should generate:\n"
                prompt += "```\n"
                prompt += str(mission_statement) + "\n"
                prompt += "```\n"
                prompt += "\n"

        if (additional_suggestions is not None) and (reflection is None): # Only include these suggestions in the initial prompt, not in the reflection prompt, since these are meant to be suggestions for the initial theory generation step, not for the reflection step
            prompt += "Here are some additional suggestions for things that you noticed in a pilot study, that may or may not be helpful:\n"
            prompt += "```\n"
            if (isinstance(additional_suggestions, dict) or isinstance(additional_suggestions, list)):
                prompt += json.dumps(additional_suggestions, indent=4) + "\n"
            else:
                prompt += str(additional_suggestions) + "\n"
            prompt += "```\n"
            prompt += "\n"

        if (max_laws_per_theory is not None):
            prompt += "# Maximum laws per theory\n"
            prompt += "When generating theories, please limit the maximum number of laws per theory to no more than the following number: " + str(max_laws_per_theory) + " .\n"
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
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output, as these are not valid JSON and will cause a parsing error.\n"
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
        prompt += "- You are encouraged to think and plan before writing your JSON response, but your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output.\n"
        prompt += "- Values that are 'none'/'null' in the JSON response should be represented as `null` in the JSON, not as an empty string, string saying \"null\", or any other value.\n"
        prompt += "- Where possible, you should prefer generating new, impactful theories.\n"
        if (max_laws_per_theory is not None):
            prompt += "- When generating theories, please limit the maximum number of laws per theory to no more than the following number: " + str(max_laws_per_theory) + " .\n"
        prompt += "- All your information must be accurate. Do not hallucinate.\n"

        return prompt

    try:
        total_cost = 0.0

        # Create the prompt
        prompt = mkPrompt(query, results, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory)

        # Call the LLM
        #responseJSON, responseText, cost = def getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True):
        all_responses = []
        MAX_GENERATION_TIME_SECS = generation_time_seconds(60 * 15)    # 15 minutes, unless MAX_GENERATION_TIME_SECONDS or the caller says otherwise
        print("## Calling LLM to generate initial theories...", flush=True)
        responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=MAX_GENERATION_TIME_SECS)
        print("## Response Received", flush=True)
        all_responses.append(responseJSON)
        total_cost += cost

        printVerbose("## Initial response:")
        printVerbose(responseText)

        # New one-theory-at-a-time reflection approach
        reflected_response = None
        if (use_reflection):
            print("### Using reflection...")
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
                    reflection_prompt = mkPrompt(query, results, reflection=theory_str, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory)

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

        print("## PACKING OUTPUT ")

        # Pack output
        packed = {
            "query": query,
            "model": model_str,
            "use_reflection": use_reflection,
            "mission_statement": mission_statement,
            "theory_response": best_response,
            "results": results,
            "raw_responses": all_responses,
            "total_cost": total_cost,
        }

        return packed

    except Exception as e:
        import traceback
        print("ERROR in build_theory_from_results_single_theory_reflection(): " + str(e))
        traceback.print_exc()
        return None

    except BaseException as e:
        print("FATAL BaseException escaped:", repr(e), flush=True)
        traceback.print_exc()
        raise


# Theory Generation: `Accuracy-Focused` Objective
# Build a set of theories from a list of results, with reflection.
# Performs the reflection one theory at a time, to encourage it to fix any issues with the single theory, more fully populate the evidence, etc.
# This one explicitly focuses on trying to abstract new qualitative and quantitative laws from the supporting evidence.
# The main change here compared to the last version is that:
#  - It's generally intended to generate only one (or a small number of) theory laws/statements per theory
#  - The 'conflicting' and 'unaccounted for' evidence is moved into the law, rather than being theory-level.
def build_theory_from_results_single_theory_reflection4(query:str, results:list, mission_statement:str=None, model_str="gpt-4o-mini", max_tokens=12000, temperature=0.0, use_reflection:bool=False, include_query_in_generation:bool=True, additional_suggestions=None, max_laws_per_theory:int=3):
    version_str = "theorizer-literature-supported-accuracy-focused-12102025"
    def mkPrompt(query:str, results:str, reflection:str=None, include_query_in_generation:bool=True, mission_statement:str=None, additional_suggestions=None, max_laws_per_theory=None):
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"

        prompt += "# Task\n"
        prompt += "Your task is to construct a theory of how a scientific phenomenon works, based on a collection of evidence that you will be provided.\n"
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
        prompt += '    "theory_statement_plain_language": "...", # A 1-2 sentence plain language version of the law/statement, that is clear, concise, and easily readable.\n'
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

        #prompt += "- `unaccounted_for`(list): A list of specific pieces of evidence that are not explained by the theory, if any.\n"
        #prompt += "- `conflicting_evidence`(list): A list of specific pieces of evidence that appear to conflict with the theory, if any.\n"
        # Conflicting Evidence
        prompt += "    \"conflicting_evidence\": [ # Conflicting evidence appears to conflict with the law/statement\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # One piece of conflicting evidence\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # Second piece of conflicting evidence\n"
        prompt += "        # ... more pieces of conflicting evidence as available\n"
        prompt += "    ],\n"
        # Unaccounted For Evidence
        prompt += "    \"unaccounted_for\": [ # Unaccounted for evidence is not explained by the law/statement\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # One piece of unaccounted-for evidence\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # Second piece of unaccounted-for evidence\n"
        prompt += "        # ... more pieces of unaccounted-for evidence as available\n"
        prompt += "    ],\n"
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
            if (mission_statement is not None):
                prompt += "Here is a mission statement that provides more detail about the goals of the theories you should generate:\n"
                prompt += "```\n"
                prompt += str(mission_statement) + "\n"
                prompt += "```\n"
                prompt += "\n"

        if (additional_suggestions is not None) and (reflection is None): # Only include these suggestions in the initial prompt, not in the reflection prompt, since these are meant to be suggestions for the initial theory generation step, not for the reflection step
            prompt += "Here are some additional suggestions for things that you noticed in a pilot study, that may or may not be helpful:\n"
            prompt += "```\n"
            if (isinstance(additional_suggestions, dict) or isinstance(additional_suggestions, list)):
                prompt += json.dumps(additional_suggestions, indent=4) + "\n"
            else:
                prompt += str(additional_suggestions) + "\n"
            prompt += "```\n"
            prompt += "\n"

        if (max_laws_per_theory is not None):
            prompt += "# Maximum laws per theory\n"
            prompt += "When generating theories, please limit the maximum number of laws per theory to no more than the following number: " + str(max_laws_per_theory) + " .\n"
            prompt += "\n"

        prompt += "Here is a list of specific results that you should use to build your theories:\n"
        # If this isn't set, just give it the raw data and ask it to brainstorm in a data-driven way rather than in a query-driven way
        prompt += "```\n"
        prompt += json.dumps(results, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"

        # Everything up to this point is identical for the initial generation call and for every
        # reflection call in this run -- the reflected theory and the output format come after.  That
        # makes it the cacheable prefix, and it is nearly the whole prompt when the evidence is large.
        cachePrefix = prompt

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
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output, as these are not valid JSON and will cause a parsing error.\n"
        prompt += "\n"
        prompt += "Your JSON response must be a dictionary, that contains two keys: `theories_general`, and `theories_specific`.  Each is a list of dictionaries, with the following keys:\n"
        prompt += "- `theory_name`(str): A short name for the theory\n"
        prompt += "- `theory_name_plain_language`(str): A short name for the theory expressed in plain language\n"
        prompt += "- `theory_description`(str): A full description of the theory\n"
        prompt += "- `theory_description_plain_language`(str): A full description of the theory expressed in plain language\n"
        #prompt += "- `supporting_evidence`(list): A list of specific supporting evidence for the theory.\n"
        #prompt += "- `theory_statements`(list): A list of specific predictive statements (either qualitative, quantitative, or both) that describe the assertions of the theory.\n"
        prompt += "- `theory_statements`(list): A list of specific laws (either qualitative, quantitative, or both) that describe the assertions of the theory.\n"
        prompt += "- `new_predictions_likely`(list): A list of several example (testable) predictions that the theory makes (the situation, and the expected results) that are not in the observed evidence. For this set of new predictions, the predictions should be fairly likely to work out.\n"
        prompt += "- `new_predictions_unknown`(list): A list of several example (testable) predictions that the theory makes (the situation, and the expected results) that are not in the observed evidence. For this set of new predictions, the predictions should be difficult to tell if they would genuinely work out or not, and have particularly impactful results if they do (or don't) work out.\n"
        prompt += "- `negative_experiments`(list): A list of several (testable) predictions that test the assumptions or statements of the theory, and if false, would call the theory statements into question.\n"
        #prompt += "- `unaccounted_for`(list): A list of specific pieces of evidence that are not explained by the theory, if any.\n"
        #prompt += "- `conflicting_evidence`(list): A list of specific pieces of evidence that appear to conflict with the theory, if any.\n"
        prompt += "- `special_cases`(list): A list of any special cases, exceptions, or boundary conditions that apply to the theory, if any.\n"
        prompt += "- `existing_theory`(dict): A dictionary that describes whether (to the best of your knowledge, and the information available in the papers) if this is a pre-existing theory, or a new theory. It has the following keys: `likely_classification`:str, which is one of `new`, `somewhat-related-to-existing`, `closely-related-to-existing`, or `existing`, and `references`:list(str), which are lists of references (one per string) to new (or related) theories in the form `First author (year) title [brief note of how related, and name(s) of relevant theories, in square brackets]`. These should help confirm the theory is new, or help confirm it is existing or closely related. It also has: `what_already_exists`(str), `what_is_novel`(str), and `classification_explanation`(str).\n"
        prompt += "\n"
        prompt += "While most fields are lists of strings, for `supporting_evidence` and `unaccounted_for`, the list should be a list of dictionaries, where each dictionary has the following keys:\n"
        prompt += "- `text`(str): A textual description of the evidence\n"
        prompt += "- `uuids`(list): A list of UUIDs that correspond to the specific pieces of evidence in the original results that support this text description. These lists can be as long or short as they need to be to encompass all the correct evidence references.\n"
        prompt += "\n"
        prompt += "Similarly, the `theory_statements` field is a list of dictionaries, where each dictionary has the following key:\n"
        #prompt += "- `law`(dict): A dictionary that describes a single qualitative or quantitative law, in the format described above (i.e. with `if` and `then` fields, each containing a list of triples), and specific supporting evidence that supports the development of this law.\n"
        prompt += "- `law`(dict): A dictionary that describes a single qualitative or quantitative law, in the format described above, and specific supporting evidence that supports the development of this law.\n"
        prompt += "\n"
        prompt += "For example (a cartoon example -- your theories should be much more detailed, and have much more supporting evidence, linked back with proper UUIDs; similarly, the unaccounted-for and conflicting evidence here is just filler, yours should be detailed and correct):\n"
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
        prompt += '                  "theory_statement_plain_language": "...",\n'
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
        # Conflicting
        prompt += '                 "conflicting_evidence": [\n'
        prompt += '                      { "text": "The observed accelerated expansion of the universe appears to conflict with the mutual attraction of massive objects.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        # Unaccounted for
        prompt += '                 "unaccounted_for": [\n'
        prompt += '                      { "text": "The phenomenon of dark matter is not explained by the mutual attraction of massive objects.", "uuids": [...] },\n'
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
        prompt += '                  "theory_statement_plain_language": "...",\n'
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
        # Conflicting
        prompt += '                 "conflicting_evidence": [\n'
        prompt += '                      { "text": "The observed accelerated expansion of the universe appears to conflict with the inverse square law of gravitational force.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        # Unaccounted for
        prompt += '                 "unaccounted_for": [\n'
        prompt += '                      { "text": "The phenomenon of dark energy is not explained by the inverse square law of gravitational force.", "uuids": [...] },\n'
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
        prompt += '                  "theory_statement_plain_language": "...",\n'
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
        # Conflicting
        prompt += '                 "conflicting_evidence": [\n'
        prompt += '                      { "text": "The observed accelerated expansion of the universe appears to conflict with the proportional relationship between mass and gravitational force.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        # Unaccounted for
        prompt += '                 "unaccounted_for": [\n'
        prompt += '                      { "text": "The phenomenon of dark energy is not explained by the proportional relationship between mass and gravitational force.", "uuids": [...] },\n'
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
        # prompt += '            "unaccounted_for": [\n'
        # prompt += '                {\"text\": "The exact mechanism of how gravity works at the quantum level is not fully understood.", \"uuids\": [7, 25]},\n'
        # prompt += '                {\"text\": "The behavior of gravity in extreme conditions, such as near black holes, is not fully understood.", \"uuids\": [12, 28]}\n'
        # prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        # prompt += '            ],\n'
        # prompt += '            "conflicting_evidence": [\n'
        # prompt += '                {\"text\": "Observations of galaxy rotation curves that do not match predictions based on visible matter challenge the theory of gravity as currently understood.", \"uuids\": [30, 32]},\n'
        # prompt += '                {\"text\": "The accelerated expansion of the universe challenges the traditional understanding of gravity.", \"uuids\": [35, 40]}\n'
        # prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        # prompt += '            ],\n'
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

        # Added this style note to help produce more readable theories.
        prompt += "# Style Notes\n"
        prompt += "- Theories should be as technical as required, but not more technical than is required (and certainly not over-simplified, such that information is lost; or over-technical/jargony, such that the important elements are obscured).\n"
        prompt += "- When possible, theories should use simpler language (e.g. that is accessible to a 1st year interdisciplinary masters student), but not at the expense of losing information or accuracy. If you need to use a highly technical term because it's important, use it. Novelty and impact are important. But readability is also important.\n"
        prompt += "- When possible, please avoid generating new terminology or jargon unless it is necessary to accurately describe a new concept or phenomenon. If you already know the correct term for a concept in a field, use it, rather than inventing a new one.  If a new one is needed, then you can create a new term, but try not to make it too difficult to understand.\n"
        prompt += "- A general comment has been that the theories are often very verbose, jargon-heavy, and sometimes hard to read by human experts who are most in a position to benefit from their knowledge, and make impactful discoveries/advances. Please keep this in mind, and aim for clarity, conciseness, and readability, while still being novel and impactful.\n"
        prompt += "\n"

        prompt += "# Important Notes\n"
        prompt += "- You must return a JSON response, and it must be valid JSON, or it will not be parsed.\n"
        prompt += "- You are encouraged to think and plan before writing your JSON response, but your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output.\n"
        prompt += "- Values that are 'none'/'null' in the JSON response should be represented as `null` in the JSON, not as an empty string, string saying \"null\", or any other value.\n"
        prompt += "- Where possible, you should prefer generating new, impactful theories.\n"
        if (max_laws_per_theory is not None):
            prompt += "- When generating theories, please limit the maximum number of laws per theory to no more than the following number: " + str(max_laws_per_theory) + " .\n"
        prompt += "- All your information must be accurate. Do not hallucinate.\n"

        return prompt, cachePrefix

    try:
        total_cost = 0.0

        # Prompt caching is only worth it when more than one call shares the prefix, which here means
        # only when reflection is on: the generation call writes the evidence to the cache and each
        # reflection call reads it back.  With reflection off there is a single call, and a cache
        # write costs more than a plain one, so leave it off.  The key is per-run, and routes this
        # run's calls to the same machine so the reads can hit.
        cacheKey = None
        if (use_reflection):
            cacheKey = "theorizer-reflection4-" + str(os.getpid()) + "-" + str(threading.get_ident()) + "-" + str(int(time.time() * 1000))

        # Create the prompt
        prompt, cachePrefix = mkPrompt(query, results, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory)

        # Call the LLM
        #responseJSON, responseText, cost = def getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True):
        all_responses = []
        MAX_GENERATION_TIME_SECS = generation_time_seconds(60 * 15)    # 15 minutes, unless MAX_GENERATION_TIME_SECONDS or the caller says otherwise
        print("## Calling LLM to generate initial theories...", flush=True)
        responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=MAX_GENERATION_TIME_SECS, cachePrefixStr=(cachePrefix if (use_reflection) else None), cacheKey=cacheKey, retry_at_lower_effort_if_empty=True, repairJSON=True)
        print("## Response Received", flush=True)
        all_responses.append(responseJSON)
        total_cost += cost

        printVerbose("## Initial response:")
        printVerbose(responseText)

        # New one-theory-at-a-time reflection approach
        reflected_response = None
        if (use_reflection):
            print("### Using reflection...")
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
                    reflection_prompt, reflection_cache_prefix = mkPrompt(query, results, reflection=theory_str, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory)

                    # Submit the LLM call to the executor
                    # Submit the LLM call to the executor.  These read the prefix that the generation
                    # call above wrote to the cache -- same prefix, same key.
                    future = executor.submit(getLLMResponseJSON, reflection_prompt, model_str, temperature, max_tokens, False,
                                             cachePrefixStr=reflection_cache_prefix, cacheKey=cacheKey,
                                             retry_at_lower_effort_if_empty=True, repairJSON=True)
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

        print("## PACKING OUTPUT ")

        # Pack output
        packed = {
            "query": query,
            "model": model_str,
            "use_reflection": use_reflection,
            "mission_statement": mission_statement,
            "theory_response": best_response,
            "results": results,
            "raw_responses": all_responses,
            "total_cost": total_cost,
        }

        return packed

    except Exception as e:
        import traceback
        print("ERROR in build_theory_from_results_single_theory_reflection(): " + str(e))
        traceback.print_exc()
        return None

    except BaseException as e:
        print("FATAL BaseException escaped:", repr(e), flush=True)
        traceback.print_exc()
        raise


# A version of the accuracy-focused, literature-supportd, that includes:
# - A list of example inference and output types
# - The ability to focus on those inference types.
def build_theory_from_results_single_theory_reflection5_inferencetypes(query:str, results:list, mission_statement:str=None, model_str="gpt-4o-mini", max_tokens=12000, temperature=0.0, use_reflection:bool=True, include_query_in_generation:bool=True, additional_suggestions=None, max_laws_per_theory:int=3, inference_type:str=None, output_type:str=None):
    version_str = "theorizer-literature-supported-accuracy-focused-reflection5-inferencetypes-08182026"
    def mkPrompt(query:str, results:str, reflection:str=None, include_query_in_generation:bool=True, mission_statement:str=None, additional_suggestions=None, max_laws_per_theory=None, inference_type:str=None, output_type:str=None):
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"

        prompt += "# Task\n"
        prompt += "Your task is to construct a theory of how a scientific phenomenon works, based on a collection of evidence that you will be provided.\n"
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
        prompt += '    "theory_statement_plain_language": "...", # A 1-2 sentence plain language version of the law/statement, that is clear, concise, and easily readable.\n'
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

        #prompt += "- `unaccounted_for`(list): A list of specific pieces of evidence that are not explained by the theory, if any.\n"
        #prompt += "- `conflicting_evidence`(list): A list of specific pieces of evidence that appear to conflict with the theory, if any.\n"
        # Conflicting Evidence
        prompt += "    \"conflicting_evidence\": [ # Conflicting evidence appears to conflict with the law/statement\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # One piece of conflicting evidence\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # Second piece of conflicting evidence\n"
        prompt += "        # ... more pieces of conflicting evidence as available\n"
        prompt += "    ],\n"
        # Unaccounted For Evidence
        prompt += "    \"unaccounted_for\": [ # Unaccounted for evidence is not explained by the law/statement\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # One piece of unaccounted-for evidence\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # Second piece of unaccounted-for evidence\n"
        prompt += "        # ... more pieces of unaccounted-for evidence as available\n"
        prompt += "    ],\n"
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
        # Knowledge/Inference Types
        prompt += '    "output_types": ["..."], # One or more knowledge/output types reflected in this law. Can specify additional type(s) if those provided on the knowledge/inference type list do not provide sufficient coverage.\n'
        prompt += '    "inference_types": ["..."], # One or more inference types reflected in this law. Can specify additional type(s) if those provided on the knowledge/inference type list do not provide sufficient coverage.\n'
        prompt += '    "inference_type_description": {"...": "..."} # For each `inference_type` in `inference_types`, provide a brief, information-dense, 1-2 sentence description of exactly how the supporting evidence was transformed into the law statement, using the inference type specified. Be very specific. Key=inference type, value=explanation. Use multiple keys in cases where there is more than one inference type.\n'
        prompt += '}\n'
        prompt += "```\n"
        #prompt += "The purpose of framing the statements/laws in this way is to make it easy to evaluate and compare across theories. The subjects, relations, and objects should be concise, but also information rich and detailed.\n"
        #prompt += "Each triple represents a single assertion about the relationship between a subject and an object, via a relation.\n"
        #prompt += "Multiple triples in the `if` section are combined with AND logic (i.e. all conditions must be met), and multiple triples in the `then` section are also combined with AND logic (i.e. all outcomes occur if the conditions are met).\n"
        prompt += "\n"


        # New section on inference types and output types
        prompt += "# Generalization Inference Types and Output Types (i.e. partial knowledge/inference type list)\n"
        prompt += "Generalizations can take on a large variety of forms. A small set of possible inference types and output types are provided below, which may be useful in framing your theories.  You are not limited to these, but they may be useful in guiding your thinking.\n"
        prompt += "Please note that a given theory/law may contain one, more than one, or none of these types (i.e. different types).\n"
        prompt += "\n"
        prompt += "## Output Types: What the generalization claims\n"
        prompt += "Here are a small set of common examples of what generalizations tend to claim.  Please note that the examples after each category type are toy/cartoon examples, and are not intended to be illustrative or restrictive:\n"
        prompt += "O1. MECHANISTIC EXPLANATION: `Y occurs because of mechanism M`. Example: The seasonal die-off happens because warm surface water blocks oxygen from below.\n"
        prompt += "O2. COMPARATIVE RANKING: `X > Y`. Example: Approach B achieves lower error than approach A.\n"
        prompt += "O3. STATE OF THE LITERATURE: `Of N studies, K address Y and none address Z`. Example: Of 70 studies, 60 were in three countries and none addressed rural settings.\n"
        prompt += "O4. CAPABILITY CLAIM: `X can do Y`. Example: Compound X can be used to treat disease Y.\n"
        prompt += "O5. CONTEXT DEPENDENCE: `Y holds in context C1 but not in C2`. Example: The component helps at high-resource sites, but not a low-resource sites.\n"
        prompt += "O6. ASSOCIATION: `X is associated with Y`. Example: Sites with higher rainfall have higher crop yields.\n"
        prompt += "O7. EFFECT OF AN INTERVENTION: `Applying X changes Y`. Example: A four-week training program reduces average error rates.\n"
        prompt += "O8. RECOMMENDATION: `Researchers should do X`. Example: Future studies should report this outcome with a standardized instrument.\n"
        prompt += "O9. DEFINITION OR TAXONOMY: `X consists of A, B, and C`. Example: Reading comprehension comprises decoding, vocabulary, and inference.\n"
        prompt += "O10. APPRAISAL OF EVIDENCE QUALITY: `The evidence for Y is inadequate`. Example: The studies are too small and heterogeneous to be compared or combined.\n"
        prompt += "O11. TEMPORAL TREND: `Y increased from T1 to T2`. Example: The number of studies using the technique grew between 2010 and 2022.\n"
        prompt += "O12. NULL RESULT: `X does not affect Y`. Example: The two groups did not differ, and the study was adequately powered.\n"
        prompt += "\n"



        prompt += "## Inference / Reasoning Types: How the generalization was reasoned frm evidence to claim:\n"
        prompt += "Here are a small set of common examples of how generalizations tend to be reasoned from evidence to claim.  Please note that the examples after each category type are toy/cartoon examples, and are not intended to be illustrative or restrictive:\n"
        prompt += "Accumulation:\n"
        prompt += "I1. ENUMERATIVE INDUCTION: Several results agree, so a broader claim is made.\n"
        prompt += "I2. TRIANGULATION: Different kinds of evidence point the same way.\n"
        prompt += "I3. META-ANALYTIC POOLING: Separate estimates are combined into one number.\n"
        prompt += "Contrast:\n"
        prompt += "I4. CONTRAST ARGUMENT (HETEROGENEITY): Results that happen to disagree are themselves the point.\n"
        prompt += "I5. CONTROLLED COMPARISON OR ABLATION: One factor is deliberatively changed, the rest are held fixed.\n"
        prompt += "Enumeration:\n"
        prompt += "I6. TABULATION OR FREQUENCY COUNTING: A fixed collection sorted into categories and counted.\n"
        prompt += "Computation:\n"
        prompt += "I7. STATISTICAL INFERENCE: A test or model applied to one assembled dataset.\n"
        prompt += "Inference:\n"
        prompt += "I8. ABDUCTION (BEST EXPLANATION): The unseen cause that best accounts for what was seen.\n"
        prompt += "I9. DEDUCTION FROM DEFINITIONS OR RULES: Follows from a definition or a stated requirement.\n"
        prompt += "I10. PROOF OF CONCEPT: Something was built, and it worked, so it is possible.\n"
        prompt += "I11. INDUCTION FROM SINGLE CASE: One source is extended to a much broader target.\n"
        prompt += "I12. ARGUMENT FROM ANALOGY: Something is true of a similar case, so it is carried across to this one.\n"
        prompt += "Appraisal:\n"
        prompt += "I13. PRACTICAL (MEANS-END) REASONING: Given the needs, this is what ought to be done.\n"
        prompt += "I14. CRITICAL APPRAISAL OF THE EVIDENCE: The evidence itself is judged, usually as lacking\n"
        prompt += "\n"

        # TODO: SPECIFIC TYPES
        if (inference_type is not None) or (output_type is not None):
            prompt += "# Inference/Output Types to Focus On\n"
            prompt += "You should focus on the following inference and/or output types when generating your theories:\n"
            if (inference_type is not None):
                prompt += "- Inference Type: " + str(inference_type) + "\n"
            if (output_type is not None):
                prompt += "- Output Type: " + str(output_type) + "\n"
            prompt += "Please keep in mind that the generalizations in theories/laws may contain one, or more than one inference type and/or output type -- so it may make sense to mix the requested type(s) with other types, as needed/appropriate, based on the data available.\n"
            prompt += "\n"


        prompt += "# Query\n"
        prompt += "Here is the general area/query that your theories should be centered around:"
        if (include_query_in_generation):
            prompt += "```\n"
            prompt += query + "\n"
            prompt += "```\n"
            prompt += "\n"
            if (mission_statement is not None):
                prompt += "Here is a mission statement that provides more detail about the goals of the theories you should generate:\n"
                prompt += "```\n"
                prompt += str(mission_statement) + "\n"
                prompt += "```\n"
                prompt += "\n"

        if (additional_suggestions is not None) and (reflection is None): # Only include these suggestions in the initial prompt, not in the reflection prompt, since these are meant to be suggestions for the initial theory generation step, not for the reflection step
            prompt += "Here are some additional suggestions for things that you noticed in a pilot study, that may or may not be helpful:\n"
            prompt += "```\n"
            if (isinstance(additional_suggestions, dict) or isinstance(additional_suggestions, list)):
                prompt += json.dumps(additional_suggestions, indent=4) + "\n"
            else:
                prompt += str(additional_suggestions) + "\n"
            prompt += "```\n"
            prompt += "\n"

        if (max_laws_per_theory is not None):
            prompt += "# Maximum laws per theory\n"
            prompt += "When generating theories, please limit the maximum number of laws per theory to no more than the following number: " + str(max_laws_per_theory) + " .\n"
            prompt += "\n"

        prompt += "Here is a list of specific results that you should use to build your theories:\n"
        # If this isn't set, just give it the raw data and ask it to brainstorm in a data-driven way rather than in a query-driven way
        prompt += "```\n"
        prompt += json.dumps(results, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"

        # Everything up to this point is identical for the initial generation call and for every
        # reflection call in this run -- the reflected theory and the output format come after.  That
        # makes it the cacheable prefix, and it is nearly the whole prompt when the evidence is large.
        cachePrefix = prompt

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
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output, as these are not valid JSON and will cause a parsing error.\n"
        prompt += "\n"
        prompt += "Your JSON response must be a dictionary, that contains two keys: `theories_general`, and `theories_specific`.  Each is a list of dictionaries, with the following keys:\n"
        prompt += "- `theory_name`(str): A short name for the theory\n"
        prompt += "- `theory_name_plain_language`(str): A short name for the theory expressed in plain language\n"
        prompt += "- `theory_description`(str): A full description of the theory\n"
        prompt += "- `theory_description_plain_language`(str): A full description of the theory expressed in plain language\n"
        #prompt += "- `supporting_evidence`(list): A list of specific supporting evidence for the theory.\n"
        #prompt += "- `theory_statements`(list): A list of specific predictive statements (either qualitative, quantitative, or both) that describe the assertions of the theory.\n"
        prompt += "- `theory_statements`(list): A list of specific laws (either qualitative, quantitative, or both) that describe the assertions of the theory.\n"
        prompt += "- `new_predictions_likely`(list): A list of several example (testable) predictions that the theory makes (the situation, and the expected results) that are not in the observed evidence. For this set of new predictions, the predictions should be fairly likely to work out.\n"
        prompt += "- `new_predictions_unknown`(list): A list of several example (testable) predictions that the theory makes (the situation, and the expected results) that are not in the observed evidence. For this set of new predictions, the predictions should be difficult to tell if they would genuinely work out or not, and have particularly impactful results if they do (or don't) work out.\n"
        prompt += "- `negative_experiments`(list): A list of several (testable) predictions that test the assumptions or statements of the theory, and if false, would call the theory statements into question.\n"
        #prompt += "- `unaccounted_for`(list): A list of specific pieces of evidence that are not explained by the theory, if any.\n"
        #prompt += "- `conflicting_evidence`(list): A list of specific pieces of evidence that appear to conflict with the theory, if any.\n"
        prompt += "- `special_cases`(list): A list of any special cases, exceptions, or boundary conditions that apply to the theory, if any.\n"
        prompt += "- `existing_theory`(dict): A dictionary that describes whether (to the best of your knowledge, and the information available in the papers) if this is a pre-existing theory, or a new theory. It has the following keys: `likely_classification`:str, which is one of `new`, `somewhat-related-to-existing`, `closely-related-to-existing`, or `existing`, and `references`:list(str), which are lists of references (one per string) to new (or related) theories in the form `First author (year) title [brief note of how related, and name(s) of relevant theories, in square brackets]`. These should help confirm the theory is new, or help confirm it is existing or closely related. It also has: `what_already_exists`(str), `what_is_novel`(str), and `classification_explanation`(str).\n"
        prompt += "\n"
        prompt += "While most fields are lists of strings, for `supporting_evidence` and `unaccounted_for`, the list should be a list of dictionaries, where each dictionary has the following keys:\n"
        prompt += "- `text`(str): A textual description of the evidence\n"
        prompt += "- `uuids`(list): A list of UUIDs that correspond to the specific pieces of evidence in the original results that support this text description. These lists can be as long or short as they need to be to encompass all the correct evidence references.\n"
        prompt += "\n"
        prompt += "Similarly, the `theory_statements` field is a list of dictionaries, where each dictionary has the following key:\n"
        #prompt += "- `law`(dict): A dictionary that describes a single qualitative or quantitative law, in the format described above (i.e. with `if` and `then` fields, each containing a list of triples), and specific supporting evidence that supports the development of this law.\n"
        prompt += "- `law`(dict): A dictionary that describes a single qualitative or quantitative law, in the format described above, and specific supporting evidence that supports the development of this law.\n"
        prompt += "\n"
        prompt += "For example (a cartoon example -- your theories should be much more detailed, and have much more supporting evidence, linked back with proper UUIDs; similarly, the unaccounted-for and conflicting evidence here is just filler, yours should be detailed and correct):\n"
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
        prompt += '                  "theory_statement_plain_language": "...",\n'
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
        # Conflicting
        prompt += '                 "conflicting_evidence": [\n'
        prompt += '                      { "text": "The observed accelerated expansion of the universe appears to conflict with the mutual attraction of massive objects.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        # Unaccounted for
        prompt += '                 "unaccounted_for": [\n'
        prompt += '                      { "text": "The phenomenon of dark matter is not explained by the mutual attraction of massive objects.", "uuids": [...] },\n'
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
        prompt += '                  },\n'
        prompt += '                  "output_types": ["..."], # One or more knowledge/output types reflected in this law. Can specify additional type(s) if those provided on the knowledge/inference type list do not provide sufficient coverage.\n'
        prompt += '                  "inference_types": ["..."], # One or more inference types reflected in this law. Can specify additional type(s) if those provided on the knowledge/inference type list do not provide sufficient coverage.\n'
        prompt += '                  "inference_type_description": {"...": "..."} # For each `inference_type` in `inference_types`, provide a brief, information-dense, 1-2 sentence description of exactly how the supporting evidence was transformed into the law statement, using the inference type specified. Be very specific. Key=inference type, value=explanation. Use multiple keys in cases where there is more than one inference type.\n'
        prompt += '                },\n'
        # Next statement/law
        prompt += '                {\n'
        prompt += '                  "statement_name": "Inverse Square Law of Gravitational Force",\n'
        prompt += '                  "theory_statement": "The gravitational force between two masses is inversely proportional to the square of the distance between them.",\n'
        prompt += '                  "theory_statement_plain_language": "...",\n'
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
        # Conflicting
        prompt += '                 "conflicting_evidence": [\n'
        prompt += '                      { "text": "The observed accelerated expansion of the universe appears to conflict with the inverse square law of gravitational force.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        # Unaccounted for
        prompt += '                 "unaccounted_for": [\n'
        prompt += '                      { "text": "The phenomenon of dark energy is not explained by the inverse square law of gravitational force.", "uuids": [...] },\n'
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
        prompt += '                  },\n'
        prompt += '                  "output_types": ["..."], # One or more knowledge/output types reflected in this law. Can specify additional type(s) if those provided on the knowledge/inference type list do not provide sufficient coverage.\n'
        prompt += '                  "inference_types": ["..."], # One or more inference types reflected in this law. Can specify additional type(s) if those provided on the knowledge/inference type list do not provide sufficient coverage.\n'
        prompt += '                  "inference_type_description": {"...": "..."} # For each `inference_type` in `inference_types`, provide a brief, information-dense, 1-2 sentence description of exactly how the supporting evidence was transformed into the law statement, using the inference type specified. Be very specific. Key=inference type, value=explanation. Use multiple keys in cases where there is more than one inference type.\n'
        prompt += '                },\n'
        # Next statement/law
        # This time, do a quantitative law
        prompt += '                {\n'
        prompt += '                  "statement_name": "Gravitational Force Proportional to Product of Masses",\n'
        prompt += '                  "theory_statement": "The gravitational force between two objects is directly proportional to the product of their masses.",\n'
        prompt += '                  "theory_statement_plain_language": "...",\n'
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
        # Conflicting
        prompt += '                 "conflicting_evidence": [\n'
        prompt += '                      { "text": "The observed accelerated expansion of the universe appears to conflict with the proportional relationship between mass and gravitational force.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        # Unaccounted for
        prompt += '                 "unaccounted_for": [\n'
        prompt += '                      { "text": "The phenomenon of dark energy is not explained by the proportional relationship between mass and gravitational force.", "uuids": [...] },\n'
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
        prompt += '                  },\n'
        prompt += '                  "output_types": ["..."], # One or more knowledge/output types reflected in this law. Can specify additional type(s) if those provided on the knowledge/inference type list do not provide sufficient coverage.\n'
        prompt += '                  "inference_types": ["..."], # One or more inference types reflected in this law. Can specify additional type(s) if those provided on the knowledge/inference type list do not provide sufficient coverage.\n'
        prompt += '                  "inference_type_description": {"...": "..."} # For each `inference_type` in `inference_types`, provide a brief, information-dense, 1-2 sentence description of exactly how the supporting evidence was transformed into the law statement, using the inference type specified. Be very specific. Key=inference type, value=explanation. Use multiple keys in cases where there is more than one inference type.\n'
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
        # prompt += '            "unaccounted_for": [\n'
        # prompt += '                {\"text\": "The exact mechanism of how gravity works at the quantum level is not fully understood.", \"uuids\": [7, 25]},\n'
        # prompt += '                {\"text\": "The behavior of gravity in extreme conditions, such as near black holes, is not fully understood.", \"uuids\": [12, 28]}\n'
        # prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        # prompt += '            ],\n'
        # prompt += '            "conflicting_evidence": [\n'
        # prompt += '                {\"text\": "Observations of galaxy rotation curves that do not match predictions based on visible matter challenge the theory of gravity as currently understood.", \"uuids\": [30, 32]},\n'
        # prompt += '                {\"text\": "The accelerated expansion of the universe challenges the traditional understanding of gravity.", \"uuids\": [35, 40]}\n'
        # prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        # prompt += '            ],\n'
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

        # Added this style note to help produce more readable theories.
        prompt += "# Style Notes\n"
        prompt += "- Theories should be as technical as required, but not more technical than is required (and certainly not over-simplified, such that information is lost; or over-technical/jargony, such that the important elements are obscured).\n"
        prompt += "- When possible, theories should use simpler language (e.g. that is accessible to a 1st year interdisciplinary masters student), but not at the expense of losing information or accuracy. If you need to use a highly technical term because it's important, use it. Novelty and impact are important. But readability is also important.\n"
        prompt += "- When possible, please avoid generating new terminology or jargon unless it is necessary to accurately describe a new concept or phenomenon. If you already know the correct term for a concept in a field, use it, rather than inventing a new one.  If a new one is needed, then you can create a new term, but try not to make it too difficult to understand.\n"
        prompt += "- A general comment has been that the theories are often very verbose, jargon-heavy, and sometimes hard to read by human experts who are most in a position to benefit from their knowledge, and make impactful discoveries/advances. Please keep this in mind, and aim for clarity, conciseness, and readability, while still being novel and impactful.\n"
        prompt += "\n"

        prompt += "# Important Notes\n"
        prompt += "- You must return a JSON response, and it must be valid JSON, or it will not be parsed.\n"
        prompt += "- You are encouraged to think and plan before writing your JSON response, but your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output.\n"
        prompt += "- Values that are 'none'/'null' in the JSON response should be represented as `null` in the JSON, not as an empty string, string saying \"null\", or any other value.\n"
        prompt += "- Where possible, you should prefer generating new, impactful theories.\n"
        if (max_laws_per_theory is not None):
            prompt += "- When generating theories, please limit the maximum number of laws per theory to no more than the following number: " + str(max_laws_per_theory) + " .\n"
        prompt += "- All your information must be accurate. Do not hallucinate.\n"

        return prompt, cachePrefix

    try:
        total_cost = 0.0

        # Prompt caching is only worth it when more than one call shares the prefix, which here means
        # only when reflection is on: the generation call writes the evidence to the cache and each
        # reflection call reads it back.  With reflection off there is a single call, and a cache
        # write costs more than a plain one, so leave it off.  The key is per-run, and routes this
        # run's calls to the same machine so the reads can hit.
        cacheKey = None
        if (use_reflection):
            cacheKey = "theorizer-reflection5-" + str(os.getpid()) + "-" + str(threading.get_ident()) + "-" + str(int(time.time() * 1000))

        # Create the prompt
        prompt, cachePrefix = mkPrompt(query, results, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory, inference_type=inference_type, output_type=output_type)

        # Call the LLM
        #responseJSON, responseText, cost = def getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True):
        all_responses = []
        MAX_GENERATION_TIME_SECS = generation_time_seconds(60 * 15)    # 15 minutes, unless MAX_GENERATION_TIME_SECONDS or the caller says otherwise
        print("## Calling LLM to generate initial theories...", flush=True)
        responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=MAX_GENERATION_TIME_SECS, cachePrefixStr=(cachePrefix if (use_reflection) else None), cacheKey=cacheKey, retry_at_lower_effort_if_empty=True, repairJSON=True)
        print("## Response Received", flush=True)
        all_responses.append(responseJSON)
        total_cost += cost

        printVerbose("## Initial response:")
        printVerbose(responseText)

        # New one-theory-at-a-time reflection approach
        reflected_response = None
        if (use_reflection):
            print("### Using reflection...")
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
                    reflection_prompt, reflection_cache_prefix = mkPrompt(query, results, reflection=theory_str, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory, inference_type=inference_type, output_type=output_type)

                    # Submit the LLM call to the executor.  These read the prefix that the generation
                    # call above wrote to the cache -- same prefix, same key.
                    future = executor.submit(getLLMResponseJSON, reflection_prompt, model_str, temperature, max_tokens, False,
                                             cachePrefixStr=reflection_cache_prefix, cacheKey=cacheKey,
                                             retry_at_lower_effort_if_empty=True, repairJSON=True)
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

        print("## PACKING OUTPUT ")

        # Pack output
        packed = {
            "query": query,
            "model": model_str,
            "use_reflection": use_reflection,
            "inference_type": inference_type,
            "output_type": output_type,
            "version": version_str,
            "mission_statement": mission_statement,
            "theory_response": best_response,
            "results": results,
            "raw_responses": all_responses,
            "total_cost": total_cost,
        }

        return packed

    except Exception as e:
        import traceback
        print("ERROR in build_theory_from_results_single_theory_reflection(): " + str(e))
        traceback.print_exc()
        return None

    except BaseException as e:
        print("FATAL BaseException escaped:", repr(e), flush=True)
        traceback.print_exc()
        raise



# A new prompt that adds attribution requirements to each statement that gets made.
def build_theory_from_results_single_theory_reflection4a_attribution(query:str, results:list, mission_statement:str=None, model_str="gpt-4o-mini", max_tokens=50000, temperature=0.0, use_reflection:bool=True, include_query_in_generation:bool=True, additional_suggestions=None, max_laws_per_theory:int=3):
    version_str = "build_theory_from_results_single_theory_reflection4a_attribution-literature-supported-accuracy-focused-09022026"
    def mkPrompt(query:str, results:str, reflection:str=None, include_query_in_generation:bool=True, mission_statement:str=None, additional_suggestions=None, max_laws_per_theory=None, filtered_reflection=None):
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"

        prompt += "# Task\n"
        prompt += "Your task is to construct a theory of how a scientific phenomenon works, based on a collection of evidence that you will be provided.\n"
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

        # NEW: ADD SOMETHING ABOUT BEING TOO WORDY.
        prompt += "# Style Guide (Being too wordy)\n"
        prompt += "Theories from past versions of this work have been praised by prominent scientists for being novel, insightful, and making genuine progress on very difficult scientific problems with near-term impact. Congratulations!\n"
        prompt += "However, nearly all those scientists have univerally criticized the *presentation* of the theories, saying that it often gets in the way of their reading, understanding, or using them. They have also said these problems are generally getting worse when new language models are being used, not better.\n"
        prompt += "Here is the sense of some of their feedback, to help you correct for these issues.\n"
        prompt += "- 'Overly wordy'\n"
        prompt += "- 'Too much jargon, and sometimes invents completely new terms or jargon for already well-established concepts, making it very difficult to decipher'\n"
        prompt += "- 'Some theories are good, while others are near gibberish'\n"
        prompt += "- 'Goes only so far, leaves obvious holes that could have been easily worked out -- should be more motivated'\n"
        prompt += "- When asked in a follow-up whether they would prefer the theories in plain language vs technical language, the scientists almost universally say they do want them in technical language, and that the problem isn't that they can't read technical language, it's that the presentation of them is often poor quality in the ways mentioned above.\n"
        prompt += "- Your goal isn't to impress anyone by using overly technical language, or inventing terms for existing concepts (but, please feel free to discover NEW concepts and explicitly name them -- that's a good thing!). The goal is to make genuine discoveries and effectively communicate them in the most useful way possible, so that they can be understood and used effectively.\n"
        prompt += "- The world's great scientists are very busy, and time is a precious resource. If it takes them an hour to read each theory, they can only read a few of them. If it takes them 2-5 minutes, they can read more. But if making those theories easier to read and understand removes core elements, then the theory is now lossy and may not have the utility or impact it once did, and that would be a major loss.\n"
        prompt += "Here are some related quotes to consider:\n"
        prompt += "\"It can scarcely be denied that the supreme goal of all theory is to make the irreducible basic elements as simple and as few as possible without having to surrender the adequate representation of a single datum of experience.\" - Albert Einstein\n"
        prompt += "\"Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex. Complex is better than complicated. Readability counts.\" - Tim Peters, The Zen of Python\n"
        prompt += "\n"


        # NEW: Attribution statement
        prompt += "# Attribution\n"
        prompt += "For users, it's very important that they see how the statements that you construct are built from the evidence, so that they can evaluate the accuracy and utility of those statements.\n"
        prompt += "As such, each law/statement that you generate must include a list of the supporting evidence that was used to construct it.\n"
        prompt += "New in this version of the prompt is that an additional copy of the law/statement is required: one that, for each sentence in the law/statement, attributes the inference in that statement to a specific set of evidence.\n"
        prompt += "## Attribution, Evidence, and Differing Kinds of Inference\n"
        prompt += "It's worth note that there are many ways that one can arrive at a law/statement, and they use evidence differently. For example:\n"
        prompt += "- Some laws/statements are an abstraction, inference, or hypothesis across many different pieces of evidence. They may use many pieces of evidence to create this abstraction.\n"
        prompt += "- Some laws/statements are an inference from using a small number (or single piece) of evidence, such as induction from a small number of examples.\n"
        prompt += "- And so on.\n"
        prompt += "As such, when you attribute each sentence in a law/statement, you will annotate both (a) the evidence used, and also (b) a description of the inference used to go from evidence to the law/statement.\n"
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
        prompt += '    "theory_statement_plain_language": "...", # A 1-2 sentence plain language version of the law/statement, that is clear, concise, and easily readable.\n'
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

        #prompt += "- `unaccounted_for`(list): A list of specific pieces of evidence that are not explained by the theory, if any.\n"
        #prompt += "- `conflicting_evidence`(list): A list of specific pieces of evidence that appear to conflict with the theory, if any.\n"
        # Conflicting Evidence
        prompt += "    \"conflicting_evidence\": [ # Conflicting evidence appears to conflict with the law/statement\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # One piece of conflicting evidence\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # Second piece of conflicting evidence\n"
        prompt += "        # ... more pieces of conflicting evidence as available\n"
        prompt += "    ],\n"
        # Unaccounted For Evidence
        prompt += "    \"unaccounted_for\": [ # Unaccounted for evidence is not explained by the law/statement\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # One piece of unaccounted-for evidence\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # Second piece of unaccounted-for evidence\n"
        prompt += "        # ... more pieces of unaccounted-for evidence as available\n"
        prompt += "    ],\n"
        # Qual/qant
        prompt += '    "qual_or_quant": "..." # Is this law/statement primarily "qualitative", "quantitative", or "mixed"\n'
        # NEW: Attribution Section
        prompt += '    "attribution": [\n'
        prompt += '        {"sentence": "...", # The first sentence from `theory_statement`\n'
        prompt += '         "plain_language_sentence": "...", # A plain-langugae version of this sentence, understandable by a 1st year interdisciplinary MSc student.\n'
        prompt += '         "supporting_evidence": [\n'
        prompt += '             {"uuid": "...", "what_it_says": "..."}, # One piece of supporting evidence, and a very brief description of what it says that supports this sentence\n'
        prompt += '             {"uuid": "...", "what_it_says": "..."}, # (Add as needed)\n'
        prompt += '         ],\n'
        prompt += '         "inference_description": "..." # A 1-2 sentence, clear, concise description of how the statement/law was made, by taking the input data/supporting evidence and transforming it through some inference process. BE SPECIFIC AND CLEAR, NOT WISHY-WASHY, AND BE HONEST AND FAITHFUL.\n'
        prompt += '        },\n'
        prompt += "        # ... repeat for each sentence in the `theory_statement`\n"
        prompt += '    ],\n'

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
            if (mission_statement is not None):
                prompt += "Here is a mission statement that provides more detail about the goals of the theories you should generate:\n"
                prompt += "```\n"
                prompt += str(mission_statement) + "\n"
                prompt += "```\n"
                prompt += "\n"

        if (additional_suggestions is not None) and (reflection is None): # Only include these suggestions in the initial prompt, not in the reflection prompt, since these are meant to be suggestions for the initial theory generation step, not for the reflection step
            prompt += "Here are some additional suggestions for things that you noticed in a pilot study, that may or may not be helpful:\n"
            prompt += "```\n"
            if (isinstance(additional_suggestions, dict) or isinstance(additional_suggestions, list)):
                prompt += json.dumps(additional_suggestions, indent=4) + "\n"
            else:
                prompt += str(additional_suggestions) + "\n"
            prompt += "```\n"
            prompt += "\n"

        if (max_laws_per_theory is not None):
            prompt += "# Maximum laws per theory\n"
            prompt += "When generating theories, please limit the maximum number of laws per theory to no more than the following number: " + str(max_laws_per_theory) + " .\n"
            prompt += "\n"

        prompt += "Here is a list of specific results that you should use to build your theories:\n"
        if (filtered_reflection is not None):
            prompt += "NOTE: This is a filtered version of the results, that only includes the evidence that is relevant to the theory you are reflecting on.  You should use this filtered evidence to reflect on the theory, and fix any errors, inconsistencies, omissions, or any other issues.\n"
        # If this isn't set, just give it the raw data and ask it to brainstorm in a data-driven way rather than in a query-driven way
        prompt += "```\n"
        prompt += json.dumps(results, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"

        # Everything up to this point is identical for the initial generation call and for every
        # reflection call in this run -- the reflected theory and the output format come after.  That
        # makes it the cacheable prefix, and it is nearly the whole prompt when the evidence is large.
        cachePrefix = prompt

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
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output, as these are not valid JSON and will cause a parsing error.\n"
        prompt += "\n"
        prompt += "Your JSON response must be a dictionary, that contains two keys: `theories_general`, and `theories_specific`.  Each is a list of dictionaries, with the following keys:\n"
        prompt += "- `theory_name`(str): A short name for the theory\n"
        prompt += "- `theory_name_plain_language`(str): A short name for the theory expressed in plain language\n"
        prompt += "- `theory_description`(str): A full description of the theory\n"
        prompt += "- `theory_description_plain_language`(str): A full description of the theory expressed in plain language\n"
        #prompt += "- `supporting_evidence`(list): A list of specific supporting evidence for the theory.\n"
        #prompt += "- `theory_statements`(list): A list of specific predictive statements (either qualitative, quantitative, or both) that describe the assertions of the theory.\n"
        prompt += "- `theory_statements`(list): A list of specific laws (either qualitative, quantitative, or both) that describe the assertions of the theory.\n"
        prompt += "- `new_predictions_likely`(list): A list of several example (testable) predictions that the theory makes (the situation, and the expected results) that are not in the observed evidence. For this set of new predictions, the predictions should be fairly likely to work out.\n"
        prompt += "- `new_predictions_unknown`(list): A list of several example (testable) predictions that the theory makes (the situation, and the expected results) that are not in the observed evidence. For this set of new predictions, the predictions should be difficult to tell if they would genuinely work out or not, and have particularly impactful results if they do (or don't) work out.\n"
        prompt += "- `negative_experiments`(list): A list of several (testable) predictions that test the assumptions or statements of the theory, and if false, would call the theory statements into question.\n"
        #prompt += "- `unaccounted_for`(list): A list of specific pieces of evidence that are not explained by the theory, if any.\n"
        #prompt += "- `conflicting_evidence`(list): A list of specific pieces of evidence that appear to conflict with the theory, if any.\n"
        prompt += "- `special_cases`(list): A list of any special cases, exceptions, or boundary conditions that apply to the theory, if any.\n"
        prompt += "- `existing_theory`(dict): A dictionary that describes whether (to the best of your knowledge, and the information available in the papers) if this is a pre-existing theory, or a new theory. It has the following keys: `likely_classification`:str, which is one of `new`, `somewhat-related-to-existing`, `closely-related-to-existing`, or `existing`, and `references`:list(str), which are lists of references (one per string) to new (or related) theories in the form `First author (year) title [brief note of how related, and name(s) of relevant theories, in square brackets]`. These should help confirm the theory is new, or help confirm it is existing or closely related. It also has: `what_already_exists`(str), `what_is_novel`(str), and `classification_explanation`(str).\n"
        prompt += "\n"
        prompt += "While most fields are lists of strings, for `supporting_evidence` and `unaccounted_for`, the list should be a list of dictionaries, where each dictionary has the following keys:\n"
        prompt += "- `text`(str): A textual description of the evidence\n"
        prompt += "- `uuids`(list): A list of UUIDs that correspond to the specific pieces of evidence in the original results that support this text description. These lists can be as long or short as they need to be to encompass all the correct evidence references.\n"
        prompt += "\n"
        prompt += "Similarly, the `theory_statements` field is a list of dictionaries, where each dictionary has the following key:\n"
        #prompt += "- `law`(dict): A dictionary that describes a single qualitative or quantitative law, in the format described above (i.e. with `if` and `then` fields, each containing a list of triples), and specific supporting evidence that supports the development of this law.\n"
        prompt += "- `law`(dict): A dictionary that describes a single qualitative or quantitative law, in the format described above, and specific supporting evidence that supports the development of this law.\n"
        prompt += "\n"
        prompt += "For example (a cartoon example -- your theories should be much more detailed, and have much more supporting evidence, linked back with proper UUIDs; similarly, the unaccounted-for and conflicting evidence here is just filler, yours should be detailed and correct):\n"
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
        prompt += '                  "theory_statement_plain_language": "...",\n'
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
        # Conflicting
        prompt += '                 "conflicting_evidence": [\n'
        prompt += '                      { "text": "The observed accelerated expansion of the universe appears to conflict with the mutual attraction of massive objects.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        # Unaccounted for
        prompt += '                 "unaccounted_for": [\n'
        prompt += '                      { "text": "The phenomenon of dark matter is not explained by the mutual attraction of massive objects.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        prompt += '                  "qual_or_quant": "qualitative",\n'
        prompt += '                  "attribution": [\n'
        prompt += '                      {"sentence": "...", "plain_language_sentence": "...", "supporting_evidence": [{"uuid": "...", "what_it_says": "..."}, ...], "inference_description": "..."},\n'
        prompt += '                      # ... repeat for each sentence in the `theory_statement`\n'
        prompt += '                  ],\n'
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
        prompt += '                  "theory_statement_plain_language": "...",\n'
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
        # Conflicting
        prompt += '                 "conflicting_evidence": [\n'
        prompt += '                      { "text": "The observed accelerated expansion of the universe appears to conflict with the inverse square law of gravitational force.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        # Unaccounted for
        prompt += '                 "unaccounted_for": [\n'
        prompt += '                      { "text": "The phenomenon of dark energy is not explained by the inverse square law of gravitational force.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        prompt += '                  "qual_or_quant": "quantitative",\n'
        prompt += '                  "attribution": [\n'
        prompt += '                      {"sentence": "...", "plain_language_sentence": "...", "supporting_evidence": [{"uuid": "...", "what_it_says": "..."}, ...], "inference_description": "..."},\n'
        prompt += '                      # ... repeat for each sentence in the `theory_statement`\n'
        prompt += '                  ],\n'
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
        prompt += '                  "theory_statement_plain_language": "...",\n'
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
        # Conflicting
        prompt += '                 "conflicting_evidence": [\n'
        prompt += '                      { "text": "The observed accelerated expansion of the universe appears to conflict with the proportional relationship between mass and gravitational force.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        # Unaccounted for
        prompt += '                 "unaccounted_for": [\n'
        prompt += '                      { "text": "The phenomenon of dark energy is not explained by the proportional relationship between mass and gravitational force.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        prompt += '                  "qual_or_quant": "quantitative",\n'
        prompt += '                  "attribution": [\n'
        prompt += '                      {"sentence": "...", "plain_language_sentence": "...", "supporting_evidence": [{"uuid": "...", "what_it_says": "..."}, ...], "inference_description": "..."},\n'
        prompt += '                      # ... repeat for each sentence in the `theory_statement`\n'
        prompt += '                  ],\n'
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
        # prompt += '            "unaccounted_for": [\n'
        # prompt += '                {\"text\": "The exact mechanism of how gravity works at the quantum level is not fully understood.", \"uuids\": [7, 25]},\n'
        # prompt += '                {\"text\": "The behavior of gravity in extreme conditions, such as near black holes, is not fully understood.", \"uuids\": [12, 28]}\n'
        # prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        # prompt += '            ],\n'
        # prompt += '            "conflicting_evidence": [\n'
        # prompt += '                {\"text\": "Observations of galaxy rotation curves that do not match predictions based on visible matter challenge the theory of gravity as currently understood.", \"uuids\": [30, 32]},\n'
        # prompt += '                {\"text\": "The accelerated expansion of the universe challenges the traditional understanding of gravity.", \"uuids\": [35, 40]}\n'
        # prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        # prompt += '            ],\n'
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

        # # Added this style note to help produce more readable theories.
        # prompt += "# Style Notes\n"
        # prompt += "- Theories should be as technical as required, but not more technical than is required (and certainly not over-simplified, such that information is lost; or over-technical/jargony, such that the important elements are obscured).\n"
        # prompt += "- When possible, theories should use simpler language (e.g. that is accessible to a 1st year interdisciplinary masters student), but not at the expense of losing information or accuracy. If you need to use a highly technical term because it's important, use it. Novelty and impact are important. But readability is also important.\n"
        # prompt += "- When possible, please avoid generating new terminology or jargon unless it is necessary to accurately describe a new concept or phenomenon. If you already know the correct term for a concept in a field, use it, rather than inventing a new one.  If a new one is needed, then you can create a new term, but try not to make it too difficult to understand.\n"
        # prompt += "- A general comment has been that the theories are often very verbose, jargon-heavy, and sometimes hard to read by human experts who are most in a position to benefit from their knowledge, and make impactful discoveries/advances. Please keep this in mind, and aim for clarity, conciseness, and readability, while still being novel and impactful.\n"
        # prompt += "\n"

        prompt += "# Important Notes\n"
        prompt += "- You must return a JSON response, and it must be valid JSON, or it will not be parsed.\n"
        prompt += "- You are encouraged to think and plan before writing your JSON response, but your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output.\n"
        prompt += "- Values that are 'none'/'null' in the JSON response should be represented as `null` in the JSON, not as an empty string, string saying \"null\", or any other value.\n"
        prompt += "- Where possible, you should prefer generating new, impactful theories.\n"
        if (max_laws_per_theory is not None):
            prompt += "- When generating theories, please limit the maximum number of laws per theory to no more than the following number: " + str(max_laws_per_theory) + " .\n"
        prompt += "- All your information must be accurate. Do not hallucinate.\n"

        return prompt, cachePrefix

    try:
        total_cost = 0.0

        # Prompt caching is only worth it when more than one call shares the prefix, which here means
        # only when reflection is on: the generation call writes the evidence to the cache and each
        # reflection call reads it back.  With reflection off there is a single call, and a cache
        # write costs more than a plain one, so leave it off.  The key is per-run, and routes this
        # run's calls to the same machine so the reads can hit.
        cacheKey = None
        if (use_reflection):
            cacheKey = "theorizer-reflection4-" + str(os.getpid()) + "-" + str(threading.get_ident()) + "-" + str(int(time.time() * 1000))

        # Create the prompt
        prompt, cachePrefix = mkPrompt(query, results, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory)

        # Call the LLM
        #responseJSON, responseText, cost = def getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True):
        all_responses = []
        MAX_GENERATION_TIME_SECS = generation_time_seconds(60 * 15)    # 15 minutes, unless MAX_GENERATION_TIME_SECONDS or the caller says otherwise
        print("## Calling LLM to generate initial theories...", flush=True)
        responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=MAX_GENERATION_TIME_SECS, cachePrefixStr=(cachePrefix if (use_reflection) else None), cacheKey=cacheKey, retry_at_lower_effort_if_empty=True, repairJSON=True)
        print("## Response Received", flush=True)
        all_responses.append(responseJSON)
        total_cost += cost

        printVerbose("## Initial response:")
        printVerbose(responseText)

        # Save a copy of the theories after the first reflection
        theories_after_first_reflection = []

        # New one-theory-at-a-time reflection approach
        reflected_response = None
        if (use_reflection):
            print("### Using reflection...")
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
                    reflection_prompt, reflection_cache_prefix = mkPrompt(query, results, reflection=theory_str, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory)

                    # Submit the LLM call to the executor
                    # Submit the LLM call to the executor.  These read the prefix that the generation
                    # call above wrote to the cache -- same prefix, same key.
                    future = executor.submit(getLLMResponseJSON, reflection_prompt, model_str, temperature, max_tokens, False,
                                             cachePrefixStr=reflection_cache_prefix, cacheKey=cacheKey,
                                             retry_at_lower_effort_if_empty=True, repairJSON=True,
                                             max_generation_time_seconds=MAX_GENERATION_TIME_SECS)
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

            # Save a copy of the theories after the first reflection, and before the second reflection.
            # Should deep copy them.
            try:
                from copy import deepcopy
                theories_after_first_reflection = deepcopy(reflected_theories_general + reflected_theories_specific)
            except Exception as e:
                print("Error deep copying theories after first reflection: " + str(e))
                theories_after_first_reflection = reflected_theories_general + reflected_theories_specific

            def collect_theory_relevant_evidence(theory, results):
                # Step 1: Collect all the UUIDs from the theory's supporting evidence and unaccounted-for evidence
                relevant_uuids = set()
                # Crawl the entire structure, looking for any keys called `uuids` or `uuid` and collect them.
                def crawl_structure(structure):
                    if isinstance(structure, dict):
                        for key, value in structure.items():
                            if key == "uuids" and isinstance(value, list):
                                relevant_uuids.update(value)
                            elif key == "uuid" and isinstance(value, str):
                                relevant_uuids.add(value)
                            else:
                                crawl_structure(value)
                    elif isinstance(structure, list):
                        for item in structure:
                            crawl_structure(item)
                # Start crawling from the top-level theory structure
                crawl_structure(theory)

                # Step 2: Filter the results to only include those with UUIDs in relevant_uuids
                filtered_results = []
                for result in results:  # Results should be a list of dictionaries, and each dictionary should have a 'uuid' key.
                    try:
                        if ("uuid" in result) and (result["uuid"] in relevant_uuids):
                            filtered_results.append(result)
                    except Exception as e:
                        print(f"Error processing result {result}: {e}")
                        continue

                print("Filtered results from " + str(len(results)) + " to " + str(len(filtered_results)) + " based on theory's relevant UUIDs.")
                if (len(filtered_results) == 0):
                    print("WARNING: No relevant results found for theory.")
                return filtered_results



            #SECOND_REFLECTION_ENABLED = True
            SECOND_REFLECTION_ENABLED = False
            reflected_theories_general2 = []
            reflected_theories_specific2 = []
            reflected_theories_other2 = []
            if (SECOND_REFLECTION_ENABLED):
                # SECOND REFLECTION: Let's to the reflection again, but a cheaper version: This time we'll filter the list of evidence to only the UUIDs that were mentioned in the first reflection.
                # That should make the call much cheaper, but should also make it a lot easier for the LLM to make the theory more consistent.
                with ThreadPoolExecutor(max_workers=MAX_THEORY_GENERATION_WORKERS) as executor:
                    future_to_theory_idx = {}
                    # All the theories should be in `reflceted_theories_general` and `reflected_theories_specific` now, so we can just iterate over those
                    for idx, theory in enumerate(reflected_theories_general + reflected_theories_specific):
                        # Create a reflection prompt for this theory
                        theory_str = json.dumps(theory, indent=4)
                        filtered_results = collect_theory_relevant_evidence(theory, results)
                        #reflection_prompt, reflection_cache_prefix = mkPrompt(query, results, reflection=theory_str, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory)
                        reflection_prompt, reflection_cache_prefix = mkPrompt(query, filtered_results, reflection=theory_str, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory, filtered_reflection=True)

                        # Submit the LLM call to the executor
                        future = executor.submit(getLLMResponseJSON, reflection_prompt, model_str, temperature, max_tokens, False,
                                                cachePrefixStr=reflection_cache_prefix, cacheKey=cacheKey,
                                                retry_at_lower_effort_if_empty=True, repairJSON=True,
                                                max_generation_time_seconds=MAX_GENERATION_TIME_SECS)
                        future_to_theory_idx[future] = (idx, theory)

                    # Process the completed futures
                    from tqdm import tqdm
                    for future in tqdm(as_completed(future_to_theory_idx), total=len(future_to_theory_idx), desc="Second reflection on theories"):
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
                                        reflected_theories_general2.append(theory_candidate)
                                        num_added += 1
                                elif ("theories_specific" in responseJSON) and (len(responseJSON["theories_specific"]) > 0):
                                    for theory_candidate in responseJSON["theories_specific"]:
                                        theory_candidate["reflected_from_theory_index"] = idx
                                        reflected_theories_specific2.append(theory_candidate)
                                        num_added += 1

                                # If we couldn't extract a theory, just use the original one
                                if (num_added == 0):
                                    if (theory["theory_type_general_specific"] == "general"):
                                        theory_candidate["reflected_from_theory_index"] = idx
                                        reflected_theories_general2.append(theory)
                                    elif (theory["theory_type_general_specific"] == "specific"):
                                        theory_candidate["reflected_from_theory_index"] = idx
                                        reflected_theories_specific2.append(theory)
                                    else:
                                        theory_candidate["reflected_from_theory_index"] = idx
                                        reflected_theories_other2.append(theory)
                            else:
                                # If we didn't get a valid response, just use the original one
                                if (theory["theory_type_general_specific"] == "general"):
                                    theory_candidate["reflected_from_theory_index"] = idx
                                    reflected_theories_general2.append(theory)
                                elif (theory["theory_type_general_specific"] == "specific"):
                                    theory_candidate["reflected_from_theory_index"] = idx
                                    reflected_theories_specific2.append(theory)
                                else:
                                    theory_candidate["reflected_from_theory_index"] = idx
                                    reflected_theories_other2.append(theory)

                        except Exception as e:
                            print(f"build_theory_from_results_single_theory_reflection(): ERROR processing theory index {idx} during second reflection: {e}")

                # So we don't break the pipeline that expects there to be only two bins (general and specific), we will just merge any "other" theories into the general bin.
                for theory in reflected_theories_other2:
                    reflected_theories_general2.append(theory)

                # Replace the first reflection with the second reflection, if enabled
                reflected_theories_general = reflected_theories_general2
                reflected_theories_specific = reflected_theories_specific2


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

        print("## PACKING OUTPUT ")

        # Pack output
        packed = {
            "query": query,
            "model": model_str,
            "use_reflection": use_reflection,
            "mission_statement": mission_statement,
            "theory_response": best_response,
            "results": results,
            "raw_responses": all_responses,
            "theories_after_first_reflection": theories_after_first_reflection,
            "total_cost": total_cost,
        }

        return packed

    except Exception as e:
        import traceback
        print("ERROR in build_theory_from_results_single_theory_reflection(): " + str(e))
        traceback.print_exc()
        return None

    except BaseException as e:
        print("FATAL BaseException escaped:", repr(e), flush=True)
        traceback.print_exc()
        raise


# A new prompt that adds attribution requirements to each statement that gets made.
def build_theory_from_results_single_theory_reflection4b_attribution_batchedevidence(query:str, results:list, results_leftover:list=None, mission_statement:str=None, model_str="gpt-4o-mini", max_tokens=50000, temperature=0.0, use_reflection:bool=True, include_query_in_generation:bool=True, additional_suggestions=None, max_laws_per_theory:int=3, max_evidence_tokens:int=800000, max_batch_reflections:int=3, rng=None):
    # `results` is the initial batch of evidence -- the one that fits in the context window, and the
    # only one the generation and the first two reflections ever see.  `results_leftover` is
    # everything that did not fit.  When a corpus is 1000+ papers the leftover is most of the
    # evidence, so after the usual reflections this makes further passes: each one draws a new batch
    # from what is left over, sized to whatever room remains once the theory's own cited evidence is
    # in the prompt, and asks the theory to take account of it.  Batches are drawn with the same
    # sort-then-random-sample procedure as the initial one, so each is a cross-section of the
    # remaining evidence rather than an alphabetical slice, and no record appears in two batches.
    #
    # max_evidence_tokens   the ceiling for all the evidence in one prompt (cited + new batch)
    # max_batch_reflections how many further passes to make at most, however much is left over
    # rng                   a random.Random for the batch draws (None: the global generator)
    version_str = "build_theory_from_results_single_theory_reflection4b_attribution_batchedevidence-literature-supported-accuracy-focused-09022026"
    def mkPrompt(query:str, results:str, reflection:str=None, include_query_in_generation:bool=True, mission_statement:str=None, additional_suggestions=None, max_laws_per_theory=None, filtered_reflection=None, new_batch_results=None):
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"

        prompt += "# Task\n"
        prompt += "Your task is to construct a theory of how a scientific phenomenon works, based on a collection of evidence that you will be provided.\n"
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

        # NEW: ADD SOMETHING ABOUT BEING TOO WORDY.
        prompt += "# Style Guide (Being too wordy)\n"
        prompt += "Theories from past versions of this work have been praised by prominent scientists for being novel, insightful, and making genuine progress on very difficult scientific problems with near-term impact. Congratulations!\n"
        prompt += "However, nearly all those scientists have univerally criticized the *presentation* of the theories, saying that it often gets in the way of their reading, understanding, or using them. They have also said these problems are generally getting worse when new language models are being used, not better.\n"
        prompt += "Here is the sense of some of their feedback, to help you correct for these issues.\n"
        prompt += "- 'Overly wordy'\n"
        prompt += "- 'Too much jargon, and sometimes invents completely new terms or jargon for already well-established concepts, making it very difficult to decipher'\n"
        prompt += "- 'Some theories are good, while others are near gibberish'\n"
        prompt += "- 'Goes only so far, leaves obvious holes that could have been easily worked out -- should be more motivated'\n"
        prompt += "- When asked in a follow-up whether they would prefer the theories in plain language vs technical language, the scientists almost universally say they do want them in technical language, and that the problem isn't that they can't read technical language, it's that the presentation of them is often poor quality in the ways mentioned above.\n"
        prompt += "- Your goal isn't to impress anyone by using overly technical language, or inventing terms for existing concepts (but, please feel free to discover NEW concepts and explicitly name them -- that's a good thing!). The goal is to make genuine discoveries and effectively communicate them in the most useful way possible, so that they can be understood and used effectively.\n"
        prompt += "- The world's great scientists are very busy, and time is a precious resource. If it takes them an hour to read each theory, they can only read a few of them. If it takes them 2-5 minutes, they can read more. But if making those theories easier to read and understand removes core elements, then the theory is now lossy and may not have the utility or impact it once did, and that would be a major loss.\n"
        prompt += "Here are some related quotes to consider:\n"
        prompt += "\"It can scarcely be denied that the supreme goal of all theory is to make the irreducible basic elements as simple and as few as possible without having to surrender the adequate representation of a single datum of experience.\" - Albert Einstein\n"
        prompt += "\"Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex. Complex is better than complicated. Readability counts.\" - Tim Peters, The Zen of Python\n"
        prompt += "\n"


        # NEW: Attribution statement
        prompt += "# Attribution\n"
        prompt += "For users, it's very important that they see how the statements that you construct are built from the evidence, so that they can evaluate the accuracy and utility of those statements.\n"
        prompt += "As such, each law/statement that you generate must include a list of the supporting evidence that was used to construct it.\n"
        prompt += "New in this version of the prompt is that an additional copy of the law/statement is required: one that, for each sentence in the law/statement, attributes the inference in that statement to a specific set of evidence.\n"
        prompt += "## Attribution, Evidence, and Differing Kinds of Inference\n"
        prompt += "It's worth note that there are many ways that one can arrive at a law/statement, and they use evidence differently. For example:\n"
        prompt += "- Some laws/statements are an abstraction, inference, or hypothesis across many different pieces of evidence. They may use many pieces of evidence to create this abstraction.\n"
        prompt += "- Some laws/statements are an inference from using a small number (or single piece) of evidence, such as induction from a small number of examples.\n"
        prompt += "- And so on.\n"
        prompt += "As such, when you attribute each sentence in a law/statement, you will annotate both (a) the evidence used, and also (b) a description of the inference used to go from evidence to the law/statement.\n"
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
        prompt += '    "theory_statement_plain_language": "...", # A 1-2 sentence plain language version of the law/statement, that is clear, concise, and easily readable.\n'
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

        #prompt += "- `unaccounted_for`(list): A list of specific pieces of evidence that are not explained by the theory, if any.\n"
        #prompt += "- `conflicting_evidence`(list): A list of specific pieces of evidence that appear to conflict with the theory, if any.\n"
        # Conflicting Evidence
        prompt += "    \"conflicting_evidence\": [ # Conflicting evidence appears to conflict with the law/statement\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # One piece of conflicting evidence\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # Second piece of conflicting evidence\n"
        prompt += "        # ... more pieces of conflicting evidence as available\n"
        prompt += "    ],\n"
        # Unaccounted For Evidence
        prompt += "    \"unaccounted_for\": [ # Unaccounted for evidence is not explained by the law/statement\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # One piece of unaccounted-for evidence\n"
        prompt += "        {\"text\": \"...\", \"uuids\": [...]}, # Second piece of unaccounted-for evidence\n"
        prompt += "        # ... more pieces of unaccounted-for evidence as available\n"
        prompt += "    ],\n"
        # Qual/qant
        prompt += '    "qual_or_quant": "..." # Is this law/statement primarily "qualitative", "quantitative", or "mixed"\n'
        # NEW: Attribution Section
        prompt += '    "attribution": [\n'
        prompt += '        {"sentence": "...", # The first sentence from `theory_statement`\n'
        prompt += '         "plain_language_sentence": "...", # A plain-langugae version of this sentence, understandable by a 1st year interdisciplinary MSc student.\n'
        prompt += '         "supporting_evidence": [\n'
        prompt += '             {"uuid": "...", "what_it_says": "..."}, # One piece of supporting evidence, and a very brief description of what it says that supports this sentence\n'
        prompt += '             {"uuid": "...", "what_it_says": "..."}, # (Add as needed)\n'
        prompt += '         ],\n'
        prompt += '         "inference_description": "..." # A 1-2 sentence, clear, concise description of how the statement/law was made, by taking the input data/supporting evidence and transforming it through some inference process. BE SPECIFIC AND CLEAR, NOT WISHY-WASHY, AND BE HONEST AND FAITHFUL.\n'
        prompt += '        },\n'
        prompt += "        # ... repeat for each sentence in the `theory_statement`\n"
        prompt += '    ],\n'

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
            if (mission_statement is not None):
                prompt += "Here is a mission statement that provides more detail about the goals of the theories you should generate:\n"
                prompt += "```\n"
                prompt += str(mission_statement) + "\n"
                prompt += "```\n"
                prompt += "\n"

        if (additional_suggestions is not None) and (reflection is None): # Only include these suggestions in the initial prompt, not in the reflection prompt, since these are meant to be suggestions for the initial theory generation step, not for the reflection step
            prompt += "Here are some additional suggestions for things that you noticed in a pilot study, that may or may not be helpful:\n"
            prompt += "```\n"
            if (isinstance(additional_suggestions, dict) or isinstance(additional_suggestions, list)):
                prompt += json.dumps(additional_suggestions, indent=4) + "\n"
            else:
                prompt += str(additional_suggestions) + "\n"
            prompt += "```\n"
            prompt += "\n"

        if (max_laws_per_theory is not None):
            prompt += "# Maximum laws per theory\n"
            prompt += "When generating theories, please limit the maximum number of laws per theory to no more than the following number: " + str(max_laws_per_theory) + " .\n"
            prompt += "\n"

        # A further batch of evidence, drawn from the records that did not fit in the original
        # context window.  It goes ABOVE the results below rather than under them, because within
        # one batch pass every theory is shown the same new batch but its own (differing) cited
        # evidence: putting the shared part first makes the whole prompt up to this point identical
        # across those calls, so it can be written to the prompt cache once and read back for the
        # rest.  Underneath it, the results the theory already had are kept, not replaced.
        # Fitting the two together within the token budget is the caller's job -- by the time it
        # gets here, both are the size they are going to be.
        batch_cache_prefix = None
        if (new_batch_results is not None):
            prompt += "# Additional Evidence (a new batch)\n"
            prompt += "The corpus this theory is built from is far larger than fits in one context window, so you have so far seen only part of it.  This is a NEW batch of evidence from that corpus, which you have not seen before.  The evidence your theory already rests on follows below it, and no record appears in both.\n"
            prompt += "Your task is to take this new evidence into account: use it to support, extend, qualify, correct, or challenge the theory as appropriate.  Evidence that genuinely supports a law should be added to that law's supporting evidence (and to its attribution), evidence that conflicts with it should be recorded as conflicting evidence, and evidence the theory does not explain should be recorded as unaccounted for.\n"
            prompt += "Do not drop the evidence you already had.  A law that was well supported before should still carry that support afterwards -- this batch adds to what you know, it does not replace it.\n"
            prompt += "If the new evidence genuinely changes the picture -- a law is too broad, a domain/scope is wrong, a relationship holds only under conditions you can now see -- then change the theory to match the evidence, and say so in the relevant fields.\n"
            prompt += "```\n"
            prompt += json.dumps(new_batch_results, indent=4) + "\n"
            prompt += "```\n"
            prompt += "\n"
            # Everything to here is the same for every theory in this batch pass.
            batch_cache_prefix = prompt

        prompt += "Here is a list of specific results that you should use to build your theories:\n"
        if (filtered_reflection is not None):
            prompt += "NOTE: This is a filtered version of the results, that only includes the evidence that is relevant to the theory you are reflecting on.  You should use this filtered evidence to reflect on the theory, and fix any errors, inconsistencies, omissions, or any other issues.\n"
        # If this isn't set, just give it the raw data and ask it to brainstorm in a data-driven way rather than in a query-driven way
        prompt += "```\n"
        prompt += json.dumps(results, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"

        # Everything up to this point is identical for the initial generation call and for every
        # reflection call in this run -- the reflected theory and the output format come after.  That
        # makes it the cacheable prefix, and it is nearly the whole prompt when the evidence is large.
        # In a batch pass the shared part ends earlier, at the end of the new batch, since the
        # results below it differ from theory to theory.
        cachePrefix = prompt
        if (batch_cache_prefix is not None):
            cachePrefix = batch_cache_prefix

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
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output, as these are not valid JSON and will cause a parsing error.\n"
        prompt += "\n"
        prompt += "Your JSON response must be a dictionary, that contains two keys: `theories_general`, and `theories_specific`.  Each is a list of dictionaries, with the following keys:\n"
        prompt += "- `theory_name`(str): A short name for the theory\n"
        prompt += "- `theory_name_plain_language`(str): A short name for the theory expressed in plain language\n"
        prompt += "- `theory_description`(str): A full description of the theory\n"
        prompt += "- `theory_description_plain_language`(str): A full description of the theory expressed in plain language\n"
        #prompt += "- `supporting_evidence`(list): A list of specific supporting evidence for the theory.\n"
        #prompt += "- `theory_statements`(list): A list of specific predictive statements (either qualitative, quantitative, or both) that describe the assertions of the theory.\n"
        prompt += "- `theory_statements`(list): A list of specific laws (either qualitative, quantitative, or both) that describe the assertions of the theory.\n"
        prompt += "- `new_predictions_likely`(list): A list of several example (testable) predictions that the theory makes (the situation, and the expected results) that are not in the observed evidence. For this set of new predictions, the predictions should be fairly likely to work out.\n"
        prompt += "- `new_predictions_unknown`(list): A list of several example (testable) predictions that the theory makes (the situation, and the expected results) that are not in the observed evidence. For this set of new predictions, the predictions should be difficult to tell if they would genuinely work out or not, and have particularly impactful results if they do (or don't) work out.\n"
        prompt += "- `negative_experiments`(list): A list of several (testable) predictions that test the assumptions or statements of the theory, and if false, would call the theory statements into question.\n"
        #prompt += "- `unaccounted_for`(list): A list of specific pieces of evidence that are not explained by the theory, if any.\n"
        #prompt += "- `conflicting_evidence`(list): A list of specific pieces of evidence that appear to conflict with the theory, if any.\n"
        prompt += "- `special_cases`(list): A list of any special cases, exceptions, or boundary conditions that apply to the theory, if any.\n"
        prompt += "- `existing_theory`(dict): A dictionary that describes whether (to the best of your knowledge, and the information available in the papers) if this is a pre-existing theory, or a new theory. It has the following keys: `likely_classification`:str, which is one of `new`, `somewhat-related-to-existing`, `closely-related-to-existing`, or `existing`, and `references`:list(str), which are lists of references (one per string) to new (or related) theories in the form `First author (year) title [brief note of how related, and name(s) of relevant theories, in square brackets]`. These should help confirm the theory is new, or help confirm it is existing or closely related. It also has: `what_already_exists`(str), `what_is_novel`(str), and `classification_explanation`(str).\n"
        prompt += "\n"
        prompt += "While most fields are lists of strings, for `supporting_evidence` and `unaccounted_for`, the list should be a list of dictionaries, where each dictionary has the following keys:\n"
        prompt += "- `text`(str): A textual description of the evidence\n"
        prompt += "- `uuids`(list): A list of UUIDs that correspond to the specific pieces of evidence in the original results that support this text description. These lists can be as long or short as they need to be to encompass all the correct evidence references.\n"
        prompt += "\n"
        prompt += "Similarly, the `theory_statements` field is a list of dictionaries, where each dictionary has the following key:\n"
        #prompt += "- `law`(dict): A dictionary that describes a single qualitative or quantitative law, in the format described above (i.e. with `if` and `then` fields, each containing a list of triples), and specific supporting evidence that supports the development of this law.\n"
        prompt += "- `law`(dict): A dictionary that describes a single qualitative or quantitative law, in the format described above, and specific supporting evidence that supports the development of this law.\n"
        prompt += "\n"
        prompt += "For example (a cartoon example -- your theories should be much more detailed, and have much more supporting evidence, linked back with proper UUIDs; similarly, the unaccounted-for and conflicting evidence here is just filler, yours should be detailed and correct):\n"
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
        prompt += '                  "theory_statement_plain_language": "...",\n'
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
        # Conflicting
        prompt += '                 "conflicting_evidence": [\n'
        prompt += '                      { "text": "The observed accelerated expansion of the universe appears to conflict with the mutual attraction of massive objects.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        # Unaccounted for
        prompt += '                 "unaccounted_for": [\n'
        prompt += '                      { "text": "The phenomenon of dark matter is not explained by the mutual attraction of massive objects.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        prompt += '                  "qual_or_quant": "qualitative",\n'
        prompt += '                  "attribution": [\n'
        prompt += '                      {"sentence": "...", "plain_language_sentence": "...", "supporting_evidence": [{"uuid": "...", "what_it_says": "..."}, ...], "inference_description": "..."},\n'
        prompt += '                      # ... repeat for each sentence in the `theory_statement`\n'
        prompt += '                  ],\n'
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
        prompt += '                  "theory_statement_plain_language": "...",\n'
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
        # Conflicting
        prompt += '                 "conflicting_evidence": [\n'
        prompt += '                      { "text": "The observed accelerated expansion of the universe appears to conflict with the inverse square law of gravitational force.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        # Unaccounted for
        prompt += '                 "unaccounted_for": [\n'
        prompt += '                      { "text": "The phenomenon of dark energy is not explained by the inverse square law of gravitational force.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        prompt += '                  "qual_or_quant": "quantitative",\n'
        prompt += '                  "attribution": [\n'
        prompt += '                      {"sentence": "...", "plain_language_sentence": "...", "supporting_evidence": [{"uuid": "...", "what_it_says": "..."}, ...], "inference_description": "..."},\n'
        prompt += '                      # ... repeat for each sentence in the `theory_statement`\n'
        prompt += '                  ],\n'
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
        prompt += '                  "theory_statement_plain_language": "...",\n'
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
        # Conflicting
        prompt += '                 "conflicting_evidence": [\n'
        prompt += '                      { "text": "The observed accelerated expansion of the universe appears to conflict with the proportional relationship between mass and gravitational force.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        # Unaccounted for
        prompt += '                 "unaccounted_for": [\n'
        prompt += '                      { "text": "The phenomenon of dark energy is not explained by the proportional relationship between mass and gravitational force.", "uuids": [...] },\n'
        prompt += '                      # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        prompt += '                  ],\n'
        prompt += '                  "qual_or_quant": "quantitative",\n'
        prompt += '                  "attribution": [\n'
        prompt += '                      {"sentence": "...", "plain_language_sentence": "...", "supporting_evidence": [{"uuid": "...", "what_it_says": "..."}, ...], "inference_description": "..."},\n'
        prompt += '                      # ... repeat for each sentence in the `theory_statement`\n'
        prompt += '                  ],\n'
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
        # prompt += '            "unaccounted_for": [\n'
        # prompt += '                {\"text\": "The exact mechanism of how gravity works at the quantum level is not fully understood.", \"uuids\": [7, 25]},\n'
        # prompt += '                {\"text\": "The behavior of gravity in extreme conditions, such as near black holes, is not fully understood.", \"uuids\": [12, 28]}\n'
        # prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        # prompt += '            ],\n'
        # prompt += '            "conflicting_evidence": [\n'
        # prompt += '                {\"text\": "Observations of galaxy rotation curves that do not match predictions based on visible matter challenge the theory of gravity as currently understood.", \"uuids\": [30, 32]},\n'
        # prompt += '                {\"text\": "The accelerated expansion of the universe challenges the traditional understanding of gravity.", \"uuids\": [35, 40]}\n'
        # prompt += '                # ... this should be as detailed/exhaustive as possible in relation to the list of results provided. Add as many elements as needed.\n'
        # prompt += '            ],\n'
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

        # # Added this style note to help produce more readable theories.
        # prompt += "# Style Notes\n"
        # prompt += "- Theories should be as technical as required, but not more technical than is required (and certainly not over-simplified, such that information is lost; or over-technical/jargony, such that the important elements are obscured).\n"
        # prompt += "- When possible, theories should use simpler language (e.g. that is accessible to a 1st year interdisciplinary masters student), but not at the expense of losing information or accuracy. If you need to use a highly technical term because it's important, use it. Novelty and impact are important. But readability is also important.\n"
        # prompt += "- When possible, please avoid generating new terminology or jargon unless it is necessary to accurately describe a new concept or phenomenon. If you already know the correct term for a concept in a field, use it, rather than inventing a new one.  If a new one is needed, then you can create a new term, but try not to make it too difficult to understand.\n"
        # prompt += "- A general comment has been that the theories are often very verbose, jargon-heavy, and sometimes hard to read by human experts who are most in a position to benefit from their knowledge, and make impactful discoveries/advances. Please keep this in mind, and aim for clarity, conciseness, and readability, while still being novel and impactful.\n"
        # prompt += "\n"

        prompt += "# Important Notes\n"
        prompt += "- You must return a JSON response, and it must be valid JSON, or it will not be parsed.\n"
        prompt += "- You are encouraged to think and plan before writing your JSON response, but your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output.\n"
        prompt += "- Values that are 'none'/'null' in the JSON response should be represented as `null` in the JSON, not as an empty string, string saying \"null\", or any other value.\n"
        prompt += "- Where possible, you should prefer generating new, impactful theories.\n"
        if (max_laws_per_theory is not None):
            prompt += "- When generating theories, please limit the maximum number of laws per theory to no more than the following number: " + str(max_laws_per_theory) + " .\n"
        prompt += "- All your information must be accurate. Do not hallucinate.\n"

        return prompt, cachePrefix

    try:
        total_cost = 0.0

        # Prompt caching is only worth it when more than one call shares the prefix, which here means
        # only when reflection is on: the generation call writes the evidence to the cache and each
        # reflection call reads it back.  With reflection off there is a single call, and a cache
        # write costs more than a plain one, so leave it off.  The key is per-run, and routes this
        # run's calls to the same machine so the reads can hit.
        cacheKey = None
        if (use_reflection):
            cacheKey = "theorizer-reflection4-" + str(os.getpid()) + "-" + str(threading.get_ident()) + "-" + str(int(time.time() * 1000))

        # Create the prompt
        prompt, cachePrefix = mkPrompt(query, results, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory)

        # Call the LLM
        #responseJSON, responseText, cost = def getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True):
        all_responses = []
        MAX_GENERATION_TIME_SECS = generation_time_seconds(60 * 15)    # 15 minutes, unless MAX_GENERATION_TIME_SECONDS or the caller says otherwise
        print("## Calling LLM to generate initial theories...", flush=True)
        responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=MAX_GENERATION_TIME_SECS, cachePrefixStr=(cachePrefix if (use_reflection) else None), cacheKey=cacheKey, retry_at_lower_effort_if_empty=True, repairJSON=True)
        print("## Response Received", flush=True)
        all_responses.append(responseJSON)
        total_cost += cost

        printVerbose("## Initial response:")
        printVerbose(responseText)

        # Save a copy of the theories after the first reflection
        theories_after_first_reflection = []
        #
        #   Keeping a record of every stage
        #
        # Each stage below (generation, each reflection, each evidence batch) appends a deep copy of
        # the theories as they stood at the end of it, together with summary statistics, so how a
        # theory changes as more evidence arrives can be looked at afterwards rather than inferred
        # from the final state.  Returned as `theories_by_stage` in the packed output.
        theories_by_stage = []

        # Every uuid a theory cites, anywhere in its structure.  Used both for the statistics below
        # and to work out which evidence to keep in front of the model in a later batch.
        def collect_uuids(structure):
            found = set()
            def crawl(node):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if (key == "uuids") and isinstance(value, list):
                            for item in value:
                                if isinstance(item, str):
                                    found.add(item)
                        elif (key == "uuid") and isinstance(value, str):
                            found.add(value)
                        else:
                            crawl(value)
                elif isinstance(node, list):
                    for item in node:
                        crawl(item)
            crawl(structure)
            return found

        def laws_of_theory(theory):
            laws = []
            for statement in (theory.get("theory_statements") or []):
                if (isinstance(statement, dict)) and (len(statement) == 1) and ("law" in statement):
                    laws.append(statement["law"])
                else:
                    laws.append(statement)
            return laws

        def summarize_theories(theories_in):
            # Counts only.  The theories themselves are kept alongside these, so anything not
            # counted here can still be recovered from the snapshot.
            num_laws = 0
            num_attribution_entries = 0
            num_attribution_citations = 0
            num_law_evidence_uuids = 0
            num_conflicting_uuids = 0
            num_unaccounted_uuids = 0
            per_theory = []

            for theory in (theories_in or []):
                laws = laws_of_theory(theory)
                theory_evidence = 0
                theory_attribution = 0
                for law in laws:
                    if (not isinstance(law, dict)):
                        continue
                    for evidence in (law.get("supporting_evidence") or []):
                        if (isinstance(evidence, dict)):
                            theory_evidence += len(evidence.get("uuids") or [])
                    for evidence in (law.get("conflicting_evidence") or []):
                        if (isinstance(evidence, dict)):
                            num_conflicting_uuids += len(evidence.get("uuids") or [])
                    for evidence in (law.get("unaccounted_for") or []):
                        if (isinstance(evidence, dict)):
                            num_unaccounted_uuids += len(evidence.get("uuids") or [])
                    for entry in (law.get("attribution") or []):
                        if (not isinstance(entry, dict)):
                            continue
                        theory_attribution += 1
                        num_attribution_citations += len(entry.get("supporting_evidence") or [])
                num_laws += len(laws)
                num_law_evidence_uuids += theory_evidence
                num_attribution_entries += theory_attribution
                per_theory.append({
                    "theory_name": theory.get("theory_name"),
                    "num_laws": len(laws),
                    "num_evidence_uuids": theory_evidence,
                    "num_unique_evidence_uuids": len(collect_uuids(theory)),
                    "num_attribution_entries": theory_attribution,
                })

            return {
                "num_theories": len(theories_in or []),
                "num_laws": num_laws,
                "num_law_evidence_uuids": num_law_evidence_uuids,
                "num_conflicting_evidence_uuids": num_conflicting_uuids,
                "num_unaccounted_for_uuids": num_unaccounted_uuids,
                "num_attribution_entries": num_attribution_entries,
                "num_attribution_citations": num_attribution_citations,
                "per_theory": per_theory,
            }

        def record_stage(stage_name, theories_in, extra=None):
            # Deep copy, so a later stage editing a theory in place cannot change what an earlier
            # stage recorded.
            try:
                from copy import deepcopy
                snapshot = deepcopy(theories_in or [])
            except Exception as e:
                print("record_stage(): Error deep copying theories for stage '" + str(stage_name) + "': " + str(e))
                snapshot = list(theories_in or [])

            entry = {
                "stage": stage_name,
                "theories": snapshot,
                "stats": summarize_theories(theories_in),
            }
            if (extra is not None):
                entry.update(extra)
            theories_by_stage.append(entry)

            stats = entry["stats"]
            print("### STAGE '" + str(stage_name) + "': " + str(stats["num_theories"]) + " theories, "
                  + str(stats["num_laws"]) + " laws, " + str(stats["num_law_evidence_uuids"]) + " evidence uuids, "
                  + str(stats["num_attribution_entries"]) + " attribution entries, "
                  + str(stats["num_attribution_citations"]) + " sentence-level citations", flush=True)
            return entry

        # The theories as generated, before any reflection
        if (responseJSON is not None):
            record_stage("generation",
                         (responseJSON.get("theories_general") or []) + (responseJSON.get("theories_specific") or []))

        # New one-theory-at-a-time reflection approach
        reflected_response = None
        if (use_reflection):
            print("### Using reflection...")
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
                    reflection_prompt, reflection_cache_prefix = mkPrompt(query, results, reflection=theory_str, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory)

                    # Submit the LLM call to the executor
                    # Submit the LLM call to the executor.  These read the prefix that the generation
                    # call above wrote to the cache -- same prefix, same key.
                    future = executor.submit(getLLMResponseJSON, reflection_prompt, model_str, temperature, max_tokens, False,
                                             cachePrefixStr=reflection_cache_prefix, cacheKey=cacheKey,
                                             retry_at_lower_effort_if_empty=True, repairJSON=True,
                                             max_generation_time_seconds=MAX_GENERATION_TIME_SECS)
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

            # Save a copy of the theories after the first reflection, and before the second reflection.
            # Should deep copy them.
            try:
                from copy import deepcopy
                theories_after_first_reflection = deepcopy(reflected_theories_general + reflected_theories_specific)
            except Exception as e:
                print("Error deep copying theories after first reflection: " + str(e))
                theories_after_first_reflection = reflected_theories_general + reflected_theories_specific

            record_stage("reflection-1-full-evidence", reflected_theories_general + reflected_theories_specific)

            def collect_theory_relevant_evidence(theory, results):
                # Step 1: Collect all the UUIDs from the theory's supporting evidence and unaccounted-for evidence
                relevant_uuids = set()
                # Crawl the entire structure, looking for any keys called `uuids` or `uuid` and collect them.
                def crawl_structure(structure):
                    if isinstance(structure, dict):
                        for key, value in structure.items():
                            if key == "uuids" and isinstance(value, list):
                                relevant_uuids.update(value)
                            elif key == "uuid" and isinstance(value, str):
                                relevant_uuids.add(value)
                            else:
                                crawl_structure(value)
                    elif isinstance(structure, list):
                        for item in structure:
                            crawl_structure(item)
                # Start crawling from the top-level theory structure
                crawl_structure(theory)

                # Step 2: Filter the results to only include those with UUIDs in relevant_uuids
                filtered_results = []
                for result in results:  # Results should be a list of dictionaries, and each dictionary should have a 'uuid' key.
                    try:
                        if ("uuid" in result) and (result["uuid"] in relevant_uuids):
                            filtered_results.append(result)
                    except Exception as e:
                        print(f"Error processing result {result}: {e}")
                        continue

                print("Filtered results from " + str(len(results)) + " to " + str(len(filtered_results)) + " based on theory's relevant UUIDs.")
                if (len(filtered_results) == 0):
                    print("WARNING: No relevant results found for theory.")
                return filtered_results



            #SECOND_REFLECTION_ENABLED = True
            SECOND_REFLECTION_ENABLED = False
            reflected_theories_general2 = []
            reflected_theories_specific2 = []
            reflected_theories_other2 = []
            if (SECOND_REFLECTION_ENABLED):
                # SECOND REFLECTION: Let's to the reflection again, but a cheaper version: This time we'll filter the list of evidence to only the UUIDs that were mentioned in the first reflection.
                # That should make the call much cheaper, but should also make it a lot easier for the LLM to make the theory more consistent.
                with ThreadPoolExecutor(max_workers=MAX_THEORY_GENERATION_WORKERS) as executor:
                    future_to_theory_idx = {}
                    # All the theories should be in `reflceted_theories_general` and `reflected_theories_specific` now, so we can just iterate over those
                    for idx, theory in enumerate(reflected_theories_general + reflected_theories_specific):
                        # Create a reflection prompt for this theory
                        theory_str = json.dumps(theory, indent=4)
                        filtered_results = collect_theory_relevant_evidence(theory, results)
                        #reflection_prompt, reflection_cache_prefix = mkPrompt(query, results, reflection=theory_str, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory)
                        reflection_prompt, reflection_cache_prefix = mkPrompt(query, filtered_results, reflection=theory_str, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, additional_suggestions=additional_suggestions, max_laws_per_theory=max_laws_per_theory, filtered_reflection=True)

                        # Submit the LLM call to the executor
                        future = executor.submit(getLLMResponseJSON, reflection_prompt, model_str, temperature, max_tokens, False,
                                                cachePrefixStr=reflection_cache_prefix, cacheKey=cacheKey,
                                                retry_at_lower_effort_if_empty=True, repairJSON=True,
                                                max_generation_time_seconds=MAX_GENERATION_TIME_SECS)
                        future_to_theory_idx[future] = (idx, theory)

                    # Process the completed futures
                    from tqdm import tqdm
                    for future in tqdm(as_completed(future_to_theory_idx), total=len(future_to_theory_idx), desc="Second reflection on theories"):
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
                                        reflected_theories_general2.append(theory_candidate)
                                        num_added += 1
                                elif ("theories_specific" in responseJSON) and (len(responseJSON["theories_specific"]) > 0):
                                    for theory_candidate in responseJSON["theories_specific"]:
                                        theory_candidate["reflected_from_theory_index"] = idx
                                        reflected_theories_specific2.append(theory_candidate)
                                        num_added += 1

                                # If we couldn't extract a theory, just use the original one
                                if (num_added == 0):
                                    if (theory["theory_type_general_specific"] == "general"):
                                        theory_candidate["reflected_from_theory_index"] = idx
                                        reflected_theories_general2.append(theory)
                                    elif (theory["theory_type_general_specific"] == "specific"):
                                        theory_candidate["reflected_from_theory_index"] = idx
                                        reflected_theories_specific2.append(theory)
                                    else:
                                        theory_candidate["reflected_from_theory_index"] = idx
                                        reflected_theories_other2.append(theory)
                            else:
                                # If we didn't get a valid response, just use the original one
                                if (theory["theory_type_general_specific"] == "general"):
                                    theory_candidate["reflected_from_theory_index"] = idx
                                    reflected_theories_general2.append(theory)
                                elif (theory["theory_type_general_specific"] == "specific"):
                                    theory_candidate["reflected_from_theory_index"] = idx
                                    reflected_theories_specific2.append(theory)
                                else:
                                    theory_candidate["reflected_from_theory_index"] = idx
                                    reflected_theories_other2.append(theory)

                        except Exception as e:
                            print(f"build_theory_from_results_single_theory_reflection(): ERROR processing theory index {idx} during second reflection: {e}")

                # So we don't break the pipeline that expects there to be only two bins (general and specific), we will just merge any "other" theories into the general bin.
                for theory in reflected_theories_other2:
                    reflected_theories_general2.append(theory)

                # Replace the first reflection with the second reflection, if enabled
                reflected_theories_general = reflected_theories_general2
                reflected_theories_specific = reflected_theories_specific2

                record_stage("reflection-2-cited-evidence", reflected_theories_general + reflected_theories_specific)

            #
            #   Batched-evidence reflection
            #
            # Everything above has only ever seen `results` -- the slice of the corpus that fit in
            # one context window.  When the corpus is 1000+ papers that is a small fraction of it.
            # Each pass below draws a new batch from the records that were left over, shows it to
            # the theory alongside the evidence that theory already cites, and asks it to take the
            # new evidence into account.  Batches are drawn with the same sort-then-random-sample
            # procedure as the original slice, so each is a cross-section of what remains, and a
            # record drawn into one batch is removed from the pool so it cannot appear in another.
            remaining_leftover = list(results_leftover) if (results_leftover is not None) else []
            evidence_seen_so_far = list(results)

            if (len(remaining_leftover) == 0):
                print("### No leftover evidence -- skipping the batched-evidence reflection.", flush=True)

            batch_index = 0
            while ((len(remaining_leftover) > 0) and (batch_index < max_batch_reflections)):
                batch_index += 1
                theories_in_play = reflected_theories_general + reflected_theories_specific
                if (len(theories_in_play) == 0):
                    break

                # How much room the new batch can have.  The evidence each theory already cites has
                # to stay in front of it (that is what the theory rests on), so the batch is sized
                # against the theory that cites the most -- one batch is shared by every theory in
                # the pass, which keeps them comparable and stops the same record being spent twice.
                largest_retained_tokens = 0
                retained_by_theory = {}
                for theory_idx, theory in enumerate(theories_in_play):
                    retained = collect_theory_relevant_evidence(theory, evidence_seen_so_far)
                    retained_by_theory[theory_idx] = retained
                    retained_tokens = countTokens(json.dumps(retained, indent=4))
                    if (retained_tokens > largest_retained_tokens):
                        largest_retained_tokens = retained_tokens

                batch_token_budget = max_evidence_tokens - largest_retained_tokens
                MIN_BATCH_TOKENS = 20000
                if (batch_token_budget < MIN_BATCH_TOKENS):
                    print("### Batch " + str(batch_index) + ": only " + str(batch_token_budget) + " tokens of room left after the cited evidence (" + str(largest_retained_tokens) + " tokens) -- stopping.", flush=True)
                    break

                # Draw the batch, and take it out of the pool
                drawn = consolodate_results_with_subsampling_and_leftover(remaining_leftover, max_tokens=batch_token_budget, rng=rng)
                if (drawn is None) or (len(drawn["data_subsampled"]) == 0):
                    print("### Batch " + str(batch_index) + ": could not draw a batch within " + str(batch_token_budget) + " tokens -- stopping.", flush=True)
                    break
                evidence_batch = drawn["data_subsampled"]
                remaining_leftover = drawn["data_leftover"]

                print("### BATCH REFLECTION " + str(batch_index) + ": " + str(len(evidence_batch)) + " new record(s) (" + str(countTokens(json.dumps(evidence_batch, indent=4))) + " tokens), " + str(len(remaining_leftover)) + " record(s) still unseen", flush=True)

                reflected_theories_general_batch = []
                reflected_theories_specific_batch = []
                results_by_theory_idx = {}

                # Build every prompt for this pass first.  They share a prefix -- the instructions
                # and the new batch, which is the same for every theory here -- and differ only in
                # the theory's own cited evidence below it.
                prompts_by_theory_idx = {}
                shared_cache_prefix = None
                for theory_idx, theory in enumerate(theories_in_play):
                    retained = retained_by_theory[theory_idx]
                    theory_str = json.dumps(theory, indent=4)
                    # The batch was sized against the theory citing the most evidence, so it fits
                    # for every theory in the pass; no further trimming is needed here.  A law may
                    # earn another law's worth of detail once it has more evidence, so the cap is
                    # raised by one for these passes.
                    batch_prompt, batch_prefix = mkPrompt(query, retained, reflection=theory_str,
                                                          include_query_in_generation=include_query_in_generation,
                                                          mission_statement=mission_statement,
                                                          additional_suggestions=additional_suggestions,
                                                          max_laws_per_theory=((max_laws_per_theory + 1) if (max_laws_per_theory is not None) else None),
                                                          filtered_reflection=True,
                                                          new_batch_results=evidence_batch)
                    prompts_by_theory_idx[theory_idx] = batch_prompt
                    shared_cache_prefix = batch_prefix

                # The first call writes that shared prefix to the cache; the rest read it back.  If
                # they all went out at once they would all miss it and each pay for the batch in
                # full, so the first one is run on its own and the others follow together.
                batch_cache_key = None
                if (cacheKey is not None):
                    batch_cache_key = str(cacheKey) + "-batch" + str(batch_index)

                ordered_indexes = sorted(prompts_by_theory_idx.keys())
                if (len(ordered_indexes) > 0):
                    first_idx = ordered_indexes[0]
                    print("### Batch " + str(batch_index) + ": warming the prompt cache with theory " + str(first_idx) + " (" + str(countTokens(shared_cache_prefix)) + " token prefix, shared by " + str(len(ordered_indexes)) + " theories)", flush=True)
                    results_by_theory_idx[first_idx] = getLLMResponseJSON(
                        prompts_by_theory_idx[first_idx], model_str, temperature, max_tokens, False,
                        cachePrefixStr=shared_cache_prefix, cacheKey=batch_cache_key,
                        retry_at_lower_effort_if_empty=True, repairJSON=True,
                        max_generation_time_seconds=MAX_GENERATION_TIME_SECS)

                    with ThreadPoolExecutor(max_workers=MAX_THEORY_GENERATION_WORKERS) as executor:
                        future_to_theory_idx = {}
                        for theory_idx in ordered_indexes[1:]:
                            future = executor.submit(getLLMResponseJSON, prompts_by_theory_idx[theory_idx],
                                                     model_str, temperature, max_tokens, False,
                                                     cachePrefixStr=shared_cache_prefix, cacheKey=batch_cache_key,
                                                     retry_at_lower_effort_if_empty=True, repairJSON=True,
                                                     max_generation_time_seconds=MAX_GENERATION_TIME_SECS)
                            future_to_theory_idx[future] = theory_idx

                        from tqdm import tqdm
                        for future in tqdm(as_completed(future_to_theory_idx), total=len(future_to_theory_idx), desc="Batch " + str(batch_index) + " reflection"):
                            theory_idx = future_to_theory_idx[future]
                            try:
                                results_by_theory_idx[theory_idx] = future.result()
                            except Exception as e:
                                print("build_theory_from_results_single_theory_reflection4b(): ERROR calling the LLM for theory index " + str(theory_idx) + " in batch " + str(batch_index) + ": " + str(e))
                                results_by_theory_idx[theory_idx] = (None, "", 0)

                for theory_idx, theory in enumerate(theories_in_play):
                    theory_type = theory.get("theory_type_general_specific", "general")
                    updated = theory                                # fall back to the theory as it was
                    try:
                        responseJSON_batch, responseText_batch, cost = results_by_theory_idx.get(theory_idx, (None, "", 0))
                        total_cost += cost

                        if (responseJSON_batch is not None):
                            candidates = (responseJSON_batch.get("theories_general") or []) + (responseJSON_batch.get("theories_specific") or [])
                            if (len(candidates) > 0):
                                updated = candidates[0]
                                updated["reflected_from_theory_index"] = theory.get("reflected_from_theory_index", theory_idx)
                                updated["theory_type_general_specific"] = theory_type
                            else:
                                print("build_theory_from_results_single_theory_reflection4b(): WARNING: no theory in the response for index " + str(theory_idx) + " in batch " + str(batch_index) + " -- keeping the previous version.")
                        else:
                            print("build_theory_from_results_single_theory_reflection4b(): WARNING: no response for theory index " + str(theory_idx) + " in batch " + str(batch_index) + " -- keeping the previous version.")

                    except Exception as e:
                        print("build_theory_from_results_single_theory_reflection4b(): ERROR processing theory index " + str(theory_idx) + " in batch " + str(batch_index) + ": " + str(e))

                    if (theory_type == "specific"):
                        reflected_theories_specific_batch.append(updated)
                    else:
                        reflected_theories_general_batch.append(updated)

                reflected_theories_general = reflected_theories_general_batch
                reflected_theories_specific = reflected_theories_specific_batch

                # The batch is now part of what has been seen, so the next pass keeps whatever the
                # theories took from it.
                evidence_seen_so_far = evidence_seen_so_far + evidence_batch

                record_stage("reflection-batch-" + str(batch_index),
                             reflected_theories_general + reflected_theories_specific,
                             extra={"batch_index": batch_index,
                                    "num_records_in_batch": len(evidence_batch),
                                    "num_records_seen_so_far": len(evidence_seen_so_far),
                                    "num_records_still_unseen": len(remaining_leftover)})

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

        print("## PACKING OUTPUT ")

        # Pack output
        packed = {
            "query": query,
            "model": model_str,
            "use_reflection": use_reflection,
            "mission_statement": mission_statement,
            "theory_response": best_response,
            "results": results,
            "raw_responses": all_responses,
            "theories_after_first_reflection": theories_after_first_reflection,
            # A deep copy of the theories at the end of every stage -- generation, each reflection,
            # and each evidence batch -- with summary statistics for each, so how they change as
            # more of the corpus arrives can be measured rather than inferred.
            "theories_by_stage": theories_by_stage,
            "results_leftover": results_leftover,
            "max_evidence_tokens": max_evidence_tokens,
            "max_batch_reflections": max_batch_reflections,
            "total_cost": total_cost,
        }

        return packed

    except Exception as e:
        import traceback
        print("ERROR in build_theory_from_results_single_theory_reflection(): " + str(e))
        traceback.print_exc()
        return None

    except BaseException as e:
        print("FATAL BaseException escaped:", repr(e), flush=True)
        traceback.print_exc()
        raise





# Parametric-only version of the above function
def build_theory_from_results_single_theory_reflection3_llm_baseline_no_evidence(query:str, original_theory_id:str, original_theory_name:str, mission_statement:str=None, provide_matched_control_thery_name:bool=False, model_str="gpt-4o-mini", max_tokens=12000, temperature=0.0, use_reflection:bool=False, include_query_in_generation:bool=True):
    #version_str = "built-theory-from-results-single-theory-reflection3-dec10-2025-LLM-BASELINE-NO-EVIDENCE"
    version_str = "theorizer-parametric-only-accuracy-focused-12102025"
    matched_control_theory_name:str=None

    if (provide_matched_control_thery_name == True):
        version_str += "-with-matched-control-theory-name"
        matched_control_theory_name = original_theory_name
    else:
        version_str += "-without-matched-control-theory-name"


    def mkPrompt(query:str, reflection:str=None, include_query_in_generation:bool=True, matched_control_theory_name:str=None, mission_statement:str=None) -> str:
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"

        prompt += "# Task\n"
        prompt += "Your task is to construct a theory of how a scientific phenomenon works.\n" #, based on a collection of evidence that you will be provided.\n"
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

            if (mission_statement is not None):
                prompt += "Here is the specific mission statement that your theories should be focused on addressing:\n"
                prompt += "```\n"
                prompt += mission_statement + "\n"
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
        prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output, as these are not valid JSON and will cause a parsing error.\n"
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
        prompt += "- You are encouraged to think and plan before writing your JSON response, but your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).  Do not include comments (e.g. // or /* */) or trailing commas in your JSON output.\n"
        prompt += "- Values that are 'none'/'null' in the JSON response should be represented as `null` in the JSON, not as an empty string, string saying \"null\", or any other value.\n"
        prompt += "- Where possible, you should prefer generating new, impactful theories.\n"
        prompt += "- All your information must be accurate. Do not hallucinate.\n"

        return prompt


    total_cost = 0.0

    # Create the prompt
    prompt = mkPrompt(query, include_query_in_generation=include_query_in_generation, matched_control_theory_name=matched_control_theory_name, mission_statement=mission_statement)

    # Call the LLM
    #responseJSON, responseText, cost = def getLLMResponseJSON(promptStr:str, model:str, temperature:float=0, maxTokens:int=DEFAULT_MAX_TOKENS, jsonOut:bool=True):
    all_responses = []
    MAX_GENERATION_TIME_SECS = generation_time_seconds(60 * 8)    # 8 minutes, unless MAX_GENERATION_TIME_SECONDS or the caller says otherwise
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
                reflection_prompt = mkPrompt(query, reflection=theory_str, include_query_in_generation=include_query_in_generation, matched_control_theory_name=matched_control_theory_name, mission_statement=mission_statement)

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

def build_theory_from_results_single_theory_reflection4_nonsafebasin(query:str, results:list, mission_statement:str=None, model_str="gpt-4o-mini", max_tokens=12000, temperature=1.0, use_reflection:bool=False, include_query_in_generation:bool=True, max_laws_per_theory=3):
    #version_str = "built-theory-from-results-single-theory-reflection4-dec27-2025-NONSAFEBASIN"
    version_str = "theorizer-literature-supported-novelty-focused-12272025"

    # This prompt re-cooked from the accuracy-focused objective to generate less conservative theories that break out of the 'safe basin' of prior knowledge. (with GPT-5.2 help)
    def mkPrompt(query: str, results:str, reflection:str=None, include_query_in_generation:bool=True, matched_control_theory_name:str=None, mission_statement:str=None, max_laws_per_theory=None):
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

            if (mission_statement is not None):
                prompt += "Here is a mission statement that provides more detail about the goals of the theories you should generate:\n"
                prompt += "```\n"
                prompt += str(mission_statement) + "\n"
                prompt += "```\n"
                prompt += "\n"

        else:
            prompt += "No specific query is provided. You can build theories as you see fit, based on the supporting evidence.\n"
            prompt += "\n"

        if (max_laws_per_theory is not None):
            prompt += "# Maximum laws per theory\n"
            prompt += "When generating theories, please limit the maximum number of laws per theory to no more than the following number: " + str(max_laws_per_theory) + " .\n"
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
        prompt += "Similarly, the `theory_statements` field is a list of dictionaries, where each dictionary has the following key:\n"
        prompt += "- `law`(dict): A dictionary that describes a single qualitative or quantitative law, in the format described above, and specific supporting evidence that supports the development of this law.\n"
        prompt += "\n"

        prompt += "NOTE: Do not hallucinate. If evidence is absent, label evidence items as background_principle or inference and do not invent studies.\n"
        prompt += "However, you must still comply with Basin-breaking mode: avoid generic safe claims; make committed, falsifiable, discriminative statements.\n"
        prompt += "\n"

        prompt += "# Important Notes\n"
        prompt += "- You must return valid JSON inside a single codeblock (```), or it will not be parsed.\n"
        prompt += "- Use `null` for none values (not empty strings).\n"
        prompt += "- Prefer generating new, impactful theories, but do not fabricate citations.\n"
        if (max_laws_per_theory is not None):
            prompt += "- Limit the maximum number of laws per theory to no more than " + str(max_laws_per_theory) + ".\n"
        prompt += "- All your information must be accurate. Do not hallucinate.\n"

        return prompt



    total_cost = 0.0

    # Create the prompt
    prompt = mkPrompt(query, results, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement)

    # Call the LLM
    all_responses = []
    MAX_GENERATION_TIME_SECS = generation_time_seconds(60 * 8)    # 8 minutes, unless MAX_GENERATION_TIME_SECONDS or the caller says otherwise
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
                reflection_prompt = mkPrompt(query, results, reflection=theory_str, include_query_in_generation=include_query_in_generation, mission_statement=mission_statement, max_laws_per_theory=max_laws_per_theory)

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
    MAX_GENERATION_TIME_SECS = generation_time_seconds(60 * 8)    # 8 minutes, unless MAX_GENERATION_TIME_SECONDS or the caller says otherwise
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
# rng: a random.Random to draw the sample with, so a caller can make the draw repeatable without
# reseeding the global generator.  None uses the global one.
def consolodate_results_with_subsampling(extracted_results_in:list, max_tokens:int=100000, rng=None):
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
        subsampled_data = (rng if (rng is not None) else random).sample(extracted_results_in, num_samples)
        # Convert the subsampled data to a string and count the tokens
        data_str = json.dumps(subsampled_data, indent=4)
        data_tokens = countTokens(data_str)

        if (data_tokens <= max_tokens):
            # If the subsampled data fits within the max_tokens limit, then we're done
            current_data = subsampled_data
            break
        else:
            # Reduce the subsampling rate.  1% steps: the coarse ladder (10%/5%/2%) overshot badly
            # near the boundary -- e.g. 70% came in 0.8% over an 800k budget, so it dropped to 60%
            # and used 691k of the 800k, leaving 14% of the budget (and 47 records) unused.
            current_subsampling_rate -= 0.01

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


# As above, but also returns everything that was NOT sampled, so the caller can keep drawing further
# batches from what is left over.  The sampling procedure is identical (sort by `name_short`, then
# random sample without replacement, stepping the rate down 1% at a time until it fits), so each
# batch drawn this way is a cross-section of the whole set rather than an alphabetical slice of it.
def consolodate_results_with_subsampling_and_leftover(extracted_results_in:list, max_tokens:int=100000, rng=None):
    packed = consolodate_results_with_subsampling(extracted_results_in, max_tokens=max_tokens, rng=rng)
    if (packed is None):
        return None

    # Which records were taken.  Compared by identity, since `random.sample` returns the same
    # objects, and two extracted records can otherwise be equal without being the same record.
    sampled_ids = set()
    for record in packed["data_subsampled"]:
        sampled_ids.add(id(record))

    data_leftover = []
    for record in packed["data_original"]:
        if (id(record) not in sampled_ids):
            data_leftover.append(record)

    packed["data_leftover"] = data_leftover

    print("consolodate_results_with_subsampling_and_leftover(): " + str(len(packed["data_subsampled"])) + " record(s) sampled, " + str(len(data_leftover)) + " left over.")
    return packed
