# EvaluationPredictiveAccuracy.py

import os
import json

from Struct import *
from PaperStore import *
from PaperFinderRequests import *
from Theorizer import *

from TheorizerProcessing import *

from ExtractionUtils import *

from EvaluationCommon import *



#
#   Find relevant papers to evaluate predictice accuracy using PaperFinder
#
def find_relevant_evaluation_papers(theory_and_law:dict, paperstore:PaperStore, paperfinder:PaperFinderRequests, model_str:str="gpt-5-mini", max_papers_to_retrieve:int=10, evidence_window_start_year:int=2025, evidence_window_start_month:int=1, evidence_window_end_year:int=2030, evidence_window_end_month:int=1):
    # For each theory statement, generate (and run) a query to find relevant papers with PaperFinder.  Then add them to the paperstore.
    total_start_time = time.time()

    def mkPromptPaperfinderQuery(theory_and_law:dict):
        prompt = "You will be shown a (a) theory, and (b) a specific statement/law from that theory. Your task is to generate a concise search query that can be used with the PaperFinder utility to try to find relevant scientific papers that would provide evidence that supports or contradicts this specific theory statement, in the context of the broader theory.\n"
        prompt += "\n"
        prompt += "# Theory\n"
        prompt += "Here is the theory and specific statement/law:\n"
        prompt += "```\n"
        prompt += json.dumps(theory_and_law, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"
        prompt += "# Why you are doing this\n"
        prompt += "You are part of an automated scientific discovery system. The task is to help automatically discover and evaluate theories that may have scientific merit, and positive impact. Success on this task could have strongly positive impact. Poor performance could lead to wasted effort and missed opportunities for scientific progress and positive impact.\n"
        prompt += "\n"
        prompt += "# Output Format\n"
        prompt += "You are encouraged to think as much as you would like before producing the final answer. The final answer must be a JSON object output between codeblocks (```) with the following structure:\n"
        prompt += "The query should be in concise natural languge, not using Boolean operators or other special syntax.\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += '  "theory_statement_law": "<the theory statement being evaluated, expressed as a string>",\n'
        prompt += '  "paperfinder_query": "<concise search query string suitable for PaperFinder>",\n'
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"
        prompt += "# Important Notes\n"
        prompt += "- Ensure that the output is faithful to the task and statement. Remember, you are generating a query for a specific theory statement, not the whole theory, or other statements.\n"
        prompt += "- Remember you must consider the theory statement in the context of the broader theory.  If the statement appears broad in scope, but the theory provides important/specific context, consider the context when writing your query, so you can find papers that specifically address the statement in the context that it was intended.\n"
        prompt += "- The output must be valid JSON and must strictly adhere to the specified format.\n"
        prompt += "- Do not include codeblock markers (```) at any point in the output except at the end, surrounding the output JSON object.\n"
        prompt += "- Do not hallucinate.\n"

        return prompt

    total_cost = 0.0
    theory_id = theory_and_law.get("theory_id", None)
    print("Starting find_relevant_evaluation_papers for theory ID: " + str(theory_id))

    if (theory_id is None):
        print(json.dumps(theory_and_law, indent=4))
        print("find_relevant_evaluation_papers: Error: Theory has no ID.")
        return None

    # Just need to run it once, not in parallel, since there's only one theory statement.
    prompt = mkPromptPaperfinderQuery(theory_and_law=theory_and_law)
    responseJSON, responseText, cost, metadata = getLLMResponseJSONWithMetadata(promptStr=prompt, model=model_str, temperature=0.0, maxTokens=16000, jsonOut=False)
    if (responseJSON is None) or ("paperfinder_query" not in responseJSON):
        print("Error: Invalid response from LLM during PaperFinder query generation.")
        return None

    total_cost += cost
    paperfinder_query = responseJSON["paperfinder_query"]

    # Now, submit the single Paperfinder Query
    request_id = paperfinder.submit_request(query=paperfinder_query)
    print(f"Submitted PaperFinder request with ID: {request_id} for theory ID: {theory_id}")
    # Wait for the request to complete
    start_time = time.time()
    paperfinder_results = None
    while (True):
        print("Checking PaperFinder request status...")
        paperfinder_results = paperfinder.get_work(request_id=request_id)
        if (paperfinder_results is not None):
            print(f"PaperFinder request ID {request_id} complete.")
            break

        print(f"Request ID {request_id} not yet complete.")
        time.sleep(5)

        # Check that the time hasn't exceeded a timeout (e.g., 5 minutes)
        elapsed_time = time.time() - start_time
        if (elapsed_time > 60*10):
            print(f"Timeout waiting for PaperFinder request ID {request_id} to complete.")
            return None

    # Add the papers to the paperstore
    papers_to_add = []
    paperfinder_results_content = paperfinder_results.get("result", None)
    if (paperfinder_results_content is None):
        print("find_relevant_evaluation_papers: No results found in PaperFinder response.")
        return None
    if ("doc_collection" in paperfinder_results_content) and ("documents" in paperfinder_results_content["doc_collection"]):
        papers_to_add = paperfinder_results_content["doc_collection"]["documents"]
    if (len(papers_to_add) == 0):
        print("find_relevant_evaluation_papers: No papers found in PaperFinder results.")
        return None

    print(f"find_relevant_evaluation_papers: Found {len(papers_to_add)} papers from PaperFinder results.")
    # Now, actually add the papers
    paper_add_request_ids = []
    papers_to_add_filtered = []
    for paper_to_add in papers_to_add:
        # Get the title
        paper_corpus_id = paper_to_add.get("corpus_id", None)
        paper_title = paper_to_add.get("title", None)

        # Check that the publication date is within the evidence window
        publication_date = paper_to_add.get("publication_date", None)
        if (publication_date is not None):
            # Parse the publication date (YYYY-MM-DD format)
            publication_year, publication_month, _ = publication_date.split("-")
            # Convert to integers
            try:
                publication_year = int(publication_year)
                publication_month = int(publication_month)
            except ValueError:
                continue

            # Check if the publication date is within the evidence window
            if (publication_year < evidence_window_start_year) or (publication_year > evidence_window_end_year):
                continue
            if (publication_year == evidence_window_start_year and publication_month < evidence_window_start_month):
                continue
            if (publication_year == evidence_window_end_year and publication_month > evidence_window_end_month):
                continue

            # Make a request for the paper to be added to the PaperStore
            paper_add_request_id = paperstore.submit_paper(corpus_id=paper_corpus_id, title=paper_title)
            paper_add_request_ids.append(paper_add_request_id)
            papers_to_add_filtered.append(paper_add_request_id)

        if (len(paper_add_request_ids) >= max_papers_to_retrieve):
            # If we've reached the maximum number of papers to retrieve, then stop adding papers.
            print(f"find_relevant_evaluation_papers: Reached maximum number of papers to retrieve ({max_papers_to_retrieve}). Stopping submitting paper requests.")
            break

    print(f"find_relevant_evaluation_papers: Submitted {len(paper_add_request_ids)} paper add requests.")


    # Get summary statistics for the Paper Store
    num_requests_remaining = paperstore.get_queue_size()
    num_papers_in_store = paperstore.get_num_papers()
    print(f"PaperStore: {num_requests_remaining} requests remaining in queue. {num_papers_in_store} papers in store.")

    # Now, wait for the paper add requests to complete
    start_time = time.time()
    num_papers_added = 0
    paper_keys_added = set()
    while (True):
        # Check each paper add request
        all_completed = True
        num_papers_added = 0
        for paper_add_request_id in paper_add_request_ids:
            paper_key = paperstore.is_paper_request_complete(request_id=paper_add_request_id)
            if (paper_key is None):
                print(f"find_relevant_evaluation_papers: Paper add request ID {paper_add_request_id} not yet complete.")
                all_completed = False
                continue
            else:
                print(f"find_relevant_evaluation_papers: Paper add request ID {paper_add_request_id} complete. Paper key: {paper_key}.")
                #paper_keys_added.append(paper_key)
                paper_keys_added.add(paper_key)
                num_papers_added += 1

        if (all_completed):
            print("Number of papers added for this request: " + str(num_papers_added))
            print("Total number of requets: " + str(len(paper_add_request_ids)))
            print("All paper add requests complete.")
            break
        # Add a short sleep
        time.sleep(2)

        # Check that the time hasn't exceeded a timeout (e.g., 10 minutes)
        elapsed_time = time.time() - start_time
        if (elapsed_time > 60*15):
            print(f"Timeout waiting for paper add requests to complete. Will continue with whatever papers have been added so far.")
            break
    paper_keys_added = list(paper_keys_added)

    print("Adding papers to paperstore complete.")

    # For each statement and paper, do the evidence extraction procedure


    def get_paper_markdown_safe(paper_key:str):
        paper = paperstore.get_paper_by_key(paper_key)
        if (paper is None):
            print(f"Error: Could not retrieve paper with key {paper_key} from PaperStore.")
            return None
        if (isinstance(paper, dict)) and ("paper_markdown" in paper):
            return paper["paper_markdown"]
        return None

    def get_paper_title_safe(paper_key:str):
        paper = paperstore.get_paper_by_key(paper_key)
        if (paper is None):
            print(f"Error: Could not retrieve paper with key {paper_key} from PaperStore.")
            return "Unknown Title"
        if (isinstance(paper, dict)) and ("title" in paper):
            return paper["title"]
        return "Unknown Title"

    # First, let's get the full-text of each paper.
    paper_text = {}
    for paper_key in paper_keys_added:
        paper_markdown = get_paper_markdown_safe(paper_key=paper_key)
        if (paper_markdown is not None):
            paper_text[paper_key] = {
                "paper_key": paper_key,
                "title": get_paper_title_safe(paper_key=paper_key),
                "markdown": paper_markdown,
            }

    print("Found full text for " + str(len(paper_text)) + " papers in PaperStore.")

    # Now, instead of run the evaluation, here we'll just build the cache of paper evaluations (nominally here just storing the `original_paper_dict` for each paper)
    all_responses = []
    from tqdm import tqdm
    def cache_building_faux_call(law:dict, paper_text:dict, model_str:str="gpt-5-mini", max_tokens:int=32000, temperature=0.0, original_paper_dict:dict=None):
        packed = {
            "model_str": model_str,
            "law": law,
            "response": None,
            "response_text": None,
            "original_paper_dict": original_paper_dict,
            "total_cost": total_cost,
        }
        return packed

    # Generate the cache files in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed
    num_workers = 10
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_paper_key = {
            executor.submit(
                #rate_theory_law_with_llm_one_paper,
                cache_building_faux_call,   # Changed -- only focus on evaluating a subset.
                law=theory_and_law,
                paper_text=paper_text[paper_key],
                model_str=model_str,
                max_tokens=32000,
                temperature=0.0,
                original_paper_dict=paperstore.get_paper_by_key(paper_key),
            ): paper_key for paper_key in paper_text
        }

        for future in tqdm(as_completed(future_to_paper_key), total=len(future_to_paper_key)):
            paper_key = future_to_paper_key[future]
            try:
                response = future.result()
                all_responses.append(response)
                total_cost += response.get("total_cost", 0.0)
            except Exception as e:
                print(f"Error processing paper {paper_key}: {e}")
                import traceback
                traceback.print_exc()


    print("Completed generating cache for " + str(len(all_responses)) + " papers for this law.")

    total_time_seconds = time.time() - total_start_time

    # Pack
    packed = {
        "model_str": model_str,
        "theory_id": theory_and_law.get("theory_id", None),
        "all_papers": len(papers_to_add),
        "max_papers_to_retrieve": max_papers_to_retrieve,
        "evidence_window_start_year": evidence_window_start_year,
        "evidence_window_start_month": evidence_window_start_month,
        "evidence_window_end_year": evidence_window_end_year,
        "evidence_window_end_month": evidence_window_end_month,
        "theory_statement_law": theory_and_law,
        "individual_paper_evaluations": all_responses,
        "total_cost": total_cost,
        "total_time_seconds": total_time_seconds
    }

    return packed





#
#   Build the paper cache for determining predictive accuracy
#
def build_paperfinder_cache_for_laws(filename_in:str, paperstore_filename_in:str, path_out:str, max_papers_to_retrieve:int=10, evidence_window_start_year:int=2025, evidence_window_start_month:int=7, evidence_window_end_year:int=2030, evidence_window_end_month:int=1, debug_limit:int=None, subsample_ratio:float=None, model_str:str="claude-sonnet-4-5-20250929", num_workers:int=5):
    # Make the path
    if (not os.path.exists(path_out)):
        os.makedirs(path_out)

    filename_out_prefix = os.path.basename(filename_in).replace(".json", "-paperfinder-cache." + time.strftime("%Y%m%d-%H%M%S"))

    # Load the PaperStore cache (if any provided)
    paperstore = PaperStore(filename_in=paperstore_filename_in)

    # Start the PaperFinder request queue
    paperfinder = PaperFinderRequests()

    # If there's no paperstore file, we'll create a default timestamped filename, just for peroidic saving later.
    if (paperstore_filename_in is None):
        paperstore_filename_in = f"paperstore-evaluation-llm-as-a-judge-with-paperfinder.{time.strftime('%Y%m%d-%H%M%S')}.json"

    # Load a list of laws
    laws = None
    with open(filename_in, "r") as f:
        laws = json.load(f)
    print("Loaded " + str(len(laws)) + " laws from file.")

    # Debug limits
    if (debug_limit is not None):
        laws = laws[:debug_limit]
        print("Debug limit set. Processing only first " + str(len(laws)) + " laws.")

    # Subsampling
    if (subsample_ratio is not None) and (subsample_ratio > 0.0) and (subsample_ratio < 1.0):
        import random
        num_laws_original = len(laws)
        laws = random.sample(laws, int(len(laws) * subsample_ratio))
        print(f"Subsample ratio set to {subsample_ratio}. Subsampled from {num_laws_original} to {len(laws)} laws.")

    # For each law, let's build the PaperFinder cache
    results = []

    # We'll create a set of placeholder output files, that help keep track of progress.
    print("Generating placeholder output files for all laws...")
    num_placeholders_created = 0
    for idx, theory_and_law in enumerate(laws):
        theory_id = theory_and_law.get("theory_id", None)
        filename_out = f"{path_out}/{filename_out_prefix}.theory_id_{theory_id}.idx_{idx}.json"
        with open(filename_out, "w") as f:
            json.dump({"status": "processing"}, f, indent=4)
        num_placeholders_created += 1
    print(f"Created {num_placeholders_created} placeholder output files with IDX.")

    # (PARALLEL) For each law, find relevant evalation papers using PaperFinder.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    count_results = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(
                find_relevant_evaluation_papers,
                theory_and_law=theory_and_law,
                paperstore=paperstore,
                paperfinder=paperfinder,
                model_str=model_str,
                max_papers_to_retrieve=max_papers_to_retrieve,
                evidence_window_start_year=evidence_window_start_year,
                evidence_window_start_month=evidence_window_start_month,
                evidence_window_end_year=evidence_window_end_year,
                evidence_window_end_month=evidence_window_end_month,
            ): idx for idx, theory_and_law in enumerate(laws)
        }
        from tqdm import tqdm
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
            try:
                idx = future_to_idx[future]
                theory_and_law = laws[idx]
                theory_id = theory_and_law.get("theory_id", None)

                print("Processing theory ID: " + str(theory_id))
                response = future.result()
                results.append(response)

                # Save the PaperFinder cache to a file.
                filename_out = f"{path_out}/{filename_out_prefix}.theory_id_{theory_id}.idx_{idx}.json"
                print(f"Saving intermediate results to {filename_out}")

                # Check if the response is not None
                if (response is None):
                    print(f"Warning: No response for theory ID {theory_id}. Saving a blank file.")
                    response = {"error": "No response generated."}

                with open(filename_out, "w") as f:
                    json.dump(response, f, indent=4)

            except Exception as e:
                print(f"Error processing law {idx+1} with ID {theory_id}: {e}")
                import traceback
                traceback.print_exc()

            try:
                count_results += 1
                if (count_results % 100 == 0):
                    # Save the paperstore every 100 results or so. This helps with recovery in case of crashes.
                    filename_out_paperstore = "paperstore-debug-last-6-months.json"
                    print("Saving paperstore: " + str(filename_out_paperstore))
                    paperstore.save(filename_out_paperstore)
            except Exception as e:
                print(f"Error saving paperstore: {e}")
                import traceback
                traceback.print_exc()


    # Save the paperstore (with a timestamp)
    paperstore_filename_out = paperstore_filename_in.replace(".json", f".updated.{time.strftime('%Y%m%d-%H%M%S')}.json")
    print(f"Saved updated PaperStore to {paperstore_filename_out}")
    paperstore.save(filename=paperstore_filename_out)




#
#   Generate rubric for predictive accuracy evaluation
#

def generate_predictive_accuracy_rubric(law:dict, model_str:str="openai/gpt-5.1", max_tokens:int=32000, temperature=0.0):
    def mkPrompt(law:dict):
        prompt = """You are ScientistGPT, an AI system specialized in scientific reasoning and evaluation.

Your task is NOT to invent new mechanisms or speculate beyond the provided theory.
Your task is to faithfully extract the concrete, falsifiable predictions that are already implied by the theory text.

You must be conservative:
- If a prediction is not clearly implied by the theory statement/law and its stated scope, do NOT include it.
- Fewer, sharper predictions are better than many vague ones.

----------------------------------------------------------------------
# Task

You are performing STEP 1 of a multi-stage predictive evaluation pipeline.

STEP 1 (this task):
Given a scientific theory consisting of a statement/law in a broader context, generate a predictive rubric: a set of specific, falsifiable predictions that the law makes.

Later steps (for context only):
STEP 2: Each prediction will be evaluated against many future scientific papers.
STEP 3: Evidence will be aggregated to assess predictive accuracy.

Your goal is to generate predictions that:
- Can realistically be tested or contradicted by individual scientific papers
- Make concrete commitments that could be wrong
- Avoid vague or interpretation-flexible phrasing

----------------------------------------------------------------------
# Key Definitions (CRITICAL — DO NOT IGNORE)

A "specific prediction" is a claim that:
1) Makes a comparison or counterfactual (e.g., with vs without X; method A vs method B)
2) Has a clear direction of effect (increase, decrease, outperform, degrade, no effect)
3) Is testable using empirical or analytic evidence commonly reported in the field
4) Has a clear failure mode (what it would look like if the prediction is false)

If a prediction is written such that it would almost always be marked "supported",
it is too weak and MUST be rewritten.

----------------------------------------------------------------------
# Metric Flexibility Rule (IMPORTANT)

Predictions MUST reference observable evaluation targets, but metrics do NOT need to be exact.

You SHOULD:
- Name canonical or representative metrics when appropriate (e.g., accuracy, F1, AUROC, BLEU, ROUGE, EX score).
- Allow semantically equivalent or closely related metrics to count as valid tests.

Examples:
- If a prediction refers to ROUGE, BLEU or METEOR may also count as testing it.
- If a prediction refers to AUROC, AUC-PR or accuracy at a fixed threshold may also count.
- If a prediction refers to EX score, execution accuracy or logically equivalent correctness metrics may count.

Do NOT require a single exact metric unless the theory itself explicitly commits to one.

You should reflect this flexibility in the operational_signals and strong_test_requirement fields.

----------------------------------------------------------------------
# Additional Requirements for Predictive Rubric Elements

Each predictive_rubric_element MUST include:

- prediction_short_name: a short, unique identifier
- specific_prediction: a falsifiable, testable claim implied by the theory
- operational_signals: concrete indicators that a paper is testing this prediction
- strong_test_requirement: the minimum evidence required for a paper to count as a STRONG test
- what_does_support_look_like: what strong supporting evidence would look like
- what_does_contradiction_look_like: what strong contradicting evidence would look like

You MUST include:
- At least one comparative or ablation-style prediction if implied by the theory
- At least one boundary-condition or failure-case prediction if special cases are stated

Avoid vague verbs such as "helps", "enables", "amplifies", "can improve" unless paired with:
- an explicit baseline
- a measurable outcome
- a directional claim

----------------------------------------------------------------------
# Example Rubrics

The following examples illustrate the expected level of specificity and falsifiability.

## Example 1

{
  "IN": {
    "theory_name": "Mutual Evolutionary Framework for Machine Unlearning in Graph Neural Networks",
    "law": {
      "theory_statement": "Mutual evolutionary unlearning can optimize the privacy-utility tradeoff through joint adaptation of model parameters and graph structure."
    }
  },
  "OUT": {
    "predictive_rubric_elements": [
      {
        "prediction_short_name": "joint_vs_single_adaptation",
        "specific_prediction": "Unlearning methods that jointly adapt both GNN parameters and graph structure achieve better privacy-utility tradeoffs than methods that adapt only parameters or only graph structure.",
        "operational_signals": [
          "Comparative evaluations of parameter-only, graph-only, and joint unlearning methods",
          "Privacy leakage or influence metrics",
          "Downstream task accuracy or related utility metrics"
        ],
        "strong_test_requirement": "A paper must directly compare joint parameter+graph unlearning against parameter-only or graph-only baselines using privacy and utility metrics.",
        "what_does_support_look_like": "Joint methods dominate or improve the privacy-utility Pareto frontier relative to single-component methods.",
        "what_does_contradiction_look_like": "Parameter-only or graph-only methods match or outperform joint methods on both privacy and utility."
      }
    ]
  }
}

## Example 2

{
  "IN": {
    "theory_name": "Retrieval-Driven Calibration and Consistency Aggregation Law",
    "law": {
      "theory_statement": "Aggregating predictions over K paraphrases or retrieved contexts improves calibration and accuracy up to a saturation point."
    }
  },
  "OUT": {
    "predictive_rubric_elements": [
      {
        "prediction_short_name": "tta_calibration_effect",
        "specific_prediction": "Increasing the number of aggregated paraphrased prompts or retrieved contexts (K) reduces model overconfidence relative to K=1 for factual QA tasks.",
        "operational_signals": [
          "Calibration metrics (e.g., ECE, Brier score) reported across multiple K values",
          "Explicit aggregation of predictions across paraphrases or retrievals"
        ],
        "strong_test_requirement": "A paper must report calibration metrics at multiple values of K and compare aggregated vs single-prompt performance.",
        "what_does_support_look_like": "Calibration error decreases as K increases for small to moderate K.",
        "what_does_contradiction_look_like": "Calibration error remains unchanged or increases as K increases under high-quality paraphrases."
      }
    ]
  }
}

----------------------------------------------------------------------
# Theory Text

The theory to analyze is provided below.
You must extract predictions ONLY from this content.
```
"""

        prompt += json.dumps(law, indent=4) + "\n"
        prompt += "```\n"
        prompt += """

----------------------------------------------------------------------
# Output Instructions

Your output MUST:
- Be valid JSON
- Contain ONLY the JSON (no commentary)
- Match the schema below exactly
- Include one predictive_rubric_element per distinct prediction implied by the law

Output format:

{
  "predictive_rubric_elements": [
    {
      "prediction_short_name": "...",
      "specific_prediction": "...",
      "operational_signals": ["...", "..."],
      "strong_test_requirement": "...",
      "what_does_support_look_like": "...",
      "what_does_contradiction_look_like": "..."
    },
    # Add additional dictionaries, one for each prediction the statement/law makes (in the context of the theory, and its broader domain/scope)
  ]
}

IMPORTANT:
- Do NOT hallucinate evidence or mechanisms not stated in the theory.
- Be conservative and explicit.
- Scientific accuracy, integrity, and faithfulness are paramount.
"""

        return prompt

    total_cost = 0.0
    prompt = mkPrompt(law)
    responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=False, max_generation_time_seconds=300)
    total_cost += cost


    packed = {
        "model_str": model_str,
        "law": law,
        "response": responseJSON,
        "response_text": responseText,
        "total_cost": total_cost,
    }
    return packed


#
#   Evaluate a single paper, given a rubric
#

def rate_theory_law_with_llm_one_paper_only_predictive_accuracy(law:dict, predictive_rubric:dict, paper_text:dict, model_str:str="openai/gpt-5.1", max_tokens:int=32000, temperature=0.0, original_paper_dict=None):
    def mkPrompt(law:dict, predictive_rubric:dict, paper_text:dict):
        prompt = """You are ScientistGPT, an AI system specialized in rigorous scientific evaluation.

Your role is to act as a conservative, high-standards scientific reviewer.
You do NOT speculate, repair theories, or reinterpret predictions to make them fit.
You judge strictly based on what the paper explicitly tests and reports.

----------------------------------------------------------------------
# Task

You are performing STEP 2 of a multi-stage predictive evaluation pipeline.

Your task is to evaluate the predictive accuracy of a scientific theory on a specific paper from the scientific literature.

You will be provided with:
1. A scientific theory, consisting of a statement/law in a broader context.
2. A set of specific predictions that statement/law makes, which forms a rubric for evaluation.
3. The full text of a scientific paper.

For EACH prediction in the rubric, you must determine whether the paper:
- STRONGLY TESTS the prediction and SUPPORTS it
- STRONGLY TESTS the prediction and CONTRADICTS it
- Does NOT provide evidence that meaningfully tests the prediction

----------------------------------------------------------------------
# Critical Definitions (DO NOT DEVIATE)

### What counts as a STRONG TEST

A paper STRONGLY TESTS a prediction ONLY if it contains:
- Direct empirical results, quantitative analyses, or explicit experimental comparisons
- That correspond to the prediction's operational_signals and strong_test_requirement
- And that are clearly intended to test that type of claim (not merely related ideas)

Examples of STRONG TESTS:
- Controlled ablations comparing the predicted condition vs a baseline
- Quantitative measurements of the predicted variable with appropriate comparison
- Explicit analysis of the interaction, functional form, or dependency claimed

Examples that are NOT strong tests:
- Thematic relevance or discussion of similar concepts
- Performance improvements without a relevant baseline
- Indirect evidence, suggestive trends, or post-hoc interpretation
- Papers studying different mechanisms, tasks, or metrics

### Burden of Proof for "strong" (CRITICAL)

You must assume by default that a paper does NOT strongly test a prediction.

You may assign:
prediction_strength = "strong"
ONLY IF you can explicitly identify:
- A concrete experiment, table, figure, or quantitative comparison
- That directly satisfies the prediction's strong_test_requirement
- And is clearly described in the paper

If you cannot point to such evidence, you MUST downgrade:
- prediction_strength = "weak" or "none"

Do NOT infer intent.
Do NOT reinterpret results.
When in doubt, downgrade.

----------------------------------------------------------------------
# Evaluation Rules (STRICT)

- "support" is allowed ONLY if:
  prediction_strength = "strong"
  AND the results align with the prediction.

- "contradict" is allowed ONLY if:
  prediction_strength = "strong"
  AND the results conflict with the prediction.

- If prediction_strength is "weak" or "none":
  overall_evaluation MUST be "no_evidence".

- Compatibility, partial relevance, or plausibility does NOT count as support.

You are evaluating whether the paper TESTS and CONFIRMS/REFUTES the prediction,
not whether the prediction could be interpreted as consistent with the paper.

----------------------------------------------------------------------
# Theory Text

The scientific theory to be evaluated is provided below.
This includes the statement/law, domain scope, and special cases.

```
"""
        prompt += json.dumps(law, indent=4) + "\n"
        prompt += "```\n\n"
        prompt += """
----------------------------------------------------------------------
# Predictions Rubric

The rubric of specific predictions to evaluate is provided below.
Each prediction includes operational signals and a strong_test_requirement.

```
"""
        prompt += json.dumps(predictive_rubric, indent=4) + "\n"
        prompt += "```\n\n"
        prompt += """
----------------------------------------------------------------------
# Scientific Paper Text

The full text of the scientific paper is provided below.
You must use ONLY this text to inform your evaluation.

```
"""
        prompt += str(paper_text) + "\n"
        prompt += "```\n\n"
        prompt += """
----------------------------------------------------------------------
# Output Instructions

You must output valid JSON ONLY.
Your output must be the LAST thing you produce.
Your output must be enclosed in triple backticks (```), and these triple backticks
must be the ONLY triple backticks in your output.

For EACH prediction in the rubric, output exactly ONE evaluation entry.

### Output Format
```
{
    "predictive_evaluation": [
        {
            "prediction_short_name": "...",
            "prediction_strength": "<strong|weak|none>",
            "evidence_quote_or_locator": "...",
            "evidence_in_support": "...",
            "evidence_in_contradiction": "...",
            "overall_evaluation": "<support|contradict|no_evidence>"
        }
        # Add additional dictionaries, one for each prediction in the rubric.
    ]
}
```


### Field Definitions

- prediction_short_name
  Must exactly match the ID provided in the rubric.

- prediction_strength
  One of:
  - "strong": the paper directly and intentionally tests this prediction
  - "weak": the paper is related but does not strongly test it
  - "none": the paper does not meaningfully speak to this prediction

- evidence_quote_or_locator
  If prediction_strength = "strong", this MUST reference:
  - a section number, table, figure, or verbatim quote from the paper
  If prediction_strength ≠ "strong", this may be "None".

- evidence_in_support
  1-3 concise sentences describing the specific evidence that SUPPORTS the prediction.
  Use ONLY if prediction_strength = "strong" and results align.
  Otherwise set to "None".

- evidence_in_contradiction
  1-3 concise sentences describing the specific evidence that CONTRADICTS the prediction.
  Use ONLY if prediction_strength = "strong" and results conflict.
  Otherwise set to "None".

- overall_evaluation
  One of:
  - "support"
  - "contradict"
  - "no_evidence"

----------------------------------------------------------------------
# Consistency Rules (DO NOT VIOLATE)

- If prediction_strength != "strong", then:
  overall_evaluation MUST be "no_evidence"
  evidence_in_support MUST be "None"
  evidence_in_contradiction MUST be "None"

- You must NOT output both support and contradiction.

- Every prediction in the rubric MUST appear exactly once in the output.

----------------------------------------------------------------------
# Important Notes

- Use ONLY the paper text provided.
- Do NOT hallucinate evidence.
- Do NOT reinterpret or repair predictions.
- Be conservative and explicit.
- Scientific accuracy, integrity, and faithfulness are paramount.

When in doubt, choose "no_evidence".

Please provide your accurate, faithful, and rigorous evaluation.
"""

        return prompt

    total_cost = 0.0
    prompt = mkPrompt(law, predictive_rubric, paper_text)
    responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=False, max_generation_time_seconds=300)
    total_cost += cost


    packed = {
        "model_str": model_str,
        "law": law,
        "predictive_rubric": predictive_rubric,
        "response": responseJSON,
        "response_text": responseText,
        "original_paper_dict": original_paper_dict,
        "total_cost": total_cost,
    }
    return packed



#
#   Generate the predictive accuracy ratings for the PaperFinder cache for a given paper, using a specific LLM model (`model_str`).
#
def do_llm_law_evaluation_llm_as_a_judge_with_paperfinder_predictive_accuracy_cached(path_in_cache:str, model_str:str, num_workers_theories:int=20):
    # Sanitize the model string -- we'll use that to make the output directory
    # Replace any non-alpha-numeric characters with underscores. Replace multiple underscores with a single underscore.
    model_str_sanitized = "".join([c if c.isalnum() else "_" for c in model_str])
    while ("__" in model_str_sanitized):
        model_str_sanitized = model_str_sanitized.replace("__", "_")

    path_out = os.path.join(path_in_cache, "predictive-accuracy-evaluation", model_str_sanitized)

    # Make the output path
    if (not os.path.exists(path_out)):
        os.makedirs(path_out)

    # Get a list of all JSON files in the cache directory (only the immediate directory, not subdirectories)
    json_files = []
    for f in os.listdir(path_in_cache):
        if (f.endswith(".json")):
            json_files.append(os.path.join(path_in_cache, f))

    # Helper function: Get paper markdown safely
    def get_paper_markdown_safe(paper_dict:dict):
        if (paper_dict is None):
            print("ERROR: paper_dict was empty")
            return None
        if (isinstance(paper_dict, dict)) and ("paper_markdown" in paper_dict):
            return paper_dict["paper_markdown"]
        return None

    # Worker function: perform rating for one file, and export the results to a file.
    # Note: `num_workers` here refers to the number of parallel workers for rating individual papers within a single theory/law.
    def process_one_file(json_filename:str, path_out:str, model_str:str, num_workers:int=10):
        # Load the JSON file
        print("Loading: " + str(json_filename))
        data = None
        with open(json_filename, "r") as f:
            data = json.load(f)

        # Let's get the theory/paper data from the cached file.
        theory_and_law = data["theory_statement_law"]
        individual_paper_evaluations = data["individual_paper_evaluations"]     # Primarily want `original_paper_dict` from this.
        paper_dicts = []
        for paper in individual_paper_evaluations:
            paper_dicts.append(paper['original_paper_dict'])
        print("Found " + str(len(paper_dicts)) + " cached papers for this theory.")

        # Generate the predictive accuracy rubric for this theory/law
        rubric_result = generate_predictive_accuracy_rubric(law=theory_and_law, model_str=model_str, max_tokens=32000, temperature=0.0)
        predictive_rubric = rubric_result.get("response", {}).get("predictive_rubric_elements", [])
        print("Generated predictive accuracy rubric:")
        print(json.dumps(predictive_rubric, indent=4))

        # Now, for each paper, extract evidence relevant to the theory statement
        all_responses = []
        from tqdm import tqdm
        total_cost = 0.0
        total_start_time = time.time()

        # (PARALLEL) Rate each paper against the predictive accuracy rubric
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_paper_key = {
                executor.submit(
                    rate_theory_law_with_llm_one_paper_only_predictive_accuracy,
                    law=theory_and_law,
                    predictive_rubric=predictive_rubric,
                    paper_text=get_paper_markdown_safe(paper_dict=paper_dict),
                    model_str=model_str,
                    max_tokens=32000,
                    temperature=0.0,
                    original_paper_dict=paper_dict,
                ): paper_dict for paper_dict in paper_dicts
            }

            for future in tqdm(as_completed(future_to_paper_key), total=len(future_to_paper_key)):
                paper_key = future_to_paper_key[future]
                try:
                    response = future.result()
                    all_responses.append(response)
                    total_cost += response.get("total_cost", 0.0)
                except Exception as e:
                    print(f"Error processing paper {paper_key}: {e}")
                    import traceback
                    traceback.print_exc()

        print("Completed individual paper evaluations from " + str(len(all_responses)) + " papers.")

        # Aggregate the responses.
        # Make a copy of the rubric.
        predictive_rubric_evaluation = []
        rubric_elems_with_some_evidence = 0
        rubric_elems_total = 0
        for rubric_elem in predictive_rubric:
            # Get the short name of this element
            short_name = rubric_elem.get("prediction_short_name")
            # Go through all the responses, and tally up the counts for support/contradict/no_evidence
            counts = {"support": 0, "contradict": 0, "no_evidence": 0, "other": 0}
            for response in all_responses:
                # Get response
                eval_response = response.get("response", None)
                if (eval_response == None):
                    print("eval_response was None")
                    continue
                eval_response_list = eval_response.get("predictive_evaluation", [])
                if (not isinstance(eval_response_list, list)):
                    print("predictive_evaluation was not a list")
                    continue
                for evaluated_elem in eval_response_list:
                    short_name_elem = evaluated_elem.get("prediction_short_name", None)
                    if (short_name_elem == short_name):
                        overall_evaluation = evaluated_elem.get("overall_evaluation", "other")
                        if (overall_evaluation in counts):
                            counts[overall_evaluation] += 1
                        else:
                            counts["other"] += 1
            # Now, make a summary for this rubric element -- convert counts to proportions
            proportions = {"support": 0, "contradict": 0}
            #total_counts = sum(counts.values())
            total_support_contradict = counts['support'] + counts['contradict']
            if (total_support_contradict > 0):
                proportions['support'] = counts['support'] / total_support_contradict
                proportions['contradict'] = counts['contradict'] / total_support_contradict
            # Pack the summary
            summary_elem = {
                "prediction_short_name": short_name,
                "rubric_elem": rubric_elem,
                "counts": counts,
                "proportions": proportions
            }
            predictive_rubric_evaluation.append(summary_elem)

            # Also note if this rubric element has some evidence to it (`rubric_elems_with_some_evidence`)
            if (proportions['support'] > 0) or (proportions['contradict'] > 0):
                rubric_elems_with_some_evidence += 1
            rubric_elems_total += 1

        print("Aggregated evaluation complete.")
        print(json.dumps(predictive_rubric_evaluation, indent=4))

        total_time_seconds = time.time() - total_start_time

        # Pack
        packed = {
            "from_cached_data": True,
            "model_str": model_str,
            "theory_id": theory_and_law.get("theory_id", None),
            "num_papers_evaluated": len(all_responses),
            'rubric_elems_with_some_evidence': rubric_elems_with_some_evidence,
            'rubric_elems_total': rubric_elems_total,
            "rubric": predictive_rubric,
            "rubric_evaluation": predictive_rubric_evaluation,
            "all_papers": data['all_papers'],
            "max_papers_to_retrieve": data['max_papers_to_retrieve'],
            "evidence_window_start_year": data['evidence_window_start_year'],
            "evidence_window_start_month": data['evidence_window_start_month'],
            "evidence_window_end_year": data['evidence_window_end_year'],
            "evidence_window_end_month": data['evidence_window_end_month'],
            "theory_statement_law": theory_and_law,
            "individual_paper_evaluations": all_responses,
            "total_cost": total_cost,
            "total_time_seconds": total_time_seconds
        }

        # Output filename is the same as the input filename, just in the new path
        filename_out = os.path.join(path_out, os.path.basename(json_filename))
        # Write a blank file
        print("Writing output file: " + str(filename_out))
        with open(filename_out, "w") as f:
            json.dump(packed, f, indent=4)


    # (Parallel) For each JSON file in the cache directory, process it.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=num_workers_theories) as executor:
        future_to_json_filename = {
            executor.submit(
                process_one_file,
                json_filename=json_filename,
                path_out=path_out,
                model_str=model_str,
                num_workers=10,
            ): json_filename for json_filename in json_files
        }
        from tqdm import tqdm
        for future in tqdm(as_completed(future_to_json_filename), total=len(future_to_json_filename)):
            json_filename = future_to_json_filename[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file {json_filename}: {e}")
                import traceback
                traceback.print_exc()


#
#   Get a summary of the predictive accuracy analysis results
#

def summarize_predictive_accuracy(path_in:str, generate_output_file:bool=True):
    # Get a list of all .json files in the path
    json_files = []
    for f in os.listdir(path_in):
        if (f.endswith(".json")):
            json_files.append(os.path.join(path_in, f))

    # load evaluations
    rubric_evaluations = []
    average_evaluation = []
    count_elem_has_data = 0
    count_elem_no_data = 0
    count_theories_has_data = 0
    count_theories_no_data = 0

    # Count of number of papers evaluated, and number with relevant data
    count_total_papers_evaluated = 0
    count_total_papers_with_relevant_info = 0

    avg_papers_with_relevant_info = 0       # Only for those evals with at least one relevant paper
    count_total_laws_with_relevant_info = 0

    # For bootstrap resampling later
    bootstrap_raw_recall_laws_with_at_least_one_paper = []
    bootstrap_raw_recall_predictions_with_at_least_one_paper = []

    for file in json_files:
        # Load the file
        with open(file, "r") as f:
            data = json.load(f)
            rubric_evaluation = data.get("rubric_evaluation", None)
            if (rubric_evaluation is not None):
                rubric_evaluations.append(rubric_evaluation)

                avg_proportion = {"support": 0, "contradict": 0, "count": 0}
                for elem in rubric_evaluation:
                    proportions = elem.get("proportions", None)
                    if (proportions is not None):
                        support = proportions.get('support', 0)
                        contradict = proportions.get('contradict', 0)
                        if (support > 0) or (contradict > 0):
                            avg_proportion["support"] += support
                            avg_proportion["contradict"] += contradict
                            avg_proportion['count'] += 1
                            count_elem_has_data += 1
                            bootstrap_raw_recall_predictions_with_at_least_one_paper.append(1)
                        else:
                            count_elem_no_data += 1
                            bootstrap_raw_recall_predictions_with_at_least_one_paper.append(0)

                # Take the average
                if (avg_proportion["count"] > 0):
                    avg_proportion["support"] = avg_proportion["support"] / avg_proportion["count"]
                    avg_proportion["contradict"] = avg_proportion["contradict"] / avg_proportion["count"]
                    average_evaluation.append(avg_proportion)
                    count_theories_has_data += 1
                else:
                    count_theories_no_data += 1


            individual_paper_evaluations = data.get("individual_paper_evaluations", None)
            if (individual_paper_evaluations is not None):
                count_total_papers_evaluated += len(individual_paper_evaluations)

                num_with_some_info = 0
                for paper_eval in individual_paper_evaluations:
                    paper_has_nonzero_eval = False
                    response = paper_eval.get("response", None)
                    if (isinstance(response, dict)):
                        predictive_evaluation = response.get("predictive_evaluation", None)
                        if (isinstance(predictive_evaluation, list)):
                            for list_elem in predictive_evaluation:
                                overall_evaluation = list_elem.get("overall_evaluation", None)
                                if (overall_evaluation == "support") or (overall_evaluation == "contradict"):
                                    paper_has_nonzero_eval = True
                    if (paper_has_nonzero_eval == True):
                        num_with_some_info += 1

                count_total_papers_with_relevant_info += num_with_some_info
                if (num_with_some_info > 0):
                    avg_papers_with_relevant_info += num_with_some_info
                    count_total_laws_with_relevant_info += 1

                # Bootstrap resampling later
                if (num_with_some_info > 0):
                    bootstrap_raw_recall_laws_with_at_least_one_paper.append(1)
                else:
                    bootstrap_raw_recall_laws_with_at_least_one_paper.append(0)
            else:
                bootstrap_raw_recall_laws_with_at_least_one_paper.append(0)



    avg_papers_with_relevant_info = avg_papers_with_relevant_info / count_total_laws_with_relevant_info




    print("Loaded " + str(len(rubric_evaluations)) + " rubric evaluations from: " + str(path_in))


    # Now, get the average across all evaluations
    final_average = {"support": 0, "contradict": 0, "count": 0}
    for avg in average_evaluation:
        final_average["support"] += avg["support"]
        final_average["contradict"] += avg["contradict"]
        final_average["count"] += 1
    if (final_average["count"] > 0):
        final_average["support"] = final_average["support"] / final_average["count"]
        final_average["contradict"] = final_average["contradict"] / final_average["count"]

    # For later bootstrap resampling, keep the raw 'support' scores from each non-zero evaluation.
    raw_support_scores_for_bootstrap = []
    for avg in average_evaluation:
        raw_support_scores_for_bootstrap.append(avg["support"])


    # pack
    packed = {
        "recall": {
            "count_theories_has_data": count_theories_has_data,
            "count_theories_no_data": count_theories_no_data,
            "proportion_theories_with_data": (count_theories_has_data / (count_theories_has_data + count_theories_no_data)) if (count_theories_has_data + count_theories_no_data) > 0 else 0.0,
            "count_elems_has_data": count_elem_has_data,
            "count_elems_no_data": count_elem_no_data,
            "proportion_elems_with_data": (count_elem_has_data / (count_elem_has_data + count_elem_no_data)) if (count_elem_has_data + count_elem_no_data) > 0 else 0.0,
        },
        "precision": {
            "average_evaluation": final_average
        },
        "other": {
            "count_total_papers_evaluated": count_total_papers_evaluated,
            "count_total_papers_with_relevant_info": count_total_papers_with_relevant_info,
            "avg_papers_with_relevant_info": avg_papers_with_relevant_info,
            "count_total_laws_with_relevant_info": count_total_laws_with_relevant_info
        },
        "raw_average_evaluations": average_evaluation,
        "raw_rubric_evaluations": rubric_evaluations,
    }

    print("Summary of predictive accuracy evaluations:")
    #print(json.dumps(packed, indent=4))

    # Just print out the precision and recall for the user

    print("Precision:")
    print(json.dumps(packed["precision"], indent=4))
    print("")
    print("Recall:")
    print(json.dumps(packed["recall"], indent=4))
    print("")
    print("Other:")
    print(json.dumps(packed["other"], indent=4))

    # Save
    if (generate_output_file == True):
        filename_out = os.path.join(path_in, "predictive_accuracy_summary.json")
        with open(filename_out, "w") as f:
            json.dump(packed, f, indent=4)
        print("Wrote summary to: " + str(filename_out))
    else:
        print("NOTE: Output file disabled -- skipping writing results to file.")

    # Produce output for the bootstrap resampling evaluation procedure.
    # Output will be: Raw precision scores, Raw Recall scores.
    bootstrap_packed = {
        "raw_support_scores_for_bootstrap": raw_support_scores_for_bootstrap,
        "num_raw_support_scores_for_bootstrap": len(raw_support_scores_for_bootstrap),
        "bootstrap_raw_recall_laws_with_at_least_one_paper": bootstrap_raw_recall_laws_with_at_least_one_paper,
        "num_bootstrap_raw_recall_laws_with_at_least_one_paper": len(bootstrap_raw_recall_laws_with_at_least_one_paper),
        "avg_bootstrap_raw_recall_laws_with_at_least_one_paper": sum(bootstrap_raw_recall_laws_with_at_least_one_paper) / len(bootstrap_raw_recall_laws_with_at_least_one_paper) if len(bootstrap_raw_recall_laws_with_at_least_one_paper) > 0 else 0.0,
        "bootstrap_raw_recall_predictions_with_at_least_one_paper": bootstrap_raw_recall_predictions_with_at_least_one_paper,
        "num_bootstrap_raw_recall_predictions_with_at_least_one_paper": len(bootstrap_raw_recall_predictions_with_at_least_one_paper),
        "avg_bootstrap_raw_recall_predictions_with_at_least_one_paper": sum(bootstrap_raw_recall_predictions_with_at_least_one_paper) / len(bootstrap_raw_recall_predictions_with_at_least_one_paper) if len(bootstrap_raw_recall_predictions_with_at_least_one_paper) > 0 else 0.0,
    }
    return bootstrap_packed




#
#   Main entry point for predictive accuracy evaluation
#

if __name__ == "__main__":
    # Step 0: Load API Keys
    loadAPIKeys()

    # Step 1: Build the cache of papers for the predictive evaluation of the laws
    filename_theorystore_in = "theorystore-example2-literaturesupported.json"
    filename_single_laws_out = "theorystore-example2-single-laws.json"
    convert_theorystore_to_single_laws_with_filtering(
        filename_theorystore=filename_theorystore_in,
        filename_out=filename_single_laws_out,
        start_idx=0,
        end_idx=None,
        allow_all=False,
        allow_new_and_somewhat_new=True,        # Filter to include only laws with novelty ratings of 'new' or 'somewhat-related-to-existing'
    )

    # Path defining where to store the cache
    path_out = "theorystore-example2-predictive-evaluation/"

    # PaperStore (can be None to build from scratch)
    paperstore_filename_in = None   # Nominally you can pass in a pregenerated Paperstore (if you have previously crawlwed papers in the evaluation period), but here we'll assume we don't have one, and need to build it from scratch.

    # The maxmimum nmber of papers to retrieve per theory/law. Nominally, 50+, but for debugging we will set it low (i.e. 10)
    max_papers_to_retrieve = 10
    #max_papers_to_retrieve = 50

    # Limit processing for debugging/development.
    debug_limit = 2     # Only process two laws for demonstration purposes.
    #debug_limit = None  # Set to None for full run

    # Model to use for the evaluation
    evaluation_model_str = "claude-sonnet-4-5-20250929"

    # Build the cache
    build_paperfinder_cache_for_laws(
        filename_in = filename_single_laws_out,
        paperstore_filename_in = paperstore_filename_in,
        path_out = path_out,
        max_papers_to_retrieve = max_papers_to_retrieve,
        evidence_window_start_year = 2025,  # This is the year/month when evidence starts. Should be after the knowledge cutoff date of the theory generation.
        evidence_window_start_month = 7,
        evidence_window_end_year = 2030,    # This is a far-off date in the future to include all possible evidence.
        evidence_window_end_month = 1,
        debug_limit = debug_limit,
        subsample_ratio = None,             # Set to a float between 0 and 1 to subsample theories for debugging; set to None for full run
        model_str = evaluation_model_str,   # Here, the model is primarily used to generate PaperFinder queries.
        num_workers=5,                      # Number of parallel workers to use when building the cache.  This is limited by PaperFinder and LLM rate limits.
    )

    # Step 2: Perform the LLM-based predictive accuracy evaluation using the cached papers
    do_llm_law_evaluation_llm_as_a_judge_with_paperfinder_predictive_accuracy_cached(
        path_in_cache = path_out,
        model_str = evaluation_model_str,
        num_workers_theories = 20,          # Number of parallel workers to use when evaluating different theories/laws. Also internally threaded, real number of workers may be 10x higher than this (e.g. num_workers_theories * 10).
    )

    # Step 3: Summarize the results for the user.
    summarize_predictive_accuracy(
        path_in = os.path.join(path_out, "predictive-accuracy-evaluation", "claude_sonnet_4_5_20250929"),
        generate_output_file = True,
    )

    print("Predictive accuracy evaluation complete.")