# EvaluationLLMAsAJudge.py

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
#   Evaluation and Prompt (one paper)
#


# Rate a single law using LLM-as-a-judge
def rate_theory_law_with_llm(law:dict, model_str:str="claude-sonnet-4-5-20250929", max_tokens:int=32000, temperature=0.0):
    def mkPrompt(law:dict):
        prompt = """# GENERAL SCIENTIFIC THEORY EVALUATION

You will evaluate a scientific or mechanistic law (in the context of a broader theory description).
Your goal is to assess its quality using general scientific knowledge, empirical findings, and established principles from any relevant field.

You are NOT restricted to using only the theory text itself.
You SHOULD use:
- widely established scientific facts,
- well-known empirical or mechanistic findings,
- broadly accepted models, laws, and results across scientific disciplines,
- general familiarity with the research literature,
- knowledge from real scientific papers.

IMPORTANT NOTE: For this evaluation, you will not be provided with external papers, and must use your own internal knowledge to make the assessments.

Do NOT fabricate specific papers, experiments, or data.
Do NOT invent evidence.
You MAY use real scientific results to assess correctness.

----------------------------------------------------------------
SCORING SCALE (1-10)
----------------------------------------------------------------

Each dimension is scored on a 1-10 scale.

- Scores of 2, 4, 6, and 8 are the primary anchored levels defined below.
- Scores of 3, 5, 7, and 9 represent intermediate cases between adjacent anchors.
- Score 1 represents a pathological or maximally poor case for that dimension.
- Score 10 represents an exceptional or near-ideal case.
- Scores of 1 and 10 should be used sparingly and only when strongly justified by the evidence.
- Avoid defaulting to mid-range scores unless the evidence is genuinely mixed.

----------------------------------------------------------------
GLOBAL INTERPRETATION GUARDRAILS (READ CAREFULLY)
----------------------------------------------------------------

- Do NOT assume a theory claims exclusivity or necessity unless it explicitly states this (e.g., "only", "must", "cannot otherwise", "necessary").
- A paper/experiment that succeeds using a different mechanism does NOT contradict a theory that claims a sufficient mechanism.
- If a paper/experiment does not evaluate or apply the theory's proposed mechanism, in an appropriate domain/scope, most dimensions should be rated as null, not low.
- Low scores (1, 2, 3, 4) are appropriate ONLY when evidence directly evaluates the theory, in an appropriate domain/scope, and shows factual error, failure, contradiction, or other relevant issue.
- Absence of evidence is NOT evidence of falsity.

----------------------------------------------------------------
SCOPE-AWARE EVALUATION PRINCIPLES
----------------------------------------------------------------

Theories are always defined over a domain or scope of applicability.

IMPORTANT:
- Evaluate all dimensions, especially factual accuracy and mechanistic plausibility, ONLY within the theory's stated scope.
- Do NOT penalize a theory for failing outside its declared domain unless it explicitly claims broader applicability.
- If a theory clearly states limitations (e.g., scale, regime, assumptions), those limitations should be respected during evaluation.
- If the theory fails to state its scope, evaluate it according to its implicit or claimed generality.

----------------------------------------------------------------
EVALUATION DIMENSIONS
----------------------------------------------------------------

## Dimension 1. FACTUAL ACCURACY (1-10)
Evaluate whether the theory's claims are true, false, or unsupported relative to established scientific knowledge, WITHIN THE THEORY'S STATED SCOPE.

### Potential Rating Pitfalls
a) Pitfall: Treating alternative successful methods as factual contradictions.
Action: If an experiment does not test the theory's mechanism, in the appropriate domain/scope, use null.
Example: An experiment achieves the same task using a different approach. [Correct rating: factual_accuracy = null]

b) Pitfall: Confusing lack of evaluation with falsification.
Action: Only assign 2-4 if evidence tests the mechanism and shows failure.
Example: Experiment implements the theory's mechanism and shows it fails. [Correct rating: factual_accuracy = 2 or 4]
Example: Experiments do not implement or test the theory's mechanism, or tests it under out-of-domain/scope conditions [Correct rating: factual_accuracy = null]

### Rating Anchors

1 - FUNDAMENTALLY FALSE / PATHOLOGICAL
- Central claims directly contradict well-established scientific facts.
- Relies on fabricated, impossible, or internally inconsistent assertions.

2 - CONTRADICTED BY KNOWN FACTS
- Claims clearly inconsistent with established scientific results or fundamental principles.
- Serious factual errors dominate the theory.

4 - MOSTLY INACCURATE OR UNSUPPORTED
- Many claims are unverified, misleading, or implausible.
- Strong overreach beyond established knowledge without clear caveats.

6 - MIXED ACCURACY
- Some claims align with known results; others are speculative or weakly supported.

8 - MOSTLY ACCURATE
- Claims largely align with established scientific findings.

10 - FULLY ACCURATE / EXEMPLARY
- All factual claims align with well-established empirical results.


## Dimension 2. SPECIFICITY AND OPERATIONALIZABILITY (1-10)
Evaluate how concrete, explicit, and experimentally actionable the theory is.

### Rating Anchors
1 - Entirely vague or non-operational.
2 - Purely high-level idea with no clear testable details.
4 - Some testable components, but key variables or mechanisms are underspecified.
6 - Reasonably specific mechanisms with identifiable variables and testable components.
8 - Well-defined variables, conditions, mechanisms, and predictions.
10 - Fully operationalizable: explicit variables, measurable predictions, and clear experimental interventions.


## Dimension 3. NOVELTY / CONTRIBUTION (1-10)
Evaluate how original and non-derivative the theory is relative to existing scientific ideas.

### Potential Rating Pitfalls
a) Pitfall: Treating synthesis or unification as non-novel.
Action: Unified framing typically merits a mid-range score.
Example: Theory unifies several known approaches that have not been previously unified, as long as this is non-trivial. [Correct rating: novelty = 6]

### Rating Anchors
1 - Entirely known, trivial, or redundant.
2 - Well-known idea with minimal or cosmetic variation.
4 - Slight extension or recombination of existing ideas.
6 - Moderately novel synthesis or reframing of known concepts.
8 - Substantially original mechanism, framing, or integration.
10 - Highly original; introduces genuinely new explanatory principles or directions.


## Dimension 4. TESTABILITY / FALSIFIABILITY (1-10)
Evaluate whether the theory makes clear, falsifiable empirical predictions.

### Potential Rating Pitfalls
a) Pitfall: Penalizing testability because experiments have not yet been run.
Action: Testability concerns structure, not evidence.
Example: Clear predictions, and clear, viable, not overly difficult or expensive methods of experimental verification, even though there's no evidence those experiments have been run yet. [Score at least a 6; `testability` concerns whether it CAN be tested, not whether it HAS been tested]

### Rating Anchors
1 - Completely unfalsifiable in principle.
2 - Nearly unfalsifiable; consequences are too vague or abstract to measure.
4 - Some empirical consequences, but weak, indirect, or impractical to test.
6 - Reasonably testable with identifiable interventions or measurements.
8 - Clear, concrete falsifiable predictions via specific experiments.
10 - Multiple independent, crisp, and feasible falsification paths.


## Dimension 5. MECHANISTIC PLAUSIBILITY (1-10)
Evaluate whether the proposed mechanism is plausible given established knowledge.

### Potential Rating Pitfalls
a) Pitfall: Treating non-uniqueness as implausibility.
Action: Multiple plausible mechanisms may coexist.
Example: Mechanism aligns with known behavior but is not unique. [Uniqueness does not affect plausibility; rate according to alignment with known mechanisms.]

### Rating Anchors
1 - Mechanistically impossible under known scientific laws.
2 - Highly implausible; contradicts fundamental principles of how systems operate.
4 - Weak plausibility; poor alignment with known mechanisms.
6 - Plausible but incomplete; broadly consistent with known scientific behavior.
8 - Strongly plausible; well-aligned with established mechanisms.
10 - Highly plausible; closely matches well-understood mechanisms observed across relevant scientific fields.


## Dimension 6. EMPIRICAL TESTING AND EVIDENCE COVERAGE (1-10)
Evaluate how extensively the theory's predictions have been empirically tested.

### Potential Rating Pitfalls
a) Pitfall: Assigning low scores when the known evidence does not test the theory.
Action: Use null when known evidence does not evaluate the theory.
Example: Known evidence studies an unrelated mechanism, or a similar mechanism but a different domain/scope. [Correct rating: empirical_testing = null]

### Rating Anchors
1 - Completely untested
2 - Minimally tested; only indirect or weak empirical contact
4 - Partially tested; some predictions examined, others untested
6 - Moderately tested; multiple predictions evaluated, but coverage incomplete
8 - Largely tested; most core predictions empirically examined
10 - Extensively tested; predictions repeatedly tested, replicated, and supported across contexts


----------------------------------------------------------------
INSTRUCTIONS FOR THE EVALUATOR
----------------------------------------------------------------

- You MAY use your full scientific and technical knowledge.
- You MAY use well-established empirical results from any scientific domain.
- You MUST NOT invent papers, experiments, or fictional evidence.
- If you do not have genuinely relevant information for a given dimension, assign a score of null for that dimension and leave the explanation blank.
- Do NOT force scores when knowledge/relevance is weak.
- Each score must be accompanied by a brief justification (2-4 concise, information-dense sentences).
- You are welcome to think as much as you like before answering.
- The final output MUST be a JSON object, enclosed in code blocks, in the exact format specified below.


----------------------------------------------------------------
JSON OUTPUT FORMAT
----------------------------------------------------------------

{
  "paper_title": "Title of the paper being used for evaluation",
  "contains_relevant_information": true or false,
  "evaluation": [
    {"factual_accuracy": 1-10 or null, "explanation": "..."},
    {"specificity": 1-10 or null, "explanation": "..."},
    {"novelty": 1-10 or null, "explanation": "..."},
    {"testability": 1-10 or null, "explanation": "..."},
    {"plausibility": 1-10 or null, "explanation": "..."},
    {"empirical_testing": 1-10 or null, "explanation": "..."}
  ]
}

# THEORY TO EVALUATE
"""

        prompt += "```\n"
        prompt += json.dumps(law, indent=4)
        prompt += "\n```\n"

        prompt += "Please provide your accurate, faithful, correct evaluation.\n"

        return prompt

    total_cost = 0.0
    prompt = mkPrompt(law)
    responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=False, max_generation_time_seconds=300)
    total_cost += cost

    evaluation = None
    try:
        evaluation = responseJSON.get("evaluation", None)
    except Exception as e:
        print("Error parsing evaluation JSON: " + str(e))
        evaluation = None

    packed = {
        "model_str": model_str,
        "law": law,
        "evaluation": evaluation,
        "response_text": responseText,
        "total_cost": total_cost,
    }
    return packed



#
#   Evaluation (batch)
#
def do_llm_law_evaluation_llm_as_a_judge(filename_in:str, filename_out:str, model_str="claude-sonnet-4-5-20250929", num_workers:int=50):

    # Load a list of laws
    laws = None
    with open(filename_in, "r") as f:
        laws = json.load(f)

    print("Loaded " + str(len(laws)) + " laws from file.")

    # For each law, rate it using the LLM (in parallel)
    results = []
    averages = {}
    average_results = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(
                rate_theory_law_with_llm,
                law,
                model_str=model_str,
                max_tokens=32000,
                temperature=0.0,
            ): idx for idx, law in enumerate(laws)
        }

        from tqdm import tqdm

        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)

                # Also keep average evaluations
                evaluation = result.get("evaluation", [])   # A list of dicts with scores and explanations
                for dim_eval in evaluation:
                    for dim_name, score in dim_eval.items():
                        if (dim_name == "explanation"):
                            continue
                        if (dim_name not in average_results):
                            average_results[dim_name] = {
                                "total_score": 0,
                                "count": 0,
                                "none": 0,
                            }
                        # Check that the score is None, or that it's an int or float
                        if (score is None) or (not isinstance(score, (int, float))):
                            average_results[dim_name]["none"] += 1
                        else:
                            average_results[dim_name]["total_score"] += score
                            average_results[dim_name]["count"] += 1

                # Calculate the average scores so far
                print("Current average evaluation scores:")
                averages = {}
                for dim_name, dim_data in average_results.items():
                    averages[dim_name] = 0.0
                    if (dim_data["count"] > 0):
                        #avg_score = dim_data["total_score"] / dim_data["count"]
                        averages[dim_name] = dim_data["total_score"] / dim_data["count"]
                        # Round to 3 decimal places
                        averages[dim_name] = round(averages[dim_name], 3)

                    print(f"  {dim_name}: {averages[dim_name]:.2f}")

                # Pack it
                packed = {
                    "model_str": model_str,
                    "total_laws_scored": len(results),
                    "average_evaluation_scores": averages,
                    "counts": dim_data,
                    "raw_results": results,
                }
                # Save the packed results
                print(f"Saving intermediate results to {filename_out}")
                with open(filename_out, "w") as f:
                    json.dump(packed, f, indent=4)

            except Exception as e:
                print(f"Error processing law {idx+1}: {e}")
                import traceback
                traceback.print_exc()

    print("Completed LLM-as-a-judge law evaluations.")
    return averages, average_results




# Stand-alone example
if __name__ == "__main__":
    # Step 0: Load the API keys
    loadAPIKeys()

    # Step 1: Convert TheoryStore (containing theories) into a file containing single laws. Also filter to include laws with a minimum self-assessed novelty rating ('new' theories, or 'somewhat-related-to-existing' theories).
    filename_theorystore_in = "theorystore-example-parametric.json"
    filename_single_laws_out = "theorystore-example-single-laws.json"
    convert_theorystore_to_single_laws_with_filtering(
        filename_theorystore=filename_theorystore_in,
        filename_out=filename_single_laws_out,
        start_idx=0,
        end_idx=None,
        allow_all=False,
        allow_new_and_somewhat_new=True,        # Filter to include only laws with novelty ratings of 'new' or 'somewhat-related-to-existing'
    )

    print("Pausing for 5 seconds to show the theorystore summary statistics before continuing...")
    time.sleep(5)

    # Step 2: Perform the LLM-as-a-judge evaluation on the filtered single laws file
    filename_out_llm_as_a_judge = filename_single_laws_out.replace(".json", ".llm_as_a_judge_evaluated." + time.strftime("%Y%m%d-%H%M%S") + ".json")
    averages, average_results = do_llm_law_evaluation_llm_as_a_judge(
        filename_in=filename_single_laws_out,
        filename_out=filename_out_llm_as_a_judge,
        model_str="claude-sonnet-4-5-20250929",
        num_workers=10,                         # Number of parallel workers (adjust based on API rate limits)
    )

    print("LLM-as-a-judge evaluation complete.")

    # As above, but make it constant width for easier reading. 3 columns (dimension name, average score, counts)
    print("\n")
    print("Final average evaluation scores (formatted):")
    print(f"{'Dimension':<30} {'Average Score':<15} {'Counts':<20}")
    for dim_name, dim_data in average_results.items():
        average_score = 0.0
        if (dim_data["count"] > 0):
            average_score = dim_data["total_score"] / dim_data["count"]
            average_score = round(average_score, 3)
        #counts_str = f"counted: {dim_data['count']}, none: {dim_data['none']}"
        counts_str = f"{dim_data['count']}"
        print(f"{dim_name:<30} {average_score:<15.2f} {counts_str:<20}")

    print("\n")
    print("Results saved to: " + filename_out_llm_as_a_judge)
