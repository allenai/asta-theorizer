# EvaluationQualifiedNovelty.py

import os
import json
import time
import random

from Struct import *
from PaperStore import *
from PaperFinderRequests import *
from Theorizer import *

from TheorizerProcessing import *

from ExtractionUtils import *

from EvaluationCommon import *

from concurrent.futures import as_completed


# Main novelty assessment prompt, for evaluating the novelty of a given theory/law with respect to a single scientific paper
def evaluate_novelty_one_paper(theory_law_dict:dict, paper_text:str, model_str:str, temperature:float=0.0, max_tokens:int=16384, use_reflection:bool=True, idx=None):

    def mkPrompt(theory_law_dict:dict, paper_text:str, reflection:bool=True):
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"
        prompt += "# Task\n"
        prompt += "You will be shown (1) a potentially novel theory/law, and (2) the text of a scientific paper. "
        prompt += "Your task is to evaluate the novelty of the theory/law with respect to the content of the paper. "
        prompt += "The results from many analyses of individual papers will then be aggregated to produce an overall evaluation of the novelty of the given theory/law.\n"
        prompt += "\n"

        prompt += "# Novelty Evaluation Operationalization\n"
        prompt += "Novelty is a blanket term, and there are many different types of novelty. "
        prompt += "For this task, we will operationalize novelty in terms of *novelty type*, and (for each type), it's *degree of novelty*.\n"
        prompt += "\n"
        prompt += "## Novelty Types\n"
        prompt += "We will consider the following (non-exclusive) types of novelty.  Note that a theory/law may be novel in multiple types with respect to a given paper.\n"
        prompt += "\n"

        # TODO

        prompt += """## Types of Novelty
You are evaluating novelty at the level of a specific theory statement (i.e., a concrete claim, law, or principle), interpreted in the context of the broader theory description, which provides motivation, scope, and grounding.

Assess whether the statement itself contributes along each novelty dimension below. Do NOT score novelty based on the overall research program, framing, or ambition unless that novelty is clearly expressed in the statement.

A single theory statement may exhibit multiple types of novelty simultaneously. For each type below, assess whether the statement contributes along that dimension, independently of the degree of novelty (which is assessed separately).

1. Type: Phenomenon / Effect Novelty
Definition:
- The statement identifies, names, or characterizes a phenomenon, effect, or regularity that was not previously recognized in the literature.

Operational criteria:
- The novelty is in *what happens*, as asserted by the statement.
- The statement makes a substantive empirical or behavioral claim.

What this is NOT:
- A new explanation for an already known effect (that is Explanatory novelty).
- A restatement or relabeling of a known phenomenon.
- Novelty that appears only in the broader theory description but not in the statement.

Examples:
- The statement claims a previously unknown empirical effect.
- The statement asserts a new failure mode or emergent behavior.

2. Type: Explanatory / Mechanistic Novelty
Definition:
- The statement proposes a new explanation, causal structure, or mechanistic account for one or more known phenomena.

Operational criteria:
- The novelty is in *why* or *how*, as asserted by the statement.
- The statement introduces new causal relations, latent variables, or mechanisms.

What this is NOT:
- Descriptive or correlational claims without explanatory structure.
- Explanatory novelty present only in surrounding text but not in the statement.

Examples:
- The statement asserts a new causal mechanism underlying known results.
- The statement introduces a new latent-variable interpretation.

3. Type: Unification Novelty
Definition:
- The statement asserts that multiple previously separate results, methods, or findings are instances of a single underlying principle or framework.

Operational criteria:
- The unification is explicit in the statement itself.
- The statement directly connects or subsumes multiple lines of prior work.

What this is NOT:
- Merely citing many papers in the broader theory description.
- Generalization without explicit conceptual unification in the statement.

Examples:
- The statement claims that several methods optimize the same objective.
- The statement asserts a single principle explaining multiple empirical findings.

4. Type: Generalization / Scope Expansion Novelty
Definition:
- The statement extends existing results to broader settings, assumptions, or domains.

Operational criteria:
- The core idea is known, but the statement explicitly broadens its scope.
- The expansion is part of the claim, not merely suggested in discussion.

Common axes of generalization:
- Modalities (e.g., unimodal → multimodal)
- Assumptions (e.g., linear → nonlinear)
- Domains (e.g., synthetic → real-world)
- Scale (e.g., small → large)

What this is NOT:
- Applying known ideas to a new dataset without analytical extension.
- Scope expansion mentioned only in the broader theory description.

Examples:
- The statement asserts that a known result holds in a broader class of settings.
- The statement explicitly weakens prior assumptions.

5. Type: Constraint / Limitation Novelty
Definition:
- The statement identifies new necessary conditions, limitations, or failure regimes of existing methods or theories.

Operational criteria:
- The statement clarifies when or why a method or result does NOT hold.
- The constraint is part of the statement itself.

What this is NOT:
- Generic caveats or obvious limitations.
- Constraints mentioned only outside the statement.

Examples:
- The statement asserts a previously unrecognized failure condition.
- The statement identifies a necessary assumption that was previously implicit.

6. Type: Conceptual Reframing / Abstraction Novelty
Definition:
- The statement introduces a new conceptual lens, abstraction, or organizing structure for interpreting prior work.

Operational criteria:
- The reframing is embodied in the statement, not just the surrounding narrative.
- The statement changes how results should be conceptualized or categorized.

What this is NOT:
- Stylistic rewriting or terminology changes without conceptual impact.
- Reframing present only in the broader theory description.

Examples:
- The statement introduces a new decomposition or abstraction.
- The statement recasts known results under a new conceptual framework.

7. Type: Empirical Synthesis / Meta-Regularity Novelty
Definition:
- The statement asserts an empirical regularity or pattern that emerges only when synthesizing evidence across multiple studies.

Operational criteria:
- The claim requires aggregation across papers.
- The synthesis is expressed as a statement-level conclusion.

What this is NOT:
- A single new experiment.
- Narrative summary without a synthesized claim.

Examples:
- The statement claims a consistent pattern across many studies.
- The statement resolves apparent empirical contradictions via synthesis.
"""

        prompt += "\n"
        prompt += """## Degree of Novelty
For each novelty type, you will assess the degree of novelty on a categorical scale.
This scale refers to the epistemic status of the claim in the prior literature,
not whether similar wording ever appeared.

1. Degree: Explicit and Established (known, recognized knowledge: not novel)
- Explicitly stated as a claim, theorem, principle, or other central component of one or more papers.
- Recognized, named, cited, or treated as known/discovered.
- A knowledgeable expert would reasonably say: “This is already known.”

Examples:
- A theorem, law, or effect with a commonly used name.
- A well-known empirical relationship repeatedly cited across papers.
- A result that later papers explicitly rely on as established fact.
- “This is standard / known in the literature.”

2. Degree: Explicit but Peripheral / Unrecognized
(stated, but not treated as knowledge: weak novelty when elevated)
- Explicitly stated somewhere in the paper (e.g., footnote, appendix, assumption, side remark, lemma).
- Not highlighted as a contribution, result, or organizing principle.
- Not discussed, named, or cited by later work as a finding.
- Key point: it is explicitly written, but not epistemically established.

Examples:
- A lemma that implies an important relationship, but is introduced only to support another result.
- A footnote remark noting a limitation or generalization, without analysis or emphasis.
- A modeling choice whose implications are stated but not discussed.
- “This was technically stated before, but no one treated it as a result.”

3. Degree: Derivable but Unstated
(logically follows from prior work, but was not recognized as knowledge)
- Can be derived from existing results, but only by:
  - synthesizing multiple papers, or
  - abstracting beyond any single experiment or theorem.
- No paper explicitly states it as a conclusion or contribution.
- Experts would not have cited it as known prior to this articulation.

Examples:
- A unifying principle that explains several separate results across papers.
- A generalization that follows from multiple special cases, but was never stated.
- A consequence of existing theory that becomes obvious only in hindsight.
- “In hindsight this follows, but I hadn't seen it stated before.”

4. Degree: Genuinely New
(not present, not assumed, not derivable: strong novelty)
- Introduces a claim, mechanism, or principle that does not appear in prior frameworks.
- Not implicit, not assumed, and not a straightforward consequence of existing results.
- Would likely surprise domain experts.
- Would plausibly be cited for this specific insight.

Examples:
- A previously unknown phenomenon or failure mode.
- A new causal or mechanistic explanation that replaces existing ones.
- A theoretical result that cannot be derived without new assumptions or analysis.
- “This is genuinely new.”

"""
        prompt += "\n"
        prompt += "## Instructions for Assessment\n"
        prompt += "For each novelty type, assess:\n"
        prompt += "- `what_is_known`: What relevant knowledge is already established in the paper?\n"
        prompt += "- `what_introduced`: What, if anything, does the theory/statement introduce on this dimension?\n"
        prompt += "- `what_novel`: What, if anything, is novel on this dimension?\n"
        prompt += "- `degree_of_novelty`: Assign a degree of novelty, which is categorical, from one of the 4 categories: `explicit_established`, `explicit_peripheral`, `derivable_unstated`, `genuinely_new`.\n"
        prompt += "\n"
        prompt += "# Theory and Statement\n"
        prompt += "Here is the theory/statement/law to be evaluated.  The theory includes the actual statement, a broader theory description (given that the theory may have more than one statement in it), as well as explicit scoping information for the statement:\n"
        prompt += "```\n"
        prompt += json.dumps(theory_law_dict, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"
        prompt += "# Paper Text\n"
        prompt += "Here is the text of the scientific paper to evaluate against:\n"
        prompt += "```\n"
        prompt += paper_text + "\n"
        prompt += "```\n"
        prompt += "\n"

        if (reflection != None):
            prompt += "# Reflection\n"
            prompt += "This is a reflection step.  Previously, you generated the following output.  Please examine it carefully, and reflect on whether it is accurate, faithful to the task, scientifically rigorous, and generally of high quality.  Please fix any issues when providing your output.\n"
            prompt += "```\n"
            prompt += json.dumps(reflection, indent=4) + "\n"
            prompt += "```\n"
            prompt += "\n"

        prompt += "# Output Format\n"
        prompt += "You are welcome (and encouraged) to think as much as possible before answering.\n"
        prompt += "Your final answer must be provided in JSON format, between triple backticks (```), and must strictly adhere to the specified structure.  You must not use triple backticks anywhere in your response except for a single time, at the end of the output, enclosing the JSON output.\n"
        prompt += "The output format is:"
        prompt += "- A dictionary, called `novelty_evaluation`.\n"
        prompt += "- The key `paper_title` with value = string (the title of the paper being evaluated)\n"
        prompt += "- The keys of the dictionary are the 7 novelty types: `phenomenon_effect`, `explanatory_mechanistic`, `unification`, `generalization_scope_expansion`, `constraint_limitation`, `conceptual_reframing_abstraction`, `empirical_synthesis_meta_regulariry`.\n"
        prompt += "- The value for each key is another dictionary with the following keys:\n"
        prompt += "  - `what_is_known`: string (1-3 detailed, concise, information-dense sentences)\n"
        prompt += "  - `what_introduced`: string (1-3 detailed, concise, information-dense sentences)\n"
        prompt += "  - `what_novel`: string (1-3 detailed, concise, information-dense sentences)\n"
        prompt += "  - `degree_of_novelty`: string, one of `explicit_established`, `explicit_peripheral`, `implicit_assumed`, `derivable_unstated`, `genuinely_new`\n"
        prompt += "\n"
        prompt += "The information should be written in such a way that it is detailed, concise, information-dense, and scientifically accurate.\n"
        prompt += "It must also be useful for aggregating the results across many papers, so the information it provides must clearly identify what this paper contributes to the overall novelty assessment.\n"
        prompt += "\n"
        prompt += "No information/Failure to Rate: If the paper does not provide any relevant information for assessing a given novelty type, simply provide a null/none in place of the dictionary.  For example, `phenomenon_effect: null`.\n"
        prompt += "\n"
        prompt += "For example:\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += '  "novelty_evaluation": {\n'
        prompt += '    "paper_title": "Title of the Paper Being Evaluated",\n'
        prompt += '    "phenomenon_effect": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    },\n'
        prompt += '    "explanatory_mechanistic": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    },\n'
        prompt += '    "unification": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    },\n'
        prompt += '    "generalization_scope_expansion": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    },\n'
        prompt += '    "constraint_limitation": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    },\n'
        prompt += '    "conceptual_reframing_abstraction": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    },\n'
        prompt += '    "empirical_synthesis_meta_regulariry": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    }\n'
        prompt += '  }\n'
        prompt += "}\n"
        prompt += "```\n"
        prompt += "\n"
        prompt += "# Important Notes\n"
        prompt += "- You must strictly adhere to the specified output format.  Failure to do so will result in an error.\n"
        prompt += "- Scientific accuracy, faithfulness to the task, integrity, and adherence to the specified format are critical.\n"
        prompt += "- You are part of an automated scientific discovery pipeline, where success could mean positive breakthroughs with highly positive societal impact, and failure could mean missed breakthroughs or wasted resources. Accuracy and integrity are of utmost importance.\n"
        prompt += "- Do not hallucinate.\n"
        return prompt


    total_cost = 0.0
    all_responses = []
    prompt = mkPrompt(theory_law_dict, paper_text, reflection=None)

    responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=False, max_generation_time_seconds=300)
    total_cost += cost

    if (responseJSON is None) or (not isinstance(responseJSON, dict)):
        print("ERROR: LLM response is not valid JSON dictionary.")
    else:
        all_responses.append(responseJSON)

    # Reflect on the response
    if (use_reflection == True):
        # Get the best response from all_responses
        best_response = None
        if (len(all_responses) > 0):
            best_response = all_responses[0]

        prompt = None
        if (best_response is not None):
            prompt = mkPrompt(theory_law_dict, paper_text, reflection=best_response)
        else:
            prompt = mkPrompt(theory_law_dict, paper_text, reflection=None)

        responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=False, max_generation_time_seconds=300)
        total_cost += cost

        if (responseJSON is None) or (not isinstance(responseJSON, dict)):
            print("ERROR: LLM response is not valid JSON dictionary (on reflection).")
        else:
            all_responses.append(responseJSON)

    # Get the best response
    best_response = None
    if (len(all_responses) > 0):
        best_response = all_responses[-1]

    # Pack output
    packed = {
        "idx": idx,
        "model_str": model_str,
        "temperature": temperature,
        "total_cost": total_cost,
        "best_response": best_response,
        "all_responses": all_responses,
        "theory_law_dict": theory_law_dict,
        "paper_text": paper_text,
    }

    return packed


# Main novelty assessment prompt, for evaluating the novelty of a given theory/law with respect to a single scientific paper
# This function aggregates the results from multiple papers to produce an overall novelty assessment
def evaluate_novelty_aggregated(theory_law_dict:dict, aggregated_results:dict, model_str:str, temperature:float=0.0, max_tokens:int=16384, use_reflection:bool=True, idx=None):

    def mkPrompt(theory_law_dict:dict, aggregated_results:dict, reflection:bool=True):
        prompt = ""
        prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
        prompt += "\n"
        prompt += "# Task\n"
        prompt += "You will be shown (1) a potentially novel theory/law, and (2) the aggregated results of assessing the novelty of that theory/law on a set of scientific papers, on a specific dimension."
        prompt += "Your task is to evaluate the novelty of the theory/law with respect to the aggregated novelty assessment results across the papers. "
        prompt += "You should take these results from many analyses of individual papers and aggregate them to produce an overall evaluation of the novelty of the given theory/law.\n"
        prompt += "\n"

        prompt += "# Novelty Evaluation Operationalization\n"
        prompt += "Novelty is a blanket term, and there are many different types of novelty. "
        prompt += "For this task, we will operationalize novelty in terms of *novelty type*, and (for each type), it's *degree of novelty*.\n"
        prompt += "\n"
        prompt += "## Novelty Types\n"
        prompt += "We will consider the following (non-exclusive) types of novelty.  Note that a theory/law may be novel in multiple types with respect to a given paper.\n"
        prompt += "\n"

        # TODO

        prompt += """## Types of Novelty
You are evaluating novelty at the level of a specific theory statement (i.e., a concrete claim, law, or principle), interpreted in the context of the broader theory description, which provides motivation, scope, and grounding.

Assess whether the statement itself contributes along each novelty dimension below. Do NOT score novelty based on the overall research program, framing, or ambition unless that novelty is clearly expressed in the statement.

A single theory statement may exhibit multiple types of novelty simultaneously. For each type below, assess whether the statement contributes along that dimension, independently of the degree of novelty (which is assessed separately).

1. Type: Phenomenon / Effect Novelty
Definition:
- The statement identifies, names, or characterizes a phenomenon, effect, or regularity that was not previously recognized in the literature.

Operational criteria:
- The novelty is in *what happens*, as asserted by the statement.
- The statement makes a substantive empirical or behavioral claim.

What this is NOT:
- A new explanation for an already known effect (that is Explanatory novelty).
- A restatement or relabeling of a known phenomenon.
- Novelty that appears only in the broader theory description but not in the statement.

Examples:
- The statement claims a previously unknown empirical effect.
- The statement asserts a new failure mode or emergent behavior.

2. Type: Explanatory / Mechanistic Novelty
Definition:
- The statement proposes a new explanation, causal structure, or mechanistic account for one or more known phenomena.

Operational criteria:
- The novelty is in *why* or *how*, as asserted by the statement.
- The statement introduces new causal relations, latent variables, or mechanisms.

What this is NOT:
- Descriptive or correlational claims without explanatory structure.
- Explanatory novelty present only in surrounding text but not in the statement.

Examples:
- The statement asserts a new causal mechanism underlying known results.
- The statement introduces a new latent-variable interpretation.

3. Type: Unification Novelty
Definition:
- The statement asserts that multiple previously separate results, methods, or findings are instances of a single underlying principle or framework.

Operational criteria:
- The unification is explicit in the statement itself.
- The statement directly connects or subsumes multiple lines of prior work.

What this is NOT:
- Merely citing many papers in the broader theory description.
- Generalization without explicit conceptual unification in the statement.

Examples:
- The statement claims that several methods optimize the same objective.
- The statement asserts a single principle explaining multiple empirical findings.

4. Type: Generalization / Scope Expansion Novelty
Definition:
- The statement extends existing results to broader settings, assumptions, or domains.

Operational criteria:
- The core idea is known, but the statement explicitly broadens its scope.
- The expansion is part of the claim, not merely suggested in discussion.

Common axes of generalization:
- Modalities (e.g., unimodal → multimodal)
- Assumptions (e.g., linear → nonlinear)
- Domains (e.g., synthetic → real-world)
- Scale (e.g., small → large)

What this is NOT:
- Applying known ideas to a new dataset without analytical extension.
- Scope expansion mentioned only in the broader theory description.

Examples:
- The statement asserts that a known result holds in a broader class of settings.
- The statement explicitly weakens prior assumptions.

5. Type: Constraint / Limitation Novelty
Definition:
- The statement identifies new necessary conditions, limitations, or failure regimes of existing methods or theories.

Operational criteria:
- The statement clarifies when or why a method or result does NOT hold.
- The constraint is part of the statement itself.

What this is NOT:
- Generic caveats or obvious limitations.
- Constraints mentioned only outside the statement.

Examples:
- The statement asserts a previously unrecognized failure condition.
- The statement identifies a necessary assumption that was previously implicit.

6. Type: Conceptual Reframing / Abstraction Novelty
Definition:
- The statement introduces a new conceptual lens, abstraction, or organizing structure for interpreting prior work.

Operational criteria:
- The reframing is embodied in the statement, not just the surrounding narrative.
- The statement changes how results should be conceptualized or categorized.

What this is NOT:
- Stylistic rewriting or terminology changes without conceptual impact.
- Reframing present only in the broader theory description.

Examples:
- The statement introduces a new decomposition or abstraction.
- The statement recasts known results under a new conceptual framework.

7. Type: Empirical Synthesis / Meta-Regularity Novelty
Definition:
- The statement asserts an empirical regularity or pattern that emerges only when synthesizing evidence across multiple studies.

Operational criteria:
- The claim requires aggregation across papers.
- The synthesis is expressed as a statement-level conclusion.

What this is NOT:
- A single new experiment.
- Narrative summary without a synthesized claim.

Examples:
- The statement claims a consistent pattern across many studies.
- The statement resolves apparent empirical contradictions via synthesis.
"""

        prompt += "\n"
        prompt += """## Degree of Novelty
For each novelty type, you will assess the degree of novelty on a categorical scale.
This scale refers to the epistemic status of the claim in the prior literature,
not whether similar wording ever appeared.

1. Degree: Explicit and Established (known, recognized knowledge: not novel)
- Explicitly stated as a claim, theorem, principle, or other central component of one or more papers.
- Recognized, named, cited, or treated as known/discovered.
- A knowledgeable expert would reasonably say: “This is already known.”

Examples:
- A theorem, law, or effect with a commonly used name.
- A well-known empirical relationship repeatedly cited across papers.
- A result that later papers explicitly rely on as established fact.
- “This is standard / known in the literature.”

2. Degree: Explicit but Peripheral / Unrecognized
(stated, but not treated as knowledge: weak novelty when elevated)
- Explicitly stated somewhere in the paper (e.g., footnote, appendix, assumption, side remark, lemma).
- Not highlighted as a contribution, result, or organizing principle.
- Not discussed, named, or cited by later work as a finding.
- Key point: it is explicitly written, but not epistemically established.

Examples:
- A lemma that implies an important relationship, but is introduced only to support another result.
- A footnote remark noting a limitation or generalization, without analysis or emphasis.
- A modeling choice whose implications are stated but not discussed.
- “This was technically stated before, but no one treated it as a result.”

3. Degree: Derivable but Unstated
(logically follows from prior work, but was not recognized as knowledge)
- Can be derived from existing results, but only by:
  - synthesizing multiple papers, or
  - abstracting beyond any single experiment or theorem.
- No paper explicitly states it as a conclusion or contribution.
- Experts would not have cited it as known prior to this articulation.

Examples:
- A unifying principle that explains several separate results across papers.
- A generalization that follows from multiple special cases, but was never stated.
- A consequence of existing theory that becomes obvious only in hindsight.
- “In hindsight this follows, but I hadn't seen it stated before.”

4. Degree: Genuinely New
(not present, not assumed, not derivable: strong novelty)
- Introduces a claim, mechanism, or principle that does not appear in prior frameworks.
- Not implicit, not assumed, and not a straightforward consequence of existing results.
- Would likely surprise domain experts.
- Would plausibly be cited for this specific insight.

Examples:
- A previously unknown phenomenon or failure mode.
- A new causal or mechanistic explanation that replaces existing ones.
- A theoretical result that cannot be derived without new assumptions or analysis.
- “This is genuinely new.”

"""
        prompt += "\n"
        prompt += "## Instructions for Assessment\n"
        prompt += "For each novelty type, assess:\n"
        prompt += "- `what_is_known`: What relevant knowledge is already established in the paper?\n"
        prompt += "- `what_introduced`: What, if anything, does the theory/statement introduce on this dimension?\n"
        prompt += "- `what_novel`: What, if anything, is novel on this dimension?\n"
        prompt += "- `degree_of_novelty`: Assign a degree of novelty, which is categorical, from one of the 4 categories: `explicit_established`, `explicit_peripheral`, `derivable_unstated`, `genuinely_new`.\n"
        prompt += "\n"
        prompt += "## Aggregation Instructions\n"
        prompt += "You are provided with aggregated results from previous steps that evaluted papers individually.\n"
        prompt += "Your task is to synthesize these results to produce an overall evaluation of the novelty of the theory/law on this dimension.\n"
        prompt += "Remember that the assessment isn't an average: Some papers may provide evidence of something not being novel, and you should pay particular attention to these, and weight them in the context of the remaining papers.\n"
        prompt += "\n"
        prompt += "# Theory and Statement\n"
        prompt += "Here is the theory/statement/law to be evaluated.  The theory includes the actual statement, a broader theory description (given that the theory may have more than one statement in it), as well as explicit scoping information for the statement:\n"
        prompt += "```\n"
        prompt += json.dumps(theory_law_dict, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"
        prompt += "# Aggregated Results\n"
        prompt += "Here are the aggregated results for a specific dimension, based on evaluating a set of papers:\n"
        prompt += "(Sorted so that the evidence suggesting the least novelty comes first)\n"
        prompt += "```\n"
        prompt += json.dumps(aggregated_results, indent=4) + "\n"
        prompt += "```\n"
        prompt += "\n"

        if (reflection != None):
            prompt += "# Reflection\n"
            prompt += "This is a reflection step.  Previously, you generated the following output.  Please examine it carefully, and reflect on whether it is accurate, faithful to the task, scientifically rigorous, and generally of high quality.  Please fix any issues when providing your output.\n"
            prompt += "```\n"
            prompt += json.dumps(reflection, indent=4) + "\n"
            prompt += "```\n"
            prompt += "\n"

        prompt += "# Output Format\n"
        prompt += "You are welcome (and encouraged) to think as much as possible before answering.\n"
        prompt += "Your final answer must be provided in JSON format, between triple backticks (```), and must strictly adhere to the specified structure.  You must not use triple backticks anywhere in your response except for a single time, at the end of the output, enclosing the JSON output.\n"
        prompt += "The output format is:"
        prompt += "- A dictionary, called `novelty_evaluation`.\n"
        prompt += "- The key `paper_title` with value = string (the title of the paper being evaluated)\n"
        prompt += "- The keys of the dictionary are the 7 novelty types: `phenomenon_effect`, `explanatory_mechanistic`, `unification`, `generalization_scope_expansion`, `constraint_limitation`, `conceptual_reframing_abstraction`, `empirical_synthesis_meta_regulariry`.\n"
        prompt += "- The value for each key is another dictionary with the following keys:\n"
        prompt += "  - `what_is_known`: string (1-3 detailed, concise, information-dense sentences)\n"
        prompt += "  - `what_introduced`: string (1-3 detailed, concise, information-dense sentences)\n"
        prompt += "  - `what_novel`: string (1-3 detailed, concise, information-dense sentences)\n"
        prompt += "  - `degree_of_novelty`: string, one of `explicit_established`, `explicit_peripheral`, `implicit_assumed`, `derivable_unstated`, `genuinely_new`\n"
        prompt += "\n"
        prompt += "The information should be written in such a way that it is detailed, concise, information-dense, and scientifically accurate.\n"
        prompt += "It must also be useful for aggregating the results across many papers, so the information it provides must clearly identify what this paper contributes to the overall novelty assessment.\n"
        prompt += "\n"
        prompt += "No information/Failure to Rate: If the paper does not provide any relevant information for assessing a given novelty type, simply provide a null/none in place of the dictionary.  For example, `phenomenon_effect: null`.\n"
        prompt += "\n"
        prompt += "For example:\n"
        prompt += "```\n"
        prompt += "{\n"
        prompt += '  "novelty_evaluation": {\n'
        prompt += '    "paper_title": "Title of the Paper Being Evaluated",\n'
        prompt += '    "phenomenon_effect": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    },\n'
        prompt += '    "explanatory_mechanistic": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    },\n'
        prompt += '    "unification": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    },\n'
        prompt += '    "generalization_scope_expansion": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    },\n'
        prompt += '    "constraint_limitation": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    },\n'
        prompt += '    "conceptual_reframing_abstraction": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    },\n'
        prompt += '    "empirical_synthesis_meta_regulariry": {\n'
        prompt += '      "what_is_known": "...\n",\n'
        prompt += '      "what_introduced": "...\n",\n'
        prompt += '      "what_novel": "...\n",\n'
        prompt += '      "degree_of_novelty": "..."\n'
        prompt += '    }\n'
        prompt += '  }\n'
        prompt += "}\n"
        prompt += "```\n"
        prompt += "NOTE: Because this is an aggregation step, and you are being provided with only evidence from one dimension, please provide `null` assessments for the other dimensions that were not provided.\n"
        prompt += "\n"
        prompt += "# Important Notes\n"
        prompt += "- You must strictly adhere to the specified output format.  Failure to do so will result in an error.\n"
        prompt += "- Scientific accuracy, faithfulness to the task, integrity, and adherence to the specified format are critical.\n"
        prompt += "- You are part of an automated scientific discovery pipeline, where success could mean positive breakthroughs with highly positive societal impact, and failure could mean missed breakthroughs or wasted resources. Accuracy and integrity are of utmost importance.\n"
        prompt += "- Do not hallucinate.\n"
        return prompt


    total_cost = 0.0
    all_responses = []
    prompt = mkPrompt(theory_law_dict, aggregated_results, reflection=None)

    responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=False, max_generation_time_seconds=300)
    total_cost += cost

    if (responseJSON is None) or (not isinstance(responseJSON, dict)):
        print("ERROR: LLM response is not valid JSON dictionary.")
    else:
        all_responses.append(responseJSON)

    # Reflect on the response
    if (use_reflection == True):
        # Get the best response from all_responses
        best_response = None
        if (len(all_responses) > 0):
            best_response = all_responses[0]

        prompt = None
        if (best_response is not None):
            prompt = mkPrompt(theory_law_dict, aggregated_results, reflection=best_response)
        else:
            prompt = mkPrompt(theory_law_dict, aggregated_results, reflection=None)

        responseJSON, responseText, cost = getLLMResponseJSON(promptStr=prompt, model=model_str, maxTokens=max_tokens, temperature=temperature, jsonOut=False, max_generation_time_seconds=300)
        total_cost += cost

        if (responseJSON is None) or (not isinstance(responseJSON, dict)):
            print("ERROR: LLM response is not valid JSON dictionary (on reflection).")
        else:
            all_responses.append(responseJSON)

    # Get the best response
    best_response = None
    if (len(all_responses) > 0):
        best_response = all_responses[-1]

    # Pack output
    packed = {
        "idx": idx,
        "model_str": model_str,
        "temperature": temperature,
        "total_cost": total_cost,
        "best_response": best_response,
        "all_responses": all_responses,
        "theory_law_dict": theory_law_dict,
        "aggregated_results": aggregated_results,
    }

    return packed



# Load theory statements from a TheoryStore
def load_theory_statements_to_evaluate_from_theorystore(filename_in:str):
    # Step 1: Load the TheoryStore
    theorystore_data = None
    with open(filename_in, "r") as f:
        theorystore_data = json.load(f)

    # Step 2: Extract the theories dict
    theories_dict = theorystore_data.get("theories", None)
    # Get the values as a list
    theories = list(theories_dict.values())

    theory_statements = []
    for theory in theories:
        components = theory.get("components", None)
        if (components is None):
            continue

        statements = components.get("theory_statements", None)
        if (statements is None):
            continue

        for idx, statement in enumerate(statements):
            # Get the novelty evaluation
            novelty_evaluation = statement.get("novelty_evaluation", None)
            if (novelty_evaluation is None):
                continue
            likely_novelty_classification = novelty_evaluation.get("likely_classification", None)
            if (likely_novelty_classification != "somewhat-related-to-existing") and (likely_novelty_classification != "new"):
                print("Skipping theory statement with original (from the theory generation model) likely_novelty_classification = " + str(likely_novelty_classification))
                continue

            packed_statement = {
                "theory_name": components.get("theory_name", None),
                "theory_description": components.get("theory_description", None),
                "statement_name": statement.get("statement_name", None),
                "theory_statement": statement.get("theory_statement", None),
                "domain_scope": statement.get("domain_scope", None),
                "special_cases": statement.get("special_cases", None),
            }

            # Repack with the theory query
            repacked = {
                "theory_id": theory.get("id", None),
                "theory_statement_idx": idx,
                "theory_query": theory.get("theory_query", None),
                "theory_statement_law": packed_statement,
            }
            theory_statements.append(repacked)
            print(json.dumps(repacked, indent=4))
            print("")

    print("Loaded " + str(len(theory_statements)) + " theory statements from TheoryStore.")
    return theory_statements



# This is the main control function to evaluate the novelty of a theory statement based on a set of input reference papers.
def evaluate_novelty_theory_statement_original_papers(path_out:str, theory_statement_dict:dict, theory_query_to_paper_ids:dict, paperstore:None, model_str_extraction:str="gpt-5-mini", model_str_decision:str="claude-sonnet-4-5-20250929"):
    print("evaluate_novelty_theory_statement_original_papers: started...")
    print("evaluate_novelty_theory_statement_original_papers: input theory dictionary:")
    print(json.dumps(theory_statement_dict, indent=4))

    theory_query = theory_statement_dict.get("theory_query", None)
    if (theory_query is None):
        print("ERROR: theory_query is None in theory_statement_dict.")
        return None

    # First, find the original theory record in the TheoryStore, so we have access to it's original extraction results/papers.
    theory_statement_law = theory_statement_dict.get("theory_statement_law", None)
    if (theory_statement_law is None):
        print("ERROR: theory_statement_law is None.")
        return None

    # Now, get the theory ID, and the specific statement we're searching for.
    theory_id = theory_statement_dict.get("theory_id", None)
    theory_statement = theory_statement_law.get("theory_statement", None)

    if (theory_id is None) or (theory_statement is None):
        print("ERROR: theory_id or theory_statement is None.")
        return None

    # Find the index of the specific statement in the original theory
    statement_index = -1
    statement_index = theory_statement_dict.get("theory_statement_idx", -1)

    if (statement_index == -1):
        print(f"ERROR: theory statement not found in original theory with ID {theory_id}.")
        return None

    # Make a verbose filename for output, that includes the theory ID, statement IDX, and a timestamp (YYYYMMDD-HHMMSS)
    timestamp_str = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    filename_out = f"novelty-evaluation-theory-{theory_id}-statement-idx-{statement_index}-{timestamp_str}.json"
    filepath_out = os.path.join(path_out, filename_out)

    # Use the theory_query_to_paper_id mapping to get paper IDs
    paper_ids = theory_query_to_paper_ids.get(theory_query, set())

    print(f"Found {len(paper_ids)} unique paper IDs supporting theory ID {theory_id}.")

    if (len(paper_ids) == 0):
        print("No paper IDs found; skipping novelty evaluation.")
        # Put a blank output file
        print("Writing: " + filepath_out)
        with open(filepath_out, "w") as f_out:
            #json.dumped({"error": "No paper IDs found for novelty evaluation."}, f_out, indent=4)
            json.dumps({"error": "No paper IDs found for novelty evaluation."}, f_out, indent=4)

        return None

    # Get the paper texts from the paperstore
    paper_texts = {}
    missing_paper_ids = []
    for paper_id in paper_ids:
        paper_text = paperstore.get_paper_text_by_key(paper_id)
        if (paper_text is None):
            print(f"WARNING: paper text for paper ID {paper_id} not found in PaperStore.")
            missing_paper_ids.append(paper_id)
            continue
        paper_texts[paper_id] = paper_text

    print(f"Retrieved {len(paper_texts)} paper texts from PaperStore.")
    if (len(missing_paper_ids) > 0):
        print(f"Missing {len(missing_paper_ids)} paper texts from PaperStore.")

    # This debug limit is to restrict the number of papers per evaluation (for debugging)
    DEBUG_LIMIT = None
    if (DEBUG_LIMIT is not None):
        paper_texts = {k: paper_texts[k] for k in list(paper_texts.keys())[:DEBUG_LIMIT]}
        print(f"DEBUG: Limiting to first {DEBUG_LIMIT} papers for novelty evaluation.")

    # Now, evaluate novelty for each paper
    all_novelty_evaluations = []
    # Parallel version
    num_workers = 50        # Number of parallel threads (using the inexpensive model for extraction)
    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_paper_id = {
            executor.submit(
                evaluate_novelty_one_paper,
                theory_law_dict=theory_statement_law,
                paper_text=paper_text,
                model_str=model_str_extraction,
                temperature=0.0,
                use_reflection=True,
                idx=paper_id
            ): paper_id
            for paper_id, paper_text in paper_texts.items()
        }

        from tqdm import tqdm
        for future in tqdm(as_completed(future_to_paper_id), total=len(future_to_paper_id)):
            paper_id = future_to_paper_id[future]
            try:
                novelty_evaluation = future.result()
                all_novelty_evaluations.append({
                    "paper_id": paper_id,
                    "novelty_evaluation": novelty_evaluation,
                })
                print(f"Completed novelty evaluation for paper ID {paper_id}.")
                total_cost += novelty_evaluation.get("total_cost", 0.0)
            except Exception as exc:
                print(f"ERROR: Paper ID {paper_id} generated an exception: {exc}")

            # Save intermediate results after each paper
            print("Saving: " + filepath_out)
            with open(filepath_out, "w") as f_out:
                packed = {
                    "model_str_extraction": model_str_extraction,
                    "model_str_decision": model_str_decision,
                    "theory_id": theory_id,
                    "total_cost": total_cost,
                    "num_papers_evaluated": len(all_novelty_evaluations),
                    "statement_index": statement_index,
                    "theory_query": theory_query,
                    "theory_statement_law": theory_statement_law,
                    "theory_statement_dict": theory_statement_dict,
                    "all_novelty_evaluations": all_novelty_evaluations,
                }
                json.dump(packed, f_out, indent=4)

            print("Total Cost: " + str(round(total_cost, 4)))


    # Now, for each dimension, collect the novelty evaluations, and separately evaluate them.
    dimension_keys = ["phenomenon_effect", "explanatory_mechanistic", "unification", "generalization_scope_expansion", "constraint_limitation", "conceptual_reframing_abstraction", "empirical_synthesis_meta_regulariry"]
    evidence_for_dimension = {}
    for dim_key in dimension_keys:
        evidence_for_dimension[dim_key] = []
    for novelty_eval_record in all_novelty_evaluations:
        novelty_evaluation = novelty_eval_record.get("novelty_evaluation", {})
        if (novelty_evaluation is None):
            continue
        best_response = novelty_evaluation.get("best_response", {})
        if (best_response is None):
            continue
        novelty_eval_dict = best_response.get("novelty_evaluation", {})
        if (novelty_eval_dict is None):
            continue
        paper_title = novelty_eval_dict.get("paper_title", None)
        for dim_key in dimension_keys:
            try:
                dim_eval = novelty_eval_dict.get(dim_key, None)
                if (dim_eval is not None):
                    evidence_for_dimension[dim_key].append({
                        "paper_title": paper_title,
                        "dimension_evaluation": dim_eval,
                    })
            except Exception as e:
                print(f"ERROR processing dimension {dim_key} for paper {paper_title}: {e}")

    # Within each dimension, we should also sort by degree_of_novelty, so that the strongest evidence of not-novelty is first.
    # `explicit_established`, `explicit_peripheral`, `derivable_unstated`, `genuinely_new`
    # Let's convert to a numeric scale for sorting:
    degree_of_novelty_order = {
        "explicit_established": 1,
        "explicit_peripheral": 2,
        "derivable_unstated": 3,
        "genuinely_new": 4,
    }
    # Add numeric degree to each evaluation
    for dim_key in dimension_keys:
        for eval_record in evidence_for_dimension[dim_key]:
            dim_eval = eval_record.get("dimension_evaluation", {})
            degree_of_novelty = dim_eval.get("degree_of_novelty", None)
            numeric_degree = degree_of_novelty_order.get(degree_of_novelty, -1)
            eval_record["numeric_degree_of_novelty"] = numeric_degree
        # Now sort
        evidence_for_dimension[dim_key] = sorted(evidence_for_dimension[dim_key], key=lambda x: x.get("numeric_degree_of_novelty", -1))

    # Store the collected evidence for each dimension
    packed = {
        "model_str_extraction": model_str_extraction,
        "model_str_decision": model_str_decision,
        "theory_id": theory_id,
        "total_cost": total_cost,
        "num_papers_evaluated": len(all_novelty_evaluations),
        "statement_index": statement_index,
        "theory_query": theory_query,
        "theory_statement_law": theory_statement_law,
        "evidence_for_dimension": evidence_for_dimension,
        "all_novelty_evaluations": all_novelty_evaluations,
    }
    print("Writing: " + filepath_out)
    with open(filepath_out, "w") as f_out:
        json.dump(packed, f_out, indent=4)

    # Now, perform the aggregation step for each dimension
    aggregated_novelty_evaluations = {}
    num_workers = 7
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_dim_key = {
            executor.submit(
                evaluate_novelty_aggregated,
                theory_law_dict=theory_statement_law,
                aggregated_results=evidence_for_dimension[dim_key],
                model_str=model_str_decision,
                temperature=0.0,
                max_tokens=16384,
                use_reflection=True,
                idx=f"{theory_id}-statement-{statement_index}-dimension-{dim_key}"
            ): dim_key
            for dim_key in dimension_keys
        }

        from tqdm import tqdm
        for future in tqdm(as_completed(future_to_dim_key), total=len(future_to_dim_key)):
            dim_key = future_to_dim_key[future]
            try:
                aggregated_novelty_evaluation = future.result()
                aggregated_novelty_evaluations[dim_key] = aggregated_novelty_evaluation
                print(f"Completed aggregated novelty evaluation for dimension: {dim_key}.")
                total_cost += aggregated_novelty_evaluation.get("total_cost", 0.0)
            except Exception as exc:
                print(f"ERROR: Dimension {dim_key} generated an exception: {exc}")

    packed["aggregated_novelty_evaluations"] = aggregated_novelty_evaluations
    packed["total_cost"] = total_cost

    # Try to generate a dictionary that's a clean final assessment of the aggregated novelty evaluations
    final_aggregated_novelty_evaluation = {}
    for dim_key in dimension_keys:
        aggregated_eval = aggregated_novelty_evaluations.get(dim_key, None)
        if (aggregated_eval is not None):
            best_response = aggregated_eval.get("best_response", {})
            novelty_evaluation = best_response.get("novelty_evaluation", {})
            final_aggregated_novelty_evaluation[dim_key] = novelty_evaluation.get(dim_key, None)
        else:
            final_aggregated_novelty_evaluation[dim_key] = None

    # Store the final aggregated novelty evaluation
    packed["final_aggregated_novelty_evaluation"] = final_aggregated_novelty_evaluation
    print("Writing final aggregated novelty evaluation: " + filepath_out)
    with open(filepath_out, "w") as f_out:
        json.dump(packed, f_out, indent=4)

    return packed




#
#   Load baseline theories (and match with papers)
#

def load_baseline_theories_from_theorystore_and_papermatch(filename_in:str, theory_idx_start:int=0, theory_idx_end:int=0):
    # Load the theorystore
    theories = None
    extraction_schemas = None
    extraction_results = None
    with open(filename_in, "r") as f:
        theorystore_data = json.load(f)
        theories = theorystore_data['theories']
        extraction_schemas = theorystore_data['extraction_schemas']
        extraction_results = theorystore_data['extraction_results']

    print(f"Loaded {len(theories)} theories, {len(extraction_schemas)} extraction schemas, {len(extraction_results)} extraction results from TheoryStore: {filename_in}")

    # Now, extract the baseline theories in the specified index range
    baseline_theories = []
    theory_ids = list(theories.keys())
    if (theory_idx_end <= 0) or (theory_idx_end > len(theory_ids)):
        theory_idx_end = len(theory_ids)

    out = []
    for theory in theories.values():
        # Theory ID is of the form 'theory-XXX', where everything after the last '-' is the index
        theory_id = theory.get("theory_id", None)
        if (theory_id is None):
            continue
        just_number = theory_id.split('-')[-1]
        # Convert to int
        theory_index = -1
        try:
            theory_index = int(just_number)
        except:
            print(f"WARNING: Could not convert theory ID {theory_id} to index.")
            continue

        if (theory_index < theory_idx_start) or (theory_index >= theory_idx_end):
            continue
        baseline_theories.append(theory)

    print(f"Extracted {len(baseline_theories)} baseline theories from index range {theory_idx_start} to {theory_idx_end}.")


# Generate a look-up table that maps all theory queries to all paper IDs that were used to build them.
def generate_lut_theory_query_to_paper_ids(theories:dict, extraction_results:dict):
    # Make a dictionary mapping theory_queries to lists of paper IDs that test them.
    theory_query_to_paper_ids = {}

    # For each theory
    for theory in theories.values():
        # Get it's theory query
        theory_query = theory.get("theory_query", None)
        if (theory_query is None):
            continue

        all_theory_queries = []
        all_theory_queries.append(theory_query)
        # We'll also normalize the theory query by removing the trailing 'based on ...' if present.
        if (", based on the following results" in theory_query):
            # Remove it
            normalized_theory_query = theory_query.replace(", based on the following results", "").strip()
            all_theory_queries.append(normalized_theory_query)

        # Check if these theory queries are already in the dictionary
        for tq in all_theory_queries:
            if tq not in theory_query_to_paper_ids:
                theory_query_to_paper_ids[tq] = set()

        # Get its list of supporting evidence IDs
        supporting_evidence_ids = theory.get("supporting_evidence_ids", [])

        for supporting_evidence_id in supporting_evidence_ids:
            extraction_result = extraction_results.get(supporting_evidence_id, None)
            if (extraction_result is None):
                continue
            paper_id = extraction_result.get("paper_id", None)
            if (paper_id is None):
                continue
            # Add the paper ID to all relevant theory queries
            for tq in all_theory_queries:
                theory_query_to_paper_ids[tq].add(paper_id)

    # Convert sets to lists
    for tq in theory_query_to_paper_ids:
        theory_query_to_paper_ids[tq] = list(theory_query_to_paper_ids[tq])

    # Calculate the average number of paper IDs per theory query
    total_papers = sum(len(paper_ids) for paper_ids in theory_query_to_paper_ids.values())
    num_theory_queries = len(theory_query_to_paper_ids)
    average_papers_per_theory_query = total_papers / num_theory_queries if num_theory_queries > 0 else 0.0
    print(f"Average number of paper IDs per theory query: {average_papers_per_theory_query:.2f}")
    # Count how many theory queries have zero paper IDs
    num_zero_paper_ids = sum(1 for paper_ids in theory_query_to_paper_ids.values() if len(paper_ids) == 0)
    print(f"Number of theory queries with zero paper IDs: {num_zero_paper_ids} out of {num_theory_queries} total theory queries.")

    # Return the look-up table
    return theory_query_to_paper_ids




#
#   Qualified Novelty Evaluation Histograms
#

# Make a histogram of scores from all the novelty evaluations
def make_novelty_evaluation_binary_one_file(path_in:str):
    import numpy as np

    # Get a list of all the JSON files in the directory (starting with 'novelty-evaluation")
    filenames = [f for f in os.listdir(path_in) if f.startswith("novelty-evaluation") and f.endswith(".json")]
    print(f"Found {len(filenames)} novelty evaluation files in: {path_in}")

    # Collect all the final aggregated novelty evaluations
    total_cost = 0.0
    papers_per_evaluation = []
    final_aggregated_novelty_evaluations = []
    for filename in filenames:
        filepath = os.path.join(path_in, filename)
        with open(filepath, "r") as f:
            data = json.load(f)
            final_aggregated_novelty_evaluation = data.get("final_aggregated_novelty_evaluation", None)
            if (final_aggregated_novelty_evaluation is not None):
                final_aggregated_novelty_evaluations.append(final_aggregated_novelty_evaluation)
            # Also accumulate total cost
            file_total_cost = data.get("total_cost", 0.0)
            total_cost += file_total_cost
            # Also store papers per evaluation
            num_papers_evaluated = data.get("num_papers_evaluated", 0)
            papers_per_evaluation.append(num_papers_evaluated)

    # Calculate the counts.
    dimension_keys = ["phenomenon_effect", "explanatory_mechanistic", "unification", "generalization_scope_expansion", "constraint_limitation", "conceptual_reframing_abstraction", "empirical_synthesis_meta_regulariry"]
    dimension_values = ["explicit_established", "explicit_peripheral", "derivable_unstated", "genuinely_new"]
    # Here, we'll remap the nuanced novelty ratings to binary ratings, since it's somewhat easier to interpret:
    remapped_dimension_values = {
        "explicit_established": "not_novel",
        "explicit_peripheral": "not_novel",
        "derivable_unstated": "novel",
        "genuinely_new": "novel",
    }

    histogram = {}
    # Initialize with zeros
    for dim_key in dimension_keys:
        #histogram[dim_key] = {val: 0 for val in dimension_values}
        histogram[dim_key] = {"not_novel": 0, "novel": 0}

    for aggregated_eval in final_aggregated_novelty_evaluations:
        for dim_key in dimension_keys:
            dim_eval = aggregated_eval.get(dim_key, None)
            if (dim_eval is not None):
                degree_of_novelty = dim_eval.get("degree_of_novelty", None)
                if (degree_of_novelty in dimension_values):
                    #histogram[dim_key][degree_of_novelty] += 1
                    # Remap
                    remapped_value = remapped_dimension_values.get(degree_of_novelty, None)
                    if (remapped_value is not None):
                        histogram[dim_key][remapped_value] += 1

    # Save a copy of the histogram, to get raw counts
    raw_count_histogram = json.loads(json.dumps(histogram))

    # Normalize to proportions
    for dim_key in dimension_keys:
        total_count = sum(histogram[dim_key].values())
        if (total_count > 0):
            for val in ["not_novel", "novel"]:
                histogram[dim_key][val] /= total_count


    packed = {
        "path_in": path_in,
        "histogram_proportions": histogram,
        "histogram_raw_counts": raw_count_histogram,
        "total_cost_all_evaluations": total_cost,
        "num_evaluations": len(final_aggregated_novelty_evaluations),
        "average_cost_per_evaluation": total_cost / len(final_aggregated_novelty_evaluations) if len(final_aggregated_novelty_evaluations) > 0 else 0.0,
        "average_papers_per_evaluation": float(np.mean(papers_per_evaluation) if len(papers_per_evaluation) > 0 else 0.0),
        "stddev_papers_per_evaluation": float(np.std(papers_per_evaluation) if len(papers_per_evaluation) > 0 else 0.0),
        "total_papers_evaluated": float(np.sum(papers_per_evaluation)),
    }

    print("Novelty Evaluation Histogram:")
    print(json.dumps(packed, indent=4))

    print("\n\n")

    # Also show the total cost, and the cost per evaluation
    num_evaluations = len(final_aggregated_novelty_evaluations)
    cost_per_evaluation = total_cost / num_evaluations if num_evaluations > 0 else 0.0
    print(f"Total Cost for all evaluations: ${total_cost:.2f}")
    print(f"Number of evaluations: {num_evaluations}")
    print(f"Average Cost per evaluation: ${cost_per_evaluation:.2f}")
    print(f"Average Number of papers evaluated per evaluation: {np.mean(papers_per_evaluation):.2f} ± {np.std(papers_per_evaluation):.2f}")
    print(f"Total papers evaluated across all evaluations: {np.sum(papers_per_evaluation)}")

    return packed


def make_novelty_evaluation_binary(path_in_baseline:str, path_in_theorizer:str, output_filename:str):
    histogram_baseline = make_novelty_evaluation_binary_one_file(path_in=path_in_baseline)
    histogram_theorizer = make_novelty_evaluation_binary_one_file(path_in=path_in_theorizer)

    packed = {
        "baseline": histogram_baseline,
        "theorizer": histogram_theorizer,
    }
    print("Writing binary novelty evaluation histogram comparison to: " + output_filename)
    with open(output_filename, "w") as f_out:
        json.dump(packed, f_out, indent=4)





#
#   Main qualified novelty evaluation
#

# This is a main high-level control function that takes a TheoryStore (and its associated PaperStore) as input, and performs qualified novelty evaluation on all theory statements in the TheoryStore.
def do_qualified_novelty_evaluation(filename_theorystore:str, filename_paperstore:str, path_out:str, model_str_extraction:str, model_str_decision:str, num_workers:int=5, DEBUG_LIMIT:int=2):
    # Make the output directory if it doesn't exist
    if (not os.path.exists(path_out)):
        os.makedirs(path_out)

    # Load the paperstore
    paperstore = PaperStore(filename_in=filename_paperstore)

    # Load the TheoryStore
    theories = None
    extraction_schemas = None
    extraction_results = None
    with open(filename_theorystore, "r") as f:
        theorystore_data = json.load(f)
        theories = theorystore_data['theories']
        extraction_schemas = theorystore_data['extraction_schemas']
        extraction_results = theorystore_data['extraction_results']

    print(f"Loaded {len(theories)} theories, {len(extraction_schemas)} extraction schemas, {len(extraction_results)} extraction results from TheoryStore: {filename_theorystore}")

    # Generate the look-up table mapping theory queries to paper IDs
    theory_query_to_paper_ids = generate_lut_theory_query_to_paper_ids(theories=theories, extraction_results=extraction_results)
    print(f"Generated look-up table mapping {len(theory_query_to_paper_ids)} theory queries to paper IDs.")

    # Load the theories that we're evaluating.
    theory_statements_to_evaluate_theorizer = load_theory_statements_to_evaluate_from_theorystore(filename_in=filename_theorystore)

    # Randomly shuffle the theory statements to evaluate, to better distribute the workload across workers.
    import random
    random.shuffle(theory_statements_to_evaluate_theorizer)
    print(f"Loaded {len(theory_statements_to_evaluate_theorizer)} theory statements to evaluate for qualified novelty.")
    # Debug limit
    if (DEBUG_LIMIT != None) and (DEBUG_LIMIT > 0):
        theory_statements_to_evaluate_theorizer = theory_statements_to_evaluate_theorizer[:DEBUG_LIMIT]
        print(f"DEBUG: Limiting to first {DEBUG_LIMIT} theory statements to evaluate for qualified novelty.")

    # Perform the qualified novelty evaluation, in parallel.
    print(f"Starting qualified novelty evaluations with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(
                evaluate_novelty_theory_statement_original_papers,
                path_out=path_out,
                theory_statement_dict=theory_statement_dict,
                theory_query_to_paper_ids=theory_query_to_paper_ids,
                paperstore=paperstore,
                model_str_extraction=model_str_extraction,
                model_str_decision=model_str_decision,
            ): idx
            for idx, theory_statement_dict in enumerate(theory_statements_to_evaluate_theorizer)
        }

        from tqdm import tqdm
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
            idx = future_to_idx[future]
            try:
                result = future.result()
                print(f"Completed novelty evaluation for theory statement {idx+1}.")
            except Exception as exc:
                print(f"ERROR: Theory statement {idx+1} generated an exception: {exc}")

    print("Completed all qualified novelty evaluations.")


#
#   Main Entry Point for Qualified Novelty Evaluation
#

if __name__ == "__main__":
    # Step 0: Load API Keys
    loadAPIKeys()

    # Step 1: Define the theories that we're evaluating (and the associated paperstore)
    filename_theorystore = "theorystore-example2-literaturesupported.json"
    filename_paperstore = "paperstore-example2-literaturesupported.json"

    # Define the extraction models
    model_str_extraction = "gpt-5-mini"
    model_str_decision = "claude-sonnet-4-5-20250929"

    # Define the output path
    path_out_literature_supported = "qualified-novelty-evaluations/literature-supported/"

    # The qualified novelty evaluation can be very expensive. Recommended to set a debug limit for testing.
    DEBUG_LIMIT = 5
    #DEBUG_LIMIT = 100

    # Step 2: Run the Qualified Novelty Evaluation
    print("Starting Qualified Novelty Evaluation...")
    do_qualified_novelty_evaluation(filename_theorystore=filename_theorystore,
                                   filename_paperstore=filename_paperstore,
                                   path_out=path_out_literature_supported,
                                   model_str_extraction=model_str_extraction,
                                   model_str_decision=model_str_decision,
                                   num_workers=5,                   # Number of parallel workers
                                   DEBUG_LIMIT=DEBUG_LIMIT
                                   )

    print("")
    print("Completed Qualified Novelty Evaluation.")
    print("")
    print("-" * 80)

    # Step 3: Perform analysis of the results
    histogram_literature_supported = make_novelty_evaluation_binary_one_file(path_in=path_out_literature_supported)

    filename_out = filename_theorystore.replace(".json", "-qualified-novelty-evaluation." + time.strftime("%Y%m%d-%H%M%S") + ".json")
    packed_overall = {
        "filename_theorystore": filename_theorystore,
        "filename_paperstore": filename_paperstore,
        "model_str_extraction": model_str_extraction,
        "model_str_decision": model_str_decision,
        "path_out_literature_supported": path_out_literature_supported,
        "histogram_literature_supported": histogram_literature_supported,
    }
    print("Writing overall qualified novelty evaluation summary to: " + filename_out)
    with open(filename_out, "w") as f_out:
        json.dump(packed_overall, f_out, indent=4)
