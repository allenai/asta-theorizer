# EvaluationCommon.py

import os
import json


from Struct import *
from PaperStore import *
from PaperFinderRequests import *
from Theorizer import *

from TheorizerProcessing import *

from ExtractionUtils import *



#
#   Filter Theory Laws
#

# Converts a TheoryStore (a JSON file containing theories) into a JSON file of individual theory statements.
# Also allows filtering based on self-generated novelty classification by the generation model.
#  - `allow_all`: If True, no novelty filtering is done, and all theory statements are included.
#  - `allow_new_and_somewhat_new`: If True, includes theory statements classified as "new" or "somewhat-related-to-existing".
# Only one of `allow_all` or `allow_new_and_somewhat_new` should be True; if both are False, only "new" statements are included.
def convert_theorystore_to_single_laws_with_filtering(filename_theorystore:str, filename_out:str, start_idx:int=0, end_idx:int=None, allow_all=False, allow_new_and_somewhat_new=False):
    # Load the theory store (as raw JSON)
    theorystore = None
    with open(filename_theorystore, "r") as f:
        theorystore = json.load(f)
    theories = theorystore.get("theories", {})

    print("Total theories in theorystore: " + str(len(theories)))

    # Get the subset of theories, based on their IDs (theory-0, theory-1, etc)
    selected_theories = []
    for theory_id, theory in theories.items():
        idx = int(theory_id.split("-")[-1])
        if (idx >= start_idx) and (end_idx is None or idx < end_idx):
            selected_theories.append(theory)

    print("Found " + str(len(selected_theories)) + " theories in the selected index range (" + str(start_idx) + " to " + str(end_idx) + ").")

    # Get the counts of the law's self-assessed novelty classification
    counts_novelty = {}
    counts_num_theory_statements = {}
    avg_num_theory_statements = 0

    num_statements_overall = 0
    avg_triples_per_statement_if = 0
    avg_triples_per_statement_then = 0
    total_theories = 0

    theory_statements_new = []

    for theory in selected_theories:
        components = theory.get("components", {})
        theory_statements = components.get("theory_statements", [])
        for statement in theory_statements:
            law = statement
            law_name = law.get("statement_name")
            theory_statement = law.get("theory_statement", "")
            domain_scope = law.get("domain_scope", "")
            special_cases = law.get("special_cases", [])
            law_supporting_evidence = law.get("supporting_evidence", [])
            law_qual_or_quant = law.get("qual_or_quant", "")
            law_existing_law = law.get("novelty_evaluation", {})
            law_likely_novelty_classification = law_existing_law.get("likely_classification", None)

            if (law_likely_novelty_classification != None):
                if (law_likely_novelty_classification not in counts_novelty):
                    counts_novelty[law_likely_novelty_classification] = 0
                counts_novelty[law_likely_novelty_classification] += 1

            avg_triples_per_statement_if += 0 # Not relevant here, not using triples
            avg_triples_per_statement_then += 0 # Not relevant here, not using triples
            num_statements_overall += 1

            # If the classification is "new", add to the list
            if (law_likely_novelty_classification == "new") or (allow_all == True) or ((allow_new_and_somewhat_new == True) and (law_likely_novelty_classification == "somewhat-related-to-existing")):
                packed_new_law = {
                    "theory_id": theory.get("id", ""),
                    "theory_name": components.get("theory_name", ""),
                    "theory_description": components.get("theory_description", ""),
                    "law": law,
                }
                theory_statements_new.append(packed_new_law)

        total_theories += 1

        # Counts for number of theory statements
        num_statements = len(theory_statements)
        if (num_statements not in counts_num_theory_statements):
            counts_num_theory_statements[num_statements] = 0
        counts_num_theory_statements[num_statements] += 1
        avg_num_theory_statements += num_statements

    # Average number of theory statements
    avg_num_theory_statements /= len(selected_theories)
    avg_triples_per_statement_if /= num_statements_overall
    avg_triples_per_statement_then /= num_statements_overall

    # Convert to proportions
    total_count = sum(counts_novelty.values())
    proportions = {k: v / total_count for k, v in counts_novelty.items()}
    print("============================================")
    print(" Summary Statistics")
    print("============================================")
    print("Number of theories: " + str(len(selected_theories)) + "   (Start idx: " + str(start_idx) + ", End idx: " + str(end_idx) + ")")
    print("Number of theory statements: " + str(num_statements_overall))
    print("")

    print("Law novelty classification counts (and proportions):  (Note, this is using the self-rated novelty from the generation model, and not the literature-based qualified novelty evaluation)")
    key_order = ["new", "somewhat-related-to-existing", "closely-related-to-existing", "existing", "unknown"]
    # Add any keys not in the key_order to the end
    for k in counts_novelty.keys():
        if (k not in key_order):
            key_order.append(k)
    # Show as a table with fixed width
    print(f"{'Classification':<30} {'Count':<10} {'Proportion':<10}")
    for k in key_order:
        print(f"{k:<30} {counts_novelty.get(k, 0):<10} {proportions.get(k, 0.0):<10.4f}")
    print("")

    # Also show other summary statistics
    print(f"Average number of theory statements per theory       : {avg_num_theory_statements:.2f}")

    print("Writing: " + filename_out + " with " + str(len(theory_statements_new)) + " theory statements.")
    # Sort by (integer-parsed) ID
    theory_statements_new = sorted(theory_statements_new, key=lambda x: int(x["theory_id"].split("-")[-1]))
    with open(filename_out, "w") as f:
        json.dump(theory_statements_new, f, indent=4)
