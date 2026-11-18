# 07_TheoriesAndModels.py
# Generating theories: the two objectives, reflection, and choosing a model per stage.
#
# generate_theories() has two objectives, which are different generators in src/, not a prompt tweak:
#
#   accuracy   THE DEFAULT.  build_theory_from_results_single_theory_reflection4 -- evidence
#              grounded, literature supported; a newer revision of the accuracy-focused generator
#              the paper's experiments used.  What you want unless you know otherwise.
#   novelty    The novelty-focused generator.  ..._reflection4_nonsafebasin, deliberately pushed out
#              of the safe basin, so it proposes riskier theories.  Same temperature as accuracy;
#              the difference is the prompt.  Choose it on purpose.
#
# This example runs both so you can compare them -- that is not a recommendation to use novelty.
#
# use_reflection runs the second stage: after the first batch, each theory is reflected on in its
# own call.  That is one extra call per theory, each re-sending the full evidence, so it costs
# about 4.8x a single-pass run.  It is normally what you want.
#
#     python examples/07_TheoriesAndModels.py <project directory with records> [model]
#
# Run 01, 03 or 06 first to have records to work from.  Cost: about $0.30 with a cheap model.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


def show_theories(result, label:str):
    response = result.get("theory_response") or {}
    general = response.get("theories_general") or []
    specific = response.get("theories_specific") or []

    print("")
    print("--- " + label + ": " + str(len(general)) + " general, " + str(len(specific)) + " specific ($" + str(round(result.get("total_cost", 0.0), 2)) + ") ---")
    for theory in (general + specific)[:5]:
        print("  - " + str(theory.get("theory_name")))


def main():
    if (len(sys.argv) < 2):
        print("Usage: python examples/07_TheoriesAndModels.py <project directory> [model]")
        print("  (run 01, 03 or 06 first, then point this at its directory)")
        return 1

    model = sys.argv[2] if (len(sys.argv) > 2) else None
    th.Config.init(project_dir=sys.argv[1], model_primary=model, model_extraction=model)

    #
    #   Load the records and the plan that an earlier session saved
    #
    records = None
    query = None
    for item in th.Project.contents():
        if ("record" in item["name"]) and (records is None):
            records = th.Project.load(item["index"])
        if ("plan" in item["name"]) and (query is None):
            query = (th.Project.load(item["index"]) or {}).get("output")

    if (records is None) or (query is None):
        print("Could not find saved records and a plan in " + str(sys.argv[1]) + ".")
        th.Project.show()
        return 1

    print("Loaded " + str(len(records)) + " record(s)")
    theory_query = query["theory_request_normalized"]

    #
    #   Setting models: globally, and per stage
    #
    # Every stage that calls a model takes model_str.  Config.set_models() sets the default for the reasoning stages
    # (primary) and the per-paper stages (extraction); model_str on any single call overrides it.
    # So a cheap sweep with one expensive final pass costs almost nothing extra:
    print("")
    print("Current defaults: primary=" + str(th.Config.DEFAULT_MODEL_PRIMARY) + ", extraction=" + str(th.Config.DEFAULT_MODEL_EXTRACTION))

    #
    #   Accuracy: the standard generator
    #
    accurate = th.Pipeline.generate_theories(records, theory_query, objective="accuracy",
                                             use_reflection=False, save_as="theories-accuracy")
    show_theories(accurate, "accuracy")

    #
    #   Novelty: a different generator, pushed away from the safe answer
    #
    novel = th.Pipeline.generate_theories(records, theory_query, objective="novelty",
                                          use_reflection=False, save_as="theories-novelty")
    show_theories(novel, "novelty")

    #
    #   The same evidence, a different model, without touching the defaults
    #
    # This is the pattern worth remembering: sweep cheaply, then spend on the one call that matters.
    # OTHER_MODEL is an example; use any model your keys reach.
    OTHER_MODEL = "gpt-6-sol"
    if (model is not None) and (model != OTHER_MODEL):
        print("")
        print("Running the accuracy generator again on a different model, per-call...")
        other = th.Pipeline.generate_theories(records, theory_query, objective="accuracy",
                                              model_str=OTHER_MODEL, use_reflection=False,
                                              save_as="theories-accuracy-other-model")
        show_theories(other, "accuracy on " + OTHER_MODEL)

    #
    #   Reflection
    #
    # use_reflection=True adds the second stage, at about 4.8x the cost.  Everything above ran with
    # it off to keep this example cheap; leave it on for real runs.
    print("")
    print("All of the above ran with use_reflection=False to keep the example cheap.")
    print("For real runs leave it on -- it is the second stage that makes the theories worth having.")

    print("")
    th.Project.show()
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
