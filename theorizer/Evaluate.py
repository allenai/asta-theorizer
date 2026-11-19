# Evaluate.py
# The three evaluations in src/Evaluation*.py, run over the project's theories and saved like any
# other stage.  The src/ drivers read a one-law-per-entry file and write JSON files, so each wrapper
# writes the store out under <project>/evaluations/<kind>/<timestamp>/, runs the driver there, and
# reads its summary back.  The drivers print their own progress and results.
#
#   judge       LLM-as-a-judge: six 1-10 scores per law from the model's own knowledge
#   belief      self-assessed belief: the model's probability that each law is true, sampled N times
#   novelty     qualified novelty: each law against every paper in the PaperStore

import glob
import json
import os
import time

from . import Config
from . import Pipeline
from . import Project


def run_dir(kind:str):
    if (not Project.is_open()):
        raise ValueError("Evaluate." + kind + "(): no project is open, and its working files go in <project>/evaluations/. Open one first: th.Config.init(project_dir=...)")
    path = os.path.join(Project.DIRECTORY, "evaluations", kind, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(path, exist_ok=True)
    return path


def newest_run_dir(kind:str):
    runs = sorted(glob.glob(os.path.join(Project.DIRECTORY, "evaluations", kind, "*")))
    return runs[-1] if runs else None


def laws_file(store:dict, only_novel:bool, path:str):
    # The theory store on disk, and the one-law-per-entry file the judge and belief
    # drivers read.  only_novel keeps the laws the generator itself called new or somewhat new.
    import EvaluationCommon
    if (len(store.get("theories") or {}) == 0):
        raise ValueError("Evaluate: the store holds no theories")
    store_path = os.path.join(path, "theorystore.json")
    Pipeline.save_theorystore(store, store_path)
    laws_path = os.path.join(path, "laws.json")
    EvaluationCommon.convert_theorystore_to_single_laws_with_filtering(filename_theorystore=store_path, filename_out=laws_path, allow_all=(not only_novel), allow_new_and_somewhat_new=only_novel)
    with open(laws_path) as f:
        laws = json.load(f)
    if (len(laws) == 0):
        raise ValueError("Evaluate: no laws to evaluate" + (" -- none were self-assessed as new; drop only_novel" if only_novel else ""))
    return laws_path, laws


@Pipeline.stage_llm_limits
def judge(store:dict, model_str:str=None, only_novel:bool=False, num_workers:int=10, save_as:str=None):
    # LLM-as-a-judge: factual accuracy, specificity, novelty, testability, plausibility and empirical
    # testing, each 1-10, per law, from the model's knowledge alone.
    import EvaluationLLMAsAJudge
    Config.ensure_initialized()
    model_str = model_str or Config.DEFAULT_MODEL_PRIMARY
    path = run_dir("judge")
    laws_path, laws = laws_file(store, only_novel, path)
    results_path = os.path.join(path, "judge.json")
    averages, counts = EvaluationLLMAsAJudge.do_llm_law_evaluation_llm_as_a_judge(filename_in=laws_path, filename_out=results_path, model_str=model_str, num_workers=num_workers)
    data = {}
    if (os.path.exists(results_path)):
        with open(results_path) as f:
            data = json.load(f)
    total_cost = sum((result or {}).get("total_cost", 0.0) for result in (data.get("raw_results") or []))
    packed = {"model": model_str, "num_laws": len(laws), "averages": averages, "counts": {k: v.get("count") for k, v in counts.items()}, "results_path": results_path,
              "total_cost": total_cost}
    Pipeline.save_output(packed, "evaluation-judge", save_as)
    return packed


@Pipeline.stage_llm_limits
def belief(store:dict, model_str:str=None, only_novel:bool=False, num_samples:int=10, num_workers:int=20, limit:int=None, save_as:str=None):
    # Self-assessed belief: the model is asked num_samples times, at temperature 1, how likely each
    # law is to be true.  Reported as the mean probability.
    import EvaluationSurprisal
    Config.ensure_initialized()
    model_str = model_str or Config.DEFAULT_MODEL_PRIMARY
    path = run_dir("belief")
    laws_path, laws = laws_file(store, only_novel, path)
    EvaluationSurprisal.do_llm_law_evaluation_surprise(filename_in=laws_path, model_str=model_str, num_workers=num_workers, num_samples=num_samples, DEBUG_LIMIT=(limit or 0))
    outputs = sorted(glob.glob(os.path.join(path, "laws.surprise_evaluated.*.json")))
    data = {}
    if (outputs):
        with open(outputs[-1]) as f:
            data = json.load(f)
    packed = {"model": model_str, "num_laws": data.get("total_laws_scored", 0), "num_samples": num_samples,
              "mean_probability_true": data.get("avg_of_avg_probability_true"), "results_path": outputs[-1] if outputs else None,
              "total_cost": sum((result or {}).get("total_cost", 0.0) for result in (data.get("raw_results") or []))}
    Pipeline.save_output(packed, "evaluation-belief", save_as)
    return packed


@Pipeline.stage_llm_limits
def novelty(store:dict, paperstore=None, model_extraction:str=None, model_decision:str=None, num_workers:int=5, limit:int=None, save_as:str=None):
    # Qualified novelty: each law against every paper with full text in the PaperStore (the project's,
    # unless paperstore= names another), on seven dimensions, each judged novel or not.  Every input
    # paper counts, not only the papers its laws cite.  It is one call per paper per law, so the cost
    # grows with the PaperStore.  Only laws the generator itself called new or somewhat new are
    # checked.  limit picks that many at random.
    import EvaluationQualifiedNovelty
    Config.ensure_initialized()
    model_extraction = model_extraction or Config.DEFAULT_MODEL_EXTRACTION
    model_decision = model_decision or Config.DEFAULT_MODEL_PRIMARY
    if (len(store.get("theories") or {}) == 0):
        raise ValueError("Evaluate: the store holds no theories")
    paper_store = paperstore or Project.get_paper_store()
    paper_ids = [key for key in list(paper_store.paperstore.keys()) if paper_store.get_paper_text_by_key(key)]
    print("Evaluate.novelty(): comparing each law against all " + str(len(paper_ids)) + " paper(s) with full text in the PaperStore (one call per paper per law)", flush=True)
    result = EvaluationQualifiedNovelty.do_qualified_novelty_evaluation_persistence(theorystore_data=store, paperstore=paper_store,
                                                                                    model_str_extraction=model_extraction, model_str_decision=model_decision,
                                                                                    num_workers=num_workers, DEBUG_LIMIT=(limit or 0), paper_ids=paper_ids)
    verdicts = {key: value.get("final_aggregated_novelty_evaluation") for key, value in (result["theorystore_data"].get("qualified_novelty_evaluations") or {}).items()}
    if (len(verdicts) == 0):
        print("Evaluate.novelty(): WARNING: no law was scored.  Only laws the generator itself rated new or somewhat new are checked; if there were some, see the errors above", flush=True)

    # Everything the evaluation produced goes into the project, where the HTML export finds it:
    # one file per law (the verdict and every paper's evaluation behind it), and the store with
    # each evaluated law's verdict attached.
    novelty_dir = None
    if (Project.is_open()):
        novelty_dir = run_dir("novelty")
        statements_dir = os.path.join(novelty_dir, "statements")
        os.makedirs(statements_dir, exist_ok=True)
        for statement in (result.get("statement_results") or []):
            filename = "qn-" + str(statement.get("theory_id")) + "-statement-" + str(statement.get("statement_index")) + ".json"
            with open(os.path.join(statements_dir, filename), "w") as f:
                json.dump(statement, f, indent=2, default=str)
        Pipeline.save_theorystore(result["theorystore_data"], os.path.join(novelty_dir, "theorystore.json"))

    packed = {"model_extraction": model_extraction, "model_decision": model_decision, "num_laws": len(verdicts), "verdicts": verdicts,
              "novelty_dir": novelty_dir, "total_cost": result.get("total_cost")}
    Pipeline.save_output(packed, "evaluation-novelty", save_as)
    return packed
