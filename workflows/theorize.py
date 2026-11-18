# theorize.py
# One theorizer run: papers in, literature-grounded theories out.
#
# HOW TO USE THIS FILE
#   Copy it into a fresh, empty directory, edit the PARAMETERS block below, and run it from there:
#
#       python theorize.py
#
#   Everything the run produces lands in that directory and nothing is ever overwritten -- the
#   stage files, the theory store, the paper store, the prompts, and the progress log.  That is
#   AGENTS.md's Rule 0, and it is why the run directory is wherever this file sits.
#
# WHAT TO EDIT
#   PARAMETERS, always.  STAGES, when the run needs a different shape: the calls there are the
#   stages of AGENTS.md's Workflow 1, in order, and the guide documents every one of them.  Leave
#   the two blocks either side of STAGES alone.
#
# WHAT IT COSTS
#   One LLM call to plan, one per paper to extract, and one per theory to write (plus one more
#   each if USE_REFLECTION is on, which roughly quintuples the cost of that last stage).  A
#   twenty-paper run on a cheap extraction model is small; a thousand-paper run is not.

import json
import os
import sys

import theorizer as th


#
#   PARAMETERS
#

# The question the theories have to answer.  One clear question; a vague one wastes the whole run.
QUESTION = "How do VIP interneurons control gain in mouse primary visual cortex?"

# Anything the theories must take into account that the question does not say by itself.  Optional.
MISSION = ""

# Where the papers come from: a directory, a glob or a list of .md/.txt files to read from disk.
# Convert PDFs to markdown first (or import them yourself with ocr="mistral").
# This build has no paper search; find papers with Semantic Scholar first.
PAPERS = None

# "accuracy" is the default, evidence-grounded generator.  "novelty" is the novelty-focused one,
# which proposes riskier theories; pick it on purpose or not at all.
OBJECTIVE = "accuracy"

# The second pass over each theory.  Normally what you want, and about 4.8x the cost of one pass.
USE_REFLECTION = True


# The run directory: this file's own, so a copy of this file IS a run.
RUN_DIR = os.path.dirname(os.path.abspath(__file__))


#
#   SETUP -- leave this alone
#

# Under --plan, only the plan goes to stdout; the stages' own chatter goes to stderr.
PLAN_ONLY = ("--plan" in sys.argv)
if (PLAN_ONLY):
    STDOUT, sys.stdout = sys.stdout, sys.stderr

# cwd first, then init: the engine writes its prompt dumps into the working directory, and the
# project directory is absolute so nothing depends on where this was launched from.
os.makedirs(RUN_DIR, exist_ok=True)
os.chdir(RUN_DIR)
th.Config.init(project_dir=RUN_DIR)


#
#   STAGES -- edit from here
#

# 1. The plan: a search query, an extraction query and an extraction schema, from the question.
#    Made once -- by `--plan`, so it can be checked before anything else is paid for, or here if
#    it was not -- and reused after, so what runs is what was shown.
try:
    plan = th.Project.load("plan-theory-query")
except (FileNotFoundError, ValueError):
    plan = th.Pipeline.plan_theory_query(QUESTION, mission_statement=MISSION)
query = plan["output"]

# `python theorize.py --plan` prints the stages and the settings each will get, as JSON, and exits.
if (PLAN_ONLY):
    stages = [{"stage": "plan_theory_query", "config": {"question": QUESTION, "mission_statement": MISSION}}]
    stages.append({"stage": "import_papers", "config": {"papers": PAPERS}})
    stages += [
        {"stage": "extract_from_papers", "config": {"extraction_query": query["extraction_query"], "extraction_schema": query["extraction_schema"]}},
        {"stage": "generate_theories", "config": {"theory_request": query["theory_request_normalized"], "objective": OBJECTIVE, "use_reflection": USE_REFLECTION}},
    ]
    print(json.dumps(stages, indent=2), file=STDOUT)
    sys.exit(0)

# 2. Get the papers from disk.
if (PAPERS is None):
    sys.exit("theorize.py: set PAPERS to a directory, a glob or a list of files (this build has no paper search)")
found = th.PaperImport.import_papers(PAPERS)
papers = found["papers"]

# 3. Pull the findings out of every paper, then flatten them into the one list the later stages
#    take.  Each record carries a uuid, which is how a law cites its evidence; the uuid leads to the
#    store's extraction result, which names the paper.
extraction = th.Pipeline.extract_from_papers(papers, query["extraction_query"], query["extraction_schema"])
records = th.Pipeline.flatten_records(extraction)

# 4. Write the theories, and register them in the project's theory store.
theories = th.Pipeline.generate_theories(
    records,
    query["theory_request_normalized"],
    objective=OBJECTIVE,
    use_reflection=USE_REFLECTION,
)
# Both kinds, each tagged: the generator returns general laws and specific ones, and keeping only
# the general half silently throws away a large part of what the run paid for.
found = []
for kind in ("general", "specific"):
    for theory in (theories["theory_response"].get("theories_" + kind) or []):
        theory["type"] = kind
        found.append(theory)

store = th.Pipeline.to_theorystore(
    found,
    query["theory_request_normalized"],
    extraction_schema=query["extraction_schema"],
)

#   STAGES -- to here


#
#   RESULT -- leave this alone
#

# The numbers worth reporting, because a run can lose papers without saying so: a file that could
# not be read is not imported, and a paper whose response fails to parse gives no records.  A
# corpus of 20 can reach theory formation as 9.
full_text = sum(1 for paper in papers if (paper.get("status", "full-text") == "full-text"))
citable = sum(1 for record in records if (record.get("uuid")))

result = {
    "question": QUESTION,
    "papers": len(papers),
    "papers_with_full_text": full_text,
    "records": len(records),
    "records_citable": citable,
    "theories": len(store.get("theories") or {}),
    "cost": round(plan["total_cost"] + extraction["total_cost"] + theories["total_cost"], 4),
    "run_dir": RUN_DIR,
}

with open(os.path.join(RUN_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

# The theory store to share: to_theorystore()'s, which carries the extraction schema its evidence
# is laid out by, plus each paper's title, authors, venue and abstract so every finding names its
# paper.  Paper text stays out.  The project's own theorystore.json has no paper details.
def paper_details(paper):
    # An imported paper keeps these at the top level; a fetched one in its Semantic Scholar record
    s2 = paper.get("s2_metadata") or {}
    authors = paper.get("authors") or s2.get("authors") or []
    return {
        "authors": [author.get("name") if isinstance(author, dict) else author for author in authors],
        "venue": paper.get("venue") or s2.get("venue"),
        "abstract": paper.get("abstract") or s2.get("abstract"),
    }


store["paperstore"] = {
    paper["paper_id"]: {
        "title": paper.get("title"),
        "corpus_id": int(paper["corpus_id"]) if str(paper.get("corpus_id") or "").isdigit() else None,
        "publication_year": paper.get("year"),
        "s2_metadata": paper_details(paper),
    }
    for paper in papers if paper.get("paper_id")
}
with open(os.path.join(RUN_DIR, "theory-store.json"), "w") as f:
    json.dump(store, f)

print("")
print("theorize.py: " + str(result["theories"]) + " theor(y/ies) from " + str(result["records"]) + " record(s)")
print("theorize.py: " + str(result["papers"]) + " paper(s), " + str(full_text) + " with full text; " + str(citable) + " record(s) can cite their evidence")
print("theorize.py: $" + str(result["cost"]) + " -- everything is in " + RUN_DIR + "; theory-store.json is the store to share")
