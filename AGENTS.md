# theorizer — guide

A thin library over the Theorizer code in `src/`, so the stages can be run one at a time instead of
as one pipeline. Every stage calls the function in `src/`; nothing here reimplements prompts,
parsing, or data structures. To see what a call really runs, look the name up in
`theorizer/Legacy.py`.

The output shown below is trimmed from real runs; model names in it are the defaults
(`gpt-6-sol`, `gpt-6-luna`).

---

## Rule 0 — everything goes in the working directory

**Every artifact a run produces goes into that run's working directory. Nothing goes anywhere else
— not `/tmp`, not a scratch directory, not a file next to the script.**

One run, one directory, holding everything needed to reproduce and inspect it: the stage outputs,
the theorystore, the paperstore, and the HTML export. The engine's own files go there too: the
prompt dumps (`prompts/`), the verbose log, the OCR cache (`ocr-cache/`, `data/`) and the Semantic
Scholar cache (`s2_cache.json`). Nothing is written beside the installed library. Only with
`project_dir=None` do these go in the working directory instead.

**A single run can cost upwards of $250.** An output written anywhere else is one that a later run
overwrites, or the system cleans up — and then the money is spent again, if the result can be got
back at all. So:

- **One directory per run.** Two runs sharing a directory, or a filename, means the second destroys
  the first.
- **Never invent a path** for a run's output. `Project.save()` numbers files in order and never
  overwrites; that is the point of it.
- **Go through the stages.** `Pipeline.generate_theories()` is what saves. Calling a
  `build_theory_*` function in `src/` directly returns the result and writes **nothing**. If a
  generator you want is not reachable through `Pipeline`, wire it in — do not work around it.
- **Check.** Opening a project does not save anything by itself. If a run finishes and the
  directory is empty, the saving did not happen.

Anything that took an LLM call to produce is expensive and unreproducible. It goes in the working
directory *before* anything else is done with it.

---

## Start

```python
import theorizer as th
th.Config.init(project_dir="my-run")
```

`init()` finds your API keys, and opens `my-run/` where every stage saves its output — see
[Rule 0](#rule-0--everything-goes-in-the-working-directory). Use a **new directory per run**.

**Keys.** A key already in the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`,
`SEMANTIC_SCHOLAR_API_KEY` or `S2_API_KEY`, ...) is used as is. For the rest, `init()` reads
`api_keys.donotcommit.json` (`{"openai": "sk-...", "anthropic": "sk-ant-...", "mistral": "..."}`) and
`s2_key.donotcommit.txt`, from the first place it finds them: `key_dir=`, the working directory,
`~/.config/asta/`, then the repository checkout (or the installed package). It prints each key's
source (environment or file), never the key. The default models are OpenAI's; with only an
Anthropic key, set them with `th.Config.set_models()` (see [Models](#models)).

**A LiteLLM proxy.** With `LITELLM_BASE_URL` and `LITELLM_USER_KEY` both set, every call goes through
that proxy: `OPENAI_API_KEY` is set to the proxy key (replacing any other), and model names get an
`openai/` prefix. Name the proxy's models with `THEORIZER_MODEL_PRIMARY` / `THEORIZER_MODEL_EXTRACTION`.

---

## The four things that flow between stages

Learn these and the rest is obvious. Everything is plain JSON.

### 1. A **paper**

```python
{"paper_id":  "paper-ahn2017-3f9c21ab",      # filename stem + a hash of the file
 "title":     "Immunoreactivities of calbindin-D28k, calretinin and parvalbumin ...",
 "year":      2017,
 "venue":     "Molecular Medicine Reports",
 "authors":   [...],
 "abstract":  "...",
 "markdown":  "# Introduction\nThe rodent somatosensory cortex ..."}
```

From `PaperImport.import_papers()` (your files) or
`Pipeline.fetch_papers()` (corpus ids).  Both produce the same shape, so everything downstream is
identical.  (This build has no paper search. Search elsewhere, with the
plan's `paper_search_query`, and fetch what it finds by corpus id.)

Papers from `fetch_papers()` also carry `corpus_id`, `s2_paper_id`, `paper_urls`, `status` and
`s2_metadata` (the Semantic Scholar record), and are kept in the project's
[PaperStore](#the-paperstore--the-papers-themselves-per-project). Imported papers have no Semantic
Scholar ids unless you pass `lookup_semantic_scholar=True`.

### 2. A **schema slot** — one thing to extract

```python
{"name":        "study_species",
 "type":        "str",
 "description": "The animal species and preparation used in the study, such as mouse, rat, ...
                 include awake versus anesthetized and in vivo versus ex vivo when reported."}
```

A **schema** is just a list of these. A **symbolic** slot adds `inclusion_criteria`,
`exclusion_criteria`, `unit`, and `categorical_values` — see [Symbolic](#workflow-3--symbolic-laws).

### 3. An **extraction record** — one entity extracted from one paper

`src/` calls these `extracted_records`; `ExtractionQueryResult.extracted_data` is the list of
them for a single paper. Shortened to "record" below.

```python
{"name_short":        "VIP interneurons",
 "name_full":         "Vasoactive intestinal peptide-expressing cortical interneurons",
 "brief_description": "The paper describes VIP interneurons as heterogeneous GABAergic cells ...",
 "citation_title":    "here",
 "study_species":     "mouse",           # <- one key per schema slot
 "visual_cortex_region": "V1",
 "uuid":              "e3.1"}            # <- stamped by the store; laws cite it
```

A record does not carry its paper. Its uuid does: each `store.extraction_results[...]` holds one
paper's records under its `paper_id`.

### 4. A **theory**

```python
{"theory_name": "Conditional VIP-SST Disinhibitory Gain",
 "theory_name_plain_language": "VIP cells boost responses mainly by releasing SST inhibition",
 "theory_description": "VIP interneurons regulate visual-cortical gain indirectly: when visual or
                        contextual inputs recruit VIP cells, VIP inhibition of SST interneurons ...",
 "theory_statements": [...],
 "new_predictions_likely": [...], "new_predictions_unknown": [...],
 "negative_experiments": [...],   "special_cases": [...]}
```

---

## The stages

Every stage returns a **packed dict**: what you gave it, plus its result, plus `total_cost`.
Stages that call a model take `model_str=`; stages that save take `save_as=`; the per-paper ones
also take `max_workers=`.

| stage | give it | get back |
|---|---|---|
| `plan_theory_query(request)` | a plain-language request | `["output"]` → `short_name`, `paper_search_query`, `extraction_query`, `extraction_schema` |
| `fetch_papers(candidates)` | dicts with a `corpus_id` (a title is optional) | `["papers"]` → the papers that got full text (PDFs go through Mistral OCR, so this needs `MISTRAL_API_KEY`); also fills and saves the project's **PaperStore** (`["paper_store_path"]`) |
| `PaperImport.import_papers(path)` | a dir/glob/file/list | `["papers"]` → list of papers |
| `extract_from_papers(papers, query, schema)` | papers + schema | `["results"]` → one entry per paper |
| `flatten_records(extraction)` | that result | a flat list of **extraction records** |
| `discover_variables(papers, query)` | papers | `["results"]` → per-paper `proposed_slots` |
| `make_schemas_from_variable_discovery(discovery, query)` | that result | `["output"]["extraction_schemas"]` → **new** schemas |
| `make_variable_report(records, query)` | records | `["output"]["suggestions"]` |
| `extend_schema_with_suggestions(schema, suggestions)` | schema + suggestions | a **bigger schema** (no LLM call) |
| `make_symbolic_schema(records, query)` | records | `["variables"]`, `["extraction_query"]` |
| `generate_theories(records, query)` | records | `["theory_response"]["theories_general"]` and `["theories_specific"]` |
| `generate_theories_over_types(records, query, output_types, inference_types)` | records | `["theories"]`, one generation per type (see [Sweeping by type](#sweeping-by-type)) |
| `to_theorystore(theories, query, model_str=, method=)` | theories | saves them to the project's theorystore, and returns a store dict with the theories, schemas and extraction results (no LLM call) |

> **The one irregularity:** theories arrive under `theory_response`, everything else under `output`.
> That is how `src/` returns them.

**Which stages call a model:** all of them except `fetch_papers`, `import_papers(..., read_metadata=False)`,
`flatten_records`, `extend_schema_with_suggestions` and `to_theorystore`. Those, and Workflow 5, need no
model key (`fetch_papers` needs `MISTRAL_API_KEY` to turn PDFs into text).

**Generated theories are not saved to the theorystore by themselves.** `generate_theories()` saves its
output file, but a theory reaches `theorystore.json` (what the exporter and reloads read) only through
`to_theorystore()`. Call it after every generation, with the model and generator, so the store says
who wrote each theory:

```python
to_theorystore(found, query["theory_request_normalized"], model_str=theories["model"], method="accuracy")
```

---

## Workflow 1 — the basic run

```python
import theorizer as th
th.Config.init(project_dir="vip-run")

plan  = th.Pipeline.plan_theory_query("Build a theory of how VIP interneurons control V1 gain")
query = plan["output"]

papers = th.PaperImport.import_papers("/path/to/papers")["papers"]

ext     = th.Pipeline.extract_from_papers(papers, query["extraction_query"],
                                          query["extraction_schema"])
records = th.Pipeline.flatten_records(ext)

theories = th.Pipeline.generate_theories(records, query["theory_request_normalized"])
found    = theories["theory_response"]["theories_general"] + theories["theory_response"]["theories_specific"]
th.Pipeline.to_theorystore(found, query["theory_request_normalized"],   # into the project's theorystore.json
                           model_str=theories["model"], method="accuracy")

for t in found:
    print(t["theory_name"])
```

```
Pipeline.plan_theory_query(): Planned 'Build a theory of how VIP...' ($0.002)
  saved: vip-run/01-plan-theory-query.json
Parallel.run_over_items(): 6/6 done (0 failed, 31s elapsed)
Pipeline.extract_from_papers(): 2 record(s) from 2/6 papers ($0.031)
  saved: vip-run/03-extraction.json
Pipeline.generate_theories(): 8 theories from 2 records ($0.07)
  saved: vip-run/05-theories.json
```

**How it hooks up:** `plan` produces the schema → `extract` consumes it → `flatten` turns the
per-paper results into one flat list of extraction records → `generate_theories` consumes those
records. Each arrow is one dict key.

`extract_from_papers()` returns results grouped **per paper**; `flatten_records()` unrolls them into
one list, read back from the store so every record has its uuid. Every later stage (variable
report, symbolic schema, theory generation) wants that flat list.

---

## Workflow 2 — improve the schema before spending at scale

The first schema is a guess made before the model has seen your papers. There are **two** ways to
improve it, and they are not interchangeable.

### (a) Ask the papers first — *before* extracting

```python
discovery = th.Pipeline.discover_variables(papers, query)     # one call per paper
schemas   = th.Pipeline.make_schemas_from_variable_discovery(discovery, query)

new = schemas["output"]["extraction_schemas"][0]              # a whole NEW schema
ext = th.Pipeline.extract_from_papers(papers, new["extraction_query"], new["extraction_schema"],
                                      extraction_schema_id="schema-2")
```

Give each schema its own `extraction_schema_id` (the first pass is `"schema-1"`), so the store can tell
whose records are whose. To build theories from each new schema, generate from its records and save
them as after the first pass:

```python
for n, new in enumerate(schemas["output"]["extraction_schemas"], start=2):
    ext      = th.Pipeline.extract_from_papers(papers, new["extraction_query"], new["extraction_schema"],
                                               extraction_schema_id=f"schema-{n}")
    theories = th.Pipeline.generate_theories(th.Pipeline.flatten_records(ext), query["theory_request_normalized"])
    found    = theories["theory_response"]["theories_general"] + theories["theory_response"]["theories_specific"]
    th.Pipeline.to_theorystore(found, query["theory_request_normalized"], model_str=theories["model"], method="accuracy")
```

Each paper answers in `discovery["results"][i]["proposed"]["proposed_slots"]`:

```python
{"slot_name":      "interneuron_marker_expression_and_calcium_buffering",
 "slot_description": "Extract changes in molecular markers of inhibitory interneurons ...",
 "impact_rating":  7}
```

### (b) Ask the records afterwards — *after* extracting

```python
report      = th.Pipeline.make_variable_report(records, query)
suggestions = report["output"]["suggestions"]
extended    = th.Pipeline.extend_schema_with_suggestions(query["extraction_schema"], suggestions)

ext2 = th.Pipeline.extract_from_papers(papers, query["extraction_query"], extended,
                                      extraction_schema_id="schema-2")
```

Each suggestion carries the slots to add:

```python
{"suggestion_short_name": "Define sign/target of VIP modulation",
 "suggested_slots_and_extraction_instructions": [
    {"slot_name": "vip_modulation_sign",
     "slot_description_and_extraction_instructions": "Extract whether VIP activation increases
                                                      or decreases the response."}]}
```

|  | (a) discovery | (b) report |
|---|---|---|
| when | before extracting | after extracting |
| gives you | **new** schemas | your **existing** schema, extended |
| model calls | one per paper, then one to design the schemas | one report call; extending the schema is free |
| use when | you want it redesigned around what's in the papers | you want the two rounds comparable |

Pilot on 2 papers, look, refine, *then* run at scale — that's the point of the library.

---

## Workflow 3 — symbolic laws

Theory generation writes prose. The symbolic path fits **equations** to a table of typed values.
Install the `symbolic` extra first (`pip install "asta-theorizer[symbolic]"`, or `uv sync --extra symbolic`
in a clone); without it the solver skips the tools that need scipy, scikit-learn or pandas.

```python
records  = th.Pipeline.flatten_records(th.Pipeline.extract_from_papers(papers, q, schema))

symbolic = th.Pipeline.make_symbolic_schema(records, query)      # typed, with units + vocabularies
typed    = th.Pipeline.flatten_records(
             th.Pipeline.extract_from_papers(papers, symbolic["extraction_query"],
                                             symbolic["variables"]))

table  = th.Symbolic.build_table(typed, symbolic)
result = th.Symbolic.run_solver(table, skip=th.Symbolic.SLOW_TOOLS)
print(th.Symbolic.report_to_markdown(result))
```

A symbolic variable is a normal slot plus constraints:

```python
{"name": "species", "type": "categorical",
 "inclusion_criteria": "Extract the species in which the relevant VIP-related result was
                        directly measured or modeled.",
 "exclusion_criteria": "Do not extract species mentioned only in background literature.",
 "categorical_values": ["mouse","rat","nonhuman_primate","human","other_mammal","other"]}
```

**Re-extracting is a normal `extract_from_papers()` call** — `src/SchemaExtractionQueue` detects a
symbolic schema by its `inclusion_criteria` and applies the typing rules itself.

**Values that don't match their declared type are dropped and counted, never repaired:**

```
Symbolic.build_table(): 62 row(s) from 62 record(s), 4 column(s)
Symbolic.build_table(): WARNING: vip_activation -- 1 value(s) did not match (e.g. not a number: 0.5 mV)
Symbolic.build_table(): WARNING: brain_state   -- 1 value(s) did not match (e.g. outside the vocabulary: asleep)
```

A repaired value becomes a confident wrong law. Read `table["coverage"]` before believing anything.

Output:

```
gain_change = 2*vip_activation + 1   [linear, R2=1.000]   (score 1.0, via functional_form)
corr(vip_activation, gain_change) = 1.000                 (score 1.0, via correlation)
```

---

## Workflow 4 — stop, and pick up next week

Nothing needs saving by hand. Every stage writes itself into the project directory, numbered, and
**nothing is ever overwritten** — three extractions leave three files to compare.

```python
th.Config.init(project_dir="vip-run")
th.Project.show()
```

```
Project.show(): vip-run (10 item(s))
  01  plan-theory-query        0.02 MB  24 extraction_schema  gpt-6-sol     $0.002
  03  extraction               0.03 MB  6 results             gpt-6-luna    $0.031
  04  records                  0.02 MB  2 record(s)
  05  theories                 0.31 MB  8 theories            gpt-6-sol     $0.07
  09  theories-novelty         0.07 MB  4 theories            gpt-6-sol     $0.008
```

```python
records = th.Project.load("records")        # or th.Project.load(4)
more    = th.Pipeline.generate_theories(records, query["theory_request_normalized"],
                                        objective="novelty", save_as="theories-novelty")
```

It is a directory of JSON files — `mv`, `cp`, `rm` all work, and `show()` stays correct.

A name matches every save with that name, and `load()` refuses a name that matches more than one.
Give a stage you will load back its own `save_as=` (`save_as="schemas-pass-2"`), or load by the
number `show()` lists.

### Long runs resume after a crash

`extract_from_papers`, `discover_variables`, `import_papers` and `generate_theories_over_types`
append each item's result the moment it lands. Re-run the same call and it continues:

```
Parallel.run_over_items(): Starting extraction over 934 item(s), 25 workers -- 900 already done, 34 to run
```

Verified by `kill -9` at 320/900: restarting resumed all 320. What is keyed, so a change runs again
instead of reusing a stale result:

- **extraction**: the query, schema, `extraction_schema_id`, model, `max_tokens`, `temperature`,
  `use_reflection`, and each paper's id and text
- **variable discovery**: the query, model, `additional_contextual_info`, `max_tokens`,
  `temperature`, `use_reflection`, and each paper's id and text
- **import**: `read_metadata`, the metadata model actually used, `ocr`, and each file's id, which
  carries a hash of its content (an edited file is read again; adding a file renames nothing)
- **type sweep**: the query, the records themselves, model, `temperature`, `max_tokens`,
  `use_reflection`, `mission_statement`, `max_laws_per_theory`, `additional_suggestions`

A rerun never adds the same results to the TheoryStore twice: a result with the same paper,
schema id and records reuses the stored one, uuids and all.

---

## Workflow 5 — write the records and theories yourself (no model calls)

For a coding agent that does the reading itself. You write the records and theories, and the store
calls below give the records their uuids and save the project, so the theories read back and export
to HTML as after a model run. No key needed.

```python
import theorizer as th

th.Config.init(project_dir="vip-run")
store = th.Project.get_theory_store()

# The person's markdown or text (PDFs: convert to markdown first, or pass ocr="mistral").  read_metadata=False skips the metadata model call, so each title is a markdown file's first "# " heading, else the filename
papers = th.PaperImport.import_papers("my-papers/", read_metadata=False)["papers"]   # [{"paper_id", "title", "markdown", ...}]
# Or open-access papers by Semantic Scholar corpus id, which calls no model (its PDFs go through Mistral OCR):
# papers = th.Pipeline.fetch_papers([{"corpus_id": "278481047"}])["papers"]

schema = [{"name": "study_species", "type": "str", "description": "The species studied"}]
store.add_extraction_schema(th.Legacy.ExtractionQuerySchema(
    schema=schema, extraction_query="VIP interneurons and gain", supporting_theory_ids=[], id="schema-1", model_str="written"))

uuids = {}
for paper in papers:                                  # read paper["markdown"], then write its records
    records = [{"name_short": "VIP gain", "name_full": "VIP activation and response gain",
                "brief_description": "VIP activation raised visual response gain.", "study_species": "mouse"}]
    result_id = store.add_extraction_result(th.Legacy.ExtractionQueryResult(
        paper_id=paper["paper_id"], extraction_schema_id="schema-1", extracted_data=records,
        potentially_relevant_new_papers=[], model_str="written"))
    uuids[paper["paper_id"]] = [record["uuid"] for record in store.extraction_results[result_id].extracted_data]
th.Project.save_theory_store()

first = next(iter(uuids.values()))[0]
theories = [{"theory_name": "VIP disinhibition", "theory_description": "VIP cells raise gain by inhibiting SST cells.",
             "theory_statements": [{"law": {"statement_name": "Disinhibition", "theory_statement": "VIP activity raises gain.",
                                            "supporting_evidence": [{"text": "VIP activation raised gain.", "uuids": [first]}]}}]}]
stored = th.Pipeline.to_theorystore(theories, "How do VIP interneurons control gain?", model_str="written", method="written")
print(list(stored["theories"]))                       # theory-1, ...; the store is saved

# Read a theory back by its id; in a later session, th.Config.init(project_dir="vip-run") reloads the store
theory = th.Project.get_theory_store().theories[list(stored["theories"])[0]]
print(theory.name, theory.supporting_evidence_ids, [s["law"]["statement_name"] for s in theory.components["theory_statements"]])

# The HTML export: the project's theorystore (and paperstore, if any) into vip-run/html/
th.Export.export_html()                               # returns vip-run/html/index.html
```

The shapes are the ones the engine asks a model for: a record in `src/SchemaExtractionQueue.py`
(`name_short`, `name_full`, `brief_description`, then one key per schema slot), a theory in
`src/TheorizerProcessing.py` (`theory_name`, `theory_description`, `theory_statements`). Wrap each
law as `{"law": {...}}`, the way the store keeps them.

These calls don't check what you wrote. A uuid that no record has links to nothing, and no error says so;
`add_extraction_result` adds the records again each time it's called. Check each law's uuids against
the records' before `to_theorystore`, and add each paper's records once.

---

## The TheoryStore — save everything to one, per project

**Assume you are saving everything to a TheoryStore.** The library now does this for you: opening a
project creates one, every extraction goes through it, and it is written to
`<project>/theorystore.json` and reloaded next session.

```python
th.Config.init(project_dir="vip-run")     # the store lives here; nothing else to do
```

### Why it matters

The store is what assigns ids — and `TheoryStore.set_id()` is what stamps a **uuid onto every
extracted record**. Those uuids are how a law cites its evidence. The theory-generation prompts ask
the model for *"the UUIDs that correspond to the specific pieces of evidence"*, so:

> **Records with no uuid produce theories with empty evidence lists, and no law can be traced back
> to the paper it came from.** The generation still succeeds and still costs money; the evidence
> chain is just silently gone.

The store also keeps ids unique. `save()` persists `all_ids` and `next_ids`, so reopening a project
**continues** the numbering:

```
session 1 -> theory-1, theory-2
session 2 -> theory-3, theory-4          (a separate process, same project)
session 3 -> theory-5, theory-6
```

Without that, every session would restart at `theory-1` and merging two runs would silently drop
theories with the same id.

### If you are not using a project

You can pass a store explicitly:

```python
store = th.Legacy.TheoryStore()
ext   = th.Pipeline.extract_from_papers(papers, q, schema, theory_store=store)
```

**Do not skip it.** A fresh store per run restarts ids at 1, so runs cannot be combined. Ids are
unique *within* a store, never across separate ones.

### Checking the chain held

```python
records = th.Pipeline.flatten_records(ext)
print(sum(1 for r in records if r.get("uuid")), "of", len(records), "records can be cited")
```

`extract_from_papers()` and `flatten_records()` both warn if any record lacks a uuid.

---

## The PaperStore — the papers themselves, per project

The TheoryStore holds the theories and the extracted records. The **PaperStore** holds the papers
those records came from: full text, and the Semantic Scholar record behind each one (title, venue,
year, abstract, tldr, url, corpus id).

Opening a project creates one, `fetch_papers()` fetches into it, and it is written to
`<project>/paperstore.json` and reloaded next session — the same arrangement as the TheoryStore.

```python
th.Config.init(project_dir="vip-run")     # the paperstore lives here too; nothing else to do

found = th.Pipeline.fetch_papers([{"corpus_id": "225095372"}, {"corpus_id": "244348308"}])
print(found["paper_store_path"])          # <project>/paperstore.json
```

Each open-access PDF goes through Mistral OCR. Without `MISTRAL_API_KEY` a paper is stored with its
Semantic Scholar record only, and `found["papers"]` leaves it out.

Because it is reloaded, a second fetch **adds to** the papers already fetched rather than starting
empty, and papers fetched last week are still there this week:

```
session 1 -> fetch_papers(...)   40 papers in the store
session 2 -> fetch_papers(...)   40 already there + whatever is new
```

You can reach it directly, and save it yourself if you have added to it outside a stage:

```python
store = th.Project.get_paper_store()      # the project's store, loaded from disk if it exists
th.Project.save_paper_store()             # returns the path it wrote
```

Passing `paper_store=` to `fetch_papers()` still overrides it, and with no project open you get a
fresh in-memory store, as before.

### Why it matters

`fetch_papers()` returns a **flattened extract** of each paper — `paper_id`, `title`, `markdown`,
`year`, `corpus_id`, `s2_paper_id`, `paper_urls`, `status`, `s2_metadata` — and that is what the
extraction stages consume. The store is the thing that holds everything else, and the thing the
HTML exporter reads paper metadata from.

### Exporting evidence pages with paper metadata

The export ships with the library. Given a project, it reads `theorystore.json`, and
`paperstore.json` when there is one, so each evidence page shows the paper's linked title, year,
venue, TL;DR and abstract. It writes into the run's own directory (`<project>/html`):

```bash
python -m theorizer.Export --project vip-run          # vip-run/html/index.html
```

```python
th.Export.export_html()                                # the open project
th.Export.export_html(store="vip-run/theorystore.json", out="vip-run/html")
```

`--store`, `--paperstore`, `--papers` and `--out` (and the same keywords in Python) name the files
explicitly. `paperstore.json` is written by `fetch_papers()`; a run that only imported its papers has
none, and `--papers <papers.jsonl>` gives titles only. Without either, pages show the paper id alone.
In a clone, `python export_theories_html.py` takes the same arguments.

---

## Settings

Three ways to set anything, in increasing precedence:

```python
th.Config.DEFAULT_MAX_WORKERS = 40                              # 1. change the default
th.Config.init(model_primary="gpt-6-sol", project_dir="vip-run") # 2. set it at startup
th.Pipeline.extract_from_papers(papers, q, s, max_workers=8)    # 3. override for one call
```

### `Config.init()` — everything you can set at startup

```python
th.Config.init(key_dir=None,                 # where to look for the key files first
               verbose=True,                 # print which API keys were loaded
               quiet_legacy_output=True,     # full ModelResponse dump -> log file (see below)
               log_path=None,                # where that dump goes (default <project_dir>/theorizer-verbose.log)
               project_dir="theorizer-run",  # where stages save; None turns saving off
               model_primary=None,           # default model for reasoning stages
               model_extraction=None,        # default model for per-paper stages
               model_repair=None)            # JSON-repair model (default: the extraction model)
```

### Defaults you can change

| setting | default | what it is |
|---|---|---|
| `Config.DEFAULT_MODEL_PRIMARY` | `gpt-6-sol` | planning, variable reports, schemas, theories |
| `Config.DEFAULT_MODEL_EXTRACTION` | `gpt-6-luna` | per-paper work, where cost multiplies |
| `Config.DEFAULT_MAX_WORKERS` | `25` | width of parallel per-paper work |
| `Config.DEFAULT_MAX_DATA_TOKENS` | `750000` | evidence per LLM call, before subsampling (see [How much evidence a call sees](#how-much-evidence-a-call-sees)) |
| `Config.DEFAULT_MAX_TOKENS_PER_PAPER` | `64000` | completion tokens for one paper's extraction or variable discovery, reasoning included; capped at the model's own output limit where litellm knows it |
| `Config.DEFAULT_EVIDENCE_SEED` | `0` | seed for that subsampling (its own generator, so reruns pick the same records); `evidence_seed=` on `generate_theories*` overrides |
| `Config.DEFAULT_LLM_TIMEOUT_SECONDS` | `3600` | before a call is abandoned and retried |
| `Config.DEFAULT_S2_SECONDS_BETWEEN_REQUESTS` | `1.1` | spacing of `import_papers(lookup_semantic_scholar=True)` lookups (see `S2_RATE_LIMIT`) |
| `Config.API_KEY_FILENAME` | `api_keys.donotcommit.json` | |
| `Config.S2_KEY_FILENAME` | `s2_key.donotcommit.txt` | |
| `PaperImport.DEFAULT_EXTENSIONS` | `.md .markdown .txt .pdf` | what `find_files()` picks up (a PDF needs `ocr="mistral"`) |
| `PaperImport.MAX_PDF_MB` | `25` | larger PDFs are refused |
| `PaperImport.MAX_INLINE_PDF_MB` | `18` | above this, upload instead of inlining |
| `PaperImport.S2_MAX_WORKERS` | `4` | S2 is rate-limited; kept narrow on purpose |
| `Symbolic.SLOW_TOOLS` | `pysr_symbolic`, `gplearn_symbolic` | the two slow solvers, usually skipped |
| `Project.DEFAULT_DIRECTORY` | `theorizer-run` | |
| `Project.THEORY_STORE_FILENAME` | `theorystore.json` | inside the project directory |
| `Project.PAPER_STORE_FILENAME` | `paperstore.json` | inside the project directory |
| `ExtractionUtils.VERBOSE` | `False` after `Config.init()` (`True` with `quiet_legacy_output=False`) | the full-response debug dump to the terminal (status lines always print) |
| `ExtractionUtils.VERBOSE_LOG_FILENAME` | `theorizer-verbose.log` | where that dump goes when off; `Config.init()` puts it in the project directory |
| `ExtractionUtils.PROMPTS_DIRECTORY` | `prompts` | every prompt and response, one file each; `Config.init()` puts it in the project directory |

`DEFAULT_LLM_TIMEOUT_SECONDS` is 3600 because `getLLMResponseJSON` defaults to 300, which is too
short for a call carrying 100k tokens of evidence, or for one paper's extraction on a slow model —
and a timed-out call is **still billed**, then retried, so a tight timeout costs money rather than
saving it.  It applies to every engine call that sets no limit of its own (planning, reports,
schemas, paper metadata) and to the per-paper stages (`extract_from_papers()`,
`discover_variables()`, which also take `llm_timeout_seconds=`).  `MAX_GENERATION_TIME_SECONDS` in the environment overrides it, read at call
time; `llm_timeout_seconds=` on any stage that calls a model overrides both for that stage.  Theory generation and the evaluations name fixed limits
inside `src/` (15 or 8 minutes; 300 s): they get the setting when one is given, otherwise the longer
of their own value and 3600 s.  Each stage saves the limit it ran with (`llm_timeout_seconds`) and
every model call that gave up (`call_failures`: model, prompt size, the engine function, the
reason).

### Environment variables

| variable | default | what it does |
|---|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, ... | unset | provider keys; one set here wins over the key file |
| `SEMANTIC_SCHOLAR_API_KEY` (or `S2_API_KEY`) | unset | the Semantic Scholar key; wins over `s2_key.donotcommit.txt` |
| `LITELLM_BASE_URL`, `LITELLM_USER_KEY` | unset | both set: every call goes through that LiteLLM proxy (see [Start](#start)) |
| `THEORIZER_MODEL_PRIMARY` | `gpt-6-sol` | the default reasoning model, read by `init()` (an argument wins) |
| `THEORIZER_MODEL_EXTRACTION` | `gpt-6-luna` | the default per-paper model, read by `init()` (an argument wins) |
| `S2_RATE_LIMIT` | `1.1` | **seconds between Semantic Scholar title lookups** in `import_papers(lookup_semantic_scholar=True)` — smaller is faster. `fetch_papers()` paces its own requests |
| `MAX_GENERATION_TIME_SECONDS` | `3600` | seconds before an LLM call is abandoned and retried (see above) |
| `THEORIZER_REASONING_EFFORT` | `high` | reasoning effort sent to reasoning models for the once-per-run stages: `none` (send none), `minimal`, `low`, `medium`, `high`, `xhigh` |
| `THEORIZER_EXTRACTION_REASONING_EFFORT` | `high` | the same, for extraction and variable discovery, and any call made with the extraction (or repair) model |
| `THEORIZER_MODEL_REPAIR` | the extraction model | the JSON-repair model |
| `THEORIZER_DATABASE_URL` | unset | a Postgres URL; when set, the OCR cache and other engine state go to that database instead of files. Needs `pip install psycopg2-binary`. The generic `DATABASE_URL` is ignored |

```bash
export S2_RATE_LIMIT=0.35        # ~3 requests/sec; needs an API key
```

Measured with a key: 45 sustained requests at 0.35s produced no 429s; a short burst at 0.2s also
none. Without a key leave it alone — unauthenticated is limited to about 1/sec. Read at call time,
so setting it in a notebook before the first lookup works.

### Models

```python
th.Config.set_models(primary="gpt-6-luna", extraction="gpt-6-luna")
th.Pipeline.generate_theories(records, q, model_str="gpt-6-sol")   # one call overrides
th.Config.set_models(repair="gpt-6-luna")   # JSON repair; default: the extraction model
```

The defaults are `gpt-6-sol` (primary) and `gpt-6-luna` (extraction).  The engine's JSON-repair pass asks `Config.get_repair_model()`, routed through the proxy
like every other call (`THEORIZER_MODEL_REPAIR` in the environment, `repair=` here).

Sweep cheaply, then spend on the call that matters.

### Verbosity

```python
th.Config.set_quiet(True)               # full ModelResponse dump -> log file
th.Config.set_quiet(False)              # dump to terminal too, for debugging
```

**Status lines always print regardless** — `Querying LLM model (...)`, `Prompt tokens`, running
cost, `Completed. Cost`. Only the block `src/` marks `# DEBUG: PRINT FULL RESPONSE` is affected.
Errors, warnings and retries always print. Safe to call mid-run; it just sets a flag.

### Per-call parameters shared by most stages

| parameter | what it does |
|---|---|
| `model_str` | the model for this call only |
| `max_workers` | parallel width (per-paper stages) |
| `save_as` | name this save, e.g. `"extraction-pilot-v2"` |
| `show_progress` | progress lines for this call |
| `max_tokens` | completion tokens |
| `temperature` | `0` for most stages, `1.0` for theory generation. Reasoning models (gpt-5/gpt-6, o-series, Claude) are sent none, so with the default models it does nothing |
| `use_reflection` | run the second refining pass; one extra call per theory (measured ~4.8x total cost) |
| `max_data_tokens` | evidence before subsampling (stages that take records) |

Notable per-stage ones: `fetch_papers(max_papers, paper_store, timeout_seconds)`; `import_papers(read_metadata, lookup_semantic_scholar,
ocr)`; `generate_theories(objective, max_laws_per_theory, mission_statement, evidence_seed)`;
`run_solver(skip, timeout, parallel, quiet, cfg)`; `build_table(include_paper_id)`;
`extend_schema_with_suggestions(max_suggestions)`.

---

## Running lots of papers

```python
ext = th.Pipeline.extract_from_papers(papers, q, schema, max_workers=25,
                                      save_as="extraction-full")
```

Same call for 2 papers or 2000 — only the number changes. One bad paper doesn't stop the rest:
`ext["failures"]` records what went wrong, `ext["results"]` has the other 997.

Each failure has a `reason`, and `ext["num_failures_by_reason"]` counts them:

| reason | what happened | what to do |
|---|---|---|
| `max_tokens` | the answer was cut off at `max_tokens` — often all of it spent reasoning | raise `max_tokens=` |
| `timeout` | three calls ran past the time limit | raise `llm_timeout_seconds=` |
| `prompt_too_long` | the paper does not fit the model's context | a model with a larger window |
| `unreadable_answer` | the model finished, but no JSON could be read from its answer | rerun; look at `prompts/` |
| `error` | the API refused or failed (the message says how) | read the message |

Re-running the same call retries only the papers that failed. A changed `max_tokens` is a new
call, so every paper runs again: to retry only the failures with a bigger budget, pass just those
papers.

---

## Evaluating theories

`th.Evaluate` scores the theories in a store (the dict `to_theorystore()` returns, or a
`theorystore.json` loaded with `json.load`). Each evaluation saves itself like a stage, and keeps its
working files under `<project>/evaluations/<kind>/<time>/`.

```python
store   = json.load(open("vip-run/theorystore.json"))
novelty = th.Evaluate.novelty(store)       # qualified novelty, against the papers each theory was built from
th.Export.export_html()                    # the theory pages now show each evaluated law's verdict
```

| evaluation | what it asks | cost |
|---|---|---|
| `novelty(store)` | for each law, does any of the papers behind it already say it? Seven dimensions (phenomenon, mechanism, unification, scope, limitation, reframing, meta-regularity), each `explicit_established`, `explicit_peripheral`, `derivable_unstated` or `genuinely_new` — or null when the papers say nothing that bears on it | one call per paper per law, with reflection: measured **$1.05–1.55 per law** at 100 papers; five laws run at once (`num_workers=5`), so 9 laws took 9 minutes |
| `judge(store)` | six 1–10 scores per law from the model's own knowledge | one call per law |
| `belief(store, num_samples=10)` | the model's probability that each law is true, sampled | `num_samples` calls per law |

**Only laws the generator itself rated new or somewhat new are scored for novelty** — on a
100-paper run, 9 of 23. `limit=` scores that many at random, for a cheap first look.
`novelty["verdicts"]` has each law's verdict. One file per law, with every paper's evaluation, goes
to `evaluations/novelty/<time>/statements/`, and the export reads the newest of those runs
(`--novelty DIR` names another).

---

## Worth knowing

### A mission statement

The papers are all Theorizer reads. Anything else the theories must answer to (a focus, results
computed elsewhere, the gaps to aim at) goes in as `mission_statement=`, on the plan and on every
generation:

```python
plan     = th.Pipeline.plan_theory_query(request, mission_statement=mission)
theories = th.Pipeline.generate_theories(records, query["theory_request_normalized"], mission_statement=mission)
```

A mission given only to `generate_theories()` steers the theories but not the search or the schema.

### Sweeping by type

`generate_theories_over_types()` asks for one kind of theory at a time: each output type (what the
theory claims: `NULL RESULT`, `CONTEXT DEPENDENCE`, ...) and each inference type (how it reasons:
`ABDUCTION (BEST EXPLANATION)`, ...). The names are `th.Legacy.OUTPUT_TYPES` and
`th.Legacy.INFERENCE_TYPES`; an unknown name is refused before any call. `None` runs every type of
that kind (26 runs in all, about 26x one generation), `[]` none.

```python
swept = th.Pipeline.generate_theories_over_types(records, query["theory_request_normalized"],
            output_types=["STATE OF THE LITERATURE", "NULL RESULT"], inference_types=[],
            mission_statement=mission)
th.Pipeline.to_theorystore(swept["theories"], query["theory_request_normalized"], model_str=swept["model"])
```

Each theory comes back with `requested_output_type` or `requested_inference_type`, and
`to_theorystore()` keeps it as the theory's method.

### How much evidence a call sees

Theory generation, the variable report and the symbolic schema each hand **all** their evidence to
one call. When the records are bigger than `max_data_tokens` (counted with the gpt-4 tokenizer), a
random sample of them is taken instead — the same sample on every rerun (`Config.DEFAULT_EVIDENCE_SEED`;
`evidence_seed=` draws another) — and the rest are never seen. The stage says so:

```
Pipeline.subsample(): Evidence subsampled to 90% (1067 of 1186 records) to fit 750000 tokens
```

and `packed["subsampling_rate"]` and `packed["num_records_used"]` record it.

The default, 750k, uses most of the default models' context (gpt-6-sol: 922k input; current Claude
models: 1M). The paper's experiments used 100k. Measured on one 100-paper corpus, accuracy objective,
reflection on:

| `max_data_tokens` | records seen | papers the theories cite | cost of the generation | time |
|---|---|---|---|---|
| 100k | 130 of 1,186 (11%) | 38 | $0.75 | ~3 min |
| 750k | 1,067 of 1,186 (90%) | 47 | $5.71 | ~4.5 min |

Most of the 750k cost is the first call writing the evidence to the prompt cache; each reflection
reads it back for about $0.45. A corpus whose evidence fits under 100k costs the same at either
setting. `generate_theories_over_types()` runs one generation per type, so its cost scales with this
too.

**A model with a smaller window** rejects a 750k prompt (`prompt_too_long`: not billed, but no
theories). Give it less:

```python
th.Pipeline.generate_theories(records, q, model_str="gpt-5", max_data_tokens=200000)
th.Config.DEFAULT_MAX_DATA_TOKENS = 200000       # or for every stage
```

### Which theory generator to use

**Use `objective="accuracy"` — the default.** It generates theories grounded in the extracted
literature evidence, with `reflection4`: a newer revision of the accuracy-focused generator the
paper's experiments used.

```python
theories = th.Pipeline.generate_theories(records, query["theory_request_normalized"])
```

The alternatives are for when you know why you want them:

| variant | how to call it | what it does |
|---|---|---|
| **accuracy** (normal) | the default | evidence-grounded, literature-supported |
| novelty | `objective="novelty"` | the novelty-focused generator, deliberately pushed out of the safe basin (`nonsafebasin`), so it proposes riskier theories. Same temperature as accuracy; the difference is the prompt |
| attribution | `objective="attribution"` | adds a per-sentence `attribution` block to every law — which evidence each sentence came from, what it says, and the inference used — plus a second reflection pass over only the evidence that theory cites. Render it with `src/ExportTheoryStoreAsHTML_NoPaperstore_attribution.py` |
| parametric | not exposed — `Legacy.build_theory_parametric_accuracy` / `..._novelty` | generates from the model's own parametric knowledge with **no evidence at all**; these exist as baselines to compare against, not as a way to build theories |

The parametric ones are deliberately not wired into `Pipeline.generate_theories()`, because
generating a theory without evidence is almost never what you want from this library — if you need
them for a baseline comparison, call them through `Legacy` on purpose.
- **`use_reflection=True`** runs the second pass over each theory. One extra call per theory, each
  re-sending the full evidence, so it costs about 4.8x a single-pass run (measured, 4 theories per
  run); normally
  what you want.
- **Evidence is subsampled** to fit `max_data_tokens` (default 750k) — see
  [How much evidence a call sees](#how-much-evidence-a-call-sees).
- **Paper titles are read by an LLM, not pattern matching** — a rule-based version produced titles
  like `Abstract` and `OPEN ACCESS` for 88 of 934 papers.
- **PDFs** need `ocr="mistral"` (Mistral OCR, needs `MISTRAL_API_KEY`, cached so re-importing is
  free; over 25 MB is refused). Without it a PDF is refused: convert PDFs to markdown first.

---

## Where things are

```
theorizer/     the library    Legacy.py is the map of what calls what in src/
tests/         python tests/RunAllTests.py
examples/      runnable examples, see examples/README.md
scripts/release/  the live check run before each release (check_release.py)
```
