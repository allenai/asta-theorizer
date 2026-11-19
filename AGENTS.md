# theorizer — guide

*This is a guide intended for agents. This guide is largely automatically generated using AI tools, from empirical
use and automated testing of the library, and may contain errors.*

A thin library over the Theorizer code in `src/`, so the stages can be run one at a time instead of
as one pipeline. Every stage calls the function in `src/`; nothing here reimplements prompts,
parsing, or data structures. To see what a call really runs, look the name up in
`theorizer/Legacy.py`.

Model names in the output shown below are the defaults (`gpt-6-sol`, `gpt-6-luna`).

---

## Rule 0 — everything goes in the working directory

**Every artifact a run produces goes into that run's working directory. Nothing goes anywhere else
— not `/tmp`, not a scratch directory, not a file next to the script.**

One run, one directory, holding everything needed to reproduce and inspect it: the stage outputs,
the theorystore, the paperstore, and the HTML export. The engine's own files go there too: the
prompt dumps (`prompts/`), the verbose log, the OCR cache (`ocr-cache/`, `data/`) and the Semantic
Scholar cache (`s2_cache.json`). Nothing is written beside the installed library. Only with
`project_dir=None` do these go in the current directory (wherever Python was started) instead.

**Theory generation costs money.** Every model call is billed, and a result that is lost has to be
paid for again, if it can be reproduced at all. An output written anywhere else is one that a later
run overwrites, or the system cleans up. So:

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

**A Semantic Scholar key is optional, but helps.** `fetch_papers()` and `import_papers(lookup_semantic_scholar=True)`
call Semantic Scholar; without a key they share its unauthenticated limit (about one request a second)
and can be refused with `429`. If the person has no key and will fetch or look up papers, tell them they
can request a free one at https://www.semanticscholar.org/product/api#api-key-form.

**A LiteLLM proxy.** With `LITELLM_BASE_URL` and `LITELLM_USER_KEY` both set, every call goes through
that proxy: `OPENAI_API_KEY` is set to the proxy key (replacing any other), and the default model
names get an `openai/` prefix. Name the proxy's models with `THEORIZER_MODEL_PRIMARY` / `THEORIZER_MODEL_EXTRACTION`.
A `model_str=` passed to a single stage is used exactly as written, so under a proxy write the prefix yourself.

---

## Theory queries and mission statements

A run starts from a **theory query**: the plain-language request given to `plan_theory_query()`. It
can be broad or narrow, one sentence or several:

```python
broad    = "Build a theory of what drives capacity fade in lithium-ion batteries."
specific = "How does charging faster than 2C below 10 °C affect lithium plating on graphite anodes?"
detailed = ("Build theories of why some lithium-ion cells fade much faster than others under the same "
            "cycling protocol. Focus on cell-to-cell variation in manufacturing. Leave out "
            "calendar-aging-only studies.")
```

The plan rewrites the query as `theory_request_normalized` ("Build theories of/about ... based on the
following results."), keeping all of its content, and the later stages use that version.

A **mission statement** (`mission_statement=`) is optional text the model keeps in mind alongside the
query. The papers are all Theorizer reads, so the mission statement is where anything else goes. It can
be the mission itself (the purpose the theories serve, the focus, the gaps to aim at), or more
information for the model to keep in context while it builds theories: data or results computed
elsewhere, or other knowledge the papers don't hold.

```python
mission  = ("We are designing a fast-charging protocol for electric buses, so the theories should name "
            "variables an operator can control. Our own cycling data, not in any paper: ...")
plan     = th.Pipeline.plan_theory_query(broad, mission_statement=mission)
query    = plan["output"]
theories = th.Pipeline.generate_theories(records, query["theory_request_normalized"], mission_statement=mission)
```

**Where it goes.** Three stages take `mission_statement=`. Pass it to each one, because none of them
carries it over from another:

| stage | what it does with the mission |
|---|---|
| `plan_theory_query()` | uses it to write the paper search query, the extraction query and the schema |
| `generate_theories()`, `generate_theories_over_types()` | sends it word for word, with the extracted records, in the first call and in every reflection call |

**Extraction gets only what the plan made from it.** `extract_from_papers()` gets the plan's extraction
query and schema, which the mission shaped, but never the mission itself, so each paper is read without
it. Theory generation then gets the mission again, together with the records extracted from the papers.
The other stages don't take it either: `discover_variables`, `make_variable_report`,
`make_schemas_from_variable_discovery` and `make_symbolic_schema` see only the plan's output, and the
evaluations see neither. A mission given only to `generate_theories()` steers the theories but not the
search or the schema. If your workflow needs the mission in one of these other stages, add it to that
stage's prompt yourself:
- **Extraction:** append it to the `extraction_query` you pass to `extract_from_papers()`. It is then
  sent with every paper, so its cost grows with the number of papers.
- **`discover_variables()`:** pass it as `additional_contextual_info=`.
- **The others:** edit the stage's prompt in `src/TheorizerProcessing.py` (the evaluations' prompts are
  in the `src/Evaluation*.py` files).

`max_data_tokens` does not count the mission. A long one, such as a table of results, adds its size to
every generation call.

---

## The four things that flow between stages

Learn these and the rest is obvious. Everything is plain JSON.

### 1. A **paper**

```python
{"paper_id":  "paper-kim2019-3f9c21ab",      # filename stem + a hash of the file
 "title":     "VIP interneuron activity during tone-reward learning in mouse auditory cortex",
 "year":      2019,
 "venue":     "...",
 "authors":   [...],
 "abstract":  "...",
 "markdown":  "# Introduction\nThe mouse auditory cortex ..."}
```

From `PaperImport.import_papers()` (your files) or
`Pipeline.fetch_papers()` (corpus ids).  Both carry `paper_id`, `title`, `year` and `markdown`, which is
all the stages read, so everything downstream is identical. An imported paper keeps `venue`, `authors`
and `abstract` at the top level; a fetched one keeps them in `s2_metadata`.  (This build has no paper search. Search elsewhere, with the
plan's `paper_search_query`, and fetch what it finds by corpus id.)

Papers from `fetch_papers()` also carry `corpus_id`, `s2_paper_id`, `paper_urls`, `status` and
`s2_metadata` (the Semantic Scholar record). Both kinds are kept in the project's
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
 "auditory_cortex_region": "A1",
 "uuid":              "e3.1"}            # <- stamped by the store; laws cite it
```

A record does not carry its paper. Its uuid does: each `store.extraction_results[...]` holds one
paper's records under its `paper_id`.

### 4. A **theory**

```python
{"theory_name": "Conditional VIP-SST Disinhibitory Plasticity",
 "theory_name_plain_language": "VIP cells open a window for plasticity mainly by releasing SST inhibition",
 "theory_description": "VIP interneurons regulate auditory-cortical plasticity indirectly: when reward or
                        contextual inputs recruit VIP cells, VIP inhibition of SST interneurons ...",
 "theory_statements": [...],
 "new_predictions_likely": [...], "new_predictions_unknown": [...],
 "negative_experiments": [...],   "special_cases": [...]}
```

---

## The stages

Every stage returns a **packed dict**: what you gave it, plus its result, plus `total_cost` (except `fetch_papers`).
`total_cost` is an estimate, from each model's reported usage and price, and it likely leaves out calls
that failed (a call that timed out or was cut off is still billed).
Stages that call a model take `model_str=`; stages that save take `save_as=`; the per-paper ones
also take `max_workers=`.

| stage | give it | get back |
|---|---|---|
| `plan_theory_query(request)` | a plain-language request | `["output"]` → `theory_request_normalized`, `short_name`, `paper_search_query`, `extraction_query`, `extraction_schema` |
| `fetch_papers(candidates)` | dicts with a `corpus_id` (a title is optional) | `["papers"]` → the papers that got full text (PDFs go through Mistral OCR, so this needs `MISTRAL_API_KEY`); also fills and saves the project's **PaperStore** (`["paper_store_path"]`) |
| `PaperImport.import_papers(path)` | a dir/glob/file/list | `["papers"]` → list of papers; also adds them to the project's **PaperStore** (`["paper_store_path"]`) |
| `extract_from_papers(papers, query, schema)` | papers + schema | `["results"]` → one entry per paper |
| `flatten_records(extraction)` | that result | a flat list of **extraction records** (saved as `records`) |
| `discover_variables(papers, query)` | papers | `["results"]` → per-paper `proposed_slots` |
| `make_schemas_from_variable_discovery(discovery, query)` | that result | `["output"]["extraction_schemas"]` → **new** schemas |
| `make_variable_report(records, query)` | records | `["output"]["suggestions"]` |
| `extend_schema_with_suggestions(schema, suggestions)` | schema + suggestions | a **bigger schema** (no LLM call), from the first 3 suggestions unless `max_suggestions=` says otherwise |
| `make_symbolic_schema(records, query)` | records | `["variables"]`, `["extraction_query"]` |
| `generate_theories(records, query)` | records | `["theory_response"]["theories_general"]` and `["theories_specific"]` |
| `generate_theories_over_types(records, query, output_types, inference_types)` | records | `["theories"]`, one generation per type (see [Sweeping by type](#sweeping-by-type)) |
| `to_theorystore(theories, query, model_str=, method=)` | theories | saves them to the project's theorystore, and returns a store dict with **just these theories**, plus all the schemas and extraction results (no LLM call) |

> **The one irregularity:** theories arrive under `theory_response`, everything else under `output`.
> That is how `src/` returns them.

`query` in this table is one of two things. For `discover_variables`, `make_schemas_from_variable_discovery`,
`make_variable_report` and `make_symbolic_schema` it is the whole plan, `plan["output"]` (a dict).
For `generate_theories`, `generate_theories_over_types` and `to_theorystore` it is the request
string, `plan["output"]["theory_request_normalized"]`.

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

papers = th.PaperImport.import_papers("/path/to/papers")["papers"]   # markdown/text; a folder of PDFs needs ocr="mistral"

plan  = th.Pipeline.plan_theory_query("Build a theory of how VIP interneurons control A1 plasticity")
query = plan["output"]

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
  saved: vip-run/01-papers.json
Pipeline.plan_theory_query(): Planned 'Build a theory of how VIP...' ($0.002)
  saved: vip-run/02-plan-theory-query.json
Parallel.run_over_items(): 6/6 done (0 failed, 31s elapsed)
Pipeline.extract_from_papers(): 2 record(s) from 2/6 papers ($0.031)
  saved: vip-run/03-extraction.json
  saved: vip-run/04-records.json
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

(a) reads each paper's full text; (b) reads the records you have already extracted. In `src/`, (a) is
`VariableDiscoveryQueue.discover_variables_from_paper` then `generate_schema_from_variable_discovery`;
(b) is `generate_variable_discovery_from_data_report`, which `src/` also calls "variable discovery".

### (a) Ask the papers first — *before* extracting

```python
discovery = th.Pipeline.discover_variables(papers, query)     # one call per paper
schemas   = th.Pipeline.make_schemas_from_variable_discovery(discovery, query)

new = schemas["output"]["extraction_schemas"][0]              # a whole NEW schema
ext = th.Pipeline.extract_from_papers(papers, new["extraction_query"], new["extraction_schema"],
                                      extraction_schema_id="schema-2")
```

Give each schema its own `extraction_schema_id` (the first pass is `"schema-1"`), so the store can tell
whose records are whose. `schema-2`, `schema-3`, ... are just unique ids: each schema needs one that no
other extraction in the project has used, whichever workflow it comes from (so if (a) and (b) both run
in one project, (b) takes a number (a) didn't). A reused id files the new records under the schema
already stored with that id, and the library only warns. To build theories from every new schema, use this loop instead of the
single extraction above (it extracts each schema, then generates and saves as after the first pass):

```python
for n, new in enumerate(schemas["output"]["extraction_schemas"], start=2):
    ext      = th.Pipeline.extract_from_papers(papers, new["extraction_query"], new["extraction_schema"],
                                               extraction_schema_id=f"schema-{n}")
    theories = th.Pipeline.generate_theories(th.Pipeline.flatten_records(ext, save_as=f"records-schema-{n}"), query["theory_request_normalized"])
    found    = theories["theory_response"]["theories_general"] + theories["theory_response"]["theories_specific"]
    th.Pipeline.to_theorystore(found, query["theory_request_normalized"], model_str=theories["model"], method="accuracy")
```

Each paper answers in `discovery["results"][i]["proposed"]["proposed_slots"]`:

```python
{"slot_name":      "interneuron_recruitment_by_behavioral_context",
 "slot_description": "Extract how behavioral context (reward, punishment, attention) changes the recruitment of inhibitory interneurons ...",
 "impact_rating":  7}
```

### (b) Ask the records afterwards — *after* extracting

```python
report      = th.Pipeline.make_variable_report(records, query)
suggestions = report["output"]["suggestions"]
extended    = th.Pipeline.extend_schema_with_suggestions(query["extraction_schema"], suggestions)

ext2 = th.Pipeline.extract_from_papers(papers, query["extraction_query"], extended,
                                      extraction_schema_id="schema-10")   # any id not yet used in this project
```

Each suggestion carries the slots to add:

```python
{"suggestion_short_name": "Define sign/target of VIP modulation",
 "suggested_slots_and_extraction_instructions": [
    {"slot_name": "vip_modulation_sign",
     "slot_description_and_extraction_instructions": "Extract whether VIP activation strengthens
                                                      or weakens the plasticity."}]}
```

|  | (a) discovery | (b) report |
|---|---|---|
| when | before extracting | after extracting |
| gives you | **new** schemas | your **existing** schema, extended |
| model calls | one per paper, then two to design the schemas (a draft and a reflection) | two report calls (a draft and a reflection); extending the schema is free |
| use when | you want it redesigned around what's in the papers | you want the two rounds comparable |

Pilot on 2 papers, look, refine, *then* run at scale — that's the point of the library.

---

## Workflow 3 — symbolic laws

Theory generation writes prose. The symbolic path fits **equations** to a table of typed values.
Install the `symbolic` extra first (`pip install "asta-theorizer[symbolic]"`, or `uv sync --extra symbolic`
in a clone); without it the solver skips the tools that need scipy, scikit-learn or pandas. Even with it,
a few tools whose packages the extra doesn't include show as skipped (`missing dep`); the rest run.

```python
records  = th.Pipeline.flatten_records(th.Pipeline.extract_from_papers(papers, query["extraction_query"],
                                                                       query["extraction_schema"]),
                                       save_as="records-symbolic-input")

symbolic = th.Pipeline.make_symbolic_schema(records, query)      # typed, with units + vocabularies
typed    = th.Pipeline.flatten_records(
             th.Pipeline.extract_from_papers(papers, symbolic["extraction_query"],
                                             symbolic["variables"], extraction_schema_id="schema-symbolic"),
             save_as="records-symbolic")

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
Symbolic.build_table(): WARNING: behavioral_state -- 1 value(s) did not match (e.g. outside the vocabulary: asleep)
```

A repaired value becomes a confident wrong law. Read `table["coverage"]` before believing anything.

Output:

```
plasticity_change = 2*vip_activation + 1   [linear, R2=1.000]   (score 1.0, via functional_form)
corr(vip_activation, plasticity_change) = 1.000                 (score 1.0, via correlation)
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
  02  plan-theory-query        0.02 MB  24 extraction_schema  gpt-6-sol     $0.002
  03  extraction               0.03 MB  6 results             gpt-6-luna    $0.031
  04  records                  0.02 MB  2 record(s)
  05  theories                 0.31 MB  8 theories            gpt-6-sol     $0.07
  09  theories-novelty         0.07 MB  4 theories            gpt-6-sol     $0.008
```

```python
records = th.Project.load("records")        # or th.Project.load(4); a name saved twice must be loaded by number
query   = th.Project.load("plan-theory-query")["output"]
more    = th.Pipeline.generate_theories(records, query["theory_request_normalized"],
                                        objective="novelty", save_as="theories-novelty")
found   = more["theory_response"]["theories_general"] + more["theory_response"]["theories_specific"]
th.Pipeline.to_theorystore(found, query["theory_request_normalized"], model_str=more["model"], method="novelty")
```

It is a directory of JSON files — `mv`, `cp`, `rm` all work, and `show()` stays correct.

A name matches every save with that name, and `load()` refuses a name that matches more than one.
A name that matches no save exactly falls back to the saves whose names contain it, so a near-miss
can load a different save than you meant; `show()` first, or load by number.
Give a stage you will load back its own `save_as=` (`save_as="schemas-pass-2"`), or load by the
number `show()` lists.

### Long runs resume after a crash

`extract_from_papers`, `discover_variables`, `import_papers` and `generate_theories_over_types`
append each item's result the moment it lands. Re-run the same call, with the same `save_as=`, and it continues:

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

**Re-running a whole script re-runs the stages before the crash too.** Load what they saved instead
of calling them again. `plan_theory_query` in particular is billed on every call and returns a
different schema each time, and a different schema matches no checkpoint, so nothing resumes:

```python
query = th.Project.load("plan-theory-query")["output"]    # not plan_theory_query() again; if a plan was saved twice, load by the number show() lists
```

Re-running `extract_from_papers()` never adds the same extraction results to the TheoryStore twice: a
result with the same paper, `extraction_schema_id` and records reuses the one already stored, uuids
and all, so laws citing those uuids still link. Theories get no such check. Every `to_theorystore()`
call adds each theory it is given as a new theory with a new id, so passing it theories that are
already stored stores a second copy of each. That happens when a script that ends in `to_theorystore()`
is run again, or when a resumed `generate_theories_over_types()` sweep (whose `["theories"]` holds the
earlier runs' theories as well as the new ones) is stored again in full.

**Known issue:** a resumed `extract_from_papers`, `discover_variables`, `import_papers` or
`generate_theories_over_types` call reports, in its `total_cost`
and in `show()`'s `$` column, the cost of the results it reused as well as the new ones. Those were billed
by the earlier run, so adding up the two saves counts them twice.

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
    schema=schema, extraction_query="VIP interneurons and plasticity", supporting_theory_ids=[], id="schema-1", model_str="written"))

uuids = {}
for paper in papers:                                  # read paper["markdown"], then write its records
    records = [{"name_short": "VIP plasticity", "name_full": "VIP activation and tone-evoked plasticity",
                "brief_description": "VIP activation enhanced tone-evoked plasticity.", "study_species": "mouse"}]
    result_id = store.add_extraction_result(th.Legacy.ExtractionQueryResult(
        paper_id=paper["paper_id"], extraction_schema_id="schema-1", extracted_data=records,
        potentially_relevant_new_papers=[], model_str="written"))
    uuids[paper["paper_id"]] = [record["uuid"] for record in store.extraction_results[result_id].extracted_data]
th.Project.save_theory_store()

first = next(iter(uuids.values()))[0]
theories = [{"theory_name": "VIP disinhibition", "theory_description": "VIP cells enable plasticity by inhibiting SST cells.",
             "theory_statements": [{"law": {"statement_name": "Disinhibition", "theory_statement": "VIP activity enables plasticity.",
                                            "supporting_evidence": [{"text": "VIP activation enhanced plasticity.", "uuids": [first]}]}}]}]
stored = th.Pipeline.to_theorystore(theories, "How do VIP interneurons control plasticity?", model_str="written", method="written")
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

**Assume you are saving everything to a TheoryStore.** The library does this for extraction: opening a
project creates one, every extraction (its schema and its records) goes through it, and it is written
to `<project>/theorystore.json` and reloaded next session. Theories are the exception: they reach
the store only through `to_theorystore()`.

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
# records: the list th.Pipeline.flatten_records(ext) returned (calling it again saves a second "records")
print(sum(1 for r in records if r.get("uuid")), "of", len(records), "records can be cited")
```

`extract_from_papers()` and `flatten_records()` both warn if any record lacks a uuid.

---

## The PaperStore — the papers themselves, per project

The TheoryStore holds the theories and the extracted records. The **PaperStore** holds the papers
those records came from: full text, and the Semantic Scholar record behind each one (title, venue,
year, abstract, tldr, url, corpus id).

Opening a project creates one, `fetch_papers()` fetches into it, `import_papers()` adds to it, and it is written to
`<project>/paperstore.json` and reloaded next session — the same arrangement as the TheoryStore.

```python
th.Config.init(project_dir="vip-run")     # the paperstore lives here too; nothing else to do

found = th.Pipeline.fetch_papers([{"corpus_id": "15600925"}, {"corpus_id": "3291104"}])
print(found["paper_store_path"])          # <project>/paperstore.json
```

Each open-access PDF goes through Mistral OCR. Without `MISTRAL_API_KEY` a paper is stored with its
Semantic Scholar record only, and `found["papers"]` leaves it out.
The same happens when none of a paper's links leads to a PDF (a DOI link often leads to a publisher's
landing page); the run prints `0 paper(s) with full text (N had metadata only)`.
**A paper already in the store is not fetched again, even one stored metadata-only**, so adding the key
and fetching again in the same project does not fill it in; fetch it in a new project. There is no
other PDF converter built in: without a Mistral key, ask the person which PDF-to-markdown converter
they prefer, convert the PDFs with it, and bring the markdown in with `import_papers()` (see
[PDF → Markdown conversion without Mistral](#pdf--markdown-conversion-without-mistral)).

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
explicitly. `paperstore.json` is written by `fetch_papers()` and `import_papers()`; an imported paper's page
shows its title and year (venue, TL;DR and abstract come only from fetched papers). `--papers <papers.jsonl>` (one JSON paper per line, not the project's `NN-papers.json`, which is
one JSON array) gives titles only. Without either, pages show the paper id alone.
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
| `ExtractionUtils.VERBOSE` | `False` after `Config.init()` (`True` with `quiet_legacy_output=False`) | the full-response debug dump (and theory generation's first draft) to the terminal (status lines always print) |
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
th.Config.set_quiet(False)              # dump to the terminal instead of the log file, for debugging
```

**Status lines always print regardless** — `Querying LLM model (...)`, `Prompt tokens`, running
cost, `Completed. Cost`. Only two dumps are affected: the full response (the block `src/` marks
`# DEBUG: PRINT FULL RESPONSE`), and theory generation's first draft, printed as `## Initial response:`
before reflection.
Errors, warnings and retries always print. Safe to call mid-run; it just sets a flag.

### Per-call parameters shared by most stages

| parameter | what it does |
|---|---|
| `model_str` | the model for this call only |
| `max_workers` | parallel width (per-paper stages) |
| `save_as` | name this save, e.g. `"extraction-pilot-v2"` |
| `show_progress` | progress lines for this call (not taken by `plan_theory_query`, `make_variable_report`, `make_schemas_from_variable_discovery`, `make_symbolic_schema`, `generate_theories` or `flatten_records`) |
| `max_tokens` | completion tokens (in `plan_theory_query`, `make_variable_report`, `make_schemas_from_variable_discovery` and `make_symbolic_schema`, the first call never gets fewer than 32000) |
| `temperature` | `0` for most stages, `1.0` for theory generation. Reasoning models (gpt-5/gpt-6, o-series, Claude) are sent none, so with the default models it does nothing |
| `use_reflection` | run a second, refining pass. On by default for `generate_theories`, `generate_theories_over_types`, `make_variable_report`, `make_schemas_from_variable_discovery` and `make_symbolic_schema`; off for `plan_theory_query`, `extract_from_papers` and `discover_variables`. In theory generation it is one extra call per theory (measured ~4.8x total cost) |
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
| `interpreter_shutdown` | Python was shutting down mid-call | rerun |

Re-running the same call retries only the papers that failed. A changed `max_tokens` is a new
call, so every paper runs again: to retry only the failures with a bigger budget, pass just those
papers.

---

## Evaluating theories

`th.Evaluate` scores the theories in a store: a `theorystore.json` loaded with `json.load` (every
theory in the project), or the dict `to_theorystore()` returns (only the theories from that call). Each evaluation saves itself like a stage, and keeps its
working files under `<project>/evaluations/<kind>/<time>/`. `judge` and `belief` need an open project
(`th.Config.init(project_dir=...)`); with `project_dir=None` they stop before any call. `novelty` runs
without one, but then saves nothing, and its verdicts are only in the dict it returns.

```python
store   = json.load(open("vip-run/theorystore.json"))
novelty = th.Evaluate.novelty(store, limit=1)   # qualified novelty, against every paper in the PaperStore; limit=1 to measure the cost first
th.Export.export_html()                    # the theory pages now show each evaluated law's verdict
```

| evaluation | what it asks | cost |
|---|---|---|
| `novelty(store)` | for each law, does any paper in the project's PaperStore already say it? Seven dimensions (phenomenon, mechanism, unification, scope, limitation, reframing, meta-regularity), each `explicit_established`, `explicit_peripheral`, `derivable_unstated` or `genuinely_new` — or null when the papers say nothing that bears on it | one call per paper per law, with reflection: measured **$1.05–1.55 per law** at 100 papers; five laws run at once (`num_workers=5`), so 9 laws took 9 minutes |
| `judge(store)` | six 1–10 scores per law from the model's own knowledge | one call per law |
| `belief(store, num_samples=10)` | the model's probability that each law is true, sampled | `num_samples` calls per law |

**Novelty can be very expensive.** Each law is compared against every paper with full text in the
PaperStore (or the one `paperstore=` names), one paper at a time, so its cost grows with the number of
papers as well as the number of laws. Run a small version first (`limit=1` scores one law), read its
`total_cost`, and multiply by the number of laws that qualify (the run prints it as `Loaded N theory
statements to evaluate for qualified novelty`) before running them all.

**Only laws the generator itself rated new or somewhat new are scored for novelty** — on a
100-paper run, 9 of 23. `novelty(limit=)` scores that many of those at random, for a cheap first look;
`belief(limit=)` scores the first that many laws; `judge` has no `limit=` and scores every law.
`novelty["verdicts"]` has each law's verdict. One file per law, with every paper's evaluation, goes
to `evaluations/novelty/<time>/statements/`, and the export reads the newest of those runs
(`--novelty DIR` names another).

`judge` and `belief` results stay in their own saves (`evaluation-judge`, `evaluation-belief`; each
law's scores are in the file at `["results_path"]`). They do not reach `theorystore.json` or the HTML
export. If the person reviewing the theories wants them, export them yourself and include them in your
records and in any HTML you produce.

---

## Worth knowing

### A mission statement

See [Theory queries and mission statements](#theory-queries-and-mission-statements).

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
| attribution | `objective="attribution"` | adds a per-sentence `attribution` block to every law — which evidence each sentence came from, what it says, and the inference used — plus a second reflection pass over only the evidence that theory cites. Render it with `th.Export.export_html(attribution=True)` (or `python -m theorizer.Export --project <dir> --attribution`), which writes `<project>/html-attribution` |
| parametric | not exposed — `Legacy.build_theory_parametric_accuracy` / `..._novelty` | generates from the model's own parametric knowledge with **no evidence at all**; these exist as baselines to compare against, not as a way to build theories |

```python
th.Export.export_html(attribution=True)        # the open project -> vip-run/html-attribution/index.html
```

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
  free; over 25 MB is refused; only the first 31 pages are read). Without it a PDF is refused: convert PDFs to markdown first.
  Mistral is the only converter built in; without a Mistral key, use the person's preferred
  PDF-to-markdown converter (ask them which) and import the markdown (see
  [PDF → Markdown conversion without Mistral](#pdf--markdown-conversion-without-mistral)).
  OCR prints `Error saving image img-N.jpeg: ...` for figures; that is harmless, and the text is imported.

### PDF → Markdown conversion without Mistral

Mistral OCR is the only PDF converter built in. Without a Mistral key, ask the person which
PDF-to-markdown converter they prefer, convert the PDFs with it, and import the folder of markdown or
text. If they have no preference, pymupdf4llm is a fast option that works reasonably well, though not
perfectly: on six ACL papers it gave clean headings, tables and spacing. It runs on CPU and installs in
seconds (`pip install pymupdf4llm`; AGPL licence):

```python
import pathlib
import pymupdf4llm

pathlib.Path("my-papers").mkdir(exist_ok=True)
for pdf in pathlib.Path("my-pdfs").glob("*.pdf"):
    pathlib.Path("my-papers", pdf.stem + ".md").write_text(pymupdf4llm.to_markdown(str(pdf)))

papers = th.PaperImport.import_papers("my-papers/")["papers"]
```

- Test any inexpensive or CPU-only PDF-to-markdown converter, this one included, on a few papers and
  check that the output is solid before converting and importing the whole folder; a bad conversion
  still imports without error. Equations and figures can be lost, and text inside figures can come out
  garbled.
- On a scanned PDF (one with no text layer), pymupdf4llm runs OCR when Tesseract is installed (it
  prints `Using Tesseract for OCR processing`); on an image-only test page it recovered the text, with
  some words garbled.
- With `read_metadata=False`, a paper's title is its file's first `# ` heading (pymupdf4llm's keep their
  `**` bold marks), or else the filename; the default, `read_metadata=True`, reads the title with a model call.

---

## Where things are

```
theorizer/     the library    Legacy.py is the map of what calls what in src/
tests/         python tests/RunAllTests.py
examples/      runnable examples, see examples/README.md
scripts/release/  the live check run before each release (check_release.py)
```
