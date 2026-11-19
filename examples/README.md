# Examples

Seven scripts covering the main stages. Each is runnable, small by default, and says what it costs.

| | what it shows | cost |
|---|---|---|
| `01_FirstRun.py` | the full flow: plan → papers → extract → theories | ~$0.45 |
| `02_BringYourOwnPapers.py` | your own markdown/text (PDFs: `ocr="mistral"`), metadata, Semantic Scholar lookup | ~$0.003/paper |
| `03_IteratingOnASchema.py` | pilot → variable **report** → extended schema → extract again | ~$0.35 |
| `04_PickingUpLater.py` | reopening a project and continuing from disk | free |
| `05_SymbolicLaws.py` | symbolic schema → typed table → fitted quantitative laws | ~$0.60 |
| `06_VariableDiscovery.py` | per-paper variable **discovery** → schemas → extract | ~$0.25 |
| `07_TheoriesAndModels.py` | accuracy vs novelty objectives, choosing a model per stage | ~$0.30 |

```bash
python examples/01_FirstRun.py /path/to/papers
```

`01`, `03`, `05` and `06` plan from a fixed request, *how VIP interneurons control plasticity in
auditory cortex*, so the papers you give them must be on that topic, or the extraction finds nothing to record.
For papers on another topic, edit `REQUEST` at the top of each script.

## Where each capability is covered

| capability | example |
|---|---|
| bringing your own markdown (or PDFs, with `ocr="mistral"`) | `02` |
| schema from the request | `01` |
| schema from per-paper variable **discovery** | `06` |
| schema from a variable **report** over records | `03` |
| **symbolic** schema (typed, units, vocabularies) | `05` |
| extraction | `01`, and every other |
| variable discovery | `06` |
| theory generation, accuracy and novelty | `07` |
| symbolic law discovery | `05` |
| resuming a session | `04` |
| choosing models per stage | `07` |

`03` and `06` are the two that matter most. The first schema a model proposes is a guess made before
it has seen any of your papers; both examples show a cheap way to find that out before running
anything at scale. They differ in when they ask: `06` asks each paper what is worth extracting
*before* any extraction, `03` reads the records afterwards and reports what is missing.

## Choosing models

Every stage that calls a model takes `model_str`. To set the defaults once (these are the
defaults; any model your keys reach works):

```python
th.Config.init(model_primary="gpt-6-sol", model_extraction="gpt-6-luna")
th.Config.set_models(primary="gpt-6-luna")        # or change them later
```

`primary` is the reasoning work (planning, variable reports, schemas, theories); `extraction` is the
per-paper work, where cost multiplies. Any single call overrides both:

```python
th.Pipeline.generate_theories(records, query, model_str="gpt-6-sol")
```

So the useful pattern is: sweep cheaply, then spend on the one call that matters.

## Running them all

```bash
python examples/RunAllExamples.py /path/to/papers gpt-6-luna
python examples/RunAllExamples.py /path/to/papers gpt-6-luna --skip 05
```

This costs real money, unlike `tests/RunAllTests.py` which is free. It is kept small — a handful of
papers per example.

Each script writes to its own project directory (`example-01-run/`, and so on). Nothing is ever
overwritten, so re-running one leaves both attempts side by side.
