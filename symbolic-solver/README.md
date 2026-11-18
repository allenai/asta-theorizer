# symbolic_solver — pattern / law-discovery solver suite

A **self-contained** tool that throws a data table at ~17 pattern/equation/rule-discovery solvers
at once and returns a single JSON report of everything they found, ranked. Give it one JSON
table; it infers the column types, runs every applicable solver, and reports the patterns
(equations, correlations, rules, clusters, trade-offs, rankings, …) each one discovered.

This directory is a standalone copy — drop it into any project and run it. Nothing here imports
from outside this folder.

```
symbolic-solver/
├── README.md            ← this file
├── requirements.txt     ← dependencies (only numpy is required; the rest are optional)
├── src/
│   └── symbolic_solver.py   ← the entire tool (one file, no internal imports)
└── examples/            ← ready-to-run example inputs (§8)
```

---

## 1. Install & quickstart

```bash
pip install -r requirements.txt

# fast pass (skips the two slow symbolic regressors):
python src/symbolic_solver.py examples/example1_scaling_law.json --skip pysr_symbolic,gplearn_symbolic
```

Only **numpy** is strictly required — with just numpy the pure-numpy built-in solvers still run,
and every other tool is **skipped gracefully** (listed in `tools_skipped`) if its library is
missing. So the suite scales from "numpy only" up to the full external tool set.

---

## 2. Input format

A **JSON table = a list of row-dicts**. Each dict is one row; keys are column/variable names,
values are that row's cells. Every solver consumes this same table.

```json
[
  {"data_size_k": 1,  "accuracy": 55.0, "model": "cnn"},
  {"data_size_k": 2,  "accuracy": 60.7, "model": "transformer"},
  {"data_size_k": 4,  "accuracy": 66.2, "model": "cnn"}
]
```

A `{"data": [...]}` wrapper is also accepted (any other top-level keys are ignored). Rows need
not share the same keys — a key missing from a row is treated as `null` (missing) there.

---

## 3. Column types (auto-detected)

Each column is classified once, from its values:

| values in the column (ignoring `null`) | inferred type |
|---|---|
| all `int`/`float` | **numeric** |
| anything else (strings, mixed, all-`bool`) | **categorical** |

- **All-or-nothing per column** — one string value makes the whole column categorical.
- `bool` → categorical (but see mirrors, §4); `null`/missing is counted, not typed.
- Numbers stored as **strings** (`"3.5"`) are categorical — but the suite auto-adds a numeric
  mirror so solvers don't miss them (§4).
- **Ordinal categoricals are NOT inferred** — `low/med/high` carries an order the data can't
  reveal, so supply your own numeric column (e.g. `severity` 1/2/3) if you want it numeric. The
  suite never guesses an ordinal encoding.

The inferred schema appears under `input_summary.schema`, with `n_missing` (int) and
`percent_missing` (float, 0–100) per column.

---

## 4. Automatic preprocessing (validation + numeric mirrors)

Before solving, `preprocess_table()` runs automatically and records everything it does in the
top-level **`warnings_errors`** list (also printed to stderr unless `--quiet`):

1. **Type validation** — any cell not `int`/`float`/`bool`/`str`/`null` (e.g. a nested list or
   object) is set to `null` with a warning; the rest of the column is kept.
2. **Bool mirror** — a `bool` column `flag` gets `flag_mirror_int` (0/1) for numeric solvers.
3. **String-numeric mirror** — an all-string column whose values all parse as numbers gets
   `count_mirror_int` (integral, e.g. `"3"`) or `rate_mirror_float` (e.g. `"1.5"`).

**Originals are never removed**; mirrors are added as `<name>_mirror_int` / `<name>_mirror_float`.
A mirror is skipped if an equivalent numeric column (or the mirror name) already exists, so
re-running is idempotent.

---

## 5. Running it

```bash
python src/symbolic_solver.py data.json --out report.json      # report to a file
python src/symbolic_solver.py data.json                        # report to stdout
cat data.json | python src/symbolic_solver.py -                # read stdin
```

Flags: `--out FILE`, `--skip a,b,c` (tool names), `--quiet`, `--timeout SECONDS`.

`--timeout` is an optional per-solver wall limit: any single tool exceeding it is stopped and
listed under `tools_skipped` (reason `"timed out after Ns"`) while **every other tool's results
are kept**. There is no default limit — tools run to completion, and being slow on more data is
expected, not an error. (A native solver like pysr can't be force-killed mid-run, so its work
keeps running in the background until the CLI exits — but the report is produced immediately.)

> Parallelism is intentionally **not** a CLI option. The suite runs serially by default because
> the heavy solvers are already multi-core internally, so threading them only oversubscribes the
> CPU and makes total wall-clock **worse** (measured ~24s serial vs ~28s parallel). The threaded
> code path is retained for the future and can be enabled by a library caller via
> `cfg={"parallel": True, "max_workers": N}`. The real speed lever is `--skip` on the slow
> symbolic regressors.

As a library (the file name has no hyphen, so it imports directly):

```python
import sys; sys.path.insert(0, "src")
import symbolic_solver as ds
report = ds.run_suite(rows, cfg={"skip_tools": ["pysr_symbolic", "gplearn_symbolic"]}, quiet=True)
```

---

## 6. Output format

```jsonc
{
  "input_summary": {
    "n_rows": 14, "n_cols": 3,
    "numeric": ["data_size_k", "accuracy"], "categorical": ["model"],
    "schema": { "accuracy": {"type":"numeric","cardinality":14,"n_missing":0,"percent_missing":0.0}, ... }
  },
  "warnings_errors": [ /* §4 */ ],
  "tools_run":      ["functional_form", "correlation", ...],
  "tools_skipped":  [ {"tool":"pysr_symbolic","reason":"missing dependency: ...","seconds":0.0} ],
  "tools_timedout": [ "pysr_symbolic" ],   // names that hit --timeout (also in tools_skipped); [] if none
  "highlights":    [ /* strongest patterns ACROSS tools, ranked */ ],
  "results_by_tool": { "functional_form": {"patterns": [ /* Pattern objects */ ]}, ... },
  "timing": { "wall_seconds": 2.9, ... }
}
```

**Pattern object:**

```json
{ "tool": "functional_form", "kind": "equation", "target": "accuracy", "scope": "global",
  "description": "accuracy = 7.968*ln(data_size_k) + 55.15   [logarithmic, R2=1.000]",
  "metric": "r2", "score": 1.0,
  "extra": {"family": "logarithmic", "feature": "data_size_k", "coef": [55.15, 7.968]} }
```

- `description` — human-readable result with the actual equation / rule / numbers.
- `score` — the tool's **native** score (R², |r|, eta², confidence, silhouette, …); patterns are
  ranked **within each tool** by it. `metric` names which score it is.
- `extra` — machine-readable details (coefficients, the pair, rule antecedent, k, …).

**Highlights** are the strongest patterns *across* tools (a heuristic 0–1 `strength` maps the
different native scores onto a common scale, for selection only — no single global ranking is
forced). When multiple tools explain the same target, the highlight notes `consensus_tools`.

---

## 7. The solvers (17)

| group | tools |
|---|---|
| equations / regression | `functional_form` (9-form library: linear, log, power, gaussian, …), `multilinear_regression`, `linear_tree`, `pysr_symbolic`, `gplearn_symbolic` |
| correlation | `correlation` (Pearson), `spearman` (monotonic) |
| group structure | `group_means` (eta²), `conditional_form` (subgroup where a relationship is stronger), `pysubgroup` |
| rules / associations | `implications` (IF a=v THEN b=w), `associations` (mlxtend itemsets), `sklearn_tree`, `classification_tree`, `imodels_rulefit` |
| unsupervised / causal | `clustering` (KMeans + silhouette), `causal_pc` (PC graph) |

Target-requiring tools follow a **find-any-pattern** policy: every column is tried as the target
in turn. The seven built-ins (`functional_form`, `multilinear_regression`, `conditional_form`,
`group_means`, `correlation`, `spearman`, `implications`) need only numpy (+ scipy for the
gaussian fit and spearman); the other ten are the optional external tools.

---

## 8. Worked examples (`examples/`)

Run any with `python src/symbolic_solver.py examples/<file> --skip pysr_symbolic,gplearn_symbolic`.
The **full JSON report for each example is committed under `examples/output/`** (paired by name,
e.g. `examples/output/example1_scaling_law.output.json`) so you can see the complete output
without running anything.

### example1_scaling_law.json — functional form (equation discovery)
A logarithmic scaling law `accuracy ~ ln(data_size_k)`:
```
functional_form  accuracy = 7.968*ln(data_size_k) + 55.15   [logarithmic, R2=1.000]
spearman         monotonic corr(data_size_k, accuracy): rho = 1.000
```

### example2_method_ranking.json — comparative ranking (categorical → numeric)
Three methods over repeated runs; the suite recovers the ordering:
```
group_means          mean(f1_score) varies with method (eta^2=0.985)
classification_tree  classification tree: 3 rules, train acc=1.000
```
(`results_by_tool.group_means[*].extra.group_means` gives the per-method means → the ranking.)

### example3_association.json — rule / association mining (categorical → categorical)
`uses_feature=yes` is strongly associated with `renewed=yes` (≈0.9); `region` is a distractor:
```
classification_tree  classification tree: 5 rules, train acc=0.825   (splits on uses_feature)
associations         {region=NA, uses_feature=no} => {renewed=no}    (in results_by_tool.associations)
```
The `implications` tool stays silent here by design — its default confidence floor is 0.95, above
this ≈0.9 effect; lower it with `cfg={"impl_min_conf": 0.6}` to surface probabilistic rules.

### example4_mixed_types.json — preprocessing showcase
Exercises every §4 feature. `warnings_errors`:
```
[warning] column 'meta': 2 cell(s) of unexpected type (not int/float/bool/str) set to null
[info]    'is_premium' already has an equivalent numeric column 'active01'; no mirror added
[info]    added numeric mirror 'visits_mirror_int' for string-numeric column 'visits'
[info]    added numeric mirror 'spend_mirror_float' for string-numeric column 'spend'
```
`input_summary.numeric` now includes `visits_mirror_int`, `spend_mirror_float`.

### example5_clusters.json — clustering (unsupervised)
Three Gaussian blobs in `(x1,x2)` plus a noise column → `clustering` finds **k=3**
(silhouette=0.86), one pattern per cluster with its distinctive feature(s).

### example6_tradeoff.json — negative-linear trade-off
```
correlation      corr(latency_ms, accuracy) = -0.998
functional_form  accuracy = -0.003668*latency_ms + 1.021   [linear, R2=0.996]
```

### example7_wrapped_input.json — the `{"data": [...]}` wrapper form (same data as #6).

---

## 9. Notes & caveats

- **It reports, it doesn't judge.** The suite surfaces patterns with their native scores; it has
  no notion of a "correct answer" — interpret the highlights/per-tool sections yourself.
- **Runs on any size, small or huge — it does not silently skip.** Every solver runs down to its
  true mathematical floor (e.g. a line needs ≥3 points; a tree ≥2; PC ≥5 and ≥2 numeric columns);
  there are **no arbitrary "needs ≥20 rows" gates**. Stress-tested on 1-row, 2-row, all-null,
  NaN/inf, constant, collinear, high-cardinality, very wide and very tall tables with no crashes.
- **Small data is flagged, not hidden (`special_note`).** When the table has fewer than
  `small_data_rows` rows (default 25), every pattern — and its copy in `highlights` — carries a
  `special_note` warning that the fit is on little data and may be overfit/low-confidence. The
  tool still runs and reports; you decide.
- **Cell repair is lenient.** A column with a few unexpected-type cells keeps its valid values
  (those cells become `null`); check `percent_missing`. A column reduced to very few values won't
  produce confident fits, and any small-data results carry the `special_note` above.
- **Performance.** `--skip pysr_symbolic,gplearn_symbolic` is the main speed lever. Slow on more
  data is expected (not capped); use `--timeout` if you want a hard per-solver bound. Parallelism
  is off by default — the heavy solvers are already multi-core internally.
- **Tuning PySR (symbolic regression).** PySR only runs on tables with at least `pysr_min_rows`
  rows (default 12) — a guard against fitting a flexible symbolic expression to too few points.
  Its search effort is `pysr_niterations` (default 40); raise it to search harder for better
  equations, at the cost of runtime. Both are cfg knobs:
  `cfg={"pysr_niterations": 120, "pysr_min_rows": 8}`. PySR is non-deterministic, so results vary
  run-to-run. (On small *noisy* data, lowering `pysr_min_rows` risks spurious equations.)
