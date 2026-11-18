#!/usr/bin/env python
# symbolic_solver.py
#
# A generic "throw the data at many solvers" harness for pattern / law discovery.
# (Standalone extraction; the run_suite/preprocess_table API is unchanged.)
#
# Input  : a JSON table -- a list of row-dicts, keys = column headers, values =
#          cell values.
# Output : a JSON report:
#            {
#              "input_summary": {...column types, n rows...},
#              "warnings_errors": [ {level, code, message, ...}, ... ],
#              "tools_run": [...], "tools_skipped": [{tool, reason}, ...],
#              "highlights": [ ...strongest patterns across all tools... ],
#              "results_by_tool": { tool_name: {"patterns": [ ... ]}, ... }
#            }
#
# Preprocessing (preprocess_table, run automatically): cell types are validated --
# anything not int/float/bool/str/None is set to null with a warning -- and numeric
# MIRROR columns are auto-added so numeric solvers don't miss data: a bool column
# gets `<col>_mirror_int` (0/1); an all-string column whose values all parse as
# numbers gets `<col>_mirror_int`/`<col>_mirror_float`. Originals are kept; a mirror
# is skipped when an equivalent numeric column already exists. All such actions are
# recorded in `warnings_errors` (and printed to stderr unless --quiet).
#
# Design decisions (per project direction):
#  - FIND ANY PATTERN. Tools that need a target column are run once PER candidate
#    target (every column enumerated as the target in turn).
#  - NO global cross-tool score is forced (native scores aren't commensurable).
#    Each tool gets its own section, ranked WITHIN-tool by its native score. A
#    top-level `highlights` section flags the strongest matches across tools (and
#    notes consensus -- the same target explained well by multiple tools).
#  - Each solver is an ADAPTER that declares its dependency and is skipped
#    gracefully (recorded in `tools_skipped`) if that dependency is absent, so the
#    suite scales from "numpy only" up to the full external tool set.
#
# Usage:
#   python src/symbolic_solver.py data.json [--out report.json] [--quiet]
#   echo '<json>' | python src/symbolic_solver.py -

import re
import os
import sys
import json
import math
import time
import argparse
import concurrent.futures
import numpy as np


# ===========================================================================
# Schema inference
# ===========================================================================

def infer_schema(rows):
    """Classify each column as 'numeric' or 'categorical'."""
    cols = {}
    keys = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols[k] = []
                keys.append(k)
    for r in rows:
        for k in keys:
            cols[k].append(r.get(k))
    schema = {}
    for k in keys:
        vals = cols[k]
        nonnull = [v for v in vals if v is not None]
        is_num = len(nonnull) > 0 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in nonnull)
        n_missing = len(vals) - len(nonnull)
        schema[k] = {
            "type": "numeric" if is_num else "categorical",
            "cardinality": len(set(map(_hashable, nonnull))),
            "n_missing": n_missing,
            "percent_missing": round(100.0 * n_missing / len(vals), 2) if vals else 0.0,
        }
    return schema, cols


def _hashable(v):
    return v if isinstance(v, (int, float, str, bool)) else str(v)


def _num(cols, k):
    return np.array([float(v) if v is not None else np.nan for v in cols[k]], float)


def _impute(X):
    """Column-median impute non-finite cells so a single missing feature does NOT drop the
    whole row. Listwise deletion (np.all(isfinite)) collapses sparse wide tables to almost no
    rows; multi-feature models (trees, regression, clustering) should impute instead. A
    fully-missing column -> 0 (becomes constant, harmless)."""
    X = np.array(X, dtype=float)
    for j in range(X.shape[1] if X.ndim == 2 else 0):
        c = X[:, j]; m = np.isfinite(c)
        c[~m] = float(np.median(c[m])) if m.any() else 0.0
        X[:, j] = c
    # clip to a finite range FAR beyond any real measurement but small enough that the
    # float32 cast sklearn does internally (max ~3.4e38), and squaring it, cannot overflow.
    # 1e18 is >> any realistic value, so no real-world data is affected.
    return np.clip(X, -1e18, 1e18)

# NOTE: ordinal categoricals (low/medium/high, ...) carry an order the data alone
# cannot reveal (e.g. low / low-med / med / med-high / high), so this is handled
# INPUT-SIDE: the producer of the JSON should emit a numeric mirror column
# alongside the categorical one. The suite does NOT guess ordinal encodings.


# ===========================================================================
# Fit helpers (numpy only)
# ===========================================================================

def _r2(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - np.sum((y - yhat) ** 2) / ss_tot)


def _lstsq_r2(design, y):
    """Fit y ~ design (incl. intercept col), return (coef, r2)."""
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef, _r2(y, design @ coef)


# --- human-readable rendering -------------------------------------------------

FORM_FAMILY = {"linear": "linear", "quadratic": "quadratic", "cubic": "cubic",
               "reciprocal": "reciprocal", "logarithmic": "logarithmic",
               "sqrt": "sqrt", "exponential": "exponential", "power": "power",
               "gaussian": "gaussian"}


def _g(v):
    return "%.4g" % float(v)


def render_equation(key, coef, yname, xname):
    """Turn a (form, coefficients) pair into the ACTUAL equation with numbers in."""
    c = [float(v) for v in coef]
    y, x = yname, xname
    if key == "linear":
        s = "%s = %s*%s + %s" % (y, _g(c[1]), x, _g(c[0]))
    elif key == "quadratic":
        s = "%s = %s*%s^2 + %s*%s + %s" % (y, _g(c[2]), x, _g(c[1]), x, _g(c[0]))
    elif key == "cubic":
        s = "%s = %s*%s^3 + %s*%s^2 + %s*%s + %s" % (
            y, _g(c[3]), x, _g(c[2]), x, _g(c[1]), x, _g(c[0]))
    elif key == "reciprocal":
        s = "%s = %s/%s + %s" % (y, _g(c[1]), x, _g(c[0]))
    elif key == "logarithmic":
        s = "%s = %s*ln(%s) + %s" % (y, _g(c[1]), x, _g(c[0]))
    elif key == "sqrt":
        s = "%s = %s*sqrt(%s) + %s" % (y, _g(c[1]), x, _g(c[0]))
    elif key == "exponential":               # fit ln y = c0 + c1 x  ->  y = e^c0 * e^(c1 x)
        s = "%s = %s*exp(%s*%s)" % (y, _g(math.exp(c[0])), _g(c[1]), x)
    elif key == "power":                      # fit ln y = c0 + c1 ln x  ->  y = e^c0 * x^c1
        s = "%s = %s*%s^%s" % (y, _g(math.exp(c[0])), x, _g(c[1]))
    elif key == "gaussian":                   # A*exp(-(x-mu)^2/(2 sig^2)) + c
        s = "%s = %s*exp(-(%s - %s)^2 / (2*%s^2)) + %s" % (
            y, _g(c[0]), x, _g(c[1]), _g(abs(c[2])), _g(c[3]))
    else:
        s = "%s = f(%s)" % (y, x)
    return s.replace("+ -", "- ")


def _gplearn_infix(expr):
    """Convert gplearn's prefix string e.g. add(mul(X0,X0),X1) to infix (X0*X0 + X1)."""
    toks = re.findall(r"[A-Za-z_]\w*|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[(),]", expr)
    pos = [0]
    BIN = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
    def parse():
        if pos[0] >= len(toks):
            return "?"
        tok = toks[pos[0]]; pos[0] += 1
        if pos[0] < len(toks) and toks[pos[0]] == "(":     # function application
            pos[0] += 1
            args = [parse()]
            while pos[0] < len(toks) and toks[pos[0]] == ",":
                pos[0] += 1; args.append(parse())
            if pos[0] < len(toks) and toks[pos[0]] == ")":
                pos[0] += 1
            if tok in BIN and len(args) == 2:
                return "(%s %s %s)" % (args[0], BIN[tok], args[1])
            if tok == "neg" and len(args) == 1:
                return "(-%s)" % args[0]
            if tok == "inv" and len(args) == 1:
                return "(1/%s)" % args[0]
            return "%s(%s)" % (tok, ", ".join(args))
        return tok                                          # variable / constant
    try:
        return parse()
    except Exception:
        return expr


def _tree_rules(tree, names, target, classes=None):
    """Flatten a fitted sklearn tree into readable IF..THEN leaf rules.

    Trees often split the same numeric feature multiple times along a path
    (e.g. X<=5.7 then X<=2.2); those are COLLAPSED into one interval (X<=2.2, or
    'lo < X <= hi'). One-hot splits 'col=val <= 0.5' render as 'col != val' /
    'col == val'. If `classes` is given (classification tree) leaves render as
    'target = class (p=purity)'; otherwise as a numeric prediction."""
    t = tree.tree_
    rules = []

    def render(cons):
        num, bools, order = {}, [], []        # name -> [lo(excl), hi(incl)]; bool list; display order
        for c in cons:
            if c[0] in ("le", "gt"):
                name, thr = c[1], c[2]
                if name not in num:
                    num[name] = [None, None]; order.append(("num", name))
                lo, hi = num[name]
                if c[0] == "le":
                    num[name][1] = thr if hi is None else min(hi, thr)
                else:
                    num[name][0] = thr if lo is None else max(lo, thr)
            else:
                order.append(("bool", len(bools))); bools.append(c)
        parts, seen = [], set()
        for typ, key in order:
            if typ == "num":
                if key in seen:
                    continue
                seen.add(key)
                lo, hi = num[key]
                if lo is not None and hi is not None:
                    parts.append("%s < %s <= %s" % (_g(lo), key, _g(hi)))
                elif hi is not None:
                    parts.append("%s <= %s" % (key, _g(hi)))
                else:
                    parts.append("%s > %s" % (key, _g(lo)))
            else:
                c = bools[key]
                parts.append("%s %s %s" % (c[1], "==" if c[0] == "eq" else "!=", c[2]))
        return " AND ".join(parts) if parts else "(always)"

    def rec(node, cons):
        if t.feature[node] == -2:                           # leaf
            v = np.asarray(t.value[node]).ravel()
            if classes is None:                             # regression leaf
                rules.append("IF %s THEN %s ~= %s" % (render(cons), target, _g(float(v[0]))))
            else:                                           # classification leaf
                k = int(np.argmax(v))
                purity = float(v[k] / v.sum()) if v.sum() > 0 else 0.0
                rules.append("IF %s THEN %s = %s  (p=%.2f, n=%d)"
                             % (render(cons), target, classes[k], purity, int(t.n_node_samples[node])))
            return
        name = names[t.feature[node]]
        thr = float(t.threshold[node])
        if "=" in name and 0.0 < thr < 1.0:                 # one-hot boolean split
            col, val = name.split("=", 1)
            rec(t.children_left[node], cons + [("ne", col, val)])
            rec(t.children_right[node], cons + [("eq", col, val)])
        else:
            rec(t.children_left[node], cons + [("le", name, thr)])
            rec(t.children_right[node], cons + [("gt", name, thr)])

    rec(0, [])
    return rules


# A library of single-variable functional forms. Each returns (r2, label, coef)
# or None if its domain doesn't apply. R^2 is always measured in ORIGINAL y-space.
def _fit_forms_1d(x, y):
    out = []
    n = len(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < 3 or np.std(x) == 0:    # need >2 points so even a line has a residual
        return out
    one = np.ones_like(x)

    def add(key, design, predict=None, yt=None):
        try:
            k = design.shape[1]          # complexity = number of fitted params
            if k >= len(y):              # fewer points than params -> exact (overfit) fit; skip this form
                return
            if yt is None:
                coef, r2 = _lstsq_r2(design, y)
            else:                       # fit in transformed y-space, score in original
                coef, *_ = np.linalg.lstsq(design, yt, rcond=None)
                r2 = _r2(y, predict(coef))
            out.append((r2, k, key, coef.tolist()))
        except Exception:
            pass

    add("linear", np.c_[one, x])
    add("quadratic", np.c_[one, x, x ** 2])
    add("cubic", np.c_[one, x, x ** 2, x ** 3])
    if np.all(x != 0):
        add("reciprocal", np.c_[one, 1.0 / x])
    if np.all(x > 0):
        add("logarithmic", np.c_[one, np.log(x)])
    if np.all(x >= 0):
        add("sqrt", np.c_[one, np.sqrt(x)])
    if np.all(y > 0):                   # exponential: ln y = a + b x
        add("exponential", np.c_[one, x],
            predict=lambda c: np.exp(c[0] + c[1] * x), yt=np.log(y))
    if np.all(x > 0) and np.all(y > 0):  # power: ln y = a + b ln x
        add("power", np.c_[one, np.log(x)],
            predict=lambda c: np.exp(c[0]) * x ** c[1], yt=np.log(y))
    # gaussian: A*exp(-(x-mu)^2/(2 sig^2)) + c -- nonlinear (offset allowed), so fit
    # by nonlinear least squares. Try several initial guesses (bump / dip / centered)
    # and keep the best, since curve_fit is sensitive to the starting point.
    if len(x) >= 5:
        try:
            import warnings
            from scipy.optimize import curve_fit

            def _g4(xx, A, mu, sig, c):
                return A * np.exp(-((xx - mu) ** 2) / (2.0 * sig * sig)) + c

            span = float(x.max() - x.min()) or 1.0
            inits = [
                (float(y.max() - y.min()) or 1.0, float(x[np.argmax(y)]), span / 4.0, float(y.min())),
                (float(y.min() - y.max()) or -1.0, float(x[np.argmin(y)]), span / 4.0, float(y.max())),
                (float(y.max() - y.min()) or 1.0, float(np.median(x)), span / 6.0, float(np.median(y))),
            ]
            best = None
            with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
                warnings.simplefilter("ignore")
                for p0 in inits:
                    try:
                        popt, _ = curve_fit(_g4, x, y, p0=p0, maxfev=10000)
                        r2 = _r2(y, _g4(x, *popt))
                        if best is None or r2 > best[0]:
                            best = (r2, [float(v) for v in popt])
                    except Exception:
                        pass
            if best is not None:
                out.append((best[0], 4, "gaussian", best[1]))
        except Exception:
            pass
    # Parsimony: within ~0.01 R^2 of the best, prefer the SIMPLEST form (fewest
    # params) -- otherwise cubic always edges out a true linear law by overfitting.
    out.sort(reverse=True, key=lambda t: (round(t[0], 2), -t[1]))
    return [(r2, key, coef) for (r2, k, key, coef) in out]


# ===========================================================================
# Built-in adapters (numpy only -- always available)
# ===========================================================================

def adapter_functional_form(rows, schema, cols, num_cols, cat_cols, cfg):
    """Best single-variable functional law y=f(x) for every numeric (target,feature)."""
    pats = []
    for y in num_cols:
        yv = _num(cols, y)
        for x in num_cols:
            if x == y:
                continue
            forms = _fit_forms_1d(_num(cols, x), yv)
            if forms and forms[0][0] >= cfg["min_r2_report"]:
                r2, key, coef = forms[0]
                eq = render_equation(key, coef, y, x)
                pats.append(_pat("functional_form", "equation", y, "global",
                                 "%s   [%s, R2=%.3f]" % (eq, FORM_FAMILY[key], r2),
                                 "r2", r2,
                                 {"equation": eq, "family": FORM_FAMILY[key], "feature": x,
                                  "coef": [round(c, 5) for c in coef],
                                  "alternatives": [{"family": FORM_FAMILY[f[1]], "r2": round(f[0], 4)}
                                                   for f in forms[1:6]]}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats


def adapter_multilinear(rows, schema, cols, num_cols, cat_cols, cfg):
    """y ~ all other numeric features (the 'no single variable / global' baseline)."""
    pats = []
    for y in num_cols:
        feats = [c for c in num_cols if c != y]
        if not feats:
            continue
        yv = _num(cols, y)
        Xf = np.column_stack([_num(cols, f) for f in feats])
        ok = np.isfinite(yv)                              # require target; impute missing features
        if ok.sum() < len(feats) + 2:
            continue
        design = np.c_[np.ones(int(ok.sum())), _impute(Xf[ok])]
        coef, r2 = _lstsq_r2(design, yv[ok])
        eq = ("%s = %s" % (y, _g(coef[0]))) + "".join(
            " + %s*%s" % (_g(c), f) for c, f in zip(coef[1:], feats))
        eq = eq.replace("+ -", "- ")
        pats.append(_pat("multilinear_regression", "equation", y, "global",
                         "%s   [R2=%.3f]" % (eq, r2), "r2", r2,
                         {"equation": eq, "features": feats,
                          "coef": [round(float(z), 5) for z in coef]}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats


def adapter_conditional_form(rows, schema, cols, num_cols, cat_cols, cfg):
    """Exceptional-model-mining-lite: a numeric law y=f(x) that holds within a
    categorical subgroup MUCH better than globally (IF C=v THEN y=f(x))."""
    pats = []
    n = len(rows)
    for y in num_cols:
        yv = _num(cols, y)
        for x in num_cols:
            if x == y:
                continue
            xv = _num(cols, x)
            gforms = _fit_forms_1d(xv, yv)
            g_r2 = gforms[0][0] if gforms else 0.0
            for c in cat_cols:
                if schema[c]["cardinality"] > cfg["max_cat_card"]:
                    continue
                for v in sorted(set(_hashable(z) for z in cols[c] if z is not None)):
                    mask = np.array([_hashable(z) == v for z in cols[c]])
                    fit_n = int((mask & np.isfinite(xv) & np.isfinite(yv)).sum())
                    if fit_n < cfg["min_subgroup_n"]:   # enough REAL points to fit (not just subgroup size)
                        continue
                    forms = _fit_forms_1d(xv[mask], yv[mask])
                    if not forms:
                        continue
                    l_r2 = forms[0][0]
                    if l_r2 >= cfg["min_local_r2"] and (l_r2 - g_r2) >= cfg["min_lift_r2"]:
                        _, key_, coef_ = forms[0]
                        eq = render_equation(key_, coef_, y, x)
                        pats.append(_pat(
                            "conditional_form", "conditional_equation", y,
                            "%s = %s" % (c, v),
                            "IF %s=%s THEN %s   (local R2=%.3f vs global %.3f)"
                            % (c, v, eq, l_r2, g_r2),
                            "local_r2", l_r2,
                            {"equation": eq, "family": FORM_FAMILY[key_], "feature": x,
                             "condition": {c: v}, "global_r2": round(g_r2, 4),
                             "coverage": round(float(mask.mean()), 3), "fit_n": fit_n,
                             "coef": [round(z, 5) for z in coef_]}))
    pats.sort(reverse=True, key=lambda p: (p["score"], p["extra"].get("global_r2", 0) * -1))
    return pats


def adapter_group_means(rows, schema, cols, num_cols, cat_cols, cfg):
    """Does a numeric column's mean differ across a categorical column? (eta^2)."""
    pats = []
    for y in num_cols:
        yv = _num(cols, y)
        ok = np.isfinite(yv)
        if ok.sum() < 4 or np.std(yv[ok]) == 0:
            continue
        grand = yv[ok].mean()
        ss_tot = np.sum((yv[ok] - grand) ** 2)
        for c in cat_cols:
            if schema[c]["cardinality"] > cfg["max_cat_card"] or schema[c]["cardinality"] < 2:
                continue
            ss_between = 0.0
            groups = {}
            for v in set(_hashable(z) for z in cols[c] if z is not None):
                m = np.array([_hashable(z) == v for z in cols[c]]) & ok
                if m.sum() == 0:
                    continue
                groups[str(v)] = round(float(yv[m].mean()), 4)
                ss_between += m.sum() * (yv[m].mean() - grand) ** 2
            eta2 = float(ss_between / ss_tot) if ss_tot > 0 else 0.0
            if eta2 >= cfg["min_eta2"]:
                pats.append(_pat("group_means", "categorical_effect", y, "global",
                                 "mean(%s) varies with %s (eta^2=%.3f)" % (y, c, eta2),
                                 "eta2", eta2, {"by": c, "group_means": groups}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats


def adapter_correlation(rows, schema, cols, num_cols, cat_cols, cfg):
    """Pairwise Pearson correlation between numeric columns."""
    pats = []
    seen = set()
    for a in num_cols:
        for b in num_cols:
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            av, bv = _num(cols, a), _num(cols, b)
            ok = np.isfinite(av) & np.isfinite(bv)
            if ok.sum() < 4 or np.std(av[ok]) == 0 or np.std(bv[ok]) == 0:
                continue
            r = float(np.corrcoef(av[ok], bv[ok])[0, 1])
            if abs(r) >= cfg["min_abs_corr"]:
                pats.append(_pat("correlation", "correlation", b, "global",
                                 "corr(%s, %s) = %.3f" % (a, b, r), "abs_r", abs(r),
                                 {"pair": [a, b], "r": round(r, 4)}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats


# ===========================================================================
# External adapters (guarded: skipped if the dependency is missing)
# ===========================================================================

def adapter_gplearn(rows, schema, cols, num_cols, cat_cols, cfg):
    from gplearn.genetic import SymbolicRegressor       # raises if missing -> skipped
    pats = []
    for y in num_cols:
        feats = [c for c in num_cols if c != y]
        if not feats:
            continue
        X = np.column_stack([_num(cols, f) for f in feats])
        yv = _num(cols, y)
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(yv)
        if ok.sum() < 3:                 # gplearn: true floor; small n flagged via special_note
            continue
        est = SymbolicRegressor(population_size=1000, generations=12,
                                function_set=("add", "sub", "mul", "div", "sqrt", "log"),
                                parsimony_coefficient=0.01, random_state=0, n_jobs=1,
                                verbose=0)
        est.fit(X[ok], yv[ok])
        raw = str(est._program)
        infix = _gplearn_infix(raw)
        for i, f in enumerate(feats):                  # X0,X1,.. -> real column names
            infix = infix.replace("X%d" % i, f)
            raw = raw.replace("X%d" % i, f)
        eq = "%s = %s" % (y, infix)
        r2 = _r2(yv[ok], est.predict(X[ok]))
        pats.append(_pat("gplearn_symbolic", "equation", y, "global",
                         "%s   [R2=%.3f]" % (eq, r2), "r2", r2,
                         {"equation": eq, "prefix_form": raw, "features": feats}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats


def adapter_sklearn_tree(rows, schema, cols, num_cols, cat_cols, cfg):
    from sklearn.tree import DecisionTreeRegressor, export_text
    from sklearn.preprocessing import OneHotEncoder
    pats = []
    feat_cols = [c for c in (num_cols + cat_cols)]
    for y in num_cols:
        feats = [c for c in feat_cols if c != y]
        if not feats:
            continue
        yv = _num(cols, y)
        ok = np.isfinite(yv)
        # build feature matrix: numeric as-is, categorical one-hot
        mats, names = [], []
        for f in feats:
            if schema[f]["type"] == "numeric":
                mats.append(_num(cols, f).reshape(-1, 1)); names.append(f)
            else:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                arr = np.array([[str(_hashable(z))] for z in cols[f]], dtype=object)  # uniform str -> encoder-safe
                m = enc.fit_transform(arr)
                mats.append(m)
                names += ["%s=%s" % (f, c) for c in enc.categories_[0]]
        X = np.hstack(mats)
        if ok.sum() < 2:                 # target present + >=2 samples (features imputed, not dropped)
            continue
        Xok = _impute(X[ok])
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(5, ok.sum() // 20),
                                     random_state=0)
        tree.fit(Xok, yv[ok])
        r2 = _r2(yv[ok], tree.predict(Xok))
        rules = _tree_rules(tree, list(names), y)
        pats.append(_pat("sklearn_tree", "tree", y, "global",
                         "regression tree: %d leaf rules, R2=%.3f" % (len(rules), r2),
                         "r2", r2, {"rules": rules}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats


def adapter_pysubgroup(rows, schema, cols, num_cols, cat_cols, cfg):
    import pandas as pd
    import pysubgroup as ps
    pats = []
    df = pd.DataFrame(rows)
    for c in cat_cols:                      # bool/categorical -> uniform strings (None stays NaN)
        df[c] = df[c].map(lambda v: str(v) if (v is not None and v == v) else None)
    for y in num_cols:
        finite = df[y].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size < 2 or finite.std() == 0:    # all-null or constant -> pysubgroup chokes
            continue
        target = ps.NumericTarget(y)
        search = [c for c in (num_cols + cat_cols) if c != y and df[c].notna().any()]
        space = ps.create_selectors(df[search + [y]], ignore=[y])
        task = ps.SubgroupDiscoveryTask(
            df, target, space, result_set_size=5, depth=2,
            qf=ps.StandardQFNumeric(a=0.5))
        result = ps.BeamSearch().execute(task)
        for desc in result.to_descriptions():       # API returns (quality, subgroup[, stats])
            q, sg = desc[0], desc[1]
            pats.append(_pat("pysubgroup", "subgroup", y, str(sg),
                             "subgroup %s has unusual mean(%s)" % (sg, y),
                             "quality", float(q), {}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats


def adapter_associations(rows, schema, cols, num_cols, cat_cols, cfg):
    import pandas as pd
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    # transactions = categorical "col=value" items + binned-numeric items
    txns = []
    for i in range(len(rows)):
        items = []
        for c in cat_cols:
            if cols[c][i] is not None:
                items.append("%s=%s" % (c, _hashable(cols[c][i])))
        txns.append(items)
    if not any(txns):
        return []
    te = TransactionEncoder()
    arr = te.fit(txns).transform(txns)
    df = pd.DataFrame(arr, columns=te.columns_)
    freq = apriori(df, min_support=cfg["min_support"], use_colnames=True, max_len=3)
    if freq.empty:
        return []
    rules = association_rules(freq, metric="confidence", min_threshold=cfg["min_confidence"])
    pats = []
    for _, row in rules.sort_values("lift", ascending=False).head(30).iterrows():
        a = ", ".join(sorted(row["antecedents"]))
        c = ", ".join(sorted(row["consequents"]))
        pats.append(_pat("associations", "association", c, "{%s}" % a,
                         "{%s} => {%s}" % (a, c), "lift", float(row["lift"]),
                         {"antecedents": sorted(row["antecedents"]),
                          "consequents": sorted(row["consequents"]),
                          "support": round(float(row["support"]), 3),
                          "confidence": round(float(row["confidence"]), 3)}))
    return pats


def adapter_spearman(rows, schema, cols, num_cols, cat_cols, cfg):
    """Spearman RANK correlation -- catches monotonic (incl. nonlinear) relations
    that Pearson misses."""
    from scipy.stats import spearmanr
    pats, seen = [], set()
    for a in num_cols:
        for b in num_cols:
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            av, bv = _num(cols, a), _num(cols, b)
            ok = np.isfinite(av) & np.isfinite(bv)
            if ok.sum() < 4 or np.std(av[ok]) == 0 or np.std(bv[ok]) == 0:
                continue
            rho = float(spearmanr(av[ok], bv[ok]).correlation)
            if abs(rho) >= cfg["min_abs_corr"]:
                pats.append(_pat("spearman", "monotonic_correlation", b, "global",
                                 "monotonic corr(%s, %s): rho = %.3f" % (a, b, rho),
                                 "abs_rho", abs(rho), {"pair": [a, b], "rho": round(rho, 4)}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats


def adapter_clf_tree(rows, schema, cols, num_cols, cat_cols, cfg):
    """Classification rule induction: for each CATEGORICAL target, a shallow
    decision tree -> IF..THEN class rules (the 'IF temp=high THEN succeeds=yes' case)."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import OneHotEncoder
    pats = []
    for y in cat_cols:
        card = schema[y]["cardinality"]
        if card < 2 or card > cfg["max_cat_card"]:
            continue
        yv = np.array([str(_hashable(z)) for z in cols[y]], dtype=object)   # str labels: bool/num/mixed -> valid classifier target
        feats = [c for c in (num_cols + cat_cols) if c != y]
        if not feats:
            continue
        mats, names = [], []
        for f in feats:
            if schema[f]["type"] == "numeric":
                mats.append(_num(cols, f).reshape(-1, 1)); names.append(f)
            else:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                arr = np.array([[str(_hashable(z))] for z in cols[f]], dtype=object)  # uniform str -> encoder-safe
                mats.append(enc.fit_transform(arr))
                names += ["%s=%s" % (f, c) for c in enc.categories_[0]]
        X = np.hstack(mats)
        ok = np.array([z is not None for z in cols[y]])  # require target; impute features
        if ok.sum() < 2:                 # a tree needs >=2 samples
            continue
        Xok = _impute(X[ok])
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(5, int(ok.sum()) // 20),
                                     random_state=0)
        clf.fit(Xok, yv[ok])
        acc = float((clf.predict(Xok) == yv[ok]).mean())
        rules = _tree_rules(clf, list(names), y, classes=[str(c) for c in clf.classes_])
        pats.append(_pat("classification_tree", "rules", y, "global",
                         "classification tree: %d rules, train acc=%.3f" % (len(rules), acc),
                         "accuracy", acc, {"rules": rules}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats


_LINEARTREE_PATCHED = [False]


def _patch_lineartree(LinearTreeRegressor):
    """lineartree calls the sklearn-private self._validate_data(), removed in
    sklearn>=1.8, and passes the renamed `force_all_finite`. Bridge it to the
    current public validate_data() so the LATEST lineartree+sklearn run together
    (this uses sklearn's real current validation -- it is NOT a version rollback)."""
    if _LINEARTREE_PATCHED[0]:
        return
    from sklearn.utils.validation import validate_data

    def _shim(self, X="no_validation", y="no_validation", **kw):
        if "force_all_finite" in kw:                 # renamed in modern sklearn
            kw["ensure_all_finite"] = kw.pop("force_all_finite")
        return validate_data(self, X, y, **kw)

    LinearTreeRegressor._validate_data = _shim
    _LINEARTREE_PATCHED[0] = True


def adapter_linear_tree(rows, schema, cols, num_cols, cat_cols, cfg):
    """Model tree (linear models at the leaves): IF split THEN y = a + b*x."""
    from lineartree import LinearTreeRegressor
    from sklearn.linear_model import LinearRegression
    _patch_lineartree(LinearTreeRegressor)
    pats = []
    for y in num_cols:
        feats = [c for c in num_cols if c != y]
        if not feats:
            continue
        X = np.column_stack([_num(cols, f) for f in feats])
        yv = _num(cols, y)
        ok = np.isfinite(yv)             # require target; impute features
        if ok.sum() < 5:                 # linear-leaf model tree
            continue
        Xok = _impute(X[ok])
        m = LinearTreeRegressor(base_estimator=LinearRegression(), max_depth=2,
                                min_samples_leaf=max(10, int(ok.sum()) // 10)).fit(Xok, yv[ok])
        r2 = _r2(yv[ok], m.predict(Xok))
        n_leaves = sum(1 for nd in m.summary().values() if "models" in nd)
        pats.append(_pat("linear_tree", "model_tree", y, "global",
                         "model tree (linear leaves): %d leaves, R2=%.3f" % (n_leaves, r2),
                         "r2", r2, {"features": feats}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats


def adapter_imodels_rulefit(rows, schema, cols, num_cols, cat_cols, cfg):
    """RuleFit: a sparse model over decision-rule + linear terms (imodels)."""
    from imodels import RuleFitRegressor
    pats = []
    for y in num_cols:
        feats = [c for c in num_cols if c != y]
        if not feats:
            continue
        X = np.column_stack([_num(cols, f) for f in feats])
        yv = _num(cols, y)
        ok = np.isfinite(yv)             # require target; impute features
        if ok.sum() < 5:                 # imodels RuleFit raises below ~5 rows
            continue
        Xok = _impute(X[ok])
        rf = RuleFitRegressor(max_rules=20, random_state=0)
        rf.fit(Xok, yv[ok], feature_names=feats)
        r2 = _r2(yv[ok], rf.predict(Xok))
        rules_df = rf._get_rules()
        nz = rules_df[rules_df.coef != 0]
        if "importance" in nz:
            nz = nz.sort_values("importance", ascending=False)
        terms = [{"rule": str(row["rule"]), "coef": round(float(row["coef"]), 4),
                  "support": round(float(row.get("support", float("nan"))), 3)}
                 for _, row in nz.head(8).iterrows()]
        pats.append(_pat("imodels_rulefit", "rule_ensemble", y, "global",
                         "RuleFit: %d active terms, R2=%.3f" % (len(nz), r2),
                         "r2", r2, {"terms": terms}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats


def adapter_causal_pc(rows, schema, cols, num_cols, cat_cols, cfg):
    """Causal structure discovery (PC algorithm) over numeric columns -> edges."""
    from causallearn.search.ConstraintBased.PC import pc
    if len(num_cols) < 2:
        return []
    data = np.column_stack([_num(cols, c) for c in num_cols])
    ok = np.all(np.isfinite(data), axis=1)
    if ok.sum() < 5:                     # PC's conditional-independence tests are unstable below ~5 rows
        return []
    D = data[ok]
    keep = [i for i in range(D.shape[1]) if np.std(D[:, i]) > 0]   # drop constants (singular matrix)
    if len(keep) < 2:
        return []
    D, names = D[:, keep], [num_cols[i] for i in keep]
    try:
        cg = pc(D, cfg["causal_alpha"], "fisherz", node_names=names, show_progress=False)
    except Exception:
        return []   # singular correlation matrix (collinear cols) / numerical failure -> no edges recoverable
    g = np.asarray(cg.G.graph)
    pats = []
    d = len(names)
    for i in range(d):
        for j in range(i + 1, d):
            a, b = names[i], names[j]
            eij, eji = g[i, j], g[j, i]
            if eij == -1 and eji == 1:
                desc, rel = "%s -> %s" % (a, b), "directed"
            elif eij == 1 and eji == -1:
                desc, rel = "%s -> %s" % (b, a), "directed"
            elif eij == -1 and eji == -1:
                desc, rel = "%s -- %s" % (a, b), "undirected"
            elif eij == 1 and eji == 1:
                desc, rel = "%s <-> %s" % (a, b), "bidirected"
            else:
                continue
            pats.append(_pat("causal_pc", "causal_edge", b, "global",
                             "causal edge: %s (%s)" % (desc, rel), "edge", 1.0,
                             {"relation": rel, "nodes": [a, b]}))
    return pats


def adapter_pysr(rows, schema, cols, num_cols, cat_cols, cfg):
    """PySR symbolic regression (needs a Julia backend; skipped if unavailable)."""
    from pysr import PySRRegressor
    pats = []
    for y in num_cols:
        feats = [c for c in num_cols if c != y]
        if not feats:
            continue
        X = np.column_stack([_num(cols, f) for f in feats])
        yv = _num(cols, y)
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(yv)
        if ok.sum() < cfg.get("pysr_min_rows", 12):       # too few points -> symbolic fit would overfit
            continue
        est = PySRRegressor(niterations=cfg.get("pysr_niterations", 40),
                            binary_operators=["+", "-", "*", "/"],
                            unary_operators=["sqrt", "log", "exp"], progress=False,
                            verbosity=0, model_selection="best",
                            temp_equation_file=True, delete_tempfiles=True)
        est.fit(X[ok], yv[ok])
        eq = str(est.sympy())
        for i, f in enumerate(feats):
            eq = eq.replace("x%d" % i, f)
        r2 = _r2(yv[ok], np.asarray(est.predict(X[ok])))
        pats.append(_pat("pysr_symbolic", "equation", y, "global",
                         "%s = %s   [R2=%.3f]" % (y, eq, r2), "r2", r2,
                         {"equation": "%s = %s" % (y, eq), "features": feats}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats


def adapter_implications(rows, schema, cols, num_cols, cat_cols, cfg):
    """Pairwise categorical IMPLICATIONS & EXCLUSIONS: IF col=v THEN col2=v2,
    ranked by confidence and filtered to informative rules (lift>=1.1). This is
    the right tool for 'if A then B' / 'A and B never co-occur' over categorical
    (incl. present/absent) data -- no apriori truncation or absent-item flooding."""
    n = len(rows)
    if not cat_cols or n < 5:
        return []
    items = {}
    for c in cat_cols:
        for i, v in enumerate(cols[c]):
            if v is not None:
                items.setdefault((c, _hashable(v)), set()).add(i)
    supp = {k: len(s) / n for k, s in items.items()}
    keys = [k for k in items if supp[k] >= cfg["impl_min_support"]]
    pats = []
    for a in keys:
        Sa = items[a]
        for b in keys:
            if a == b or a[0] == b[0]:
                continue
            conf = len(Sa & items[b]) / len(Sa)
            lift = conf / supp[b] if supp[b] else 0.0
            if conf >= cfg["impl_min_conf"] and lift >= 1.1:
                pats.append(_pat("implications", "implication", "%s=%s" % b, "%s=%s" % a,
                                 "IF %s=%s THEN %s=%s" % (a[0], a[1], b[0], b[1]),
                                 "confidence", conf,
                                 {"antecedent": "%s=%s" % a, "consequent": "%s=%s" % b,
                                  "lift": round(lift, 3),
                                  "antecedent_support": round(supp[a], 3)}))
    pats.sort(reverse=True, key=lambda p: p["score"])
    return pats[:60]


def adapter_clustering(rows, schema, cols, num_cols, cat_cols, cfg):
    """Unsupervised CLUSTERS (taxonomy): KMeans over numeric + one-hot categorical
    features; pick k by silhouette; report each cluster's distinctive features."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import OneHotEncoder
    if len(rows) < 4:
        return []
    mats, names = [], []
    for c in num_cols:
        mats.append(_num(cols, c).reshape(-1, 1)); names.append(c)
    for c in cat_cols:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        arr = np.array([[str(_hashable(z)) if z is not None else "?"] for z in cols[c]], dtype=object)  # uniform str
        mats.append(enc.fit_transform(arr))
        names += ["%s=%s" % (c, v) for v in enc.categories_[0]]
    X = _impute(np.hstack(mats))        # impute rather than drop rows with any missing feature
    if len(X) < 4:
        return []
    best = None
    for k in range(2, min(8, max(2, len(X) // 2)) + 1):   # >=2 points per cluster
        try:
            labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(X, labels)
            if best is None or sil > best[0]:
                best = (sil, k, labels)
        except Exception:
            pass
    if best is None:
        return []
    sil, k, labels = best
    names = np.array(names)
    all_rate = X.mean(axis=0)
    pats = []
    for cl in range(k):
        members = X[labels == cl]
        if len(members) == 0:
            continue
        in_rate = members.mean(axis=0)
        distinctive = [str(names[j]) for j in range(X.shape[1])
                       if in_rate[j] >= 0.8 and in_rate[j] - all_rate[j] >= 0.2]
        pats.append(_pat("clustering", "cluster", "cluster%d" % cl, "global",
                         "cluster of %d entities; distinctive: %s" % (len(members), distinctive),
                         "silhouette", float(sil),
                         {"size": int(len(members)), "characteristic_features": distinctive, "k": k}))
    return pats


BUILTIN_ADAPTERS = [
    ("functional_form", adapter_functional_form),
    ("multilinear_regression", adapter_multilinear),
    ("conditional_form", adapter_conditional_form),
    ("group_means", adapter_group_means),
    ("correlation", adapter_correlation),
    ("spearman", adapter_spearman),
    ("implications", adapter_implications),
]
EXTERNAL_ADAPTERS = [
    ("gplearn_symbolic", adapter_gplearn),
    ("sklearn_tree", adapter_sklearn_tree),
    ("classification_tree", adapter_clf_tree),
    ("pysubgroup", adapter_pysubgroup),
    ("associations", adapter_associations),
    ("clustering", adapter_clustering),
    ("linear_tree", adapter_linear_tree),
    ("imodels_rulefit", adapter_imodels_rulefit),
    ("causal_pc", adapter_causal_pc),
    ("pysr_symbolic", adapter_pysr),
]


# ===========================================================================
# Pattern + highlights
# ===========================================================================

def _pat(tool, kind, target, scope, description, metric, score, extra, special_note=None):
    p = {"tool": tool, "kind": kind, "target": target, "scope": scope,
         "description": description, "metric": metric,
         "score": round(float(score), 4), "extra": extra}
    if special_note:
        p["special_note"] = special_note
    return p


# Map each metric onto a rough 0-1 "strength" so highlights can be picked across
# tools. This is a heuristic for the highlights list ONLY -- the per-tool sections
# keep each tool's native score untouched.
def _strength(p):
    m, s = p["metric"], p["score"]
    if m in ("r2", "local_r2", "abs_r", "abs_rho", "eta2", "confidence", "accuracy"):
        return max(0.0, min(1.0, s))
    if m == "lift":
        return max(0.0, min(1.0, (s - 1.0) / 2.0))   # lift 1->0, 3+->1
    if m == "quality":
        return None                                  # not comparable; include if top-of-tool
    return None


def build_highlights(all_pats, cfg):
    scored = []
    for p in all_pats:
        st = _strength(p)
        if st is not None and st >= cfg["highlight_strength"]:
            scored.append((st, p))
    scored.sort(reverse=True, key=lambda t: t[0])
    # consensus: targets independently flagged strong by >1 distinct tool
    from collections import defaultdict
    by_target = defaultdict(set)
    for st, p in scored:
        by_target[p["target"]].add(p["tool"])
    highlights = []
    for st, p in scored[:cfg["max_highlights"]]:
        h = dict(p)
        h["strength"] = round(st, 3)
        consensus = sorted(by_target[p["target"]])
        if len(consensus) > 1:
            h["consensus_tools"] = consensus
        highlights.append(h)
    return highlights


DEFAULT_CFG = {
    "min_r2_report": 0.5, "min_local_r2": 0.9, "min_lift_r2": 0.25,
    "min_subgroup_n": 8, "max_cat_card": 12, "min_eta2": 0.1,
    "min_abs_corr": 0.5, "min_support": 0.1, "min_confidence": 0.6,
    "causal_alpha": 0.05,
    "impl_min_support": 0.1, "impl_min_conf": 0.95,
    "highlight_strength": 0.8, "max_highlights": 15,
    # PySR symbolic regression effort knobs. niterations is the main "search harder" lever
    # (raise it for better equations at the cost of runtime). min_rows guards against fitting
    # a flexible symbolic expression to too few points (overfit): lower it to let pysr run on
    # small tables, but on small NOISY data the equation may be spurious. Tune via cfg.
    "pysr_niterations": 40, "pysr_min_rows": 12,
    # below this many rows, stamp a low-confidence "special_note" on every pattern (a small
    # table can't support a confident fit). This NEVER skips a tool -- it only annotates.
    "small_data_rows": 25,
    # Parallelism is OFF by default: the bottleneck tools (pysr, gplearn) are
    # CPU-bound and multi-core internally, so threading them only oversubscribes the
    # CPU and made wall time WORSE in testing (24s serial vs 28s parallel). The real
    # speed lever is --skip pysr_symbolic,gplearn_symbolic for a fast (~3s) pass.
    "skip_tools": [], "parallel": False, "max_workers": None,
    "tool_timeout": None,        # optional per-solver wall limit (seconds); None = no limit
}


# ===========================================================================
# Input validation + numeric mirrors
# ===========================================================================

_EXPECTED_CELL = (bool, int, float, str)   # None is allowed (= missing)


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def preprocess_table(rows):
    """Make a JSON table robust + solver-friendly before the suite runs:
      1. VALIDATE cell types -- anything not int/float/bool/str/None (e.g. a list or dict)
         is replaced with null (filtered out) and a warning is recorded.
      2. MIRROR bool columns -> add `<col>_mirror_int` (0/1) so numeric solvers can use them.
      3. MIRROR all-string columns whose values all parse as numbers -> add
         `<col>_mirror_int` (integral) or `<col>_mirror_float`.
    Originals are NEVER removed. A mirror is skipped if an equivalent numeric column (or the
    mirror name) already exists. Returns (new_rows, warnings); warnings is a list of
    {level, code, message, ...} dicts (also surfaced in the report under 'warnings_errors')."""
    rows = [dict(r) for r in rows]            # don't mutate the caller's dicts
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    warnings = []

    # 1) type validation -- unexpected cell types -> null
    bad = {}
    for r in rows:
        for k, v in list(r.items()):
            if v is not None and not isinstance(v, _EXPECTED_CELL):
                bad[k] = bad.get(k, 0) + 1
                r[k] = None
    for k in sorted(bad):
        warnings.append({"level": "warning", "code": "unexpected_type", "column": k,
                         "message": "column %r: %d cell(s) of unexpected type (not int/float/bool/str) "
                                    "set to null" % (k, bad[k])})

    colvals = {k: [r.get(k) for r in rows] for k in keys}
    nonnull = {k: [v for v in colvals[k] if v is not None] for k in keys}
    numeric = {k: colvals[k] for k in keys if nonnull[k] and all(_is_number(v) for v in nonnull[k])}

    def equivalent_numeric_exists(target):
        for k, vals in numeric.items():
            if all((a is None) == (b is None) and (a is None or float(a) == float(b))
                   for a, b in zip(target, vals)):
                return k
        return None

    def add_mirror(src, name, target, kind):
        if name in keys:                      # idempotent (already added / user-supplied name)
            return
        ex = equivalent_numeric_exists(target)
        if ex:
            warnings.append({"level": "info", "code": "mirror_exists", "column": src,
                             "message": "%r already has an equivalent numeric column %r; no mirror added" % (src, ex)})
            return
        for r, v in zip(rows, target):
            r[name] = v
        keys.append(name); colvals[name] = target; numeric[name] = target
        warnings.append({"level": "info", "code": "added_mirror", "column": src, "mirror": name,
                         "message": "added numeric mirror %r for %s column %r" % (name, kind, src)})

    # 2) bool -> int(0/1) mirror ; 3) all-string-numeric -> int/float mirror
    for k in list(keys):
        nn = nonnull[k]
        if not nn:
            continue
        if all(isinstance(v, bool) for v in nn):
            target = [int(v) if v is not None else None for v in colvals[k]]
            add_mirror(k, "%s_mirror_int" % k, target, "bool")
        elif all(isinstance(v, str) for v in nn):
            try:
                parsed = [float(v) for v in nn]
            except (ValueError, TypeError):
                continue
            all_int = all(p.is_integer() for p in parsed)
            suffix = "int" if all_int else "float"
            cast = (lambda s: int(float(s))) if all_int else float
            target = [cast(v) if v is not None else None for v in colvals[k]]
            add_mirror(k, "%s_mirror_%s" % (k, suffix), target, "string-numeric")

    return rows, warnings


# ===========================================================================
# Driver
# ===========================================================================

def run_suite(rows, cfg=None, quiet=False):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    rows, warnings = preprocess_table(rows)
    if not quiet:
        for w in warnings:
            print("  [%-7s] %s" % (w["level"], w["message"]), file=sys.stderr)
    schema, cols = infer_schema(rows)
    num_cols = [k for k, v in schema.items() if v["type"] == "numeric"]
    cat_cols = [k for k, v in schema.items() if v["type"] == "categorical"]

    skip = set(cfg.get("skip_tools") or [])
    adapters = [(n, f) for n, f in (BUILTIN_ADAPTERS + EXTERNAL_ADAPTERS) if n not in skip]
    order = {n: i for i, (n, _) in enumerate(adapters)}

    def _run_one(name, fn):
        """Run one adapter; never raises (returns a tagged outcome). The adapters
        are independent and only read shared data, so they are thread-safe."""
        t0 = time.perf_counter()
        try:
            pats = fn(rows, schema, cols, num_cols, cat_cols, cfg)
            return (name, "ran", pats, round(time.perf_counter() - t0, 3), None)
        except ImportError as e:
            return (name, "skip", None, round(time.perf_counter() - t0, 3),
                    "missing dependency: %s" % e)
        except Exception as e:
            return (name, "error", None, round(time.perf_counter() - t0, 3),
                    "%s: %s" % (type(e).__name__, e))

    def _log(o):
        if quiet:
            return
        name, status, pats, dt, reason = o
        if status == "ran":
            print("  [ran]     %-22s %4d patterns  %7.2fs" % (name, len(pats), dt), file=sys.stderr)
        elif status == "skip":
            print("  [skipped] %-22s (missing dep)" % name, file=sys.stderr)
        elif status == "timeout":
            print("  [timeout] %-22s %s" % (name, reason), file=sys.stderr)
        else:
            print("  [error]   %-22s %s" % (name, reason), file=sys.stderr)

    # Adapters are independent -> run concurrently in THREADS (the heavy tools spend
    # their time in GIL-releasing native code: pysr in Julia, gplearn/sklearn in
    # numpy). Use only a FEW workers on purpose: the cheap tools (<0.3s) clear almost
    # immediately, then the 2-3 heavy tools (pysr/gplearn/imodels) run with little
    # mutual contention. Many workers is COUNTERPRODUCTIVE here -- pysr already uses
    # all cores internally, so a wide pool just oversubscribes the CPU and thrashes
    # the GIL (measured: 11 workers made pysr 2x SLOWER). Wall ~= the slowest tool.
    # NOTE -- parallelism is RETAINED FOR THE FUTURE but is OFF by default and is NOT exposed on
    # the CLI. Current testing shows the suite is already effectively parallel: the bottleneck
    # tools (pysr, gplearn) saturate all cores internally, so running the suite "in parallel" (or
    # with more workers) only oversubscribes the CPU and makes the TOTAL wall-clock time WORSE,
    # not better (measured: 24s serial vs 28s parallel; an 11-worker pool made pysr ~2x slower).
    # The code is kept in case a future tool mix is genuinely I/O- or single-core-bound, where
    # threading would help. Opt in (library only) with cfg={"parallel": True, "max_workers": N}.
    max_workers = cfg.get("max_workers") or min(2, len(adapters))
    parallel = bool(cfg.get("parallel", False)) and max_workers > 1 and len(adapters) > 1

    # Submit the heaviest tools FIRST so each grabs a worker for its whole run while
    # the other worker streams through the many cheap tools. (Few workers + heavy-
    # first avoids several multi-core tools running at once and oversubscribing.)
    heavy = {"pysr_symbolic", "gplearn_symbolic", "imodels_rulefit"}
    sub_order = sorted(adapters, key=lambda nf: 0 if nf[0] in heavy else 1)

    timeout = cfg.get("tool_timeout")
    t_wall = time.perf_counter()
    outcomes = []
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_run_one, n, f): n for n, f in sub_order}
            for fut in concurrent.futures.as_completed(futs):
                o = fut.result(); outcomes.append(o); _log(o)
    else:
        # Run light tools FIRST so they always finish before any timed-out heavy/native tool
        # (pysr/gplearn) leaves an orphan thread competing for the CPU/GIL -- otherwise that
        # orphan could starve a cheap tool past its --timeout. Report order is restored by the
        # outcomes.sort() below, so execution order doesn't affect the output.
        for n, f in sorted(adapters, key=lambda nf: 1 if nf[0] in heavy else 0):
            if timeout:
                # Optional per-solver wall limit: stop WAITING on a tool that runs too long and
                # keep every other tool's results. A native solver (pysr/Julia) cannot be force-
                # killed from Python, so its orphan thread keeps running until the process exits
                # via os._exit -- but the report is produced immediately without it.
                _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                _fut = _ex.submit(_run_one, n, f)
                try:
                    o = _fut.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    o = (n, "timeout", None, round(float(timeout), 3),
                         "timed out after %gs (--timeout); other tools' results kept" % timeout)
                _ex.shutdown(wait=False)
            else:
                o = _run_one(n, f)
            outcomes.append(o); _log(o)
    wall = round(time.perf_counter() - t_wall, 3)

    outcomes.sort(key=lambda o: order[o[0]])
    results, ran, skipped, timings, timedout = {}, [], [], {}, []
    for name, status, pats, dt, reason in outcomes:
        if status == "ran":
            timings[name] = dt
            results[name] = {"patterns": pats, "n": len(pats), "seconds": dt}
            ran.append(name)
        else:
            skipped.append({"tool": name, "reason": reason, "seconds": dt})
            if status == "timeout":          # also surfaced as its own list (handy in debug logs)
                timedout.append(name)

    serial = round(sum(timings.values()), 3)
    by_tool = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
    slow = by_tool[0] if by_tool else None
    if not slow:
        note = "no tools ran"
    elif parallel:
        note = ("wall %.1fs on %d workers; slowest tool: %s (%.1fs). per-tool times are "
                "measured under concurrency -- run serially (cfg parallel=False) for solo timings."
                % (wall, max_workers, slow[0], slow[1]))
    else:
        note = "serial %.1fs; slowest tool: %s (%.1fs)" % (serial, slow[0], slow[1])

    all_pats = [p for r in results.values() for p in r["patterns"]]
    if 0 < len(rows) < cfg.get("small_data_rows", 25):   # annotate (never skip) on small tables
        _note = ("the input table has only %d rows; patterns are fit on little data and may be "
                 "overfit or unstable -- treat as low-confidence." % len(rows))
        for _p in all_pats:
            _p.setdefault("special_note", _note)
    return {
        "input_summary": {"n_rows": len(rows), "n_cols": len(schema),
                          "numeric": num_cols, "categorical": cat_cols, "schema": schema},
        "warnings_errors": warnings,
        "timing": {"wall_seconds": wall, "serial_equivalent_seconds": serial,
                   "parallel": parallel, "max_workers": max_workers,
                   "per_tool_sorted": by_tool, "note": note},
        "tools_run": ran, "tools_skipped": skipped, "tools_timedout": timedout,
        "highlights": build_highlights(all_pats, cfg),
        "results_by_tool": results,
    }


def main():
    ap = argparse.ArgumentParser(description="Run a suite of pattern/law-discovery solvers on a JSON table.")
    ap.add_argument("input", help="JSON file (list of row-dicts), or '-' for stdin")
    ap.add_argument("--out", help="write JSON report here (default: stdout)")
    ap.add_argument("--skip", help="comma-separated tool names to skip (e.g. pysr_symbolic,gplearn_symbolic)")
    ap.add_argument("--timeout", type=float, help="optional per-solver wall limit in seconds; a tool "
                    "exceeding it is stopped and listed under tools_skipped, all other results kept")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    raw = sys.stdin.read() if args.input == "-" else open(args.input).read()
    data = json.loads(raw)
    rows = data["data"] if isinstance(data, dict) and "data" in data else data

    cfg = {}
    if args.skip:
        cfg["skip_tools"] = args.skip.split(",")
    if args.timeout:
        cfg["tool_timeout"] = args.timeout
    # Parallelism is off by default (the heavy solvers are already multi-core internally, so
    # threading them oversubscribes the CPU and was slower in testing). It is intentionally NOT
    # exposed on the CLI; a library caller can still opt in with cfg={"parallel": True, ...}.
    report = run_suite(rows, cfg=cfg or None, quiet=args.quiet)

    out = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        if not args.quiet:
            print("\nWrote report to %s" % args.out, file=sys.stderr)
            _print_highlights(report)
    else:
        print(out)

    # Force a clean exit: pysr's Julia runtime spawns non-daemon background threads
    # that otherwise keep the interpreter alive after the report is fully written.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


def _print_highlights(report):
    print("\n=== HIGHLIGHTS (strongest across tools) ===", file=sys.stderr)
    for h in report["highlights"]:
        tag = (" [consensus: %s]" % ",".join(h["consensus_tools"])) if "consensus_tools" in h else ""
        print("  (%.2f) %-22s %s%s" % (h.get("strength", 0), h["tool"], h["description"], tag),
              file=sys.stderr)
    if not report["highlights"]:
        print("  (none above threshold)", file=sys.stderr)
    t = report.get("timing", {})
    print("\n=== TIMING (wall %.2fs%s) ==="
          % (t.get("wall_seconds", 0),
             ", serial-equiv %.2fs" % t["serial_equivalent_seconds"]
             if t.get("parallel") else ""), file=sys.stderr)
    for name, secs in t.get("per_tool_sorted", []):
        print("  %7.2fs  %s" % (secs, name), file=sys.stderr)
    print("  -> %s" % t.get("note", ""), file=sys.stderr)


if __name__ == "__main__":
    main()
