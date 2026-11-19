# Export.py
# A theorystore to HTML: an index of the theories, a page per theory, and a page per piece of
# evidence, each naming the paper it came from when the papers' details are given.
#
# Ships with the library, so it works from a pip install as well as from a clone.  The package
# installs no commands; run it as a module:
#
#     python -m theorizer.Export --project my-run          # my-run/theorystore.json -> my-run/html/
#     python -m theorizer.Export --store my-run/theorystore.json --paperstore my-run/paperstore.json --out my-run/html
#
# or from Python:
#
#     th.Export.export_html()                              # the open project, into <project>/html
#     th.Export.export_html(store="my-run/theorystore.json", out="my-run/html")
#
# Uses the engine's ExportTheoryStoreAsHTML_NoPaperstore -- the original exporter patched to render
# the newer fields (method, plain-language names, source schema) and to work without a paperstore.
# With a paperstore (or a papers.jsonl) each evidence page shows the paper's linked title, year,
# venue, TL;DR and abstract; without one, pages show the paper id alone.
# With attribution=True (--attribution) it uses ExportTheoryStoreAsHTML_NoPaperstore_attribution
# instead, which also renders the per-sentence attribution block objective="attribution" adds to
# each law, into <project>/html-attribution.

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from . import Legacy  # noqa: F401  (puts the engine on sys.path)
from . import Project


def _load_paper_details(paperstore:str=None, papers:str=None):
    # (paperstore, paperstore_lut) in the shape the exporter reads
    if (paperstore is not None):
        with open(paperstore) as handle:
            saved = json.load(handle)
        return (saved.get("paperstore", {}) or {}), (saved.get("paperstore_lut", {}) or {})

    store, lut = {}, {}
    if (papers is not None):
        with open(papers) as handle:
            for line in handle:
                line = line.strip()
                if (len(line) == 0):
                    continue
                paper = json.loads(line)
                paper_id = paper.get("paper_id")
                if (paper_id is None):
                    continue
                store[paper_id] = paper
                lut[paper_id] = paper_id
    return store, lut


def newest_novelty_statements(project_dir:str):
    # The per-law files the newest Evaluate.novelty() run wrote into this project, or None
    root = os.path.join(project_dir, "evaluations", "novelty")
    if (not os.path.isdir(root)):
        return None
    for run in sorted(os.listdir(root), reverse=True):
        statements = os.path.join(root, run, "statements")
        if (os.path.isdir(statements)) and any(name.endswith(".json") for name in os.listdir(statements)):
            return statements
    return None


def export_html(store:str=None, out:str=None, paperstore:str=None, papers:str=None, limit:int=None, project_dir:str=None, novelty:str=None, show_progress:bool=True, attribution:bool=False) -> str:
    # Write the HTML and return the path of its index.html.
    #
    # store       a theorystore JSON.  Default: the project's theorystore.json (project_dir, else
    #             the open project)
    # out         where the HTML goes.  Default: <project>/html, next to the store -- the run's own
    #             directory, like everything else a run writes
    # paperstore  a paperstore.json (what fetch_papers() saves).  Default: the project's, if it has one
    # papers      a papers.jsonl (paper_id -> title/year/url), used when there is no paperstore
    # limit       export only this many items, for a quick look
    # novelty     a directory of qualified-novelty results (one JSON per law).  Default: the newest
    #             Evaluate.novelty() run in the project, if there is one.  Each evaluated law's theory
    #             page then shows its verdict and links to a page with the per-paper evidence
    # attribution render each law's per-sentence attribution block (objective="attribution") with the
    #             engine's attribution exporter.  Default out: <project>/html-attribution
    if (attribution):
        import ExportTheoryStoreAsHTML_NoPaperstore_attribution as exporter
    else:
        import ExportTheoryStoreAsHTML_NoPaperstore as exporter

    if (project_dir is None) and (store is None) and (Project.is_open()):
        project_dir = Project.DIRECTORY
    if (store is None):
        if (project_dir is None):
            raise ValueError("Export.export_html(): give store= (a theorystore JSON) or project_dir=, or open a project with Config.init()")
        store = os.path.join(project_dir, Project.THEORY_STORE_FILENAME)
    if (project_dir is None):
        project_dir = os.path.dirname(os.path.abspath(store))
    if (out is None):
        out = os.path.join(project_dir, "html-attribution" if attribution else "html")
    if (novelty is None):
        novelty = newest_novelty_statements(project_dir)
    if (paperstore is None) and (papers is None):
        candidate = os.path.join(project_dir, Project.PAPER_STORE_FILENAME)
        if (os.path.exists(candidate)):
            paperstore = candidate

    if (not os.path.exists(store)):
        raise FileNotFoundError("Export.export_html(): no theorystore at " + str(store) + " -- theories reach it through Pipeline.to_theorystore(); give store= to name another")
    with open(store) as handle:
        saved = json.load(handle)
    theories = saved.get("theories", {})
    schemas = saved.get("extraction_schemas", {})
    results = saved.get("extraction_results", {})
    evaluations = saved.get("theory_evaluations", {})
    paper_store, paper_lut = _load_paper_details(paperstore, papers)

    if (show_progress):
        methods = {}
        for theory in theories.values():
            # to_theorystore() keeps the generator in components["method"]
            method = theory.get("method") or (theory.get("components") or {}).get("method") or "unset"
            methods[method] = methods.get(method, 0) + 1
        print("Export.export_html(): " + str(len(theories)) + " theories, " + str(len(schemas)) + " schemas, " + str(len(results))
              + " extraction results; methods " + str(methods), flush=True)
        if (paperstore is not None):
            with_metadata = sum(1 for paper in paper_store.values() if (paper.get("s2_metadata")))
            print("Export.export_html(): " + str(len(paper_store)) + " papers from " + str(paperstore) + " (" + str(with_metadata) + " with Semantic Scholar metadata)", flush=True)
        elif (papers is not None):
            print("Export.export_html(): " + str(len(paper_store)) + " papers from " + str(papers), flush=True)
        else:
            print("Export.export_html(): no paperstore -- evidence pages will show paper ids only", flush=True)
        if (novelty is not None):
            num = sum(1 for name in os.listdir(novelty) if name.endswith(".json"))
            print("Export.export_html(): qualified novelty for " + str(num) + " law(s) from " + str(novelty), flush=True)

    os.makedirs(out, exist_ok=True)
    exporter.export_all(theories, schemas, results, evaluations, paper_store, paper_lut, out,
                        None,       # no predictive-accuracy evaluations
                        novelty,    # qualified-novelty results, or None
                        DEBUG_LIMIT=limit)

    # The theories and the store itself go alongside, so the export stands on its own
    json_dir = os.path.join(out, exporter.PATH_JSON)
    os.makedirs(json_dir, exist_ok=True)
    theories_only = {tid: {k: v for k, v in theory.items() if (k != "supporting_evidence_ids")} for tid, theory in theories.items()}
    with open(os.path.join(json_dir, "theories.json"), "w") as handle:
        json.dump(theories_only, handle, indent=4, default=str)
    target = os.path.join(json_dir, os.path.basename(store))
    if (os.path.abspath(target) != os.path.abspath(store)):
        shutil.copy(store, target)

    index = os.path.join(out, "index.html")
    if (show_progress):
        print("Export.export_html(): " + index, flush=True)
    return index


def main(argv:list=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m theorizer.Export", description="Export a theorystore to HTML.")
    parser.add_argument("--project", default=None,
                        help="a run's project directory: reads its theorystore.json (and paperstore.json, if any) "
                             "and writes <project>/html unless --out says otherwise")
    parser.add_argument("--store", default=None, help="the theorystore JSON to export (instead of --project)")
    parser.add_argument("--out", default=None, help="the directory to write the HTML into (default: <project>/html)")
    parser.add_argument("--paperstore", default=None,
                        help="a paperstore.json saved by a project; evidence pages then get each paper's linked "
                             "title, venue, year, TL;DR and abstract")
    parser.add_argument("--papers", default=None, help="a papers.jsonl (paper_id -> title/year/url), when there is no paperstore")
    parser.add_argument("--novelty", default=None,
                        help="a directory of qualified-novelty results (default: the newest Evaluate.novelty() run in the project)")
    parser.add_argument("--limit", type=int, default=None, help="cap how many items are exported (for a quick look)")
    parser.add_argument("--attribution", action="store_true",
                        help="also render each law's per-sentence attribution (objective=\"attribution\" theories), "
                             "into <project>/html-attribution unless --out says otherwise")
    args = parser.parse_args(argv)

    if (args.project is None) and (args.store is None):
        parser.error("give --project DIR or --store FILE")

    export_html(store=args.store, out=args.out, paperstore=args.paperstore, papers=args.papers, limit=args.limit,
                project_dir=args.project, novelty=args.novelty, attribution=args.attribution)
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
