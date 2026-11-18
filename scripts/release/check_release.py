# check_release.py
# The live release check for the theorizer library: one real run, from the INSTALLED package, of the
# main flow (AGENTS.md Workflow 1) and the variable-discovery flow (Workflow 2a), optionally scored
# for qualified novelty and exported to HTML.  5 papers for every release; 100 papers with --novelty for a minor release.
#
#     python scripts/release/check_release.py RUN_DIR "QUESTION" --papers 5
#     python scripts/release/check_release.py RUN_DIR "QUESTION" --papers 100 --novelty --s2-query "short keywords"
#
# Run it from a fresh virtualenv that has the wheel (or the PyPI release) installed, and from outside
# the repository, so `import theorizer` is the installed package, not the checkout.  It refuses
# otherwise (--allow-checkout to override).
#
# Keys come from the environment only: OPENAI_API_KEY (or ANTHROPIC_API_KEY), MISTRAL_API_KEY,
# SEMANTIC_SCHOLAR_API_KEY.  Everything the run writes goes in RUN_DIR (use a new one per run), plus
# check-summary.json there.  Exit status 0 means every hard check passed.
#
# Papers: this build has no paper search, so the script searches Semantic Scholar with the plan's
# paper_search_query (often too long for S2's keyword search: give --s2-query as a fallback), keeps
# open-access results, puts repository-hosted PDFs (arXiv, PMC, ...) first because publisher sites
# usually refuse Mistral's fetcher, and fetches them with Pipeline.fetch_papers().

import argparse
import json
import os
import sys
import time
import traceback

import requests

# Hosts whose PDFs Mistral OCR can fetch.  Publisher sites (MDPI, Wiley, T&F, OUP, BMJ, Elsevier) mostly
# answer 403.
REPOSITORY_HOSTS = ("arxiv.org", "ncbi.nlm.nih.gov", "europepmc.org", "biorxiv.org", "medrxiv.org", "aclanthology.org",
                    "openreview.net", "proceedings.neurips.cc", "jmlr.org", "proceedings.mlr.press")


def parse_args():
    parser = argparse.ArgumentParser(description="Live release check for the theorizer library.")
    parser.add_argument("run_dir", help="a new, empty directory for this run")
    parser.add_argument("question", help="the theory question")
    parser.add_argument("--papers", type=int, default=5, help="papers to run on (default 5)")
    parser.add_argument("--flows", default="main,discovery", help="main, discovery, or both (default both)")
    parser.add_argument("--discovery-schemas", type=int, default=1,
                        help="how many of the discovery schemas to extract and theorize (default 1; 0 for all)")
    parser.add_argument("--novelty", action="store_true", help="score the theories with Evaluate.novelty() (~$1-1.50 per eligible law at 100 papers)")
    parser.add_argument("--s2-query", default=None, help="a short keyword query, used when the plan's query finds too few papers")
    parser.add_argument("--allow-checkout", action="store_true", help="allow testing a repository checkout instead of an installed package")
    return parser.parse_args()


def s2_search(query, limit):
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
    found, offset = [], 0
    while (len(found) < limit) and (offset < 1000):
        response = None
        for attempt in range(5):
            response = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                                    params={"query": query, "fields": "title,year,corpusId,openAccessPdf", "openAccessPdf": "", "limit": 100, "offset": offset},
                                    headers={"x-api-key": key} if key else {}, timeout=60)
            if (response.status_code == 429):
                time.sleep(3 * (attempt + 1))
                continue
            response.raise_for_status()
            break
        body = response.json()
        data = body.get("data") or []
        if (not data):
            break
        for paper in data:
            url = (paper.get("openAccessPdf") or {}).get("url")
            if paper.get("corpusId") and url:
                found.append({"corpus_id": str(paper["corpusId"]), "title": paper.get("title"), "year": paper.get("year"), "pdf_url": url})
        offset += 100
        if (offset >= int(body.get("total") or 0)):
            break
        time.sleep(1.2)
    return found


def all_theories(theories):
    response = theories.get("theory_response") or {}
    found = []
    for kind in ("general", "specific"):
        for theory in (response.get("theories_" + kind) or []):
            theory["type"] = kind
            found.append(theory)
    return found


def evidence_check(found, records):
    # Every uuid a law cites should name a record that was extracted
    known = {r.get("uuid") for r in records if r.get("uuid")}
    laws = without = cited = dangling = 0
    for theory in found:
        for statement in (theory.get("theory_statements") or []):
            law = statement.get("law", statement) if isinstance(statement, dict) else {}
            laws += 1
            uuids = []
            for evidence in (law.get("supporting_evidence") or []):
                if isinstance(evidence, dict):
                    uuids += evidence.get("uuids") or []
            if (not uuids):
                without += 1
            cited += len(uuids)
            dangling += sum(1 for u in uuids if u not in known)
    return {"laws": laws, "laws_without_evidence": without, "uuids_cited": cited, "uuids_dangling": dangling}


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    flows = [f.strip() for f in args.flows.split(",") if f.strip()]
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    import theorizer as th
    if ("site-packages" not in th.__file__) and (not args.allow_checkout):
        sys.exit("check_release.py: theorizer was imported from " + th.__file__ + ", not an installed package. "
                 "Run from a fresh virtualenv, outside the repository (or pass --allow-checkout).")
    th.Config.init(project_dir=run_dir)

    summary = {"question": args.question, "papers_wanted": args.papers, "flows": flows, "theorizer": th.__file__,
               "version": _version(), "stages": [], "hard_failures": [], "notes": []}
    start = time.time()

    def stage(name, packed, **extra):
        entry = {"stage": name, "cost": round(float((packed or {}).get("total_cost", 0.0) or 0.0), 4), "elapsed_s": round(time.time() - start)}
        entry.update(extra)
        summary["stages"].append(entry)
        print("CHECK: " + json.dumps(entry), flush=True)

    def theorize(records, query, schema, label):
        theories = th.Pipeline.generate_theories(records, query["theory_request_normalized"], save_as="theories-" + label)
        found = all_theories(theories)
        stored = th.Pipeline.to_theorystore(found, query["theory_request_normalized"], model_str=theories["model"], extraction_schema=schema, method="accuracy")
        evidence = evidence_check(found, records)
        stage(label + "/theories", theories, theories=len(found), stored=len(stored.get("theories") or {}),
              records_seen=theories.get("num_records_used"), subsampling_rate=theories.get("subsampling_rate"), evidence=evidence)
        if (len(found) == 0):
            summary["hard_failures"].append(label + ": no theories")
        if (evidence["uuids_dangling"] > 0) or (evidence["laws_without_evidence"] > 0):
            summary["hard_failures"].append(label + ": evidence chain broken " + json.dumps(evidence))

    def extract(papers, query_text, schema, schema_id, label):
        ext = th.Pipeline.extract_from_papers(papers, query_text, schema, extraction_schema_id=schema_id, save_as="extraction-" + label)
        records = th.Pipeline.flatten_records(ext, save_as="records-" + label)
        stage(label + "/extract", ext, results=len(ext["results"]), failures=ext.get("num_failures_by_reason") or {},
              records=len(records), records_with_uuid=sum(1 for r in records if r.get("uuid")))
        if (len(records) == 0):
            summary["hard_failures"].append(label + ": no records")
        if (sum(1 for r in records if r.get("uuid")) < len(records)):
            summary["hard_failures"].append(label + ": records without uuids")
        if (len(ext.get("failures") or []) > max(1, len(papers) // 20)):
            summary["hard_failures"].append(label + ": " + str(len(ext["failures"])) + " papers failed extraction")
        return records

    try:
        try:
            plan = th.Project.load("plan-theory-query")
        except (FileNotFoundError, ValueError):
            plan = th.Pipeline.plan_theory_query(args.question)
            stage("plan", plan, slots=len(plan["output"]["extraction_schema"]), paper_search_query=plan["output"].get("paper_search_query"))
        query = plan["output"]

        candidates_path = os.path.join(run_dir, "s2-candidates.json")
        if os.path.exists(candidates_path):
            with open(candidates_path) as f:
                candidates = json.load(f)
        else:
            want = int(args.papers * 1.6) + 4
            candidates = s2_search(query.get("paper_search_query") or args.question, 1000)
            summary["s2_hits_for_plan_query"] = len(candidates)
            if (len(candidates) < want * 2) and (args.s2_query):
                seen = {c["corpus_id"] for c in candidates}
                candidates += [c for c in s2_search(args.s2_query, 1000) if c["corpus_id"] not in seen]
            candidates = sorted(candidates, key=lambda c: 0 if any(h in c["pdf_url"] for h in REPOSITORY_HOSTS) else 1)[:want]
            with open(candidates_path, "w") as f:
                json.dump(candidates, f, indent=2)

        fetched = th.Pipeline.fetch_papers(candidates, paper_search_query=query.get("paper_search_query"))
        papers = fetched["papers"][:args.papers]
        stage("fetch", fetched, candidates=len(candidates), with_full_text=fetched["num_with_full_text"], used=len(papers))
        if (len(papers) < args.papers):
            summary["notes"].append("only " + str(len(papers)) + " papers had full text (wanted " + str(args.papers) + ")")
        if (len(papers) == 0):
            summary["hard_failures"].append("no papers with full text")
            raise RuntimeError("no papers")

        if ("main" in flows):
            records = extract(papers, query["extraction_query"], query["extraction_schema"], "schema-1", "main")
            if records:
                theorize(records, query, query["extraction_schema"], "main")

        if ("discovery" in flows):
            discovery = th.Pipeline.discover_variables(papers, query, save_as="variable-discovery")
            slots = th.Pipeline.collect_proposed_slots(discovery)
            stage("discovery/discover", discovery, results=len(discovery["results"]), failures=discovery.get("num_failures_by_reason") or {},
                  slots=sum(len(v) for v in slots.values()))
            schemas = th.Pipeline.make_schemas_from_variable_discovery(discovery, query, save_as="schemas-from-discovery")
            proposed = (schemas.get("output") or {}).get("extraction_schemas") or []
            stage("discovery/schemas", schemas, schemas=len(proposed))
            if (not proposed):
                summary["hard_failures"].append("discovery: no schemas")
            chosen = proposed if (args.discovery_schemas == 0) else proposed[:args.discovery_schemas]
            for n, schema in enumerate(chosen, start=2):
                label = "schema-" + str(n)
                records = extract(papers, schema["extraction_query"], schema["extraction_schema"], label, label)
                if records:
                    theorize(records, query, schema["extraction_schema"], label)

        if (args.novelty):
            with open(os.path.join(run_dir, "theorystore.json")) as f:
                store = json.load(f)
            novelty = th.Evaluate.novelty(store)
            stage("novelty", novelty, laws_scored=novelty.get("num_laws"), failed_calls=len(novelty.get("call_failures") or []))
            if (novelty.get("num_laws", 0) == 0):
                summary["notes"].append("novelty: no law was rated new or somewhat new, so none was scored")

        index = th.Export.export_html()
        out = os.path.dirname(index)
        pages = len(os.listdir(os.path.join(out, "theories")))
        novelty_pages = len(os.listdir(os.path.join(out, "qualifiednovelty"))) if os.path.isdir(os.path.join(out, "qualifiednovelty")) else 0
        summary["export"] = {"index": index, "theory_pages": pages, "novelty_pages": novelty_pages}
        if (pages == 0):
            summary["hard_failures"].append("export: no theory pages")

        th.Config.init(project_dir=run_dir, verbose=False)      # reopen from disk: the store must reload
        store = th.Project.get_theory_store()
        summary["reloaded_store"] = {"theories": len(store.theories), "extraction_results": len(store.extraction_results)}
    except Exception as e:
        summary["hard_failures"].append("crashed: " + repr(e))
        summary["traceback"] = traceback.format_exc()
        traceback.print_exc()

    summary["total_cost"] = round(sum(s["cost"] for s in summary["stages"]), 2)
    summary["elapsed_min"] = round((time.time() - start) / 60, 1)
    with open(os.path.join(run_dir, "check-summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("")
    print("check_release.py: theorizer " + str(summary["version"]) + " -- $" + str(summary["total_cost"]) + " in " + str(summary["elapsed_min"]) + " min")
    for note in summary["notes"]:
        print("  note: " + note)
    for failure in summary["hard_failures"]:
        print("  FAIL: " + failure)
    print("check_release.py: " + ("PASS" if not summary["hard_failures"] else "FAIL") + " -- " + os.path.join(run_dir, "check-summary.json"))
    return 0 if not summary["hard_failures"] else 1


def _version():
    try:
        from importlib.metadata import version
        return version("asta-theorizer")
    except Exception:
        return "unknown"


if (__name__ == "__main__"):
    sys.exit(main())
