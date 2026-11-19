# Pipeline.py
# The Theorizer stages, callable one at a time.
#
# Each function here wraps one function in src/TheorizerProcessing.py, or runs one of the per-paper
# functions over many papers at once.  They do three things and nothing else:
#
#   - fill in the library's defaults (model, worker count, timeout) when the caller does not
#   - reshape data between stages
#   - return the packed dict the src/ function returned, with a little context added
#
# The prompts, the parsing and the retry logic are all in src/.  If a result looks wrong, the answer
# is in the function named in Legacy.py, not here.
#
# A typical order, the one the paper's pipeline ran, is:
#
#   plan_theory_query -> find papers -> discover_variables -> extract_from_papers
#     -> make_variable_report -> extend_schema_with_suggestions -> extract_from_papers again
#     -> make_symbolic_schema -> extract again -> generate_theories
#
# You are not obliged to follow it.  Any stage can be run on its own as long as you have its inputs.

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import time

from . import Config
from . import Legacy
from . import Parallel
from . import Project


#
#   The time limit of a once-per-run stage, and the calls that gave up
#
# A once-per-run stage (planning, reports, schemas, theory generation) makes its model calls deep in
# src/, each with a limit src/ chose: theory generation's was a fixed 15 minutes, and a 169k-token
# prompt on a slow model ran past it and into the retry loop.  @stage_llm_limits gives such a stage
# an llm_timeout_seconds= keyword: every model call it makes, on any thread, gets that limit
# (Config.llm_timeout), and its saved output records the limit and every call that gave up.

class StageCalls:
    # What one stage's model calls did: the limit they had, and the ones that gave up
    def __init__(self, llm_timeout_seconds:int=None):
        self.requested = llm_timeout_seconds
        self.limit = None
        self.start = 0

    def failures(self):
        return [dict(f) for f in Legacy.ExtractionUtils.CALL_FAILURES[self.start:]]


_current_stage = None


def stage_llm_limits(function):
    # Decorator: add llm_timeout_seconds= to a once-per-run stage (see above)
    import functools

    @functools.wraps(function)
    def wrapper(*args, llm_timeout_seconds:int=None, **kwargs):
        global _current_stage
        calls = StageCalls(llm_timeout_seconds)
        calls.start = len(Legacy.ExtractionUtils.CALL_FAILURES)
        saved_stage = _current_stage
        _current_stage = calls
        try:
            with Config.llm_timeout(llm_timeout_seconds):
                calls.limit = Config.get_llm_timeout_seconds()
                packed = function(*args, **kwargs)
        finally:
            _current_stage = saved_stage
        if (isinstance(packed, dict)):
            packed.setdefault("llm_timeout_seconds", calls.limit)
            packed["call_failures"] = calls.failures()
        return packed

    signature = inspect.signature(function)
    parameters = list(signature.parameters.values()) + [inspect.Parameter("llm_timeout_seconds", inspect.Parameter.KEYWORD_ONLY, default=None)]
    wrapper.__signature__ = signature.replace(parameters=parameters)
    return wrapper


#
#   Planning
#

@stage_llm_limits
def plan_theory_query(theory_request:str, model_str:str=None, mission_statement:str=None, max_tokens:int=16000, temperature:float=0.0, use_reflection:bool=False, save_as:str=None):
    # Turn a plain-language request ("build me a theory of X") into a normalised query, a paper
    # search query, an extraction query, and a first extraction schema.
    #
    # Returns the packed dict from convert_theory_request_to_query_and_schema().  The parts you
    # usually want are in packed["output"]: theory_request_normalized, paper_search_query,
    # extraction_query, extraction_schema, short_name.
    Config.ensure_initialized()
    if (model_str is None):
        model_str = Config.DEFAULT_MODEL_PRIMARY

    print("Pipeline.plan_theory_query(): Starting on '" + str(theory_request)[:60] + "...' with " + str(model_str), flush=True)
    packed = Legacy.plan_query_and_schema(theory_request, model_str, max_tokens=max_tokens, temperature=temperature, use_reflection=use_reflection, mission_statement=mission_statement)
    packed["mission_statement"] = mission_statement
    print("Pipeline.plan_theory_query(): Planned '" + str(theory_request)[:60] + "...' ($" + str(round(packed.get("total_cost", 0.0), 4)) + ")", flush=True)
    save_output(packed, "plan_theory_query", save_as)
    return packed


#
#   Fetching papers
#
# Papers come in through PaperImport.import_papers() (files or folders) or through fetch_papers()
# by corpus id.  This build has no paper search: finding
# papers is the caller's job.

def fetch_papers(candidates:list, max_papers:int=None, paper_search_query:str=None, paper_store=None, timeout_seconds:int=1800, poll_seconds:int=5, show_progress:bool=True, save_as:str=None):
    # Fetch full text for a list of candidates (anything with a corpus_id; a title is optional),
    # returning papers in the shape PaperImport.import_papers() does.  Each paper is a
    # download plus an OCR conversion, so this is the slow, paid step; max_papers=None fetches all.
    # Fetched PDFs are converted with Mistral OCR (MISTRAL_API_KEY); without a key a paper stays
    # metadata-only, and only papers with full text are returned.
    Config.ensure_initialized()
    if (max_papers is None):
        max_papers = len(candidates)

    # Submit for full text.  A corpus id is all a paper needs: the Semantic Scholar record it
    # names supplies the title and the open-access links.
    #
    # The store comes from the project, so it is the same one across calls and across sessions: a
    # second search adds to the papers already fetched instead of starting empty, and it is written
    # back to the project directory below.  An explicit paper_store= still wins, and with no project
    # open this is a fresh in-memory store, as before.
    if (paper_store is None):
        paper_store = Project.get_paper_store()
    paper_store.start_thread()

    request_ids = []
    seen = set()
    num_without_corpus_id = 0
    for paper in candidates:
        title = paper.get("title")
        corpus_id = paper.get("corpus_id")
        if (corpus_id is None) or (len(str(corpus_id).strip()) == 0):
            num_without_corpus_id += 1
            continue
        keys = {"corpus:" + str(corpus_id).strip()}
        if (title is not None) and (len(str(title).strip()) > 0):
            keys.add("title:" + str(title).strip().lower())
        if (len(keys & seen) > 0):
            continue

        request_ids.append(paper_store.submit_paper(corpus_id=str(corpus_id).strip(), title=title))
        seen |= keys
        if (len(request_ids) >= max_papers):
            break
    if (num_without_corpus_id > 0) and (show_progress):
        print("Pipeline.fetch_papers(): WARNING: " + str(num_without_corpus_id) + " candidate(s) have no corpus_id and were skipped", flush=True)

    if (show_progress):
        print("Pipeline.fetch_papers(): Fetching full text for " + str(len(request_ids)) + " paper(s) -- each is a download plus an OCR conversion", flush=True)

    # Poll until they are all done or the timeout passes
    start_time = time.time()
    keys = []
    num_reported = 0
    while ((time.time() - start_time) < timeout_seconds):
        keys = []
        for request_id in request_ids:
            key = paper_store.is_paper_request_complete(request_id)
            if (isinstance(key, str)):
                keys.append(key)

        if (show_progress) and (len(keys) > num_reported):
            print("Pipeline.fetch_papers(): " + str(len(keys)) + "/" + str(len(request_ids)) + " ready (" + str(round(time.time() - start_time)) + "s elapsed)", flush=True)
            num_reported = len(keys)

        if (len(keys) >= len(request_ids)):
            break
        time.sleep(poll_seconds)

    # Collect them in the shape the extraction stages expect
    papers = []
    num_without_text = 0
    for key in keys:
        record = paper_store.get_paper_by_key(key) or {}
        markdown = paper_store.get_paper_text_by_key(key)
        if (markdown is None) or (len(str(markdown).strip()) == 0):
            num_without_text += 1
            continue

        papers.append({
            "paper_id": str(key),
            "title": record.get("title"),
            "markdown": markdown,
            "year": record.get("publication_year"),
            "corpus_id": record.get("corpus_id"),
            "s2_paper_id": record.get("s2_id"),
            "paper_urls": record.get("paper_urls"),
            "status": record.get("status"),
            # The Semantic Scholar record: venue, abstract, tldr, url, and the rest.  The HTML
            # exporter reads paper metadata out of this field, so leaving it behind is what made
            # exported evidence pages show a bare paper id with no title and no link.
            "s2_metadata": record.get("s2_metadata"),
        })

    if (show_progress):
        print("Pipeline.fetch_papers(): " + str(len(papers)) + " paper(s) with full text" + (" (" + str(num_without_text) + " had metadata only)" if (num_without_text > 0) else ""), flush=True)
        if (len(keys) < len(request_ids)):
            print("Pipeline.fetch_papers(): WARNING: " + str(len(request_ids) - len(keys)) + " paper(s) did not finish within " + str(timeout_seconds) + "s -- raise timeout_seconds, or carry on with what arrived", flush=True)

    # Pack output
    packed = {
        "paper_search_query": paper_search_query,
        "papers": papers,
        "num_candidates": len(candidates),
        "num_requested": len(request_ids),
        "num_with_full_text": len(papers),
        "paper_store": paper_store,
    }

    if (Project.is_open()):
        # Two writes, because they are two different things.  The packed output is this call's
        # result and is saved like every other stage's.  The PaperStore is the project's corpus --
        # every paper fetched so far, with its full text and Semantic Scholar record -- and has to
        # be written separately or it dies with the process when this function returns.
        paper_store_path = Project.save_paper_store()
        packed["paper_store_path"] = paper_store_path

        # The live PaperStore object itself is not JSON, so it is excluded here; save_paper_store()
        # above is what persists it.
        saved_path = Project.save(save_as if (save_as is not None) else "papers_found",
                                  {k: v for k, v in packed.items() if (k != "paper_store")})
        if (show_progress):
            print("  saved: " + str(saved_path), flush=True)
            if (paper_store_path is not None):
                print("  paperstore: " + str(paper_store_path) + " (" + str(len(paper_store.paperstore)) + " papers)", flush=True)

    return packed



#
#   Per-paper model calls that give up
#
# The engine returns the same empty answer for a call that timed out, a call that failed, and a
# call whose answer held no readable JSON.  Saved as an empty result, a timed-out paper looked
# exactly like a paper that was read in full and had nothing to extract -- and the checkpoint
# marked it done, so a rerun never tried it again.  So a paper with no answer raises instead:
# Parallel records it as a failure with the reason, the checkpoint leaves it out, and a rerun of the
# same command tries it again.

class PaperCallFailed(Exception):
    # reason: "timeout", "prompt_too_long", "max_tokens" (the answer was cut off at max_tokens --
    # often all of it spent on reasoning), "error", "interpreter_shutdown" (from the engine), or
    # "unreadable_answer" (the model finished its answer, but no JSON could be read from it)
    def __init__(self, reason:str, message:str):
        super().__init__(message)
        self.reason = reason


def paper_call_failure(stage:str):
    # The exception for a paper whose call came back with no answer, saying why
    failure = Legacy.ExtractionUtils.get_last_call_failure()
    if (failure is not None):
        return PaperCallFailed(failure.get("reason") or "error", stage + ": " + str(failure.get("message")))
    return PaperCallFailed("unreadable_answer", stage + ": the model answered, but no JSON could be read from its answer")


def text_hash(value) -> str:
    # A short, stable digest of a paper's text (or any JSON-able value), for checkpoint keys
    if (not isinstance(value, str)):
        value = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def paper_checkpoint_key(paper:dict) -> str:
    # A paper's checkpoint key: its id AND its text, so a paper whose text changed (re-converted,
    # edited, or a different paper that came to share the id) is run again, not served the old result
    return str(paper.get("paper_id")) + ":" + text_hash(paper.get("markdown") or "")


def paper_failures(failures:list):
    # Parallel's failure entries carry the whole paper (its full text); keep the paper's id and title
    out = []
    for failure in failures:
        paper = failure.get("item") or {}
        out.append({
            "paper_id": paper.get("paper_id") if isinstance(paper, dict) else None,
            "paper_title": paper.get("title") if isinstance(paper, dict) else None,
            "reason": failure.get("reason") or "error",
            "error": failure.get("error"),
            "error_type": failure.get("error_type"),
        })
    return out


def mark_reported_per_paper(start:int):
    # The calls that gave up during a per-paper stage are reported as failed papers, so a caller
    # reading CALL_FAILURES need not list them a second time as calls
    for failure in Legacy.ExtractionUtils.CALL_FAILURES[start:]:
        failure["reported_per_paper"] = True


def count_by_reason(failures:list):
    counts = {}
    for failure in failures:
        reason = failure.get("reason") or "error"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


#
#   Extraction
#

def extract_from_papers(papers:list, extraction_query:str, extraction_schema:list, model_str:str=None, max_workers:int=None, max_tokens:int=None, temperature:float=0.0, use_reflection:bool=False, theory_store=None, extraction_schema_id:str="schema-1", show_progress:bool=True, save_as:str=None, llm_timeout_seconds:int=None):
    # Run the extraction schema over every paper.
    #
    # papers            list of paper dicts (from PaperImport, or anything with paper_id/markdown)
    # extraction_query  the natural-language request that goes with the schema
    # extraction_schema list of slot dicts: {"name", "type", "description"}
    #
    # Returns a packed dict whose "results" is one record per paper, each carrying the paper's id
    # and title alongside what was extracted -- so a record says where it came from without having
    # to look the paper up again.
    #
    # llm_timeout_seconds  seconds before one paper's model call is abandoned and retried (three
    #                      timeouts and the paper fails); default Config.get_llm_timeout_seconds()
    Config.ensure_initialized()
    if (model_str is None):
        model_str = Config.DEFAULT_MODEL_EXTRACTION
    if (llm_timeout_seconds is None):
        llm_timeout_seconds = Config.get_llm_timeout_seconds()
    if (max_tokens is None):
        max_tokens = Config.DEFAULT_MAX_TOKENS_PER_PAPER
    max_tokens = Config.cap_output_tokens(model_str, max_tokens)

    def extract_one(paper):
        paper_text = paper.get("markdown") or ""
        output = Legacy.extract_entities_from_paper(extraction_query, extraction_schema, paper_text, model_str=model_str, max_tokens=max_tokens, temperature=temperature, use_reflection=use_reflection, max_generation_time_seconds=llm_timeout_seconds)

        response = (output or {}).get("response")
        if (response is None):
            raise paper_call_failure("extraction")
        records = []
        if (isinstance(response, dict)):
            records = response.get("results") or []
        if (not isinstance(records, list)):
            records = [records]

        # Pack output
        packed = {
            "paper_id": paper.get("paper_id"),
            "paper_title": paper.get("title"),
            "records": records,
            "suggested_papers": (response or {}).get("follow_on_papers") or [],
            "cost": (output or {}).get("total_cost", 0.0),
            "model": model_str,
        }
        return packed

    # Checkpoint per paper, so a run that dies partway can be restarted without re-extracting (or
    # re-paying for) the papers already done.  The file is keyed to everything that changes what
    # comes back -- query, schema, schema id, model, max_tokens, temperature, use_reflection -- and
    # each paper within it to its id and its text.
    checkpoint = Project.checkpoint_path(save_as if (save_as is not None) else "extraction",
                                         [extraction_query, extraction_schema, model_str, max_tokens, temperature, use_reflection, extraction_schema_id])
    calls_before = len(Legacy.ExtractionUtils.CALL_FAILURES)
    with Config.per_paper_stage():
        batch = Parallel.run_over_items(extract_one, papers, max_workers=max_workers, description="extraction", show_progress=show_progress,
                                        checkpoint_path=checkpoint, item_key=paper_checkpoint_key)
    mark_reported_per_paper(calls_before)

    # Stamp a uuid onto every extracted record, by putting each paper's records into an
    # ExtractionQueryResult and adding it to a TheoryStore.
    #
    # This is NOT bookkeeping.  The theory-generation prompts ask the model to cite the uuids of the
    # evidence each law rests on; records with no uuid produce theories with empty evidence lists and
    # no way to trace a law back to a paper.  TheoryStore.add_extraction_result() is what assigns the
    # ids, and Struct.py notes that it "must use this method to correctly set the ID and add UUIDs to
    # the extracted data" -- so the ids come from there rather than from a scheme invented here.
    if (theory_store is None):
        theory_store = Project.get_theory_store()

    # A rerun (a resumed checkpoint, or the same papers again) hands back results the store already
    # holds.  Adding them again would give the same records a second id and second uuids, and the
    # theory prompts would see every finding twice.  So a result the store already has -- same
    # paper, same schema id, same records -- reuses the stored one instead.
    # The schema goes in the store under its id, so the store -- and anything exported from it --
    # carries the schema its results were extracted against
    register_schema(theory_store, extraction_schema_id, extraction_schema, extraction_query, model_str)

    held = stored_results_by_content(theory_store)
    num_reused = 0
    for result in batch["results"]:
        records = result.get("records") or []
        if (len(records) == 0):
            continue
        key = result_content_key(result.get("paper_id"), extraction_schema_id, records)
        if (key in held):
            result["extraction_result_id"] = held[key]
            num_reused += 1
            continue
        extraction_result = Legacy.ExtractionQueryResult(
            paper_id=result.get("paper_id"), extraction_schema_id=extraction_schema_id,
            extracted_data=records, potentially_relevant_new_papers=result.get("suggested_papers") or [],
            cost=result.get("cost", 0.0), model_str=model_str)
        result["extraction_result_id"] = theory_store.add_extraction_result(extraction_result)
        held[key] = result["extraction_result_id"]

    # The uuids live on the stored records; give the batch the stored copies, so what this returns
    # and what flatten_records() reads are the same records
    for result in batch["results"]:
        stored = theory_store.get_extraction_result(id=result["extraction_result_id"]) if (result.get("extraction_result_id")) else None
        if (stored is not None) and (isinstance(stored.extracted_data, list)):
            result["records"] = stored.extracted_data

    if (show_progress) and (num_reused > 0):
        print("Pipeline.extract_from_papers(): " + str(num_reused) + " result(s) were already in the store -- reused, not added again", flush=True)

    total_cost = 0.0
    num_records = 0
    num_papers_with_records = 0
    for result in batch["results"]:
        total_cost += result.get("cost", 0.0)
        num_records += len(result.get("records") or [])
        if (len(result.get("records") or []) > 0):
            num_papers_with_records += 1

    num_with_uuid = 0
    for result in batch["results"]:
        for record in result.get("records") or []:
            if (isinstance(record, dict)) and (record.get("uuid")):
                num_with_uuid += 1

    if (show_progress) and (num_records > 0) and (num_with_uuid < num_records):
        print("Pipeline.extract_from_papers(): WARNING: only " + str(num_with_uuid) + " of " + str(num_records) + " record(s) carry a uuid -- theories built on these cannot cite their evidence", flush=True)

    failures = paper_failures(batch["failures"])
    if (show_progress):
        print("Pipeline.extract_from_papers(): " + str(num_records) + " record(s) from " + str(num_papers_with_records) + "/" + str(len(papers)) + " papers ($" + str(round(total_cost, 2)) + ")", flush=True)
        for failure in failures:
            print("Pipeline.extract_from_papers(): WARNING: no extraction for " + str(failure.get("paper_id")) + " (" + str(failure.get("reason")) + "): " + str(failure.get("error"))[:200], flush=True)

    # Pack output
    packed = {
        "extraction_query": extraction_query,
        "extraction_schema": extraction_schema,
        "model": model_str,
        "llm_timeout_seconds": llm_timeout_seconds,
        "results": batch["results"],
        "failures": failures,
        "num_failures_by_reason": count_by_reason(failures),
        "num_records": num_records,
        "num_papers_with_records": num_papers_with_records,
        "num_records_with_uuid": num_with_uuid,
        "theory_store": theory_store,
        "total_cost": total_cost,
    }
    # The TheoryStore is a live object, not data -- save everything else, and write the store itself
    # into the project directory so the next session continues its ids rather than restarting.
    save_output({k: v for k, v in packed.items() if (k != "theory_store")}, "extraction", save_as)
    Project.save_theory_store()
    return packed


def register_schema(theory_store, extraction_schema_id:str, extraction_schema:list, extraction_query:str, model_str:str=None):
    # Add the schema to the store under extraction_schema_id, unless that id is already there.  An
    # id that already holds a DIFFERENT schema is kept, with a warning: results under one id should
    # come from one schema, so a changed schema wants its own extraction_schema_id.
    schemas = getattr(theory_store, "extraction_schemas", None)
    if (schemas is None) or (not hasattr(theory_store, "add_extraction_schema")):
        return None
    existing = schemas.get(extraction_schema_id)
    if (existing is not None):
        existing_slots = existing.schema if hasattr(existing, "schema") else (existing or {}).get("schema")
        if (json.dumps(existing_slots, sort_keys=True, default=str) != json.dumps(extraction_schema, sort_keys=True, default=str)):
            print("Pipeline.extract_from_papers(): WARNING: " + str(extraction_schema_id) + " already holds a different schema in this project; these records are filed under it. Use an unused extraction_schema_id= to keep the two apart", flush=True)
        return extraction_schema_id
    return theory_store.add_extraction_schema(Legacy.ExtractionQuerySchema(
        schema=extraction_schema, extraction_query=extraction_query, supporting_theory_ids=[],
        id=extraction_schema_id, model_str=model_str))


def result_content_key(paper_id, extraction_schema_id, records:list):
    # What makes two extraction results the same: the paper, the schema id, and the records
    # themselves (without the uuids the store stamps on)
    cleaned = []
    for record in (records or []):
        if (isinstance(record, dict)):
            cleaned.append({k: v for k, v in record.items() if (k != "uuid")})
        else:
            cleaned.append(record)
    return (str(paper_id), str(extraction_schema_id), json.dumps(cleaned, sort_keys=True, default=str))


def stored_results_by_content(theory_store):
    # {result_content_key: result id} for every extraction result the store holds
    held = {}
    for result_id, result in (getattr(theory_store, "extraction_results", None) or {}).items():
        if (isinstance(result, dict)):
            paper_id, schema_id, data = result.get("paper_id"), result.get("extraction_schema_id"), result.get("extracted_data")
        else:
            paper_id, schema_id, data = result.paper_id, result.extraction_schema_id, result.extracted_data
        held.setdefault(result_content_key(paper_id, schema_id, data), result_id)
    return held


def flatten_records(extraction_result:dict, save_as:str=None, save:bool=True):
    # Concatenate every paper's extracted data into the one flat list the later stages consume
    # (variable reports, symbolic schemas, theory generation).
    #
    # This reads the records back OUT OF THE THEORYSTORE, as the paper's pipeline did --
    # get_extraction_result(id).extracted_data, extended into one list. Reading them
    # from the store rather than from the batch dict matters because the store is what stamped the
    # uuids onto them; anything that reads around it gets records a theory cannot cite.
    #
    # Nothing is added to a record here. An earlier version of this injected paper_id and
    # paper_title into every record, which the paper's pipeline never did -- so the theory generator was
    # being shown two fields it never normally sees. Provenance comes from the uuid instead: it
    # resolves to the ExtractionQueryResult, which carries the paper_id.
    store = extraction_result.get("theory_store") if isinstance(extraction_result, dict) else None

    records = []
    if (store is not None):
        for result in extraction_result.get("results") or []:
            result_id = result.get("extraction_result_id")
            if (result_id is None):
                continue
            stored = store.get_extraction_result(id=result_id)
            if (stored is None):
                print("Pipeline.flatten_records(): WARNING: extraction result " + str(result_id) + " is not in the store -- skipping", flush=True)
                continue
            if (isinstance(stored.extracted_data, list)):
                records.extend(stored.extracted_data)
    else:
        # No store (e.g. records reloaded from disk in a later session). Fall back to the batch, and
        # say so, because these records may predate the uuid stamping.
        for result in extraction_result.get("results") or []:
            for record in result.get("records") or []:
                if (isinstance(record, dict)):
                    records.append(record)
        if (len(records) > 0):
            print("Pipeline.flatten_records(): NOTE: no TheoryStore on this extraction, reading records directly", flush=True)

    num_with_uuid = 0
    for record in records:
        if (record.get("uuid")):
            num_with_uuid += 1

    print("Pipeline.flatten_records(): " + str(len(records)) + " record(s) from " + str(len(extraction_result.get("results") or [])) + " paper(s)", flush=True)
    if (len(records) > 0) and (num_with_uuid < len(records)):
        print("Pipeline.flatten_records(): WARNING: " + str(len(records) - num_with_uuid) + " record(s) have no uuid -- theories built on these cannot cite their evidence", flush=True)

    if (save):
        save_output(records, "records", save_as)
    return records


#
#   Variable discovery
#

def discover_variables(papers:list, theory_query_information:dict, model_str:str=None, max_workers:int=None, additional_contextual_info=None, max_tokens:int=None, temperature:float=0.0, use_reflection:bool=False, show_progress:bool=True, save_as:str=None, llm_timeout_seconds:int=None):
    # Ask each paper what variables would be worth extracting, before committing to a schema.
    Config.ensure_initialized()
    if (model_str is None):
        model_str = Config.DEFAULT_MODEL_EXTRACTION
    if (llm_timeout_seconds is None):
        llm_timeout_seconds = Config.get_llm_timeout_seconds()
    if (max_tokens is None):
        max_tokens = Config.DEFAULT_MAX_TOKENS_PER_PAPER
    max_tokens = Config.cap_output_tokens(model_str, max_tokens)

    def discover_one(paper):
        paper_text = paper.get("markdown") or ""
        output = Legacy.discover_variables_from_paper(theory_query_information, paper_text, additional_contextual_info=additional_contextual_info, model_str=model_str, max_tokens=max_tokens, temperature=temperature, use_reflection=use_reflection, max_generation_time_seconds=llm_timeout_seconds)
        if ((output or {}).get("response") is None):
            raise paper_call_failure("variable discovery")

        # Pack output
        packed = {
            "paper_id": paper.get("paper_id"),
            "paper_title": paper.get("title"),
            "proposed": (output or {}).get("response"),
            "cost": (output or {}).get("total_cost", 0.0),
            "model": model_str,
        }
        return packed

    # Keyed like extraction: everything that changes a proposal, and each paper's id and text
    checkpoint = Project.checkpoint_path(save_as if (save_as is not None) else "variable_discovery",
                                         [theory_query_information, model_str, additional_contextual_info, max_tokens, temperature, use_reflection])
    calls_before = len(Legacy.ExtractionUtils.CALL_FAILURES)
    with Config.per_paper_stage():
        batch = Parallel.run_over_items(discover_one, papers, max_workers=max_workers, description="variable discovery", show_progress=show_progress,
                                        checkpoint_path=checkpoint, item_key=paper_checkpoint_key)
    mark_reported_per_paper(calls_before)

    total_cost = 0.0
    for result in batch["results"]:
        total_cost += result.get("cost", 0.0)

    failures = paper_failures(batch["failures"])
    if (show_progress):
        print("Pipeline.discover_variables(): Proposals from " + str(len(batch["results"])) + "/" + str(len(papers)) + " papers ($" + str(round(total_cost, 2)) + ")", flush=True)
        for failure in failures:
            print("Pipeline.discover_variables(): WARNING: no proposals for " + str(failure.get("paper_id")) + " (" + str(failure.get("reason")) + "): " + str(failure.get("error"))[:200], flush=True)

    # Pack output
    packed = {
        "theory_query_information": theory_query_information,
        "model": model_str,
        "llm_timeout_seconds": llm_timeout_seconds,
        "results": batch["results"],
        "failures": failures,
        "num_failures_by_reason": count_by_reason(failures),
        "total_cost": total_cost,
    }
    save_output(packed, "variable_discovery", save_as)
    return packed


@stage_llm_limits
def make_variable_report(records:list, theory_query_information, model_str:str=None, max_data_tokens:int=None, max_tokens:int=32000, temperature:float=0.0, use_reflection:bool=True, save_as:str=None):
    # Look across everything extracted so far and report which variables are missing, implicit, or
    # worth splitting out.  Its suggestions feed extend_schema_with_suggestions(), NOT
    # make_schemas_from_variable_discovery() -- see that function's comment for why.
    Config.ensure_initialized()
    if (model_str is None):
        model_str = Config.DEFAULT_MODEL_PRIMARY

    require_records(records, "make_variable_report")
    records, subsampling_rate = subsample(records, max_data_tokens)
    print("Pipeline.make_variable_report(): Starting on " + str(len(records)) + " record(s) with " + str(model_str) + " (one call, this can take several minutes)", flush=True)
    packed = Legacy.generate_variable_report(theory_query_information, records, model_str, max_tokens=max_tokens, temperature=temperature, use_reflection=use_reflection)
    packed["subsampling_rate"] = subsampling_rate
    packed["num_records_used"] = len(records)

    suggestions = ((packed.get("output") or {}).get("suggestions")) or []
    print("Pipeline.make_variable_report(): " + str(len(suggestions)) + " suggestion(s) ($" + str(round(packed.get("total_cost", 0.0), 2)) + ")", flush=True)
    save_output(packed, "variable_report", save_as)
    return packed


#
#   Schemas
#

@stage_llm_limits
def make_schemas_from_variable_discovery(variable_discovery, theory_query_information, model_str:str=None, max_tokens:int=32000, temperature:float=0.0, use_reflection:bool=True, save_as:str=None):
    # Turn per-paper variable discovery into one or more NEW, self-contained extraction schemas.
    # Each comes back with its own extraction_query and its own complete slot list, so each can be
    # extracted with on its own.  The model decides how many.
    #
    # Accepts the packed dict from discover_variables(), or the {paper id: proposed_slots} dict the
    # src/ function wants directly.
    #
    # This is the FIRST of the two ways a schema gets improved.  The other is
    # extend_schema_with_suggestions(), below -- see its comment for when each applies.
    Config.ensure_initialized()
    if (model_str is None):
        model_str = Config.DEFAULT_MODEL_PRIMARY

    discovery = collect_proposed_slots(variable_discovery)
    if (len(discovery) == 0):
        raise ValueError("Pipeline.make_schemas_from_variable_discovery(): No proposed_slots found. Pass the packed dict from discover_variables(); each paper's proposal is in result['proposed']['proposed_slots'].")

    num_slots = 0
    for slots in discovery.values():
        num_slots += len(slots)

    print("Pipeline.make_schemas_from_variable_discovery(): Starting on " + str(num_slots) + " proposed slot(s) from " + str(len(discovery)) + " paper(s) with " + str(model_str), flush=True)
    packed = Legacy.generate_schemas_from_variable_discovery(theory_query_information, discovery, model_str, max_tokens=max_tokens, temperature=temperature, use_reflection=use_reflection)

    schemas = ((packed.get("output") or {}).get("extraction_schemas")) or []
    print("Pipeline.make_schemas_from_variable_discovery(): " + str(len(schemas)) + " schema(s) proposed ($" + str(round(packed.get("total_cost", 0.0), 2)) + ")", flush=True)
    save_output(packed, "schemas", save_as)
    return packed


def collect_proposed_slots(variable_discovery):
    # Normalise whatever the caller has into the {paper id: proposed_slots} dict that
    # generate_schema_from_variable_discovery() expects.  The paper's pipeline filtered the same way: a
    # paper whose proposal did not come back with a proposed_slots list is left out.
    if (isinstance(variable_discovery, dict)) and ("results" in variable_discovery):
        collected = {}
        for result in variable_discovery.get("results") or []:
            proposed = result.get("proposed")
            if (isinstance(proposed, dict)):
                slots = proposed.get("proposed_slots")
                if (isinstance(slots, list)) and (len(slots) > 0):
                    collected[str(result.get("paper_id"))] = slots
        return collected

    if (isinstance(variable_discovery, dict)):
        return variable_discovery

    if (isinstance(variable_discovery, list)):
        return {"proposals": variable_discovery}

    return {}


def extend_schema_with_suggestions(extraction_schema:list, suggestions:list, max_suggestions:int=3, show_progress:bool=True):
    # Add the variable report's suggestions to an existing schema as new slots, and return the
    # extended schema.
    #
    # This is the SECOND of the two ways a schema gets improved, and it is not the same as the
    # first.  make_schemas_from_variable_discovery() asks a model to design whole new schemas from
    # what the papers proposed BEFORE any extraction.  This one takes the report written AFTER
    # extraction and appends its suggested slots to the schema you already have -- so the columns
    # you already have keep their meaning and the records stay comparable.
    #
    # There is no LLM call here: the suggestions already contain the slot names and instructions.
    # This mirrors how the paper's pipeline extended a schema with the report's suggestions.
    if (max_suggestions is not None) and (max_suggestions > 0):
        suggestions = (suggestions or [])[:max_suggestions]

    extended = []
    for slot in (extraction_schema or []):
        extended.append(dict(slot))

    existing_names = set()
    for slot in extended:
        existing_names.add(str(slot.get("name")))

    num_added = 0
    for suggestion in (suggestions or []):
        if (not isinstance(suggestion, dict)):
            continue
        for slot in (suggestion.get("suggested_slots_and_extraction_instructions") or []):
            name = slot.get("slot_name")
            description = slot.get("slot_description_and_extraction_instructions")
            if (name is None) or (description is None):
                continue
            if (str(name) in existing_names):
                continue

            # "str" is the default type, the same default the paper's pipeline used here
            extended.append({"name": str(name), "type": "str", "description": str(description)})
            existing_names.add(str(name))
            num_added += 1

    if (show_progress):
        print("Pipeline.extend_schema_with_suggestions(): " + str(len(extraction_schema or [])) + " slot(s) + " + str(num_added) + " from the report = " + str(len(extended)), flush=True)
        if (num_added == 0) and (len(suggestions or []) > 0):
            print("Pipeline.extend_schema_with_suggestions(): WARNING: None of the suggestions carried suggested_slots_and_extraction_instructions, so nothing was added", flush=True)

    return extended


@stage_llm_limits
def make_symbolic_schema(records:list, theory_query_information, model_str:str=None, max_data_tokens:int=None, max_tokens:int=64000, temperature:float=0.0, use_reflection:bool=True, save_as:str=None):
    # Derive a strongly typed schema -- declared types, units, closed vocabularies, inclusion and
    # exclusion criteria -- from records already extracted.  This is the schema the symbolic
    # extractor uses, and what turns extracted text into a table a solver can work on.
    Config.ensure_initialized()
    if (model_str is None):
        model_str = Config.DEFAULT_MODEL_PRIMARY

    require_records(records, "make_symbolic_schema")
    records, subsampling_rate = subsample(records, max_data_tokens)
    print("Pipeline.make_symbolic_schema(): Starting on " + str(len(records)) + " record(s) with " + str(model_str) + " (one call, this can take several minutes)", flush=True)
    packed = Legacy.generate_symbolic_schema(theory_query_information, records, model_str, max_tokens=max_tokens, temperature=temperature, use_reflection=use_reflection)
    packed["subsampling_rate"] = subsampling_rate
    packed["num_records_used"] = len(records)

    # The prompt asks the model to reply with a single wrapper key so there is one predictable JSON
    # object to parse: {"extraction_schema": {short_name, extraction_query, extraction_schema, ...}}.
    # That wrapper has done its job once the response is parsed, so unwrap it HERE, once, rather than
    # storing the envelope and making every later caller dig through it.  The inner
    # `extraction_schema` is the list of variables; the outer one was just the envelope.
    schema_object = (packed.get("output") or {}).get("extraction_schema")
    if (isinstance(schema_object, dict)):
        packed["symbolic_schema"] = schema_object
        packed["variables"] = schema_object.get("extraction_schema") or []
        packed["extraction_query"] = schema_object.get("extraction_query")
        packed["short_name"] = schema_object.get("short_name")
    else:
        packed["symbolic_schema"] = None
        packed["variables"] = []
        packed["extraction_query"] = None
        print("Pipeline.make_symbolic_schema(): WARNING: the response did not contain a schema under output['extraction_schema']", flush=True)

    print("Pipeline.make_symbolic_schema(): " + str(len(packed["variables"])) + " variable(s) in the schema ($" + str(round(packed.get("total_cost", 0.0), 2)) + ")", flush=True)
    save_output(packed, "symbolic_schema", save_as)
    return packed


#
#   Evidence
#

def evidence_rng(evidence_seed:int=None):
    # A private generator for the evidence draw: seeded (Config.DEFAULT_EVIDENCE_SEED unless given),
    # so the same records give the same sample every run, and the global generator is left alone.
    return random.Random(Config.DEFAULT_EVIDENCE_SEED if (evidence_seed is None) else evidence_seed)


def require_records(records, stage:str):
    # A stage that takes records would otherwise still make its call on an empty list, and bill it.
    if (records is None) or (len(records) == 0):
        raise ValueError("Pipeline." + stage + "(): no records to work from, so no call was made.  Check that the extraction found any (extract_from_papers() prints the count)")


def subsample(records:list, max_data_tokens:int=None, evidence_seed:int=None):
    # Everything that hands evidence to a single LLM call goes through here first.
    #
    # consolodate_results_with_subsampling() sorts by name_short, then takes a random sample without
    # replacement at decreasing rates until the evidence fits.  The sample is drawn with
    # evidence_rng(evidence_seed), so it is the same on every run.  Returns (records, rate), where
    # a rate of 1.0 means nothing was dropped.
    if (max_data_tokens is None):
        max_data_tokens = Config.DEFAULT_MAX_DATA_TOKENS

    result = Legacy.subsample_records(extracted_results_in=records, max_tokens=max_data_tokens, rng=evidence_rng(evidence_seed))
    if (result is None):
        raise RuntimeError("Pipeline.subsample(): Could not reduce the evidence to " + str(max_data_tokens) + " tokens. Raise max_data_tokens, or use fewer records.")

    subsampled = result.get("data_subsampled") or []
    rate = result.get("subsampling_rate", 1.0)

    # Say how big the evidence actually is, so the number in a context_length_exceeded error can be
    # compared against it.  Note this count uses the gpt-4 tokenizer (countTokens), which does not
    # match the model's own: measured 340000 here came back as 284914 from the API, a ratio of 0.84.
    if (len(subsampled) > 0):
        print("Pipeline.subsample(): evidence is ~" + str(Legacy.count_tokens(json.dumps(subsampled, default=str))) + " tokens by the gpt-4 tokenizer (the model's own count will differ)", flush=True)
    if (rate < 1.0):
        print("Pipeline.subsample(): Evidence subsampled to " + str(round(rate * 100)) + "% (" + str(len(subsampled)) + " of " + str(len(records)) + " records) to fit " + str(max_data_tokens) + " tokens", flush=True)

    return subsampled, rate


#
#   Theories
#

def warn_if_no_reflection(use_reflection:bool):
    # Theory generation normally runs with reflection on (the paper's pipeline always did).  Turning
    # it off produces first-pass theories that are not comparable to anything generated normally,
    # and the run is expensive either way, so say so loudly rather than letting it pass silently.
    if (use_reflection):
        return
    print("", flush=True)
    print("*** WARNING: use_reflection=False -- the per-theory reflection stage will NOT run. ***", flush=True)
    print("*** Theories will be first-pass only, and not comparable to normally-generated runs.  ***", flush=True)
    print("", flush=True)


@stage_llm_limits
def generate_theories(records:list, theory_query:str, mission_statement:str=None, objective:str="accuracy", model_str:str=None, max_data_tokens:int=None, max_tokens:int=80000, temperature:float=1.0, use_reflection:bool=True, max_laws_per_theory:int=3, additional_suggestions=None, evidence_seed:int=None, save_as:str=None):
    # Build theories from extracted evidence.
    #
    # objective   "accuracy" is the default: evidence-grounded theories built from the extracted
    #             literature (reflection4, a newer revision of the accuracy-focused generator the
    #             paper's experiments used).  What you want unless you know otherwise.
    #
    #             "novelty" is the novelty-focused generator (nonsafebasin), deliberately pushed away
    #             from the safe answer.  Same temperature as accuracy (1.0, which reasoning models are
    #             not sent at all); the difference is the prompt.  Choose it on purpose.
    #
    #             "attribution" is a research variant that adds a per-sentence `attribution` block
    #             to every law -- the evidence each sentence came from, what that evidence says, and
    #             the inference used to get from one to the other -- and runs a second reflection
    #             pass over only the evidence the theory cites.  Less well studied.
    #
    #             The parametric baselines (Legacy.build_theory_parametric_*) generate with NO
    #             evidence at all, and exist for baseline comparisons.  They are deliberately not
    #             offered here.
    # use_reflection  runs the second stage: after the initial batch, each theory is reflected on in
    #                 a separate call.  This is normally what you want.  It is one extra call per
    #                 theory, each re-sending the full evidence, so a 4-theory run costs about 4.8x
    #                 the single-pass version (measured with 342 records).
    # temperature     1.0, as the paper's pipeline used.  Reasoning models (gpt-5/gpt-6, o-series,
    #                 Claude) are sent no temperature, so with the default models it has no effect.
    # evidence_seed   the seed for subsampling evidence that does not fit (default
    #                 Config.DEFAULT_EVIDENCE_SEED), so a rerun sees the same records
    #
    # The theories come back in packed["theory_response"]["theories_general"] and
    # packed["theory_response"]["theories_specific"].  Note that this is "theory_response" and not
    # "output", which is what the other stages use.
    Config.ensure_initialized()
    if (model_str is None):
        model_str = Config.DEFAULT_MODEL_PRIMARY

    if (objective == "accuracy"):
        build = Legacy.build_theory_accuracy_focused
    elif (objective == "novelty"):
        build = Legacy.build_theory_novelty_focused
    elif (objective == "attribution"):
        build = Legacy.build_theory_attribution
    elif (objective == "attribution-batched"):
        build = Legacy.build_theory_attribution_batched
    else:
        raise ValueError("Pipeline.generate_theories(): objective must be 'accuracy', 'novelty', 'attribution' or 'attribution-batched', not " + str(objective))
    require_records(records, "generate_theories")

    # The batched generator is the one case where what does not fit is not thrown away: it takes the
    # subsample as its first batch and the rest as `results_leftover`, and reflects over further
    # batches drawn from it.  Every other generator sees only the subsample.
    records_leftover = None
    if (objective == "attribution-batched"):
        rng = evidence_rng(evidence_seed)
        split = Legacy.subsample_records_with_leftover(extracted_results_in=records,
                                                       max_tokens=(max_data_tokens if (max_data_tokens is not None) else Config.DEFAULT_MAX_DATA_TOKENS),
                                                       rng=rng)
        if (split is None):
            raise RuntimeError("Pipeline.generate_theories(): Could not reduce the evidence to fit. Raise max_data_tokens, or use fewer records.")
        num_records_in = len(records)
        records = split["data_subsampled"]
        records_leftover = split["data_leftover"]
        subsampling_rate = (len(records) / num_records_in) if (num_records_in > 0) else 1.0
        print("Pipeline.generate_theories(): First batch is " + str(len(records)) + " of " + str(num_records_in) + " record(s); " + str(len(records_leftover)) + " left over for the batched reflections", flush=True)
    else:
        records, subsampling_rate = subsample(records, max_data_tokens, evidence_seed=evidence_seed)

    warn_if_no_reflection(use_reflection)
    print("Pipeline.generate_theories(): Starting on " + str(len(records)) + " record(s) with " + str(model_str) + ", objective=" + str(objective) + ", use_reflection=" + str(use_reflection) + " (this can take several minutes)", flush=True)
    # The novelty generator does not take additional_suggestions; the accuracy one does.  Passing it
    # blindly is a TypeError after the subsampling has already been done.
    arguments = {"mission_statement": mission_statement, "model_str": model_str, "max_tokens": max_tokens,
                 "temperature": temperature, "use_reflection": use_reflection, "max_laws_per_theory": max_laws_per_theory}
    if (additional_suggestions is not None):
        if ("additional_suggestions" in inspect.signature(build).parameters):
            arguments["additional_suggestions"] = additional_suggestions
        else:
            print("Pipeline.generate_theories(): WARNING: the '" + str(objective) + "' generator does not accept additional_suggestions -- ignoring them", flush=True)

    if (records_leftover is not None):
        arguments["results_leftover"] = records_leftover
        # The later batches are drawn from the same seeded generator, so they repeat too
        arguments["rng"] = rng

    packed = build(theory_query, records, **arguments)
    packed["subsampling_rate"] = subsampling_rate
    packed["num_records_used"] = len(records)

    # The theory builders return their result under "theory_response", not "output" like every other
    # stage in TheorizerProcessing does.  The theories themselves are in theories_general and
    # theories_specific.
    theory_response = packed.get("theory_response") or {}
    num_theories = len(theory_response.get("theories_general") or []) + len(theory_response.get("theories_specific") or [])
    print("Pipeline.generate_theories(): " + str(num_theories) + " theories from " + str(len(records)) + " records ($" + str(round(packed.get("total_cost", 0.0), 2)) + ")", flush=True)
    save_output(packed, "theories", save_as)
    return packed

#
#   Saving stage output
#
# Every stage saves itself into the open project directory (see Project.py), so nothing expensive is
# lost by closing a notebook.  save_as="..." gives a save a meaningful name -- "extraction-v2-large"
# rather than "extract_from_papers" -- which is worth doing as soon as there is more than one of
# something.

def save_output(packed, stage_name:str, save_as:str=None):
    # Save a stage's output, and say where it went.  Does nothing if no project is open.
    if (not Project.is_open()):
        return None

    name = save_as if (save_as is not None) else stage_name
    if (_current_stage is not None) and (isinstance(packed, dict)):
        # A once-per-run stage records the limit its calls had and the calls that gave up
        packed.setdefault("llm_timeout_seconds", _current_stage.limit)
        packed["call_failures"] = _current_stage.failures()
    saved_path = Project.save(name, packed)
    print("  saved: " + str(saved_path), flush=True)
    return saved_path

#
#   Theories over the knowledge/inference type vocabularies
#

def stamp_requested_types(theory_response, requested_output_type:str, requested_inference_type:str):
    # Record on every theory WHICH type was asked for.
    #
    # The model already puts its own `output_types` and `inference_types` on each law -- its
    # judgement of what that law turned out to be.  Those are kept untouched.  These two extra
    # fields are a different thing: the type the generator was INSTRUCTED to produce.  Telling the
    # two apart afterwards is the whole point, so they get names that cannot be confused with the
    # model's own (singular "requested_", versus the model's plural "output_types").
    #
    # Theory dicts become Theory.components in the TheoryStore, which is a free-form dict, so these
    # travel with the theory rather than living only in the run's packed output.
    stamped = []
    response = theory_response or {}
    for key in ("theories_general", "theories_specific"):
        for theory in (response.get(key) or []):
            if (not isinstance(theory, dict)):
                continue
            theory["requested_output_type"] = requested_output_type
            theory["requested_inference_type"] = requested_inference_type
            stamped.append(theory)

    return stamped


@stage_llm_limits
def generate_theories_over_types(records:list, theory_query:str, output_types:list=None, inference_types:list=None, mission_statement:str=None, model_str:str=None, max_workers:int=5, max_data_tokens:int=None, max_tokens:int=80000, temperature:float=1.0, use_reflection:bool=True, max_laws_per_theory:int=3, additional_suggestions=None, debug_limit:int=None, debug_seed:int=None, evidence_seed:int=None, evidence=None, show_progress:bool=True, save_as:str=None):
    # Generate theories once per knowledge/output type, then once per inference type.
    #
    # This is N + M runs, not N x M: each output type is requested on its own (with no inference
    # type named), then each inference type on its own.  With the default vocabularies that is
    # 12 + 14 = 26 generation calls, so it is roughly 26x the cost and time of one
    # generate_theories() call.  max_workers runs them concurrently; 5 is deliberately modest,
    # because each call is a large-evidence generation and the reflection pass multiplies it by
    # roughly the number of theories per run.
    #
    # Every theory comes back stamped with requested_output_type / requested_inference_type, so the
    # results can be separated afterwards without trusting the model's own classification.
    #
    # debug_limit runs only that many of the requests, chosen at random across both vocabularies, so
    # a wiring change can be checked for a fraction of the cost.  debug_seed makes that choice
    # repeatable.  Which ones were picked is always printed -- a truncated run must never be
    # mistaken for a complete one.
    Config.ensure_initialized()
    if (model_str is None):
        model_str = Config.DEFAULT_MODEL_PRIMARY
    if (output_types is None):
        output_types = Legacy.OUTPUT_TYPES
    if (inference_types is None):
        inference_types = Legacy.INFERENCE_TYPES

    # An unknown name would reach the model as free text and be paid for; refuse it before any call.
    unknown_output_types = [name for name in (output_types or []) if (name not in Legacy.OUTPUT_TYPES)]
    unknown_inference_types = [name for name in (inference_types or []) if (name not in Legacy.INFERENCE_TYPES)]
    if (len(unknown_output_types) > 0) or (len(unknown_inference_types) > 0):
        message = "Pipeline.generate_theories_over_types(): unknown type name(s)."
        if (len(unknown_output_types) > 0):
            message += "\n  Unknown output_types: " + str(unknown_output_types) + "\n  Valid output_types (Legacy.OUTPUT_TYPES): " + str(Legacy.OUTPUT_TYPES)
        if (len(unknown_inference_types) > 0):
            message += "\n  Unknown inference_types: " + str(unknown_inference_types) + "\n  Valid inference_types (Legacy.INFERENCE_TYPES): " + str(Legacy.INFERENCE_TYPES)
        raise ValueError(message)
    require_records(evidence if (evidence is not None) else records, "generate_theories_over_types")

    # Subsample ONCE, not once per run: every run must see the same evidence, or the comparison
    # between types is confounded by which records each one happened to get.
    #
    # The subsample is seeded (Config.DEFAULT_EVIDENCE_SEED, or evidence_seed), so a later sweep
    # over the same records draws the same subset.  The evidence actually used is also saved, so it
    # can be handed back in as `evidence` to extend a sweep with exactly what the earlier runs saw.
    if (evidence is not None):
        records = evidence
        subsampling_rate = 1.0
        if (show_progress):
            print("Pipeline.generate_theories_over_types(): using the " + str(len(records)) + " record(s) supplied as evidence -- not resubsampling", flush=True)
    else:
        records, subsampling_rate = subsample(records, max_data_tokens, evidence_seed=evidence_seed)
        save_output(records, "evidence", (save_as + "-evidence") if (save_as is not None) else None)

    requests = []
    for output_type in (output_types or []):
        requests.append({"kind": "output", "output_type": output_type, "inference_type": None})
    for inference_type in (inference_types or []):
        requests.append({"kind": "inference", "output_type": None, "inference_type": inference_type})

    num_requested_in_full = len(requests)
    if (debug_limit is not None) and (debug_limit < len(requests)):
        chooser = random.Random(debug_seed)
        requests = chooser.sample(requests, debug_limit)
        print("Pipeline.generate_theories_over_types(): DEBUG_LIMIT is set -- running " + str(len(requests)) + " of " + str(num_requested_in_full) + " request(s). THIS IS NOT A COMPLETE SWEEP.", flush=True)
        for request in requests:
            print("    " + str(request["kind"]) + ": " + str(request["output_type"] or request["inference_type"]), flush=True)

    warn_if_no_reflection(use_reflection)
    if (show_progress):
        print("Pipeline.generate_theories_over_types(): " + str(len(requests)) + " run(s) -- " + str(len(output_types or [])) + " output type(s) + " + str(len(inference_types or [])) + " inference type(s) -- on " + str(len(records)) + " record(s) with " + str(model_str) + ", " + str(max_workers) + " at a time", flush=True)
        print("Pipeline.generate_theories_over_types(): this is roughly " + str(len(requests)) + "x the cost of one generate_theories() call", flush=True)

    def run_one(request):
        packed = Legacy.build_theory_with_types(theory_query, records, mission_statement=mission_statement,
                                                model_str=model_str, max_tokens=max_tokens, temperature=temperature,
                                                use_reflection=use_reflection, max_laws_per_theory=max_laws_per_theory,
                                                additional_suggestions=additional_suggestions,
                                                inference_type=request["inference_type"],
                                                output_type=request["output_type"])
        if (packed is None):
            raise RuntimeError("build_theory_with_types() returned None for " + str(request["output_type"] or request["inference_type"]))

        theories = stamp_requested_types(packed.get("theory_response"), request["output_type"], request["inference_type"])

        # Pack output
        return {
            "requested_kind": request["kind"],
            "requested_output_type": request["output_type"],
            "requested_inference_type": request["inference_type"],
            "num_theories": len(theories),
            "theories": theories,
            "theory_response": packed.get("theory_response"),
            "model": model_str,
            "total_cost": packed.get("total_cost", 0.0),
        }

    # Checkpointed per run, so a crash partway through 26 expensive calls does not throw away the
    # ones that finished.  Keyed to everything that changes a run: the query, the evidence itself
    # (a hash of the records, not just how many), the model and every generation setting.
    checkpoint = Project.checkpoint_path(save_as if (save_as is not None) else "theories_over_types",
                                         [theory_query, model_str, use_reflection, text_hash(records), temperature, max_tokens,
                                          mission_statement, max_laws_per_theory, additional_suggestions])
    batch = Parallel.run_over_items(run_one, requests, max_workers=max_workers, description="theory generation over types",
                                    show_progress=show_progress, checkpoint_path=checkpoint,
                                    item_key=lambda request: str(request["kind"]) + ":" + str(request["output_type"] or request["inference_type"]))

    all_theories = []
    total_cost = 0.0
    for result in batch["results"]:
        all_theories.extend(result.get("theories") or [])
        total_cost += result.get("total_cost", 0.0)

    # A run whose generator came back with nothing parseable is NOT a failure as far as the thread
    # pool is concerned -- it returns cleanly with an empty theory_response.  Say so, or a type that
    # produced no theories at all looks the same as one that was never requested.
    empty_runs = []
    for result in batch["results"]:
        if (result.get("num_theories", 0) == 0):
            empty_runs.append(str(result.get("requested_output_type") or result.get("requested_inference_type")))

    if (show_progress):
        print("Pipeline.generate_theories_over_types(): " + str(len(all_theories)) + " theories from " + str(len(batch["results"])) + "/" + str(len(requests)) + " run(s) ($" + str(round(total_cost, 2)) + ")", flush=True)
        if (len(empty_runs) > 0):
            print("Pipeline.generate_theories_over_types(): WARNING: " + str(len(empty_runs)) + " run(s) produced NO theories (the generator returned nothing parseable): " + ", ".join(empty_runs), flush=True)
            print("Pipeline.generate_theories_over_types(): two causes look identical here. (a) The response was truncated mid-JSON -- max_tokens is currently " + str(max_tokens) + "; a stronger model writes longer answers and needs more, and those calls ARE billed. (b) The prompt was rejected for being too long (context_length_exceeded); those calls are not billed, and the fix is a smaller max_data_tokens, currently " + str(max_data_tokens if (max_data_tokens is not None) else Config.DEFAULT_MAX_DATA_TOKENS) + ". A run costing $0.00 points at (b).", flush=True)
        for failure in batch["failures"]:
            request = failure.get("item") or {}
            print("Pipeline.generate_theories_over_types(): FAILED " + str(request.get("output_type") or request.get("inference_type")) + ": " + str(failure.get("error"))[:90], flush=True)

    # Pack output
    packed = {
        "theory_query": theory_query,
        "model": model_str,
        "output_types": output_types,
        "inference_types": inference_types,
        "mission_statement": mission_statement,
        "num_runs": len(requests),
        "runs": batch["results"],
        "failures": batch["failures"],
        "theories": all_theories,
        "num_theories": len(all_theories),
        "subsampling_rate": subsampling_rate,
        "num_records_used": len(records),
        "evidence_seed": Config.DEFAULT_EVIDENCE_SEED if (evidence_seed is None) else evidence_seed,
        "max_tokens": max_tokens,
        "num_requests_in_full_sweep": num_requested_in_full,
        "debug_limit": debug_limit,
        "is_complete_sweep": (len(requests) == num_requested_in_full),
        "runs_with_no_theories": empty_runs,
        "total_cost": total_cost,
    }
    save_output(packed, "theories_over_types", save_as)
    return packed

#
#   Exporting to a TheoryStore
#

def _supporting_evidence_ids(theory:dict, theory_store) -> list:
    # The extraction results THIS theory's statements cite, not the whole training set and not
    # nothing.  A statement carries the uuids of the data it was drawn from; each datum's uuid was
    # stamped by ExtractionResult.set_id, so the results themselves say which result owns which
    # uuid.  Left empty, every theory renders with no supporting evidence at all.
    index = {}
    for result_id, result in (getattr(theory_store, "extraction_results", None) or {}).items():
        data = getattr(result, "extracted_data", None)
        if (data is None) and isinstance(result, dict):
            data = result.get("extracted_data")
        for datum in (data or []):
            if isinstance(datum, dict) and datum.get("uuid"):
                index[datum["uuid"]] = result_id

    ids, seen = [], set()

    def walk(node):
        if isinstance(node, dict):
            for uuid in (node.get("uuids") or []):
                result_id = index.get(uuid)
                if (result_id is not None) and (result_id not in seen):
                    seen.add(result_id)
                    ids.append(result_id)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(theory)
    return ids


def to_theorystore(theories:list, theory_query:str, model_str:str=None, extraction_schema:list=None, method:str=None, theory_store=None):
    # Assemble generated theories into the theorystore dict that src/ExportTheoryStoreAsHTML*.py
    # reads, so the existing HTML exporter can render them.
    #
    # Theory ids come from TheoryStore.add_theory(), which calls generate_id() -- NOT from a scheme
    # invented here. An earlier version numbered them "theory-0001" upward per call, so two separate
    # sweeps both produced theory-0001 and merging their stores silently dropped one of the two.
    # Passing the same store through successive calls now keeps every id distinct.
    # Default to the session's store.  Ids are unique WITHIN a store, and a fresh one restarts at 1,
    # so two runs that each made their own store would both produce theory-1 and merging them would
    # lose one.  The session store is reloaded from the project directory, so ids keep counting.
    if (theory_store is None):
        theory_store = Project.get_theory_store()

    store = {"theories": {}, "extraction_schemas": {}, "extraction_results": {}, "theory_evaluations": {}}

    # The schemas the store already holds, under their own ids.  An extraction result names the
    # schema it was produced against, and a reader that cannot find that schema shows the theory
    # with no evidence at all -- so the schema has to travel whether or not a caller passed one.
    for schema_id, schema in (getattr(theory_store, "extraction_schemas", None) or {}).items():
        store["extraction_schemas"][schema_id] = schema.to_dict() if hasattr(schema, "to_dict") else schema

    # The extraction results the theories were built from.  Without these the evidence uuids cited by
    # each law have nothing to link to, and the exported pages show evidence text with no references.
    for result_id, result in (getattr(theory_store, "extraction_results", None) or {}).items():
        store["extraction_results"][result_id] = result.to_dict() if hasattr(result, "to_dict") else result

    # Every schema an extraction result points at must exist, whatever it is called.
    # extract_from_papers() puts each schema in the store, so normally they are all there already.
    # For one that is not (a store from an older run), fall back to the schema this call was given
    # rather than leaving a dangling reference.
    if (extraction_schema is not None):
        referenced_ids = {"schema-1"}
        for result in store["extraction_results"].values():
            if (isinstance(result, dict)) and (result.get("extraction_schema_id")):
                referenced_ids.add(result.get("extraction_schema_id"))
        for referenced in sorted(referenced_ids):
            if (referenced not in store["extraction_schemas"]):
                store["extraction_schemas"][referenced] = {
                    "id": referenced, "schema": extraction_schema,
                    "extraction_query": theory_query, "model_str": model_str}

    for theory in (theories or []):
        requested = theory.get("requested_output_type") or theory.get("requested_inference_type")
        # The generator's name, kept on the theory itself so it survives the save (the exporter
        # and the workspace pages read components["method"])
        if ((method if (method is not None) else requested) is not None):
            theory.setdefault("method", method if (method is not None) else requested)
        stored = Legacy.Theory(
            name=theory.get("theory_name", "Unnamed"),
            description=theory.get("theory_description", ""),
            type=theory.get("type", "general"),
            theory_query=theory_query, derived_from=[], theory_evaluation_ids=[],
            components=theory,
            supporting_evidence_ids=_supporting_evidence_ids(theory, theory_store),
            knowledge_cutoff_year=None, knowledge_cutoff_month=None, model_str=model_str)
        theory_id = theory_store.add_theory(stored)

        entry = stored.to_dict()
        # Fields the exporter reads that Theory does not carry
        entry["method"] = method if (method is not None) else requested
        # The schema of the first result the theory cites, else the given one's
        entry["schema_id"] = None
        for result_id in (stored.supporting_evidence_ids or []):
            cited = store["extraction_results"].get(result_id)
            if (isinstance(cited, dict)) and (cited.get("extraction_schema_id")):
                entry["schema_id"] = cited.get("extraction_schema_id")
                break
        if (entry["schema_id"] is None) and (extraction_schema is not None):
            entry["schema_id"] = "schema-1"
        store["theories"][theory_id] = entry

    Project.save_theory_store()
    return store


def save_theorystore(store:dict, filename:str):
    directory = os.path.dirname(os.path.abspath(filename))
    if (len(directory) > 0):
        os.makedirs(directory, exist_ok=True)

    with open(filename, "w") as f:
        json.dump(store, f, indent=2, default=str)

    print("Pipeline.save_theorystore(): Wrote " + str(len(store.get("theories") or {})) + " theories to " + str(filename), flush=True)
    return filename
