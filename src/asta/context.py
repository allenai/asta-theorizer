"""Stamp BYO inputs onto a workflow so downstream executors can skip the
phases that would normally have produced them."""

from __future__ import annotations

from Struct import ExtractionQuerySchema as StructSchema

from .schemas import (
    ExtractionSchemaInput,
    ExtractionResultsInput,
    PaperStore,
    TheoryStore,
)


def _load_paper_store_from_uri(uri: str) -> PaperStore:
    """Fetch a paper-store JSON from a file://, s3://, or https:// URI and
    validate it into a PaperStore.

    Implements the schemas-extension contract ("a model-typed input also
    accepts a string URI") for paper_store, which this asta-agent build does
    not resolve server-side. Lets callers hardcode a paper store in S3 (the
    Titanic/DataVoyager pattern) instead of inlining a large payload.
    """
    import json

    if uri.startswith("file://"):
        with open(uri[len("file://"):]) as f:
            data = json.load(f)
    elif uri.startswith("s3://"):
        import boto3
        bucket, key = uri[len("s3://"):].split("/", 1)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        data = json.loads(body)
    elif uri.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(uri, timeout=60) as r:
            data = json.loads(r.read())
    else:
        raise ValueError(
            f"paper_store string must be a file://, s3://, or https:// URI — got {uri[:60]!r}"
        )
    return PaperStore.model_validate(data)


def _hydrate_paper_requests(workflow, request_ids) -> None:
    """Poll injected paper requests to completion (full text) before downstream
    steps run, mirroring make_paperfinder_request_executor's fetch+poll. Used
    when no FIND_PAPERS step will consume the requests; otherwise the OCR work
    is orphaned and the papers stay metadata-only for extraction/novelty."""
    import time
    from Theorizer1 import PAPER_CONVERSION_TIMEOUT_SECONDS

    paperstore = workflow.paper_store
    deadline = time.time() + PAPER_CONVERSION_TIMEOUT_SECONDS
    while time.time() < deadline:
        completed = sum(
            1 for rid in request_ids
            if isinstance(paperstore.is_paper_request_complete(rid), str)
        )
        if completed == len(request_ids):
            break
        time.sleep(5)
    # Cancel any that didn't finish so they don't linger in the worker pool.
    paperstore.cancel_incomplete(request_ids)


def inject_papers(workflow, paper_store: PaperStore, will_find_papers: bool = False) -> None:
    """Merge full-text entries directly; for identifier-only entries, submit
    them to the paperstore's worker pool for hydration.

    When a FIND_PAPERS_WITH_PAPERFINDER step will run (``will_find_papers``),
    that step's poll/cancel loop picks up the request_ids. Otherwise no step
    would, so we poll them to completion here — leaving them un-polled would
    orphan the OCR work and feed metadata-only papers to extraction/novelty."""
    # String URIs aren't resolved server-side in this build; resolve here.
    if isinstance(paper_store, str):
        paper_store = _load_paper_store_from_uri(paper_store)
    # asta-agent 1.0.15 only coerces dict inputs into Pydantic models, so a JSON-array paper_store arrives as a plain list; reject upfront with the expected shape instead of an AttributeError mid-task.
    if not isinstance(paper_store, PaperStore):
        raise ValueError(
            "paper_store must be a PaperStore object with shape "
            '{"paperstore": {<key>: {<paper-fields>}, ...}, '
            '"paperstore_lut": {<alt-key>: <key>, ...}} — '
            f"got {type(paper_store).__name__}."
        )
    raw = paper_store.model_dump(by_alias=True)
    workflow.paper_store.merge_from_dict(raw)

    request_ids = []
    for paper in (raw.get("paperstore") or {}).values():
        if paper.get("paper_markdown"):
            continue
        corpus_id = paper.get("corpus_id")
        s2_id = paper.get("s2_id")
        title = paper.get("title")
        if (corpus_id is None) and (s2_id is None) and (title is None):
            continue
        rid = workflow.paper_store.submit_paper(
            corpus_id=str(corpus_id) if corpus_id is not None else None,
            s2_sha=s2_id,
            title=title,
        )
        if rid is not None:
            request_ids.append(rid)

    if request_ids:
        existing = workflow.theory_workflow_structure.get("injected_paper_request_ids", []) or []
        workflow.theory_workflow_structure["injected_paper_request_ids"] = existing + request_ids
        if not will_find_papers:
            _hydrate_paper_requests(workflow, request_ids)

    existing_keys = workflow.theory_workflow_structure.get("paper_keys", [])
    injected = list(workflow.paper_store.paperstore.keys())
    workflow.theory_workflow_structure["paper_keys"] = list(dict.fromkeys(existing_keys + injected))


def inject_schema(workflow, schema_in: ExtractionSchemaInput) -> None:
    schemas = schema_in.extraction_schemas
    if not schemas:
        raise ValueError("extraction_schema input has no `extraction_schemas` entries")
    schema_id, schema_pyd = next(iter(schemas.items()))
    schema = StructSchema.from_dict(schema_pyd.model_dump(by_alias=True))
    schema.id = schema_id
    workflow.theory_store.add_extraction_schema(schema)
    _stamp_schema(workflow, schema, schema_id)


def inject_extractions(workflow, results_in: ExtractionResultsInput) -> None:
    if not results_in.extraction_schemas:
        raise ValueError("extraction_results input has no `extraction_schemas` entries")
    if not results_in.extraction_results:
        raise ValueError("extraction_results input has no `extraction_results` entries")
    _load_and_stamp(workflow, results_in.model_dump(by_alias=True))


def inject_theories(workflow, theory_store: TheoryStore) -> None:
    if not theory_store.theories:
        raise ValueError("theory_store input has no `theories` entries")
    # Don't stamp a schema: novelty reads the stores from disk, not the schema fields, and a BYO theory_store may carry no extraction_schemas (which used to crash with StopIteration).
    workflow.theory_store.load_from_dict(_as_partial_store(theory_store.model_dump(by_alias=True)))


def _load_and_stamp(workflow, store_dict: dict) -> None:
    workflow.theory_store.load_from_dict(_as_partial_store(store_dict))
    schema_id = next(iter(store_dict["extraction_schemas"].keys()))
    schema = workflow.theory_store.get_extraction_schema(schema_id)
    if schema is None:
        raise ValueError(f"could not load extraction schema {schema_id}")
    _stamp_schema(workflow, schema, schema_id)
    _stamp_completed_results(workflow)


def _stamp_schema(workflow, schema, schema_id: str) -> None:
    # Core seam: records the schema as if BUILD_PAPERFINDER_REQUEST produced it.
    workflow.seed_extraction_schema(schema, schema_id)


def _stamp_completed_results(workflow) -> None:
    # Core seam: records extraction_results as completed so FORM_THEORY skips extraction.
    workflow.seed_completed_extractions()


def _as_partial_store(d: dict) -> dict:
    # TheoryStore.load_from_dict KeyErrors on missing top-level keys; pad them.
    return {
        "theories": d.get("theories", {}),
        "extraction_schemas": d.get("extraction_schemas", {}),
        "extraction_results": d.get("extraction_results", {}),
        "theory_evaluations": d.get("theory_evaluations", {}),
        "all_ids": list(d.get("all_ids", [])),
        "next_ids": d.get("next_ids") or {
            "theory": 0, "extraction-schema": 0,
            "extraction-result": 0, "theory-evaluation": 0,
        },
    }
