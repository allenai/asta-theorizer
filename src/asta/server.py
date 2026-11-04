"""A2A tasks for the Theorizer agent."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime

# Without this, module loggers default to WARNING and the log.info breadcrumbs below
# (e.g. RUN_VIA_QUEUE routing) never emit in prod — exactly the observability gap that
# made the last incident harder to triage. basicConfig is a no-op if handlers already exist.
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

from ExtractionUtils import loadAPIKeys
from Theorizer1 import (
    Step,
    Theorizer,
    TheoryGenerationWorkflowStructure,
    GENERATION_INPUT_LITERATURE_SUPPORT,
    MAX_WORKFLOW_TIMEOUT_SECONDS,
)
from asta_agent import Asta, AstaContext, NullContext

from .workflow import AstaWorkflow
from .context import inject_papers, inject_schema, inject_extractions, inject_theories
from .credits import CREDIT_TYPE, estimate_paper_count, calc_credits
from .messages import theory_message, schema_message
from .schemas import (
    Query, Mission, RunId,
    PaperStoreInput, RequiredPaperStore,
    ExtractionSchemaIn, ExtractionResultsIn, TheoryStoreIn,
    MaxPapers, CutoffYear, CutoffMonth,
    PrimaryModel, ExtractionModel,
    Objective, MaxTokens, Temperature,
    DoNovelty, SearchPapers,
    DEFAULT_MAX_PAPERS, DEFAULT_KNOWLEDGE_CUTOFF_YEAR, DEFAULT_KNOWLEDGE_CUTOFF_MONTH,
    DEFAULT_MAX_DATA_TOKENS, DEFAULT_GENERATION_TEMPERATURE,
    DEFAULT_MODEL_PRIMARY, DEFAULT_MODEL_EXTRACTION,
    GENERATION_OBJECTIVE_ACCURACY_FOCUS,
)


loadAPIKeys()
theorizer = Theorizer()
asta = Asta(
    "Theorizer",
    description=(
        "Reads scientific papers about a research question and writes theories from them. "
        "Each theory is a small set of laws — short statements like 'X causes Y' or "
        "'increasing A increases B' — with the evidence from the papers that supports "
        "(or contradicts) each one."
    ),
    version="0.1.0",
)


@asta.task("Literature Theory Generation")
def generate_theory(
    theory_query: Query,
    mission_statement: Mission = "",
    paper_store: PaperStoreInput = None,
    extraction_schema: ExtractionSchemaIn = None,
    ctx: AstaContext = NullContext(),
    max_papers_to_retrieve: MaxPapers = DEFAULT_MAX_PAPERS,
    knowledge_cutoff_year: CutoffYear = DEFAULT_KNOWLEDGE_CUTOFF_YEAR,
    knowledge_cutoff_month: CutoffMonth = DEFAULT_KNOWLEDGE_CUTOFF_MONTH,
    model_str_primary: PrimaryModel = DEFAULT_MODEL_PRIMARY,
    model_str_extraction: ExtractionModel = DEFAULT_MODEL_EXTRACTION,
    generation_objective: Objective = GENERATION_OBJECTIVE_ACCURACY_FOCUS,
    max_data_tokens: MaxTokens = DEFAULT_MAX_DATA_TOKENS,
    do_qualified_novelty_evaluation: DoNovelty = True,
    search_additional_papers: SearchPapers = True,
    generation_temperature: Temperature = DEFAULT_GENERATION_TEMPERATURE,
):
    """End-to-end pipeline: find papers, extract evidence, write theories.

    Searches for papers about the research question, pulls relevant
    findings out of each, and writes a small set of theories backed by
    that evidence. Each theory is a few short laws (e.g. "X causes Y")
    with the supporting and conflicting papers attached. Optionally
    checks how novel each law is against the retrieved literature.

    Use this when you have a research question and want literature-grounded
    theories in one shot. If you'd rather inspect the evidence first, use
    Find and Extract; if you already have evidence, use Form Theory."""
    workflow = _new_workflow(theory_query, ctx,
        mission_statement=mission_statement,
        max_papers_to_retrieve=max_papers_to_retrieve,
        knowledge_cutoff_year=knowledge_cutoff_year,
        knowledge_cutoff_month=knowledge_cutoff_month,
        model_str_primary=model_str_primary,
        model_str_extraction=model_str_extraction,
        generation_objective=generation_objective,
        generation_input=GENERATION_INPUT_LITERATURE_SUPPORT,
        max_data_tokens=max_data_tokens,
        do_qualified_novelty_evaluation=do_qualified_novelty_evaluation,
    )
    workflow.theory_workflow_structure["generation_temperature"] = generation_temperature
    if paper_store is not None:
        inject_papers(workflow, paper_store, will_find_papers=search_additional_papers)
    if extraction_schema is not None:
        inject_schema(workflow, extraction_schema)

    steps = []
    if extraction_schema is None:
        steps.append(Step.BUILD_PAPERFINDER_REQUEST)
    if search_additional_papers:
        steps.append(Step.FIND_PAPERS_WITH_PAPERFINDER)
    steps += [Step.EXTRACT_FROM_PAPERS, Step.FORM_THEORY_FROM_LITERATURE, Step.SAVE_THEORY_TO_DISK]
    if do_qualified_novelty_evaluation:
        steps.append(Step.QUALIFIED_NOVELTY_EVALUATION)

    _run_via_queue(ctx, workflow, steps, theory_message)


@asta.task("Build Extraction Schema")
def build_extraction_schema(
    theory_query: Query,
    ctx: AstaContext = NullContext(),
    model_str_primary: PrimaryModel = DEFAULT_MODEL_PRIMARY,
):
    """Build a structured extraction schema for a research question.

    Normalizes the research question, drafts a paper search query, and
    generates a structured extraction schema describing the fields that
    should be pulled from each paper.

    Use this to inspect or curate the schema before running Find and
    Extract or Literature Theory Generation; the produced schema can be
    passed to either via its `extraction_schema` parameter."""
    workflow = _new_workflow(theory_query, ctx,
        model_str_primary=model_str_primary,
    )
    _run_via_queue(ctx, workflow, [Step.BUILD_PAPERFINDER_REQUEST], schema_message)


@asta.task("Find and Extract")
def find_and_extract(
    theory_query: Query,
    mission_statement: Mission = "",
    paper_store: PaperStoreInput = None,
    extraction_schema: ExtractionSchemaIn = None,
    ctx: AstaContext = NullContext(),
    max_papers_to_retrieve: MaxPapers = DEFAULT_MAX_PAPERS,
    knowledge_cutoff_year: CutoffYear = DEFAULT_KNOWLEDGE_CUTOFF_YEAR,
    knowledge_cutoff_month: CutoffMonth = DEFAULT_KNOWLEDGE_CUTOFF_MONTH,
    model_str_primary: PrimaryModel = DEFAULT_MODEL_PRIMARY,
    model_str_extraction: ExtractionModel = DEFAULT_MODEL_EXTRACTION,
    generation_objective: Objective = GENERATION_OBJECTIVE_ACCURACY_FOCUS,
    max_data_tokens: MaxTokens = DEFAULT_MAX_DATA_TOKENS,
    search_additional_papers: SearchPapers = True,
):
    """Find papers and extract theory-relevant findings (no theories).

    Searches for papers about the research question and pulls
    theory-relevant findings — entities, variables, results — out of
    each into a structured table. Stops short of writing theories.

    Use this when you want to inspect or curate the evidence base before
    generating theories, or to feed it into Form Theory in a separate step."""
    workflow = _new_workflow(theory_query, ctx,
        mission_statement=mission_statement,
        max_papers_to_retrieve=max_papers_to_retrieve,
        knowledge_cutoff_year=knowledge_cutoff_year,
        knowledge_cutoff_month=knowledge_cutoff_month,
        model_str_primary=model_str_primary,
        model_str_extraction=model_str_extraction,
        generation_objective=generation_objective,
        generation_input=GENERATION_INPUT_LITERATURE_SUPPORT,
        max_data_tokens=max_data_tokens,
    )
    if paper_store is not None:
        inject_papers(workflow, paper_store, will_find_papers=search_additional_papers)
    if extraction_schema is not None:
        inject_schema(workflow, extraction_schema)

    steps = []
    if extraction_schema is None:
        steps.append(Step.BUILD_PAPERFINDER_REQUEST)
    if search_additional_papers:
        steps.append(Step.FIND_PAPERS_WITH_PAPERFINDER)
    steps += [Step.EXTRACT_FROM_PAPERS, Step.SAVE_THEORY_TO_DISK]

    _run_via_queue(ctx, workflow, steps, "Done extracting from papers.")


@asta.task("Form Theory")
def form_theory(
    run_id: RunId = None,
    theory_query: Query = "",
    extraction_results: ExtractionResultsIn = None,
    paper_store: PaperStoreInput = None,
    ctx: AstaContext = NullContext(),
    model_str_primary: PrimaryModel = DEFAULT_MODEL_PRIMARY,
    generation_objective: Objective = GENERATION_OBJECTIVE_ACCURACY_FOCUS,
    max_data_tokens: MaxTokens = DEFAULT_MAX_DATA_TOKENS,
    generation_temperature: Temperature = DEFAULT_GENERATION_TEMPERATURE,
):
    """Write theories from existing extraction results (skip paper finding).

    Pass either a prior run's id, or your own extraction results.

    Use this after Find and Extract, or to re-run theory generation with
    different settings (model, temperature, accuracy vs novelty focus)
    without paying for extraction again."""
    if not _require_exactly_one(ctx, run_id=run_id, extraction_results=extraction_results):
        return
    if run_id:
        workflow = _resume(run_id, ctx,
            model_str_primary=model_str_primary,
            generation_objective=generation_objective,
            max_data_tokens=max_data_tokens,
        )
        if workflow is None:
            return
    else:
        if not theory_query:
            ctx.fail("theory_query is required when passing extraction_results")
            return
        workflow = _new_workflow(theory_query, ctx,
            model_str_primary=model_str_primary,
            generation_objective=generation_objective,
            max_data_tokens=max_data_tokens,
        )
        inject_extractions(workflow, extraction_results)
        if paper_store is not None:
            inject_papers(workflow, paper_store)

    workflow.theory_workflow_structure["generation_temperature"] = generation_temperature
    _run_via_queue(ctx, workflow, [
        Step.FORM_THEORY_FROM_LITERATURE,
        Step.SAVE_THEORY_TO_DISK,
    ], theory_message)


@asta.task("Evaluate Novelty")
def evaluate_novelty(
    run_id: RunId = None,
    theory_store: TheoryStoreIn = None,
    paper_store: PaperStoreInput = None,
    ctx: AstaContext = NullContext(),
    model_str_primary: PrimaryModel = DEFAULT_MODEL_PRIMARY,
    model_str_extraction: ExtractionModel = DEFAULT_MODEL_EXTRACTION,
):
    """Score each theory's statements against the retrieved literature.

    For each statement, checks whether the retrieved literature already
    says it. Returns a per-statement assessment marking the claim as
    already established, derivable from known work, or genuinely new.
    Pass either a prior run's id, or your own theories.

    Use this to filter theory-generation output, or to revisit novelty
    on theories from an earlier run."""
    if not _require_exactly_one(ctx, run_id=run_id, theory_store=theory_store):
        return
    if run_id:
        workflow = _resume(run_id, ctx,
            do_qualified_novelty_evaluation=True,
            model_str_primary=model_str_primary,
            model_str_extraction=model_str_extraction,
        )
        if workflow is None:
            return
        steps = [Step.QUALIFIED_NOVELTY_EVALUATION]
    else:
        workflow = _new_workflow("", ctx,
            do_qualified_novelty_evaluation=True,
            model_str_primary=model_str_primary,
            model_str_extraction=model_str_extraction,
        )
        inject_theories(workflow, theory_store)
        if paper_store is not None:
            inject_papers(workflow, paper_store)
        # BYO needs SAVE_THEORY_TO_DISK so novelty can read the injected stores back; resumed runs already have them on disk.
        steps = [Step.SAVE_THEORY_TO_DISK, Step.QUALIFIED_NOVELTY_EVALUATION]

    _run_via_queue(ctx, workflow, steps, "Novelty evaluation complete.")


@asta.task("Resume Extraction")
def resume_extraction(
    run_id: RunId,
    paper_store: RequiredPaperStore,
    ctx: AstaContext = NullContext(),
    model_str_extraction: ExtractionModel = DEFAULT_MODEL_EXTRACTION,
    max_data_tokens: MaxTokens = DEFAULT_MAX_DATA_TOKENS,
):
    """Add more papers to a prior run and extract findings from them.

    Use this to recover from failed paper downloads, or to grow the
    evidence base of a prior run without starting over."""
    workflow = _resume(run_id, ctx,
        model_str_extraction=model_str_extraction,
        max_data_tokens=max_data_tokens,
    )
    if workflow is None:
        return
    inject_papers(workflow, paper_store)

    _run_via_queue(ctx, workflow, [
        Step.EXTRACT_FROM_PAPERS,
        Step.SAVE_THEORY_TO_DISK,
    ], "Done extracting from added papers.")


# Helpers


def _require_exactly_one(ctx, **named) -> bool:
    """Validate that exactly one of the named inputs was provided. On violation, fail the
    task with a clear message and return False (callers must `return`). Centralizes the
    run_id-vs-BYO xor check that several tasks share."""
    provided = [name for name, val in named.items() if val is not None and val != ""]
    if len(provided) == 1:
        return True
    names = " or ".join(named)
    if len(provided) > 1:
        ctx.fail(f"Pass exactly one of {names}, not several ({', '.join(provided)} given).")
    else:
        ctx.fail(f"Provide exactly one of {names}.")
    return False


def _new_workflow(theory_query, ctx, **kwargs):
    gen_id = f"gen-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:8]}"
    ctx.set_metadata("run_id", gen_id)
    ctx.set_metadata("workflow_id", gen_id)
    kwargs.setdefault("request_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    kwargs.setdefault("user_id", "api")
    return TheoryGenerationWorkflowStructure(
        generation_id=gen_id, theory_query=theory_query, **kwargs)


def _resume(run_id, ctx, **overrides):
    """Reconstruct a prior run, or fail the task with a clear message and return None
    if the run_id is unknown/expired (callers must `return` on None)."""
    ctx.set_metadata("run_id", run_id)
    ctx.set_metadata("workflow_id", run_id)
    try:
        workflow = TheoryGenerationWorkflowStructure.from_persistence(run_id)
    except ValueError:
        ctx.fail(f"No run found for run_id {run_id!r}; it may have expired or never existed.")
        return None
    workflow.theory_workflow_structure.update(overrides)
    return workflow


def _run_via_queue(ctx, workflow, steps, completion_message):
    """completion_message can be a plain str or a callable(workflow, credits_charged) -> str."""
    if ctx.credits:
        # Gate on the estimated credit cost before queueing work; credits.py owns how
        # that estimate is derived. check() is fail-open, so this is a no-op until a
        # credits backend is configured.
        est_credits = calc_credits(num_papers=estimate_paper_count(workflow, steps))
        if est_credits > 0 and not ctx.credits.check(CREDIT_TYPE, est_credits):
            log.info("CREDITS_GATE blocked gen_id=%s estimate=%s type=%s",
                    workflow.theory_workflow_structure.get("generation_id"),
                    est_credits, CREDIT_TYPE)
            ctx.fail(
                f"Insufficient {CREDIT_TYPE} credits: this run needs about {est_credits}.",
                code="INSUFFICIENT_CREDITS",
            )
            return
    # Log gen_id, declared steps, and completion-fn so bug reports can tell routing bugs from worker bugs.
    log.info(
        "RUN_VIA_QUEUE gen_id=%s steps=%s completion=%s",
        workflow.theory_workflow_structure.get("generation_id"),
        [s.value for s in [Step.INIT] + list(steps) + [Step.COMPLETE]],
        getattr(completion_message, "__name__", repr(completion_message)),
    )
    workflow.set_steps([Step.INIT] + list(steps) + [Step.COMPLETE])
    # Persist before submitting so a follow-up task using the streamed run_id doesn't hit a "no workflow found" window before the worker's first save.
    workflow.save_workflow_to_file()

    msg_fn = completion_message if callable(completion_message) else (lambda _w, _c: completion_message)
    tracker = AstaWorkflow(ctx, workflow, completion_msg_fn=msg_fn)
    theorizer.submit_workflow(workflow)
    # Bounded wait: if the worker never signals completion (a missed terminal callback or a
    # dead worker thread), fail the task instead of hanging the request forever. The grace
    # buffer exceeds the worker's own per-workflow timeout so this only fires on a real stall.
    if not tracker.done.wait(timeout=MAX_WORKFLOW_TIMEOUT_SECONDS + 300):
        log.error("workflow %s never signaled completion; failing task",
                  workflow.theory_workflow_structure.get("generation_id"))
        ctx.fail("Theory generation did not complete in time (worker did not signal completion).")


# ASGI entry point: `uvicorn asta.server:app`
app = asta.app


def run():
    asta.run(port=int(os.environ.get("THEORIZER_PORT", 9000)))
