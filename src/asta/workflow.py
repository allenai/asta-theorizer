"""Bridges Theorizer workflow callbacks to AstaContext."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from asta_agent import ArtifactRef, AstaContext, PaperRef
from asta_credits import ContextGoneError

from Theorizer1 import Step

from .credits import CREDIT_TYPE, actual_paper_count, calc_credits

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Substep:
    """A substep label that flips from active text to past-tense once it's done."""
    active: str
    completed: str


# Step labels for the agent UI (None hides the step); each carries an ordered list of substeps to walk through.
STEPS: dict[Step, tuple[str | None, list[Substep]]] = {
    Step.INIT: ("Initializing", []),
    Step.BUILD_PAPERFINDER_REQUEST: (
        "Building extraction schema",
        [
            Substep(
                active="Reformulating the theory query into a literature search query",
                completed="Reformulated the theory query into a literature search query",
            ),
            Substep(
                active="Generating a structured extraction schema for the query",
                completed="Generated a structured extraction schema for the query",
            ),
        ],
    ),
    Step.BUILD_EXTRACTION_SCHEMA: (None, []),
    Step.FIND_PAPERS_WITH_PAPERFINDER: (
        "Finding papers",
        [
            Substep(
                active="Searching for relevant papers via PaperFinder and Semantic Scholar",
                completed="Searched for relevant papers via PaperFinder and Semantic Scholar",
            ),
        ],
    ),
    Step.EXTRACT_FROM_PAPERS: ("Extracting from papers", []),
    Step.FORM_THEORY_PARAMETRIC: (
        "Forming theory (parametric)",
        [
            Substep(
                active="Generating candidate theories from parametric knowledge",
                completed="Generated candidate theories from parametric knowledge",
            ),
        ],
    ),
    Step.FORM_THEORY_FROM_LITERATURE: (
        "Forming theories from extracted data",
        [
            Substep(
                active="Synthesizing candidate theories from extracted evidence and refining via self-reflection",
                completed="Synthesized candidate theories from extracted evidence and refined via self-reflection",
            ),
        ],
    ),
    Step.SAVE_THEORY_TO_DISK: (None, []),
    Step.QUALIFIED_NOVELTY_EVALUATION: (
        "Evaluating novelty",
        [
            Substep(
                active="Assessing novelty of each theory law against the retrieved papers across seven dimensions",
                completed="Assessed novelty of each theory law against the retrieved papers across seven dimensions",
            ),
        ],
    ),
}


class AstaWorkflow:
    """Renders step progress, publishes artifacts, and signals completion."""

    def __init__(self, ctx: AstaContext, workflow, completion_msg_fn):
        """completion_msg_fn(workflow, credits_charged) -> str; credits_charged is None
        when no charge landed."""
        self.ctx = ctx
        self.workflow = workflow
        self.wf_data = workflow.theory_workflow_structure
        self.done = threading.Event()
        self.completion_msg_fn = completion_msg_fn

        self.step = None
        self.substep = None
        self.current: Substep | None = None
        self.upcoming: list[Substep] = []
        self._seen_papers: set[str] = set()

        # Single streaming theory-store artifact, opened lazily; each callback appends an RFC 7396 JSON Merge Patch fragment that consumers fold in order to rebuild the TheoryStore + PaperStore.
        self.raw_store = None
        # Serializes the lazy one-time inits (raw_store / substep), which fire from concurrent extraction worker threads.
        self._init_lock = threading.Lock()

        # Subscribe to the workflow's progress events through the core's single observer
        # seam. The core wires these hooks (some live on the workflow, some on its
        # theory_store) and guards each one, so a render error here can't escape into and
        # kill the worker thread — and the SDK example no longer monkey-patches six
        # attributes across two objects.
        workflow.set_observer(self)

        # BYO inputs were stamped before this tracker existed and skip the append callbacks; seed them now so the artifact round-trips the injected entities, not just what this run produced.
        self._seed_injected_store()

    # -- streaming round-trip seed -----------------------------------------

    def _seed_injected_store(self):
        """Append entities already present at construction time (BYO inputs) to
        the streaming theory_store, WITHOUT publishing discrete per-entity
        artifacts — those belong only to entities the skill actually produces."""
        ts = self.workflow.theory_store
        parts: dict = {}
        schemas = {str(sid): s.to_dict() for sid, s in (ts.extraction_schemas or {}).items()}
        if schemas:
            parts["extraction_schemas"] = schemas
        results = {str(rid): r.to_dict() for rid, r in (ts.extraction_results or {}).items()}
        if results:
            parts["extraction_results"] = results
        theories = {str(tid): t.to_dict() for tid, t in (ts.theories or {}).items()}
        if theories:
            parts["theories"] = theories
        # paper_markdown stripped — full text lives only on the asta-theorizer side.
        paperstore = ts.paperstore.paperstore or {}
        lut = ts.paperstore.paperstore_lut or {}
        if paperstore:
            parts["paperstore"] = {
                str(lut.get(pk, pk)): {"id": str(lut.get(pk, pk)),
                                       **{f: v for f, v in paper.items() if f != "paper_markdown"}}
                for pk, paper in paperstore.items()
            }
            parts["paperstore_lut"] = {str(k): str(v) for k, v in lut.items()}
        if parts:
            self._ensure_raw_store()
            self.raw_store.append_part(parts)

    # -- step transitions --------------------------------------------------

    def on_step_change(self, wf):
        # Close out the active substep before the step transitions.
        if self.substep and self.current:
            self.substep.update(self.current.completed)

        if wf.is_at_terminal_step():
            self._terminate(wf)
            return

        label, substeps = STEPS.get(wf.get_current_step_name(), (None, []))
        if not label:
            return

        self.step = self.ctx.start_step(label)
        self.upcoming = list(substeps)
        self.current = None
        self.substep = None
        if self.upcoming:
            self.current = self.upcoming.pop(0)
            self.substep = self.ctx.start_step(self.current.active, parent=self.step)

    def _terminate(self, wf):
        if self.done.is_set():
            return
        try:
            if wf.is_in_error_state():
                self._fail()
            else:
                self._succeed(wf)
        except Exception:
            log.exception("Error finalizing the workflow")
            try:
                self.ctx.fail("An unexpected error occurred while finalizing theory generation.")
            except Exception:
                log.exception("Failed to surface terminal error to the client")
        finally:
            self.done.set()

    def _succeed(self, wf):
        self.ctx.set_metadata("cost", round(wf.get_total_cost(), 4))
        self.ctx.set_metadata("time", round(wf.get_total_time(), 1))
        credits_charged = self._charge_credits()
        # Best-effort, like every other streaming write (Theorizer1.set_observer's
        # guarded()): success was already decided from workflow state in _terminate, and a
        # bookkeeping write must not overturn it -- least of all this one, whose payload
        # consumers re-derive when merging patches.
        try:
            self._on_complete()
        except Exception:
            log.exception("TERMINAL_ARTIFACT_EVENT_FAILED gen_id=%s; completing the run",
                          self.wf_data.get("generation_id"))
        # A graceful empty completion (e.g. no full-text papers, or no relevant results found)
        # records an explanatory note via workflow.set_completed_state(); prefer it over the
        # generic per-skill completion message.
        completion_note = self.wf_data.get("completion_note")
        message = completion_note or self.completion_msg_fn(wf, credits_charged)
        # Append a link to the streaming theory_store artifact; its metadata.type "theory_store"
        # lets the front-end render it by type. The tag is what makes the store reachable at all
        # (the webapp opens the panel only on seeing one), so it is emitted whether or not the
        # finalize above landed -- the id resolves off the streamed parts either way.
        # Skipped for a graceful empty completion -- there's nothing worth showing (no theories,
        # possibly no papers/extractions either), so the link would be pointless at best and
        # confusing at worst.
        if self.raw_store is not None and not completion_note:
            message = f"{message}\n\n{ArtifactRef(self.raw_store.artifact_id, 'View theories').tag()}"
        self.ctx.complete(message)

    def _charge_credits(self) -> int | None:
        """Charge the caller for a successful run; returns the amount charged, or None.

        Best-effort and only on success: failures route through _fail() and cost
        nothing. ctx.credits.consume uses the task id as its idempotency key, so a
        retried terminal callback won't double-charge. We deliberately do NOT fail
        a finished run if the charge can't be applied (e.g. a 402 because the
        balance dropped since the up-front check, or a backend hiccup)."""
        if not self.ctx.credits or not self.ctx.credits.has_credits_backend():
            return None
        num_papers = actual_paper_count(self.workflow)
        num_credits = calc_credits(num_papers)
        if num_credits <= 0:
            log.info("CREDITS_NOT_CHARGED gen_id=%s papers=%s amount=%s type=%s; "
                     "no credits to charge", self.wf_data.get("generation_id"), num_papers, num_credits, CREDIT_TYPE)
            return None
        gen_id = self.wf_data.get("generation_id")
        try:
            result = self.ctx.credits.consume(
                CREDIT_TYPE, num_credits,
                metadata={"generation_id": str(gen_id or ""), "num_papers": num_papers},
            )
            self.ctx.set_metadata("credits_charged", num_credits)
            log.info("CREDITS_CHARGED gen_id=%s papers=%s amount=%s type=%s remaining=%s",
                     gen_id, num_papers, num_credits, CREDIT_TYPE,
                     getattr(result, "available_credits", "?"))
            return num_credits
        except ContextGoneError:
            # The context resolved at the start-of-run check but no longer exists, most
            # likely because the user deleted the thread mid-run; don't charge for it.
            log.info("CREDITS_NOT_CHARGED gen_id=%s papers=%s amount=%s type=%s; "
                     "context no longer exists (thread likely deleted mid-run)",
                     gen_id, num_papers, num_credits, CREDIT_TYPE)
            return None
        except Exception:
            log.warning("CREDITS_CHARGE_FAILED gen_id=%s papers=%s amount=%s type=%s; "
                        "completing run without charging", gen_id, num_papers, num_credits, CREDIT_TYPE,
                        exc_info=True)
            return None

    def _fail(self):
        errors = self.wf_data.get("errors", [])
        raw = errors[-1] if errors else "An unexpected error occurred during theory generation."
        msg = raw.split(": ", 1)[-1] if raw.startswith("ERROR in") else raw
        if self.substep:
            self.substep.update(msg)
            self.substep.finish(is_success=False)
        if self.raw_store is not None:
            # Best-effort for the same reason as in _succeed, and here it also guards the
            # diagnosis: an unguarded raise would swap the actual reason below for
            # _terminate's generic "error while finalizing" message.
            try:
                self.raw_store.finalize()
            except Exception:
                log.exception("TERMINAL_ARTIFACT_EVENT_FAILED gen_id=%s; failing the run",
                              self.wf_data.get("generation_id"))
        # ctx.fail() emits a `failed` event with no message body, so the reason would otherwise
        # only live on the step. Emit it as an interim agent message first (symmetric with the
        # completion message) so the chat surfaces *why* it failed.
        self._emit_message(msg)
        self.ctx.fail(msg, code=self.wf_data.get("error_code"))

    def _emit_message(self, text: str) -> None:
        """Send a free-text agent message as a `working` status-update (an interim chat message).
        TECH DEBT: reaches into the SDK-internal _update_task_status because asta_agent has no
        public 'emit interim message' method yet — replace with ctx.emit_message() once the SDK
        ships one. Best-effort: a delivery failure must never block the terminal path."""
        try:
            import uuid
            from a2a.types import Message, TextPart, TaskState
            self.ctx._run_async(self.ctx._update_task_status(
                TaskState.working,
                message=Message(
                    message_id=str(uuid.uuid4()), role="agent",
                    parts=[TextPart(text=text)],
                    task_id=self.ctx.task_id, context_id=self.ctx.context_id,
                ),
            ))
        except Exception:
            log.warning("could not emit interim agent message", exc_info=True)

    def _advance_substep(self):
        """Flip the active substep to past tense, finish it, and begin the next (if any)."""
        if self.substep and self.current:
            self.substep.update(self.current.completed)
            self.substep.finish()
        if self.upcoming:
            self.current = self.upcoming.pop(0)
            self.substep = self.ctx.start_step(self.current.active, parent=self.step)
        else:
            self.current = None
            self.substep = None

    # -- streaming raw artifact -------------------------------------------

    def _ensure_raw_store(self):
        """Open the streaming theory_store artifact on first emit (exactly once,
        even when concurrent extraction worker threads race to be first)."""
        if self.raw_store is not None:
            return
        with self._init_lock:
            if self.raw_store is not None:
                return
            gen_id = str(self.wf_data.get("generation_id") or "unknown")
            self.raw_store = self.ctx.open_streaming_artifact(
                artifact_id=f"{gen_id}-theory-store-raw",
                subtype="theory_store",
                name="Theory Store",
                description=self.wf_data.get("theory_query", "") or "",
                metadata={
                    "merge_format": "json-merge-patch-rfc7396",
                    "schema_version": "1",
                },
            )

    # -- artifact publish events -------------------------------------------
    # These are the WorkflowObserver hooks the core invokes (wired via set_observer,
    # which guards each so a render error is logged and skipped, never failing the run).

    def on_theory_added(self, store, theory_id):
        theory = store.get_theory(theory_id)
        if not theory:
            return
        if not self.substep:
            self.substep = self.ctx.start_step("Generating theories", parent=self.step)
        self._ensure_raw_store()
        self.raw_store.append_part({"theories": {str(theory.id): theory.to_dict()}})

    def on_extraction_schema_added(self, store, schema_id):
        schema = store.get_extraction_schema(schema_id)
        if not schema:
            return
        self._ensure_raw_store()
        self.raw_store.append_part({"extraction_schemas": {str(schema.id): schema.to_dict()}})
        self._advance_substep()

    def on_extraction_result_added(self, store, result_id):
        result = store.get_extraction_result(result_id)
        if not result:
            return
        # Fired concurrently from extraction worker threads; init the substep once.
        if not self.substep:
            with self._init_lock:
                if not self.substep:
                    self.substep = self.ctx.start_step("Extracting data from papers", parent=self.step)
        self._ensure_raw_store()
        self.raw_store.append_part({"extraction_results": {str(result.id): result.to_dict()}})

    def on_novelty_statement_evaluated(self, theory_id, idx, name, eval, total):
        if not self.substep:
            self.substep = self.ctx.start_step("Evaluating novelty of theory statements", parent=self.step)
        eval_id = f"novelty-{theory_id}-{idx}"
        self._ensure_raw_store()
        self.raw_store.append_part({
            "theory_evaluations": {eval_id: {"id": eval_id, **(eval or {})}},
        })

    def on_paper_retrieved(self, wf, paper_key):
        if paper_key in self._seen_papers:
            return
        self._seen_papers.add(paper_key)
        paper = wf.theory_store.paperstore.get_paper_by_key(paper_key)
        if paper is None:
            return
        s2 = paper.get("s2_metadata") or {}
        corpus_id = s2.get("corpus_id") or s2.get("corpusId") or paper.get("corpus_id")
        ref = PaperRef(title=paper.get("title", paper_key), corpus_id=corpus_id)
        # First paper: end the "Searching..." substep, switch to a "Retrieving papers" container.
        if len(self._seen_papers) == 1:
            self._advance_substep()
            self.substep = self.ctx.start_step("Retrieving papers", parent=self.step)
            self.current = None
        self.ctx.append_item(ref.tag() or ref.title, parent=self.substep)

        # paper_markdown stripped — full text lives only on the asta-theorizer side.
        primary = wf.theory_store.paperstore.paperstore_lut.get(paper_key, paper_key)
        entry = {"id": str(primary), **{f: v for f, v in paper.items() if f != "paper_markdown"}}
        self._ensure_raw_store()
        self.raw_store.append_part({
            "paperstore":     {str(primary): entry},
            "paperstore_lut": {str(paper_key): str(primary)},
        })
    def _on_complete(self):
        """Persist the workflow file and finalize the streaming theory_store
        artifact. The final part carries the canonical `all_ids` / `next_ids`
        so consumers can round-trip the rebuilt store back into a
        TheoryStore.load_from_dict() without re-deriving them."""
        self.workflow.save_workflow_to_file()
        if self.raw_store is not None:
            ts = self.workflow.theory_store
            self.raw_store.finalize({
                "all_ids":  list(getattr(ts, "all_ids", []) or []),
                "next_ids": dict(getattr(ts, "next_ids", {}) or {}),
                "short_name": self.wf_data.get("short_name"),
            })
