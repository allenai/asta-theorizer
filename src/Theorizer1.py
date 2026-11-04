# Theorizer.py
from __future__ import annotations

import os
import json
import time
import traceback
import logging

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Deque, Dict, Optional, Any

import uuid

from Struct import *
from SemanticScholar import *
from PaperStore import *
from PaperFinderRequests import *
from SchemaExtractionQueue import *
from MistralOCRStore import *

from ExtractionUtils import *
from TheorizerProcessing import *

import Persistence
from EvaluationQualifiedNovelty import *

# Default model, if no other model is specified
#MODEL_STR_TO_USE = "gpt-4.1-2025-04-14"

#DEFAULT_MODEL_PRIMARY = "gpt-4.1-2025-04-14"
DEFAULT_MODEL_PRIMARY = "gpt-5.2"
DEFAULT_MODEL_EXTRACTION = "gpt-5-mini"

# Generation objectives and inputs
GENERATION_OBJECTIVE_ACCURACY_FOCUS = "accuracy-focused"
GENERATION_OBJECTIVE_NOVELTY_FOCUS = "novelty-focused"
GENERATION_INPUT_LITERATURE_SUPPORT = "literature-support"
GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY = "parametric knowledge only"

ERROR_CODE_PAPERS_NOT_FOUND = "Papers_not_found"

# With 10 parallel threads, 100 papers completes in ~6m. 20m timeout gives generous buffer.
PAPER_CONVERSION_TIMEOUT_SECONDS = 1200

# Directory for debug output files (extraction results, theory results, etc.)
DEBUG_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "debug")
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

# Catch-all workflow timeouts (overridable via environment variables for testing)
MAX_STEP_TIMEOUT_SECONDS = int(os.environ.get("MAX_STEP_TIMEOUT_SECONDS", 60 * 90))
MAX_WORKFLOW_TIMEOUT_SECONDS = int(os.environ.get("MAX_WORKFLOW_TIMEOUT_SECONDS", 60 * 60 * 3))
# How many finished workflows to keep in memory. Each retained workflow pins its
# PaperStore (full paper text/OCR markdown), so an unbounded list is a memory leak
# that OOM-kills the container under sustained load. Completed runs remain
# retrievable from persistence by run_id regardless of this cap.
MAX_COMPLETED_RETAINED = int(os.environ.get("MAX_COMPLETED_RETAINED", 20))

# A PaperFinder search is waited on at two levels, because one shared PaperFinderRequests worker
# serializes searches across all workflows: the client (SemanticScholar) bounds the search itself,
# while the workflow additionally sits through however long it is queued behind other workflows.
# Derived from the client's budget rather than configured separately, so the outer wait can never be
# set below the inner one -- which would make a search time out here while it was still running.
# Stays well under MAX_STEP_TIMEOUT_SECONDS.
_PAPERFINDER_QUEUE_ALLOWANCE_SECONDS = 60 * 20
PAPERFINDER_TOTAL_WAIT_SECONDS = PAPER_FINDER_SEARCH_TIMEOUT_SECONDS + _PAPERFINDER_QUEUE_ALLOWANCE_SECONDS

# Enumeration of steps.
from enum import Enum
class Step(str, Enum):
    INIT = "initialization"
    BUILD_PAPERFINDER_REQUEST = "build-paperfinder-request"
    BUILD_EXTRACTION_SCHEMA = "build-extraction-schema"
    FIND_PAPERS_WITH_PAPERFINDER = "find-papers-with-paperfinder-and-convert"
    EXTRACT_FROM_PAPERS = "extract-from-papers"
    FORM_THEORY_PARAMETRIC = "form-theory-parametric"
    FORM_THEORY_FROM_LITERATURE = "form-theory-from-literature"
    SAVE_THEORY_TO_DISK = "save-theory-to-disk"
    QUALIFIED_NOVELTY_EVALUATION = "qualified-novelty-evaluation"
    COMPLETE = "complete"

    ERROR = "error"
    UNKNOWN = "unknown-step" # For testing if the workflow dispatcher correctly handles unknown steps.

LIST_OF_END_STEPS = [Step.COMPLETE, Step.ERROR]

#
#   Refactor: Take it out of the 'Workflow' classes, and make it a datastructure, that passes between different threads for the different stages (e.g. schema generation, paper finding, paper extraction, theory generation, etc.)
#

# Make a data structure that keeps track of the theory generation process.
class TheoryGenerationWorkflowStructure():
    def __init__(self,
        generation_id:str,
        user_id:str="api",
        request_timestamp:str="",
        theory_query:str="",
        mission_statement:str="",
        max_papers_to_retrieve:int=10,
        knowledge_cutoff_year:int=2025,
        knowledge_cutoff_month:int=1,
        model_str_primary:str=DEFAULT_MODEL_PRIMARY,
        model_str_extraction:str=DEFAULT_MODEL_EXTRACTION,
        generation_objective:str=GENERATION_OBJECTIVE_ACCURACY_FOCUS,
        generation_input:str=GENERATION_INPUT_LITERATURE_SUPPORT,
        base_id_level:int=0,
        max_data_tokens:int=100000,
        do_qualified_novelty_evaluation:bool=False,
    ):

        self.on_step_change = None
        self.on_paper_retrieved = None

        # Each workflow has it's own TheoryStore
        self.paper_store = PaperStore(filename_in=None, generation_id=generation_id) # TODO: Should share a reference to a global paperstore, instead of each workflow having its own paperstore instance.
        self.theory_store = TheoryStore(paperstore=self.paper_store, base_id_level=base_id_level)

        # Background worker-threaded services this workflow owns (beyond its PaperStore),
        # e.g. per-extraction SchemaExtractionQueues. cleanup() stops them all so the
        # workflow can be garbage-collected once retired instead of leaking threads.
        self._extra_services = []

        # Store original parameters
        self.theory_workflow_structure = {
            "generation_id": generation_id,
            "user_id": user_id,
            "request_timestamp": request_timestamp,
            "theory_query": theory_query,
            "mission_statement": mission_statement,
            "max_papers_to_retrieve": max_papers_to_retrieve,
            "knowledge_cutoff_year": knowledge_cutoff_year,
            "knowledge_cutoff_month": knowledge_cutoff_month,
            "model_str_primary": model_str_primary,
            "model_str_extraction": model_str_extraction,
            "generation_objective": generation_objective,
            "generation_input": generation_input,
            "base_id_level": base_id_level,
            "max_data_tokens": max_data_tokens,
            "do_qualified_novelty_evaluation": do_qualified_novelty_evaluation,
        }

        # Other fields that will be populated during the workflow, as different steps are completed:
        self.theory_workflow_structure["save_time"] = None
        self.theory_workflow_structure["total_cost"] = None
        self.theory_workflow_structure["total_time"] = None
        self.theory_workflow_structure["current_step"] = None
        self.theory_workflow_structure["was_successful"] = None
        self.theory_workflow_structure["is_finished"] = None

        # Keep track of any errors
        self.theory_workflow_structure["errors"] = []
        self.theory_workflow_structure["error_code"] = None
        self.theory_workflow_structure["notes"] = []
        self.theory_workflow_structure["costs"] = {}


        # Infer a workflow based on the generation parameters
        if (self.theory_workflow_structure.get("generation_input", None) == GENERATION_INPUT_LITERATURE_SUPPORT):
            # If the generation input is 'literature support', then we need to find papers, convert them, and extract from them before we can form the theory.
            self.theory_workflow_structure["steps"] = [
                Step.INIT,
                Step.BUILD_PAPERFINDER_REQUEST,
                Step.BUILD_EXTRACTION_SCHEMA,
                Step.FIND_PAPERS_WITH_PAPERFINDER,
                Step.EXTRACT_FROM_PAPERS,
                #Step.UNKNOWN,   # DEBUG TEST
                Step.FORM_THEORY_FROM_LITERATURE,
                Step.SAVE_THEORY_TO_DISK,
                Step.QUALIFIED_NOVELTY_EVALUATION,
                Step.COMPLETE
            ]
        elif (self.theory_workflow_structure.get("generation_input", None) == GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY):
            # If the generation input is 'parametric knowledge only', then we can skip the steps related to finding papers, converting them, and extracting from them, and go straight to forming the theory.
            self.theory_workflow_structure["steps"] = [
                Step.INIT,
                Step.FORM_THEORY_PARAMETRIC,
                Step.SAVE_THEORY_TO_DISK,
                Step.COMPLETE
            ]
        else:
            # Unknown workflow
            self.theory_workflow_structure["steps"] = [
                Step.ERROR
            ]
            self.append("ERROR: Unknown generation input type '" + str(self.theory_workflow_structure.get("generation_input", None)) + "'. Can not generate workflow steps.")

        # Set the current step to the first step
        self.set_to_first_step()

        # Intialize a structure to keep track of the time starting/ending each step.
        self.theory_workflow_structure["time_components"] = []
        for step in self.theory_workflow_structure["steps"]:
            self.theory_workflow_structure["time_components"].append({
                "step": step,
                "start_time": None,
                "end_time": None,
                "total_time_secs": None
            })

    def register_background_service(self, service):
        """Track a background service (anything with a stop_thread()) so cleanup()
        can stop its worker thread when the workflow is retired."""
        self._extra_services.append(service)

    def cleanup(self):
        """Stop every background worker thread this workflow owns (its PaperStore
        plus any registered services), so the workflow — and the paper full-text it
        pins — can be garbage-collected once finished. Idempotent and best-effort:
        a stop failure is logged, never raised, so retiring one workflow can't wedge
        the worker loop."""
        for svc in [self.paper_store, *self._extra_services]:
            try:
                svc.stop_thread()
            except Exception as e:
                print(f"TheoryGenerationWorkflowStructure.cleanup(): error stopping {type(svc).__name__}: {e!r}")
        self._extra_services = []

    # --- Stable seams for the A2A layer ----------------------------------
    # These let a caller scope a run and supply bring-your-own inputs WITHOUT
    # reaching into theory_workflow_structure's private keys. The field names a
    # skipped pipeline step would have produced stay private to the core, here,
    # next to the executors that read them.

    def set_steps(self, steps):
        """Scope the workflow to an explicit ordered step list, resetting per-step
        timing and positioning at the first step. Lets a caller run a subset of the
        pipeline without hand-rebuilding the parallel time_components list."""
        self.theory_workflow_structure["steps"] = list(steps)
        self.theory_workflow_structure["time_components"] = [
            {"step": s, "start_time": None, "end_time": None, "total_time_secs": None}
            for s in self.theory_workflow_structure["steps"]
        ]
        self.set_to_first_step()

    def seed_extraction_schema(self, schema, schema_id):
        """Record a pre-built extraction schema as if BUILD_PAPERFINDER_REQUEST produced
        it, so EXTRACT/FORM_THEORY can run with the build step skipped."""
        wf = self.theory_workflow_structure
        theory_query = wf.get("theory_query", "")
        wf["theory_request_normalized"] = wf.get("theory_request_normalized") or theory_query
        wf["paper_search_query"] = wf.get("paper_search_query") or theory_query
        wf["extraction_query"] = schema.extraction_query
        wf["extraction_schema"] = schema
        wf["extraction_schema_raw"] = schema.schema
        wf["extraction_schema_id"] = schema_id

    def seed_completed_extractions(self):
        """Record the theory store's extraction_results as completed, as if
        EXTRACT_FROM_PAPERS ran, so FORM_THEORY reads them instead of re-extracting."""
        completed_ids, completed_results = [], []
        for rid, result in self.theory_store.extraction_results.items():
            completed_ids.append(rid)
            data = result.extracted_data
            if isinstance(data, list):
                completed_results.extend(data)
            else:
                completed_results.append(data)
        self.theory_workflow_structure["completed_extraction_result_ids"] = completed_ids
        self.theory_workflow_structure["completed_extraction_results"] = completed_results

    def set_observer(self, observer):
        """Wire one observer to every progress callback (some live on the workflow,
        some on its theory_store) so the A2A layer subscribes via a single typed seam
        instead of monkey-patching six attributes across two objects. Each hook is
        guarded so an observer exception can't escape into — and kill — the worker thread."""
        def guarded(fn):
            def wrapped(*args):
                try:
                    return fn(*args)
                except Exception:
                    print(f"observer callback {getattr(fn, '__name__', fn)} failed:\n{traceback.format_exc()}")
            return wrapped
        self.on_step_change = guarded(observer.on_step_change)
        self.on_paper_retrieved = guarded(observer.on_paper_retrieved)
        self.on_novelty_statement_evaluated = guarded(observer.on_novelty_statement_evaluated)
        self.theory_store.on_theory_added = guarded(observer.on_theory_added)
        self.theory_store.on_extraction_schema_added = guarded(observer.on_extraction_schema_added)
        self.theory_store.on_extraction_result_added = guarded(observer.on_extraction_result_added)

    @staticmethod
    def from_persistence(generation_id: str):
        """Reconstruct a workflow from its persisted state, theory store, and paper store."""
        store = Persistence.store()
        state = store.get_workflow(generation_id)
        if state is None:
            raise ValueError(f"No workflow found for: {generation_id}")

        wf = TheoryGenerationWorkflowStructure(generation_id=state["generation_id"])
        wf.theory_workflow_structure = dict(state)
        wf.theory_workflow_structure.setdefault("error_code", None)

        paper_store = store.get_paperstore(generation_id)
        if paper_store is not None:
            # The PaperStore that __init__ just spun up (with its monitor thread) is about
            # to be replaced; stop it first so its thread doesn't leak on every resume.
            try:
                wf.paper_store.stop_thread()
            except Exception as e:
                print(f"from_persistence(): error stopping placeholder PaperStore: {e!r}")
            wf.paper_store = paper_store
            wf.theory_store.paperstore = paper_store

        theory_store_dict = store.get_theorystore(generation_id)
        if theory_store_dict is not None:
            wf.theory_store.load_from_dict(theory_store_dict)

        # Step enums get serialized as strings — convert back.
        wf.theory_workflow_structure["steps"] = [Step(v) for v in wf.theory_workflow_structure.get("steps", [])]
        return wf

    # Accesors/setters/etc.
    def set_to_first_step(self):
        self.theory_workflow_structure["error_code"] = None
        self.theory_workflow_structure["current_step_idx"] = 0
        self.theory_workflow_structure["is_current_step_completed"] = False
        # Don't set a start time for the initial step here, since it's just a 'waiting for its turn to start working' step.

    # Set whether the current step is completed or not.
    # This is normally called by a worker thread, to signal to the workflow manager that this step is completed, and the workflow needs to move on to the next step.
    def set_current_step_completed(self):
        self.theory_workflow_structure["is_current_step_completed"] = True
        # Set the end time
        current_step_idx = self.theory_workflow_structure.get("current_step_idx", None)
        if (current_step_idx is not None):
            time_components = self.theory_workflow_structure.get("time_components", [])
            if (current_step_idx < len(time_components)):
                current_time_component = time_components[current_step_idx]
                current_time_component["end_time"] = time.time()
                if (current_time_component["start_time"] is not None):
                    current_time_component["total_time_secs"] = current_time_component["end_time"] - current_time_component["start_time"]

        # Terminal steps don't go through increment_step, so fire the callback here.
        if self.on_step_change and self.is_at_terminal_step():
            self.on_step_change(self)

    # Check whether the current step is marked as completed
    def is_current_step_completed(self):
        return self.theory_workflow_structure.get("is_current_step_completed", False)

    # Get the current step name
    def get_current_step_name(self):
        current_step_idx = self.theory_workflow_structure.get("current_step_idx", None)
        steps = self.theory_workflow_structure.get("steps", [])
        if (current_step_idx is None) or (current_step_idx >= len(steps)):
            return None
        return steps[current_step_idx]

    # Increment to the next step.
    def increment_step(self):
        current_step_idx = self.theory_workflow_structure.get("current_step_idx", None)
        if (current_step_idx is None):
            self.theory_workflow_structure.errors.append("ERROR: Current step index is not set. Can not increment step.")
            return

        steps = self.theory_workflow_structure.get("steps", [])
        if (current_step_idx >= len(steps) - 1):
            return

        # Increment the current step index
        self.theory_workflow_structure["current_step_idx"] = current_step_idx + 1

        # Mark the new current step as not completed
        self.theory_workflow_structure["is_current_step_completed"] = False

        # Set the start time for the new current step
        new_current_step_idx = self.theory_workflow_structure["current_step_idx"]
        time_components = self.theory_workflow_structure.get("time_components", [])
        if (new_current_step_idx < len(time_components)):
            new_current_time_component = time_components[new_current_step_idx]
            new_current_time_component["start_time"] = time.time()

        if self.on_step_change:
            self.on_step_change(self)


    # Get the current time spent on this step
    def get_time_on_this_step(self):
        current_step_idx = self.theory_workflow_structure.get("current_step_idx", None)
        time_components = self.theory_workflow_structure.get("time_components", [])
        if (current_step_idx is None) or (current_step_idx >= len(time_components)):
            return None
        current_time_component = time_components[current_step_idx]
        if (current_time_component["start_time"] is None):
            return 0.0
        elif (current_time_component["end_time"] is None):
            return round((time.time() - current_time_component["start_time"]), 2)
        else:
            return round((current_time_component["total_time_secs"]), 2)

    def get_total_time(self):
        total_time = 0.0
        time_components = self.theory_workflow_structure.get("time_components", [])
        for time_component in time_components:
            if (time_component["total_time_secs"] is not None):
                total_time += time_component["total_time_secs"]

        # If the current step is not completed, then also add the time spent on the current step so far.
        if (not self.is_current_step_completed()):
            current_time_on_step = self.get_time_on_this_step()
            if (current_time_on_step is not None):
                total_time += current_time_on_step

        return round(total_time, 2)

    # This returns true if the current step is one of the 'end steps' (i.e. compelted, error, etc.), which means the workflow is effectively done.
    def is_at_terminal_step(self):
        current_step_name = self.get_current_step_name()
        if (current_step_name in LIST_OF_END_STEPS):
            return True
        return False

    # Set into an error state.
    def set_error_state(self, error_message:str, error_code:Optional[str]=None):
        """The code describes the latest error, which A2A reports as the failure reason.
        An uncoded error clears the prior code so it cannot classify a different failure.
        """
        self.theory_workflow_structure["errors"].append(error_message)
        self.theory_workflow_structure["error_code"] = error_code

        # Set the end time on the current step.
        current_step_idx = self.theory_workflow_structure.get("current_step_idx", None)
        time_components = self.theory_workflow_structure.get("time_components", [])
        if (current_step_idx is not None) and (current_step_idx < len(time_components)):
            current_time_component = time_components[current_step_idx]
            current_time_component["end_time"] = time.time()
            if (current_time_component["start_time"] is not None):
                current_time_component["total_time_secs"] = current_time_component["end_time"] - current_time_component["start_time"]

        # Add a final state to the list of steps, called Step.ERROR.
        self.theory_workflow_structure["steps"].append(Step.ERROR)
        # Set the current step to this new error step.
        self.theory_workflow_structure["current_step_idx"] = len(self.theory_workflow_structure["steps"]) - 1
        # Set the current state to be completed, since it's a terminal.
        self.set_current_step_completed()

    # Gracefully finish in a SUCCESS terminal state, skipping any remaining steps. The
    # symmetric counterpart to set_error_state(), for legitimate empty results (e.g. no
    # full-text papers) where nothing went wrong but there's no more work to do. An optional
    # `note` is stored as the user-facing completion message.
    def set_completed_state(self, note: str = ""):
        if note:
            self.theory_workflow_structure["completion_note"] = note

        # Close out the current step's timing.
        current_step_idx = self.theory_workflow_structure.get("current_step_idx", None)
        time_components = self.theory_workflow_structure.get("time_components", [])
        if (current_step_idx is not None) and (current_step_idx < len(time_components)):
            current_time_component = time_components[current_step_idx]
            current_time_component["end_time"] = time.time()
            if (current_time_component["start_time"] is not None):
                current_time_component["total_time_secs"] = current_time_component["end_time"] - current_time_component["start_time"]

        # Jump straight to a terminal COMPLETE step.
        self.theory_workflow_structure["steps"].append(Step.COMPLETE)
        self.theory_workflow_structure["current_step_idx"] = len(self.theory_workflow_structure["steps"]) - 1
        self.set_current_step_completed()

    # Check if this workflow is in an error state.
    def is_in_error_state(self):
        current_step_name = self.get_current_step_name()
        if (current_step_name == Step.ERROR):
            return True
        return False



    #
    #   Costs
    #
    def add_cost_to_key(self, key:str, cost:float):
        if "costs" not in self.theory_workflow_structure:
            self.theory_workflow_structure["costs"] = {}
        self.theory_workflow_structure["costs"][key] = self.theory_workflow_structure["costs"].get(key, 0.0) + cost

    def get_total_cost(self):
        total_cost = 0.0
        costs = self.theory_workflow_structure.get("costs", {})
        for key, cost in costs.items():
            total_cost += cost
        return total_cost


    #
    #   Saving the workflow (for record keeping/debugging purposes)
    #
    def save_workflow_to_file(self):
        try:
            generation_id_sanitized = str(self.theory_workflow_structure.get("generation_id", "unknown")).strip().replace(" ", "_")

            # Add a 'save time' field to the workflow structure, for record keeping.
            self.theory_workflow_structure["save_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Add a 'total_cost' and 'total_time' field to the workflow structure, for record keeping.
            self.theory_workflow_structure["total_cost"] = self.get_total_cost()
            self.theory_workflow_structure["total_time"] = self.get_total_time()

            # Add 'current_step' and 'is_complete' fields
            self.theory_workflow_structure["current_step"] = self.get_current_step_name()
            self.theory_workflow_structure["was_successful"] = (self.get_current_step_name() == Step.COMPLETE)
            self.theory_workflow_structure["is_finished"] = self.is_at_terminal_step() or self.is_in_error_state()

            # Clone it, since we're going to remove a few large fields
            deep_clone = json.loads(json.dumps(self.theory_workflow_structure, default=str))
            if ("papers_from_paperfinder_unfiltered" in deep_clone):
                deep_clone["papers_from_paperfinder_unfiltered"] = "Removed for size limits."
            if ("papers_from_paperfinder" in deep_clone):
                deep_clone["papers_from_paperfinder"] = "Removed for size limits."

            Persistence.store().save_workflow(generation_id_sanitized, deep_clone)

        except Exception as e:
            print(f"save_workflow_to_file(): Error saving workflow to file: {e!r}\n{traceback.format_exc()}")

    #
    #   String methods
    #
    def add_note(self, note:str):
        if ("notes" not in self.theory_workflow_structure):
            self.theory_workflow_structure["notes"] = []
        self.theory_workflow_structure["notes"].append(note)

    def get_notes_debug_str(self):
        outStr = ("-"*50) + "\n"
        outStr += "Workflow ID: " + str(self.theory_workflow_structure.get("generation_id", "N/A")) + ", User ID: " + str(self.theory_workflow_structure.get("user_id", "N/A")) + "\n"
        outStr += "\tQuery: '" + str(self.theory_workflow_structure.get("theory_query", "N/A")) + "'\n"
        current_step_name = self.get_current_step_name()
        outStr += "\tCurrent step: '" + str(current_step_name) + "'.\n"
        outStr += "Notes:\n"
        for idx, note in enumerate(self.theory_workflow_structure.get("notes", [])):
            outStr += f"\t{idx+1}. " + str(note) + "\n"
        outStr += "Errors:\n"
        if (len(self.theory_workflow_structure.get("errors", [])) == 0):
            outStr += "\tNone.\n"
        else:
            for idx, error in enumerate(self.theory_workflow_structure.get("errors", [])):
                outStr += f"\t{idx+1}. " + str(error) + "\n"

        outStr += ("-"*50)

        return outStr


#
#   Scaffold for a StepProcessor class, that essentially keeps a queue of work that it's working on, and processes it in the background (in a threaded way).
#
class StepProcessor:
    """
    Generic base class for a step processor that:
    - maintains an unstarted queue
    - runs up to max_simultaneous_work concurrently
    - tracks inflight futures and marks workflows completed on success
    """

    def __init__(self, step_name: str, max_simultaneous_work: int = 5):
        self.step_name = step_name
        self.max_simultaneous_work = max_simultaneous_work

        # Thread-safe state guarded by self._cv
        self._cv = threading.Condition()
        self._unstarted: Deque[Any] = deque()
        self._inflight: Dict[Future, Any] = {}

        self._stop = False
        self._worker_thread: Optional[threading.Thread] = None

        # Executor runs the work
        self._executor = ThreadPoolExecutor(max_workers=self.max_simultaneous_work)

    # ---- public API ----
    # Add work to the queue
    def add_workflow_to_queue(self, workflow: Any) -> None:
        with self._cv:
            if self._stop:
                return
            self._unstarted.append(workflow)
            self._cv.notify()

    # Start the worker thread that schedules work.
    def start_thread(self) -> None:
        with self._cv:
            if self._worker_thread is not None:
                return
            self._stop = False
            self._worker_thread = threading.Thread(
                target=self._process_work_monitor,
                name=f"StepProcessor[{self.step_name}]",
                daemon=True,
            )
            self._worker_thread.start()

    # Stop the worker thread and shut down the executor.
    def stop_thread(self, wait: bool = True) -> None:
        print(f"StepProcessor.stop_thread(): Stopping worker thread for step '{self.step_name}'...")
        with self._cv:
            self._stop = True
            self._cv.notify_all()

        # Stop accepting new work; cancel queued (not-started) futures where possible
        self._executor.shutdown(wait=False, cancel_futures=True)

        if wait and self._worker_thread is not None:
            self._worker_thread.join()

        self._worker_thread = None
        print(f"StepProcessor.stop_thread(): Worker thread for step '{self.step_name}' stopped.")

    # ---- subclass hooks ----
    def submit_callable(self, workflow: Any) -> Future:
        """
        Override in subclass to submit the correct external function.
        Must return a Future.
        """
        raise NotImplementedError

    def set_workflow_step_completed(self, workflow: Any) -> None:
        """
        Default: calls workflow.set_current_step_completed().
        Override if your workflow tracks steps differently.
        """
        with getattr(workflow, "lock", _NullLock()):
            workflow.set_current_step_completed()

    def on_error(self, workflow: Any, exc: BaseException) -> None:
        """
        Default error handling: store error and print.
        """
        with getattr(workflow, "lock", _NullLock()):
            if hasattr(workflow, "step_errors") and isinstance(workflow.step_errors, dict):
                workflow.step_errors[self.step_name] = repr(exc)
        print(f"[{self.step_name}] ERROR for {workflow}: {exc!r}")

    # ---- internal ----
    def _process_work_monitor(self) -> None:
        """
        Monitor thread:
        - schedules up to max_simultaneous_work
        - polls for completions
        - marks workflows complete on success
        """
        while True:
            # 1) Launch work up to capacity
            with self._cv:
                if self._stop:
                    return

                while (len(self._inflight) < self.max_simultaneous_work) and self._unstarted:
                    workflow = self._unstarted.popleft()
                    fut = self.submit_callable(workflow)
                    self._inflight[fut] = workflow

                # Nothing queued or running: wait for new work or stop
                if not self._inflight and not self._unstarted:
                    self._cv.wait(timeout=1.0)
                    continue

                inflight_futures = list(self._inflight.keys())

            # 2) Poll for completions
            completed = [f for f in inflight_futures if f.done()]
            if not completed:
                time.sleep(0.1)
                continue

            # 3) Harvest completions
            for fut in completed:
                with self._cv:
                    workflow = self._inflight.pop(fut, None)

                if workflow is None:
                    continue

                try:
                    fut.result()  # raises if worker failed
                except BaseException as e:
                    self.on_error(workflow, e)
                    continue

                # NOTE: PJ: Commented out
                # # Worker mutated workflow; now mark step complete
                # try:
                #     self.set_workflow_step_completed(workflow)
                # except BaseException as e:
                #     self.on_error(workflow, e)

                # Wake scheduler in case capacity freed
                with self._cv:
                    self._cv.notify()


class _NullLock:
    """Fallback context manager if workflow.lock doesn't exist."""
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False


#
#   Step Processor: Build PaperFinder Request
#

def build_paperfinder_request_executor(workflow: TheoryGenerationWorkflowStructure) -> None:
    try:
        workflow.add_note("I'm working on normalizing the theory query and building an extraction schema...")

        # Step 1: Perform the work (i.e. run the theory request prompt)
        result = convert_theory_request_to_query_and_schema(
            theory_request=workflow.theory_workflow_structure.get("theory_query", ""),
            model_str=workflow.theory_workflow_structure.get("model_str_primary", None),
            max_tokens=16000,
            temperature=0.0,
            use_reflection=False
        )

        # Step 2: Parse the output, and save it back into the workflow structure for the next steps to use.

        # Cost
        workflow.add_cost_to_key("convert_theory_request_to_query_and_schema", result.get("total_cost", 0.0))

        # Get the individual components from the result
        conversion_result = result.get("output", None)
        if (conversion_result is None):
            workflow.set_error_state("ERROR in `build-paperfinder-request`: The initial step to convert the theory request to a query and schema failed.")
            return

        workflow.theory_workflow_structure["theory_request_normalized"] = conversion_result.get("theory_request_normalized", None)
        workflow.theory_workflow_structure["paper_search_query"] = conversion_result.get("paper_search_query", None)
        workflow.theory_workflow_structure["short_name"] = conversion_result.get("short_name", None)
        workflow.theory_workflow_structure["extraction_query"] = conversion_result.get("extraction_query", None)
        workflow.theory_workflow_structure["extraction_schema_raw"] = conversion_result.get("extraction_schema", None)

        #print(json.dumps(conversion_result, indent=4))

        # If the query or the schema are empty, throw an error.
        if (not isinstance(workflow.theory_workflow_structure.get("paper_search_query", None), str)):
            workflow.set_error_state("ERROR in `build-paperfinder-request`: The paper search query generated from the theory request is empty or not the expected type (" + str(type(workflow.theory_workflow_structure.get("paper_search_query", None))) + ").")
            return
        if (not isinstance(workflow.theory_workflow_structure.get("extraction_query", None), str)):
            workflow.set_error_state("ERROR in `build-paperfinder-request`: The extraction query generated from the theory request is empty or not the expected type (" + str(type(workflow.theory_workflow_structure.get("extraction_query", None))) + ").")
            return
        if (not isinstance(workflow.theory_workflow_structure.get("extraction_schema_raw", None), list)):
            workflow.set_error_state("ERROR in `build-paperfinder-request`: The extraction schema generated from the theory request is empty or not the expected type (" + str(type(workflow.theory_workflow_structure.get("extraction_schema_raw", None))) + ").")
            return

        # Save the extraction schema (combined with the extraction query) in its storage class
        workflow.theory_workflow_structure["extraction_schema"] = ExtractionQuerySchema(
            schema=workflow.theory_workflow_structure.get("extraction_schema_raw", None),
            extraction_query=workflow.theory_workflow_structure.get("extraction_query", None),
            supporting_theory_ids=[],    # Populate after the theories are generated
            model_str=workflow.theory_workflow_structure.get("model_str_primary", None)
        )

        # Add the extraction schema to the theory store, and save the ID back into the workflow structure for the next steps to use.
        workflow.theory_workflow_structure["extraction_schema_id"] = workflow.theory_store.add_extraction_schema(schema=workflow.theory_workflow_structure.get("extraction_schema", None))

        #workflow.add_note("build_paperfinder_request_executor(): Completed...")
        schema_raw = workflow.theory_workflow_structure.get("extraction_schema_raw", None)
        num_schema_fields = 0
        try:
            num_schema_fields = len(schema_raw)
        except Exception as e:
            pass
        workflow.add_note("Theory query normalization and extraction schema generated. (Info: The schema has " + str(num_schema_fields) + " fields.)")

        # Mark that the work is done, and the workflow can move on to the next step.
        workflow.set_current_step_completed()

    except Exception as e:
        error_message = f"Exception in build_paperfinder_request_executor: {e!r}\n{traceback.format_exc()}"
        workflow.set_error_state(error_message)
        print(error_message)



class StepProcessorBuildPaperFinderRequest(StepProcessor):
    def __init__(self, max_simultaneous_work: int = 5):
        super().__init__(Step.BUILD_EXTRACTION_SCHEMA.value, max_simultaneous_work=max_simultaneous_work)

    def submit_callable(self, workflow: TheoryGenerationWorkflowStructure) -> Future:
        # Submit the EXTERNAL executor function
        return self._executor.submit(build_paperfinder_request_executor, workflow)



#
#   Step Processor: Make PaperFinder request
#

def make_paperfinder_request_executor(workflow: TheoryGenerationWorkflowStructure, paperfinder_requests:PaperFinderRequests) -> None:
    try:
        workflow.add_note("I'm checking PaperFinder for relevant papers...")

        # Step 1: Perform the work (i.e. submit the paperfinder request, and monitor for results until they come back, or a timeout is reached)
        # Carry the workflow's generation_id as Paper Finder's cost_trace_id, so its spend attributes
        # to the run that caused it.
        paperfinder_request_id = paperfinder_requests.submit_request(
            workflow.theory_workflow_structure.get("paper_search_query", ""),
            cost_trace_id=workflow.theory_workflow_structure.get("generation_id"),
        )

        # Step 2: Wait for the results to come back: the search itself, plus time queued behind
        # other workflows. See PAPERFINDER_TOTAL_WAIT_SECONDS.
        timeout_seconds = PAPERFINDER_TOTAL_WAIT_SECONDS
        poll_interval_seconds = 5
        start_time = time.time()
        paperfinder_results = None
        while ((time.time() - start_time) < timeout_seconds):
            paperfinder_results = paperfinder_requests.get_work(paperfinder_request_id)
            if (paperfinder_results is not None):
                break
            time.sleep(poll_interval_seconds)

        if (paperfinder_results is None):
            workflow.set_error_state("ERROR in `make-paperfinder-request`: Timeout reached while waiting for PaperFinder results.")
            return

        # The queue stores a named reason when the PaperFinder call itself failed (rejected request,
        # failed task, timeout). Report it verbatim -- otherwise every failure below reads as
        # "No candidate papers found", which looks like a search that legitimately found nothing.
        if isinstance(paperfinder_results, dict) and paperfinder_results.get("error"):
            workflow.set_error_state("ERROR in `make-paperfinder-request`: " + str(paperfinder_results["error"]))
            return

        # Step 3: Handle the knowledge cutoff by filtering out papers published after the cutoff date.
        # Get the list of papers
        result = paperfinder_results.get("result") if isinstance(paperfinder_results, dict) else None
        doc_collection = result.get("doc_collection") if isinstance(result, dict) else None
        candidate_papers = doc_collection.get("documents") if isinstance(doc_collection, dict) else None
        if not isinstance(candidate_papers, list):
            workflow.set_error_state(
                "ERROR in `make-paperfinder-request`: Invalid response from PaperFinder; "
                "expected result.doc_collection.documents to be a list."
            )
            return

        if (not candidate_papers):
            # Only an explicit empty list confirms that the search matched nothing.
            workflow.set_error_state(
                "ERROR: PaperFinder returned no papers for this query.",
                error_code=ERROR_CODE_PAPERS_NOT_FOUND,
            )
            return

        # Store the unfiltered papers in the workflow structure, for the next steps to use (and for record-keeping).
        workflow.theory_workflow_structure["papers_from_paperfinder_unfiltered"] = candidate_papers

        # Filter to papers whose publication dates are within the evidence window, if the knowledge cutoff date is set (i.e. ints, not None)
        knowledge_cutoff_year = workflow.theory_workflow_structure.get("knowledge_cutoff_year", None)
        knowledge_cutoff_month = workflow.theory_workflow_structure.get("knowledge_cutoff_month", None)
        if (isinstance(knowledge_cutoff_year, int) and isinstance(knowledge_cutoff_month, int)):
            papers_within_cutoff = []
            for paper in candidate_papers:
                publication_date = paper.get("publication_date", None)
                # Handle case for when no publication date is available
                if (publication_date is None):
                    # If the knowledge cutoff date is some far off date in the future (i.e. a future year), then we can include papers with missing publication dates, since they're likely to be old papers that just don't have a recorded publication date in the database. But if the knowledge cutoff date is in the past, then we should exclude papers with missing publication dates, since they might be recent papers that just don't have a recorded publication date yet.
                    current_year = time.localtime().tm_year
                    if (knowledge_cutoff_year > current_year+1):
                        papers_within_cutoff.append(paper)
                    else:
                        continue

                # Parse the publication date (YYYY-MM-DD format)
                try:
                    publication_year, publication_month, _ = publication_date.split("-")
                    publication_year = int(publication_year)
                    publication_month = int(publication_month)
                except Exception as e:
                    continue

                # Check if the publication date is BEFORE the knowledge cutoff
                if (publication_year < knowledge_cutoff_year) or ((publication_year == knowledge_cutoff_year) and (publication_month < knowledge_cutoff_month)):
                    papers_within_cutoff.append(paper)

            workflow.theory_workflow_structure["papers_from_paperfinder"] = papers_within_cutoff
        else:
            # If the knowledge cutoff isn't set, then just use the unfiltered list of papers.
            workflow.theory_workflow_structure["papers_from_paperfinder"] = candidate_papers

        # The list of papers to use is in `papers_from_paperfinder` in the workflow structure, and the original (unfiltered) list is in `papers_from_paperfinder_unfiltered`.
        # Add a workflow note about how many papers were found (before and after filtering).
        num_papers_unfiltered = len(workflow.theory_workflow_structure.get("papers_from_paperfinder_unfiltered", []))
        num_papers_filtered = len(workflow.theory_workflow_structure.get("papers_from_paperfinder", []))
        workflow.add_note(f"PaperFinder returned {num_papers_unfiltered} papers for the query. After applying the knowledge cutoff filter (if applicable), {num_papers_filtered} papers remain.")

        injected_request_ids = list(workflow.theory_workflow_structure.pop("injected_paper_request_ids", []) or [])

        # Now, add the papers to the processing queue.
        if (num_papers_filtered == 0) and (not injected_request_ids):
            workflow.set_error_state("ERROR in `make-paperfinder-request`: No papers found from PaperFinder for the query, after applying the knowledge cutoff filter (if applicable).")
            return

        workflow.add_note("Submitting requests for paper full text...")

        # Note: Only fetch `max_papers_to_retrieve` papers.
        max_papers_to_retrieve = workflow.theory_workflow_structure.get("max_papers_to_retrieve", 10)

        DEBUG_LIMIT_RETRIEVE = None
        #DEBUG_LIMIT_RETRIEVE = 10

        workflow.add_note(f"Note: `max_papers_to_retrieve` is set to: {workflow.theory_workflow_structure.get('max_papers_to_retrieve', None)}. Only the top {workflow.theory_workflow_structure.get('max_papers_to_retrieve', None)} papers will attempt to be fetched.")
        # Submit requests for the full text of the papers to the paper store, and save the returned request IDs in the workflow structure for the next steps to use.
        # Allow callers to inject papers they want included alongside PaperFinder's discoveries.
        paper_store_request_ids = list(injected_request_ids)
        paper_titles_submitted = set()
        for paper in workflow.theory_workflow_structure.get("papers_from_paperfinder", []):
            paper_title = paper.get("title", None)
            # PaperFinder returns corpus_id as int (or absent); downstream code expects str-or-None.
            corpus_id = paper.get("corpus_id")
            paper_corpus_id = str(corpus_id) if corpus_id is not None else None
            paper_title_sanitized = paper_title.strip().lower() if paper_title is not None else None

            if (paper_title is not None) and (paper_corpus_id is not None) and (paper_title_sanitized not in paper_titles_submitted):
                #request_id = workflow.theory_store.paperstore.request_paper_by_corpus_id(paper_corpus_id, paper_title)
                # def submit_paper(self, corpus_id:str=None, s2_sha:str=None, title:str=None):
                request_id = workflow.theory_store.paperstore.submit_paper(corpus_id=paper_corpus_id, title=paper_title)
                paper_store_request_ids.append(request_id)
                paper_titles_submitted.add(paper_title_sanitized)

            # Check if we've reached the `max_papers_to_retrieve` limit, and if so, break out of the loop.
            if (len(paper_store_request_ids) >= max_papers_to_retrieve):
                break
            if (DEBUG_LIMIT_RETRIEVE is not None) and (len(paper_store_request_ids) >= DEBUG_LIMIT_RETRIEVE):
                workflow.add_note(f"DEBUG Mode is enabled (likely to test back-filling): Reached the debug limit for number of papers to retrieve ({DEBUG_LIMIT_RETRIEVE}). Stopping submission of paper retrieval requests.")
                break
        workflow.theory_workflow_structure["paper_titles_submitted"] = list(paper_titles_submitted)
        workflow.theory_workflow_structure["paper_store_request_ids"] = paper_store_request_ids

        workflow.add_note(f"Submitted {len(paper_store_request_ids)} requests for paper full text to the paper store...")

        # Wait for the paper store requests to be fulfilled, with a timeout.
        poll_interval_seconds = 5
        start_time = time.time()
        papers_converted = []
        num_papers_converted_previous = 0
        while ((time.time() - start_time) < PAPER_CONVERSION_TIMEOUT_SECONDS):
            papers_converted = []
            for paper_store_request_id in workflow.theory_workflow_structure.get("paper_store_request_ids", []):
                paper_key = workflow.theory_store.paperstore.is_paper_request_complete(paper_store_request_id)
                if isinstance(paper_key, str):  # PaperStore returns error dicts on failure; ignore those.
                    papers_converted.append(paper_key)
                    if workflow.on_paper_retrieved:
                        workflow.on_paper_retrieved(workflow, paper_key)

            # Check to see if we've made progress since the last poll, and if so, add a note to the workflow about how many papers have been converted so far.
            num_papers_converted = len(papers_converted)
            if (num_papers_converted > num_papers_converted_previous):
                time_elapsed_seconds = round((time.time() - start_time), 1)
                workflow.add_note(f"{num_papers_converted} out of {len(workflow.theory_workflow_structure.get('paper_store_request_ids', []))} papers have processed so far ({time_elapsed_seconds} seconds elapsed).")
                num_papers_converted_previous = num_papers_converted

            if (len(papers_converted) == len(workflow.theory_workflow_structure.get("paper_store_request_ids", []))):
                workflow.add_note(f"All papers have been processsed by the full-text extraction pipeline! ({len(papers_converted)} papers, {round((time.time() - start_time), 1)} seconds elapsed)")
                break

            time.sleep(poll_interval_seconds)

        # Append rather than overwrite: BYO papers may already be in paper_keys.
        existing = workflow.theory_workflow_structure.get("paper_keys", [])
        workflow.theory_workflow_structure["paper_keys"] = list(dict.fromkeys(existing + papers_converted))

        # Cancel any remaining queued requests so they don't block follow-on papers
        workflow.theory_store.paperstore.cancel_incomplete(paper_store_request_ids)

        # As long as at least one paper has been converted, we can proceed to the next step. If no papers were converted, then we should set an error state.
        # Check for 3 cases: Full list converted, partial list converted, no papers converted.
        if (len(papers_converted) == len(workflow.theory_workflow_structure.get("paper_store_request_ids", []))):
            workflow.add_note("All papers processed by the full-text extraction pipeline within the timeout period.")
        elif (len(papers_converted) > 0):
            workflow.add_note(f"Only {len(papers_converted)} out of {len(workflow.theory_workflow_structure.get('paper_store_request_ids', []))} papers were processed through the full-text extraction pipeline within the timeout period. Proceeding with the converted papers.")
            # Update the workflow structure to only include the converted papers for the next steps to use.
        else:
            workflow.set_error_state("ERROR in `make-paperfinder-request`: No papers were converted within the timeout period.")
            return

        # Check how many of the papers were actually able to retreive full-text (`PaperStatus.FULL_TEXT.value`)
        num_papers_with_full_text, valid_ids = workflow.theory_store.paperstore.get_num_papers_with_full_text(paper_keys=workflow.theory_workflow_structure.get("paper_keys", []))
        workflow.add_note(f"Out of the {len(workflow.theory_workflow_structure.get('paper_keys', []))} papers that were processed through the full-text extraction pipeline, {num_papers_with_full_text} have full text available.")
        # Replace the paper keys with `valid_ids`
        workflow.theory_workflow_structure["paper_keys"] = valid_ids

        # Set this step as completed, so the workflow can move on to the next step.
        workflow.set_current_step_completed()



    except Exception as e:
        error_message = f"Exception in make_paperfinder_request_executor: {e!r}\n{traceback.format_exc()}"
        workflow.set_error_state(error_message)
        print(error_message)


class StepProcessorMakePaperFinderRequest(StepProcessor):
    def __init__(self, max_simultaneous_work: int = 5):
        super().__init__(Step.FIND_PAPERS_WITH_PAPERFINDER.value, max_simultaneous_work=max_simultaneous_work)
        # Store an instance of PaperFinderRequests to use for making the requests.
        self.paperfinder_requests = PaperFinderRequests()


    def submit_callable(self, workflow: TheoryGenerationWorkflowStructure) -> Future:
        # Submit the EXTERNAL executor function
        return self._executor.submit(make_paperfinder_request_executor, workflow, self.paperfinder_requests)


#
#   Step Processor: Extract from Papers
#

def extract_from_papers_executor(workflow: TheoryGenerationWorkflowStructure) -> None:
    # Make a fresh instance of the paper extraction queue.
    paper_store = workflow.paper_store
    theory_store = workflow.theory_store
    workflow.add_note("I'm initializing the SchemaExtractionQueue...")
    try:
        schema_extraction_queue = SchemaExtractionQueue(theorystore=theory_store, paperstore=paper_store)
        # Hand ownership to the workflow so its monitor thread is stopped when the
        # workflow retires, instead of leaking one thread per extraction step.
        workflow.register_background_service(schema_extraction_queue)
    except Exception as e:
        error_message = f"Exception while initializing SchemaExtractionQueue: {e!r}\n{traceback.format_exc()}"
        workflow.set_error_state(error_message)
        print(error_message)
        return

    # Step 1: Submit the extraction requests for the papers, using the extraction schema that was generated in the first step, and save the request IDs in the workflow structure for the next steps to use.
    workflow.add_note("I'm extracting information from the papers using the extraction schema I generated earlier...")
    workflow.theory_workflow_structure["extraction_request_ids"] = []
    extraction_schema_id = workflow.theory_workflow_structure.get("extraction_schema_id", None)
    if (extraction_schema_id is None):
        workflow.set_error_state("ERROR in `extract-from-papers`: Extraction schema ID is not set in the workflow structure.")
        return

    # Submit the extraction requests on a batch of papers
    extraction_request_ids = schema_extraction_queue.submit_extraction_request(extraction_schema_id=extraction_schema_id, paper_ids_to_extract_from=workflow.theory_workflow_structure.get("paper_keys"), extraction_model_str=workflow.theory_workflow_structure.get("model_str_extraction"))
    workflow.theory_workflow_structure["extraction_request_ids"] = extraction_request_ids

    workflow.add_note(f"Submitted {len(workflow.theory_workflow_structure.get('extraction_request_ids', []))} extraction requests for the papers...")

    # Zero requests means no papers had accessible full text to extract from (PaperFinder may
    # have found papers, but none converted — common for recent/paywalled literature). That's a
    # legitimate empty result, not a failure: complete gracefully with an explanatory message
    # instead of erroring with a misleading "timeout".
    if (len(extraction_request_ids) == 0):
        num_found = len(workflow.theory_workflow_structure.get("papers_from_paperfinder", []))
        if num_found > 0:
            note = (f"Found {num_found} paper{'s' if num_found != 1 else ''} for this query, but none had "
                    "accessible full text to extract from, so no theories were generated. This often "
                    "happens for very recent or paywalled literature.")
        else:
            note = ("No papers with accessible full text were available to extract from, so no theories "
                    "were generated.")
        workflow.add_note(note)
        workflow.set_completed_state(note)
        return

    # Step 2: Wait for the extraction requests to be completed, and save the extraction results in the workflow structure for the next steps to use.
    # 20-minute floor, 5s per paper for larger batches.
    timeout_seconds = max(60 * 20, len(workflow.theory_workflow_structure.get("extraction_request_ids", [])) * 5)
    poll_interval_seconds = 5
    start_time = time.time()
    completed_extraction_result_ids = []
    completed_extraction_results = []
    num_extraction_results_completed_previous = 0
    # Whether the loop below exhausted its time budget without every request finishing. Tracked
    # explicitly (flipped at the same point we `break`) so downstream code can ask "did this
    # genuinely time out?" directly, instead of inferring it after the fact from the *content* of
    # the extraction results -- which conflates two different questions (did the requests finish
    # vs. did they find anything) and was the root cause of a prior bug (see nora-issues#3150).
    timed_out = True
    while ((time.time() - start_time) < timeout_seconds):
        completed_extraction_result_ids = []
        for extraction_request_id in workflow.theory_workflow_structure.get("extraction_request_ids", []):
            extraction_result_id = schema_extraction_queue.is_request_complete(request_id=extraction_request_id)
            if (extraction_result_id is not None):
                completed_extraction_result_ids.append(extraction_result_id)

        # Check to see if we've made progress since the last poll, and if so, add a note to the workflow about how many extraction results have been completed so far.
        num_extraction_results_completed = len(completed_extraction_result_ids)
        print("Number of completed extraction results: " + str(num_extraction_results_completed) + " out of " + str(len(workflow.theory_workflow_structure.get("extraction_request_ids", []))))

        if (num_extraction_results_completed > num_extraction_results_completed_previous):
            time_elapsed_seconds = round((time.time() - start_time), 1)
            workflow.add_note(f"{num_extraction_results_completed} out of {len(workflow.theory_workflow_structure.get('extraction_request_ids', []))} extraction requests have been completed so far ({time_elapsed_seconds} seconds elapsed).")
            num_extraction_results_completed_previous = num_extraction_results_completed

        if (len(completed_extraction_result_ids) == len(workflow.theory_workflow_structure.get("extraction_request_ids", []))):
            workflow.add_note(f"All extraction requests have been completed! ({len(completed_extraction_result_ids)} requests, {round((time.time() - start_time), 1)} seconds elapsed)")
            timed_out = False
            break

        time.sleep(poll_interval_seconds)

    # Now, try to extract the actual extracted data from the completed extraction results, and save it in the workflow structure for the next steps to use.
    for extraction_result_id in completed_extraction_result_ids:
        try:
            extraction_result = theory_store.get_extraction_result(id=extraction_result_id)     # This returns a ExtractionQueryResult -- need to access the `extracted_data` field to get the actual extracted data.
            if (extraction_result is None):
                workflow.add_note(f"Warning: Extraction result with ID {extraction_result_id} is None. Skipping.")
                continue
            actual_extraction_results = extraction_result.extracted_data
            if (isinstance(actual_extraction_results, list)):
                # If the extracted data is a list, we can assume it's a list of individual extraction results, and we can add them to the completed_extraction_results list.
                completed_extraction_results.extend(actual_extraction_results)
            else:
                completed_extraction_results.append(actual_extraction_results)
        except Exception as e:
            import traceback
            print(f"Error while accessing extracted_data from extraction result with ID {extraction_result_id}: {e!r}\n{traceback.format_exc()}")
            workflow.add_note("Warning: Failed to access extracted data from extraction result with ID " + str(extraction_result_id) + ". Error: " + repr(e) + "\n" + traceback.format_exc())
            continue

    # Save the extraction results to a debug file
    filename_debug_out = os.path.join(DEBUG_OUTPUT_DIR, "completed-extraction-results-" + time.strftime("%Y%m%d-%H%M%S") + ".json")
    try:
        with open(filename_debug_out, "w") as f:
            json.dump(completed_extraction_results, f, indent=4)
        workflow.add_note(f"Exported the completed extraction results to a JSON file for debugging: {filename_debug_out}")
    except Exception as e:
        workflow.add_note(f"Warning: Failed to export the completed extraction results to a JSON file for debugging. Error: {e!r}")

    if (len(completed_extraction_results) == 0):
        # Reachable only when requests WERE submitted (the no-full-text case returned above).
        # This can happen for two very different reasons, which we need to tell apart:
        #   1. A genuine timeout: not every request came back within the time budget.
        #   2. Every request came back, but each paper's extraction yielded an empty `results`
        #      list -- e.g. the LLM judged none of the papers relevant to the query, per the
        #      off-ramp instruction in the extraction prompt. That's a legitimate empty outcome,
        #      not a failure, and shouldn't be reported as a timeout.
        n_requests = len(workflow.theory_workflow_structure.get("extraction_request_ids", []))

        if timed_out:
            minutes = timeout_seconds // 60
            workflow.set_error_state(
                f"ERROR in `extract-from-papers`: Submitted {n_requests} extraction request(s) but "
                f"none completed within {minutes} minutes."
            )
        else:
            note = (f"{n_requests} extraction request{'s' if n_requests != 1 else ''} completed, but no "
                    "relevant information was found in any of the papers, so no theories were generated.")
            workflow.add_note(note)
            workflow.set_completed_state(note)
        return

    # Merge with prior results so resume_extraction grows the set instead of replacing it.
    existing_ids = workflow.theory_workflow_structure.get("completed_extraction_result_ids", [])
    completed_extraction_result_ids = list(dict.fromkeys(existing_ids + completed_extraction_result_ids))
    completed_extraction_results = workflow.theory_workflow_structure.get("completed_extraction_results", []) + completed_extraction_results

    # Save the completed extraction result IDs and results in the workflow structure for the next steps to use.
    workflow.theory_workflow_structure["completed_extraction_result_ids"] = completed_extraction_result_ids
    workflow.theory_workflow_structure["completed_extraction_results"] = completed_extraction_results

    # Count number of blank extraction results (i.e. extraction results where the extracted data is empty or None), and add a note to the workflow about it.
    num_blank_extraction_results = 0
    for extraction_result_id in completed_extraction_result_ids:
        if (isinstance(extraction_result_id, str)):
            if (len(extraction_result_id.strip()) == 0):
                num_blank_extraction_results += 1
                continue
        else:
            num_blank_extraction_results += 1

    # Tell the user how many extraction requests were completed within the timeout period.
    workflow.add_note(f"{len(completed_extraction_result_ids)} out of {len(workflow.theory_workflow_structure.get('extraction_request_ids', []))} extraction requests were completed within the timeout period.")
    workflow.add_note(f"{num_blank_extraction_results} out of {len(completed_extraction_result_ids)} completed extraction results were blank (i.e. had empty or None extracted data), which is likely due to failure to retrieve an open access version of the paper.")
    workflow.add_note(f"{len(completed_extraction_results)} extraction results were obtained from the completed extraction requests.")

    # Remove any blank IDs from the list.
    workflow.theory_workflow_structure["completed_extraction_result_ids"] = [id for id in completed_extraction_result_ids if not (isinstance(id, str) and len(id.strip()) == 0)]

    # TODO: Do additional stages (check to see if enough papers, then backfill, etc.)
    # Step 3: Check to see if we've processed the number of papers we were hoping to process.
    # If not, try to add in more papers to the extraction queue from the extraction results
    num_retrieved_papers = len(completed_extraction_result_ids) - num_blank_extraction_results
    max_papers_to_retrieve = workflow.theory_workflow_structure.get("max_papers_to_retrieve", 10)
    if (num_retrieved_papers < max_papers_to_retrieve):
        workflow.add_note(f"Only {num_retrieved_papers} papers were successfully retrieved and processed, which is less than the `max_papers_to_retrieve` value of {max_papers_to_retrieve}. In the next steps, I will attempt to find additional papers to process by looking at the potentially relevant papers that were identified in the extraction results, and adding them to the extraction queue.")

        # Step 3A: Make a list of potentially relevant papers from the extraction results, and rank them.
        # For each extraction result, get a list of potentially relevant papers
        potentially_relevant_papers = []
        for extraction_result_id in completed_extraction_result_ids:
            # Get the extraction result from the theory store
            extraction_result = workflow.theory_store.get_extraction_result(id=extraction_result_id)

            if (extraction_result is not None):
                # Part 1: Get a list of the potentially relevant papers that were found from reading this paper
                new_papers_found = extraction_result.potentially_relevant_new_papers
                if (new_papers_found is not None) and (isinstance(new_papers_found, list)) and (len(new_papers_found) > 0):
                    # If there are new papers found, add them to the list of potentially relevant papers
                    for paper_record in new_papers_found:
                        # Check that the record has at least two keys: `title` and `rating`
                        if ("paper_title" in paper_record) and ("rating" in paper_record):
                            if (not isinstance(paper_record["paper_title"], str)) or (not isinstance(paper_record["rating"], int)) or (len(paper_record["paper_title"]) < 10):
                                # If the title is not a string or the rating is not an integer, or the title is too short, skip this paper
                                continue
                            # Add the paper record to the list of potentially relevant papers
                            sanitized_title = workflow.theory_store.paperstore._sanitize_paper_title(paper_record["paper_title"])
                            paper_record["sanitized_title"] = sanitized_title
                            potentially_relevant_papers.append(paper_record)

        # Sort the potentially relevant papers by their rating (descending). If no rating, assume -1
        potentially_relevant_papers.sort(key=lambda x: x.get("rating", -1), reverse=True)

        # Only keep papers with ratings >= 1
        filtered_potentially_relevant_papers = [paper for paper in potentially_relevant_papers if paper.get("rating", -1) >= 1]
        # Print the number of potentially relevant papers found
        workflow.add_note(f"Found {len(potentially_relevant_papers)} potentially relevant follow-on papers from the extraction results.")
        workflow.add_note(f"Filtered down to {len(filtered_potentially_relevant_papers)} potentially relevant follow-on papers with rating >= 1.")

        # Get the top-rated N papers to add to the extraction queue (where N is the number of papers we need to reach the maximum)
        num_papers_to_add = max_papers_to_retrieve - num_retrieved_papers
        workflow.add_note(f"Adding up to {num_papers_to_add} of the top-rated potentially relevant follow-on papers to the extraction queue, in order to try to reach the `max_papers_to_retrieve` value of {max_papers_to_retrieve}...")
        num_papers_added = 0

        workflow.theory_workflow_structure["follow_on_paper_ids"] = []
        workflow.theory_workflow_structure["follow_on_paper_request_ids"] = []
        paper_titles_submitted = set(workflow.theory_workflow_structure.get("paper_titles_submitted", []))
        for potentially_relevant_paper in filtered_potentially_relevant_papers:
            if (num_papers_added >= num_papers_to_add):
                break

            paper_title = potentially_relevant_paper.get("paper_title", None)
            paper_title_sanitized = paper_title.strip().lower() if paper_title is not None else None

            if (paper_title is not None) and (paper_title_sanitized not in paper_titles_submitted) and (len(paper_title) >= 10):
                # Add the paper to the extraction queue by submitting a request to the paper store, and then adding the returned request ID to the list of paper store request IDs in the workflow structure for the next steps to use.
                #request_id = workflow.theorizer.paperstore.submit_paper(title=potentially_relevant_paper["paper_title"])
                request_id = workflow.theory_store.paperstore.submit_paper(title=paper_title)
                workflow.theory_workflow_structure["follow_on_paper_request_ids"].append(request_id)
                paper_titles_submitted.add(paper_title_sanitized)
                num_papers_added += 1

        workflow.theory_workflow_structure["paper_titles_submitted"] = list(paper_titles_submitted)

        workflow.add_note(f"Added {num_papers_added} follow-on papers to the extraction queue. Now waiting for these papers to be processed...")

        # Wait for the newly added papers to be added.
        poll_interval_seconds = 5
        start_time = time.time()
        last_num_processed = 0
        follow_on_papers_converted = set()
        while ((time.time() - start_time) < PAPER_CONVERSION_TIMEOUT_SECONDS):
            # Check to see if the newly added papers have been added to the paper store by checking for the presence of their request IDs in the paper store.
            num_papers_added_and_converted = 0
            for request_id in workflow.theory_workflow_structure.get("follow_on_paper_request_ids", []):
                paper_key = workflow.theory_store.paperstore.is_paper_request_complete(request_id)
                if (paper_key is not None):
                    num_papers_added_and_converted += 1
                    follow_on_papers_converted.add(paper_key)

            if (num_papers_added_and_converted > 0):
                if (num_papers_added_and_converted > last_num_processed):
                    time_elapsed_seconds = round((time.time() - start_time), 1)
                    workflow.add_note(f"{num_papers_added_and_converted} out of {len(workflow.theory_workflow_structure.get('follow_on_paper_request_ids', []))} follow-on papers have been processed by the full-text extraction pipeline so far ({time_elapsed_seconds} seconds elapsed).")
                    last_num_processed = num_papers_added_and_converted
            else:
                workflow.add_note(f"No follow-on papers have been added and converted yet. Still waiting... ({round((time.time() - start_time), 1)} seconds elapsed)")

            if (num_papers_added_and_converted == len(workflow.theory_workflow_structure.get("follow_on_paper_request_ids", []))):
                workflow.add_note(f"All follow-on papers have been processed by the full-text extraction pipeline! ({num_papers_added_and_converted} papers, {round((time.time() - start_time), 1)} seconds elapsed)")
                break

            time.sleep(poll_interval_seconds)

        workflow.theory_workflow_structure["follow_on_paper_ids"] = list(follow_on_papers_converted)
        workflow.theory_workflow_structure["follow_on_paper_ids_before_filtering"] = list(follow_on_papers_converted)

        # Cancel any remaining queued follow-on requests
        workflow.theory_store.paperstore.cancel_incomplete(
            workflow.theory_workflow_structure.get("follow_on_paper_request_ids", [])
        )

        # After waiting, check to see how many of the follow-on papers were successfully added and converted, and add a note to the workflow about it.
        num_papers_added_and_converted = 0
        for request_id in workflow.theory_workflow_structure.get("follow_on_paper_request_ids", []):
            paper_key = workflow.theory_store.paperstore.is_paper_request_complete(request_id)
            if (paper_key is not None):
                num_papers_added_and_converted += 1
        workflow.add_note(f"{num_papers_added_and_converted} out of {len(workflow.theory_workflow_structure.get('follow_on_paper_ids', []))} follow-on papers were successfully added and converted within the timeout period.")

        # Check how many have full-text available
        try:
            num_papers_with_full_text, valid_ids = workflow.theory_store.paperstore.get_num_papers_with_full_text(paper_keys=workflow.theory_workflow_structure.get("follow_on_paper_ids", []))
            workflow.add_note(f"Out of the {len(workflow.theory_workflow_structure.get('follow_on_paper_request_ids', []))} follow-on papers that were processed by the full-text extraction pipeline, {num_papers_with_full_text} have full text available.")
            # Replace the follow-on paper keys with `valid_ids`
            workflow.theory_workflow_structure["follow_on_paper_ids"] = valid_ids
            num_papers_added_and_converted = len(workflow.theory_workflow_structure.get("follow_on_paper_ids", []))

            # Add these to the list of paper keys in the workflow, so we have one big list of all the paper ids.
            workflow.theory_workflow_structure["paper_keys"].extend(workflow.theory_workflow_structure.get("follow_on_paper_ids", []))

        except Exception as e:
            import traceback
            #workflow.add_note(f"Warning: Failed to get the number of follow-on papers with full text available. Error: {e!r}")
            workflow.add_note(f"Warning: Failed to get the number of follow-on papers with full text available. Error: {e!r}\n{traceback.format_exc()}")

        # If at least one follow-on paper was successfully added and converted, we should submit the extraction requests for those papers as well, and wait for them to finish.
        if (num_papers_added_and_converted > 0):
            workflow.add_note(f"Submitting extraction requests for the {num_papers_added_and_converted} follow-on papers...")
            # Submit the extraction requests for the follow-on papers
            follow_on_extraction_request_ids = schema_extraction_queue.submit_extraction_request(extraction_schema_id=extraction_schema_id, paper_ids_to_extract_from=workflow.theory_workflow_structure.get("follow_on_paper_ids"), extraction_model_str=workflow.theory_workflow_structure.get("model_str_extraction"))
            workflow.theory_workflow_structure["follow_on_extraction_request_ids"] = follow_on_extraction_request_ids

            workflow.add_note(f"Submitted {len(workflow.theory_workflow_structure.get('follow_on_extraction_request_ids', []))} extraction requests for the follow-on papers...")

        # Wait for the follow-on extraction requests to be completed, and save the results in the workflow structure for the next steps to use.
        timeout_seconds = max(60 * 20, len(workflow.theory_workflow_structure.get("follow_on_extraction_request_ids", [])) * 5)
        poll_interval_seconds = 5
        start_time = time.time()
        completed_follow_on_extraction_result_ids = []
        num_follow_on_extraction_results_completed_previous = 0
        while ((time.time() - start_time) < timeout_seconds):
            completed_follow_on_extraction_result_ids = []
            for extraction_request_id in workflow.theory_workflow_structure.get("follow_on_extraction_request_ids", []):
                extraction_result_id = schema_extraction_queue.is_request_complete(request_id=extraction_request_id)
                if (extraction_result_id is not None):
                    completed_follow_on_extraction_result_ids.append(extraction_result_id)

            # Check to see if we've made progress since the last poll, and if so, add a note to the workflow about how many follow-on extraction results have been completed so far.
            num_follow_on_extraction_results_completed = len(completed_follow_on_extraction_result_ids)
            print("Number of completed follow-on extraction results: " + str(num_follow_on_extraction_results_completed) + " out of " + str(len(workflow.theory_workflow_structure.get("follow_on_extraction_request_ids", []))))

            if (num_follow_on_extraction_results_completed > num_follow_on_extraction_results_completed_previous):
                time_elapsed_seconds = round((time.time() - start_time), 1)
                workflow.add_note(f"{num_follow_on_extraction_results_completed} out of {len(workflow.theory_workflow_structure.get('follow_on_extraction_request_ids', []))} follow-on extraction requests have been completed so far ({time_elapsed_seconds} seconds elapsed).")
                num_follow_on_extraction_results_completed_previous = num_follow_on_extraction_results_completed

            if (len(completed_follow_on_extraction_result_ids) == len(workflow.theory_workflow_structure.get("follow_on_extraction_request_ids", []))):
                workflow.add_note(f"All follow-on extraction requests have been completed! ({len(completed_follow_on_extraction_result_ids)} requests, {round((time.time() - start_time), 1)} seconds elapsed)")
                break

            time.sleep(poll_interval_seconds)

        # After waiting, check to see how many of the follow-on extraction requests were completed, and add a note to the workflow about it.
        num_follow_on_extraction_results_completed = len(completed_follow_on_extraction_result_ids)
        workflow.add_note(f"{num_follow_on_extraction_results_completed} out of {len(workflow.theory_workflow_structure.get('follow_on_extraction_request_ids', []))} follow-on extraction requests were completed within the timeout period.")

        # Now, join both the lists together, so the next steps can use the combined list of completed extraction results (from both the original papers and the follow-on papers).
        workflow.theory_workflow_structure["completed_extraction_result_ids"] = workflow.theory_workflow_structure.get("completed_extraction_result_ids", []) + workflow.theory_workflow_structure.get("follow_on_extraction_request_ids", [])
        # Get the actual extracted data for the follow-on extraction results, and add it to the list of completed extraction results in the workflow structure for the next steps to use.
        for extraction_result_id in completed_follow_on_extraction_result_ids:
            try:
                extraction_result = theory_store.get_extraction_result(id=extraction_result_id)     # This returns a ExtractionQueryResult -- need to access the `extracted_data` field to get the actual extracted data.
                actual_extraction_results = extraction_result.extracted_data
                if (isinstance(actual_extraction_results, list)):
                    # If the extracted data is a list, we can assume it's a list of individual extraction results, and we can add them to the completed_extraction_results list.
                    workflow.theory_workflow_structure["completed_extraction_results"].extend(actual_extraction_results)
                else:
                    workflow.theory_workflow_structure["completed_extraction_results"].append(actual_extraction_results)
            except Exception as e:
                import traceback
                print(f"Error while accessing extracted_data from follow-on extraction result with ID {extraction_result_id}: {e!r}\n{traceback.format_exc()}")
                continue

        # Write a note to the workflow about the total number of completed extraction results we have now (from both the original papers and the follow-on papers).
        total_completed_extraction_results = len(workflow.theory_workflow_structure.get("completed_extraction_results", []))
        workflow.add_note(f"After processing the follow-on papers, we now have a total of {total_completed_extraction_results} completed extraction results to work with for the next steps.")

    # Try to measure the cost
    extraction_total_cost = 0.0
    for extraction_request_id in workflow.theory_workflow_structure.get("extraction_request_ids", []):
        try:
            extraction_result = theory_store.get_extraction_result(id=extraction_request_id)
            if (extraction_result is not None):
                extraction_total_cost += extraction_result.cost
        except Exception as e:
            import traceback
            print(f"Error while accessing cost from extraction result with ID {extraction_request_id}: {e!r}\n{traceback.format_exc()}")
            continue

    # Set the cost in the workflow
    if (isinstance(extraction_total_cost, (int, float))):
        workflow.add_cost_to_key("paper_extraction", extraction_total_cost)

    # Set completed
    workflow.set_current_step_completed()



class StepProcessorExtractFromPapers(StepProcessor):
    def __init__(self, max_simultaneous_work: int = 5):
        super().__init__(Step.EXTRACT_FROM_PAPERS.value, max_simultaneous_work=max_simultaneous_work)

    def submit_callable(self, workflow: TheoryGenerationWorkflowStructure) -> Future:
        # Submit the EXTERNAL executor function
        return self._executor.submit(extract_from_papers_executor, workflow)



#
#   Step Processor: Make theory from extraction results
#

def make_theory_from_literature_executor(workflow: TheoryGenerationWorkflowStructure) -> None:
    start_time = time.time()

    # Step 1: Get the theory query
    theory_query = workflow.theory_workflow_structure.get("theory_request_normalized", None)

    # Step 2: Get the extraction results from the workflow structure
    extraction_results = workflow.theory_workflow_structure.get("completed_extraction_results", None)
    if (extraction_results is None):
        workflow.set_error_state("ERROR in `make-theory-from-literature`: Completed extraction results not found in the workflow structure.")
        return

    # Step 3: Subsample the results to some maximum number of tokens, if they exceed it
    #MAX_DATA_TOKENS = 100000   # TODO: Get this value from the workflow structure instead of hardcoding it here
    MAX_DATA_TOKENS = workflow.theory_workflow_structure.get("max_data_tokens", 100000)

    workflow.add_note("I'm checking to see if the extracted evidence exceeds the maximum token limit for the model...")
    # Export 'extraction_results'
    filename_debug_out = os.path.join(DEBUG_OUTPUT_DIR, "extraction-results-" + time.strftime("%Y%m%d-%H%M%S") + ".json")
    try:
        with open(filename_debug_out, "w") as f:
            json.dump(extraction_results, f, indent=4)
        workflow.add_note(f"Exported the completed extraction results to a JSON file for debugging: {filename_debug_out}")
    except Exception as e:
        workflow.add_note(f"Warning: Failed to export the completed extraction results to a JSON file for debugging. Error: {e!r}")

    consolidated_subsampled_results = consolodate_results_with_subsampling(extracted_results_in=extraction_results, max_tokens=MAX_DATA_TOKENS)
    if (consolidated_subsampled_results is None):
        print("consolodate_results_with_subsampling() returned None, which indicates an error during subsampling.")
        # Some terrible error has happened in the subsampling -- the results are too large to fit in the model.
        workflow.set_error_state("ERROR in `make-theory-from-literature`: The extracted evidence is too large to fit in the model, even after subsampling. Please reduce the number of papers or the amount of evidence extracted from each paper.")
        return

    data_after_subsampling = consolidated_subsampled_results.get("data_subsampled", None)
    if (data_after_subsampling is None):
        print("consolodate_results_with_subsampling() returned None for the subsampled data, which indicates an error during subsampling.")
        workflow.set_error_state("ERROR in `make-theory-from-literature`: An error occurred during subsampling, and the subsampled data is None.")
        return

    print("Getting subsampling info...")
    subsampling_rate = consolidated_subsampled_results.get("subsampling_rate", None)
    if (subsampling_rate >= 1.0):
        workflow.add_note("The data was not subsampled.")
    else:
        workflow.add_note("The data was subsampled at the following rate: " + str(subsampling_rate))

    # If we reach here, subsampling was successful.
    # Save the data to the workflow.
    workflow.theory_workflow_structure["data_after_subsampling"] = data_after_subsampling
    workflow.theory_workflow_structure["subsampling_rate"] = subsampling_rate

    #print("Consolidated_subsampled_results:")
    #print(json.dumps(consolidated_subsampled_results, indent=4))

    workflow.add_note("Starting theory generation...")

    # Check the generation objective
    known_generation_objectives = [GENERATION_OBJECTIVE_ACCURACY_FOCUS, GENERATION_OBJECTIVE_NOVELTY_FOCUS]
    generation_objective = workflow.theory_workflow_structure["generation_objective"]
    if (generation_objective not in known_generation_objectives):
        workflow.add_note("Warning: Unknown generation objective '" + str(generation_objective) + "'. Defaulting to 'accuracy-focused' theory generation.")
        generation_objective = GENERATION_OBJECTIVE_ACCURACY_FOCUS
        workflow.theory_workflow_structure["generation_objective"] = generation_objective

    workflow.add_note(f"Theory generation will be performed with the following objective: {generation_objective}")

    # Perform theory generation
    theory_result = None
    mission_statement = None
    try:
        generation_temperature = workflow.theory_workflow_structure.get("generation_temperature", 1.0)
        mission_statement = workflow.theory_workflow_structure.get("mission_statement", None)
        # Check the mission statement is the correct type, and if not, set it to None (to disable).
        if (not isinstance(mission_statement, str)):
            mission_statement = None
        else:
            if (len(mission_statement.strip()) == 0):
                mission_statement = None

        MAX_TOKENS_OUT=32000
        if (generation_objective == GENERATION_OBJECTIVE_ACCURACY_FOCUS):
            workflow.add_note(f"Calling build_theory_from_results_single_theory_reflection3...")
            theory_result = build_theory_from_results_single_theory_reflection3(
                query=theory_query,
                results=data_after_subsampling,
                mission_statement=mission_statement,
                model_str=workflow.theory_workflow_structure.get("model_str_primary", None),
                max_tokens=MAX_TOKENS_OUT,
                temperature=generation_temperature,
                use_reflection=True,
                include_query_in_generation=True
            )

        elif (generation_objective == GENERATION_OBJECTIVE_NOVELTY_FOCUS):
            workflow.add_note(f"Calling build_theory_from_results_single_theory_reflection4_nonsafebasin...")
            theory_result = build_theory_from_results_single_theory_reflection4_nonsafebasin(
                query = theory_query,
                results = data_after_subsampling,
                mission_statement=mission_statement,
                model_str = workflow.theory_workflow_structure.get("model_str_primary", None),
                max_tokens = MAX_TOKENS_OUT,
                temperature = generation_temperature,
                use_reflection = True,
                include_query_in_generation = True
            )

        else:
            workflow.set_error_state("ERROR in `make-theory-from-literature`: Unknown generation objective '" + str(generation_objective) + "'. This is a critical error.")

        # DEBUG: Save the Theory result.
        filename_debug_out = os.path.join(DEBUG_OUTPUT_DIR, "theory-result-" + time.strftime("%Y%m%d-%H%M%S") + ".json")
        with open(filename_debug_out, "w") as f:
            json.dump(theory_result, f, indent=4)
        workflow.add_note(f"Exported the theory generation result to a JSON file for debugging: {filename_debug_out}")


    except Exception as e:
        import traceback
        workflow.add_note("An error occurred during theory generation.")
        workflow.add_note(f"Exception details: {e!r}")

        error_message = f"Exception during theory generation: {e!r}\n{traceback.format_exc()}"
        workflow.set_error_state(error_message)
        print(error_message)

        # Print the traceback to the console
        print("Traceback (most recent call last):")
        traceback.print_exc()

        return



    workflow.add_note("After theory generation...")

    # Add the time and cost information
    workflow.add_cost_to_key("theory_formation", theory_result.get("total_cost", 0.0))
    workflow.theory_workflow_structure["theory_result"] = theory_result

    # Add theories to Theory Store
    theory_response = theory_result.get("theory_response", None)
    if (theory_response is None):
        workflow.set_error_state("ERROR in `make-theory-from-literature`: Theory formation failed. No theory response returned.")
        return

    theories_flat_to_convert = []
    for theory in theory_response.get("theories_general", []):
        theory["type"] = "general"
        theories_flat_to_convert.append(theory)
    for theory in theory_response.get("theories_specific", []):
        theory["type"] = "specific"
        theories_flat_to_convert.append(theory)

    # Resolve the datum uuids a theory's statements cite back to their extraction-result ids (deduped, order-preserving) via the store's uuid_index.
    def gather_supporting_evidence_ids(theory_components: dict, uuid_index: dict) -> list:
        result_ids, seen = [], set()
        def walk(node):
            if isinstance(node, dict):
                for u in (node.get("uuids") or []):
                    entry = uuid_index.get(u)
                    if entry and entry[1] not in seen:
                        seen.add(entry[1])
                        result_ids.append(entry[1])
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(theory_components)
        return result_ids

    theory_ids_added_to_store = []
    try:
        for theory in theories_flat_to_convert:
            # Create the Theory objects

            # Add generation objective
            theory_name = theory.get("theory_name", "")
            theory_description = theory.get("theory_description", "")
            theory_type = theory.get("type", "")
            # Supporting evidence = extraction results THIS theory's statements cite, not the whole training set.
            theory_supporting_evidence_ids = gather_supporting_evidence_ids(theory, workflow.theory_store.uuid_index)

            theory["generation_objective"] = generation_objective
            theory["mission_statement"] = mission_statement

            workflow.add_note(f"Adding a theory to the Theory Store with the following name: '{theory_name}'")
            # def __init__(self, name:str, description:str, type:str, theory_query:str, derived_from:list, theory_evaluation_ids:list, components:dict, supporting_evidence_ids:list[str], knowledge_cutoff_year:int, knowledge_cutoff_month:int, change_log:list=None, id:str=None, model_str:str=None):
            theory_object = Theory(
                name=theory_name,
                description=theory_description,
                type=theory_type,
                theory_query=theory_query,
                derived_from=None,
                theory_evaluation_ids=[],
                components=theory,
                supporting_evidence_ids=theory_supporting_evidence_ids,
                knowledge_cutoff_year=workflow.theory_workflow_structure.get("knowledge_cutoff_year", None),
                knowledge_cutoff_month=workflow.theory_workflow_structure.get("knowledge_cutoff_month", None),
                model_str=workflow.theory_workflow_structure.get("model_str_primary", None)
            )
            # Add the Theory object to the Theory Store
            theory_id = workflow.theory_store.add_theory(theory_object)
            theory_ids_added_to_store.append(theory_id)
    except Exception as e:
        import traceback
        workflow.add_note("An error occurred while adding theories to the Theory Store.")
        workflow.add_note(f"Exception details: {e!r}")

        error_message = f"Exception while adding theories to the Theory Store: {e!r}\n{traceback.format_exc()}"
        workflow.set_error_state(error_message)
        print(error_message)

         # Print the traceback to the console
        print("Traceback (most recent call last):")
        traceback.print_exc()

        return

    end_time = time.time()
    time_elapsed_seconds = round((end_time - start_time), 1)
    workflow.add_note(f"Theory formation completed in {time_elapsed_seconds} seconds. Added {len(theory_ids_added_to_store)} theories to the Theory Store.")

    # Export the theory store as a debug step.
    filename_out_debug = os.path.join(DEBUG_OUTPUT_DIR, "theory_store_debug_after_theory_formation." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    print("DEBUG: Saving file: " + str(filename_out_debug))
    workflow.theory_store.save(filename_out_debug)

    # Set this step as completed.
    workflow.set_current_step_completed()



class StepProcessorMakeTheoryFromLiterature(StepProcessor):
    def __init__(self, max_simultaneous_work: int = 5):
        super().__init__(Step.FORM_THEORY_FROM_LITERATURE.value, max_simultaneous_work=max_simultaneous_work)

    def submit_callable(self, workflow: TheoryGenerationWorkflowStructure) -> Future:
        # Submit the EXTERNAL executor function
        return self._executor.submit(make_theory_from_literature_executor, workflow)



#
#   Step Processor: Saving Theories
#

def save_theorystore_to_disk_executor(workflow: TheoryGenerationWorkflowStructure) -> None:
    try:
        workflow.add_note("I'm saving the Theory Store...")

        generation_id = workflow.theory_workflow_structure.get("generation_id", None)
        if generation_id is not None and not isinstance(generation_id, str):
            generation_id = str(generation_id)
        if not isinstance(generation_id, str):
            generation_id = "unknown"

        # Build the full store dict (matches TheoryStore.save() format exactly)
        store_dict = {
            'theories': workflow.theory_store.get_all_theories_as_dict(),
            'extraction_schemas': workflow.theory_store.get_all_extraction_schemas_as_dict(),
            'extraction_results': workflow.theory_store.get_all_extraction_results_as_dict(),
            'theory_evaluations': workflow.theory_store.get_all_theory_evaluations_as_dict(),
            'all_ids': list(workflow.theory_store.all_ids),
            'next_ids': workflow.theory_store.next_ids,
        }
        Persistence.store().save_theorystore(generation_id, store_dict)
        workflow.add_note("Theory Store saved successfully.")

        # Also save the paperstore via persistence
        workflow.add_note("I'm also saving the Paper Store...")
        with workflow.theory_store.paperstore.THREAD_LOCK_PAPERSTORE:
            # paperstore_dict = {
            #     "paperstore": workflow.theory_store.paperstore.paperstore,
            #     "paperstore_lut": workflow.theory_store.paperstore.paperstore_lut,
            # }
            paperstore_dict = workflow.theory_store.paperstore.serialize()
        Persistence.store().save_paperstore(generation_id, paperstore_dict)
        workflow.add_note("Paper Store saved successfully.")

        workflow.set_current_step_completed()

    except Exception as e:
        error_message = f"Exception in save_theorystore_to_disk_executor: {e!r}\n{traceback.format_exc()}"
        workflow.set_error_state(error_message)
        print(error_message)


class StepProcessorSaveTheoriesToDisk(StepProcessor):
    def __init__(self, max_simultaneous_work: int = 5):
        super().__init__(Step.SAVE_THEORY_TO_DISK.value, max_simultaneous_work=max_simultaneous_work)

    def submit_callable(self, workflow: TheoryGenerationWorkflowStructure) -> Future:
        # Submit the EXTERNAL executor function
        return self._executor.submit(save_theorystore_to_disk_executor, workflow)





#
#   Step Processor: Qualified Novelty Evaluation
#

    # Example of using the Persistence layer.
        # generation_id = workflow.theory_workflow_structure.get("generation_id", None)
        # if generation_id is not None and not isinstance(generation_id, str):
        #     generation_id = str(generation_id)
        # if not isinstance(generation_id, str):
        #     generation_id = "unknown"

        # # Build the full store dict (matches TheoryStore.save() format exactly)
        # store_dict = {
        #     'theories': workflow.theory_store.get_all_theories_as_dict(),
        #     'extraction_schemas': workflow.theory_store.get_all_extraction_schemas_as_dict(),
        #     'extraction_results': workflow.theory_store.get_all_extraction_results_as_dict(),
        #     'theory_evaluations': workflow.theory_store.get_all_theory_evaluations_as_dict(),
        #     'all_ids': list(workflow.theory_store.all_ids),
        #     'next_ids': workflow.theory_store.next_ids,
        # }
        # Persistence.store().save_theorystore(generation_id, store_dict)
        # workflow.add_note("Theory Store saved successfully.")

        # # Also save the paperstore via persistence
        # workflow.add_note("I'm also saving the Paper Store...")
        # with workflow.theory_store.paperstore.THREAD_LOCK_PAPERSTORE:
        #     paperstore_dict = {
        #         "paperstore": workflow.theory_store.paperstore.paperstore,
        #         "paperstore_lut": workflow.theory_store.paperstore.paperstore_lut,
        #     }
        # Persistence.store().save_paperstore(generation_id, paperstore_dict)
        # workflow.add_note("Paper Store saved successfully.")


def qualified_novelty_evaluation_executor(workflow: TheoryGenerationWorkflowStructure) -> None:
    try:
        # Step 1: Check if the qualified novelty evaluation is enabled in the workflow structure
        if (not workflow.theory_workflow_structure.get("do_qualified_novelty_evaluation", False)):
            workflow.add_note("Qualified novelty evaluation is not enabled for this workflow. Skipping this step.")
            workflow.set_current_step_completed()
            return
        else:
            workflow.add_note("Qualified novelty evaluation is enabled for this workflow. Starting this step...")


        #filename_theorystore = workflow.theory_workflow_structure.get("filename_out_theory_store", None)
        #filename_paperstore = workflow.theory_workflow_structure.get("filename_out_paperstore", None)
        # Persistence layer: Reference by generation ID instead of filename
        generation_id = workflow.theory_workflow_structure.get("generation_id", None)
        if generation_id is not None and not isinstance(generation_id, str):
            generation_id = str(generation_id)
        if not isinstance(generation_id, str):
            generation_id = "unknown"

        # Get the theorystore, and the paperstore
        theorystore = Persistence.store().get_theorystore(generation_id)
        paperstore = Persistence.store().get_paperstore(generation_id)

        model_str_primary = workflow.theory_workflow_structure.get("model_str_primary", None)
        model_str_extraction = workflow.theory_workflow_structure.get("model_str_extraction", None)
        num_workers = 3
        QUALIFIED_NOVELTY_DEBUG_LIMIT = None
        #QUALIFIED_NOVELTY_DEBUG_LIMIT = 2

        if (theorystore is None) or (paperstore is None):
            #workflow.add_note("Error: Missing filename for theorystore or paperstore in the workflow structure. Cannot perform qualified novelty evaluation.")
            workflow.set_error_state("ERROR in `qualified_novelty_evaluation_executor`: Failed to load the TheoryStore or PaperStore from the Persistence layer. Cannot perform qualified novelty evaluation.")
            return

        # get_paperstore() spins up a PaperStore monitor thread; hand it to the workflow so
        # cleanup() stops it at retirement (otherwise every novelty run — the default path —
        # leaks a thread straight through the OOM fix).
        workflow.register_background_service(paperstore)

        # This should automatically save the results back to the TheoryStore, but `results` will contain other important information (like cost)
        workflow.add_note("Qualified novelty evaluation has started...")
        start_time_qualified_novelty = time.time()
        # TODO: Replace this with actual loaded copies of the theorystore/paperstore. Have it return the updated TheoryStore, so we can save it.
        on_statement_cb = getattr(workflow, 'on_novelty_statement_evaluated', None)
        result = do_qualified_novelty_evaluation_persistence(
            theorystore_data = theorystore,
            paperstore = paperstore,
            model_str_extraction=model_str_extraction,
            model_str_decision=model_str_primary,
            num_workers=num_workers,
            DEBUG_LIMIT=QUALIFIED_NOVELTY_DEBUG_LIMIT,
            on_statement_evaluated=on_statement_cb)
        total_time_qualified_novelty = round((time.time() - start_time_qualified_novelty), 1)
        #workflow.add_note("Qualified novelty evaluation has completed.")
        workflow.add_note(f"Qualified novelty evaluation has completed in {total_time_qualified_novelty} seconds.")

        # TODO: Save the new updated TheoryStore that includes the qualified novelty evaluation.
        theorystore_data = result.get("theorystore_data", None)
        if (theorystore_data is not None):
            Persistence.store().save_theorystore(generation_id, theorystore_data)
            workflow.add_note("The updated TheoryStore with qualified novelty evaluation results has been saved to disk.")
            # Re-sync the in-memory store with the updated data (novelty added evaluations).
            workflow.theory_store.load_from_dict(theorystore_data)
        else:
            workflow.add_note("Warning: Theorystore data with qualified novelty evaluation results is None. The updated TheoryStore will not be saved.")

        total_cost_qualified_novelty = result.get("total_cost", None)
        if (isinstance(total_cost_qualified_novelty, (int, float))):
            workflow.add_cost_to_key("qualified_novelty_evaluation", total_cost_qualified_novelty)
        else:
            workflow.add_note("Warning: Total cost for qualified novelty evaluation is not a valid number. Received value: " + str(total_cost_qualified_novelty))

        workflow.set_current_step_completed()

    except Exception as e:
        error_message = f"Exception in qualified_novelty_evaluation_executor: {e!r}\n{traceback.format_exc()}"
        workflow.set_error_state(error_message)
        print(error_message)
        return




class StepProcessorQualifiedNoveltyEvaluation(StepProcessor):
    def __init__(self, max_simultaneous_work: int = 5):
        super().__init__(Step.QUALIFIED_NOVELTY_EVALUATION.value, max_simultaneous_work=max_simultaneous_work)

    def submit_callable(self, workflow: TheoryGenerationWorkflowStructure) -> Future:
        # Submit the EXTERNAL executor function
        return self._executor.submit(qualified_novelty_evaluation_executor, workflow)


#
#   (New) Main Theorizer class
#
class Theorizer:
    # Constructor
    def __init__(self, path_in:str="data/"):
        # External stores
        self.paperstore = PaperStore()
        #self.theory_store = TheoryStore(paperstore=self.paperstore, filename_in=filename_theory_store_in)
        #self.paperfinder_requests = PaperFinderRequests()
        #self.schema_extraction_queue = SchemaExtractionQueue(theorystore=self.theory_store, paperstore=self.paperstore)

        # Make the path_in if it doesn't exist
        if (not os.path.exists(path_in)):
            print("Path does not exist, creating: " + str(path_in))
            os.makedirs(path_in)

        # Queues
        self.workflow_queue_unstarted = []
        self.workflow_queue_in_process = []
        # Bounded so finished workflows (and the paper full-text they pin) get
        # garbage-collected instead of accumulating until OOM. Append auto-evicts the
        # oldest atomically; evicted workflows were already cleaned up at retirement.
        self.workflows_completed = deque(maxlen=MAX_COMPLETED_RETAINED)

        self.finished_work = {}
        self.next_request_id = 1

        # Update message: Show the user the Theorizer heartbeat every 10 seconds
        self.heartbeat_interval = 5
        self.heartbeat_last_time = 0

        # Thread lock
        self.THREAD_LOCK_THEORIZER_REQUESTS = threading.Lock()
        self.THREAD_ACTIVE = False
        # Thread lock for base ID level
        self.THREAD_LOCK_BASE_ID_LEVEL = threading.Lock()
        self.next_base_id_level = 0

        self.load_base_id_level_from_file()

        # Start a background thread to process the queue
        self.worker_thread = None
        self.actively_processing_work = False
        self.start_thread()

        # Step Processors
        self.step_processors = {}
        self.step_processors[Step.BUILD_PAPERFINDER_REQUEST] = StepProcessorBuildPaperFinderRequest(max_simultaneous_work=5)
        self.step_processors[Step.FIND_PAPERS_WITH_PAPERFINDER] = StepProcessorMakePaperFinderRequest(max_simultaneous_work=5)
        self.step_processors[Step.EXTRACT_FROM_PAPERS] = StepProcessorExtractFromPapers(max_simultaneous_work=5)
        self.step_processors[Step.FORM_THEORY_FROM_LITERATURE] = StepProcessorMakeTheoryFromLiterature(max_simultaneous_work=5)
        self.step_processors[Step.SAVE_THEORY_TO_DISK] = StepProcessorSaveTheoriesToDisk(max_simultaneous_work=5)
        self.step_processors[Step.QUALIFIED_NOVELTY_EVALUATION] = StepProcessorQualifiedNoveltyEvaluation(max_simultaneous_work=5)

        # Start all step processor threads
        for step_name, step_processor in self.step_processors.items():
            step_processor.start_thread()

        # A UUID generator for generating unique IDs for workflows and requests

    def generate_uuid(self):
        return str(uuid.uuid4())

    def load_base_id_level_from_file(self):
        try:
            data = Persistence.store().get_kv("base_id_level")
            if data is not None:
                base_id_level = data.get("base_id_level", 1)
                if isinstance(base_id_level, int) and base_id_level >= 0:
                    self.next_base_id_level = base_id_level
        except Exception as e:
            print(f"Error loading base ID level: {e!r}. Starting with default: {self.next_base_id_level}")

    def get_next_base_id_level(self):
        with self.THREAD_LOCK_BASE_ID_LEVEL:
            value_to_write = self.next_base_id_level
            self.next_base_id_level += 20
            try:
                Persistence.store().set_kv("base_id_level", {"base_id_level": self.next_base_id_level})
            except Exception as e:
                print(f"Error saving base ID level: {e!r}")
            return value_to_write

    #
    #   Get current workflow statuses or data.
    #

    # Get the status of a workflow by generation ID.
    def get_workflow_status_by_generation_id(self, generation_id:str):
        try:
            generation_id_sanitized = generation_id.strip().replace(" ", "_")
            workflow_data = Persistence.store().get_workflow(generation_id_sanitized)
            if workflow_data is not None:
                return workflow_data
            else:
                # Check in the waiting queue to see if it's there, and if so, what it's position is.
                position_in_queue = None
                for idx, workflow in enumerate(self.workflow_queue_unstarted):
                    workflow_generation_id = workflow.theory_workflow_structure.get("generation_id", None)
                    if (workflow_generation_id == generation_id):
                        position_in_queue = idx + 1   # +1 to convert from 0-indexed to 1-indexed position
                        break

                if (position_in_queue is not None):
                    return {
                        "generation_id": generation_id,
                        "theory_query": workflow.theory_workflow_structure.get("theory_query", None),
                        "queue_position": position_in_queue,
                    }

                # Otherwise, no workflow found with this generation ID in either the active workflows or the waiting queue.
                return {"error": "Workflow not found for generation ID: " + generation_id + ".  If you are sure the generation id is correct, then the workflow likely has not yet started."}
        except Exception as e:
            print(f"Error while loading workflow status: {e!r}")
            return {"error": "Error while loading workflow status for generation ID: " + generation_id + ". Error details: " + str(e)}

    # Get the output theorystore from a given workflow, by generation ID.
    def get_theorystore_by_generation_id(self, generation_id:str):
        try:
            theorystore_data = Persistence.store().get_theorystore(generation_id)
            if theorystore_data is not None:
                return theorystore_data
            else:
                return {"error": "Theorystore not found for generation ID: " + generation_id + ". If you are sure the generation id is correct, then the workflow likely has not yet completed."}
        except Exception as e:
            print(f"Error while loading theorystore: {e!r}")
            return {"error": "Error while loading theorystore for generation ID: " + generation_id + ". Error details: " + str(e)}


    # Show the status of the server, including all active workflows, and all queued workflows.
    def get_server_status(self):
        # Not using the thread lock here, since we're just reading, and it's generally OK if this information is slightly out of date.
        active_workflows = []
        try:
            for workflow in self.workflow_queue_in_process:
                # Show the generation ID, current step, theory query, total cost, total runtime
                active_workflows.append(
                    {
                        "generation_id": workflow.theory_workflow_structure.get("generation_id", None),
                        "current_step": workflow.get_current_step_name(),
                        "theory_query": workflow.theory_workflow_structure.get("theory_query", None),
                        "is_completed": workflow.is_at_terminal_step(),
                        "has_errors": workflow.is_in_error_state(),
                        "errors": workflow.theory_workflow_structure.get("errors", []),
                        "cost": workflow.get_total_cost(),
                        "runtime_sec": workflow.get_total_time(),
                        "runtime_steps": workflow.theory_workflow_structure.get("time_components", [])
                    }
                )
        except Exception as e:
            print(f"Error while getting server status for active workflows: {e!r}")


        queued_workflows = []
        try:
            for workflow in self.workflow_queue_unstarted:
                queued_workflows.append(
                    {
                        "generation_id": workflow.theory_workflow_structure.get("generation_id", None),
                        "current_step": workflow.get_current_step_name(),
                        "theory_query": workflow.theory_workflow_structure.get("theory_query", None),
                        "is_completed": workflow.is_completed,
                        "has_errors": workflow.is_in_error_state(),
                        "errors": workflow.theory_workflow_structure.get("errors", []),
                        "cost": workflow.get_total_cost(),
                        "runtime_sec": workflow.get_total_time(),
                        "runtime_steps": workflow.theory_workflow_structure.get("time_components", [])
                    }
                )
        except Exception as e:
            print(f"Error while getting server status for queued workflows: {e!r}")

        # Completed workflows
        completed_workflows = []
        try:
            # Snapshot first: workflows_completed is a deque the worker thread appends to, and
            # deque raises "deque mutated during iteration" if it grows mid-iterate (a list won't).
            for workflow in list(self.workflows_completed):
                completed_workflows.append(
                    {
                        "generation_id": workflow.theory_workflow_structure.get("generation_id", None),
                        "current_step": workflow.get_current_step_name(),
                        "theory_query": workflow.theory_workflow_structure.get("theory_query", None),
                        "is_completed": workflow.is_at_terminal_step(),
                        "has_errors": workflow.is_in_error_state(),
                        "errors": workflow.theory_workflow_structure.get("errors", []),
                        "cost": workflow.get_total_cost(),
                        "runtime_sec": workflow.get_total_time(),
                        "runtime_steps": workflow.theory_workflow_structure.get("time_components", [])
                    }
                )
        except Exception as e:
            print(f"Error while getting server status for completed workflows: {e!r}")


        packed = {
            "active_workflows": active_workflows,
            "queued_workflows": queued_workflows,
            "completed_workflows": completed_workflows
        }

        # Return the active and queued workflow statuses
        return packed


    #
    #   Work Queueing/Threading
    #

    # # Submit a request to the PaperFinder API, which will add it to the request queue.
    # # This is the normal (literature-supported) theory generation workflow.
    # def submit_theory_query(self, query:str, max_papers_to_retrieve:int=10, extraction_evaluation_year:int=2025, extraction_evaluation_month:int=1, model_str_build_new_request:str=MODEL_STR_TO_USE, model_str_extraction:str="gpt-5-mini", generation_objective:str=GENERATION_OBJECTIVE_ACCURACY_FOCUS):
    #     # Make a workflow for this query
    #     new_workflow = TheorizerNewTheoryGeneration(theorizer=self, name="Theory Query: " + query, theory_query=query, theory_id=None, max_papers_to_retrieve=max_papers_to_retrieve, extraction_evaluation_year=extraction_evaluation_year, extraction_evaluation_month=extraction_evaluation_month, model_str_build_new_request=model_str_build_new_request, model_str_extraction=model_str_extraction, generation_objective=generation_objective)
    #     with self.THREAD_LOCK_THEORIZER_REQUESTS:
    #         self.workflow_queue.append(new_workflow)

    # # TODO: Make a function to submit a Parametric-only theory workflow.
    # def submit_theory_query_parametric_only(self, query:str, model_str_build_new_request:str=MODEL_STR_TO_USE, generation_objective:str=GENERATION_OBJECTIVE_ACCURACY_FOCUS):
    #     # Make a workflow for this query
    #     new_workflow = TheorizerNewTheoryGenerationParametricOnly(theorizer=self, name="Parametric-Only Theory Query: " + query, theory_query=query, theory_id=None, model_str_build_new_request=model_str_build_new_request, generation_objective=generation_objective)
    #     with self.THREAD_LOCK_THEORIZER_REQUESTS:
    #         self.workflow_queue.append(new_workflow)

    # As above, but using the new `TheoryGenerationWorkflowStructure` storage class.
    def submit_theory_query(self, user_id:str,
                            theory_query:str, mission_statement:str,
                            max_papers_to_retrieve:int=10, knowledge_cutoff_year:int=2050, knowledge_cutoff_month:int=1,
                            model_str_primary:str=DEFAULT_MODEL_PRIMARY, model_str_extraction:str=DEFAULT_MODEL_EXTRACTION,
                            generation_objective:str=GENERATION_OBJECTIVE_ACCURACY_FOCUS,
                            generation_input:str=GENERATION_INPUT_LITERATURE_SUPPORT,
                            max_data_tokens:int=100000,
                            do_qualified_novelty_evaluation:bool=False,
                            ):

        # Make a new UUID for this workflow
        generation_id = self.generate_uuid()

        base_id_level = self.get_next_base_id_level()

        # Make a workflow structure for this query
        new_workflow_structure = TheoryGenerationWorkflowStructure(
            generation_id=generation_id,
            user_id=user_id,
            request_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            theory_query=theory_query,
            mission_statement=mission_statement,
            max_papers_to_retrieve=max_papers_to_retrieve,
            knowledge_cutoff_year=knowledge_cutoff_year,
            knowledge_cutoff_month=knowledge_cutoff_month,
            model_str_primary=model_str_primary,
            model_str_extraction=model_str_extraction,
            generation_objective=generation_objective,
            generation_input=generation_input,
            base_id_level=base_id_level,
            max_data_tokens=max_data_tokens,
            do_qualified_novelty_evaluation=do_qualified_novelty_evaluation,
        )

        # Add this into the workflow queue
        with self.THREAD_LOCK_THEORIZER_REQUESTS:
            self.workflow_queue_unstarted.append(new_workflow_structure)

        # Return the generation ID
        submission_time = time.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "generation_id": generation_id,
            "submission_time": submission_time,
            "theory_query": theory_query,
            "message": "Successfully submitted to queue."
        }


    def submit_workflow(self, workflow):
        """Submit an already-constructed workflow into the worker queue."""
        with self.THREAD_LOCK_THEORIZER_REQUESTS:
            self.workflow_queue_unstarted.append(workflow)


    # Get the number of requests in the queue
    def get_num_active_workflows(self):
        with self.THREAD_LOCK_THEORIZER_REQUESTS:
            return len(self.workflow_queue_unstarted) + len(self.workflow_queue_in_process)

    # # Get the status of all active workflows in the system
    # def get_workflow_statuses(self):
    #     # Not using the thread lock here, since we're just reading, and it's generally OK if this information is slightly out of date.
    #     out_active = []
    #     for workflow in self.workflow_queue:
    #         has_errors, errors = workflow.has_errors()
    #         out_active.append({"name": workflow.get_name(), "current_step": workflow.get_current_step_name(), "is_completed": workflow.is_completed, "has_errors": has_errors, "errors": errors, "status_str": workflow.get_status_str(), "cost": workflow.get_cost(), "runtime_sec": workflow.get_runtime(), "runtime_steps": workflow.get_time_components()})

    #     out_completed = []
    #     for workflow in self.workflows_completed:
    #         has_errors, errors = workflow.has_errors()
    #         out_completed.append({"name": workflow.get_name(), "current_step": workflow.get_current_step_name(), "is_completed": workflow.is_completed, "has_errors": has_errors, "errors": errors, "status_str": workflow.get_status_str(), "cost": workflow.get_cost(), "runtime_sec": workflow.get_runtime(), "runtime_steps": workflow.get_time_components()})

    #     packed = {
    #         "active_workflows": out_active,
    #         "completed_workflows": out_completed,
    #     }
    #     # Return the active and completed workflow statuses
    #     return packed

    # Returns true if there is anything in the queue, or anything currently being processed
    def is_busy(self):
        # Check 1: Are there any requests in the queue?
        if (self.get_num_active_workflows() > 0):
            return True

        # If we reach here, then we're not busy
        return False


    # Stop the worker thread
    def stop_thread(self):
        print("Theorizer.start_thread(): Stopping worker thread...")
        self.THREAD_ACTIVE = False
        if (self.worker_thread is not None):
            self.worker_thread.join()
        self.worker_thread = None

        # Stop all step processor threads
        for step_name, step_processor in self.step_processors.items():
            step_processor.stop_thread()
        print("Theorizer.start_thread(): Worker thread stopped.")

    def start_thread(self):
        with self.THREAD_LOCK_THEORIZER_REQUESTS:
            if (self.worker_thread is not None) and (self.worker_thread.is_alive()):
                print("Theorizer.start_thread(): WARNING: Worker thread is already running -- not starting a new one.")
                return
            print("Theorizer.start_thread(): Starting worker thread...")

            self.THREAD_ACTIVE = True
            # Start the worker thread
            self.worker_thread = threading.Thread(target=self.process_work_monitor, daemon=True)
            self.worker_thread.start()

            print("Theorizer.start_thread(): Worker thread started.")


    # Check to see if a given workflow is waiting to move to the next step.  If it is, send it there.
    def dispatch_workflow(self, workflow_structure:TheoryGenerationWorkflowStructure):
        # Dispatch the workflow to the appropriate worker thread, based on the new current step.
        current_step_name = workflow_structure.get_current_step_name()

        # Fail loudly if the dispatcher tries to run a step not in the workflow's declared step list (per-skill scoping is what keeps it from running the full pipeline).
        declared_steps = workflow_structure.theory_workflow_structure.get("steps", [])
        if current_step_name is not None and current_step_name not in declared_steps:
            workflow_structure.set_error_state(
                f"Dispatcher tried to run step '{current_step_name}' but it is not in the "
                f"workflow's declared step list {[str(s) for s in declared_steps]}."
            )
            return False

        # DEBUG: Show the current workflow notes.
        print("-"*50)
        print("DispatchWorkflow: Workflow ID: " + str(workflow_structure.theory_workflow_structure.get("generation_id", "N/A")) + ", User ID: " + str(workflow_structure.theory_workflow_structure.get("user_id", "N/A")))
        print("\tQuery: '" + str(workflow_structure.theory_workflow_structure.get("theory_query", "N/A")))
        #print(+ "'. Current step: '" + str(current_step_name) + "'. Notes: " + str(workflow_structure.theory_workflow_structure.get("notes", [])))
        print("\tCurrent step: '" + str(current_step_name) + "'.")
        print("Notes:")
        for idx, note in enumerate(workflow_structure.theory_workflow_structure.get("notes", [])):
            print(f"\t{idx+1}. " + str(note))
        print("-"*50)


        # Switch/case on all the steps.
        match current_step_name:
            #
            # Initialization state
            #
            case Step.INIT:
                # This step doesn't have any work to do, so we can just mark it as completed.
                workflow_structure.add_note("Workflow is in the INIT step. Initializing workflow.")
                workflow_structure.set_current_step_completed()

                return True

            #
            # Terminal states
            #

            # Complete state
            case Step.COMPLETE:
                # This step doesn't have any work to do, so we can just mark it as completed.
                workflow_structure.add_note("Workflow has reached the COMPLETE step. Workflow is now marked as complete.")
                workflow_structure.set_current_step_completed()
                return True

            # Error state
            case Step.ERROR:
                # This step doesn't have any work to do, so we can just mark it as completed.
                workflow_structure.add_note("Workflow has reached the ERROR step. Workflow is now marked as complete, but with errors. Please check the error messages for details.")
                workflow_structure.set_current_step_completed()
                return True

            #
            # Work states
            #

            case Step.BUILD_PAPERFINDER_REQUEST:
                # Dispatch this workflow to the worker thread that builds PaperFinder requests.
                # TODO: Just skip over for now, and set complete.
                workflow_structure.add_note("Building PaperFinder request for query '" + workflow_structure.theory_workflow_structure.get("theory_query", "N/A") + "' with model '" + workflow_structure.theory_workflow_structure.get("model_str_primary", "N/A") + "' and generation objective '" + workflow_structure.theory_workflow_structure.get("generation_objective", "N/A") + "'.")

                # Submit to the appropriate step processor: build paperfinder request
                self.step_processors[Step.BUILD_PAPERFINDER_REQUEST].add_workflow_to_queue(workflow_structure)

                return True

            case Step.BUILD_EXTRACTION_SCHEMA:
                # Dispatch this workflow to the worker thread that builds schema extraction requests.
                # TODO: Just skip over for now, and set complete.
                # NOTE: This one isn't actually used, since the query currently gets generated in the PaperFinder request step.
                #workflow_structure.add_note("Faux dispatch: Building extraction schema for query '" + workflow_structure.theory_workflow_structure.get("theory_query", "N/A") + "' with model '" + workflow_structure.theory_workflow_structure.get("model_str_extraction", "N/A") + "' and generation objective '" + workflow_structure.theory_workflow_structure.get("generation_objective", "N/A") + "'.")

                workflow_structure.set_current_step_completed()
                return True

            case Step.FIND_PAPERS_WITH_PAPERFINDER:
                # Dispatch this workflow to the worker thread that finds papers.
                # TODO: Just skip over for now, and set complete.
                #workflow_structure.add_note("Faux dispatch: Finding papers for query '" + workflow_structure.theory_workflow_structure.get("theory_query", "N/A") + "' with max papers '" + str(workflow_structure.theory_workflow_structure.get("max_papers_to_retrieve", "N/A")) + "' and knowledge cutoff '" + str(workflow_structure.theory_workflow_structure.get("knowledge_cutoff_year", "N/A")) + "-" + str(workflow_structure.theory_workflow_structure.get("knowledge_cutoff_month", "N/A")) + "'.")
                #workflow_structure.set_current_step_completed()
                workflow_structure.add_note("Finding papers for query '" + workflow_structure.theory_workflow_structure.get("theory_query", "N/A") + "' with max papers '" + str(workflow_structure.theory_workflow_structure.get("max_papers_to_retrieve", "N/A")) + "' and knowledge cutoff '" + str(workflow_structure.theory_workflow_structure.get("knowledge_cutoff_year", "N/A")) + "-" + str(workflow_structure.theory_workflow_structure.get("knowledge_cutoff_month", "N/A")) + "'.")
                # Submit to the appropriate step processor: find papers with paperfinder
                self.step_processors[Step.FIND_PAPERS_WITH_PAPERFINDER].add_workflow_to_queue(workflow_structure)

                return True

            # case Step.CONVERT_PAPERS:
            #     # Dispatch this workflow to the worker thread that collects and converts papers.
            #     # TODO: Just skip over for now, and set complete.
            #     workflow_structure.add_note("Faux dispatch: Collecting and converting papers for query '" + workflow_structure.theory_workflow_structure.get("theory_query", "N/A") + "'.")
            #     workflow_structure.set_current_step_completed()
            #     return True

            case Step.EXTRACT_FROM_PAPERS:
                # Dispatch this workflow to the worker thread that extracts from papers.
                workflow_structure.add_note("Extracting from papers for query '" + workflow_structure.theory_workflow_structure.get("theory_query", "N/A") + "' with extraction model '" + workflow_structure.theory_workflow_structure.get("model_str_extraction", "N/A") + "'.")
                # Submit to the appropriate step processor: extract from papers
                self.step_processors[Step.EXTRACT_FROM_PAPERS].add_workflow_to_queue(workflow_structure)

                return True

            case Step.FORM_THEORY_PARAMETRIC:
                # Dispatch this workflow to the worker thread that forms theories from parametric knowledge.
                # TODO: Just skip over for now, and set complete.
                workflow_structure.add_note("Faux dispatch: Forming theory from parametric knowledge for query '" + workflow_structure.theory_workflow_structure.get("theory_query", "N/A") + "' with model '" + workflow_structure.theory_workflow_structure.get("model_str_primary", "N/A") + "' and generation objective '" + workflow_structure.theory_workflow_structure.get("generation_objective", "N/A") + "'.")
                workflow_structure.set_current_step_completed()
                return True

            case Step.FORM_THEORY_FROM_LITERATURE:
                # Dispatch this workflow to the worker thread that forms theories from literature.
                # TODO: Just skip over for now, and set complete.
                #workflow_structure.add_note("Faux dispatch: Forming theory from literature for query '" + workflow_structure.theory_workflow_structure.get("theory_query", "N/A") + "' with model '" + workflow_structure.theory_workflow_structure.get("model_str_primary", "N/A") + "' and generation objective '" + workflow_structure.theory_workflow_structure.get("generation_objective", "N/A") + "'.")
                #workflow_structure.set_current_step_completed()
                self.step_processors[Step.FORM_THEORY_FROM_LITERATURE].add_workflow_to_queue(workflow_structure)
                return True

            case Step.SAVE_THEORY_TO_DISK:
                # Dispatch this workflow to the worker thread that saves theories to disk.
                # TODO: Just skip over for now, and set complete.
                #workflow_structure.add_note("Faux dispatch: Saving theory to disk for query '" + workflow_structure.theory_workflow_structure.get("theory_query", "N/A") + "'.")
                #workflow_structure.set_current_step_completed()
                self.step_processors[Step.SAVE_THEORY_TO_DISK].add_workflow_to_queue(workflow_structure)
                return True


            case Step.QUALIFIED_NOVELTY_EVALUATION:
                self.step_processors[Step.QUALIFIED_NOVELTY_EVALUATION].add_workflow_to_queue(workflow_structure)
                return True


            case _:
                # Unknown step -- set into an error state.
                workflow_structure.add_note("ERROR: Unknown step '" + str(current_step_name) + "'. Can not dispatch workflow.")
                workflow_structure.set_error_state("Unknown step '" + str(current_step_name) + "'. Can not dispatch workflow.")
                return False


    # Retire a finished workflow: stop its background threads, then hand it to the
    # bounded completed deque (which auto-evicts the oldest so memory stays flat).
    def _retire_workflow(self, workflow_structure):
        try:
            workflow_structure.cleanup()
        except Exception as e:
            print(f"_retire_workflow(): cleanup error: {e!r}")
        self.workflows_completed.append(workflow_structure)

    # Move a workflow from the in-process queue to completed, in an error state.
    def _fail_workflow(self, workflow_structure, error_message:str):
        workflow_structure.add_note(f"ERROR: {error_message}")
        workflow_structure.set_error_state(error_message)
        if workflow_structure in self.workflow_queue_in_process:
            self.workflow_queue_in_process.remove(workflow_structure)
        self._retire_workflow(workflow_structure)
        workflow_structure.save_workflow_to_file()

    # Main thread: Process the request queue
    # This should be running in the background.
    def process_work_monitor(self):
        print("Theorizer.process_work_monitor(): Worker thread started")
        while (self.THREAD_ACTIVE):
            # Step 0: Sleep for 1 second to avoid busy waiting
            time.sleep(5)

            # Step 1: Check if the heartbeat interval has passed
            time_since_last_heartbeat = time.time() - self.heartbeat_last_time
            if (time_since_last_heartbeat >= self.heartbeat_interval):
                # Print a heartbeat message

                # Count how many finished in an error state
                num_finished_in_error_state = 0
                for workflow in self.workflows_completed:
                    if (workflow.is_in_error_state()):
                        num_finished_in_error_state += 1

                print("Theorizer.process_work_monitor(): Heartbeat: Worker thread is alive. (Unstarted workflows: " + str(len(self.workflow_queue_unstarted)) + ", In-process workflows: " + str(len(self.workflow_queue_in_process)) + ", Completed workflows: " + str(len(self.workflows_completed)) + ", Finished in error state: " + str(num_finished_in_error_state) + ")")
                self.heartbeat_last_time = time.time()

            # Step 2: Look for work to perform.

            # Poll each function in the workflow.  This will handle the workflow progressing (or being marked as finished).
            MAX_SIMULTANEOUS_WORKFLOWS = int(os.environ.get("MAX_SIMULTANEOUS_WORKFLOWS", 5))

            # For each workflow in the 'in process' queue, see if it's completed it's current step.
            cur_in_process_idx = 0
            while (cur_in_process_idx < len(self.workflow_queue_in_process)):
                workflow_structure = self.workflow_queue_in_process[cur_in_process_idx]

                # Save the workflow
                workflow_structure.save_workflow_to_file()

                # Step 1: Check to see if work is completed.
                if (not workflow_structure.is_current_step_completed()):
                    # Work isn't completed -- check for timeout.
                    time_on_step = workflow_structure.get_time_on_this_step()
                    total_time = workflow_structure.get_total_time()

                    timed_out = False
                    if (time_on_step is not None and time_on_step > MAX_STEP_TIMEOUT_SECONDS):
                        reason = f"Step '{workflow_structure.get_current_step_name()}' timed out after {time_on_step:.0f} seconds (max {MAX_STEP_TIMEOUT_SECONDS}s)."
                        timed_out = True
                    elif (total_time is not None and total_time > MAX_WORKFLOW_TIMEOUT_SECONDS):
                        reason = f"Workflow total time exceeded {total_time:.0f} seconds (max {MAX_WORKFLOW_TIMEOUT_SECONDS}s)."
                        timed_out = True

                    if timed_out:
                        gen_id = workflow_structure.theory_workflow_structure.get('generation_id', 'N/A')
                        print(f"ERROR: {reason} Forcing error state. (Generation ID: {gen_id})")
                        self._fail_workflow(workflow_structure, reason)
                        continue

                    print("Workflow is still in progress (Generation ID: " + str(workflow_structure.theory_workflow_structure.get("generation_id", "N/A")) + ", User ID: " + str(workflow_structure.theory_workflow_structure.get("user_id", "N/A")) + "). Current step: '" + str(workflow_structure.get_current_step_name()) + "'. Time on this step: " + str(workflow_structure.get_time_on_this_step()) + " seconds. Total time: " + str(workflow_structure.get_total_time()) + " seconds.")
                    cur_in_process_idx += 1

                    # Debug: Show the workflow
                    print(workflow_structure.get_notes_debug_str())
                    continue

                # Step 2: If we reach here, work is completed.
                # First, check if we're in a terminal step, and need to remove this workflow from the 'in process' queue and move it to the 'completed' queue.
                if (workflow_structure.is_at_terminal_step()):
                    # If we're at a terminal step, then we can move this workflow to the 'completed' queue.
                    self.workflow_queue_in_process.remove(workflow_structure)
                    self._retire_workflow(workflow_structure)
                    print("Workflow completed, moving to completed queue. (Generation ID: " + str(workflow_structure.theory_workflow_structure.get("generation_id", "N/A")) + ", User ID: " + str(workflow_structure.theory_workflow_structure.get("user_id", "N/A")) + ", Query: '" + str(workflow_structure.theory_workflow_structure.get("theory_query", "N/A")) + "'). Completed step: '" + str(workflow_structure.get_current_step_name()) + "'. Time on this step: " + str(workflow_structure.get_time_on_this_step()) + " seconds. Total time: " + str(workflow_structure.get_total_time()) + " seconds.")
                    # Debug: Show the workflow
                    print(workflow_structure.get_notes_debug_str())

                else:
                    # If we're not at a terminal step, then we need to increment to the next step, and dispatch the work for that step.
                    print("Workflow step completed, incrementing step. (Generation ID: " + str(workflow_structure.theory_workflow_structure.get("generation_id", "N/A")) + ", User ID: " + str(workflow_structure.theory_workflow_structure.get("user_id", "N/A")) + ", Query: '" + str(workflow_structure.theory_workflow_structure.get("theory_query", "N/A")) + "'). Completed step: '" + str(workflow_structure.get_current_step_name()) + "'. Time on this step: " + str(workflow_structure.get_time_on_this_step()) + " seconds. Total time: " + str(workflow_structure.get_total_time()) + " seconds.")
                    workflow_structure.increment_step()
                    # Isolate dispatch: an exception here must fail this one workflow, never
                    # escape and kill the single worker thread (which would freeze the whole
                    # queue and hang every in-flight task on done.wait()).
                    try:
                        self.dispatch_workflow(workflow_structure)
                        cur_in_process_idx += 1   # advance only when the workflow stayed in the in-process queue
                    except Exception as e:
                        print(f"ERROR: dispatch_workflow raised: {e!r}\n{traceback.format_exc()}")
                        self._fail_workflow(workflow_structure, f"Dispatch error: {e!r}")
                        # _fail_workflow removed it from in_process; the next item shifted into this index, so don't advance.


            # Calculate the number of in-process workflows.  If we're not at the maximum number of simultaneous workflows, then we can move some workflows from the 'unstarted' queue to the 'in process' queue.
            while (len(self.workflow_queue_in_process) < MAX_SIMULTANEOUS_WORKFLOWS) and (len(self.workflow_queue_unstarted) > 0):
                # Move the next workflow from the 'unstarted' queue to the 'in process' queue
                next_workflow_structure = self.workflow_queue_unstarted.pop(0)
                self.workflow_queue_in_process.append(next_workflow_structure)
                print("Moving workflow from unstarted queue to in process queue. (Generation ID: " + str(next_workflow_structure.theory_workflow_structure.get("generation_id", "N/A")) + ", User ID: " + str(next_workflow_structure.theory_workflow_structure.get("user_id", "N/A")) + ", Query: '" + str(next_workflow_structure.theory_workflow_structure.get("theory_query", "N/A")) + "'). Current step: '" + str(next_workflow_structure.get_current_step_name()) + "'.")

                # Dispatch this workflow to start working on the first step (isolated so a
                # dispatch failure fails just this workflow instead of killing the worker).
                try:
                    self.dispatch_workflow(next_workflow_structure)
                except Exception as e:
                    print(f"ERROR: dispatch_workflow (start) raised: {e!r}\n{traceback.format_exc()}")
                    self._fail_workflow(next_workflow_structure, f"Dispatch error on start: {e!r}")


            # If there's nothing in the queue for more than 10 seconds, exit.
#             if (len(self.workflow_queue_unstarted) == 0) and (len(self.workflow_queue_in_process) == 0):
#                 # Sleep for 5 seconds to see if any new work arrives
#                 time.sleep(5)
#                 if (len(self.workflow_queue_unstarted) == 0) and (len(self.workflow_queue_in_process) == 0):
#                     print("Theorizer.process_work_monitor(): No work in the queue for 10 seconds -- continuing to wait.")
#                     self.THREAD_ACTIVE = False
#                     # Just hard halt
# #                    exit(1)

        # If we reach here, then the thread is being stopped
        print("Theorizer.process_work_monitor(): Worker thread has completed.")




    #
    #   Loading/Saving
    #

    # # Save/load the contents of the Theorizer and Paperstore to/from a folder
    # def save(self, filename_prefix:str, only_theorystore:bool=False):
    #     print("Theorizer.save(): Saving state... (" + filename_prefix + ")")

    #     # Save the theory store
    #     self.theory_store.save(filename_prefix + ".theorystore.json")

    #     # Save the paperstore
    #     if (not only_theorystore):
    #         self.paperstore.save(filename_prefix + ".paperstore.json")

    #     print(f"Theorizer: Saved state to {filename_prefix}.")


    # def load(self, filename_prefix:str):
    #     print("Theorizer.load(): Loading state... (" + filename_prefix + ")")

    #     # Load the theory store
    #     self.theory_store.load(filename_prefix + ".theorystore.json")

    #     # Load the paperstore
    #     self.paperstore.load(filename_prefix + ".paperstore.json")

    #     print(f"Theorizer: Loaded state from {filename_prefix}.")


    # # Load only a paperstore
    # def load_paperstore(self, filename_paperstore:str):
    #     print("Theorizer.load_paperstore(): Loading paperstore from file... (" + filename_paperstore + ")")
    #     self.paperstore.load(filename_paperstore)
    #     print("Theorizer.load_paperstore(): Paperstore loaded.")



# Quick test of the persistence layer
def quick_test_persistence_layer_save():
    # Make a faux paperstore
    paperstore = PaperStore()

    # Add a paper
    request_id = paperstore.submit_paper(corpus_id="247451124")

    # Wait for the paper to be ready, and then get the paper data.
    MAX_WAIT_TIME_SEC = 60*2
    current_time = 0

    while (current_time < MAX_WAIT_TIME_SEC):
        print("Waiting for request to complete... (Current wait time: " + str(current_time) + " seconds)")
        time.sleep(2)
        current_time += 2

        paper_key = paperstore.is_paper_request_complete(request_id)
        if (paper_key is not None):
            print("Paper request is complete! Paper key: " + str(paper_key))
            paper_data = paperstore.get_paper_by_key(paper_key)
            print("Paper data: " + str(paper_data))
            break

    # Save the paperstore using the Persistence layer
    print("Saving paperstore...")
    paperstore_dict = paperstore.serialize()
    Persistence.store().save_paperstore(generation_id="test-123", data=paperstore_dict)

    print("Saved paperstore...")


def quick_test_persistence_layer_load():
    # Load the paperstore using the Persistence layer
    print("Loading paperstore...")
    paperstore = Persistence.store().get_paperstore(generation_id="test-123")

    if (paperstore is not None):
        # Get the paper from the paperstore
        paper_key = "paper-247451124"
        paper_data = paperstore.get_paper_by_key(paper_key)
        if (paper_data is None):
            print("Error: Paper with key '" + paper_key + "' not found in paperstore.")
            return
        paper_data_str = json.dumps(paper_data, indent=4)
        # Show the first 10 lines
        print("First 10 lines of Paper Data:\n")
        paper_data_lines = paper_data_str.split("\n")
        for line in paper_data_lines[:10]:
            print(line)

    else:
        print("No paperstore found for generation ID 'test-123'.")





#
#   Quick test
#
if __name__ == "__main__":
    loadAPIKeys()

    #quick_test_persistence_layer_save()
    quick_test_persistence_layer_load()
    exit(1)

    # Create a Theorizer instance (new)
    theorizer = Theorizer()

    # Generate a UUID
    #uuid = theorizer.generate_uuid()
    #print(uuid)
    #exit(1)

    # def submit_theory_query(self, generation_id:str, user_id:str,
    #                         theory_query:str, mission_statement:str,
    #                         max_papers_to_retrieve:int=10, knowledge_cutoff_year:int=2050, knowledge_cutoff_month:int=1,
    #                         model_str_primary:str=DEFAULT_MODEL_PRIMARY, model_str_extraction:str=DEFAULT_MODEL_EXTRACTION,
    #                         generation_objective:str=GENERATION_OBJECTIVE_ACCURACY_FOCUS,
    #                         generation_input:str=GENERATION_INPUT_LITERATURE_SUPPORT):


    submission_id_struct1 = theorizer.submit_theory_query(user_id="test_user",
                                    theory_query="Build a theory about how language model agents can best use memory to solve text games.",
                                    mission_statement="I want to understand how language model agents can best use memory to solve text games, so that I can build better agents in the future.",
                                    max_papers_to_retrieve=10, knowledge_cutoff_year=2050, knowledge_cutoff_month=1,
                                    model_str_primary=DEFAULT_MODEL_PRIMARY, model_str_extraction=DEFAULT_MODEL_EXTRACTION,
                                    #model_str_primary="gpt-5-mini", model_str_extraction="gpt-5-mini",  # For speed in debugging
                                    generation_objective=GENERATION_OBJECTIVE_ACCURACY_FOCUS, generation_input=GENERATION_INPUT_LITERATURE_SUPPORT)

    submission_id_struct2 = theorizer.submit_theory_query(user_id="test_user",
                                    theory_query="Build a theory of the key factors that contribute to the development of Alzheimer's disease.",
                                    mission_statement="I want to understand the key factors that contribute to the development of Alzheimer's disease, so that I can contribute to research in this area.",
                                    max_papers_to_retrieve=10, knowledge_cutoff_year=2050, knowledge_cutoff_month=1,
                                    model_str_primary=DEFAULT_MODEL_PRIMARY, model_str_extraction=DEFAULT_MODEL_EXTRACTION,
                                    generation_objective=GENERATION_OBJECTIVE_ACCURACY_FOCUS, generation_input=GENERATION_INPUT_LITERATURE_SUPPORT)
                                    #generation_objective=GENERATION_OBJECTIVE_ACCURACY_FOCUS, generation_input=GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY)

    # theorizer.submit_theory_query(generation_id=3, user_id="test_user",
    #                                 theory_query="Build a theory of the key factors that contribute to how language models perform arithmetic.",
    #                                 mission_statement="",
    #                                 max_papers_to_retrieve=25, knowledge_cutoff_year=2050, knowledge_cutoff_month=1,
    #                                 model_str_primary=DEFAULT_MODEL_PRIMARY, model_str_extraction=DEFAULT_MODEL_EXTRACTION,
    #                                 generation_objective=GENERATION_OBJECTIVE_ACCURACY_FOCUS, generation_input=GENERATION_INPUT_LITERATURE_SUPPORT)
    #                                 #generation_objective=GENERATION_OBJECTIVE_ACCURACY_FOCUS, generation_input=GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY)

    # theorizer.submit_theory_query(generation_id=4, user_id="test_user",
    #                                 theory_query="Build a theory of the how language models understand theory-of-mind",
    #                                 mission_statement="",
    #                                 max_papers_to_retrieve=25, knowledge_cutoff_year=2050, knowledge_cutoff_month=1,
    #                                 model_str_primary=DEFAULT_MODEL_PRIMARY, model_str_extraction=DEFAULT_MODEL_EXTRACTION,
    #                                 generation_objective=GENERATION_OBJECTIVE_ACCURACY_FOCUS, generation_input=GENERATION_INPUT_LITERATURE_SUPPORT)
    #                                 #generation_objective=GENERATION_OBJECTIVE_ACCURACY_FOCUS, generation_input=GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY)

    # theorizer.submit_theory_query(generation_id=5, user_id="test_user",
    #                                 theory_query="Build a theory of how language models can best use memory to solve text games.",
    #                                 mission_statement="",
    #                                 max_papers_to_retrieve=25, knowledge_cutoff_year=2050, knowledge_cutoff_month=1,
    #                                 model_str_primary=DEFAULT_MODEL_PRIMARY, model_str_extraction=DEFAULT_MODEL_EXTRACTION,
    #                                 generation_objective=GENERATION_OBJECTIVE_ACCURACY_FOCUS, generation_input=GENERATION_INPUT_LITERATURE_SUPPORT)
    #                                 #generation_objective=GENERATION_OBJECTIVE_ACCURACY_FOCUS, generation_input=GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY)

    # theorizer.submit_theory_query(generation_id=6, user_id="test_user",
    #                                 theory_query="Build a theory of the key factors that contribute to the development of Alzheimer's disease.",
    #                                 mission_statement="I want to understand the key factors that contribute to the development of Alzheimer's disease, so that I can contribute to research in this area.",
    #                                 max_papers_to_retrieve=25, knowledge_cutoff_year=2050, knowledge_cutoff_month=1,
    #                                 model_str_primary=DEFAULT_MODEL_PRIMARY, model_str_extraction=DEFAULT_MODEL_EXTRACTION,
    #                                 generation_objective=GENERATION_OBJECTIVE_ACCURACY_FOCUS, generation_input=GENERATION_INPUT_LITERATURE_SUPPORT)
    #                                 #generation_objective=GENERATION_OBJECTIVE_ACCURACY_FOCUS, generation_input=GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY)


    # Keep going until there is a signal to end (e.g. CTRL-C)
    try:
        while True:
            time.sleep(5)

            # Test exports
            #def get_workflow_status_by_generation_id(self, generation_id:str):
            #def get_theorystore_by_generation_id(self, generation_id:str):
            #def get_server_status(self):

            id1 = submission_id_struct1["generation_id"]

            try:
                out1 = theorizer.get_workflow_status_by_generation_id(id1)
                filename_out = "debug.workflowstatus.1.json"
                with open(filename_out, "w") as f:
                    json.dump(out1, f, indent=4)
                print(f"Exported workflow status for generation ID 1 to {filename_out}.")

                out2 = theorizer.get_theorystore_by_generation_id(id1)
                filename_out = "debug.theorystore.1.json"
                with open(filename_out, "w") as f:
                    json.dump(out2, f, indent=4)
                print(f"Exported theorystore for generation ID 1 to {filename_out}.")

                out3 = theorizer.get_server_status()
                filename_out = "debug.serverstatus.json"
                with open(filename_out, "w") as f:
                    json.dump(out3, f, indent=4)
                print(f"Exported server status to {filename_out}.")
            except Exception as e:
                print("Error during export: " + str(e))



    except KeyboardInterrupt:
        print("Main thread: Received KeyboardInterrupt. Stopping worker thread...")
        theorizer.stop_thread()
        print("Main thread: Worker thread stopped. Exiting.")