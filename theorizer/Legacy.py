# Legacy.py
# Bindings to the Theorizer code in src/.  This module does not implement anything -- it puts src/
# on the path and gives its functions and classes short, stable names for the wrappers to call.
#
# Everything the library does eventually comes through here, so if you want to know what a wrapper
# really runs, look up the name in this file and read that function in src/.

from __future__ import annotations

import os
import sys


#
#   Paths
#

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)

# In a checkout, the engine is the repo's src/ and symbolic-solver/src/.  In an installed wheel the
# build has copied both into theorizer/_engine/ (see pyproject.toml); that copy is used when present.
ENGINE_DIR = os.path.join(PACKAGE_DIR, "_engine")

if (os.path.isdir(os.path.join(ENGINE_DIR, "src"))):
    SRC_DIR = os.path.join(ENGINE_DIR, "src")
    SYMBOLIC_SOLVER_DIR = os.path.join(ENGINE_DIR, "symbolic_solver")
else:
    SRC_DIR = os.path.join(REPO_ROOT, "src")
    SYMBOLIC_SOLVER_DIR = os.path.join(REPO_ROOT, "symbolic-solver", "src")


def add_to_path(path:str):
    # The src/ modules import each other by bare name (e.g. "from PaperStore import *"), so src/
    # itself has to be on sys.path rather than being imported as a package.
    if (os.path.isdir(path)) and (path not in sys.path):
        sys.path.insert(0, path)


add_to_path(SRC_DIR)


import ExtractionUtils as _extraction_utils          # noqa: E402
import MistralOCRStore as _mistral_ocr               # noqa: E402
import PaperStore as _paper_store                    # noqa: E402
import Persistence as _persistence                   # noqa: E402
import SchemaExtractionQueue as _schema_extraction    # noqa: E402
import SemanticScholar as _semantic_scholar          # noqa: E402
import Struct as _struct                             # noqa: E402
import Throttle as _throttle                         # noqa: E402
import TheorizerProcessing as _processing            # noqa: E402
import VariableDiscoveryQueue as _variable_discovery  # noqa: E402


#
#   Where the engine writes
#

def point_engine_at(directory:str=None):
    # Everything the engine writes on its own -- prompt dumps, the OCR cache and the file backend's
    # data/, the Semantic Scholar response cache -- goes into the open project directory, so a run
    # writes nothing beside the installed code or in the working directory.  With no project open
    # (directory None) they go in the working directory, as the engine always did.
    if (directory is not None):
        _extraction_utils.PROMPTS_DIRECTORY = os.path.join(directory, "prompts")
        _persistence.BASE_DIRECTORY = os.path.abspath(directory)
        _semantic_scholar.FILENAME_S2_CACHE = os.path.join(directory, "s2_cache.json")
    else:
        _extraction_utils.PROMPTS_DIRECTORY = "prompts"
        _persistence.BASE_DIRECTORY = None
        _semantic_scholar.FILENAME_S2_CACHE = "s2_cache.json"

    # Caches already loaded belong to the previous location
    _persistence.reset()
    _semantic_scholar.global_s2cache = None


#
#   Data structures (src/Struct.py)
#

TheoryStore = _struct.TheoryStore
Theory = _struct.Theory
ExtractionQuerySchema = _struct.ExtractionQuerySchema
ExtractionQueryResult = _struct.ExtractionQueryResult
TheoryEvaluation = _struct.TheoryEvaluation


#
#   Papers (src/PaperStore.py, src/SemanticScholar.py)
#

PaperStore = _paper_store.PaperStore
PaperStatus = _paper_store.PaperStatus

# MistralOCRStore.process_pdf() takes a URL.  Local files are handled in PaperImport, which uploads
# them to Mistral and OCRs the signed URL.
MistralOCRStore = _mistral_ocr.MistralOCRStore

# SemanticScholar is the client class; find_paper_by_title() is what the paper-import step uses to
# fill in metadata a user-supplied file does not carry.
SemanticScholar = _semantic_scholar.SemanticScholar

# SemanticScholar(api_key=None) is unauthenticated, which is a much stricter rate limit.  The key
# has to be loaded and passed in; Config.init() puts it in the environment, where this finds it.
load_s2_api_key = _semantic_scholar.load_s2_api_key


#
#   Persistence (src/Persistence.py)
#

FilePersistence = _persistence.FilePersistence
PersistenceBackend = _persistence.PersistenceBackend


#
#   LLM helpers (src/ExtractionUtils.py)
#

# The module itself, so Config can set its VERBOSE flag (see Config.set_quiet()).
ExtractionUtils = _extraction_utils

count_tokens = _extraction_utils.countTokens
truncate_to_model_context = _extraction_utils.truncate_to_model_context

_get_llm_response_json = _extraction_utils.getLLMResponseJSON


def _repair_model():
    # The engine's JSON-repair pass asks default_repair_model() for a model when its caller names
    # none (theory generation never does).  Answer with the library's configured model, routed like
    # every other call, instead of the engine's fixed gpt-6-luna.
    from . import Config
    return Config.get_repair_model()


_extraction_utils.default_repair_model = _repair_model


def _reasoning_effort(model:str):
    # The engine sent reasoning_effort="high" on every call to a reasoning model.  Answer with the
    # library's setting for that model (Config.get_reasoning_effort()), "high" unless changed.
    from . import Config
    return Config.get_reasoning_effort(model)


_extraction_utils.default_reasoning_effort = _reasoning_effort


def _generation_time_seconds():
    # The limit for an engine call that sets none of its own -- planning, variable reports, schemas,
    # evaluations' helpers: the library's (Config.get_llm_timeout_seconds(), 3600 s unless
    # MAX_GENERATION_TIME_SECONDS says otherwise) rather than the engine's 300.  Theory generation
    # passes its own fixed limits and is not affected.
    from . import Config
    return Config.get_llm_timeout_seconds()


_extraction_utils.default_generation_time_seconds = _generation_time_seconds


def _fixed_generation_time_seconds(fixed_seconds:int):
    # Theory generation and the evaluations name fixed limits (15 or 8 minutes, 300 s).  A 169k-token
    # generation on a slow model ran past the 15 minutes; give such calls the library's setting
    # instead (never less than their own value unless a setting asks for it).
    from . import Config
    return Config.get_generation_time_seconds(fixed_seconds)


_extraction_utils._generation_time_resolver = _fixed_generation_time_seconds


def get_llm_response_json(*args, **kwargs):
    # getLLMResponseJSON already handles timeouts and retries (a call that runs past
    # max_generation_time_seconds is abandoned and retried, up to MAX_RETRIES).  Its default is 300
    # seconds, which is too short for a large-evidence call, so the library's default is applied
    # here when the caller does not give one.  Note that a timed-out call is still billed, so the
    # timeout wants to be generous rather than tight.
    from . import Config

    if ("max_generation_time_seconds" not in kwargs):
        kwargs["max_generation_time_seconds"] = Config.get_llm_timeout_seconds()

    return _get_llm_response_json(*args, **kwargs)


#
#   Rate limiting (src/Throttle.py)
#

Throttle = _throttle.Throttle


#
#   Stage functions (src/TheorizerProcessing.py)
#
# These are the pieces the pipeline is made of.  Each takes plain arguments and returns a packed
# dict, so each can be called on its own.

plan_query_and_schema = _processing.convert_theory_request_to_query_and_schema
generate_schemas_from_variable_discovery = _processing.generate_schema_from_variable_discovery
generate_symbolic_schema = _processing.generate_schema_from_extraction_results_for_symbolic_solvers
generate_variable_report = _processing.generate_variable_discovery_from_data_report
subsample_records = _processing.consolodate_results_with_subsampling
subsample_records_with_leftover = _processing.consolodate_results_with_subsampling_and_leftover

# Theory generation.  reflection4 is the current two-stage generator; the "nonsafebasin" variant is
# the novelty-focused one; the "llm_baseline_no_evidence" variants generate without evidence, for
# parametric baselines.
build_theory_accuracy_focused = _processing.build_theory_from_results_single_theory_reflection4
build_theory_novelty_focused = _processing.build_theory_from_results_single_theory_reflection4_nonsafebasin
# reflection4a adds a per-sentence `attribution` block to every law -- which evidence each sentence
# came from, and by what inference -- and a second reflection pass over only the evidence the
# theory cites.
build_theory_attribution = _processing.build_theory_from_results_single_theory_reflection4a_attribution
# reflection4b is the batched-evidence variant: it takes the records that did not fit the context
# window as `results_leftover` and reflects over further batches drawn from them.
build_theory_attribution_batched = _processing.build_theory_from_results_single_theory_reflection4b_attribution_batchedevidence
# reflection5 takes a requested inference_type / output_type, and records both in its packed output.
# OUTPUT_TYPES and INFERENCE_TYPES are the two vocabularies it enumerates over.
build_theory_with_types = _processing.build_theory_from_results_single_theory_reflection5_inferencetypes
OUTPUT_TYPES = _processing.OUTPUT_TYPES
INFERENCE_TYPES = _processing.INFERENCE_TYPES

build_theory_parametric_accuracy = _processing.build_theory_from_results_single_theory_reflection3_llm_baseline_no_evidence
build_theory_parametric_novelty = _processing.build_theory_from_results_single_theory_reflection4_llm_baseline_no_evidence_nonsafebasin


#
#   Per-paper work (src/SchemaExtractionQueue.py, src/VariableDiscoveryQueue.py)
#
# These are the single-paper units the engine's work queues call.  The library calls them directly
# and does its own pooling, so that a caller can run one paper or a thousand from the same function.

extract_entities_from_paper = _schema_extraction.SchemaExtractionQueue.extract_entities_from_paper
discover_variables_from_paper = _variable_discovery.VariableDiscoveryQueue.discover_variables_from_paper

SchemaExtractionQueue = _schema_extraction.SchemaExtractionQueue
VariableDiscoveryQueue = _variable_discovery.VariableDiscoveryQueue


#
#   Symbolic solver (symbolic-solver/src/symbolic_solver.py)
#

def load_symbolic_solver():
    # Imported on demand: it pulls in numpy/pandas/scikit-learn and several optional solvers, and a
    # caller who never runs the solver should not pay that import cost.
    add_to_path(SYMBOLIC_SOLVER_DIR)
    import symbolic_solver
    return symbolic_solver


SYMBOLIC_SOLVER_SCRIPT = os.path.join(SYMBOLIC_SOLVER_DIR, "symbolic_solver.py")
