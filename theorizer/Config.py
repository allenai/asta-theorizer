# Config.py
# API keys and library-wide defaults.
#
# The src/ modules read their keys from files in the current working directory, which is awkward
# for a library that gets called from anywhere.  init() finds those files wherever they live and
# promotes them to environment variables, which litellm, the Mistral client and the Semantic
# Scholar client all read as well.

from __future__ import annotations

import json
import os

from . import Legacy
from . import Project


#
#   Defaults
#

# The reasoning-heavy model: query planning, schema generation, variable reports, theories.
DEFAULT_MODEL_PRIMARY = "gpt-6-sol"

# The cheap per-paper model: extraction and variable discovery run once per paper, so cost adds up.
DEFAULT_MODEL_EXTRACTION = "gpt-6-luna"

# The model that repairs an answer whose JSON is malformed (the engine's third repair pass, used by
# theory generation's reflection).  None means the extraction model: the repair is a cheap,
# mechanical call, and the extraction model is one the run already reaches.  The engine on its own
# names gpt-6-luna, which fails behind a proxy or key that does not serve it.
DEFAULT_MODEL_REPAIR = None

# Per-paper work is always parallel.  This is the default width; every function that uses it also
# takes max_workers so a caller can run a 10-paper test or a 1000-paper job from the same call.
DEFAULT_MAX_WORKERS = 25

# Evidence handed to a single LLM call, before subsampling kicks in (counted with the gpt-4
# tokenizer, which runs ~16% above gpt-6's own count).
#
# 750000, to use the context of the default models (gpt-6-sol: 922k input; current Claude: 1M)
# rather than the 100k the paper's experiments used.  Measured on a 100-paper corpus (1,186 records,
# ~820k tokens of evidence): at 100k the theories saw 11% of the records and cited 38 papers, for
# $0.75; at 750k they saw 90% and cited 47 papers, for $5.71 -- most of it the first call writing the
# evidence to the prompt cache, which the reflection calls then read for ~$0.45 each.  A run whose
# evidence already fits under 100k costs the same either way.
#
# A model with a smaller window rejects the prompt (prompt_too_long, which is not billed): pass
# max_data_tokens= on the stage, or set this lower, for such a model.
DEFAULT_MAX_DATA_TOKENS = 750000

# Completion tokens for one paper's call (extraction, variable discovery).  A reasoning model at
# reasoning effort high counts its thinking against this too, and at 32000 it ran out on dense
# papers: on a 100-paper run, 8 calls spent all 32000 on reasoning and wrote nothing, and 11 more
# used over 28000.  The same papers at 64000 finished with 18k-37k.  Billing is for tokens used, not
# for the ceiling, so a higher one costs nothing on a paper that does not need it.  It is capped at
# the model's own output limit where that is known (see cap_output_tokens).
DEFAULT_MAX_TOKENS_PER_PAPER = 64000


def cap_output_tokens(model_str:str, max_tokens:int, show:bool=True):
    # A model rejects a request for more completion tokens than it can write (gpt-4.1: 32768,
    # gpt-4o: 16384), so cap at its published limit.  A model litellm does not know is left alone.
    try:
        import litellm
        name = str(model_str)
        info = None
        for candidate in (name, name.split("/", 1)[-1]):
            try:
                info = litellm.get_model_info(candidate)
                break
            except Exception:
                continue
        limit = (info or {}).get("max_output_tokens")
    except Exception:
        limit = None
    if (isinstance(limit, int)) and (limit > 0) and (max_tokens > limit):
        if (show):
            print("Config.cap_output_tokens(): " + str(model_str) + " writes at most " + str(limit) + " tokens; using that instead of max_tokens=" + str(max_tokens), flush=True)
        return limit
    return max_tokens

# The seed for that subsampling.  The draw uses its own random.Random(seed), never the global
# generator, so a rerun over the same records picks the same evidence and nothing else's random
# numbers change.  generate_theories() and generate_theories_over_types() take evidence_seed= to
# draw a different sample.
DEFAULT_EVIDENCE_SEED = 0

# Context limits are PER MODEL and are not recorded here.  An earlier version of this file carried a
# single "safe" number taken from one model's rejection message and applied it to all of them, which
# was simply wrong -- one model rejected above 272000 input tokens, while another has a far larger
# window.  litellm.get_max_tokens() is not a substitute: it reported 128000 for every model of
# that family, which contradicts both measurements.
#
# So nothing guesses a ceiling.  If a prompt is too long the API says so, and Pipeline.subsample()
# reports the size it produced so the number in the error can be compared against it.

# Seconds before an LLM call is abandoned and retried.  A timed-out call has still been paid for
# and is then retried, so a tight timeout costs money rather than saving it; the retry loop gives
# up after 3 timeouts, which bounds a stuck call at 3x this.
#
# 3600, because theory generation hands its whole evidence budget to one call: measured at ~900k
# tokens of prompt on a 1,103-paper corpus at max_data_tokens=750000, at reasoning effort high,
# with up to 128k tokens of answer to write.  Three such batches ran past 30 minutes and returned
# nothing.  Per-paper calls are a thousandth of that and return in seconds, so a generous ceiling
# costs an ordinary run nothing -- nothing waits on the timeout unless a call is genuinely stuck.
#
# It applies to every engine call that does not set its own limit (Legacy makes it the engine's
# default: planning, reports, schemas, paper metadata) and is passed explicitly to the per-paper
# stages (extraction, variable discovery).  The MAX_GENERATION_TIME_SECONDS
# environment variable overrides it, read at call time; llm_timeout_seconds= on a Pipeline stage
# overrides both for that stage.  Theory generation
# and the evaluations name fixed limits inside src/ (15 or 8 minutes; 300 s): they get the setting
# when it is given, else the longer of their own value and this default (get_generation_time_seconds()).
DEFAULT_LLM_TIMEOUT_SECONDS = 3600

# Set for the length of one stage by llm_timeout(seconds) -- the stage's llm_timeout_seconds -- and
# read by every call made meanwhile, on any thread.
_llm_timeout_override = None


class llm_timeout:
    # with Config.llm_timeout(seconds): ...   every call in the block gets `seconds` (None: no change)
    def __init__(self, seconds:int=None):
        self.seconds = seconds

    def __enter__(self):
        global _llm_timeout_override
        self.saved = _llm_timeout_override
        if (self.seconds is not None):
            _llm_timeout_override = int(self.seconds)
        return self

    def __exit__(self, *exc):
        global _llm_timeout_override
        _llm_timeout_override = self.saved
        return False


def get_generation_time_seconds(fixed_seconds:int):
    # The limit for an engine call that names a fixed one of its own (theory generation, the
    # evaluations): the stage's or the environment's setting when there is one, else the longer of
    # its own value and the library default -- so no call gets less than it had.
    if (_llm_timeout_override is not None) or (os.environ.get("MAX_GENERATION_TIME_SECONDS")):
        return get_llm_timeout_seconds()
    return max(int(fixed_seconds), DEFAULT_LLM_TIMEOUT_SECONDS)


def get_llm_timeout_seconds():
    # The stage's setting, else MAX_GENERATION_TIME_SECONDS (read at call time rather than import
    # time, as get_s2_seconds_between_requests() is), else DEFAULT_LLM_TIMEOUT_SECONDS.
    if (_llm_timeout_override is not None):
        return _llm_timeout_override
    value = os.environ.get("MAX_GENERATION_TIME_SECONDS")
    if (value is None) or (len(str(value).strip()) == 0):
        return DEFAULT_LLM_TIMEOUT_SECONDS

    try:
        return int(float(value))
    except ValueError:
        print("Config.get_llm_timeout_seconds(): WARNING: MAX_GENERATION_TIME_SECONDS is not a number (" + str(value) + "), using the default of " + str(DEFAULT_LLM_TIMEOUT_SECONDS), flush=True)
        return DEFAULT_LLM_TIMEOUT_SECONDS

# Minimum seconds between Semantic Scholar title lookups made by
# PaperImport.import_papers(lookup_semantic_scholar=True).  (fetch_papers() goes through PaperStore,
# which paces its own requests.)  The default is deliberately conservative and safe for a caller
# without an API key.
#
# Set the S2_RATE_LIMIT environment variable to go faster if you have a key -- it is the number of
# seconds between requests, so a SMALLER number is faster:
#
#     export S2_RATE_LIMIT=0.35        # about three requests a second
#
# Measured with a key: 45 sustained requests at 0.35s produced no 429s, and a short burst at 0.2s
# did not either.  Without a key, Semantic Scholar's guidance is one request per second, so leave
# this alone.
DEFAULT_S2_SECONDS_BETWEEN_REQUESTS = 1.1


def get_s2_seconds_between_requests():
    # Read at call time rather than import time, so setting the variable in a notebook or a script
    # before the first lookup still takes effect.
    value = os.environ.get("S2_RATE_LIMIT")
    if (value is None) or (len(str(value).strip()) == 0):
        return DEFAULT_S2_SECONDS_BETWEEN_REQUESTS

    try:
        return float(value)
    except ValueError:
        print("Config.get_s2_seconds_between_requests(): WARNING: S2_RATE_LIMIT is not a number (" + str(value) + "), using the default of " + str(DEFAULT_S2_SECONDS_BETWEEN_REQUESTS), flush=True)
        return DEFAULT_S2_SECONDS_BETWEEN_REQUESTS


#
#   Models
#

def set_models(primary:str=None, extraction:str=None, repair:str=None):
    # Set the default model for the reasoning-heavy stages (planning, variable reports, schemas,
    # theories) and for the per-paper stages (extraction, variable discovery, paper metadata).
    #
    # Every stage also takes model_str, which overrides this for that one call -- so a run can use a
    # cheap model everywhere and an expensive one for theory generation alone:
    #
    #     th.Config.set_models(primary="gpt-6-luna", extraction="gpt-6-luna")
    #     th.Pipeline.generate_theories(records, query, model_str="gpt-6-sol")
    #
    # repair is the JSON-repair model (default: the extraction model; see DEFAULT_MODEL_REPAIR).
    global DEFAULT_MODEL_PRIMARY, DEFAULT_MODEL_EXTRACTION, DEFAULT_MODEL_REPAIR

    if (primary is not None):
        DEFAULT_MODEL_PRIMARY = primary
    if (extraction is not None):
        DEFAULT_MODEL_EXTRACTION = extraction
    if (repair is not None):
        DEFAULT_MODEL_REPAIR = repair

    # Re-route whatever is set now, including a name that was already there: under a proxy every
    # name needs a provider prefix, and the defaults above do not carry one.
    DEFAULT_MODEL_PRIMARY = proxy_model(DEFAULT_MODEL_PRIMARY)
    DEFAULT_MODEL_EXTRACTION = proxy_model(DEFAULT_MODEL_EXTRACTION)
    DEFAULT_MODEL_REPAIR = proxy_model(DEFAULT_MODEL_REPAIR)

    print("Config.set_models(): primary=" + str(DEFAULT_MODEL_PRIMARY) + ", extraction=" + str(DEFAULT_MODEL_EXTRACTION) + ", repair=" + str(get_repair_model()), flush=True)
    return {"primary": DEFAULT_MODEL_PRIMARY, "extraction": DEFAULT_MODEL_EXTRACTION, "repair": get_repair_model()}


# The reasoning effort sent to reasoning models (gpt-5/gpt-6, o-series, claude): "low", "medium",
# "high" or "xhigh", or "none" to send no effort and leave the model at its own default.  The
# engine sent "high" for every call.  One setting for the once-per-run stages and one for the
# per-paper stages (and JSON repair): every call inside extraction or variable discovery, and any
# other call made with the extraction or repair model, gets the extraction effort.  Read from the environment at
# call time: THEORIZER_REASONING_EFFORT and THEORIZER_EXTRACTION_REASONING_EFFORT.
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_REASONING_EFFORT_EXTRACTION = "high"
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def _effort_from(name:str, default:str):
    value = (os.environ.get(name) or "").strip().lower()
    if (len(value) == 0):
        return default
    if (value not in REASONING_EFFORTS):
        print("Config.get_reasoning_effort(): WARNING: " + name + " is not one of " + ", ".join(REASONING_EFFORTS) + " (" + str(value) + "), using " + str(default), flush=True)
        return default
    return value


# Set for the length of a per-paper stage (extraction, variable discovery) by per_paper_stage(), and
# read by every call made meanwhile, on any thread: those calls get the extraction effort whatever
# model they use.
_per_paper_stage = False


class per_paper_stage:
    def __enter__(self):
        global _per_paper_stage
        self.saved = _per_paper_stage
        _per_paper_stage = True
        return self

    def __exit__(self, *exc):
        global _per_paper_stage
        _per_paper_stage = self.saved
        return False


def get_reasoning_effort(model:str=None, per_paper:bool=None):
    # The effort for a call made with `model`: the extraction effort inside a per-paper stage or for
    # the extraction (or repair) model, the primary effort for anything else.  per_paper= decides it
    # outright.  None means send none.
    primary = _effort_from("THEORIZER_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    extraction = _effort_from("THEORIZER_EXTRACTION_REASONING_EFFORT", DEFAULT_REASONING_EFFORT_EXTRACTION)
    if (per_paper is None):
        per_paper_models = {proxy_model(DEFAULT_MODEL_EXTRACTION), DEFAULT_MODEL_EXTRACTION, get_repair_model()}
        per_paper = _per_paper_stage or ((model is not None) and (model in per_paper_models) and (model != DEFAULT_MODEL_PRIMARY))
    effort = extraction if (per_paper) else primary
    return None if (effort == "none") else effort


def get_repair_model():
    # The JSON-repair model, routed like every other name: DEFAULT_MODEL_REPAIR, else the extraction
    # model.  Legacy points the engine's repair pass here.
    return proxy_model(DEFAULT_MODEL_REPAIR or DEFAULT_MODEL_EXTRACTION)


#
#   An OpenAI-compatible proxy (LiteLLM), when one is configured
#
# With LITELLM_BASE_URL and LITELLM_USER_KEY both set, every model call goes through that proxy,
# which needs no provider keys of its own.  With neither set nothing here does anything, and calls
# go to the providers directly with the keys from the environment or the key file.

def proxy_configured():
    return bool(os.environ.get("LITELLM_BASE_URL")) and bool(os.environ.get("LITELLM_USER_KEY"))


def proxy_model(name:str):
    # litellm picks the provider from the SHAPE of the model name, so a bare name or a claude-*
    # one goes to its own provider and ignores api_base entirely.  Prefixing with "openai/" says
    # "speak the OpenAI protocol to whatever api_base points at", which is what a LiteLLM proxy
    # serves.  A name that already carries a prefix is left alone.
    if (not name) or ("/" in str(name)) or (not proxy_configured()):
        return name

    return "openai/" + str(name)


def configure_proxy():
    # Point litellm at the proxy and give it the proxy's key.  Returns the base URL, or None when
    # no proxy is configured.
    base_url = os.environ.get("LITELLM_BASE_URL")
    key = os.environ.get("LITELLM_USER_KEY")
    if (not base_url) or (not key):
        return None

    import litellm
    # api_base on the module is what every call falls back to, whichever provider prefix the model
    # name carries.
    litellm.api_base = base_url
    os.environ["OPENAI_API_KEY"] = key
    os.environ.setdefault("OPENAI_API_BASE", base_url)
    os.environ.setdefault("OPENAI_BASE_URL", base_url)
    return base_url


#
#   Verbose output from src/
#

def set_quiet(quiet:bool=True, log_path:str=None):
    # Send the full ModelResponse dump to a log file (quiet=True) instead of the terminal.
    #
    # This covers ONLY that dump -- the block ExtractionUtils marks "# DEBUG: PRINT FULL RESPONSE",
    # which is many lines per call.  The per-query status lines that tell you a run is progressing
    # (querying model X, prompt tokens, running cost, completed cost) always print, as do errors,
    # warnings and retries.  Suppressing those made a live run indistinguishable from a hung one.
    #
    # Safe to call at any time, including while work is running -- it sets a flag.
    if (log_path is not None):
        Legacy.ExtractionUtils.VERBOSE_LOG_FILENAME = log_path

    Legacy.ExtractionUtils.VERBOSE = (not quiet)
    return Legacy.ExtractionUtils.VERBOSE_LOG_FILENAME


API_KEY_FILENAME = "api_keys.donotcommit.json"
S2_KEY_FILENAME = "s2_key.donotcommit.txt"

# Which key in api_keys.donotcommit.json maps to which environment variable.
KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "togetherai": "TOGETHERAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

_initialized = False

# The values init() itself put into the environment from a key file, by variable.  A second init()
# may replace these (a different key_dir), but never a value that was in the environment before.
_keys_from_file = {}


#
#   Key loading
#
# A key already in the environment wins; a key file only fills in the ones that are missing.  That
# is the usual order for credentials, and it means a key exported for one run is the one used,
# whatever file happens to sit in the working directory.

# Where a key file lives when it is not tied to one project.  A working directory is usually a
# repository, so a key kept there is one `git add -A` from being committed -- its filename is the
# only guard, and only this repository's .gitignore knows about it.  This is the per-user location
# other tools use, outside any checkout, set once and found from anywhere.
USER_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "asta")


def find_key_file(filename:str, search_dir:str=None):
    # Look in the caller-supplied directory first, then the working directory, then the per-user
    # config directory, then the repo, then src/ -- the first hit wins.  The working directory
    # comes first so a project can override the per-user key with one of its own.
    candidates = []
    if (search_dir is not None):
        candidates.append(search_dir)
    candidates.append(os.getcwd())
    candidates.append(USER_CONFIG_DIR)
    candidates.append(Legacy.REPO_ROOT)
    candidates.append(Legacy.SRC_DIR)

    for directory in candidates:
        candidate = os.path.join(directory, filename)
        if (os.path.isfile(candidate)):
            return candidate

    return None


def init(key_dir:str=None, verbose:bool=True, quiet_legacy_output:bool=True, log_path:str=None, project_dir:str=Project.DEFAULT_DIRECTORY, model_primary:str=None, model_extraction:str=None, model_repair:str=None):
    # Load API keys into the environment, and open a directory to save stage output into.  Safe to
    # call more than once.
    #
    # quiet_legacy_output routes the src/ modules' printing to a log file.  Set it False when you
    # are debugging and want to watch the prompts and responses go by.
    #
    # project_dir is where every stage's output gets saved, so that a stage which cost real money is
    # still there tomorrow.  Point it at a directory per investigation ("vip-gain-control/"), or
    # pass None to turn saving off.
    global _initialized

    if (project_dir is not None):
        Project.use(project_dir)
    else:
        Project.close()

    # The engine dumps every prompt and response into prompts/ (it can reach megabytes), the full
    # response dump into the verbose log, and its OCR and Semantic Scholar caches into files.  Keep
    # all of them with the project they belong to (Project.use() points the engine there); with no
    # project they go in the working directory.
    if (Project.DIRECTORY is not None):
        default_log_path = os.path.join(Project.DIRECTORY, "theorizer-verbose.log")
    else:
        default_log_path = "theorizer-verbose.log"
    if (log_path is None):
        log_path = default_log_path

    resolved_log_path = set_quiet(quiet_legacy_output, log_path=log_path)

    # Where each key came from: "environment", "file", or None when there is none
    key_sources = {}

    def from_environment(env_var:str):
        # Set, and not by an earlier init() from a key file
        value = os.environ.get(env_var)
        return bool(value) and (_keys_from_file.get(env_var) != value)

    def fill_from_file(env_var:str, value:str):
        os.environ[env_var] = value
        _keys_from_file[env_var] = value

    api_key_path = find_key_file(API_KEY_FILENAME, key_dir)
    keys = {}
    if (api_key_path is not None):
        with open(api_key_path) as f:
            keys = json.load(f)
    for name, env_var in KEY_ENV_VARS.items():
        if (from_environment(env_var)):
            key_sources[name] = "environment"
        elif (keys.get(name)):
            fill_from_file(env_var, keys[name])
            key_sources[name] = "file"
        elif (os.environ.get(env_var)):
            key_sources[name] = "file"          # left by an earlier init() from a key file
        else:
            key_sources[name] = None

    # Semantic Scholar: SEMANTIC_SCHOLAR_API_KEY (what src/ reads), then S2_API_KEY (another name
    # for it), both from the environment; then the key file
    s2_key_path = find_key_file(S2_KEY_FILENAME, key_dir)
    if (from_environment("SEMANTIC_SCHOLAR_API_KEY")):
        key_sources["semantic_scholar"] = "environment"
    elif (os.environ.get("S2_API_KEY")):
        os.environ["SEMANTIC_SCHOLAR_API_KEY"] = os.environ["S2_API_KEY"]
        _keys_from_file.pop("SEMANTIC_SCHOLAR_API_KEY", None)
        key_sources["semantic_scholar"] = "environment"
    else:
        value = ""
        if (s2_key_path is not None):
            with open(s2_key_path) as f:
                value = f.read().strip()
        if (len(value) > 0):
            fill_from_file("SEMANTIC_SCHOLAR_API_KEY", value)
            key_sources["semantic_scholar"] = "file"
        else:
            key_sources["semantic_scholar"] = "file" if (os.environ.get("SEMANTIC_SCHOLAR_API_KEY")) else None

    # After the key file, so a configured proxy wins over any provider key found on disk.
    proxy_base_url = configure_proxy()

    # A proxy serves its own list of model names, which the defaults above are not on, so the
    # environment says which ones to use.  An explicit argument still wins.
    resolved_primary = model_primary if (model_primary is not None) else os.environ.get("THEORIZER_MODEL_PRIMARY")
    resolved_extraction = model_extraction if (model_extraction is not None) else os.environ.get("THEORIZER_MODEL_EXTRACTION")
    resolved_repair = model_repair if (model_repair is not None) else os.environ.get("THEORIZER_MODEL_REPAIR")
    if (resolved_primary is not None) or (resolved_extraction is not None) or (resolved_repair is not None) or (proxy_base_url is not None):
        set_models(primary=resolved_primary, extraction=resolved_extraction, repair=resolved_repair)

    _initialized = True

    present = {}
    for name, env_var in KEY_ENV_VARS.items():
        present[name] = bool(os.environ.get(env_var))
    present["semantic_scholar"] = bool(os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))

    if (verbose):
        found = []
        for name in sorted(present):
            if (present[name]):
                source = key_sources.get(name)
                if (source == "file"):
                    where = "file " + str(s2_key_path if (name == "semantic_scholar") else api_key_path)
                else:
                    where = "environment"
                found.append(name + " from " + where)
        if (proxy_base_url is not None):
            route = "proxy " + str(proxy_base_url) + " (OPENAI_API_KEY is the proxy key)"
        else:
            route = "the providers directly"
        print("Config.init(): LLM calls go to " + route + "; keys: " + (", ".join(found) if (len(found) > 0) else "none found"), flush=True)

    # Pack output
    packed = {
        "api_key_file": api_key_path,
        "s2_key_file": s2_key_path,
        "keys_present": present,
        "key_sources": key_sources,
        "log_path": resolved_log_path,
        "quiet_legacy_output": quiet_legacy_output,
        "project_dir": Project.DIRECTORY,
    }
    return packed


def ensure_initialized():
    # Called by the wrappers, so a simple script does not have to remember to call init().
    if (not _initialized):
        init(verbose=False)
