# TestModels.py
# Tests for which model each call goes to: the library's defaults, the JSON-repair model, and the
# names they carry behind a LiteLLM proxy.
#
# The engine's JSON-repair pass (used by theory generation's reflection) asked a fixed gpt-6-luna,
# whatever the run's models were; behind a proxy that serves no such name the repair failed, and the
# theory it would have saved was lost.  It now asks the run's repair model, which defaults to the
# extraction model and is routed like every other call.
#
# Runs offline.  litellm's own mock_response stands in for the provider: no API keys, no cost.
#
#     python tests/TestModels.py

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


PROXY_URL = "http://litellm.test:4000"
PROXY_KEY = "sk-test-proxy-key"

ENV_KEYS = [
    "LITELLM_BASE_URL", "LITELLM_USER_KEY",
    "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY",
    "THEORIZER_MODEL_PRIMARY", "THEORIZER_MODEL_EXTRACTION", "THEORIZER_MODEL_REPAIR",
    "THEORIZER_MODEL", "THEORIZER_EXTRACTION_MODEL", "THEORIZER_REPAIR_MODEL",
    "THEORIZER_REASONING_EFFORT", "THEORIZER_EXTRACTION_REASONING_EFFORT",
]


def clean_env():
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    import litellm
    litellm.api_base = None
    th.Config.DEFAULT_MODEL_PRIMARY = "gpt-6-sol"
    th.Config.DEFAULT_MODEL_EXTRACTION = "gpt-6-luna"
    th.Config.DEFAULT_MODEL_REPAIR = None


def use_proxy():
    os.environ["LITELLM_BASE_URL"] = PROXY_URL
    os.environ["LITELLM_USER_KEY"] = PROXY_KEY


def init_in(directory:str, **kwargs):
    return th.Config.init(project_dir=directory, verbose=False, **kwargs)


#
#   A fake provider: records every completion() the engine makes and answers from a script
#

class RecordingCompletion:
    def __init__(self, answers:list):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, **kwargs):
        import litellm
        self.calls.append({k: v for k, v in kwargs.items() if (k != "messages")})
        answer = self.answers.pop(0) if (len(self.answers) > 0) else "```json\n{}\n```"
        return litellm.completion(**kwargs, mock_response=answer)


class EngineFakes:
    # Install the recording completion, and (optionally) make the library-tier JSON repair give up so
    # the LLM tier runs.  The engine writes prompt dumps to ./prompts, so run from a scratch directory.
    def __init__(self, completion, library_repair_fails:bool=False):
        self.completion = completion
        self.library_repair_fails = library_repair_fails

    def __enter__(self):
        utils = th.Legacy.ExtractionUtils
        self.saved = (utils.completion, utils.jsonRepairWithLibrary, os.getcwd())
        self.directory = tempfile.mkdtemp()
        os.chdir(self.directory)
        utils.completion = self.completion
        if (self.library_repair_fails):
            utils.jsonRepairWithLibrary = lambda jsonStr: None
        return self.completion

    def __exit__(self, *exc):
        utils = th.Legacy.ExtractionUtils
        utils.completion, utils.jsonRepairWithLibrary, cwd = self.saved
        os.chdir(cwd)
        shutil.rmtree(self.directory, ignore_errors=True)
        return False


#
#   Tests: defaults
#

def test_the_library_default_models(directory):
    clean_env()
    init_in(directory)
    assert (th.Config.DEFAULT_MODEL_PRIMARY == "gpt-6-sol"), th.Config.DEFAULT_MODEL_PRIMARY
    assert (th.Config.DEFAULT_MODEL_EXTRACTION == "gpt-6-luna"), th.Config.DEFAULT_MODEL_EXTRACTION


#
#   Tests: the repair model
#

def test_the_repair_model_follows_the_extraction_model(directory):
    clean_env()
    init_in(directory, model_extraction="cheap-model")
    assert (th.Config.get_repair_model() == "cheap-model"), th.Config.get_repair_model()
    assert (th.Legacy.ExtractionUtils.default_repair_model() == "cheap-model"), "the engine's repair pass does not follow the library: " + str(th.Legacy.ExtractionUtils.default_repair_model())


def test_the_repair_model_is_routed_through_the_proxy(directory):
    clean_env()
    use_proxy()
    init_in(directory, model_extraction="gpt-6-luna")
    assert (th.Config.get_repair_model() == "openai/gpt-6-luna"), "the repair model was not routed: " + str(th.Config.get_repair_model())


def test_an_explicit_repair_model_wins(directory):
    clean_env()
    os.environ["THEORIZER_MODEL_REPAIR"] = "env-repair-model"
    init_in(directory, model_extraction="cheap-model")
    assert (th.Config.get_repair_model() == "env-repair-model"), th.Config.get_repair_model()

    init_in(directory, model_repair="arg-repair-model")
    assert (th.Config.get_repair_model() == "arg-repair-model"), "the argument should beat the environment: " + str(th.Config.get_repair_model())


def test_the_engine_repair_pass_asks_the_configured_model(directory):
    # A malformed answer, a library repair that gives up, and an LLM repair: the repair call must go
    # to the run's repair model through the proxy, not to a fixed gpt-6-luna.
    clean_env()
    use_proxy()
    init_in(directory, model_primary="big-model", model_extraction="small-model")
    broken = "```json\n{\"finding\": \"the gain rises with VIP activity\", \"n\": 3,,}\n```"
    fixed = "```json\n{\"finding\": \"the gain rises with VIP activity\", \"n\": 3}\n```"
    fake = RecordingCompletion([broken, fixed])
    with EngineFakes(fake, library_repair_fails=True):
        parsed, text, cost = th.Legacy.ExtractionUtils.getLLMResponseJSON("a prompt", "openai/big-model", jsonOut=False, repairJSON=True)

    models = [call["model"] for call in fake.calls]
    assert (models == ["openai/big-model", "openai/small-model"]), "the repair went to the wrong model: " + str(models)
    assert (parsed == {"finding": "the gain rises with VIP activity", "n": 3}), parsed


#
#   Tests: the reasoning effort
#

def efforts_sent(primary:str, extraction:str):
    # Make one call with the primary model and one with the extraction model; return the
    # reasoning_effort each carried (None when none was sent)
    fake = RecordingCompletion(["```json\n{\"a\": 1}\n```", "```json\n{\"a\": 1}\n```"])
    with EngineFakes(fake):
        th.Legacy.ExtractionUtils.getLLMResponseJSON("a prompt", primary, jsonOut=False)
        th.Legacy.ExtractionUtils.getLLMResponseJSON("a prompt", extraction, jsonOut=False)
    return [call.get("reasoning_effort") for call in fake.calls]


def test_the_reasoning_effort_is_high_unless_set(directory):
    clean_env()
    use_proxy()
    init_in(directory)
    assert (efforts_sent("openai/gpt-6-sol", "openai/gpt-6-luna") == ["high", "high"]), "the default effort changed"


def test_the_extraction_effort_is_its_own_setting(directory):
    clean_env()
    use_proxy()
    init_in(directory)
    os.environ["THEORIZER_EXTRACTION_REASONING_EFFORT"] = "low"
    assert (efforts_sent("openai/gpt-6-sol", "openai/gpt-6-luna") == ["high", "low"]), "the extraction model should get the extraction effort"

    os.environ["THEORIZER_REASONING_EFFORT"] = "medium"
    os.environ["THEORIZER_EXTRACTION_REASONING_EFFORT"] = "none"
    assert (efforts_sent("openai/gpt-6-sol", "openai/gpt-6-luna") == ["medium", None]), "none should send no reasoning_effort"

    # One model for both roles: a per-paper stage still gets the extraction effort
    os.environ["THEORIZER_REASONING_EFFORT"] = "high"
    os.environ["THEORIZER_EXTRACTION_REASONING_EFFORT"] = "low"
    th.Config.set_models(primary="gpt-6-luna", extraction="gpt-6-luna")
    assert (efforts_sent("openai/gpt-6-luna", "openai/gpt-6-other") == ["high", "high"]), "outside a per-paper stage the primary effort applies"
    with th.Config.per_paper_stage():
        assert (efforts_sent("openai/gpt-6-luna", "openai/gpt-6-other") == ["low", "low"]), "inside a per-paper stage every call should get the extraction effort"
    assert (th.Config.get_reasoning_effort("openai/gpt-6-luna", per_paper=True) == "low")

    os.environ["THEORIZER_REASONING_EFFORT"] = "very high"
    assert (th.Config.get_reasoning_effort("openai/gpt-6-sol") == "high"), "an unknown effort should fall back to the default"


def main():
    print("TestModels.py")
    tests = [
        test_the_library_default_models,
        test_the_repair_model_follows_the_extraction_model,
        test_the_repair_model_is_routed_through_the_proxy,
        test_an_explicit_repair_model_wins,
        test_the_engine_repair_pass_asks_the_configured_model,
        test_the_reasoning_effort_is_high_unless_set,
        test_the_extraction_effort_is_its_own_setting,
    ]

    saved_env = {key: os.environ.get(key) for key in ENV_KEYS}
    saved_models = (th.Config.DEFAULT_MODEL_PRIMARY, th.Config.DEFAULT_MODEL_EXTRACTION, th.Config.DEFAULT_MODEL_REPAIR)
    try:
        for test in tests:
            directory = tempfile.mkdtemp()
            try:
                test(directory)
                print("  " + test.__name__ + ": OK")
            finally:
                shutil.rmtree(directory, ignore_errors=True)
    finally:
        th.Config.DEFAULT_MODEL_PRIMARY, th.Config.DEFAULT_MODEL_EXTRACTION, th.Config.DEFAULT_MODEL_REPAIR = saved_models
        import litellm
        litellm.api_base = None
        for key, value in saved_env.items():
            if (value is None):
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("TestModels.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
