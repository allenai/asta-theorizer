# TestConfig.py
# Tests for theorizer/Config.py -- where LLM calls are routed, and which model names carry.
#
# With LITELLM_BASE_URL / LITELLM_USER_KEY set, every model call goes through that LiteLLM proxy.
# These check that the proxy wins over a key file when one is configured, that model names get the
# prefix litellm needs to honour api_base, and that a run with no proxy env goes to the providers.
#
#     python tests/TestConfig.py

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


PROXY_URL = "http://litellm.test:4000"
PROXY_KEY = "sk-test-proxy-key"

ENV_KEYS = [
    "LITELLM_BASE_URL", "LITELLM_USER_KEY",
    "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL",
    "THEORIZER_MODEL_PRIMARY", "THEORIZER_MODEL_EXTRACTION",
    "S2_API_KEY", "SEMANTIC_SCHOLAR_API_KEY",
    "ANTHROPIC_API_KEY", "MISTRAL_API_KEY",
]


def clean_env():
    # Start every test from a known environment: no proxy, no model names, no keys.  The saved
    # values are put back by main().
    for key in ENV_KEYS:
        os.environ.pop(key, None)


def init_in_temp_dir(**kwargs):
    # init() opens a project directory; give each test one of its own and throw it away after.
    directory = tempfile.mkdtemp()
    return th.Config.init(project_dir=directory, verbose=False, **kwargs)


#
#   Tests
#

def test_no_proxy_leaves_everything_alone():
    clean_env()
    th.Config.DEFAULT_MODEL_PRIMARY = "gpt-5.2"
    th.Config.DEFAULT_MODEL_EXTRACTION = "gpt-5-mini"

    import litellm
    litellm.api_base = None
    init_in_temp_dir()

    assert (th.Config.proxy_configured() is False), "proxy reported as configured with no env"
    assert (litellm.api_base is None), "api_base was set without a proxy: " + str(litellm.api_base)
    assert (th.Config.DEFAULT_MODEL_PRIMARY == "gpt-5.2"), "a model name was rewritten without a proxy: " + str(th.Config.DEFAULT_MODEL_PRIMARY)
    assert (th.Config.DEFAULT_MODEL_EXTRACTION == "gpt-5-mini"), "a model name was rewritten without a proxy: " + str(th.Config.DEFAULT_MODEL_EXTRACTION)
    print("  test_no_proxy_leaves_everything_alone: OK")


def test_proxy_takes_the_calls_and_the_key():
    clean_env()
    os.environ["LITELLM_BASE_URL"] = PROXY_URL
    os.environ["LITELLM_USER_KEY"] = PROXY_KEY

    import litellm
    litellm.api_base = None
    init_in_temp_dir()

    assert (litellm.api_base == PROXY_URL), "api_base is not the proxy: " + str(litellm.api_base)
    # The repo may carry a key file, so this also checks the proxy beats a provider key on disk.
    assert (os.environ.get("OPENAI_API_KEY") == PROXY_KEY), "OPENAI_API_KEY is not the proxy key"
    assert (os.environ.get("OPENAI_BASE_URL") == PROXY_URL), "OPENAI_BASE_URL is not the proxy"
    print("  test_proxy_takes_the_calls_and_the_key: OK")


def test_proxy_prefixes_bare_model_names():
    clean_env()
    os.environ["LITELLM_BASE_URL"] = PROXY_URL
    os.environ["LITELLM_USER_KEY"] = PROXY_KEY
    os.environ["THEORIZER_MODEL_PRIMARY"] = "gpt-5.6-sol"
    os.environ["THEORIZER_MODEL_EXTRACTION"] = "gpt-5.6-luna"

    init_in_temp_dir()

    # litellm picks the provider from the name's shape, so a bare name ignores api_base entirely.
    assert (th.Config.DEFAULT_MODEL_PRIMARY == "openai/gpt-5.6-sol"), "primary was not routed: " + str(th.Config.DEFAULT_MODEL_PRIMARY)
    assert (th.Config.DEFAULT_MODEL_EXTRACTION == "openai/gpt-5.6-luna"), "extraction was not routed: " + str(th.Config.DEFAULT_MODEL_EXTRACTION)

    # A claude-* name would go to Anthropic and ignore the proxy; it has to be routed too.
    assert (th.Config.proxy_model("claude-opus-5") == "openai/claude-opus-5"), "a claude name was not routed"
    # One that already names a provider is left alone.
    assert (th.Config.proxy_model("openai/gpt-5.6-sol") == "openai/gpt-5.6-sol"), "an already-routed name was rewritten"
    print("  test_proxy_prefixes_bare_model_names: OK")


def test_an_explicit_argument_beats_the_environment():
    clean_env()
    os.environ["LITELLM_BASE_URL"] = PROXY_URL
    os.environ["LITELLM_USER_KEY"] = PROXY_KEY
    os.environ["THEORIZER_MODEL_PRIMARY"] = "gpt-5.6-sol"

    init_in_temp_dir(model_primary="claude-opus-5")

    assert (th.Config.DEFAULT_MODEL_PRIMARY == "openai/claude-opus-5"), "the argument did not win: " + str(th.Config.DEFAULT_MODEL_PRIMARY)
    print("  test_an_explicit_argument_beats_the_environment: OK")


def test_s2_key_is_read_under_either_name():
    # S2_API_KEY is another name for SEMANTIC_SCHOLAR_API_KEY, which src/SemanticScholar.py reads.
    # Point the filename at one that does not exist, so a real key file cannot fill it instead.
    clean_env()
    os.environ["S2_API_KEY"] = "s2-test-key"

    saved_filename = th.Config.S2_KEY_FILENAME
    th.Config.S2_KEY_FILENAME = "s2_key.does-not-exist.txt"
    try:
        init_in_temp_dir()
    finally:
        th.Config.S2_KEY_FILENAME = saved_filename

    assert (os.environ.get("SEMANTIC_SCHOLAR_API_KEY") == "s2-test-key"), "S2_API_KEY was not carried over"
    print("  test_s2_key_is_read_under_either_name: OK")


def write_key_files(directory, keys:dict, s2_key:str=None):
    with open(os.path.join(directory, th.Config.API_KEY_FILENAME), "w") as f:
        json.dump(keys, f)
    if (s2_key is not None):
        with open(os.path.join(directory, th.Config.S2_KEY_FILENAME), "w") as f:
            f.write(s2_key + "\n")


def test_a_key_in_the_environment_beats_the_key_file():
    # The environment wins; the file fills in only the keys that are missing
    clean_env()
    os.environ["OPENAI_API_KEY"] = "env-openai"
    with tempfile.TemporaryDirectory() as key_dir:
        write_key_files(key_dir, {"openai": "file-openai", "anthropic": "file-anthropic"})
        packed = init_in_temp_dir(key_dir=key_dir)
        assert (os.environ.get("OPENAI_API_KEY") == "env-openai"), "the key file overwrote OPENAI_API_KEY from the environment"
        assert (os.environ.get("ANTHROPIC_API_KEY") == "file-anthropic"), "the key file did not fill in a missing key"
        assert (packed["key_sources"]["openai"] == "environment"), "openai's source: " + str(packed["key_sources"]["openai"])
        assert (packed["key_sources"]["anthropic"] == "file"), "anthropic's source: " + str(packed["key_sources"]["anthropic"])

        # A second init() keeps the environment's key, and may refresh the one it took from a file
        write_key_files(key_dir, {"openai": "file-openai-2", "anthropic": "file-anthropic-2"})
        init_in_temp_dir(key_dir=key_dir)
        assert (os.environ.get("OPENAI_API_KEY") == "env-openai"), "a second init() overwrote the environment's key"
        assert (os.environ.get("ANTHROPIC_API_KEY") == "file-anthropic-2"), "a second init() did not refresh a key it had read from a file"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    print("  test_a_key_in_the_environment_beats_the_key_file: OK")


def test_an_s2_key_in_the_environment_beats_the_key_file():
    # Same order for Semantic Scholar: SEMANTIC_SCHOLAR_API_KEY, then S2_API_KEY, then the file
    with tempfile.TemporaryDirectory() as key_dir:
        write_key_files(key_dir, {}, s2_key="file-s2")

        clean_env()
        os.environ["S2_API_KEY"] = "env-s2"
        packed = init_in_temp_dir(key_dir=key_dir)
        assert (os.environ.get("SEMANTIC_SCHOLAR_API_KEY") == "env-s2"), "the key file beat S2_API_KEY"
        assert (packed["key_sources"]["semantic_scholar"] == "environment")
        assert (th.Legacy.load_s2_api_key() == "env-s2"), "the engine's own lookup does not prefer the environment"

        clean_env()
        os.environ["SEMANTIC_SCHOLAR_API_KEY"] = "env-semantic"
        os.environ["S2_API_KEY"] = "env-s2"
        init_in_temp_dir(key_dir=key_dir)
        assert (os.environ.get("SEMANTIC_SCHOLAR_API_KEY") == "env-semantic"), "S2_API_KEY or the file beat SEMANTIC_SCHOLAR_API_KEY"

        clean_env()
        packed = init_in_temp_dir(key_dir=key_dir)
        assert (os.environ.get("SEMANTIC_SCHOLAR_API_KEY") == "file-s2"), "with nothing in the environment the key file was not used"
        assert (packed["key_sources"]["semantic_scholar"] == "file")
    print("  test_an_s2_key_in_the_environment_beats_the_key_file: OK")


def test_init_says_where_each_key_came_from():
    import contextlib
    import io

    clean_env()
    os.environ["OPENAI_API_KEY"] = "env-openai-secret"
    with tempfile.TemporaryDirectory() as key_dir:
        write_key_files(key_dir, {"mistral": "file-mistral-secret"})
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            th.Config.init(project_dir=tempfile.mkdtemp(), key_dir=key_dir, verbose=True)
    text = printed.getvalue()
    assert ("openai from environment" in text), "the message does not say openai came from the environment: " + text
    assert ("mistral from file" in text), "the message does not say mistral came from a file: " + text
    assert ("secret" not in text), "a key value was printed"
    os.environ.pop("MISTRAL_API_KEY", None)
    print("  test_init_says_where_each_key_came_from: OK")


def test_a_key_file_is_found_in_the_user_config_dir():
    # A working directory is usually a repository, so the per-user config directory is where a key
    # can sit without being one `git add -A` from a commit.  It is searched after the working
    # directory, so a project can still override it with a key file of its own.
    import tempfile

    clean_env()
    with tempfile.TemporaryDirectory() as fake_home:
        config_dir = os.path.join(fake_home, ".config", "asta")
        os.makedirs(config_dir)
        with open(os.path.join(config_dir, th.Config.API_KEY_FILENAME), "w") as f:
            f.write('{"openai": "sk-from-user-config"}')

        saved = th.Config.USER_CONFIG_DIR
        th.Config.USER_CONFIG_DIR = config_dir
        try:
            with tempfile.TemporaryDirectory() as empty_cwd:
                cwd = os.getcwd()
                os.chdir(empty_cwd)                      # nothing to find in the working directory
                try:
                    found = th.Config.find_key_file(th.Config.API_KEY_FILENAME)
                finally:
                    os.chdir(cwd)
        finally:
            th.Config.USER_CONFIG_DIR = saved

    assert (found is not None), "the key file in the per-user config directory was not found"
    assert (found.startswith(config_dir)), "found " + str(found) + ", not the one in the config directory"
    print("  test_a_key_file_is_found_in_the_user_config_dir: OK")


def test_logs_and_prompts_go_in_the_project():
    # The engine dumps every prompt and response into prompts/ (megabytes on a real run) and the
    # full response dump into theorizer-verbose.log.  Both belong with the project, not wherever the
    # script happened to be started from; with no project they stay in the working directory.
    import tempfile

    clean_env()
    extraction_utils = th.Legacy.ExtractionUtils

    class Captured(Exception):
        pass

    def fake_completion(**kwargs):
        raise Captured()

    with tempfile.TemporaryDirectory() as project_dir, tempfile.TemporaryDirectory() as cwd_dir:
        cwd = os.getcwd()
        real_completion = extraction_utils.completion
        extraction_utils.completion = fake_completion
        os.chdir(cwd_dir)
        try:
            th.Config.init(project_dir=project_dir, verbose=False, quiet_legacy_output=True)
            extraction_utils.printVerbose("a line for the verbose log")
            try:
                extraction_utils._getLLMResponseJSONMaybeCached("a prompt", "gpt-6-luna", 0.0, 100, False)
            except Captured:
                pass
        finally:
            extraction_utils.completion = real_completion
            os.chdir(cwd)

        assert os.path.isfile(os.path.join(project_dir, "theorizer-verbose.log")), "the verbose log is not in the project directory"
        prompt_files = os.listdir(os.path.join(project_dir, "prompts"))
        assert (len(prompt_files) == 1), "expected one prompt dump in <project>/prompts, got " + str(prompt_files)
        assert (os.listdir(cwd_dir) == []), "files were written to the working directory: " + str(os.listdir(cwd_dir))

        th.Config.init(project_dir=None, verbose=False)
        assert (extraction_utils.PROMPTS_DIRECTORY == "prompts"), "with no project, prompts/ should be in the working directory"
        assert (extraction_utils.VERBOSE_LOG_FILENAME == "theorizer-verbose.log"), "with no project, the log should be in the working directory"

    print("  test_logs_and_prompts_go_in_the_project: OK")


#
#   Runner
#

def main():
    print("TestConfig.py")
    tests = [
        test_no_proxy_leaves_everything_alone,
        test_proxy_takes_the_calls_and_the_key,
        test_proxy_prefixes_bare_model_names,
        test_an_explicit_argument_beats_the_environment,
        test_s2_key_is_read_under_either_name,
        test_a_key_in_the_environment_beats_the_key_file,
        test_an_s2_key_in_the_environment_beats_the_key_file,
        test_init_says_where_each_key_came_from,
        test_a_key_file_is_found_in_the_user_config_dir,
        test_logs_and_prompts_go_in_the_project,
    ]

    saved_env = {key: os.environ.get(key) for key in ENV_KEYS}
    saved_models = (th.Config.DEFAULT_MODEL_PRIMARY, th.Config.DEFAULT_MODEL_EXTRACTION)
    try:
        for test in tests:
            th.Config.DEFAULT_MODEL_PRIMARY, th.Config.DEFAULT_MODEL_EXTRACTION = saved_models
            test()
    finally:
        th.Config.DEFAULT_MODEL_PRIMARY, th.Config.DEFAULT_MODEL_EXTRACTION = saved_models
        for key, value in saved_env.items():
            if (value is None):
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        th.Project.close()

    print("TestConfig.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
