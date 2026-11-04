"""Deterministic unit test for the asta-papers license decision on a fetch that returned
no markdown.

asta-papers resolves a paper's license from the bytes it fetched, while `result.markdown` is
only populated when a converter ran - with no MISTRAL_API_KEY set, no PDF fetch produces
markdown at all. `_get_fulltext_from_asta_papers` therefore branches on `result.success`
first and records `llm_usable` (plus the license fields) whenever the fetch succeeded, so
`AstaPapersFullTextResult.paper_restricted()` can answer for those papers instead of
returning None ("we don't know"), which stops the source chain before legacy OCR.

asta-papers, the Mistral SDK and func_timeout are stubbed, so this runs without network
access, an API key, or the full dependency set:

    python3 tests/test_asta_papers_license_capture.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# --- Stub the dependencies PaperStore imports at module scope ------------------------------

def _stub_module(name, **attrs):
    """Install a stub module, leaving a real (or already-stubbed) one alone.

    The guard matters because tests/test_mistral_ocr_error_logging.py stubs `mistralai`
    too: under a whole-directory pytest run both modules share one sys.modules, so
    whichever imports first must leave a stub the other can still use.
    """
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _StubFunctionTimedOut(Exception):
    pass


class _StubMistralError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __str__(self):
        return self.message


class _StubMistral:
    """The shape MistralOCRStore depends on, so this stub works for that test too."""

    def __init__(self, api_key=None):
        self.ocr = types.SimpleNamespace(process=None)


_stub_module(
    "func_timeout",
    func_timeout=lambda *args, **kwargs: None,
    FunctionTimedOut=_StubFunctionTimedOut,
)
_stub_mistral = _stub_module("mistralai", Mistral=_StubMistral)
_stub_mistral.models = _stub_module("mistralai.models", MistralError=_StubMistralError)

import PaperStore  # noqa: E402


# --- Fakes for the asta-papers client and its result ----------------------------------------
# Mirrors the shape `_get_fulltext_from_asta_papers` reads: `success`, `markdown`,
# `may_use_for_tdm`, `license_class`/`source_type` (enum-like, `.value`), `license_url`,
# and `provenance.primary` (enum-like).

class _Enum:
    def __init__(self, value):
        self.value = value


class _FakeResult:
    def __init__(self, success=True, markdown=None, may_use_for_tdm=True,
                 license_class="CCBY", source_type="pdf", channel="unpaywall"):
        self.success = success
        self.markdown = markdown
        self.may_use_for_tdm = may_use_for_tdm
        self.license_class = _Enum(license_class) if license_class else None
        self.license_url = "https://creativecommons.org/licenses/by/4.0/"
        self.source_type = _Enum(source_type) if source_type else None
        self.provenance = types.SimpleNamespace(primary=_Enum(channel) if channel else None)


class _FakeClient:
    def __init__(self, result):
        self._result = result

    def fetch(self, **kwargs):
        return self._result


class _FakeStore:
    """Just the two attributes the method touches, so no PaperStore construction is needed."""

    def __init__(self, result):
        self._result = result
        self.lines = []

    def log_and_print(self, message):
        self.lines.append(message)

    def _get_asta_papers_client(self):
        return _FakeClient(self._result)


def _fetch(result):
    """Drive the method with asta-papers enabled.

    The env var is set here rather than by the caller so the test is correct however it is
    invoked - without it the method takes its "not using asta-papers at all" branch and
    every assertion below reads a bare FullTextResult.
    """
    store = _FakeStore(result)
    previous = os.environ.get("USE_ASTA_PAPERS")
    os.environ["USE_ASTA_PAPERS"] = "true"
    try:
        out = PaperStore.PaperStore._get_fulltext_from_asta_papers(
            store, corpus_id=123, paper_title="A paper", paper_data={})
    finally:
        if previous is None:
            del os.environ["USE_ASTA_PAPERS"]
        else:
            os.environ["USE_ASTA_PAPERS"] = previous
    return out, store.lines


def _check(name, condition):
    if not condition:
        raise AssertionError(name)


def test_success_without_markdown_records_the_license_decision():
    """The case this test exists for: no converter ran, so no markdown - but the fetch
    succeeded, so the TDM decision is real and must not be discarded."""
    out, lines = _fetch(_FakeResult(success=True, markdown=None, may_use_for_tdm=True))
    _check("no fulltext", out.fulltext is None)
    _check("llm_usable recorded", out.llm_usable is True)
    _check("license recorded", out.license_class == "CCBY")
    _check("channel recorded", out.fetch_channel == "unpaywall")
    _check("source type recorded", out.source_type == "pdf")
    _check("not restricted", out.paper_restricted() is False)
    _check("logged as a miss",
           any("asta-papers miss" in line for line in lines))


def test_success_without_markdown_and_tdm_denied_blocks_the_chain():
    out, _ = _fetch(_FakeResult(success=True, markdown=None, may_use_for_tdm=False))
    _check("no fulltext", out.fulltext is None)
    _check("llm_usable recorded", out.llm_usable is False)
    _check("restricted", out.paper_restricted() is True)


def test_success_with_markdown_and_tdm_permitted_serves_the_text():
    out, lines = _fetch(_FakeResult(success=True, markdown="# Body", may_use_for_tdm=True))
    _check("fulltext served", out.fulltext == "# Body")
    _check("found_fulltext", out.found_fulltext() is True)
    _check("not restricted", out.paper_restricted() is False)
    _check("logged as a hit", any("asta-papers hit " in line for line in lines))


def test_success_with_markdown_and_tdm_denied_withholds_the_text():
    """Markdown came back but the licence denies TDM, so the chain gets no text - which
    is what "miss" reports, the same as a paper-restricted answer on the nora path."""
    out, lines = _fetch(_FakeResult(success=True, markdown="# Body", may_use_for_tdm=False))
    _check("text withheld", out.fulltext is None)
    _check("llm_usable recorded", out.llm_usable is False)
    _check("restricted", out.paper_restricted() is True)
    _check("logged as a miss", any("asta-papers miss" in line for line in lines))
    _check("licence decision still on the line",
           any("llm_usable=False" in line for line in lines))


def test_failed_fetch_decides_nothing():
    """No successful fetch means no license was resolved, so there is no decision to
    record and paper_restricted() stays None."""
    out, lines = _fetch(_FakeResult(success=False, markdown=None, may_use_for_tdm=False))
    _check("no fulltext", out.fulltext is None)
    _check("no llm_usable", out.llm_usable is None)
    _check("no license", out.license_class is None)
    _check("no decision", out.paper_restricted() is None)
    _check("logged as a miss", any("asta-papers miss" in line for line in lines))


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{'FAILED' if failures else 'OK'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)
