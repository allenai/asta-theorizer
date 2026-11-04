"""Deterministic unit test for Mistral OCR failure classification and logging.

Covers the structured `MISTRAL_OCR_ERROR` / `MISTRAL_OCR_OK` lines added so that a billing
refusal (402), a rate limit, a bad PDF and a Mistral-side outage are distinguishable in the
logs instead of all collapsing to one generic "Error processing PDF" line, which is what let a
whole run of 402s go unnoticed (allenai/nora-issues#3197).

The Mistral SDK is stubbed, so this runs without network access or an API key:

    .venv/bin/python tests/test_mistral_ocr_error_logging.py
"""

import io
import logging
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# --- Stub the Mistral SDK -------------------------------------------------------------------
# Mirrors the real shape MistralOCRStore depends on: `mistralai.Mistral` and the
# `mistralai.models.MistralError` base class carrying `status_code`.

class _StubMistralError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __str__(self):
        return self.message


class _StubMistral:
    def __init__(self, api_key=None):
        self.ocr = types.SimpleNamespace(process=None)


if "mistralai" not in sys.modules:
    _stub_root = types.ModuleType("mistralai")
    _stub_root.Mistral = _StubMistral
    _stub_models = types.ModuleType("mistralai.models")
    _stub_models.MistralError = _StubMistralError
    _stub_root.models = _stub_models
    sys.modules["mistralai"] = _stub_root
    sys.modules["mistralai.models"] = _stub_models

import MistralOCRStore as M  # noqa: E402


def _store():
    """An OCR store with the network, throttle and cache stubbed out."""
    store = M.MistralOCRStore(api_key="test-key", generation_id="gen-20260831-14f9b043")
    store._ocr_throttle = types.SimpleNamespace(wait=lambda url: None)
    store.load_cache = lambda url: None
    store.save_cache = lambda url, response_json: M.MistralOCRStore.get_markdown_from_ocr_response(store, response_json)
    return store


def _run_process_pdf(store, url="https://arxiv.org/pdf/2001.02007.pdf"):
    captured = io.StringIO()
    handler = logging.StreamHandler(captured)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = M.logger.level
    M.logger.setLevel(logging.INFO)
    M.logger.addHandler(handler)
    try:
        result = store.process_pdf(url)
    finally:
        M.logger.removeHandler(handler)
        M.logger.setLevel(previous_level)
    return result, captured.getvalue()


def test_extract_http_status_prefers_the_sdk_attribute():
    assert M.extract_http_status(_StubMistralError("boom", 402)) == 402


def test_extract_http_status_falls_back_to_the_message():
    # A wrapped/re-raised SDK error keeps the status only in its rendered message.
    wrapped = RuntimeError('API error occurred: Status 429. Body: {"detail":"rate limited"}')
    assert M.extract_http_status(wrapped) == 429


def test_extract_http_status_is_none_without_a_response():
    assert M.extract_http_status(OSError("connection reset by peer")) is None


def test_classify_ocr_error_separates_the_actionable_failures():
    cases = {
        401: "auth_or_billing",
        402: "auth_or_billing",   # the failure mode from nora-issues#3197
        403: "auth_or_billing",
        429: "rate_limit",
        404: "client_error",
        413: "client_error",
        500: "server_error",
        503: "server_error",
    }
    for status_code, expected_category in cases.items():
        error = _StubMistralError("boom", status_code)
        status, category = M.classify_ocr_error(status_code, error)
        assert status == str(status_code), f"{status_code}: got status {status}"
        assert category == expected_category, f"{status_code}: got category {category}"


def test_classify_ocr_error_labels_transport_failures_as_network():
    class ConnectTimeout(Exception):
        pass

    status, category = M.classify_ocr_error(None, ConnectTimeout("timed out"))
    assert (status, category) == ("none", "network")


def test_classify_ocr_error_labels_anything_else_as_unexpected():
    status, category = M.classify_ocr_error(None, ValueError("not an HTTP failure at all"))
    assert (status, category) == ("none", "unexpected")


def test_process_pdf_logs_a_greppable_line_for_a_billing_refusal():
    store = _store()

    def _refuse(**kwargs):
        raise _StubMistralError(
            'API error occurred: Status 402. Body: {"detail":"Check your subscription on '
            'https://admin.mistral.ai/subscription"}',
            402,
        )

    store.client.ocr.process = _refuse
    result, output = _run_process_pdf(store)

    assert result is None, "a refused OCR call must still return None to the caller"
    line = next((l for l in output.splitlines() if l.startswith(M.LOG_TOKEN_OCR_ERROR)), None)
    assert line is not None, f"no {M.LOG_TOKEN_OCR_ERROR} line emitted:\n{output}"
    assert "status=402" in line, line
    assert "category=auth_or_billing" in line, line
    assert "generation_id=gen-20260831-14f9b043" in line, line
    assert "url=https://arxiv.org/pdf/2001.02007.pdf" in line, line
    assert "Check your subscription" in line, line


def test_process_pdf_logs_the_same_shape_for_a_non_sdk_failure():
    # The generic catch stays, but a failure mode we didn't anticipate is now visible too.
    store = _store()

    def _explode(**kwargs):
        raise ValueError("unexpected response shape")

    store.client.ocr.process = _explode
    result, output = _run_process_pdf(store)

    assert result is None
    line = next((l for l in output.splitlines() if l.startswith(M.LOG_TOKEN_OCR_ERROR)), None)
    assert line is not None, f"no {M.LOG_TOKEN_OCR_ERROR} line emitted:\n{output}"
    assert "status=none" in line, line
    assert "category=unexpected" in line, line
    assert "error_type=ValueError" in line, line


def test_process_pdf_logs_a_success_line_so_the_error_rate_has_a_denominator():
    store = _store()
    page = {"markdown": "# Title\n\nSome extracted body text."}
    store.client.ocr.process = lambda **kwargs: types.SimpleNamespace(dict=lambda: {"pages": [page]})

    result, output = _run_process_pdf(store)

    assert result == "# Title\n\nSome extracted body text."
    line = next((l for l in output.splitlines() if l.startswith(M.LOG_TOKEN_OCR_OK)), None)
    assert line is not None, f"no {M.LOG_TOKEN_OCR_OK} line emitted:\n{output}"
    assert "pages=1" in line, line
    assert "generation_id=gen-20260831-14f9b043" in line, line
    assert M.LOG_TOKEN_OCR_ERROR not in output, "a successful call must not log an error line"


def test_log_detail_is_single_line_and_bounded():
    detail = M._log_detail(RuntimeError("a\nb\n" + "x" * 5000))
    assert "\n" not in detail, "a multi-line body would break one-line log parsing"
    assert len(detail) <= M._MAX_DETAIL_LEN + len("...(truncated)")


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
