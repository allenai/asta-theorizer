# Offline tests for NoraRetrievalFullText -- no network, no credentials.
#
#   cd tests && python3 -m unittest test_nora_retrieval_fulltext

import os
import sys
import unittest
from contextlib import contextmanager
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import NoraRetrievalFullText as nrft


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeGet:
    """Stands in for requests.get: records each request and returns a canned response."""

    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.calls = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        if self.raises is not None:
            raise self.raises
        return self.response


@contextmanager
def fake_requests(response=None, raises=None):
    """Patch out the module's one network call for the duration of the block."""
    fake = FakeGet(response, raises)
    with mock.patch.object(nrft.requests, "get", fake):
        yield fake


def body(text, oa=None):
    payload = {"paperId": "abc", "body": {"full": {"text": text}}}
    if oa is not None:
        payload["openAccessPdf"] = oa
    return payload


class NoraRetrievalFullTextTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in (
            nrft.ENABLED_ENV_VAR, nrft.ENV_ENV_VAR,
            nrft.BEARER_TOKEN_ENV_VAR, "NORA_RETRIEVAL_BASE_URL_TEMPLATE")}
        for k in self._saved:
            os.environ.pop(k, None)
        os.environ[nrft.BEARER_TOKEN_ENV_VAR] = "test-token"
        # No default env, so every test that expects a request has to name one.
        os.environ[nrft.ENV_ENV_VAR] = "prod"
        os.environ["NORA_RETRIEVAL_BASE_URL_TEMPLATE"] = "https://retrieval.{env}.example.org"

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # -- the gate ---------------------------------------------------------

    def test_disabled_unless_explicitly_on(self):
        os.environ.pop(nrft.ENABLED_ENV_VAR, None)
        self.assertFalse(nrft.is_enabled())
        os.environ[nrft.ENABLED_ENV_VAR] = "True"      # not the literal we accept
        self.assertFalse(nrft.is_enabled())
        os.environ[nrft.ENABLED_ENV_VAR] = "true"
        self.assertTrue(nrft.is_enabled())

    def test_no_token_means_no_request(self):
        os.environ.pop(nrft.BEARER_TOKEN_ENV_VAR)
        with fake_requests(FakeResponse(payload=body("x" * 500))) as http:
            self.assertIsNone(nrft.fetch_fulltext("abc"))
            self.assertEqual(http.calls, [])

    # -- the request we make ----------------------------------------------

    def test_asks_for_olmocr_and_the_flat_text_field_only(self):
        with fake_requests(FakeResponse(payload=body("hello"))) as http:
            nrft.fetch_fulltext("abc", s2_api_key="key-1")
            call = http.calls[0]
        self.assertEqual(call["params"],
                         {"fields": "body.full.text", "source": "olmocr",
                          "include_no_body_reason": "true"})
        self.assertEqual(call["headers"]["Authorization"], "Bearer test-token")
        self.assertEqual(call["headers"]["x-api-key"], "key-1")
        self.assertEqual(
            call["url"],
            "https://retrieval.prod.example.org"
            "/corpus/v1/vespa/publicapi/paper/abc")

    def test_env_selects_the_deployment_and_nothing_else_does(self):
        # NORA_RETRIEVAL_ENV picks which nora-retrieval we talk to; it cannot
        # name a host.
        os.environ[nrft.ENV_ENV_VAR] = "dev"
        with fake_requests(FakeResponse(payload=body("hello"))) as http:
            nrft.fetch_fulltext("abc")
        self.assertEqual(
            http.calls[0]["url"],
            "https://retrieval.dev.example.org"
            "/corpus/v1/vespa/publicapi/paper/abc")

    def test_unset_env_makes_no_request_rather_than_defaulting_to_prod(self):
        os.environ.pop(nrft.ENV_ENV_VAR)
        with fake_requests(FakeResponse(payload=body("hello"))) as http:
            self.assertIsNone(nrft.fetch_fulltext("abc"))
            self.assertEqual(http.calls, [])

    def test_unknown_env_makes_no_request_rather_than_falling_back_to_prod(self):
        for value in ("develop", "eval", "https://example.test", "", "  "):
            os.environ[nrft.ENV_ENV_VAR] = value
            with fake_requests(FakeResponse(payload=body("hello"))) as http:
                self.assertIsNone(nrft.fetch_fulltext("abc"), value)
                self.assertEqual(http.calls, [], value)

    def test_no_api_key_header_when_none_supplied(self):
        with fake_requests(FakeResponse(payload=body("hello"))) as http:
            nrft.fetch_fulltext("abc")
        self.assertNotIn("x-api-key", http.calls[0]["headers"])

    def test_request_carries_the_fixed_timeout(self):
        with fake_requests(FakeResponse(payload=body("hello"))) as http:
            nrft.fetch_fulltext("abc")
        self.assertEqual(http.calls[0]["timeout"], nrft.TIMEOUT_SECONDS)

    # -- what we do with the answer ---------------------------------------

    def test_hit_returns_text(self):
        with fake_requests(FakeResponse(payload=body("full text here"))):
            result = nrft.fetch_fulltext("abc")
        self.assertEqual(result.text, "full text here")

    def test_a_hit_carries_the_text_and_nothing_about_licensing(self):
        # We don't ask for openAccessPdf, and if a deployment volunteers it
        # anyway its license belongs to the cluster's advertised PDF, not to the
        # PDF this text was extracted from -- so nothing here records it.
        oa = {"url": "https://example.test/p.pdf", "status": "GOLD", "license": "CCBY"}
        for payload in (body("full text here", oa=oa), body("full text here")):
            with fake_requests(FakeResponse(payload=payload)):
                result = nrft.fetch_fulltext("abc")
            self.assertEqual(result.text, "full text here", payload)
            self.assertEqual(result._fields, ("text", "no_body_reason"), payload)

    def test_null_body_is_a_miss_not_an_error(self):
        payload = {"paperId": "abc", "body": None}
        with fake_requests(FakeResponse(payload=payload)):
            result = nrft.fetch_fulltext("abc")
        self.assertIsNotNone(result)
        self.assertIsNone(result.text)

    def test_missing_or_blank_text_is_a_miss(self):
        for payload in ({"body": {"full": {}}},
                        {"body": {"full": None}},
                        {"body": {}},
                        body("   "),
                        body(None),
                        {}):
            with fake_requests(FakeResponse(payload=payload)):
                result = nrft.fetch_fulltext("abc")
            self.assertIsNotNone(result, payload)
            self.assertIsNone(result.text, payload)

    def test_a_miss_carries_the_reason_it_stopped_at(self):
        for reason in (nrft.NO_BODY_PAPER_RESTRICTED, nrft.NO_BODY_NO_UNRESTRICTED_FULLTEXT):
            payload = {"paperId": "abc", "body": None, "noBodyReason": reason}
            with fake_requests(FakeResponse(payload=payload)):
                result = nrft.fetch_fulltext("abc")
            self.assertEqual(result.no_body_reason, reason)

    def test_a_deployment_that_sends_no_reason_carries_none(self):
        # The reason is opt-in and a nora-retrieval that predates it just ignores
        # the parameter, so a miss from one carries no reason at all. The caller
        # is what decides what to do with that; here we only pin that we pass on
        # the absence rather than inventing a reason.
        with fake_requests(FakeResponse(payload={"paperId": "abc", "body": None})):
            result = nrft.fetch_fulltext("abc")
        self.assertIsNone(result.no_body_reason)

    def test_a_hit_carries_no_reason(self):
        # A reason says which check a *miss* stopped at, so text served means
        # there is nothing to report -- the text is the answer.
        with fake_requests(FakeResponse(payload=body("full text here"))):
            result = nrft.fetch_fulltext("abc")
        self.assertIsNone(result.no_body_reason)

    def test_error_statuses_are_misses(self):
        for status in (400, 403, 404, 500, 503):
            with fake_requests(FakeResponse(status_code=status, payload={"detail": "nope"})):
                self.assertIsNone(nrft.fetch_fulltext("abc"), status)

    def test_transport_failure_is_a_miss(self):
        with fake_requests(raises=RuntimeError("connection reset")):
            self.assertIsNone(nrft.fetch_fulltext("abc"))

    def test_non_json_response_is_a_miss(self):
        with fake_requests(FakeResponse(payload=None, text="<html>gateway</html>")):
            self.assertIsNone(nrft.fetch_fulltext("abc"))

    def test_empty_paper_id_makes_no_request(self):
        with fake_requests(FakeResponse(payload=body("x"))) as http:
            self.assertIsNone(nrft.fetch_fulltext(""))
            self.assertEqual(http.calls, [])

    def test_log_callback_receives_misses(self):
        lines = []
        with fake_requests(FakeResponse(status_code=404, payload={})):
            nrft.fetch_fulltext("abc", log=lines.append)
        self.assertTrue(any("not found" in line for line in lines), lines)

    def test_log_callback_receives_hits(self):
        lines = []
        with fake_requests(FakeResponse(payload=body("x" * 500))):
            nrft.fetch_fulltext("abc", log=lines.append)
        self.assertTrue(any("500 chars" in line for line in lines), lines)

    def test_log_callback_gets_every_line_and_nothing_is_printed_twice(self):
        # PaperStore passes log_and_print, which prints as well as records, so this
        # module must not print the same line itself.
        import io
        from contextlib import redirect_stdout

        lines = []
        captured = io.StringIO()
        with fake_requests(FakeResponse(payload=body("x" * 500))):
            with redirect_stdout(captured):
                nrft.fetch_fulltext("abc", log=lines.append)
        self.assertEqual(len(lines), 1, lines)
        self.assertEqual(captured.getvalue(), "")

    def test_prints_when_no_log_callback_is_given(self):
        import io
        from contextlib import redirect_stdout

        captured = io.StringIO()
        with fake_requests(FakeResponse(payload=body("x" * 500))):
            with redirect_stdout(captured):
                nrft.fetch_fulltext("abc")
        self.assertIn("500 chars", captured.getvalue())

    def test_log_callback_distinguishes_an_empty_body_from_a_failure(self):
        lines = []
        with fake_requests(FakeResponse(payload={"paperId": "abc", "body": None})):
            self.assertIsNone(nrft.fetch_fulltext("abc", log=lines.append).text)
        self.assertTrue(any("no body.full.text" in line for line in lines), lines)
