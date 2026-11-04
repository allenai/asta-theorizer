# NoraRetrievalFullText.py
# olmOCR full text for a paper, read from the nora-retrieval API.
#
# The text comes from nora-retrieval's public-API full-text endpoint, so this
# module follows the same logic that endpoint applies internally.
#
#     GET {base}/corpus/v1/vespa/publicapi/paper/{paper_id}
#         ?fields=body.full.text&source=olmocr
#
#     -> {"paperId": ...,
#         "body": {"full": {"text": "..."}}}     # or "body": null on a miss
#
# `source=olmocr` is a deliberate choice of extractor, not a fallback: the
# endpoint consults exactly the source you name, so a hit is always olmOCR text
# and a miss is a miss, never silently something else. `source=vespa` (the
# endpoint's default) is the older PES2O-style body and is *not* what this
# module asks for.
#
# A miss is the normal case for now and costs one round trip. Everything here
# returns None on a miss rather than raising, so the caller's next full-text
# source runs unchanged.

import os
from typing import NamedTuple

import requests


class FullText(NamedTuple):
    """What one answered lookup gives us: the text, or why there isn't any."""

    text: str | None
    no_body_reason: str | None  # noBodyReason when there is no text; None on a hit.


# The `source` the endpoint knows this extractor by.
BODY_SOURCE = "olmocr"

BODY_FIELDS = "body.full.text"

# Opt-in diagnostic on the same request: when the response carries no body, which
# check it stopped at. Without it, `body: null` is one shape for two situations we
# treat differently -- a paper stopped before the usable-pdf check, and a paper
# that got past it but has no olmOCR text yet. A deployment that predates the
# parameter ignores it and simply returns no reason.
NO_BODY_REASON_PARAM = "include_no_body_reason"
NO_BODY_REASON_KEY = "noBodyReason"

# One of the options for noBodyReason.
NO_BODY_PAPER_RESTRICTED = "paper-restricted"

# One of the options for noBodyReason.
NO_BODY_NO_UNRESTRICTED_FULLTEXT = "no-unrestricted-fulltext"

PAPER_PATH = "/corpus/v1/vespa/publicapi/paper/{paper_id}"

BASE_URL_TEMPLATE = ""  # Set NORA_RETRIEVAL_BASE_URL_TEMPLATE to enable retrieval.

# There is no default env. A deployment that hasn't said which nora-retrieval it
# talks to doesn't use this path at all -- defaulting would mean an unconfigured
# (or misconfigured) Theorizer reading production text by omission.
KNOWN_ENVS = ("prod", "dev")

# Off unless a deployment turns it on, matching the USE_ASTA_PAPERS convention
# in PaperStore: this path needs a token configured alongside it, and an
# unconfigured deployment should not spend a round trip per paper to discover
# that.
ENABLED_ENV_VAR = "USE_NORA_RETRIEVAL_FULLTEXT"
ENV_ENV_VAR = "NORA_RETRIEVAL_ENV"
BEARER_TOKEN_ENV_VAR = "NORA_RETRIEVAL_BEARER_TOKEN"

TIMEOUT_SECONDS = 20.0


def is_enabled() -> bool:
    return os.environ.get(ENABLED_ENV_VAR) == "true"


def base_url() -> "str | None":
    """The deployment we talk to, or None unless NORA_RETRIEVAL_ENV names one we know.

    Unset and unrecognized are both None, and neither falls back to prod.
    """
    env = os.environ.get(ENV_ENV_VAR, "").strip().lower()
    if env not in KNOWN_ENVS:
        return None
    template = os.environ.get("NORA_RETRIEVAL_BASE_URL_TEMPLATE", BASE_URL_TEMPLATE).strip()
    return template.format(env=env) if template else None


def _extract_text(payload) -> str | None:
    """Pull body.full.text out of the response, tolerating every level being null."""
    if not isinstance(payload, dict):
        return None
    body = payload.get("body")
    if not isinstance(body, dict):
        # "body": null is the endpoint's way of saying "no full text for this
        # paper from this source" -- a miss, not an error.
        return None
    full = body.get("full")
    if not isinstance(full, dict):
        return None
    text = full.get("text")
    if isinstance(text, str) and text.strip():
        return text
    return None


def fetch_fulltext(paper_id: str, s2_api_key: str = None, log=None) -> "FullText | None":
    """Return what the endpoint says about olmOCR full text for `paper_id`.

    A `FullText` whenever the request was answered: `.text` is the full text on a
    hit and None on a miss, and `.no_body_reason` says which check a miss stopped
    at. None when the request was never answered at all -- the source is off or
    unconfigured, or the call failed -- so that case is "we learned nothing",
    not "there is no text".

    `paper_id` is any id the S2 public API accepts -- an S2 paper sha, or a
    prefixed form such as `CorpusId:12345`.

    `s2_api_key` is forwarded as `x-api-key`; the endpoint passes it through to
    the public API, so supplying it means the lookup runs against our own rate
    limit rather than the shared anonymous one.

    Never raises for an expected outcome (miss, timeout, auth, disabled source):
    the caller's next full-text source should still get its turn.
    """

    def _log(message):
        line = f"NoraRetrievalFullText: {message}"
        # `log` is expected to both record and print (PaperStore.log_and_print),
        # so printing here as well would double every line.
        if log is not None:
            log(line)
        else:
            print(line)

    if not paper_id:
        return None

    token = os.environ.get(BEARER_TOKEN_ENV_VAR)
    if not token:
        _log(f"{BEARER_TOKEN_ENV_VAR} is not set; skipping (the public deployment requires it)")
        return None

    base = base_url()
    if base is None:
        configured = os.environ.get(ENV_ENV_VAR)
        why = ("is not set" if configured is None
               else f"is {configured!r}, not one of {', '.join(KNOWN_ENVS)}")
        _log(f"{ENV_ENV_VAR} {why}; skipping rather than guessing a deployment")
        return None

    url = base + PAPER_PATH.format(paper_id=paper_id)
    headers = {"Authorization": f"Bearer {token}"}
    if s2_api_key:
        headers["x-api-key"] = s2_api_key
    params = {
        "fields": BODY_FIELDS,
        "source": BODY_SOURCE,
        NO_BODY_REASON_PARAM: "true",
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=TIMEOUT_SECONDS)
    except Exception as e:
        _log(f"request for paper_id={paper_id} raised {type(e).__name__}: {e}")
        return None

    status = getattr(response, "status_code", None)
    if status == 404:
        _log(f"paper_id={paper_id} not found")
        return None
    if status == 400:
        # The one 400 worth distinguishing: the deployment has the olmOCR path
        # switched off (OLMOCR_FULLTEXT_ENABLED=false). Every paper will miss
        # until that changes, so say which it is rather than logging a bare 400.
        _log(f"paper_id={paper_id}: 400 from source={BODY_SOURCE} "
             f"(is it enabled in this deployment?): {getattr(response, 'text', '')[:200]}")
        return None
    if status is None or status >= 300:
        _log(f"paper_id={paper_id}: HTTP {status}: {getattr(response, 'text', '')[:200]}")
        return None

    try:
        payload = response.json()
    except Exception as e:
        _log(f"paper_id={paper_id}: response was not JSON ({type(e).__name__}: {e})")
        return None

    reason = payload.get(NO_BODY_REASON_KEY) if isinstance(payload, dict) else None

    text = _extract_text(payload)
    if text is None:
        # A 200 carrying "body": null -- the request was fine and this source simply
        # has no text for this paper. Worth saying apart from the failures above,
        # and worth returning rather than collapsing to None: the reason is the
        # thing the caller can act on, and only an answered request carries one.
        _log(f"paper_id={paper_id}: 200 but no body.full.text from source={BODY_SOURCE} "
             f"{NO_BODY_REASON_KEY}={reason}")
        return FullText(text=None, no_body_reason=reason)

    _log(f"paper_id={paper_id}: {len(text)} chars from source={BODY_SOURCE}")
    return FullText(text=text, no_body_reason=None)
