# (COMPLETED PASS)

# MistralOCRStore.py
# This is used to extract the full-text of scientific paper PDFs using the Mistral OCR API, with caching.


import os
import json
import logging
import re
import time

from mistralai import Mistral
from mistralai.models import MistralError

import Persistence
from Throttle import Throttle

# Thread lock
from threading import Lock
mistral_cache_thread_lock = Lock()

logger = logging.getLogger(__name__)


# Greppable log tokens. A log-based metric and its alert policy key off these lines
# (allenai/nora-issues#3197), so both the token and the `key=value` shape are
# load-bearing: don't reword them without updating the metric filters.
LOG_TOKEN_OCR_ERROR = "MISTRAL_OCR_ERROR"
LOG_TOKEN_OCR_OK = "MISTRAL_OCR_OK"

# Substrings of exception class names that mean "the call never got an HTTP response".
# httpx transport errors reach us unwrapped, and matching by name avoids importing httpx
# just to classify them.
_NETWORK_ERROR_NAME_TOKENS = ("Timeout", "Connect", "Network", "Protocol", "Proxy")

# Longest error body echoed into a log line, so one huge HTML error page can't flood the logs.
_MAX_DETAIL_LEN = 300


def extract_http_status(exception:Exception):
    """Best-effort HTTP status for a failed OCR call, or None if there wasn't a response.

    `MistralError` carries `status_code` directly. Anything else may still be a wrapped
    SDK error whose message ends up as "...: Status 402. Body: ...", so fall back to that.
    """
    status_code = getattr(exception, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    match = re.search(r"\bStatus (\d{3})\b", str(exception))
    if match is not None:
        return int(match.group(1))

    return None


def classify_ocr_error(status_code, exception:Exception):
    """Map a failed OCR call onto (status, category) strings for alerting.

    `status` is the HTTP status, or "none" when the call never reached a response.
    Categories:
      auth_or_billing - 401/402/403: credentials or subscription. Always actionable.
      rate_limit      - 429: alert on a sustained rate, not a single occurrence.
      client_error    - other 4xx: usually the specific PDF (unfetchable, too large, ...).
      server_error    - 5xx: Mistral-side; alert on a sustained rate.
      network         - no response at all (timeout, DNS, connection reset).
      unexpected      - everything else, including bugs on our side of the call.
    """
    if (status_code is None):
        exception_name = type(exception).__name__
        if any(token in exception_name for token in _NETWORK_ERROR_NAME_TOKENS):
            return "none", "network"
        return "none", "unexpected"

    if (status_code in (401, 402, 403)):
        return str(status_code), "auth_or_billing"
    if (status_code == 429):
        return str(status_code), "rate_limit"
    if (400 <= status_code < 500):
        return str(status_code), "client_error"
    if (status_code >= 500):
        return str(status_code), "server_error"

    return str(status_code), "unexpected"


def _log_detail(exception:Exception):
    """Collapse an exception message to a single truncated line safe to embed in a log line."""
    detail = " ".join(str(exception).split())
    if (len(detail) > _MAX_DETAIL_LEN):
        detail = detail[:_MAX_DETAIL_LEN] + "...(truncated)"
    return detail.replace('"', "'")


class MistralOCRStore:
    # Shared across all instances/threads — prevents hammering publisher servers.
    _ocr_throttle = Throttle(3.0)

    def __init__(self, api_key:str=None, generation_id:str=None):
        # Tagged onto every OCR log line so a failure can be traced back to the workflow that hit it.
        self.generation_id = generation_id

        # Check if the API key is set
        if (api_key == None):
            # Check if the environment variable is set
            api_key = os.environ.get("MISTRAL_API_KEY", None)
            if (api_key == None):
                # Try to load the API key from a file
                api_key = self.load_api_key("api_keys.donotcommit.json")

        if (api_key == None):
            raise ValueError("Mistral API key is not set.")

        print ("First two characters of Mistral API key: " + api_key[:2] + "**")

        # Initialize the Mistral client
        self.client = Mistral(api_key=api_key)

    # Load the API key from a JSON file
    def load_api_key(self, file_path:str):
        # Load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        if ("mistral" in data):
            return data["mistral"]

        return None


    def load_cache(self, cached_url:str):
        cache_wait_start_time = time.time()
        with mistral_cache_thread_lock:
            total_wait_time = time.time() - cache_wait_start_time
            if (total_wait_time > 1.0):
                print(f"load_cache(): Waited {total_wait_time:.2f} seconds to acquire cache lock for {cached_url}")
            try:
                cached = Persistence.store().load_ocr_cache_markdown(cached_url)
                if cached is not None:
                    print(f"Cache loaded for {cached_url}")
                    return cached
                return None
            except Exception as e:
                print(f"Error loading cache for {cached_url}: {e}")
                return None


    def save_cache(self, cached_url:str, ocr_response_json:dict):
        cache_wait_start_time = time.time()
        with mistral_cache_thread_lock:
            total_wait_time = time.time() - cache_wait_start_time
            if (total_wait_time > 1.0):
                print(f"save_cache(): Waited {total_wait_time:.2f} seconds to acquire cache lock for {cached_url}")
            try:
                markdown_str = self.get_markdown_from_ocr_response(ocr_response_json)
                Persistence.store().save_ocr_cache(cached_url, ocr_response_json, markdown_str)
                print(f"Cache saved for {cached_url}")
                return markdown_str
            except Exception as e:
                print(f"Error saving cache for {cached_url}: {e}")
                return self.get_markdown_from_ocr_response(ocr_response_json)



    def get_markdown_from_ocr_response(self, ocr_response:dict):
        # Extract the markdown content from the OCR response
        markdown_str = ""
        if ("pages" in ocr_response):
            for page in ocr_response.get("pages", []):
                try:
                    if ("markdown" in page):
                        markdown_str += page["markdown"] + "\n\n"
                except Exception as e:
                    print(f"Error processing page {page.get('id', 'unknown')}: {e}")

        return markdown_str.strip()


    # Emit one structured, greppable line for a failed OCR call (see LOG_TOKEN_OCR_ERROR).
    def _log_ocr_error(self, pdf_url:str, exception:Exception, elapsed_sec:float, generation_id:str=None):
        status, category = classify_ocr_error(extract_http_status(exception), exception)
        logger.error(
            '%s status=%s category=%s error_type=%s generation_id=%s '
            'elapsed_sec=%.2f url=%s detail="%s"',
            LOG_TOKEN_OCR_ERROR,
            status,
            category,
            type(exception).__name__,
            generation_id or self.generation_id or "unknown",
            elapsed_sec,
            pdf_url,
            _log_detail(exception),
        )

    # Process a PDF
    def process_pdf(self, pdf_url:str, max_pages:int=20, generation_id:str=None):
        print("process_pdf: started for url: " + str(pdf_url))

        start_time = time.time()
        try:
            # Step 1: First, check for a cached version
            cached_response = self.load_cache(pdf_url)
            if (cached_response is not None):
                print(f"Cache hit for {pdf_url}")
                return cached_response
            else:
                print(f"No cache hit for {pdf_url}")


            self._ocr_throttle.wait(pdf_url)

            # Step 2: If no cached version, call the Mistral OCR API
            start_time = time.time()      # Reset, so elapsed_sec measures the API call, not the cache lookup
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": pdf_url,
                },
                include_image_base64=True,
                pages=range(0, max_pages + 1),  # Process the first `max_pages` pages
            )
            processing_time = time.time() - start_time
            print(f"OCR processing time for {pdf_url}: {processing_time:.2f} seconds")

            # Step 3: Save the response to a cache
            # Convert the response to JSON
            ocr_response_json = ocr_response.dict()
            markdown_str = self.save_cache(pdf_url, ocr_response_json)

            # The success counterpart of MISTRAL_OCR_ERROR: without it the error rate has no
            # denominator, so a rate-based alert can't tell "5 failures out of 5" from "out of 500".
            logger.info(
                '%s category=success generation_id=%s elapsed_sec=%.2f '
                'pages=%d chars=%d url=%s',
                LOG_TOKEN_OCR_OK,
                generation_id or self.generation_id or "unknown",
                processing_time,
                len(ocr_response_json.get("pages", []) or []),
                len(markdown_str or ""),
                pdf_url,
            )

            # Return the markdown
            return markdown_str


        except MistralError as e:
            # An API-level refusal: the status separates a billing/credential failure (402/401/403,
            # always actionable) from a rate limit, a bad PDF, or a Mistral-side outage. Swallowing
            # these made all four look identical in the logs — see allenai/nora-issues#3197.
            self._log_ocr_error(pdf_url, e, time.time() - start_time, generation_id)
            return None

        except Exception as e:
            # Everything else (transport errors, response parsing, bugs on our side). Still logged in
            # the same shape so a failure mode we haven't anticipated is visible to the same metric.
            self._log_ocr_error(pdf_url, e, time.time() - start_time, generation_id)
            return None




# Example stand-alone usage
if __name__ == "__main__":
    # Initialize the MistralOCRStore with your API key
    ocr_store = MistralOCRStore(api_key=None)

    # Process a PDF URL
    pdf_url = "https://arxiv.org/pdf/2503.22708"
    response = ocr_store.process_pdf(pdf_url)

    # Print the response
    print(json.dumps(response, indent=4))
