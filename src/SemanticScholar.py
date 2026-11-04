# SemanticScholar.py
# Wrapper functions for Semantic Scholar API

import os
import json
import requests
import time
import random

from datetime import datetime


# Thread lock for saving the cache
from threading import Lock
THREAD_LOCK_CACHE = Lock()


FILENAME_S2_CACHE = "s2_cache.json"

global_s2cache = None
global_num_cache_writes_since_save = 0


class SemanticScholar():
    def __init__(self, api_key=None):
        self.api_key = api_key
        # Try to load the cache file by using a faux query
        self.query_cache("init", "init", 0, None, None)


    def _paper_search(self, query, limit=10, latest_year=None, latest_month=None):
        # Rate limit
        time.sleep(2)
        # Set the fields to return, which are the above, plus the paper title and publication date
        fields = "paperId,corpusId,externalIds,url,title,abstract,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,authors,citations,references,tldr"

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": fields,
        }
        if (latest_year is not None):
            if (latest_month is not None):
                # Add the year and month to the query (up to latest_year and latest_month)
                params["publicationDateOrYear"] = f":{latest_year}-{latest_month:02d}"
            else:
                # Add the year range to the query (up to latest_year)
                params["publicationDateOrYear"] = f":{latest_year}"
        headers = {}
        if self.api_key:
            # Use the API key string, not the object
            headers["x-api-key"] = self.api_key
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Error:", response.status_code)
            print(response.text)
            return None


    # Bulk paper search, with abstracts
    def _bulk_abstract_search(self, query, limit=10, latest_year=None, latest_month=None):
        # Rate limit
        time.sleep(2)

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,corpusId,publicationDate"
        }
        if (latest_year is not None):
            if (latest_month is not None):
                # Add the year and month to the query (up to latest_year and latest_month)
                params["publicationDateOrYear"] = f":{latest_year}-{latest_month:02d}"
            else:
                # Add the year range to the query (up to latest_year)
                params["publicationDateOrYear"] = f":{latest_year}"

        headers = {}
        if self.api_key:
            # Use the API key string, not the object
            headers["x-api-key"] = self.api_key
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Error:", response.status_code)
            print(response.text)
            return None



    # Get a list of papers by an author (by author_id)
    def get_papers_by_author(self, author_id:int):
        # Rate limit
        time.sleep(2.0)

        url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"
        params = {
            "fields": "paperId,title,abstract,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount"
        }

        headers = {}
        if self.api_key:
            # Use the API key string, not the object
            headers["x-api-key"] = self.api_key
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Error:", response.status_code)
            print(response.text)
            return None

    #
    #   Caching
    #
    def query_cache(self, query_type:str, query, limit, date_cutoff_yyyy=None, date_cutoff_mm=None):
        global global_s2cache
        # Check if the cache file exists
        if os.path.exists(FILENAME_S2_CACHE):
            # Make a key for this query
            try:
                query_key = str(query_type) + "-" + str(query) + "-limit" + str(limit) + "-datecutoff" + str(date_cutoff_yyyy) + "-" + str(date_cutoff_mm)

                # Load the cache file, if we haven't already
                if (global_s2cache is None):
                    print("Loading cache file from disk: " + FILENAME_S2_CACHE)
                    with open(FILENAME_S2_CACHE, 'r') as f:
                        # Faster JSON loading
                        import orjson
                        global_s2cache = orjson.loads(f.read())
                        #data = json.load(f)
                # Check if the query is in the cache
                if (query_key in global_s2cache):
                    return global_s2cache[query_key]

            except Exception as e:
                print(f"Error loading cache file {FILENAME_S2_CACHE}: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            # No cache file exists -- initialize the cache
            global_s2cache = {}
        return None

    def save_cache(self, query_type:str, query, limit, date_cutoff_yyyy=None, date_cutoff_mm=None, data_to_store=None, force_save=False):
        # Check if the cache file exists
        global global_s2cache
        global global_num_cache_writes_since_save
        with THREAD_LOCK_CACHE:
            data = {}
            query_key = str(query_type) + "-" + str(query) + "-limit" + str(limit) + "-datecutoff" + str(date_cutoff_yyyy) + "-" + str(date_cutoff_mm)
            # Set the key
            global_s2cache[query_key] = data_to_store
            global_num_cache_writes_since_save += 1

            # Save the data to the cache file
            if ((global_num_cache_writes_since_save > 0) and (global_num_cache_writes_since_save % 25 == 0)) or (force_save == True):
                # Save the cache file
                print("Saving cache file " + FILENAME_S2_CACHE)
                with open(FILENAME_S2_CACHE, 'w') as f:
                    json.dump(global_s2cache, f, indent=4)
                global_num_cache_writes_since_save = 0




    #
    #   Regular paper search based on query (with date cutoffs, and with caching)
    #
    def search_papers(self, query, limit=10, date_cutoff_yyyy=None, date_cutoff_mm=None):
        # Check if there's a cache hit -- if so, return it
        cached_data = self.query_cache("paper", query, limit, date_cutoff_yyyy, date_cutoff_mm)
        if (cached_data is not None):
            print("Cache hit for query: " + str(query))
            return cached_data

        # Do the regular paper search
        MAX_RETRIES = 5
        for i in range(MAX_RETRIES):
            data = self._paper_search(query, limit, latest_year=date_cutoff_yyyy, latest_month=date_cutoff_mm)
            if (data is not None):
                break
            else:
                delay_time = 5*(i+1)
                print("Error: No data found for query " + str(query) + ". Retrying... (" + str(i+1) + "/" + str(MAX_RETRIES) + ")")
                time.sleep(delay_time)

        #print("Data: ")
        #print(json.dumps(data, indent=4))
        if data is None:
            return None
        if ("data" in data):
            data = data["data"]
        else:
            return None
        # Pack it
        papers = {
            "query": query,
            "limit": limit,
            "date_cutoff_yyyy": date_cutoff_yyyy,
            "date_cutoff_mm": date_cutoff_mm,
            "papers": data
        }
        # Save the data to the cache file
        self.save_cache("paper", query, limit, date_cutoff_yyyy, date_cutoff_mm, data_to_store=papers)
        return papers


    #
    #   S2 Bulk Paper Abstract Search based on query (with date cutoffs, and with caching)
    #
    def search_bulk_abstracts(self, query, limit=10, date_cutoff_yyyy=None, date_cutoff_mm=None):
        # Check if there's a cache hit -- if so, return it
        cached_data = self.query_cache("bulk_abstract", query, limit, date_cutoff_yyyy, date_cutoff_mm)
        if (cached_data is not None):
            print("Cache hit for query: " + str(query))
            return cached_data

        # Do the regular bulk abstract search
        MAX_RETRIES = 5
        for i in range(MAX_RETRIES):
            data = self._bulk_abstract_search(query, limit, latest_year=date_cutoff_yyyy, latest_month=date_cutoff_mm)
            if (data is not None):
                break
            else:
                delay_time = 5*(i+1)
                print("Error: No data found for query " + str(query) + ". Retrying... (" + str(i+1) + "/" + str(MAX_RETRIES) + ")")
                time.sleep(delay_time)

        #print("Data: ")
        #print(json.dumps(data, indent=4))
        if data is None:
            return None
        if ("data" in data):
            data = data["data"]
        else:
            return None

        # Pack it
        abstracts = {
            "query": query,
            "limit": limit,
            "date_cutoff_yyyy": date_cutoff_yyyy,
            "date_cutoff_mm": date_cutoff_mm,
            "abstracts": data,
        }
        # Save the data to the cache file
        self.save_cache("bulk_abstract", query, limit, date_cutoff_yyyy, date_cutoff_mm, data_to_store=abstracts)
        return abstracts


    # Find a paper by its title
    def find_paper_by_title(self, title:str):
        time.sleep(1)

        def find_paper(title:str):
            url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
            params = {
                "query": title,
                "fields": "paperId,corpusId,externalIds,url,title,abstract,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,authors,citations,references,tldr"
            }

            headers = {}
            if self.api_key:
                # Use the API key string, not the object
                headers["x-api-key"] = self.api_key
            response = requests.get(url, params=params, headers=headers)
            #print("Response Code: " + str(response.status_code))
            if response.status_code == 200:
                # Success
                data = response.json()
                return data
            elif response.status_code == 404:
                # No results found
                return {}
            elif response.status_code == 500:
                # Server error: In case the service is temporarily unavailable.
                print("Warning: Semantic Scholar API returned 500 Internal Server Error -- Service is temporarily unavailable.")
                return None
            else:
                print("Error:", response.status_code)
                print(response.text)
                return None


        # Try to find the paper
        MAX_RETRIES = 3
        data = None
        for i in range(MAX_RETRIES):
            data = find_paper(title)
            if (data is not None):
                break
            else:
                delay_time = 5*(i+1)
                print("Error: No data found for title " + str(title) + ". Retrying... (" + str(i+1) + "/" + str(MAX_RETRIES) + ")")
                time.sleep(delay_time)

        # Interpret the results
        if (data is None):
            # Error retrieving data
            return None
        elif (data == {}):
            # No results found
            return None
        else:
            # Results found
            return data


    # Find a paper by its title
    def find_paper_by_paperid(self, paper_id:str):
        time.sleep(2)

        def find_paper(paper_id:str):
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
            params = {
                "fields": "paperId,corpusId,externalIds,url,title,abstract,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,authors,citations,references,tldr"
            }

            headers = {}
            if self.api_key:
                # Use the API key string, not the object
                headers["x-api-key"] = self.api_key
            response = requests.get(url, params=params, headers=headers)

            if response.status_code == 200:
                # Success
                data = response.json()
                return data
            elif response.status_code == 404:
                # No results found
                return {}
            else:
                print("Error:", response.status_code)
                print(response.text)
                return None

        # Try to find the paper
        MAX_RETRIES = 5
        data = None
        for i in range(MAX_RETRIES):
            data = find_paper(paper_id)
            if (data is not None):
                break
            else:
                delay_time = 5*(i+1)
                print("Error: No data found for paper_id `" + str(paper_id) + "`.  Retrying... (" + str(i+1) + "/" + str(MAX_RETRIES) + ")")
                time.sleep(delay_time)

        # Interpret the results
        if (data is None):
            # Error retrieving data
            return None
        elif (data == {}):
            # No results found
            return None
        else:
            # Results found
            return data


    # Find a paper by its Corpus ID
    def find_paper_by_corpusid(self, corpus_id:str):
        # Reassmble into a query that uses the paper_id function
        paper_id = "CorpusID:" + str(corpus_id)
        # Use the find_paper_by_paperid function to find the paper
        return self.find_paper_by_paperid(paper_id)


    # Get all papers from a given venue and year (using paper bulk search)
    def get_papers_by_venue_and_year(self, venue_name: str, year: str, fields: str = None):
        # Rate limit a bit between calls
        time.sleep(2)

        url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

        if fields is None:
            fields = (
                "paperId,title,abstract,url,venue,year,publicationDate,"
                "referenceCount,citationCount,influentialCitationCount,authors"
            )

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        # NOTE: query is required by the API; we use a very broad wildcard query.
        params = {
            "query": "*",
            "fields": fields,
            "venue": venue_name,
            "year": year,
            # Optional: sort by paperId or publicationDate to have a stable order
            "sort": "paperId:asc",
        }

        all_papers = []
        token = None

        while True:
            if token is not None:
                params["token"] = token
            else:
                params.pop("token", None)

            response = requests.get(url, params=params, headers=headers)

            if response.status_code != 200:
                print("Error:", response.status_code)
                print(response.text)
                break

            data = response.json()

            # Bulk search returns results under 'data'
            batch = data.get("data", [])
            print(f"Fetched {len(batch)} papers in this batch")

            if not batch:
                # Nothing more to fetch (or filters are too strict)
                break

            all_papers.extend(batch)

            # Pagination: 'token' is the continuation token
            token = data.get("token")
            if not token:
                break

            # Be nice to the API
            time.sleep(0.1)

        print(f"Total papers fetched: {len(all_papers)}")
        return all_papers


#
#   Paper Finder
#

# Paper Finder settings. Most are plain constants; only three are read from the environment, each
# for a reason that already exists rather than in case it is wanted later:
#   - PAPER_FINDER_BASE_URL selects which Paper Finder deployment to talk to. Paper Finder runs
#     several (prod-web, main-web, and a local instance on :8000); before this, switching meant
#     editing the URL literal in this file and redeploying (see c79df5b).
#   - PAPER_FINDER_WORKER_PROFILE so "deep" (this PR's change) can be A/B'd against "standard"
#     without a rebuild.
#   - PAPERFINDER_SEARCH_TIMEOUT_SECONDS so the new budget can be tuned if 600s proves wrong.
# Module-level names are `_`-prefixed because PaperFinderRequests.py does
# `from SemanticScholar import *` and would otherwise pick them up.


def _env_or_default(name, default):
    """Read a setting from the environment, treating an empty value as unset.

    Not about deployment environments -- this is `os.environ` with one wrinkle: a name present
    but blank (which is what `cp .env.example .env`, or an unfilled `.env` line, produces) reads
    as "" rather than falling back. Sending that through would mean an empty worker_profile
    (rejected 422) or a crash at import on int("").
    """
    value = os.environ.get(name)
    return value if value else default


_PAPER_FINDER_BASE_URL = _env_or_default("PAPER_FINDER_BASE_URL", "").rstrip("/")
# A stable *service* identity, matching Paper Finder's own "nora-handler"/"debug-console" convention.
# Deliberately not `user_id`, which is hardcoded to "api" for every request and carries no real identity.
_PAPER_FINDER_CALLER_ACTOR_ID = "theorizer"
# "infer" lets Paper Finder pick search effort per query; matches nora's configured value.
_PAPER_FINDER_OPERATION_MODE = "infer"
# "deep" is the captain-orchestrated flow nora's review path runs. The /api/3/tasks request model
# defaults to "standard", so this must be sent explicitly or Theorizer silently gets the weaker profile.
_PAPER_FINDER_WORKER_PROFILE = _env_or_default("PAPER_FINDER_WORKER_PROFILE", "deep")
# How long one search may take. The previous implementation gave up after ~110s, far shorter than a
# deep search needs -- it returned None, which surfaced downstream as "no papers found".
# Deliberately public (no leading underscore): Theorizer1 imports it via `from SemanticScholar import *`
# and derives its own, longer wait from it, so the two layers cannot drift apart.
PAPER_FINDER_SEARCH_TIMEOUT_SECONDS = int(_env_or_default("PAPERFINDER_SEARCH_TIMEOUT_SECONDS", 600))
# Per-request ceiling, not per-search: the search is polled, so this only bounds a single HTTP call.
_PAPER_FINDER_HTTP_TIMEOUT_SECONDS = 30
_PAPER_FINDER_POLL_INTERVAL_SECONDS = 5


class PaperFinderError(Exception):
    """A Paper Finder request failed for a reason we can name.

    Raised instead of returning None so the failure reason survives up to the workflow:
    PaperFinderRequests stores it as {"error": ...}, and the executor reports it verbatim
    rather than collapsing every failure into a misleading "No candidate papers found".
    """
    pass


# Get PaperFinder results for a query, via the /api/3/tasks REST API.
#
# `/api/2/rounds` is deprecated (it is now a thin shim over these same v3 handlers and returns a
# Sunset header). /api/3/tasks is the REST successor: same POST -> 204 + Location -> poll flow, but it
# also accepts caller_actor_id/cost_trace_id/worker_profile, and its `/enriched` response preserves
# `doc_collection.documents[].publication_date`, which the knowledge-cutoff filter depends on.
def _get_paperfinder_results_internal(query:str, cost_trace_id:str=None):
    print("PaperFinder: get_paperfinder_results(): Started... (query = '" + query + "')")

    # Step 1: Pack the request
    paperfinder_request = {
        "paper_description": query,
        "caller_actor_id": _PAPER_FINDER_CALLER_ACTOR_ID,
        "cost_trace_id": cost_trace_id,
        "operation_mode": _PAPER_FINDER_OPERATION_MODE,
        "worker_profile": _PAPER_FINDER_WORKER_PROFILE,
        "generate_widget_state": False,
    }

    if not _PAPER_FINDER_BASE_URL:
        raise PaperFinderError("Set PAPER_FINDER_BASE_URL to your authorized Paper Finder service")
    url = f"{_PAPER_FINDER_BASE_URL}/api/3/tasks"

    # Send the POST request
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(paperfinder_request),
            timeout=_PAPER_FINDER_HTTP_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        raise PaperFinderError(f"Could not reach Paper Finder at {url}: {e!r}")

    # Step 2: Check the response. A 204 carries the result location in the `Location` header.
    if (response.status_code != 204):
        # Includes 503 when Paper Finder's task queue is full.
        raise PaperFinderError(f"Paper Finder rejected the request: {response.status_code} - {response.text}")

    relative_url = response.headers.get("location")
    if not relative_url:
        raise PaperFinderError("Paper Finder returned a 204 with no Location header.")

    full_url_enriched = f"{_PAPER_FINDER_BASE_URL}{relative_url}/enriched"
    print(f"Waiting for results to be available at: {full_url_enriched}")

    # Step 3: Poll until the task reaches a terminal state, or we run out of budget.
    #   404 -> still running     200 -> done     410 -> the task failed (detail in the body)
    start_time = time.time()
    while ((time.time() - start_time) < PAPER_FINDER_SEARCH_TIMEOUT_SECONDS):
        try:
            response = requests.get(full_url_enriched, timeout=_PAPER_FINDER_HTTP_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            # A transient network blip shouldn't abort a search that may already be nearly done.
            print(f"PaperFinder: transient error while polling for results (will retry): {e!r}")
            time.sleep(_PAPER_FINDER_POLL_INTERVAL_SECONDS)
            continue

        if (response.status_code == 200):
            print(f"Paperfinder results received. ({round(time.time() - start_time, 1)} seconds elapsed)")
            return response.json()

        if (response.status_code == 410):
            raise PaperFinderError(f"Paper Finder task failed: {response.text}")

        # Transient at the edge (load-balancer hiccup, rate limit). The task is still running
        # server-side, so retry rather than throwing away a search that is already in flight --
        # same reasoning as the RequestException branch above. Deliberately NOT extended to other
        # 4xx: an auth or bad-URL error cannot succeed on retry, and would just burn the budget.
        if (response.status_code == 429) or (response.status_code >= 500):
            print(f"PaperFinder: transient {response.status_code} while polling for results (will retry)")
            time.sleep(_PAPER_FINDER_POLL_INTERVAL_SECONDS)
            continue

        if (response.status_code != 404):
            raise PaperFinderError(f"Unexpected response while polling Paper Finder: {response.status_code} - {response.text}")

        time.sleep(_PAPER_FINDER_POLL_INTERVAL_SECONDS)

    raise PaperFinderError(f"Paper Finder timed out after {PAPER_FINDER_SEARCH_TIMEOUT_SECONDS} seconds.")


# Get PaperFinder results based on a query.
def get_paperfinder_results(query:str, cost_trace_id:str=None):
    return _get_paperfinder_results_internal(query=query, cost_trace_id=cost_trace_id)



# Load the S2 API key
def load_s2_api_key():
    filename_s2_api_key = "s2_key.donotcommit.txt"
    if os.path.exists(filename_s2_api_key):
        with open(filename_s2_api_key, 'r') as f:
            s2_api_key = f.read().strip()
        return s2_api_key
    else:
        # Fall back to environment variable (e.g., on Modal)
        env_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", None)
        if env_key:
            return env_key
        print(f"Warning: Semantic Scholar API key not found in file or environment.")
        return None


# Debug/Tests
if __name__ == "__main__":
    # Load the S2 API key
    s2_api_key = load_s2_api_key()
    if (s2_api_key is None):
        print("Exiting due to missing Semantic Scholar API key.")
        exit(1)

    # Initialize the SemanticScholar API client
    s2_client = SemanticScholar(api_key=s2_api_key)

    # Try out the author endpoint
    results = s2_client.get_papers_by_author(author_id = 144949918)
    print(json.dumps(results, indent=4))

    # Try out finding a paper by title
    title = "ScienceWorld: Is your Agent Smarter"
    paper_data = s2_client.find_paper_by_title(title)
    print("Find paper by title: " + title)
    print(json.dumps(paper_data, indent=4))

    # Dump to file
    with open("test_find_paper_by_title.json", 'w') as f:
        json.dump(paper_data, f, indent=4)

    # # PaperFinder test
    # query = "Papers about virtual environments for scientific discovery."
    # pf_results = get_paperfinder_results(query=query)
    # print(json.dumps(pf_results, indent=4))