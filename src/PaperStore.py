# PaperStore.py
# A class to handle finding papers, retrieving their full-text, and making it available to threads that ask for it.

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

# Timeout
from func_timeout import func_timeout, FunctionTimedOut

# Threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading


from SemanticScholar import *
from MistralOCRStore import *
from Throttle import Throttle
import NoraRetrievalFullText

# Make an enumeration for different paper statuses (e.g. full-text, metadata-only, failed, timeout)
from enum import Enum
class PaperStatus(Enum):
    FULL_TEXT = "full-text"               # A status for papers that have full-text available
    METADATA_ONLY = "metadata-only"       # A status for papers that have metadata only (no full-text)"
    FAILED = "failed"                     # A status for papers that failed to be processed (for whatever reason)
    TIMEOUT = "timeout"                   # A status for papers that timed out during processing

# Titles go into log lines truncated, so one long title can't push the
# corpus_id and the outcome off the end of a line. A paper entered by sha or
# by URL may have no title at all.
def log_title(paper_title: Optional[str]) -> str:
    return (paper_title or "")[:60]


# Global thread lock/Rate limiting for Arxiv API
#THREAD_LOCK_ARXIV_API = Lock()
# Time of the last Arxiv API call
#LAST_ARXIV_API_CALL = 0


@dataclass
class FullTextResult:
    fulltext: Optional[str] = None
    # asta-papers only. here instead of under AstaPapersFullTextResult
    # because we put them in a paper record
    fetch_channel: Optional[str] = None
    license_class: Optional[str] = None
    license_url: Optional[str] = None
    source_type: Optional[str] = None
    llm_usable: Optional[bool] = None
    # only the legacy path collects these. here instead of under
    # LegacyFullTextResult because we put them in a paper record, where
    # PaperEntry.paper_urls is a list - so the other paths carry [], not None.
    paper_urls: List[str] = field(default_factory=list)

    def paper_restricted(self) -> Optional[bool]:
        return None

    def found_fulltext(self) -> bool:
        return self.fulltext is not None


@dataclass
class NoraRetrievalFullTextResult(FullTextResult):
    no_body_reason: Optional[str] = None

    def paper_restricted(self) -> Optional[bool]:
        if self.fulltext is not None:
            return False
        if self.no_body_reason == NoraRetrievalFullText.NO_BODY_PAPER_RESTRICTED:
            return True
        if self.no_body_reason == NoraRetrievalFullText.NO_BODY_NO_UNRESTRICTED_FULLTEXT:
            return False
        return None


@dataclass
class AstaPapersFullTextResult(FullTextResult):
    def paper_restricted(self) -> Optional[bool]:
        if self.llm_usable is not None:
            return not self.llm_usable
        return None


@dataclass
class LegacyFullTextResult(FullTextResult):
    def paper_restricted(self) -> Optional[bool]:
        # we don't know
        return None


class PaperStore():
    #
    #   Constructor
    #
    def __init__(self, filename_in:str=None, deserialize_from_dict:dict=None, generation_id:str=None):
        # The workflow this store belongs to, tagged onto OCR log lines so a failure is traceable
        # back to a run. None when the store is built outside a workflow (tests, stand-alone use).
        self.generation_id = generation_id

        # Initialize the Mistral OCR store
        self.ocr_store = MistralOCRStore(api_key=None, generation_id=generation_id)      # Will automatically load the API key

        # Initialize the Semantic Scholar API
        s2_api_key = load_s2_api_key()
        self.s2_api = SemanticScholar(api_key=s2_api_key)

        # Initialize the paperstore
        self.paperstore = {}        # A dictinonary to store the papers with some metadata.
        self.paperstore_lut = {}    # A look-up table that maps paper identifiers (like corpus_id or s2_sha) to the keys in the paperstore dictionary.

        # Queue for adding papers to the paperstore
        self.queue = []
        self.finished_work = {}
        self.next_request_id = 1

        # Rate limiter for Semantic Scholar API calls
        self._api_throttle = Throttle(0.2)

        # Parallel paper retrieval
        self.active_threads = set()
        self.MAX_THREADS = 10

        # Skip S2ORC internal cache by default (use Mistral OCR only)
        self.USE_S2ORC_INTERNAL_CACHE = False

        # Lazily-initialized asta-papers Client (only built when USE_ASTA_PAPERS=true)
        self._asta_papers_client = None

        # Update message: Show the user the PaperFinder heartbeat every 10 seconds
        self.heartbeat_interval = 10
        self.heartbeat_last_time = 0

        # Thread lock
        self.THREAD_LOCK_PAPERSTORE = threading.Lock()
        self.THREAD_ACTIVE = False

        # Logging (for debugging/timing)
        self.FILENAME_LOG_FILE = "paperstore_log.txt"
        self.LOG_ENABLED = False

        # Load the existing paperstore from the file
        if (filename_in is not None):
            success = self.load(filename_in)
            if (not success):
                print("PaperStore.__init__(): WARNING: Failed to load paperstore from file: " + filename_in)
                exit(1)
        elif (deserialize_from_dict is not None):
            success = self.deserialize(deserialize_from_dict)
            if (not success):
                print("PaperStore.__init__(): WARNING: Failed to load paperstore from dict.")
                exit(1)

        # Start a background thread to process the queue
        self.worker_thread = None
        self.start_thread()


    #
    #   Logging
    #
    def add_log_entry(self, message:str):
        if (not self.LOG_ENABLED):
            return

        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        packed = {"timestamp": timestamp_str, "message": message}

        # Append to the log file immediately
        with open(self.FILENAME_LOG_FILE, 'a') as f:
            f.write(json.dumps(packed) + "\n")


    def log_and_print(self, message:str):
        # Say it on stdout as well as in the log file. add_log_entry() is a no-op when
        # LOG_ENABLED is false, so the print is the only line guaranteed to show up when
        # you are watching a running Theorizer decide where a paper's full text came from.
        print(message)
        self.add_log_entry(message)



    #
    #   Size
    #

    def get_num_papers(self):
        with self.THREAD_LOCK_PAPERSTORE:
            return len(self.paperstore)

    def get_queue_size(self):
        with self.THREAD_LOCK_PAPERSTORE:
            return len(self.queue)

    #
    #   Getters/Setters for accessing (or storing) papers
    #

    # Try to retrieve a paper by its key
    def get_paper_by_key(self, key:str):
        # First, look up the key in the paperstore LUT
        paper = self._paper_lookup_by_key(key)
        return paper

    # Retrieve a paper's text by its key
    def get_paper_text_by_key(self, key:str):
        # Get the paper by its key
        paper = self.get_paper_by_key(key)
        if (paper is None):
            #print("PaperStore.get_paper_text_by_key(): No paper found for key: " + str(key))
            return None

        # Return the full-text Markdown of the paper
        paper_markdown = paper.get("paper_markdown", None)
        if (paper_markdown is None):
            #print("PaperStore.get_paper_text_by_key(): No full-text Markdown found for paper with key: " + str(key))
            return None
        return paper_markdown

    # Lookup a paper record.  Handles looking up by alternate keys
    def _paper_lookup_by_key(self, alternate_key:str):
        # First, get the primary key for the paper by looking it up in the LUT
        with self.THREAD_LOCK_PAPERSTORE:
            if (alternate_key in self.paperstore_lut):
                primary_key = self.paperstore_lut[alternate_key]
                #print("PaperStore._paper_lookup_by_key(): Found primary key for alternate key '" + str(alternate_key) + "': " + str(primary_key))
                # Now, get the paper by its primary key
                paper = self.paperstore.get(primary_key, None)
                return paper
            else:
                return None


    # Sanitize paper titles
    def _sanitize_paper_title(self, title:str):
        import re
        # Convert to lowercase, remove non-alphanumeric characters, remove multiple whitespace, and replace spaces with underscores
        sanitized_title = title.lower()
        sanitized_title = ''.join(c for c in sanitized_title if c.isalnum() or c.isspace())
        # Remove multiple whitespace with a regular expression
        sanitized_title = re.sub(r'\s+', ' ', sanitized_title).strip()
        sanitized_title = sanitized_title.replace(' ', '_')
        return sanitized_title

    def _mk_key_corpus_id(self, corpus_id:str):
        if (corpus_id is None):
            return None
        return "paper-" + corpus_id

    def _mk_key_s2_sha(self, s2_sha:str):
        if (s2_sha is None):
            return None
        return "paper-" + s2_sha

    def _mk_key_title(self, title:str):
        if (title is None):
            return None
        return "paper-" + self._sanitize_paper_title(title)

    # Add a paper to the paperstore, by its keys.
    # Returns the primary key
    def _add_paper_by_keys(self, paper:dict, corpus_id:str=None, s2_sha:str=None, title:str=None, alt_title:str=None):
        print("PaperStore._add_paper_by_keys(): Adding paper with keys: corpus_id=" + str(corpus_id) + ", s2_sha=" + str(s2_sha) + ", title=" + str(title))
        # Check: At least one of the parameters must be provided
        if (corpus_id is None) and (s2_sha is None) and (title is None):
            return None

        # Step 1: Make the keys that this paper can be referenced by
        primary_key = None      # The key for the paperstore
        alternate_keys = []     # A list of alternate keys that can also be used to look up this paper
        if (s2_sha is not None):
            primary_key = self._mk_key_s2_sha(s2_sha)
            alternate_keys.append(self._mk_key_s2_sha(s2_sha))
        if (corpus_id is not None):
            if (primary_key is None):
                primary_key = self._mk_key_corpus_id(corpus_id)
            alternate_keys.append(self._mk_key_corpus_id(corpus_id))
        if (title is not None):
            if (primary_key is None):
                primary_key = self._mk_key_title(title)
            alternate_keys.append(self._mk_key_title(title))
        if (alt_title is not None):
            # If an alternate title is provided, use it as well
            alternate_keys.append(self._mk_key_title(alt_title))

        # Step 2: Then, store this paper under each key
        with self.THREAD_LOCK_PAPERSTORE:
            # Add to the paperstore
            self.paperstore[primary_key] = paper
            # Add to the paperstore LUT
            for key in alternate_keys:
                # Note, in rare cases, this will overwrite the existing key in the LUT (i.e. if papers have duplicate titles)
                self.paperstore_lut[key] = primary_key

        # Return the primary key for the paper
        return primary_key


    #
    #   Threading/Queue/Work Processing
    #

    # Add a new paper to the queue to add papers, by one (or more) of: corpus_id, s2_sha, or title.
    # This is the main submission entrypoint to request a paper be added to the paperstore.
    def submit_paper(self, corpus_id:str=None, s2_sha:str=None, title:str=None):
        # Check: At least one of the parameters must be provided
        if (corpus_id is None) and (s2_sha is None) and (title is None):
            return None

        self.add_log_entry("PaperStore.submit_paper(): (waiting for threadlock) Submitting paper with corpus_id=" + str(corpus_id) + ", s2_sha=" + str(s2_sha) + ", title=" + str(title))
        with self.THREAD_LOCK_PAPERSTORE:
            self.add_log_entry("PaperStore.submit_paper(): (threadlock open) Submitting paper with corpus_id=" + str(corpus_id) + ", s2_sha=" + str(s2_sha) + ", title=" + str(title))

            # Generate an ID for the request
            request_id = "paper-add-request-" + str(self.next_request_id)
            self.next_request_id += 1

            # Add the request to the queue
            self.queue.append({
                'id': request_id,
                'corpus_id': corpus_id,
                's2_sha': s2_sha,
                'title': title,
            })

            return request_id


    # Check to see if a paper has been added to the store, by its request ID.
    # Returns `None` if the paper has not been added yet.
    # Returns the key to access the paper, if it has been added
    def is_paper_request_complete(self, request_id:str):
        with self.THREAD_LOCK_PAPERSTORE:
            # Return a copy of the finished work
            if (request_id in self.finished_work):
                return self.finished_work[request_id]
            else:
                return None

    # Get the number of requests in the queue
    def get_num_requests(self):
        with self.THREAD_LOCK_PAPERSTORE:
            return len(self.queue)

    # Returns true if there is anything in the queue, or anything currently being processed
    # TODO: Depricate this?
    def is_busy(self):
        # Check 1: Are there any requests in the queue?
        if (self.get_num_requests() > 0):
            return True
        # Check 2: Are we actively processing work?
        with self.THREAD_LOCK_PAPERSTORE:
            if len(self.active_threads) > 0:
                return True

        # If we reach here, then we're not busy
        return False


    # Cancel any requests that haven't completed yet — removes them from the queue
    # and marks them as finished so callers aren't left waiting.
    def cancel_incomplete(self, request_ids):
        with self.THREAD_LOCK_PAPERSTORE:
            stale = {rid for rid in request_ids if rid not in self.finished_work}
            if not stale:
                return
            # Only remove matching IDs; other workflows may have requests in the same queue.
            self.queue = [r for r in self.queue if r['id'] not in stale]
            for rid in stale:
                self.finished_work[rid] = None
        print(f"PaperStore.cancel_incomplete(): Cancelled {len(stale)} incomplete requests")


    # Add finished work to the directory of finished work
    # When a paper request is finished, call this to set the nominal key that that paper should be referenced by.
    def _set_paper_as_processed(self, request_id:str, reference_key:str):
        with self.THREAD_LOCK_PAPERSTORE:
            # Set the finished work
            self.finished_work[request_id] = reference_key

    # Stop the worker thread
    def stop_thread(self):
        print("PaperStore.start_thread(): Stopping worker thread...")
        self.THREAD_ACTIVE = False
        if (self.worker_thread is not None):
            self.worker_thread.join()
        self.worker_thread = None

    def start_thread(self):
        if (self.worker_thread is not None) and (self.worker_thread.is_alive()):
            print("PaperStore.start_thread(): WARNING: Worker thread is already running -- not starting a new one.")
            return
        print("PaperStore.start_thread(): Starting worker thread...")

        self.THREAD_ACTIVE = True
        # Start the worker thread
        self.worker_thread = threading.Thread(target=self.process_work_monitor, daemon=True)
        self.worker_thread.start()

        print("PaperStore.start_thread(): Worker thread started.")



    # Worker thread: process a single paper retrieval request.
    def _perform_work(self, request):
        try:
            self._api_throttle.wait()
            self.add_log_entry(f"Processing: corpus_id={request.get('corpus_id')}, title={request.get('title')}")
            paper_key = self.add_paper_start(
                corpus_id=request.get('corpus_id', None),
                s2_id=request.get('s2_sha', None),
                paper_title=request.get('title', None)
            )
            self._set_paper_as_processed(request['id'], paper_key)
        except Exception as e:
            import traceback
            error_str = f"Error '{request['id']}': {str(e)}\n{traceback.format_exc()}"
            self._set_paper_as_processed(request['id'], {"error": error_str})
            self.add_log_entry("ERROR: " + error_str)

    # Main thread: Process the request queue
    def process_work_monitor(self):
        print("PaperStore.process_work_monitor(): Worker thread started.")
        while (self.THREAD_ACTIVE):
            time.sleep(1)

            # Heartbeat
            time_since_last_heartbeat = time.time() - self.heartbeat_last_time
            if (time_since_last_heartbeat >= self.heartbeat_interval):
                print(f"PaperStore.process_work_monitor(): Heartbeat (active: {len(self.active_threads)}, queue: {len(self.queue)})")
                self.add_log_entry(f"PaperStore.process_work_monitor(): Heartbeat (active: {len(self.active_threads)}, queue: {len(self.queue)})")
                self.heartbeat_last_time = time.time()

            # Step A: Remove completed threads from active set
            with self.THREAD_LOCK_PAPERSTORE:
                for request_id in list(self.active_threads):
                    if request_id in self.finished_work:
                        self.active_threads.remove(request_id)

            # Step B: Start new threads up to MAX_THREADS
            with self.THREAD_LOCK_PAPERSTORE:
                while (len(self.active_threads) < self.MAX_THREADS) and (len(self.queue) > 0):
                    request = self.queue.pop(0)
                    thread = threading.Thread(target=self._perform_work, args=(request,), daemon=True)
                    thread.start()
                    self.active_threads.add(request['id'])

        print("PaperStore.process_work_monitor(): Worker thread has completed.")



    #
    #   Loading/Saving
    #
    def save(self, filename:str):
        print("PaperStore.save(): Saving paperstore to file: " + filename)
        # Pack the paperstore and paperstore LUT
        print("PaperStore.save(): Waiting for threadlock to be released...")
        with self.THREAD_LOCK_PAPERSTORE:
            print("PaperStore.save(): Threadlock available... continuing to save.")
            # packed = {
            #     "paperstore_lut": self.paperstore_lut,
            #     "paperstore": self.paperstore,
            # }
            packed = self.serialize()

            # Save to a file
            with open(filename, 'w') as f:
                json.dump(packed, f, indent=4)

            print("PaperStore.save(): Save complete, releasing threadlock.")


    def load(self, filename:str):
        print("PaperStore.load(): Loading paperstore from file: " + filename)
        # Load the paperstore from a file
        if (not os.path.exists(filename)):
            print("PaperStore.load(): File does not exist: " + filename)
            return False

        with open(filename, 'r') as f:
            packed = json.load(f)

        # Unpack the paperstore and paperstore LUT
        self.deserialize(packed)

        print("PaperStore.load(): Loaded " + str(len(self.paperstore)) + " papers from the file.")
        return True

    def serialize(self):
        print("PaperStore.serialize(): Serializing paperstore to dict.")
        # Pack the paperstore and paperstore LUT
        packed = {
            "paperstore_lut": self.paperstore_lut,
            "paperstore": self.paperstore,
        }
        return packed

    def deserialize(self, deserialize_from_dict:dict):
        print("PaperStore.deserialize(): Deserializing paperstore from dict.")
        # Unpack the paperstore and paperstore LUT
        self.paperstore_lut = deserialize_from_dict.get("paperstore_lut", {})
        self.paperstore = deserialize_from_dict.get("paperstore", {})

        print("PaperStore.deserialize(): Deserialized " + str(len(self.paperstore)) + " papers from the dict.")
        return True

    def merge_from_dict(self, data:dict):
        """Merge papers with full text from a wrapped paperstore dict; skip ones without paper_markdown."""
        for key, paper in data.get("paperstore", {}).items():
            if paper.get("paper_markdown"):
                self.paperstore[key] = paper
                self.paperstore_lut[key] = key
                # Register the corpus_id alias so a later PaperFinder discovery of the same
                # paper dedupes against this seed (it looks papers up by _mk_key_corpus_id)
                # instead of re-OCRing and storing a duplicate full-text copy.
                corpus_id = paper.get("corpus_id")
                if corpus_id is not None:
                    self.paperstore_lut.setdefault(self._mk_key_corpus_id(str(corpus_id)), key)
        self.paperstore_lut.update(data.get("paperstore_lut", {}))


    #
    #   Performing Work: Downloading and Converting Papers
    #

    # Add a paper to the paperstore.  Handles fetching the paper, downloading it, converting it, and adding it to the paperstore.
    # Returns the ID of the paper in the paperstore.
    # If it fails to add the paper for any reason, it will still make a 'failure' record in the paperstore, so that we don't try to spend resources trying to add the paper again.
    def add_paper_start(self, corpus_id:str=None, s2_id:str=None, paper_title:str=None):
        # Parameter checking: Check that the keys aren't empty
        if (corpus_id is not None) and (len(corpus_id.strip()) == 0):
            print("WARNING: add_paper_start(): corpus_id is empty. Skipping.")
            corpus_id = None
        if (s2_id is not None) and (len(s2_id.strip()) == 0):
            print("WARNING: add_paper_start(): s2_id is empty. Skipping.")
            s2_id = None
        if (paper_title is not None) and (len(paper_title.strip()) == 0):
            print("WARNING: add_paper_start(): paper_title is empty. Skipping.")
            paper_title = None

        # Parameter Checking: At least one of the parameters must be provided
        if (corpus_id is None) and (s2_id is None) and (paper_title is None):
            print("WARNING: add_paper_start(): Must provide at least one of corpus_id, s2_id, or paper_title.")
            return None

        # If there's no corpusID, then try to get a corpusID, because our internal cache uses it as a lookup key and it's vastly faster than re-fetching the paper.
        if (corpus_id is None):
            # If there's an S2 ID, then try to look it up
            if (s2_id is not None):
                print("add_paper_start(): Looking up paper by S2 ID using S2 API: " + s2_id)
                paper_data = self.s2_api.find_paper_by_paperid(s2_id)
                if (isinstance(paper_data, dict)):
                    corpus_id = paper_data.get("corpus_id", None)
                    if (corpus_id is not None):
                        print(f"add_paper_start(): Found corpus_id: {corpus_id} for paper with S2 ID: {s2_id}")

            # If there's still no corpus ID but is a paper title, then try to look it up
            if (corpus_id is None) and (paper_title is not None):
                self.add_log_entry(f"add_paper_start(): Looking up paper by title using S2 API: {paper_title}")
                paper_data = self.s2_api.find_paper_by_title(paper_title)
                if (paper_data is None):
                    # We weren't able to look-up the paper by title
                    pass
                else:
                    data = paper_data.get("data", [])
                    paper_record = None
                    if (data is not None) and (len(data) > 0):
                        paper_record = data[0]  # Get the first paper record
                    if (paper_record is not None):
                        corpus_id = paper_record.get("corpus_id", None)
                        if (corpus_id is not None):
                            print(f"add_paper_start(): Found corpus_id: {corpus_id} for paper with title: {paper_title}")


        # Duplicate checking: Check if the paper already exists in the paperstore
        if (corpus_id is not None):
            corpus_id_key = self._mk_key_corpus_id(corpus_id)
            with self.THREAD_LOCK_PAPERSTORE:
                if (corpus_id_key in self.paperstore_lut):
                    print("add_paper_start(): Paper already exists in the paperstore with corpus_id: " + corpus_id)
                    self.add_log_entry("add_paper_start(): Paper already exists in the paperstore with corpus_id: " + corpus_id)
                    return self.paperstore_lut[corpus_id_key]
        if (s2_id is not None):
            s2_id_key = self._mk_key_s2_sha(s2_id)
            with self.THREAD_LOCK_PAPERSTORE:
                if (s2_id_key in self.paperstore_lut):
                    print("add_paper_start(): Paper already exists in the paperstore with s2_id: " + s2_id)
                    self.add_log_entry
                    return self.paperstore_lut[s2_id_key]
        if (paper_title is not None):
            title_key = self._mk_key_title(paper_title)
            with self.THREAD_LOCK_PAPERSTORE:
                if (title_key in self.paperstore_lut):
                    print("add_paper_start(): Paper already exists in the paperstore with title: " + paper_title)
                    self.add_log_entry("add_paper_start(): Paper already exists in the paperstore with title: " + paper_title)
                    return self.paperstore_lut[title_key]

        # Wrap in a timeout with a 60 second limit
        MAX_ADD_PAPER_TIMEOUT = 60

        ###################################
        ### INTERNAL (S2ORC cache — disabled by default)
        ###################################

        if self.USE_S2ORC_INTERNAL_CACHE:
            self.add_log_entry("add_paper_start(): Looking for paper in internal cache...  corpus_id=" + str(corpus_id) + ", s2_id=" + str(s2_id) + ", paper_title=" + str(paper_title) + " using internal cache.")
            try:
                primary_key = func_timeout(MAX_ADD_PAPER_TIMEOUT, self._add_paper_internal, args=(corpus_id, s2_id, paper_title))
                if (primary_key is not None):
                    print(f"Successfully added paper: {paper_title} (S2 ID: {s2_id}) with primary key: {primary_key}")
                    self.add_log_entry("add_paper_start(): Successfully added paper from internal cache.  corpus_id=" + str(corpus_id) + ", s2_id=" + str(s2_id) + ", paper_title=" + str(paper_title) + ", primary_key=" + str(primary_key))
                    return primary_key
            except FunctionTimedOut:
                print(f"ERROR: add_paper() timed out after {MAX_ADD_PAPER_TIMEOUT} seconds. Paper title: {paper_title}, S2 ID: {s2_id}")
            except Exception as e:
                print(f"ERROR: add_paper() encountered an error: {e}. Paper title: {paper_title}, S2 ID: {s2_id}")

        ###################################
        ###################################

        # Try to fetch the paper externally
        self.add_log_entry("add_paper_start(): Looking for paper by fetching manually...  corpus_id=" + str(corpus_id) + ", s2_id=" + str(s2_id) + ", paper_title=" + str(paper_title))
        try:
            primary_key = func_timeout(MAX_ADD_PAPER_TIMEOUT, self._add_paper, args=(corpus_id, s2_id, paper_title))
            if (primary_key != None):
                print(f"Successfully added paper: {paper_title} (S2 ID: {s2_id}) with primary key: {primary_key}")
                self.add_log_entry("add_paper_start(): Successfully added paper by fetching manually.  corpus_id=" + str(corpus_id) + ", s2_id=" + str(s2_id) + ", paper_title=" + str(paper_title) + ", primary_key=" + str(primary_key))
                return primary_key

        except FunctionTimedOut:
            print(f"ERROR: add_paper() timed out after {MAX_ADD_PAPER_TIMEOUT} seconds. Paper title: {paper_title}, S2 ID: {s2_id}")
        except Exception as e:
            print(f"ERROR: add_paper() encountered an error: {e}. Paper title: {paper_title}, S2 ID: {s2_id}")

        # If we reach here, then the paper could not be added (for whatever reason).
        # Add a faux record, with a failure status, so that we don't try to go through the time-consuming process of trying to add this paper again.
        print(f"Failed to add paper: {paper_title} (S2 ID: {s2_id}).")
        self.add_log_entry("add_paper_start(): Failed to add paper.  corpus_id=" + str(corpus_id) + ", s2_id=" + str(s2_id) + ", paper_title=" + str(paper_title))
        paper_record = {
            "status": PaperStatus.FAILED.value,
        }
        # Add this paper to the paperstore with a failure status
        primary_key = self._add_paper_by_keys(paper_record, corpus_id=corpus_id, s2_sha=s2_id, title=paper_title, alt_title=None)
        return primary_key


    #########################################################################
    ########################## INTERNAL ######################################
    #########################################################################

    # Try to retrieve the paper using the internal cache
    def _add_paper_internal(self, corpus_id:str=None, s2_id:str=None, paper_title:str=None):
        original_title = paper_title # Save the original title, in case we need to use it later
        if (corpus_id is None):
            return None

        # Try to retrieve the paper by its corpus ID
        cache_url = os.environ.get("PAPER_CACHE_BASE_URL", "").rstrip("/")
        if not cache_url:
            return None
        paper_url = cache_url + "/paper/" + str(corpus_id)

        # Get the JSON response using the requests library
        def get_paper_json(url):
            import requests
            try:
                response = requests.get(url, timeout=10)  # Set a timeout for the request
                response.raise_for_status()  # Raise an error for bad responses
                return response.json()
            except requests.RequestException as e:
                print(f"Error retrieving paper data from {url}: {e}")
                return None
        # Use the function timeout to limit the time spent on this request
        paper_data = None
        try:
            self.add_log_entry(f"_add_paper_internal(): Retrieving paper data from URL: {paper_url}")
            paper_data = func_timeout(10, get_paper_json, args=(paper_url,))
        except FunctionTimedOut:
            print(f"ERROR: Timeout while trying to retrieve paper data from {paper_url}")
            self.add_log_entry(f"ERROR: Timeout while trying to retrieve paper data from {paper_url}")
            return None

        if (paper_data is None):
            print(f"ERROR: Could not retrieve paper data from {paper_url}")
            return None

        # It should return as a dictionary
        if (not isinstance(paper_data, dict)):
            print(f"ERROR: Paper data from {paper_url} is not a valid JSON dictionary.")
            print("Paper data:")
            print(str(paper_data))
            return None

        # Step 1B: Get the corpus_id, s2_id, and title from the paper data
        failure = False

        # Title
        metadata = paper_data.get("metadata", {})
        paper_title = metadata.get("title", None)
        if (paper_title is None):
            print("ERROR: Paper data does not contain a valid title.")
            failure = True

        # Failure to retrieve any of the above items is considered a failure
        if (failure):
            # Add a record to the failed papers
            return None

        # Step 2: Get a list of possible URLs to find this paper at
        paper_urls = []
        print("add_paper(): Found paper URLs for paper (`" + paper_title + "`):")
        content = paper_data.get("content", {})
        source = content.get("source", {})
        if ("pdf_uri" in source):
            pdf_uri = source["pdf_uri"]
            if (isinstance(pdf_uri, str)):
                paper_urls.append(pdf_uri)
            elif (isinstance(pdf_uri, list)):
                for url in pdf_uri:
                    if (isinstance(url, str)):
                        paper_urls.append(url)
            else:
                print(f"ERROR: Unexpected type for pdf_uri: {type(pdf_uri)}. Expected str or list of str.")

        paper_text = content.get("grobid", {}).get("contents", None)
        # It's not markdown, but we'll call it markdown.
        paper_markdown = paper_text

        # Pick a status
        status = PaperStatus.FULL_TEXT.value        # Default: assume the paper processing worked

        # If we have no paper markdown, then set the status to NO_PDF
        if (paper_markdown is None):
            status = PaperStatus.METADATA_ONLY.value
        # If the markdown is empty or too short, set the status to NO_PDF
        elif (len(paper_markdown.strip()) <= 100):
            status = PaperStatus.METADATA_ONLY.value
        else:
            print("add_paper(): Appear to have successfully retrieved paper markdown for paper (`" + paper_title + "`). Length of markdown: " + str(len(paper_markdown)))

        # Additional metadata: Break down the publication date into year and month
        publication_date = metadata.get("publication_date", {})
        publication_year = publication_date.get("year", None)
        publication_month = publication_date.get("month", None)

        # Create a paper record
        packed = {
            "status": status,
            "s2_id": s2_id,
            "corpus_id": corpus_id,
            "title": paper_title,
            "publication_year": publication_year,       # Publication year
            "publication_month": publication_month,     # Publication month
            "s2_metadata": paper_data,
            "paper_urls": paper_urls,                   # List of URLs for the paper PDF
            "paper_markdown": paper_markdown,
        }

        # Add this paper to the paperstore
        primary_key = self._add_paper_by_keys(packed, corpus_id=corpus_id, s2_sha=s2_id, title=paper_title, alt_title=original_title)

        return primary_key

    #########################################################################
    #########################################################################

    # Check how many papers have full-text available
    def get_num_papers_with_full_text(self, paper_keys:list):
        count = 0
        ids = set()
        for key in paper_keys:
            try:
                if (key is not None):
                    status = self.get_paper_by_key(key).get("status", None)
                    if (status == PaperStatus.FULL_TEXT.value):
                        count += 1
                        ids.add(key)
            except Exception as e:
                print(f"Error checking paper with key {key}: {e}")
        return count, list(ids)

    # Lazily build the asta-papers Client (primary full-text fetch via legal OA channels, JATS/XML preferred); returns None if the library isn't installed so the caller falls back to legacy URL+OCR.
    def _get_asta_papers_client(self):
        if self._asta_papers_client is not None:
            return self._asta_papers_client
        try:
            from asta_papers import Client
            from asta_papers.enums import FetchSource
        except ImportError:
            self.add_log_entry("asta-papers not installed; using legacy URL+OCR path only")
            return None
        converters = []
        if os.environ.get("MISTRAL_API_KEY"):
            try:
                from asta_papers.converters.mistral import MistralConverter
                converters.append(MistralConverter())
            except Exception as e:
                self.add_log_entry(f"asta-papers MistralConverter unavailable: {e!r}")
        kwargs = dict(
            email=os.environ.get("ASTA_PAPERS_EMAIL", "theorizer@allenai.org"),
            ncbi_api_key=os.environ.get("NCBI_API_KEY"),
            s2_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
            workers=int(os.environ.get("ASTA_PAPERS_WORKERS", "4")),
            enrich_identifiers=True,   # cross-resolve corpus_id -> DOI/PMID/PMCID before fetching
            disable_sources={FetchSource.UNPAYWALL},
        )
        if converters:
            kwargs["converters"] = converters
        self._asta_papers_client = Client(**kwargs)
        return self._asta_papers_client

    def _get_fulltext_from_nora_retrieval(self, corpus_id, paper_title):
        if NoraRetrievalFullText.is_enabled():
            try:
                nora_result = NoraRetrievalFullText.fetch_fulltext(
                    paper_id=f"CorpusId:{corpus_id}",
                    s2_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
                    log=self.log_and_print,
                )
                # Any text the endpoint serves counts as a hit. Text too short to be
                # a real extraction still gets caught below, where the paper's status
                # is set to METADATA_ONLY on a short fulltext whatever produced it.
                if (nora_result is not None) and nora_result.text:
                    self.log_and_print(
                        f"add_paper(): nora-retrieval olmOCR hit corpus_id={corpus_id} "
                        f"title={log_title(paper_title)} chars={len(nora_result.text)}")
                    return NoraRetrievalFullTextResult(fulltext=nora_result.text)
                else:
                    reason = nora_result.no_body_reason if nora_result is not None else None
                    self.log_and_print(
                        f"add_paper(): nora-retrieval olmOCR miss corpus_id={corpus_id} "
                        f"title={log_title(paper_title)} noBodyReason={reason}")
                    return NoraRetrievalFullTextResult(no_body_reason=reason)
            except Exception as e:
                self.log_and_print(
                    f"add_paper(): nora-retrieval olmOCR raised {type(e).__name__} "
                    f"corpus_id={corpus_id} title={log_title(paper_title)}: {e}")
                return NoraRetrievalFullTextResult()
        else:
            self.log_and_print(
                f"add_paper(): not using nora-retrieval olmOCR at all for "
                f"corpus_id={corpus_id} title={log_title(paper_title)}")
            return FullTextResult()

    def _get_fulltext_from_asta_papers(self, corpus_id, paper_title, paper_data):
        if os.environ.get("USE_ASTA_PAPERS") == "true":
            client = self._get_asta_papers_client()
            if client is not None:
                ext = paper_data.get("externalIds", {}) or {}
                try:
                    result = client.fetch(
                        doi=ext.get("DOI"),
                        pmid=str(ext.get("PubMed")) if ext.get("PubMed") else None,
                        pmcid=str(ext.get("PubMedCentral")) if ext.get("PubMedCentral") else None,
                        arxiv=ext.get("ArXiv"),
                        corpus_id=str(paper_data.get("corpusId")) if paper_data.get("corpusId") else None,
                        title=paper_title,
                    )
                    if result.success:
                        # A successful fetch always carries a license decision, whether or
                        # not markdown came back: asta-papers resolves the license from the
                        # bytes it fetched, while result.markdown is only populated when a
                        # converter ran (with no MISTRAL_API_KEY set, that is no PDF at all).
                        # So branch on success first and record the decision either way -
                        # it is what lets paper_restricted() answer for those papers instead
                        # of saying "we don't know" and stopping the chain.
                        # Capture provenance regardless of the license decision (for auditing).
                        prov = getattr(result, "provenance", None)
                        fetch_channel = prov.primary.value if prov and prov.primary else None
                        license_class = result.license_class.value if result.license_class else None
                        license_url = result.license_url
                        source_type = result.source_type.value if result.source_type else None
                        # License gate: only feed full text to the LLM when text-and-data mining is permitted.
                        llm_usable = bool(result.may_use_for_tdm)
                        # "hit" means the chain got usable text, not that asta-papers had
                        # the paper - so a TDM-denied fetch is a miss here, the same way a
                        # paper-restricted answer is a miss on the nora-retrieval path.
                        # The license= and llm_usable= fields below still say which it was.
                        outcome = "hit" if (result.markdown and llm_usable) else "miss"
                        self.log_and_print(
                            f"add_paper(): asta-papers {outcome} channel={fetch_channel} license={license_class} "
                            f"llm_usable={llm_usable} corpus_id={corpus_id} title={log_title(paper_title)}")
                        return AstaPapersFullTextResult(
                            fulltext=result.markdown if llm_usable else None,
                            fetch_channel=fetch_channel,
                            license_class=license_class,
                            license_url=license_url,
                            source_type=source_type,
                            llm_usable=llm_usable,
                        )
                    else:
                        self.log_and_print(
                            f"add_paper(): asta-papers miss "
                            f"corpus_id={corpus_id} title={log_title(paper_title)}")
                        return AstaPapersFullTextResult()
                except Exception as e:
                    self.log_and_print(
                        f"add_paper(): asta-papers raised {type(e).__name__} "
                        f"corpus_id={corpus_id} title={log_title(paper_title)}: {e}")
                    return AstaPapersFullTextResult()
            else:
                self.log_and_print(
                    f"add_paper(): not using asta-papers at all (no client) for "
                    f"corpus_id={corpus_id} title={log_title(paper_title)}")
                return FullTextResult()
        else:
            self.log_and_print(
                f"add_paper(): not using asta-papers at all for "
                f"corpus_id={corpus_id} title={log_title(paper_title)}")
            return FullTextResult()

    def _get_fulltext_from_legacy_path(self, corpus_id, paper_title, paper_data):
        paper_urls = self.populate_paper_pdf_urls(paper_data)
        print(
            f"add_paper(): legacy path Found paper URLs for paper "
            f"corpus_id={corpus_id} title={log_title(paper_title)}:")
        for url in paper_urls:
            print(f"\t- {url}")

        # Try to download the paper and convert it to text/markdown
        # How many of the paper's URLs to try before giving up. attempt is
        # 0-based, so the bound below compares attempt + 1 against it.
        max_attempts = 4
        for attempt, pdf_url in enumerate(paper_urls):
            # Use the first URL as the PDF URL
            print(f"Using PDF URL: {pdf_url}")
            self.log_and_print(
                f"add_paper(): legacy path Attempting to process PDF from URL: {pdf_url} (attempt {attempt + 1} of {max_attempts}) "
                f"corpus_id={corpus_id} title={log_title(paper_title)}")
            paper_markdown = self.ocr_store.process_pdf(pdf_url, generation_id=self.generation_id)

            if (paper_markdown is not None) and (len(paper_markdown.strip()) > 100):
                self.log_and_print(
                    f"add_paper(): legacy path hit Successfully processed PDF from URL: {pdf_url} "
                    f"corpus_id={corpus_id} title={log_title(paper_title)} (attempt {attempt + 1})")
                return LegacyFullTextResult(fulltext=paper_markdown, paper_urls=paper_urls)

            if (attempt + 1 >= max_attempts):
                self.log_and_print(
                    f"add_paper(): legacy path miss Failed to process PDFs from available URLs after {max_attempts} attempts. "
                    f"corpus_id={corpus_id} title={log_title(paper_title)} Skipping.")
                return LegacyFullTextResult(paper_urls=paper_urls)

        self.log_and_print(
            f"add_paper(): legacy path miss Failed to process PDFs from available URLs "
            f"corpus_id={corpus_id} title={log_title(paper_title)} Skipping.")
        return LegacyFullTextResult(paper_urls=paper_urls)

    # Try to retrieve a paper's S2 record (that may include open access URLs), download the paper, and convert it to full-text Markdown.
    def _add_paper(self, corpus_id:str=None, s2_id:str=None, paper_title:str=None):
        original_title = paper_title # Save the original title, in case we need to use it later

        # Minimally, we'll be given the paper title.  The S2 ID may need to be looked up.
        if (corpus_id is None) and (s2_id is None) and (paper_title is None):
            print("ERROR: add_paper(): Must provide at least one of corpus_id, s2_id, or paper_title.")
            return

        # Step 1: Look up the paper by s2_id, corpus_id, or title
        paper_data = None
        self.add_log_entry("add_paper(): Looking up paper... corpus_id=" + str(corpus_id) + ", s2_id=" + str(s2_id) + ", paper_title=" + str(paper_title))
        if (s2_id != None):
            paper_data = self.s2_api.find_paper_by_paperid(s2_id)
        elif (corpus_id != None):
            paper_data = self.s2_api.find_paper_by_corpusid(corpus_id)
        elif (paper_title is not None):
            # If the paper title is provided, look it up by title
            print("add_paper(): Finding paper with title: `" + str(paper_title) + "`")
            paper_data = self.s2_api.find_paper_by_title(paper_title)
            print("add_paper(): Found paper with title: `" + str(paper_title) + "`")

        # If we couuldn't find the paper data, then we can't proceed.
        if (paper_data is None):
            print("ERROR: add_paper(): Could not find paper data. (s2_id: " + str(s2_id) + ", corpus_id: " + str(corpus_id) + ", paper_title: `" + str(paper_title) + "`)")
            self.add_log_entry("ERROR: add_paper(): Could not find paper data. (s2_id: " + str(s2_id) + ", corpus_id: " + str(corpus_id) + ", paper_title: `" + str(paper_title) + "`)")
            return None

        # It may return as a dictionary with a single key, "data", containing the paper information
        if (isinstance(paper_data, dict) and ("data" in paper_data) and (isinstance(paper_data["data"], list))):
            paper_data = paper_data["data"][0]

        # Step 1B: Get the corpus_id, s2_id, and title from the paper data
        failure = False

        # Corpus ID
        corpusId = paper_data.get("corpusId", None)
        if (corpusId is None):
            print("ERROR: Paper data does not contain a valid corpus ID.")
            failure = True

        # S2 Paper ID
        paperId = paper_data.get("paperId", None)
        if (paperId is None):
            print("ERROR: Paper data does not contain a valid S2 ID.")
            failure = True
        s2_id = paperId

        # Title
        paper_title = paper_data.get("title", None)
        if (paper_title is None):
            print("ERROR: Paper data does not contain a valid title.")
            failure = True

        # Failure to retrieve any of these 3 items is considered a failure
        if (failure):
            # Add a record to the failed papers
            return None

        # Full-text sources, in the order they're consulted:
        #   1. nora-retrieval's olmOCR path (env-gated by USE_NORA_RETRIEVAL_FULLTEXT)
        #   2. asta-papers (env-gated by USE_ASTA_PAPERS)
        #   3. legacy: S2 open-access URLs + Mistral OCR
        # Each one runs only if the ones before it produced nothing.

        nora_retrieval_result = self._get_fulltext_from_nora_retrieval(
            corpus_id=corpusId,
            paper_title=paper_title,
        )
        if nora_retrieval_result.paper_restricted():
            self.log_and_print(
                f"add_paper(): nora-retrieval says this paper is restricted, not getting "
                f"fulltext for corpus_id={corpusId} title={log_title(paper_title)}")
            fulltext_result = nora_retrieval_result

        elif nora_retrieval_result.paper_restricted() is False:
            # check specifically for False, not just not X - None means we don't know

            if nora_retrieval_result.found_fulltext():
                fulltext_result = nora_retrieval_result

            else:

                asta_papers_result = self._get_fulltext_from_asta_papers(
                    corpus_id=corpusId,
                    paper_title=paper_title,
                    paper_data=paper_data,
                )
                if asta_papers_result.found_fulltext():
                    fulltext_result = asta_papers_result

                else:

                    # we don't check asta-papers' restriction info
                    # because we did a check through nora-retrieval
                    legacy_result = self._get_fulltext_from_legacy_path(
                        corpus_id=corpusId,
                        paper_title=paper_title,
                        paper_data=paper_data,
                    )
                    if not legacy_result.found_fulltext():
                        self.log_and_print(
                            f"add_paper(): no hits across the sources we tried for paper "
                            f"corpus_id={corpusId} title={log_title(paper_title)}")
                    fulltext_result = legacy_result

        else:

            # we didn't get an answer from nora-retrieval about this paper's restrictions
            # maybe nora-retrieval isn't enabled, maybe it hit an error
            # in any case, fallback to a decision from asta-papers
            asta_papers_result = self._get_fulltext_from_asta_papers(
                corpus_id=corpusId,
                paper_title=paper_title,
                paper_data=paper_data,
            )
            if asta_papers_result.paper_restricted():
                self.log_and_print(
                    f"add_paper(): asta-papers says this paper is restricted, not getting "
                    f"fulltext for corpus_id={corpusId} title={log_title(paper_title)}")
                fulltext_result = asta_papers_result

            elif asta_papers_result.paper_restricted() is False:
                # check specifically for False, not just not X - None means we don't know

                if asta_papers_result.found_fulltext():
                    fulltext_result = asta_papers_result

                else:

                    legacy_result = self._get_fulltext_from_legacy_path(
                        corpus_id=corpusId,
                        paper_title=paper_title,
                        paper_data=paper_data,
                    )
                    if not legacy_result.found_fulltext():
                        self.log_and_print(
                            f"add_paper(): no hits across the sources we tried for paper "
                            f"corpus_id={corpusId} title={log_title(paper_title)}")
                    fulltext_result = legacy_result

            else:
                # neither nora-retrieval nor asta-papers could tell us if this paper
                # is restricted
                self.log_and_print(
                    f"add_paper(): we don't know if this paper is restricted, not getting "
                    f"fulltext for corpus_id={corpusId} title={log_title(paper_title)}")
                fulltext_result = FullTextResult()

        # Pick a status
        status = PaperStatus.FULL_TEXT.value        # Default: assume the paper processing worked

        # If we have no paper markdown, then set the status to NO_PDF
        if (not fulltext_result.found_fulltext()):
            status = PaperStatus.METADATA_ONLY.value
        # If the markdown is empty or too short, set the status to NO_PDF
        elif (len(fulltext_result.fulltext.strip()) <= 100):
            status = PaperStatus.METADATA_ONLY.value

        self.log_and_print(
            f"add_paper(): status={status} corpus_id={corpusId} title={log_title(paper_title)}")

        # Additional metadata: Break down the publication date into year and month
        publication_date = paper_data.get("publicationDate", None)
        publication_year = None
        publication_month = None
        if (publication_date is not None):
            fields = publication_date.split("-")
            if (len(fields) >= 3):
                try:
                    publication_year = int(fields[0])
                except ValueError:
                    print(f"Error parsing publication year from date: {publication_date}")

                try:
                    publication_month = int(fields[1])
                except ValueError:
                    print(f"Error parsing publication month from date: {publication_date}")

        # Create a paper record
        packed = {
            "status": status,
            "s2_id": s2_id,
            "corpus_id": corpus_id,
            "title": paper_title,
            "publication_year": publication_year,       # Publication year
            "publication_month": publication_month,     # Publication month
            "s2_metadata": paper_data,
            "paper_urls": fulltext_result.paper_urls,                   # List of URLs for the paper PDF (only the legacy path collects these; [] otherwise)
            "paper_markdown": fulltext_result.fulltext,
            "fetch_channel": fulltext_result.fetch_channel,             # an asta-papers channel (None on the legacy and nora-retrieval paths)
            "license_class": fulltext_result.license_class,             # asta-papers LicenseClass value (None on the legacy and nora-retrieval paths)
            "license_url": fulltext_result.license_url,                 # asta-papers license URL (None if none, and on the legacy and nora-retrieval paths))
            "source_type": fulltext_result.source_type,                 # asta-papers SourceType value (None on the legacy and nora-retrieval paths)
            "llm_usable": fulltext_result.llm_usable,                   # True/False if asta-papers license-gated; None on the legacy and nora-retrieval paths (markdown kept)
        }

        # Add this paper to the paperstore
        self.add_log_entry("add_paper(): Adding paper to paperstore... corpus_id=" + str(corpus_id) + ", s2_id=" + str(s2_id) + ", paper_title=" + str(paper_title))
        primary_key = self._add_paper_by_keys(packed, corpus_id=corpus_id, s2_sha=s2_id, title=paper_title, alt_title=original_title)

        return primary_key


    # Helper function to try to populare URLs for papers from the S2 data
    def populate_paper_pdf_urls(self, s2_data:dict):
        self.add_log_entry("populate_paper_pdf_urls(): Populating paper PDF URLs from S2 data.")

        urls = []
        # Populate best-guess URLs for the paper PDF
        if (s2_data is None):
            return []

        paper_title = s2_data.get("title", "")
        if (paper_title is None) or (len(paper_title.strip()) < 10):
            print("ERROR: Paper data does not contain a valid title.")
            return []

        # Check if the S2 data has a PDF URL
        try:
            if ("openAccessPdf" in s2_data) and ("url" in s2_data["openAccessPdf"]):
                url = s2_data["openAccessPdf"]["url"]
                if (url is not None) and (len(url.strip()) > 0):
                    urls.append(url.strip())
        except Exception as e:
            print(f"Error extracting openAccessPdf URL: {e}")

        # Check for ArXiv ID
        try:
            if ("externalIds" in s2_data) and ("ArXiv" in s2_data["externalIds"]):
                arxiv_id = s2_data["externalIds"]["ArXiv"]
                if (arxiv_id is not None) and (len(arxiv_id.strip()) > 0):
                    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    urls.append(url.strip())
        except Exception as e:
            print(f"Error extracting ArXiv ID: {e}")

        # Check for ACL ID
        try:
            if ("externalIds" in s2_data) and ("ACL" in s2_data["externalIds"]):
                acl_id = s2_data["externalIds"]["ACL"]
                if (acl_id is not None) and (len(acl_id.strip()) > 0):
                    url = f"https://aclanthology.org/{acl_id}.pdf"
                    #urls.append(url.strip())
                    # This is high quality -- put it on the top of the list
                    urls.insert(0, url.strip())
        except Exception as e:
            print(f"Error extracting ACL ID: {e}")

        # If there are still no PDF links, then try an Arxiv search to see if this paper is on Arxiv
        if (len(urls) == 0):
            self.add_log_entry("populate_paper_pdf_urls(): No PDF URLs found in S2 data. Trying to find paper using Arxiv API by title: `" + paper_title + "`")
            best_match = find_pdf_arxiv(paper_title)
            if (best_match is not None) and ("id" in best_match):
                arxiv_id = best_match["id"].split("/")[-1]
                url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                urls.append(url.strip())
            self.add_log_entry("populate_paper_pdf_urls(): Arxiv API search complete.")

        # Return the list of URLs
        return urls


# Helper function to query the Arxiv API for a paper (by title)
def find_pdf_arxiv(paper_title:str):
    #global LAST_ARXIV_API_CALL
    #MIN_TIME_BETWEEN_REQUESTS_SEC = 3.5
    # Use the Arxiv API to find a paper by title
    import requests
    import time

#   with THREAD_LOCK_ARXIV_API:

    try:
        # # Step 0: Check how long it's been since the last Arxiv API call
        # current_time = time.time()
        # if (current_time - LAST_ARXIV_API_CALL < MIN_TIME_BETWEEN_REQUESTS_SEC):
        #     # If it's been less than the minimum time, wait for the remaining time
        #     wait_time = MIN_TIME_BETWEEN_REQUESTS_SEC - (current_time - LAST_ARXIV_API_CALL)
        #     print(f"Waiting for {wait_time:.2f} seconds before making the next Arxiv API call...")
        #     time.sleep(wait_time)
        # # Update the last Arxiv API call time
        # LAST_ARXIV_API_CALL = time.time()

        # Step 0 (replaced): The reality is that the calls are not quick, and we're likely not hitting the rate limit, even across threads.
        time.sleep(2.0)  # Sleep for 2 seconds before making the API call, just to be safe and avoid hitting rate limits.


        # Step 1: Make a request to the Arxiv API
        # Endpoint: http://export.arxiv.org/api/{method_name}?{parameters}
        base_url = "http://export.arxiv.org/api/query?"
        search_query = f"search_query={paper_title.replace(' ', '+')}"
        url = base_url + search_query
        print(f"Searching Arxiv for paper with title: `{paper_title}`")
        response = requests.get(url)

        if (response.status_code != 200):
            print(f"Error: {response.status_code} - {response.text}")
            return None

        # Step 2: Extract the response (not XML/etc, just the raw text content)
        content = response.text

        # Convert all the Atom XML entires to JSON
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)
        entries = root.findall('{http://www.w3.org/2005/Atom}entry')
        papers = []
        for entry in entries:
            paper = {
                "id": entry.find('{http://www.w3.org/2005/Atom}id').text,
                "updated": entry.find('{http://www.w3.org/2005/Atom}updated').text,
                "published": entry.find('{http://www.w3.org/2005/Atom}published').text,
                "title": entry.find('{http://www.w3.org/2005/Atom}title').text,
                "summary": entry.find('{http://www.w3.org/2005/Atom}summary').text,
                "authors": [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
                "links": {link.get('rel'): link.get('href') for link in entry.findall('{http://www.w3.org/2005/Atom}link')}
            }
            papers.append(paper)

        # Step 3: Try to find the paper with the best title match
        def normalize_title(s:str):
            import re
            s = s.lower()
            s = re.sub(r"[^a-z0-9\s]", "", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        # Returns a float
        def similarity(a:str, b:str):
            from difflib import SequenceMatcher
            return SequenceMatcher(None, a, b).ratio()

        # Find the best match
        best_match = None
        best_score = 0.0
        MIN_THRESH = 0.95
        normalized_query_str = normalize_title(paper_title)
        for paper in papers:
            normalized_paper_title = normalize_title(paper.get("title", ""))
            score = similarity(normalized_query_str, normalized_paper_title)
            print(f"Title: {normalized_paper_title}, Score: {score}")

            if score > best_score:
                best_score = score
                best_match = paper

        if (best_match is None) or (best_score < MIN_THRESH):
            print(f"No good match found for paper title: {paper_title} (best score: {best_score})")
            return None
        print(f"Best match found: {best_match['title']} (score: {best_score})")
        # Step 4: Print the best match and return it
        return best_match

    except Exception as e:
        print(f"Error finding paper on Arxiv: {e}")
        return None



# Stand-alone examples/tests.
if __name__ == "__main__":
    # Make an instance of the class
    paperstore = PaperStore()

    s2_id = "0e41fd22d483047bd8fdb1757d90c7702493567e"
    paper_data = paperstore.s2_api.find_paper_by_paperid(s2_id)
    print(json.dumps(paper_data, indent=4))
    print("\n\n")
    #paper_title = "DiscoveryWorld: a virtual environment for developing and evaluating automated scientific discovery agents"
    #paper_data = paperstore.s2_api.find_paper_by_title(paper_title)
    #print(json.dumps(paper_data, indent=4))

    exit(1)

    # Submit a number of sample requests
    request_ids = []
    # request_ids.append(paperstore.submit_paper(title="DiscoveryWorld: a virtual environment for developing and evaluating automated scientific discovery agents"))
    # request_ids.append(paperstore.submit_paper(corpus_id="12345", title="A Study on Birds"))
    # request_ids.append(paperstore.submit_paper(s2_sha="abcde12345", title="A Study on Turtles"))
    # request_ids.append(paperstore.submit_paper(title="ScienceWorld: Is your Agent Smarter than a 5th Grader?"))
    # request_ids.append(paperstore.submit_paper(title="A Study on Large Language Models"))

    # # Duplicate detection
    # request_ids.append(paperstore.submit_paper(title="DiscoveryWorld: a virtual environment for developing and evaluating automated scientific discovery agents"))
    # request_ids.append(paperstore.submit_paper(corpus_id="12345", title="A Study on Birds"))
    # request_ids.append(paperstore.submit_paper(s2_sha="abcde12345", title="A Study on Turtles"))
    # request_ids.append(paperstore.submit_paper(title="ScienceWorld: Is your Agent Smarter than a 5th Grader?"))
    # request_ids.append(paperstore.submit_paper(title="A Study on Large Language Models"))

    # Test of the (internal) cache
    request_ids.append(paperstore.submit_paper(corpus_id="215416146"))

    # Example of waiting for the requests to complete
    completed_requests = {}
    while (len(completed_requests) < len(request_ids)):
        # Check to see if the work is completed
        for request_id in request_ids:
            # Check to see if their work has already been retrieved
            if (request_id not in completed_requests):
                paper_key = paperstore.is_paper_request_complete(request_id)
                if (paper_key is not None):
                    completed_requests[request_id] = paper_key
                    print("main: Retrieved work for request ID '" + str(request_id) + "': " + str(paper_key))

        # Sleep (so we're not busy waiting)
        time.sleep(1)

    # Show the completed requests
    print("Completed requests:")
    for request_id, paper_key in completed_requests.items():
        print(f"Request ID: {request_id}, Paper Key: {paper_key}")

    # Save the paperstore to a test file
    paperstore.save("paperstore.debug.json")

    # Try to get the paper text of each paper
    print("Getting paper text (by key):")
    for primary_key in completed_requests.values():
        print(f"Retrieving paper text for primary key: {primary_key}")
        # Get the paper text by its key
        paper_text = paperstore.get_paper_text_by_key(primary_key)
        # Show the first 100 characters of the paper text
        if (paper_text is not None):
            print(f"Paper text for {primary_key}: {paper_text[:100]}...")
        else:
            print(f"Paper text for {primary_key} is None.")


    time.sleep(10)

    # Stop the thread
    paperstore.stop_thread()
