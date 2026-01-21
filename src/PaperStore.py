# PaperStore.py
# A class to handle finding papers, retrieving their full-text, and making it available to threads that ask for it.

from __future__ import annotations

import os
import json
import time

# Timeout
from func_timeout import func_timeout, FunctionTimedOut

# Threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading


from SemanticScholar import *
from MistralOCRStore import *

# Make an enumeration for different paper statuses (e.g. full-text, metadata-only, failed, timeout)
from enum import Enum
class PaperStatus(Enum):
    FULL_TEXT = "full-text"               # A status for papers that have full-text available
    METADATA_ONLY = "metadata-only"       # A status for papers that have metadata only (no full-text)"
    FAILED = "failed"                     # A status for papers that failed to be processed (for whatever reason)
    TIMEOUT = "timeout"                   # A status for papers that timed out during processing

# Global thread lock/Rate limiting for Arxiv API
THREAD_LOCK_ARXIV_API = Lock()
# Time of the last Arxiv API call
LAST_ARXIV_API_CALL = 0




class PaperStore():
    #
    #   Constructor
    #
    def __init__(self, filename_in:str=None):
        # Initialize the Mistral OCR store
        self.ocr_store = MistralOCRStore(api_key=None)      # Will automatically load the API key

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

        # Parameters
        self.MIN_TIME_BETWEEN_REQUESTS_SECS = 0.2        # The minimum time between subsequent paper retrieval requests
        self.required_wait_time_last_request = 1.0       # A request-dependent wait time after the last request. (Certain API calls require longer wait times).
        self.last_request_time = 0                       # The timestamp of the last request

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

        # Start a background thread to process the queue
        self.worker_thread = None
        self.actively_processing_work = False
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
            if (self.actively_processing_work):
                return True

        # If we reach here, then we're not busy
        return False


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



    #
    #   Main thread: Process the request queue
    #

    # This should be running in the background.
    def process_work_monitor(self):
        #MIN_WAIT_TIME = 0.2
        MIN_WAIT_TIME = 2.0

        print("PaperStore.process_work_monitor(): Worker thread started.")
        while (self.THREAD_ACTIVE):
            # Step 0: Check if the heartbeat interval has passed
            time_since_last_heartbeat = time.time() - self.heartbeat_last_time
            if (time_since_last_heartbeat >= self.heartbeat_interval):
                # Print a heartbeat message
                print("PaperStore.process_work_monitor(): Heartbeat: Worker thread is alive. (queue size: " + str(len(self.queue)) + ")")
                self.add_log_entry("PaperStore.process_work_monitor(): Heartbeat: Worker thread is alive. (queue size: " + str(len(self.queue)) + ")")
                self.heartbeat_last_time = time.time()


            #print("PaperStore.process_work_monitor(): Worker thread is running...")
            # Step 1: See if enough time has passed since the last request
            time_since_last_request = time.time() - self.last_request_time
            #if (time_since_last_request < self.MIN_TIME_BETWEEN_REQUESTS_SECS):
            if (time_since_last_request < self.required_wait_time_last_request):
                # Sleep for a short time to avoid busy waiting
                time.sleep(MIN_WAIT_TIME)
                continue

            # Step 2: If we're here, then enough time has passed since the last request.  Look for work to perform
            request = None
            with self.THREAD_LOCK_PAPERSTORE:
                # Show queue size periodically
                #print("PaperStore.process_work_monitor(): Checking for requests in the queue... (queue size: " + str(len(self.queue)) + ")")

                # Check if there are any requests in the queue
                if (len(self.queue)) == 0:
                    # No requests in the queue, so wait a bit before checking again
                    time.sleep(MIN_WAIT_TIME)
                    continue

                # Get the next request
                request = self.queue.pop(0)


            # Step 2A: If the request is not valid, then sleep and check again in the next iteration
            if (request is None):
                # No request to process, so wait a bit before checking again
                time.sleep(MIN_WAIT_TIME)
                continue


            # Step 2B: If we reach here, then it's time to process the request
            # TODO: Process request
            with self.THREAD_LOCK_PAPERSTORE:
                self.actively_processing_work = True

            try:
                self.add_log_entry("PaperStore.process_work_monitor(): Processing request to add paper: corpus_id=" + str(request.get('corpus_id', None)) + ", s2_sha=" + str(request.get('s2_sha', None)) + ", title=" + str(request.get('title', None)))
                paper_key = self.add_paper_start(corpus_id=request.get('corpus_id', None), s2_id=request.get('s2_sha', None), paper_title=request.get('title', None))
                # Step 2C: Store the work result
                self._set_paper_as_processed(request['id'], paper_key)
            except Exception as e:
                import traceback
                error_str = "PaperStore.process_work_monitor(): Error processing request '" + request['id'] + "': " + str(e) + "\n" + traceback.format_exc()
                self._set_paper_as_processed(request['id'], {"error": error_str})
                self.add_log_entry("ERROR: " + error_str)
                self.required_wait_time_last_request = 1.0   # On error, increase the wait time before the next request

            with self.THREAD_LOCK_PAPERSTORE:
                self.actively_processing_work = False

            # Step 3: Set the time of the last request
            self.last_request_time = time.time()

        # If we reach here, then the thread is being stopped
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
            packed = {
                "paperstore_lut": self.paperstore_lut,
                "paperstore": self.paperstore,
            }

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
        self.paperstore_lut = packed.get("paperstore_lut", {})
        self.paperstore = packed.get("paperstore", {})

        print("PaperStore.load(): Loaded " + str(len(self.paperstore)) + " papers from the file.")
        return True



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


        self.required_wait_time_last_request = 0.1   # Reset the wait time to the default value, signifying a fast request.

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

        self.required_wait_time_last_request = 2.0   # External (2 second refresh)

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


        # Step 2: Get a list of possible URLs to find this paper at
        paper_urls = self.populate_paper_pdf_urls(paper_data)
        print("add_paper(): Found paper URLs for paper (`" + paper_title + "`):")
        for url in paper_urls:
            print(f"\t- {url}")

        # Step 3: Try to download the paper and convert it to text/markdown
        paper_markdown = None
        max_attempts = 3    # Try at most 3 URLs
        for attempt, pdf_url in enumerate(paper_urls):
            # Use the first URL as the PDF URL
            print(f"Using PDF URL: {pdf_url}")
            self.add_log_entry(f"add_paper(): Attempting to process PDF from URL: {pdf_url} (attempt {attempt + 1} of {max_attempts})")
            paper_markdown = self.ocr_store.process_pdf(pdf_url)

            if (paper_markdown is not None) and (len(paper_markdown.strip()) > 100):
                print(f"Successfully processed PDF from {pdf_url} (attempt {attempt + 1})")
                self.add_log_entry(f"add_paper(): Successfully processed PDF from URL: {pdf_url} (attempt {attempt + 1})")
                break

            if (attempt >= max_attempts):
                print(f"Failed to process PDF from {pdf_url} after {max_attempts} attempts. Skipping.")
                self.add_log_entry(f"add_paper(): Failed to process PDF from URL: {pdf_url} after {max_attempts} attempts. Skipping.")
                paper_markdown = None
                break


        # Pick a status
        status = PaperStatus.FULL_TEXT.value        # Default: assume the paper processing worked

        # If we have no paper markdown, then set the status to NO_PDF
        if (paper_markdown is None):
            status = PaperStatus.METADATA_ONLY.value
        # If the markdown is empty or too short, set the status to NO_PDF
        elif (len(paper_markdown.strip()) <= 100):
            status = PaperStatus.METADATA_ONLY.value

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
            "paper_urls": paper_urls,                   # List of URLs for the paper PDF
            "paper_markdown": paper_markdown,
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
    global LAST_ARXIV_API_CALL
    MIN_TIME_BETWEEN_REQUESTS_SEC = 3.5
    # Use the Arxiv API to find a paper by title
    import requests
    import time

    with THREAD_LOCK_ARXIV_API:
        try:
            # Step 0: Check how long it's been since the last Arxiv API call
            current_time = time.time()
            if (current_time - LAST_ARXIV_API_CALL < MIN_TIME_BETWEEN_REQUESTS_SEC):
                # If it's been less than the minimum time, wait for the remaining time
                wait_time = MIN_TIME_BETWEEN_REQUESTS_SEC - (current_time - LAST_ARXIV_API_CALL)
                print(f"Waiting for {wait_time:.2f} seconds before making the next Arxiv API call...")
                time.sleep(wait_time)
            # Update the last Arxiv API call time
            LAST_ARXIV_API_CALL = time.time()

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
