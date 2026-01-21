# SchemaExtractionQueue.py
# A class to handle extracting information from papers using schemas.  The list of work is held in a queue, and processed in a background thread.

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

from Struct import *
from Theorizer import *
from PaperStore import *
from ExtractionUtils import *



class SchemaExtractionQueue():
    #
    #   Constructor
    #
    def __init__(self, theorystore:TheoryStore, paperstore:PaperStore):
        self.theorystore = theorystore
        self.paperstore = paperstore

        # Queue for adding papers to the paperstore
        self.queue = []
        self.finished_work = {}
        self.next_request_id = 1

        # Parameters
        self.MIN_TIME_BETWEEN_REQUESTS_SECS = 1        # The minimum time between subsequent requests
        self.last_request_time = 0                     # The timestamp of the last request

        # Update message: Show the user the PaperFinder heartbeat every 10 seconds
        self.heartbeat_interval = 10
        self.heartbeat_last_time = 0

        # Thread lock
        self.THREAD_LOCK_EXTRACTION_QUEUE = threading.Lock()
        self.THREAD_ACTIVE = False

        # Start a background thread to process the queue
        self.worker_thread = None
        self.actively_processing_work = False
        self.start_thread()

        # Keep track of the number of active threads
        self.active_threads = set()
        self.MAX_THREADS = 10           # The maximum number of concurrent extraction threads (each contains an LLM call)

    #
    #   Threading/Queue/Work Processing
    #



    # Add a new paper to the queue to add papers, by one (or more) of: corpus_id, s2_sha, or title.
    def submit_extraction_request(self, extraction_schema_id:str, paper_ids_to_extract_from:list[str], extraction_model_str:str='gpt-5-mini'):
        # Check: parameters must not be empty
        if (extraction_schema_id is None) or (len(extraction_schema_id.strip()) == 0):
            print("ERROR: SchemaExtractionQueue.submit_extraction_request(): extraction_schema_id is empty. Cannot submit request.")
            return None

        request_ids = []
        print("SchemaExtractionQueue.submit_extraction_request(): Submitting extraction request for schema ID '" + extraction_schema_id + "' applied to " + str(len(paper_ids_to_extract_from)) + " papers.")
        with self.THREAD_LOCK_EXTRACTION_QUEUE:
            # Generate an ID for the requests
            for paper_id in paper_ids_to_extract_from:
                request_id = "schema-extraction-request-" + str(self.next_request_id)
                self.next_request_id += 1

                # Add the request to the queue
                self.queue.append({
                    'id': request_id,
                    'extraction_schema_id': extraction_schema_id,
                    'paper_id': paper_id,
                    'extraction_model_str': extraction_model_str,
                })
                request_ids.append(request_id)

        print("SchemaExtractionQueue.submit_extraction_request(): Submitted extraction request for schema ID '" + extraction_schema_id + "' applied to " + str(len(paper_ids_to_extract_from)) + " papers.")
        return request_ids


    # Check to see if a paper has been added to the store, by its request ID.
    # Returns `None` if the paper has not been added yet.
    # Returns the key to access the paper, if it has been added
    def is_request_complete(self, request_id:str):
        with self.THREAD_LOCK_EXTRACTION_QUEUE:
            # Return a copy of the finished work
            if (request_id in self.finished_work):
                return self.finished_work[request_id]
            else:
                return None

    def get_completed_work(self, request_id:str):
        return self.is_request_complete(request_id)

    # Get the number of requests in the queue
    def get_num_requests(self):
        with self.THREAD_LOCK_EXTRACTION_QUEUE:
            return len(self.queue)

    def get_num_completed_requests(self):
        with self.THREAD_LOCK_EXTRACTION_QUEUE:
            return len(self.finished_work)

    # Returns true if there is anything in the queue, or anything currently being processed
    # TODO: Depricate this?
    def is_busy(self):
        # Check 1: Are there any requests in the queue?
        if (self.get_num_requests() > 0):
            return True
        # Check 2: Are we actively processing work?
        with self.THREAD_LOCK_EXTRACTION_QUEUE:
            if (self.actively_processing_work):
                return True

        # If we reach here, then we're not busy
        return False


    # Add finished work to the directory of finished work
    # When a paper request is finished, call this to set the nominal key that that paper should be referenced by.
    def _set_work_completed(self, request_id:str, extraction_result_id:str):
        with self.THREAD_LOCK_EXTRACTION_QUEUE:
            # Set the finished work
            self.finished_work[request_id] = extraction_result_id

    # Stop the worker thread
    def stop_thread(self):
        print("SchemaExtractionQueue.stop_thread(): Stopping worker thread...")
        self.THREAD_ACTIVE = False
        if (self.worker_thread is not None):
            self.worker_thread.join()
        self.worker_thread = None

    def start_thread(self):
        if (self.worker_thread is not None) and (self.worker_thread.is_alive()):
            print("SchemaExtractionQueue.start_thread(): WARNING: Worker thread is already running -- not starting a new one.")
            return
        print("SchemaExtractionQueue.start_thread(): Starting worker thread...")

        self.THREAD_ACTIVE = True
        # Start the worker thread
        self.worker_thread = threading.Thread(target=self.process_work_monitor, daemon=True)
        self.worker_thread.start()

        print("SchemaExtractionQueue.start_thread(): Worker thread started.")


    # Main thread: Process the request queue
    # This should be running in the background.
    def process_work_monitor(self):
        print("SchemaExtractionQueue.process_work_monitor(): Worker thread started.")
        while (self.THREAD_ACTIVE):
            time.sleep(1)   # Sleep for a bit to avoid busy waiting

            # Step 0: Check if the heartbeat interval has passed
            time_since_last_heartbeat = time.time() - self.heartbeat_last_time
            if (time_since_last_heartbeat >= self.heartbeat_interval):
                # Print a heartbeat message
                print("SchemaExtractionQueue.process_work_monitor(): Heartbeat: Worker thread is alive. (active threads: " + str(len(self.active_threads)) + ", queue size: " + str(len(self.queue)) + ")")
                self.heartbeat_last_time = time.time()

            # Perform one unit of work
            def perform_work(request):
                try:
                    print("SchemaExtractionQueue.process_work_monitor(): Processing request '" + str(request['id']) + "' for extraction schema ID '" + str(request['extraction_schema_id']) + "' on paper ID '" + str(request['paper_id']) + "'...")
                    extraction_result_id = self.extract_from_paper(extraction_schema_id=request['extraction_schema_id'], paper_id=request['paper_id'], model_str=request['extraction_model_str'])
                    print("SchemaExtractionQueue.process_work_monitor(): Request Returned: '" + str(request['id']) + "' for extraction schema ID '" + str(request['extraction_schema_id']) + "' on paper ID '" + str(request['paper_id']) + "'...")
                    # Step 2C: Store the work result
                    if (extraction_result_id is None):
                        print("SchemaExtractionQueue.process_work_monitor(): Request '" + str(request['id']) + "' failed to extract from paper ID '" + str(request['paper_id']) + "'.")
                        self._set_work_completed(request['id'], extraction_result_id="") # Set the result to "" on failure, to signify the result was completed
                    else:
                        # Success
                        self._set_work_completed(request['id'], extraction_result_id=extraction_result_id)
                    print("SchemaExtractionQueue.process_work_monitor(): Request '" + str(request['id']) + "' completed successfully with extraction result ID: " + str(extraction_result_id))

                except Exception as e:
                    import traceback
                    error_str = f"SchemaExtractionQueue.process_work_monitor(): Error processing request '{request['id']}': {str(e)}\n{traceback.format_exc()}"
                    print(error_str)
                    self._set_work_completed(request['id'], extraction_result_id={"error": error_str})  # Set the result to None on failure

            # Multi-threaded version
            # Step A: Check if any of the active threads have returned results
            with self.THREAD_LOCK_EXTRACTION_QUEUE:
                for request_id in list(self.active_threads):
                    if (request_id in self.finished_work):
                        # If the request is complete, remove it from the active threads
                        print(f"SchemaExtractionQueue.process_work_monitor(): Request '{request_id}' has completed. Removing from active threads.")
                        self.active_threads.remove(request_id)

            # Step B: If we're less than the maximum number of threads, and the queue has work, then start new threads up to the maximum number of threads
            with self.THREAD_LOCK_EXTRACTION_QUEUE:
                while (len(self.active_threads) < self.MAX_THREADS) and (len(self.queue) > 0):
                    # Get the next request
                    request = self.queue.pop(0)
                    # Start a new thread to process the request
                    print(f"SchemaExtractionQueue.process_work_monitor(): Starting new thread for request '{request['id']}'...")
                    thread = threading.Thread(target=perform_work, args=(request,), daemon=True)
                    thread.start()
                    # Add the thread to the active threads
                    self.active_threads.add(request['id'])


        # If we reach here, then the thread is being stopped
        print("SchemaExtractionQueue.process_work_monitor(): Worker thread has completed.")


    #
    #   Performing Work: Extracting content from papers
    #
    def extract_from_paper(self, extraction_schema_id:str, paper_id:str, model_str:str):
        # Wrapper for below, with a timeout
        MAX_EXTRACTION_TIME = 60*5  # Maximum time to wait for extraction to complete (in seconds)
        try:
            # Call the extraction function with a timeout
            result = func_timeout(MAX_EXTRACTION_TIME, self._extract_from_paper, args=(extraction_schema_id, paper_id, model_str))
            return result
        except FunctionTimedOut:
            print(f"ERROR: SchemaExtractionQueue.extract_from_paper(): Extraction for paper ID '{paper_id}' timed out after {MAX_EXTRACTION_TIME} seconds.")
            return ""   # Return blank, so signifies that this result was tried, but not extracted successfully
        # Catch all other exceptions
        except Exception as e:
            print(f"ERROR: SchemaExtractionQueue.extract_from_paper(): An error occurred while extracting from paper ID '{paper_id}': {str(e)}")
            return ""   # Return blank, so signifies that this result was tried, but not extracted successfully


    def _extract_from_paper(self, extraction_schema_id:str, paper_id:str, model_str:str="gpt-5-mini"):
        # TODO: This whole function should be wrapped in a timeout that gracefully fails if the LLM extraction request times out.

        # Step 1: Retrieve the extraction schema
        extraction_schema = self.theorystore.get_extraction_schema(id=extraction_schema_id)
        if (extraction_schema is None):
            print("ERROR: SchemaExtractionQueue.extract_from_paper(): Extraction schema with ID '" + extraction_schema_id + "' not found.")
            return None

        # Step 2: Retrieve the paper
        paper_text = self.paperstore.get_paper_text_by_key(paper_id)
        if (paper_text is None):
            print("ERROR: SchemaExtractionQueue.extract_from_paper(): Full-text for paper with ID '" + paper_id + "' not found.")
            return None

        # Step 3: Perform the extraction
        max_tokens = 32000

        # Call the extraction function
        print("SchemaExtractionQueue.extract_from_paper(): Starting extraction for paper ID: " + paper_id + ".")
        extraction_query = extraction_schema.extraction_query
        extraction_schema = extraction_schema.schema
        extraction_result_output = self.extract_entities_from_paper(extraction_query, extraction_schema, paper_text, model_str=model_str, max_tokens=max_tokens, use_reflection=False)
        print("SchemaExtractionQueue.extract_from_paper(): Worker thread for paper ID: " + paper_id + " has received LLM extraction results.")

        response = extraction_result_output.get("response", None)
        if (response is None):
            print("ERROR: SchemaExtractionQueue.extract_from_paper(): Extraction failed for paper ID '" + paper_id + "'. No response received from LLM.")
            return None

        extraction_results = response.get("results", [])
        follow_on_papers = response.get("follow_on_papers", [])

        # Pack the extraction result
        extraction_result = ExtractionQueryResult(paper_id=paper_id, extraction_schema_id=extraction_schema_id, extracted_data=extraction_results, potentially_relevant_new_papers=follow_on_papers, cost=extraction_result_output.get("total_cost", 0.0), model_str=model_str)

        # Step 4: Store the extraction result
        extraction_result_id = self.theorystore.add_extraction_result(result=extraction_result)

        # Step 5: Return the extraction result ID
        return extraction_result_id


    # This is the main function that extracts extraction-schema-specified information (`added_keys`) from a scientific paper (`paper_text`), with the `query` in mind.
    def extract_entities_from_paper(self, query:str, added_keys:list, paper_text:str, model_str="gpt-4o-mini", max_tokens=8000, temperature=0.0, use_reflection:bool=False):
        def mkPrompt(query:str, paper_text:str, reflection:str=None):
            prompt = ""
            prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
            prompt += "\n"

            prompt += "# Task\n"
            prompt += "You will be given a scientific paper, and you will have to extract specific types of information from it based on the query, below.  You will return the results in a JSON format, also described below.\n"
            prompt += "The overall goal of the entities you extract is to use them for building scientific theories in the context of the query, so the knowledge that you extract must be very detailed, accurate, faithful, and scientifically rigorous.\n"
            prompt += "\n"

            prompt += "# Tools\n"
            prompt += "If you are a tool-calling model, it is important that you do not call any external web-search tools for this task.  Your information should come only from the scientific paper provided, and not (for example) a web search.\n"
            prompt += "\n"

            prompt += "# Query\n"
            prompt += "Here is the query:\n"
            prompt += "```\n"
            prompt += query + "\n"
            prompt += "```\n"
            prompt += "\n"

            prompt += "# Paper Text\n"
            prompt += "Here is the text of the scientific paper:\n"
            prompt += "```\n"
            prompt += paper_text + "\n"
            prompt += "```\n"

            if (reflection is not None):
                prompt += "\n"
                prompt += "# Reflection\n"
                prompt += "This is a reflection step. Previously, you generated a response (below).  Now, your task is to reflect on that response, and fix any errors, inconsistencies, omissions, or any other issues.\n"
                prompt += "```\n"
                prompt += reflection + "\n"
                prompt += "```\n"

            prompt += "# Output Format\n"
            prompt += "You must return your results in JSON format, between a single set of codeblocks (```).  While you can write any text to think and plan before writing your JSON response, your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
            prompt += "\n"
            prompt += "Your JSON response must be a dictionary, that contains two keys: `results`, and `follow_on_papers`."
            prompt += "`results` is a list of dictionaries, where each dictionary contains the following keys:\n"
            prompt += "- `name_short`(str): A short name for the entity (e.g. an acronym)\n"
            prompt += "- `name_full`(str): The full name of the entity (e.g. the full name of an acronym)\n"
            prompt += "- `brief_description`(str): A concise but information-dense description of the entity, in 1-2 sentences\n"
            prompt += "- `citation_title`(str): The exact title (and only the title) of the paper this entity appears to be from.  If the entity was first introduced in this paper, this response should be `here`.  If this information is not available with high confidence, leave this a blank string ("").\n"
            prompt += "- `mention_or_use`(str): A categorical string (`mention` or `use`) that indicates whether this entity is only mentioned in the paper (e.g. in an introduction/related work), or whether the paper actually uses this entity in any kind of experiment or analysis.\n"
            if (len(added_keys) > 0):
                prompt += "In addition, results must contain the following task-specific keys:\n"
                for key in added_keys:
                    key_name = key.get("name", "")
                    key_type = key.get("type", "str")
                    key_description = key.get("description", "")
                    if (key_name is not None) and (len(key_name.strip()) > 0):
                        prompt += f"- `{key_name}`({key_type}): {key_description}\n"

            prompt += "\n"
            prompt += "`follow_on_papers` is a list of dictionaries that suggest references from the paper that are likely to contain additional examples of these entities. The format is:\n"
            prompt += "- `paper_title`(str): The exact paper title (and only the title)\n"
            prompt += "- `rating`(int): A rating value. 2 means the paper almost certainly contains more examples, and 1 means the paper may. Do not include papers that are unlikely to contain additional examples.\n"
            prompt += "\n"
            prompt += "For example:\n"
            prompt += "```\n"
            prompt += "{\n"
            prompt += "    \"results\": ["
            prompt += '        {\n'
            prompt += '            "name_short": "GPT-4",\n'
            prompt += '            "name_full": "Generative Pre-trained Transformer 4",\n'
            prompt += '            "brief_description": "A large pre-trained transformer model developed by OpenAI. Trained on A, B, and C, and shows strong performance on a variety of benchmarks (e.g. X, Y, Z).",\n'
            prompt += '            "citation_title": "here",\n'
            prompt += '            "mention_or_use": "use"\n'
            for key in added_keys:
                key_name = key.get("name", "")
                if (key_name is not None) and (len(key_name.strip()) > 0):
                    prompt += f'            "{key_name}": ... # example_value,\n'
            prompt += '        },\n'
            prompt += "        ... # Add more entities as needed\n"
            prompt += '    ],\n'
            prompt += '    "follow_on_papers": [\n'
            prompt += '        {\n'
            prompt += '            "paper_title": "A Comprehensive Study on GPT-4",\n'
            prompt += '            "rating": 2\n'
            prompt += '        },\n'
            prompt += '        {\n'
            prompt += '            "paper_title": "Exploring the Capabilities of GPT-4",\n'
            prompt += '            "rating": 1\n'
            prompt += '        },\n'
            prompt += '        ... # Add more follow-on papers as needed\n'
            prompt += '    ]\n'
            prompt += "```\n"
            prompt += "\n"
            prompt += "# Important Notes\n"
            prompt += "- You must return a JSON response, and it must be valid JSON, or it will not be parsed.\n"
            prompt += "- You are encouraged to think and plan before writing your JSON response, but your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
            prompt += "- Values that are 'none'/'null' in the JSON response should be represented as `null` in the JSON, not as an empty string, string saying \"null\", or any other value.\n"
            prompt += "- Off-ramp: If this paper looks completely irrelevant/off-topic to the query, return blank lists for `results` and `follow_on_papers`, and we'll detect that and stop.\n"
            prompt += "- All your information must be accurate. Do not hallucinate.\n"

            return prompt


        total_cost = 0.0

        # Create the prompt
        prompt = mkPrompt(query, paper_text)

        # Call the LLM
        all_responses = []
        responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False)
        all_responses.append(responseJSON)
        total_cost += cost

        # Reflection: If we are using reflection, we will call the LLM again with the reflection step
        if (use_reflection):
            # Create the reflection prompt
            reflection_prompt = mkPrompt(query, paper_text, responseText)
            # Call the LLM again
            responseJSON, responseText, cost = getLLMResponseJSON(reflection_prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False)
            all_responses.append(responseJSON)
            total_cost += cost

        # Get the best response (the last one that's not None)
        best_response = None
        for response in reversed(all_responses):
            if (response is not None):
                best_response = response
                break

        # Pack output
        packed = {
            "query": query,
            "paper_text": paper_text,
            "model": model_str,
            "use_reflection": use_reflection,
            "raw_responses": all_responses,
            "response": best_response,
            "total_cost": total_cost,
        }

        return packed



# Example: Quick stand-alone example/test
if __name__ == "__main__":
    # Make placeholder instances of the TheoryStore and PaperStore
    theorystore = TheoryStore()
    paperstore = PaperStore()

    # Create an instance of the SchemaExtractionQueue
    extraction_queue = SchemaExtractionQueue(theorystore=theorystore, paperstore=paperstore)

    # Add 2 faux papers
    request_ids = []
    request_ids.append(paperstore.submit_paper(title="DiscoveryWorld: a virtual environment for developing and evaluating automated scientific discovery agents"))
    request_ids.append(paperstore.submit_paper(title="ScienceWorld: Is your Agent Smarter than a 5th Grader?"))

    # Wait until the papers are added
    paper_ids = []
    while (len(paper_ids) < len(request_ids)):
        paper_ids = []
        print("Waiting for papers to be added to the store...")
        for request_id in request_ids:
            paper_id = paperstore.is_paper_request_complete(request_id)
            if (paper_id is not None):
                paper_ids.append(paper_id)
                print("Paper added with ID: " + paper_id)
        time.sleep(1)

    paperstore.save("paperstore.debug.json")

    print("Adding extraction query...")
    extraction_schema = {"todo": "todo"}
    extraction_query_schema = ExtractionQuerySchema(schema=extraction_schema, extraction_query="Extract information about the paper", supporting_theory_ids=[])
    extraction_schema_id = theorystore.add_extraction_schema(extraction_query_schema)

    # Submit an extraction request for the papers
    extraction_request_ids = extraction_queue.submit_extraction_request(extraction_schema_id=extraction_schema_id, paper_ids_to_extract_from=paper_ids)

    # Wait for the extraction requests to complete
    completed_requests = []
    print("Extraction request ids: " + str(extraction_request_ids))
    while (len(completed_requests) < len(extraction_request_ids)):
        completed_requests = []
        print("Waiting for extraction requests to complete...")
        for request_id in extraction_request_ids:
            if (extraction_queue.is_request_complete(request_id) is not None):
                completed_requests.append(request_id)
                print("Extraction request completed: " + request_id)
        print("Waiting for extraction requests to complete... (completed: " + str(len(completed_requests)) + "/" + str(len(extraction_request_ids)) + ")")

        # Sleep for a bit before checking again
        time.sleep(1)

    print("All extraction requests completed.")

    pass
