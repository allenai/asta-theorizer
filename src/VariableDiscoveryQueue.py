# VariableDiscoveryQueue.py
# A class to handle trying to discover relevant classes of variables to extract from schemes.  The list of work is held in a queue, and processed in a background thread.

from __future__ import annotations

import os
import json
import time
import traceback

# Threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading

from Struct import *
from PaperStore import *
from ExtractionUtils import *



class VariableDiscoveryQueue():
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

        # Update message: Show the user the heartbeat every 10 seconds
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
        self.MAX_THREADS = 25           # The maximum number of concurrent extraction threads (each contains an LLM call)

    #
    #   Threading/Queue/Work Processing
    #



    # Add a new paper to the queue to add papers, by one (or more) of: corpus_id, s2_sha, or title.
    def submit_discovery_request(self, theory_query_information:dict, paper_ids_to_extract_from:list[str], additional_contextual_info=None, extraction_model_str:str='gpt-5-mini'):
        # Check: parameters must not be empty

        # Check if the theory_query_information is empty.
        if (not theory_query_information) or (len(theory_query_information) == 0):
            print("ERROR: VariableDiscoveryQueue.submit_discovery_request(): theory_query_information is empty. Cannot submit request.")
            return None

        request_ids = []
        print("VariableDiscoveryQueue.submit_discovery_request(): Submitting variable discovery request applied to " + str(len(paper_ids_to_extract_from)) + " papers.")
        with self.THREAD_LOCK_EXTRACTION_QUEUE:
            # Generate an ID for the requests
            for paper_id in paper_ids_to_extract_from:
                request_id = "variable-discovery-request-" + str(self.next_request_id)
                self.next_request_id += 1

                # Add the request to the queue
                self.queue.append({
                    'id': request_id,
                    'theory_query_information': theory_query_information,
                    'additional_contextual_info': additional_contextual_info,
                    'paper_id': paper_id,
                    'extraction_model_str': extraction_model_str,
                })
                request_ids.append(request_id)

        print("VariableDiscoveryQueue.submit_discovery_request(): Submitted variable discovery request applied to " + str(len(paper_ids_to_extract_from)) + " papers.")
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
    def _set_work_completed(self, request_id:str, discovery_result:str):
        with self.THREAD_LOCK_EXTRACTION_QUEUE:
            # Set the finished work
            self.finished_work[request_id] = discovery_result

    # Stop the worker thread
    def stop_thread(self):
        print("VariableDiscoveryQueue.stop_thread(): Stopping worker thread...")
        self.THREAD_ACTIVE = False
        if (self.worker_thread is not None):
            self.worker_thread.join()
        self.worker_thread = None

    def start_thread(self):
        if (self.worker_thread is not None) and (self.worker_thread.is_alive()):
            print("VariableDiscoveryQueue.start_thread(): WARNING: Worker thread is already running -- not starting a new one.")
            return
        print("VariableDiscoveryQueue.start_thread(): Starting worker thread...")

        self.THREAD_ACTIVE = True
        # Start the worker thread
        self.worker_thread = threading.Thread(target=self.process_work_monitor, daemon=True)
        self.worker_thread.start()

        print("VariableDiscoveryQueue.start_thread(): Worker thread started.")


    # Main thread: Process the request queue
    # This should be running in the background.
    def process_work_monitor(self):
        print("VariableDiscoveryQueue.process_work_monitor(): Worker thread started.")
        while (self.THREAD_ACTIVE):
            time.sleep(1)   # Sleep for a bit to avoid busy waiting

            # Step 0: Check if the heartbeat interval has passed
            time_since_last_heartbeat = time.time() - self.heartbeat_last_time
            if (time_since_last_heartbeat >= self.heartbeat_interval):
                # Print a heartbeat message
                print("VariableDiscoveryQueue.process_work_monitor(): Heartbeat: Worker thread is alive. (active threads: " + str(len(self.active_threads)) + ", queue size: " + str(len(self.queue)) + ")")
                self.heartbeat_last_time = time.time()

            # Perform one unit of work
            def perform_work(request):
                try:
                    print("VariableDiscoveryQueue.process_work_monitor(): Processing request '" + str(request['id']) + "' for variable discovery on paper ID '" + str(request['paper_id']) + "'...")
                    #discovery_result_id = self.extract_from_paper(extraction_schema_id=request['extraction_schema_id'], paper_id=request['paper_id'], model_str=request['extraction_model_str'])
                    discovery_result = self.extract_from_paper(theory_query_information=request['theory_query_information'], paper_id=request['paper_id'], additional_contextual_info=request['additional_contextual_info'], model_str=request['extraction_model_str'])
                    print("VariableDiscoveryQueue.process_work_monitor(): Request Returned: '" + str(request['id']) + "' on paper ID '" + str(request['paper_id']) + "'...")

                    # Step 2C: Store the work result
                    if (discovery_result is None):
                        print("VariableDiscoveryQueue.process_work_monitor(): Request '" + str(request['id']) + "' failed to extract from paper ID '" + str(request['paper_id']) + "'.")
                        self._set_work_completed(request['id'], discovery_result="") # Set the result to "" on failure, to signify the result was completed
                    else:
                        # Success
                        self._set_work_completed(request['id'], discovery_result=discovery_result)
                    #print("VariableDiscoveryQueue.process_work_monitor(): Request '" + str(request['id']) + "' completed successfully with extraction result ID: " + str(discovery_result_id))
                    print("VariableDiscoveryQueue.process_work_monitor(): Request '" + str(request['id']) + "' completed successfully for paper ID '" + str(request['paper_id']) + "'.")

                except Exception as e:
                    import traceback
                    error_str = f"VariableDiscoveryQueue.process_work_monitor(): Error processing request '{request['id']}': {str(e)}\n{traceback.format_exc()}"
                    print(error_str)
                    self._set_work_completed(request['id'], discovery_result={"error": error_str})  # Set the result to None on failure

            # Multi-threaded version
            # Step A: Check if any of the active threads have returned results
            with self.THREAD_LOCK_EXTRACTION_QUEUE:
                for request_id in list(self.active_threads):
                    if (request_id in self.finished_work):
                        # If the request is complete, remove it from the active threads
                        print(f"VariableDiscoveryQueue.process_work_monitor(): Request '{request_id}' has completed. Removing from active threads.")
                        self.active_threads.remove(request_id)

            # Step B: If we're less than the maximum number of threads, and the queue has work, then start new threads up to the maximum number of threads
            with self.THREAD_LOCK_EXTRACTION_QUEUE:
                while (len(self.active_threads) < self.MAX_THREADS) and (len(self.queue) > 0):
                    # Get the next request
                    request = self.queue.pop(0)
                    # Start a new thread to process the request
                    print(f"VariableDiscoveryQueue.process_work_monitor(): Starting new thread for request '{request['id']}'...")
                    thread = threading.Thread(target=perform_work, args=(request,), daemon=True)
                    thread.start()
                    # Add the thread to the active threads
                    self.active_threads.add(request['id'])


        # If we reach here, then the thread is being stopped
        print("VariableDiscoveryQueue.process_work_monitor(): Worker thread has completed.")


    #
    #   Performing Work: Extracting content from papers
    #
    def extract_from_paper(self, theory_query_information:str, paper_id:str, additional_contextual_info, model_str:str):
        try:
            result = self._extract_from_paper(theory_query_information, paper_id, additional_contextual_info=additional_contextual_info, model_str=model_str)
            return result
        except Exception as e:
            print(f"ERROR: VariableDiscoveryQueue.extract_from_paper(): An error occurred while extracting from paper ID '{paper_id}': {str(e)}\n{traceback.format_exc()}")
            return ""   # Return blank, so signifies that this result was tried, but not extracted successfully


    def _extract_from_paper(self, theory_query_information:str, paper_id:str, additional_contextual_info=None, model_str:str="gpt-5-mini"):
        # TODO: This whole function should be wrapped in a timeout that gracefully fails if the LLM extraction request times out.

        # Step 1: Retrieve extraction schema (NOT REQUIRED FOR VARIABLE DISCOVERY)

        # Step 2: Retrieve the paper
        paper_text = self.paperstore.get_paper_text_by_key(paper_id)
        if (paper_text is None):
            print("ERROR: VariableDiscoveryQueue.extract_from_paper(): Full-text for paper with ID '" + paper_id + "' not found.")
            return None

        paper_text = truncate_to_model_context(paper_text, model_str)

        # Step 3: Perform the extraction
        max_tokens = 32000

        # Call the extraction function
        print("VariableDiscoveryQueue.extract_from_paper(): Starting extraction for paper ID: " + paper_id + ".")
        #extraction_query = extraction_schema.extraction_query
        #extraction_schema = extraction_schema.schema
        extraction_result_output = self.discover_variables_from_paper(theory_query_information, paper_text, additional_contextual_info=additional_contextual_info, model_str=model_str, max_tokens=max_tokens, use_reflection=False)
        print("VariableDiscoveryQueue.extract_from_paper(): Worker thread for paper ID: " + paper_id + " has received LLM extraction results.")

        response = extraction_result_output.get("response", None)
        if (response is None):
            print("ERROR: VariableDiscoveryQueue.extract_from_paper(): Extraction failed for paper ID '" + paper_id + "'. No response received from LLM.")
            return None

        extraction_results = response.get("results", [])
        follow_on_papers = response.get("follow_on_papers", [])

        # Pack the extraction result
        ## TODO
        #extraction_result = ExtractionQueryResult(paper_id=paper_id, extraction_schema_id=extraction_schema_id, extracted_data=extraction_results, potentially_relevant_new_papers=follow_on_papers, cost=extraction_result_output.get("total_cost", 0.0), model_str=model_str)
        # Step 4: Store the extraction result
        #extraction_result_id = self.theorystore.add_extraction_result(result=extraction_result)
        # Step 5: Return the extraction result ID
        #return extraction_result_id

        # Hack: return the entire discovery result, since there isn't a storage class in the TheoryStore made for it yet
        return extraction_result_output


    # This is the main function that extracts extraction-schema-specified information (`added_keys`) from a scientific paper (`paper_text`), with the `query` in mind.
    # NOTE: This is a `staticmethod` so that it can also be called directly (i.e. `VariableDiscoveryQueue.discover_variables_from_paper(...)`) without having to
    # construct a queue.  Calling it on an instance (`self.discover_variables_from_paper(...)`) works exactly as before.
    @staticmethod
    def discover_variables_from_paper(theory_query_information:dict, paper_text:str, additional_contextual_info=None, model_str="gpt-5-mini", max_tokens=8000, temperature=0.0, use_reflection:bool=False, max_generation_time_seconds:int=None):
        def mkPrompt(theory_query_information:dict, paper_text:str, additional_contextual_info=None, reflection:str=None):
            prompt = ""
            prompt += "You are ScientistGPT, the most advanced AI scientist in the world.  You can answer any scientific question, and if you don't know the answer, you can use your enormous intellect to find it.  You answer every question accurately, faithfully, and with the highest level of scientific integrity.\n\n"
            prompt += "\n"

            prompt += "# Task\n"
            prompt += "This is part of a theory-building process.\n"

            prompt += "## High-Level Task Context\n"
            prompt += "The overall task is to formulate new theories/laws based on scientific evidence, in part through a large-scale literature search.\n"
            prompt += "The general process is: (a) extract information from papers using a detailed extraction schema, that can supply information useful for theory building, then (b) use the extracted information across many (dozens/hundreds) of papers to infer patterns and build theories.\n"
            prompt += "The slots in the extraction schema can be high-level/broad, or low-level/specific.\n"
            prompt += "\n"
            prompt += "Here are examples of high-level/open-ended slots:\n"
            prompt += "- `key_factors`: Key factors that contribute to the phenomenon of interest (e.g. Alzheimer's disease development)\n"
            prompt += "- `challenges`: Challenges that make it difficult to solve the problem of interest\n"
            prompt += "- `proposed_approaches`: Proposed approaches to solving the problem of interest\n"
            prompt += "- `research_methods`: A description of how the problem was studied/the research methods used\n"
            prompt += "- `findings`: The key findings/results of the paper\n"
            prompt += "\n"
            prompt += "Here are examples of low-level/specific slots:\n"
            prompt += "- `datasets_used`: A list of datasets used in the paper\n"
            prompt += "- `model_architectures_used`: A list of model architectures used in the paper\n"
            prompt += "- `evaluation_metrics_used`: A list of evaluation metrics used in the paper\n"
            prompt += "- `model_performance_on_question_answering`: The performance of the model(s) used on a specific task (e.g. question answering), if applicable\n"
            prompt += "- `model_performance_after_treatment`: The performance of the model(s) after a specific treatment/intervention (e.g. memory augmentation), if applicable\n"
            prompt += "\n"
            prompt += "## Thinking about slots\n"
            prompt += "The slots in the extraction schema are designed to extract information that is useful for theory building.\n"
            prompt += "Theory building in this way is essentially an exercise in finding patterns in the extracted information across many papers, and then using those patterns to build theories that explain the phenomenon of interest.\n"
            prompt += "Some of these patterns can be sparse (i.e. occuring in only a small number of papers), and some can be dense (i.e. occurring in many papers).\n"
            prompt += "\n"
            prompt += "## Slots and their relationship to variables in theories:\n"
            prompt += "Sometimes slots are specific enough, that they themselves are very close to variables that would appear in a theory.  As a toy example, if the extraction schema contained slots for 'force', 'mass', and 'acceleration', and the papers all described experiments measuring the force, mass, and acceleration of objects, then it'd be likely to discover a direct quantitative relationship between these variables (i.e. force = mass * acceleration) -- though of course this is a toy example because it's already known.\n"
            prompt += "Other times, slots are more high-level and broad, and they might contain a mix of different kinds of information that would need to be further processed/analyzed to discover finer-grained (or even more abstracted) patterns.\n"
            prompt += "Some slots can be filled with quantitative information (e.g. a number), some qualitative information, some categorical information, some prose descriptions, short lists, or other formats -- whatever is needed.\n"
            prompt += "In this way, some theories/laws that are synthesized may be quantitative, qualitative, or a mix of both.\n"
            prompt += "\n"
            prompt += "## Implicit/Hidden Variable Discovery\n"
            prompt += "In some cases, the most important variables to extract may be implicit, hidden, or otherwise non-obvious.\n"
            prompt += "Here is a real (but somewhat toy) example from the Chemistry domain, and an early qualitative law discovery system.\n"
            prompt += "The law that was discovered by the system was: `acids and alkalis react to produce a salt and water`.\n"
            prompt += "You can imagine that if the extraction slots directly included this classification information (i.e. whether a substance was an acid, alkali, salt), then it would be relatively straightforward to discover this law.\n"
            prompt += "It's also important to node that this information could have been included in a variety of ways:\n"
            prompt += "- Possibility: `reactant_1_classification`, `reactant_2_classification`: These are high-level slots, that aren't restricted to being a specific variable type (e.g. acid/alkali/salt), but it might get lucky and fill those classifications in for a subset of papers, and then the broader analysis would discover the pattern (there's a precision/recall tradeoff analog here: being specific makes it easier to discover a specific pattern, but harder to go fishing/abstract/discover somewhat unexpected patterns).\n"
            prompt += "- Possibility: `reactant_1_properties`, `reactant_2_properties`: These are more general slots, that could contain a variety of information about the reactants, but if they contain the right information (e.g. pH level, or other chemical properties), then the broader analysis might discover the pattern (though this is harder than the previous possibility, because the information is less standardized and more variable across papers, so it's harder to discover patterns in it).\n"
            prompt += "- Possibility: `is_reactant_1_an_acid`, `is_reactant_1_an_alkali`, `is_reactant_1_a_salt`: These are very specific slots (and could have been more categorical: e.g. `classification_acid_alkali_salt`), that directly ask for classification based on a specific categorization scheme.\n"
            prompt += "- Possibility: `reactant_1_chemical_formula`, `reactant_2_chemical_formula`: These are more indirect, that ask for chemical formulas, but if the analysis is sophisticated enough, it might be able to infer the classifications from the chemical formulas (though this is much harder than the previous possibilities, because it requires more chemical knowledge and reasoning to notice the pattern/infer the classifications from the chemical formulas).\n"
            prompt += "- Possibility: `reactant_1_chemical_name`, `reactant_2_chemical_name`: These are even harder, as there would be quite a bit of effort required to discover that chemical X is an acid, chemical Y is an alkali, and chemical Z is a salt, based on their names alone (though in some cases it might be possible to infer this from the names, if the names contain indicative information).\n"
            prompt += "One of the more interesting things is that the most important or useful variables might be non-obvious, or classifications that people haven't thought about before.\n"
            prompt += "That is: some discovered patterns might be rules over known classes (e.g. acids+alkalis -> salts, if this rule wasn't already known, but the classes of `acid`, `alkali`, and `salt` were known)\n"
            prompt += "BUT: some discovered patterns might be over new classes that haven't been explicitly recognized before. One of my favorite examples is that a colleague recently discovered (in a natural langugae processing task) that the standard input people were using actually contained two large, completely independent classes of problems that had been unrecognized, and that if you studied them independent, you discovered patterns that were obscured or difficult to see when you didn't know the classes existed.\n"
            prompt += "While that's a more classification-centered, splitting-hypothesis-centered example, there are also classic versions of other examples: such as the idea that tectonic plates are a hidden variable/mechanistic explanation that unifies (rather than split) seeminly disparate phenomena (e.g. earthquakes, volcanoes, mountain formation, continental drift) into a unifying theory, that also allows making better predictions, and asking better research questions.\n"
            prompt += "\n"
            prompt += "In summary:\n"
            prompt += "- Discovering novel rules over known variables can be impactful\n"
            prompt += "- Discovering novel classifications, or ways of looking at the data, that reveal new variables can be impactful\n"
            prompt += "- New variables can be low-level and specific, high-level and abstract, or anywhere in between, and still potentially be useful for theory building\n"
            prompt += "\n"

            prompt += "# Specific Task\n"
            prompt += "You will be provided with:\n"
            prompt += "- `theory_query_information`: This is a dictionary that contains information about the specific theory-building question or topic of interest.\n"
            prompt += "- The text of a single scientific paper\n"
            prompt += "- (Optional) Data to reflect upon, which could be: data from an earlier response; a brainstorming schema generated with a different/simpler method; data extracted from papers using a different schema; or any other kind of data that might be useful for reflection.  This is optional, and may not be provided.\n"
            prompt += "\n"
            prompt += "Your task is to examine the theory query, paper text, and optional data provided, and suggest schema slots that you think would be useful for extracting information from a much broader corpus of papers in support of the theory building process.\n"
            prompt += "\n"


            # TODO: Add contextual schema / previous schema, to work off of?

            #prompt += "You will be given a scientific paper, and you will have to extract specific types of information from it based on the query, below.  You will return the results in a JSON format, also described below.\n"
            #prompt += "The overall goal of the entities you extract is to use them for building scientific theories in the context of the query, so the knowledge that you extract must be very detailed, accurate, faithful, and scientifically rigorous.\n"
            #prompt += "\n"

            prompt += "# Tools\n"
            prompt += "If you are a tool-calling model, it is important that you do not call any external web-search tools for this task.  Your information should come only from the scientific paper provided, and not (for example) a web search.\n"
            prompt += "\n"

            prompt += "# Query\n"
            prompt += "Here is the theory query:\n"
            prompt += "```\n"
            if (isinstance(theory_query_information, dict)):
                prompt += json.dumps(theory_query_information, indent=4)
            else:
                prompt += str(theory_query_information)
            prompt += "\n"
            prompt += "```\n"
            prompt += "\n"

            prompt += "# Paper Text\n"
            prompt += "Here is the text of the scientific paper:\n"
            prompt += "```\n"
            prompt += paper_text + "\n"
            prompt += "```\n"
            prompt += "\n"

            if (additional_contextual_info is not None):
                prompt += "# Optional Additional Contextual Information\n"
                prompt += "Here is some additional contextual information that may be useful for the variable discovery / schema slot suggestion process.\n"
                prompt += "```\n"
                if (isinstance(additional_contextual_info, dict) or isinstance(additional_contextual_info, list)):
                    prompt += json.dumps(additional_contextual_info, indent=4)
                else:
                    prompt += str(additional_contextual_info)
                prompt += "\n"
                prompt += "```\n"
                prompt += "\n"


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

            prompt += "Your JSON response must be a dictionary, that contains one key: `proposed_slots`, a list of dictionaries.  Each dictionary must contain the following keys:\n"
            prompt += "- `slot_name`(str): A short name for the proposed slot (e.g. 'key_factors_in_development')\n"
            prompt += "- `slot_description`(str): A concise but information-dense description of what kind of information should be filled in this slot.  This should be in 1-2 sentences.\n"
            prompt += "- `how_useful_for_theory_building`(str): A concise but information-dense description of how the information filled in this slot might be useful for theory building.  This should be in 1-2 sentences.\n"
            prompt += "- `what_variables`(str): A specific description of what variable (or variables) this slot might contain/measure, and whether they are high-level or low-level variables.\n"
            prompt += "- `possible_implicit_or_hidden_variables`(str): What variables, if any, are implicit, hidden, or non-obvious in this slot?\n"
            prompt += "- `expected_frequency`(str): Do you expect this slot to be filled with information in many papers, or only a few papers? (and, why?)\n"
            prompt += "- `impact_rating`(int): On a scale of 1-10, if this slot was filled with high-quality information across a large corpus of papers, how impactful do you think it would be for building useful theories that can lead to impactful theories that significantly expand knowlege or ability? (1 = not at all, 10 = exceptional)\n"
            prompt += "- `impact_rating_explanation`(str): A concise but information-dense explanation of why you gave the impact rating you did.  This should be in 1-2 sentences.\n"
            prompt += "\n"
            prompt += "For example:\n"
            prompt += "Note: This is a cartoon example to demonstrate the format, and your example should be significantly more detailed, thought-out, and information dense than this.\n"
            prompt += "```\n"
            prompt += "{\n"
            prompt += "    \"proposed_slots\": [\n"
            prompt += "        {\n"
            prompt += "            \"slot_name\": \"key_factors_in_development\",\n"
            prompt += "            \"slot_description\": \"...\",\n"
            prompt += "            \"how_useful_for_theory_building\": \"...\",\n"
            prompt += "            \"what_variables\": \"...\",\n"
            prompt += "            \"possible_implicit_or_hidden_variables\": \"...\",\n"
            prompt += "            \"expected_frequency\": \"...\"\n"
            prompt += "            \"impact_rating\": 8,\n"
            prompt += "            \"impact_rating_explanation\": \"...\"\n"
            prompt += "        },\n"
            prompt += "        ... # Add more proposed slots as needed; but no more than 10\n"
            prompt += "    ]\n"
            prompt += "}\n"
            prompt += "```\n"
            prompt += "\n"

            prompt += "# Important Notes\n"
            prompt += "- You must return a JSON response, and it must be valid JSON, or it will not be parsed.\n"
            prompt += "- You are encouraged to think and plan before writing your JSON response, but your JSON response must be the last thing you write, and it must be between a single set of codeblocks (```), and contain valid JSON, or it will not be parsed (which will be a critical error).\n"
            prompt += "- Values that are 'none'/'null' in the JSON response should be represented as `null` in the JSON, not as an empty string, string saying \"null\", or any other value.\n"
            prompt += "- Off-ramp: If this paper looks completely irrelevant/off-topic to the query, return blank lists for `proposed_slots`, and we'll detect that and stop.\n"
            prompt += "- All your information must be accurate. Do not hallucinate.\n"

            return prompt


        total_cost = 0.0

        # Create the prompt
        prompt = mkPrompt(theory_query_information, paper_text, additional_contextual_info=additional_contextual_info, reflection=None)

        # Call the LLM
        all_responses = []
        responseJSON, responseText, cost = getLLMResponseJSON(prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=max_generation_time_seconds, repairJSON=True)
        all_responses.append(responseJSON)
        total_cost += cost

        # Reflection: If we are using reflection, we will call the LLM again with the reflection step
        if (use_reflection):
            # Create the reflection prompt
            reflection_prompt = mkPrompt(theory_query_information, paper_text, additional_contextual_info=additional_contextual_info, reflection=responseText)
            # Call the LLM again
            responseJSON, responseText, cost = getLLMResponseJSON(reflection_prompt, model_str, temperature=temperature, maxTokens=max_tokens, jsonOut=False, max_generation_time_seconds=max_generation_time_seconds, repairJSON=True)
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
            "theory_query_information": theory_query_information,
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
    loadAPIKeys()

    # Make placeholder instances of the TheoryStore and PaperStore
    theorystore = TheoryStore()
    paperstore = PaperStore()

    # Create an instance of the VariableDiscoveryQueue
    extraction_queue = VariableDiscoveryQueue(theorystore=theorystore, paperstore=paperstore)

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
    #extraction_query_schema = ExtractionQuerySchema(schema=extraction_schema, extraction_query="Extract information about the paper", supporting_theory_ids=[])
    #extraction_schema_id = theorystore.add_extraction_schema(extraction_query_schema)

    # Submit an extraction request for the papers
    #extraction_request_ids = extraction_queue.submit_extraction_request(extraction_schema_id=extraction_schema_id, paper_ids_to_extract_from=paper_ids)
    # def submit_discovery_request(self, theory_query_information:dict, paper_ids_to_extract_from:list[str], extraction_model_str:str='gpt-5-mini'):
    theory_query_information = {"query": "How can LLM-based agents generate novel scientific theories?"}
    extraction_request_ids = extraction_queue.submit_discovery_request(theory_query_information=theory_query_information, paper_ids_to_extract_from=paper_ids, extraction_model_str="gpt-5-mini")

    # Wait for the extraction requests to complete
    completed_requests_id = []
    print("Extraction request ids: " + str(extraction_request_ids))
    completed_requests = {}
    while (len(completed_requests) < len(extraction_request_ids)):
        completed_requests_id = []
        print("Waiting for extraction requests to complete...")
        for request_id in extraction_request_ids:
            if (extraction_queue.is_request_complete(request_id) is not None):
                completed_requests_id.append(request_id)
                completed_requests[request_id] = extraction_queue.get_completed_work(request_id)
                print("Extraction request completed: " + request_id)
        print("Waiting for extraction requests to complete... (completed: " + str(len(completed_requests)) + "/" + str(len(extraction_request_ids)) + ")")

        # Sleep for a bit before checking again
        time.sleep(1)

    print("All extraction requests completed.")

    # Export to file
    packed = {
        "theory_query_information": theory_query_information,
        "paper_ids": paper_ids,
        "extraction_request_ids": extraction_request_ids,
        "completed_requests": completed_requests,
    }
    filename_out = "variable_discovery_queue_output.debug.json"
    print("Saving: " + str(filename_out))
    with open(filename_out, "w") as f:
        json.dump(packed, f, indent=4)
