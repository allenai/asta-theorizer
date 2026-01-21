# Theorizer.py
from __future__ import annotations

import os
import json
import time


from Struct import *
from SemanticScholar import *
from PaperStore import *
from PaperFinderRequests import *
from SchemaExtractionQueue import *
from MistralOCRStore import *

from ExtractionUtils import *
from TheorizerProcessing import *

# Default model, if no other model is specified
MODEL_STR_TO_USE = "gpt-4.1-2025-04-14"


# Generation objectives and inputs
GENERATION_OBJECTIVE_ACCURACY_FOCUS = "accuracy-focused"
GENERATION_OBJECTIVE_NOVELTY_FOCUS = "novelty-focused"
GENERATION_INPUT_LITERATURE_SUPPORT = "literature-support"
GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY = "parametric knowledge only"


# Generic class for representing a Theorizer workflow
# The general form of the workflow is as follows:
# - Each workflow is a series of steps
# - At the start of each step, a bunch of work is queued up in the Theorizer queues (paperfinder, paper extraction, theory building, etc.).
# - The workflow monitors the queues to see when the work for a given stage is all completed and available to the next stage.
# - When the work is available, the workflow can proceed to the next step.
# - This process continues until all the steps in the workflow are completed.
class TheorizerWorkflow():
    def __init__(self, theorizer=None, name:str="Default Name"):
        self.theorizer = theorizer if theorizer is not None else None
        self.name = name            # A name for the workflow (for the user's benefit)
        self.current_step = 0       # 0 is the first step, 1 is the second step, etc.
        self.is_completed = False
        self.step_names = ["not started", "completed"] # The first and last steps are always "not started" and "completed", and should be empty of work.
        self.errors = []            # Hopefully these never happen, but if they do, error messages are stored here.
        self.total_cost = 0.0       # Total cost of this workflow
        self.costs = {}
        self.status_str = ""        # A current status string, for the user's benefit.

        self.start_time = time.time()  # The time when the workflow was started
        self.end_time = None

        self.time_components = {}   # A dictionary to store the times of each component of the workflow


    def get_name(self):
        # Return the name of the workflow
        return self.name

    def get_cost(self):
        # Return the total cost of the workflow
        # Sum the costs
        self.total_cost = 0.0
        for cost in self.costs.values():
            self.total_cost += cost
        return self.total_cost

    # Get the current runtime of this workflow (in seconds)
    def get_runtime(self):
        if (self.end_time is None):
            # If the workflow is still running, return the current runtime (in seconds)
            return time.time() - self.start_time
        else:
            # If the workflow is completed, return the total runtime (in seconds)
            return self.end_time - self.start_time

    def get_time_components(self):
        # Return the time components of the workflow
        # This is a dictionary of the time components, where each key is a string representing the component name, and the value is the time taken for that component (in seconds)
        return self.time_components

    def get_status_str(self):
        # Return the current status string of the workflow
        return self.status_str

    # Get the maximum number of steps in the workflow
    def max_steps(self):
        return len(self.step_names)

    # Check whether the workflow is in an error state
    # Returns: (bool, lis of errors)
    def has_errors(self):
        return (len(self.errors) > 0, self.errors)

    def add_external_error(self, error_message:str):
        # Add an error message to the list of errors
        self.errors.append(error_message)

    # Check if this workflow is completed
    def is_completed(self):
        return self.is_completed

    # Return the name of the current step in the workflow
    def get_current_step_name(self):
        if (self.current_step < len(self.step_names)):
            return self.step_names[self.current_step]
        return None

    # Signal to move to the next step in the workflow
    def _increment_step(self):
        if (self.is_completed):
            # If the workflow is already completed, do nothing
            return

        # If the current step is not the last step, return the next step
        self.current_step += 1

        # Check if we've exceeded the number of steps in the workflow
        if (self.current_step >= len(self.step_names)):
            self.current_step = len(self.step_names) - 1
            self.is_completed = True


    # The main polling function that should be called periodically to check the status of the workflow
    def poll(self):
        if (self.is_completed):
            if (self.end_time is None):
                # If the workflow is completed, set the end time
                self.end_time = time.time()
            # If the workflow is already completed, do nothing
            return

        # If there are errors, then stop, and mark this workflow as completed.
        has_errors, errors = self.has_errors()
        if (has_errors):
            # If there are errors, then mark this workflow as completed.
            self.is_completed = True
            print("TheorizerWorkflow.poll(): Workflow has errors, marking as completed.\nERRORS:" + "\n".join(errors))
            return

        # Perform the work for the current step
        work_completed = self._perform_work_for_current_step()

        # Check if we can proceed to the next step
        if (work_completed):
            # If we can proceed to the next step, increment the step and start the work for the current step
            self._increment_step()


    # This function should be called (exactly once) when the work for the current step is ready to be started.
    # Returns 'true' if the work for the current step was started, and 'false' otherwise/if there was an error.
    def _perform_work_for_current_step(self):
        # Switch/Case for the step names.
        current_step_name = self.get_current_step_name()
        match current_step_name:
            case "not started":
                # This is the first step, and always has no work.
                return True

            case "completed":
                # This is the last step, and always has no work.
                return True

            # TODO: Add additional steps for your workflow here


            # Default case to catch undefined steps
            case _:
                # For all non-defined steps, write an error message.
                error_message = "TheorizerWorkflow.start_work_for_current_step(): Error: Current step '" + current_step_name + "' is not defined in the workflow."
                self.errors.append(error_message)
                print(error_message)
                return False



# An instance of a workflow
# This one is left mostly blank just to show how to create a new workflow (and, as a test case, to ensure the workflow processing works).
class TheorizerNewTheoryGenerationExample(TheorizerWorkflow):
    def __init__(self, theorizer:Theorizer, name:str, theory_query:str, theory_id:str=None):
        super().__init__(theorizer=theorizer, name=name)
        self.step_names = ["not started", "build-new-request", "paperfinder-request", "paper-conversion", "paper-extraction", "theory-formation", "completed"]


    def _perform_work_for_current_step(self):
        current_step_name = self.get_current_step_name()
        match current_step_name:
            # Start
            case "not started":
                # This is the first step, and always has no work.
                return True

            # Complete
            case "completed":
                # This is the last step, and always has no work.
                return True

            # Build a new request
            case "build-new-request":
                return True

            # PaperFinder request
            case "paperfinder-request":
                return True

            # Paper conversion
            case "paper-conversion":
                return True

            # Paper extraction
            case "paper-extraction":
                return True

            # Theory formation
            case "theory-formation":
                return True

            # Default case to catch undefined steps
            case _:
                # For all non-defined steps, write an error message.
                error_message = "TheorizerNewTheoryGeneration.start_work_for_current_step(): Error: Current step '" + current_step_name + "' is not defined in the workflow."
                self.errors.append(error_message)
                print(error_message)
                return False


#
# Literature supported theory generation workflow.
#

# `extraction_evaluation_year` and `extraction_evaluation_month` are the cutoff date used for dividing extracted results into 'train' (for building the theory) and 'eval' (for evaluating the theory).
# They are inclusive for the evaluation set (i.e. if the evaluation year is 2025 and the evaluation month is 1, then all papers extracted in January 2025 or afterwards are included in the evaluation set).
class TheorizerNewTheoryGeneration(TheorizerWorkflow):
    def __init__(self, theorizer:Theorizer, name:str, theory_query:str, theory_id:str=None, max_papers_to_retrieve:int=10, extraction_evaluation_year:int=2025, extraction_evaluation_month:int=1, model_str_build_new_request:str=MODEL_STR_TO_USE, model_str_extraction:str="gpt-5-mini", generation_objective:str=GENERATION_OBJECTIVE_ACCURACY_FOCUS):
        super().__init__(theorizer=theorizer, name=name)
        self.step_names = ["not started", "build-new-request", "paperfinder-request", "paper-conversion", "paper-extraction", "theory-formation", "completed"]
        self.theory_query = theory_query
        self.theory_id = theory_id if theory_id is not None else None ### NOTE: Not currently used, may not make sense to include (since each request usually generates multiple theories)
        self.max_papers_to_retrieve = max_papers_to_retrieve    # The maximum number of papers to retrieve for this theory request

        # Evaluation parameters
        self.extraction_evaluation_year = extraction_evaluation_year
        self.extraction_evaluation_month = extraction_evaluation_month

        # Components
        self.paperfinder_request_id = None
        self.extraction_request_ids = None
        self.extraction_stage_idx = 0
        self.submitted_paper_titles = set()     # A list of paper titles that we've already submitted to the extraction system (to avoid sending duplicate papers for extraction)

        self.completed_extraction_result_ids_after_follow_on = []

        # Which model to use
        self.model_str_build_new_request = model_str_build_new_request
        self.model_str_extraction = model_str_extraction

        # Generation objective (i.e. accuracy-focused, novelty-focused, etc.)
        self.generation_objective = generation_objective



    # 'True' means the work is done.  'False' means remain on the current step.
    # Adding to the errors list will cause the workflow to be marked as completed (and effectively stopped).
    def _perform_work_for_current_step(self):
        current_step_name = self.get_current_step_name()
        match current_step_name:
            # Start
            case "not started":
                # This is the first step, and always has no work.
                self.status_str = "Work has not started"
                return True

            # Complete
            case "completed":
                # This is the last step, and always has no work.
                self.status_str = "Work has completed"
                self.end_time = time.time()  # Set the end time
                return True

            # Build a new request
            case "build-new-request":
                self.status_str = "Building new request."

                self.start_time = time.time()  # The time when the workflow was started

                # Record start time
                if (self.time_components.get("build-new-request-llm-conversion-start", None) is None):
                    # If the time component is not set, set it to the current time
                    self.time_components["build-new-request-llm-conversion-start"] = time.time()

                # Perform the request

                result = convert_theory_request_to_query_and_schema(theory_request=self.theory_query, model_str=self.model_str_build_new_request, max_tokens=16000, temperature=0.0, use_reflection=False)

                # Parse the output
                #self.total_cost += result.get("total_cost", 0.0)
                self.costs["convert_theory_request_to_query_and_schema"] = result.get("total_cost", 0.0)

                # Get the individual components from the result
                conversion_result = result.get("output", None)
                if (conversion_result is None):
                    self.errors.append("TheorizerNewTheoryGeneration: ERROR in `build-new-request`: The initial step to convert the theory request to a query and schema failed.")
                    return False

                self.theory_request_normalized = conversion_result.get("theory_request_normalized", None)
                self.paper_search_query = conversion_result.get("paper_search_query", None)
                self.short_name = conversion_result.get("short_name", None)
                self.extraction_query = conversion_result.get("extraction_query", None)
                self.extraction_schema = conversion_result.get("extraction_schema", None)

                # Print
                print("TheorizerNewTheoryGeneration: Successfully built new request:")
                print("  Theory Request Normalized: ", self.theory_request_normalized)
                print("  Paper Search Query: ", self.paper_search_query)
                print("  Short Name: ", self.short_name)
                print("  Extraction Query: ", self.extraction_query)
                print("  Extraction Schema: ", self.extraction_schema)

                # Save the extraction schema (combined with the extraction query)
                self.extraction_schema = ExtractionQuerySchema(
                    schema=self.extraction_schema,
                    extraction_query=self.extraction_query,
                    supporting_theory_ids=[],    # Populate after the theories are generated
                    model_str=self.model_str_build_new_request
                )
                # Add the extraction schema to the theory store
                print("Adding extraction schema...")
                self.extraction_schema_id = self.theorizer.theory_store.add_extraction_schema(schema=self.extraction_schema)
                print("Extraction schema ID: " + str(self.extraction_schema_id))

                # Store end time
                self.time_components["build-new-request-llm-conversion-total"] = time.time() - self.time_components["build-new-request-llm-conversion-start"]

                return True

            # PaperFinder request
            case "paperfinder-request":
                self.status_str = "Finding relevant papers for the theory request."

                if (self.time_components.get("paperfinder-request-start", None) is None):
                    # If the time component is not set, set it to the current time
                    self.time_components["paperfinder-request-start"] = time.time()

                # Two substeps: Submit the Paperfinder request, and monitor the PaperFinder queue for results.

                # Submit the PaperFinder request (if not already submitted)
                if (self.paperfinder_request_id == None):
                    self.paperfinder_request_id = self.theorizer.paperfinder_requests.submit_request(self.paper_search_query)

                else:
                    # Monitor the PaperFinder queue for results
                    paperfinder_results = self.theorizer.paperfinder_requests.get_work(self.paperfinder_request_id)
                    if (paperfinder_results is None):
                        # The work isn't yet completed. Keep waiting.
                        ## TODO: Add a timeout here, to put the workflow in an error state if the request takes too long.
                        return False
                    else:
                        # The work is completed -- store it.
                        self.papers_to_add = []
                        # dump paperfinder results to file
                        with open('paperfinder_results.json', 'w') as f:
                            json.dump(paperfinder_results, f, indent=4)
                        paperfinder_results = paperfinder_results.get("result", None)
                        if ("doc_collection" in paperfinder_results) and ("documents" in paperfinder_results["doc_collection"]):
                            self.papers_to_add = paperfinder_results["doc_collection"]["documents"]

                        # Print how many papers were found (before filtering)
                        print(f"TheorizerNewTheoryGeneration: Found {len(self.papers_to_add)} papers for query '{self.paper_search_query}' from the PaperFinder, before filtering.")

                        # Filter to papers whose publication dates are within the evidence window
                        self.papers_to_add_filtered = []
                        for paper_to_add in self.papers_to_add:
                            publication_date = paper_to_add.get("publication_date", None)
                            if (publication_date is None):
                                continue
                            # Parse the publication date (YYYY-MM-DD format)
                            publication_year, publication_month, _ = publication_date.split("-")
                            # Convert to integers
                            try:
                                publication_year = int(publication_year)
                                publication_month = int(publication_month)
                            except ValueError:
                                continue

                            # Check if the publication date is BEFORE the knowledge cutoff
                            if (publication_year < self.extraction_evaluation_year) or ((publication_year == self.extraction_evaluation_year) and (publication_month < self.extraction_evaluation_month)):
                               # The paper was published before the knowledge cutoff, so add it
                               self.papers_to_add_filtered.append(paper_to_add)
                            else:
                                # The paper was published after the knowledge cutoff, so skip it
                                pass

                        # Switch the list to the new list
                        self.papers_to_add = self.papers_to_add_filtered
                        #print(f"TheorizerTheoryEvaluationCollectEvidence: Filtered papers to {len(self.papers_to_add)} papers within the evidence window ({self.evidence_window_start_year}-{self.evidence_window_start_month} to {self.evidence_window_end_year}-{self.evidence_window_end_month}).")
                        print(f"TheorizerNewTheoryGeneration: Filtered papers to {len(self.papers_to_add)} papers before the cutoff date of ({self.extraction_evaluation_year}-{self.extraction_evaluation_month}).")


                        # If there are no papers to add, then mark this as an error.
                        if (len(self.papers_to_add) == 0):
                            print("TheorizerNewTheoryGeneration: No papers found for query '" + self.paper_search_query + "'.")
                            self.errors.append("TheorizerNewTheoryGeneration: No papers found for query '" + self.paper_search_query + "'.")
                            return False

                        # Print the number of papers found
                        print(f"TheorizerNewTheoryGeneration: Found {len(self.papers_to_add)} papers for query '{self.paper_search_query}'.")
                        self.status_str = "Submitting paper requests to the PaperStore."

                        # Add the papers to the processing queue.
                        self.paper_add_request_ids = []
                        for paper_to_add in self.papers_to_add:
                            # Get the title
                            paper_corpus_id = paper_to_add.get("corpus_id", None)
                            paper_title = paper_to_add.get("title", None)
                            paper_title_sanitized = self.theorizer.paperstore._sanitize_paper_title(paper_title)

                            # Make a request for the paper to be added to the PaperStore
                            paper_add_request_id = self.theorizer.paperstore.submit_paper(corpus_id=paper_corpus_id, title=paper_title)
                            self.paper_add_request_ids.append(paper_add_request_id)

                            # Keep track of the paper titles that we've submitted, so we don't submit duplicates
                            self.submitted_paper_titles.add(paper_title)
                            self.submitted_paper_titles.add(paper_title_sanitized)

                            # NOTE: This is submitting `max request` paper add requests, some of which may fail.  A better way of doing this would be monitoring for max_request successful conversions, though this will take much longer, and would need timeouts, since papers (especially in some fields) may be unavailable.
                            # Will keep it simple for now.
                            if (len(self.paper_add_request_ids) >= self.max_papers_to_retrieve):
                                print(f"TheorizerNewTheoryGeneration: Reached maximum number of papers to retrieve ({self.max_papers_to_retrieve}). Stopping submitting paper requests.")
                                break

                        # End time
                        self.time_components["paperfinder-request-total"] = time.time() - self.time_components["paperfinder-request-start"]

                        # If we reach here, then we've submitted the requests to add the papers to the paperstore -- now we just need to wait for them to finish.
                        return True

                return False

            # Paper conversion
            case "paper-conversion":
                self.status_str = "Converting papers to the PaperStore format."

                # Start time
                if (self.time_components.get("paper-conversion-start", None) is None):
                    # If the time component is not set, set it to the current time
                    self.time_components["paper-conversion-start"] = time.time()

                # This step waits for all the papers to be added to the paperstore.
                completed = True
                self.paper_keys = []
                for paper_add_request_id in self.paper_add_request_ids:
                    # Check if the paper has been added to the paperstore
                    paper_key = self.theorizer.paperstore.is_paper_request_complete(paper_add_request_id)
                    if (paper_key != None):
                        # If the paper has been added, add it to the list of paper keys
                        self.paper_keys.append(paper_key)
                    else:
                        # If the paper has not been added, then we're not done yet.
                        completed = False

                # Calculate what percentage of the papers have been added
                completed_proportion = (len(self.paper_keys) / len(self.paper_add_request_ids))
                print(f"TheorizerNewTheoryGeneration: Paper conversion step: {len(self.paper_keys)} out of {len(self.paper_add_request_ids)} papers added to the paperstore ({completed_proportion * 100:.2f}%).")
                self.status_str = "Waiting for papers to be added to the PaperStore (" + str(len(self.paper_keys)) + " / " + str(len(self.paper_add_request_ids)) + " papers added)."

                # TODO: Need to handle potential errors: What if some of the papers fail to be added, etc. -- incorporate timeout?

                if (completed):
                    # DEBUG: Save the paperstore
                    self.theorizer.paperstore.save("paperstore.debug.json")
                    # If all the papers have been added, then we can proceed to the next step.
                    self.time_components["paper-conversion-total"] = time.time() - self.time_components["paper-conversion-start"]
                    return True
                else:
                    return False

            # Paper extraction
            case "paper-extraction":
                self.current_status = "Extracting information from the papers."
                completed = False

                # Start time
                if (self.time_components.get("paper-extraction-start", None) is None):
                    # If the time component is not set, set it to the current time
                    self.time_components["paper-extraction-start"] = time.time()

                # Two steps: Submitting the extraction requests, and waiting for the results to complete.
                if (self.extraction_request_ids == None):

                    # Start time (extraction on initial papers)
                    if (self.time_components.get("paper-extraction-initial-start", None) is None):
                        # If the time component is not set, set it to the current time
                        self.time_components["paper-extraction-initial-start"] = time.time()

                    self.status_str = "Submitting extraction requests for the papers."
                    # Step 1: Submit the extraction requests
                    self.extraction_request_ids = []
                    # Submit the extraction request for all the papers (this function handles the whole paper list, and automatically splits each paper into a single request)
                    extraction_request_ids = self.theorizer.schema_extraction_queue.submit_extraction_request(extraction_schema_id=self.extraction_schema_id, paper_ids_to_extract_from=self.paper_keys, extraction_model_str=self.model_str_extraction)
                    self.extraction_request_ids = extraction_request_ids
                    # Keep track of the number of papers that were submitted for extraction
                    self.num_papers_initial = len(self.paper_keys)
                    self.num_papers_follow_on = 0

                    # Signal that there is still more work to perform
                    self.extraction_stage_idx = 1   # Signal to move to next stage
                    return False

                elif (self.extraction_stage_idx == 1):
                    self.extraction_stage_idx = 2    # By default, bump up the stage
                    # Step 2: Wait for the extraction requests to complete
                    self.completed_extraction_result_ids = []
                    self.completed_extraction_results = []
                    for extraction_request_id in self.extraction_request_ids:
                        # Check if the extraction request is complete
                        extraction_result_id = self.theorizer.schema_extraction_queue.is_request_complete(request_id=extraction_request_id)
                        if (extraction_result_id is not None):
                            # If the extraction result is available, add it to the list of completed results
                            self.completed_extraction_result_ids.append(extraction_result_id)
                            extraction_result = self.theorizer.theory_store.get_extraction_result(id=extraction_result_id)
                            self.completed_extraction_results.append(extraction_result)
                        else:
                            # If the extraction result is not available, then we're not done yet.
                            self.extraction_stage_idx = 1       # Still waiting for work -- reduce the stage back down to the current stage

                    # Calculate what percentage of the extraction requests have been completed
                    completed_proportion = (len(self.completed_extraction_results) / len(self.extraction_request_ids))
                    print(f"TheorizerNewTheoryGeneration: Paper extraction step: {len(self.completed_extraction_results)} out of {len(self.extraction_request_ids)} extraction requests completed ({completed_proportion * 100:.2f}%).")
                    self.status_str = "Waiting for extraction results to complete (" + str(len(self.completed_extraction_results)) + " / " + str(len(self.extraction_request_ids)) + " extraction requests completed)."

                    # End time (if finished)
                    if (self.extraction_stage_idx == 2):
                        self.time_components["paper-extraction-initial-total"] = time.time() - self.time_components["paper-extraction-initial-start"]

                elif (self.extraction_stage_idx == 2):
                    print(f"TheorizerNewTheoryGeneration: Checking to see if we have enough papers to form the theory, or if we need to add in follow-on papers")

                    # Start time
                    if (self.time_components.get("paper-extraction-middle-process-extraction-results-start", None) is None):
                        # If the time component is not set, set it to the current time
                        self.time_components["paper-extraction-middle-process-extraction-results-start"] = time.time()

                    # Step 3: Check to see if we've processed the number of papers we were hoping to process.
                    # If not, try to add in more papers to the extraction queue from the extraction results.
                    if (len(self.completed_extraction_results) < self.max_papers_to_retrieve):
                        print("TheorizerNewTheoryGeneration: Processed " + str(len(self.completed_extraction_results)) + " papers, but user requested to process " + str(self.max_papers_to_retrieve) + " papers. Adding follow-on papers to the extraction queue.")
                        # We can potentially add more papers here.
                        # For each extraction result, get a list of potentially relevant papers
                        self.potentially_relevant_papers = []
                        for extraction_result_id in self.completed_extraction_result_ids:
                            print("** Extraction Result ID: " + str(extraction_result_id))
                            # Get the extraction result from the theory store
                            extraction_result = self.theorizer.theory_store.get_extraction_result(id=extraction_result_id)
                            print("** Extraction Result: " + str(extraction_result))

                            if (extraction_result is not None):
                                # Part 1: Get a list of the potentially relevant papers that were found from reading this paper
                                new_papers_found = extraction_result.potentially_relevant_new_papers
                                print("Extraction Result ID: " + str(extraction_result_id) + "\nPotentially relevant papers: " + str(new_papers_found))
                                if (new_papers_found is not None) and (isinstance(new_papers_found, list)) and (len(new_papers_found) > 0):
                                    # If there are new papers found, add them to the list of potentially relevant papers
                                    for paper_record in new_papers_found:
                                        print("*\t" + str(paper_record))
                                        # Check that the record has at least two keys: `title` and `rating`
                                        if ("paper_title" in paper_record) and ("rating" in paper_record):
                                            if (not isinstance(paper_record["paper_title"], str)) or (not isinstance(paper_record["rating"], int)) or (len(paper_record["paper_title"]) < 10):
                                                # If the title is not a string or the rating is not an integer, or the title is too short, skip this paper
                                                continue
                                            # Add the paper record to the list of potentially relevant papers
                                            sanitized_title = self.theorizer.paperstore._sanitize_paper_title(paper_record["paper_title"])
                                            paper_record["sanitized_title"] = sanitized_title
                                            self.potentially_relevant_papers.append(paper_record)
                                            print("Added " + str(paper_record["paper_title"]) + " to the list of potentially relevant papers with rating " + str(paper_record["rating"]))

                                # Part 2: Also store the name of the paper that was processed, so we don't accidentally add it again
                                paper_id = extraction_result.paper_id
                                paper_record = self.theorizer.paperstore.get_paper_by_key(key=paper_id)
                                if (paper_record is not None):
                                    paper_title = paper_record.get("paper_title", None)
                                    if (paper_title is not None):
                                        # Sanitize the paper title
                                        sanitized_paper_title = self.theorizer.paperstore._sanitize_paper_title(paper_title)
                                        # Add the title to the list of already explored paper titles
                                        self.submitted_paper_titles.add(sanitized_paper_title)
                                        self.submitted_paper_titles.add(paper_title)


                        # Sort the potentially relevant papers by their rating (descending). If no rating, assume -1
                        self.potentially_relevant_papers.sort(key=lambda x: x.get("rating", -1), reverse=True)

                        # Only keep papers with ratings >= 1
                        self.filtered_potentially_relevant_papers = [paper for paper in self.potentially_relevant_papers if paper.get("rating", -1) >= 1]
                        # Print the number of potentially relevant papers found
                        print(f"TheorizerNewTheoryGeneration: Found {len(self.potentially_relevant_papers)} potentially relevant follow-on papers from the extraction results.")
                        print(f"TheorizerNewTheoryGeneration: Filtered down to {len(self.filtered_potentially_relevant_papers)} potentially relevant follow-on papers with rating >= 1.")

                        # If our initial paperfinder search didn't find `max_papers_to_retrieve`, then here backfill by adding the top-rated N papers to the extraction queue (where N is the number of papers we need to reach the maximum)
                        num_papers_to_add = self.max_papers_to_retrieve - len(self.completed_extraction_results)
                        num_papers_added = 0
                        self.num_papers_follow_on = 0
                        if (num_papers_to_add > 0):
                            for potentially_relevant_paper in self.filtered_potentially_relevant_papers:
                                # Check if the paper title has already been submitted
                                if (potentially_relevant_paper["sanitized_title"] in self.submitted_paper_titles) or (potentially_relevant_paper["paper_title"] in self.submitted_paper_titles):
                                    # If the paper title has already been submitted, skip it
                                    continue

                                # These additions are title-only (since they are extracted from mentions/citations in a paper, and don't include S2 IDs or corpus IDs).
                                paper_add_request_id = self.theorizer.paperstore.submit_paper(title=potentially_relevant_paper["paper_title"])
                                self.paper_add_request_ids.append(paper_add_request_id)

                                self.submitted_paper_titles.add(potentially_relevant_paper["sanitized_title"])
                                self.submitted_paper_titles.add(potentially_relevant_paper["paper_title"])
                                num_papers_added += 1
                                self.num_papers_follow_on += 1

                                # If we haven't added enough papers yet, then add this paper to the list of papers to add
                                if (num_papers_added >= num_papers_to_add):
                                    break

                        print("TheorizerNewTheoryGeneration: Added " + str(num_papers_added) + " follow-on papers to the extraction queue.")

                        # If we did NOT add any papers, then we're done this stage.
                        if (num_papers_added <= 0):
                            # End time
                            self.time_components["paper-extraction-total"] = time.time() - self.time_components["paper-extraction-start"]
                            self.time_components["paper-extraction-middle-process-extraction-results-total"] = time.time() - self.time_components["paper-extraction-middle-process-extraction-results-start"]
                            # Add a rate of seconds-per-paper for the initial papers
                            if (self.num_papers_initial > 0):
                                self.time_components["paper-extraction-rate-seconds-per-paper-initial-rate"] = self.time_components["paper-extraction-total"] / self.num_papers_initial

                            completed = True
                        # If we DID add papers, then we need to move onto the next stage (waiting for the paper add requests to complete, then submitting extraction requests and waiting for them to complete)
                        else:
                            self.extraction_stage_idx = 3
                            self.time_components["paper-extraction-middle-process-extraction-results-total"] = time.time() - self.time_components["paper-extraction-middle-process-extraction-results-start"]
                            # Add a rate of seconds-per-paper for the initial papers
                            if (self.num_papers_initial > 0):
                                paper_extraction_total_so_far = time.time() - self.time_components["paper-extraction-start"]
                                self.time_components["paper-extraction-rate-seconds-per-paper-initial-rate"] = paper_extraction_total_so_far / self.num_papers_initial

                            completed = False

                    else:
                        # We've already processed enough papers, so we can move on to the next stage.

                        # End time
                        self.time_components["paper-extraction-total"] = time.time() - self.time_components["paper-extraction-start"]
                        self.time_components["paper-extraction-middle-process-extraction-results-total"] = time.time() - self.time_components["paper-extraction-middle-process-extraction-results-start"]
                        # Add a rate of seconds-per-paper for the initial papers
                        if (self.num_papers_initial > 0):
                            self.time_components["paper-extraction-rate-seconds-per-paper-initial-rate"] = self.time_components["paper-extraction-total"] / self.num_papers_initial

                        completed = True


                elif (self.extraction_stage_idx == 3):
                    print("TheorizerNewTheoryGeneration: Waiting for follow-on papers to be added to the PaperStore.")

                    # Start time (follow-on papers)
                    if (self.time_components.get("paper-extraction-follow-on-add-papers-start", None) is None):
                        # If the time component is not set, set it to the current time
                        self.time_components["paper-extraction-follow-on-add-papers-start"] = time.time()


                    # This stage is waiting for the new paper requests (from the follow-on paper additions) to finish.
                    self.extraction_stage_idx = 4    # Bump up the stage by default
                    #self.paper_keys = []
                    self.paper_keys_follow_on = []

                    total_papers_converted = 0
                    for paper_add_request_id in self.paper_add_request_ids:
                        # Check if the paper has been added to the paperstore
                        paper_key = self.theorizer.paperstore.is_paper_request_complete(paper_add_request_id)
                        if (paper_key != None):
                            # If the paper has been added, add it to the list of paper keys
                            #self.paper_keys.append(paper_key)
                            if (paper_key not in self.paper_keys):
                                self.paper_keys_follow_on.append(paper_key)
                            total_papers_converted += 1
                        else:
                            # If the paper has not been added, then we're not done yet.
                            self.extraction_stage_idx = 3       # Still waiting for work -- reduce the stage back down to the current stage

                    # Calculate what percentage of the papers have been added
                    completed_proportion = (total_papers_converted / len(self.paper_add_request_ids))
                    print(f"TheorizerNewTheoryGeneration: Follow-on Paper conversion step: {total_papers_converted} out of {len(self.paper_add_request_ids)} papers added to the paperstore ({completed_proportion * 100:.2f}%).")
                    self.status_str = "Waiting for follow-on papers to be added to the PaperStore (" + str(total_papers_converted) + " / " + str(len(self.paper_add_request_ids)) + " papers added)."

                    # If all the papers have been added, then we can move on to the next stage.
                    if (completed_proportion >= 1.0):
                        self.extraction_stage_idx = 4    # Bump up the stage

                    completed = False

                elif (self.extraction_stage_idx == 4):

                    # End time for the follow-on papers
                    self.time_components["paper-extraction-follow-on-add-papers-total"] = time.time() - self.time_components["paper-extraction-follow-on-add-papers-start"]
                    # Start time for the follow-on papers extraction
                    if (self.time_components.get("paper-extraction-follow-on-extraction-start", None) is None):
                        # If the time component is not set, set it to the current time
                        self.time_components["paper-extraction-follow-on-extraction-start"] = time.time()

                    print("TheorizerNewTheoryGeneration: Submitting extraction requests for the follow-on papers.")
                    # This stage is submitting the extraction requests for the follow-on papers.
                    self.status_str = "Submitting extraction requests for the follow-on papers."

                    extraction_request_ids_follow_on = self.theorizer.schema_extraction_queue.submit_extraction_request(extraction_schema_id=self.extraction_schema_id, paper_ids_to_extract_from=self.paper_keys_follow_on, extraction_model_str=self.model_str_extraction)
                    self.extraction_request_ids.extend(extraction_request_ids_follow_on)

                    self.extraction_stage_idx = 5    # Bump up the stage
                    completed = False


                elif (self.extraction_stage_idx == 5):
                    print("TheorizerNewTheoryGeneration: Waiting for extraction results to complete for the follow-on papers.")
                    self.extraction_stage_idx = 6
                    # In this stage, we wait for the extraction requests for all the papers to complete.
                    self.status_str = "Waiting for extraction results to complete for the follow-on papers."

                    # Step 2: Wait for the extraction requests to complete
                    self.completed_extraction_result_ids_after_follow_on = []
                    for extraction_request_id in self.extraction_request_ids:
                        # Check if the extraction request is complete
                        extraction_result_id = self.theorizer.schema_extraction_queue.is_request_complete(request_id=extraction_request_id)
                        if (extraction_result_id is not None):
                            # If the extraction result is available, add it to the list of completed results
                            self.completed_extraction_result_ids_after_follow_on.append(extraction_result_id)

                        else:
                            # If the extraction result is not available, then we're not done yet.
                            self.extraction_stage_idx = 5       # Still waiting for work -- reduce the stage back down to the current stage

                    # Calculate what percentage of the extraction requests have been completed
                    completed_proportion = (len(self.completed_extraction_result_ids_after_follow_on) / len(self.extraction_request_ids))
                    print(f"TheorizerNewTheoryGeneration: Paper extraction step follow-on: {len(self.completed_extraction_result_ids_after_follow_on)} out of {len(self.extraction_request_ids)} extraction requests completed ({completed_proportion * 100:.2f}%).")
                    self.status_str = "Waiting for extraction results to complete in follow-on stage (" + str(len(self.completed_extraction_result_ids_after_follow_on)) + " / " + str(len(self.extraction_request_ids)) + " extraction requests completed)."

                elif (self.extraction_stage_idx == 6):
                    print("TheorizerNewTheoryGeneration: Follow-on extraction results completed.")
                    # If we reach here, then the follow-on extraction requests have completed.
                    self.completed_extraction_result_ids = self.completed_extraction_result_ids_after_follow_on

                    # Also combine the 'paper_keys' and 'paper_keys_follow_on' lists
                    self.paper_keys.extend(self.paper_keys_follow_on)

                    # End time (follow-on papers)
                    self.time_components["paper-extraction-follow-on-extraction-total"] = time.time() - self.time_components["paper-extraction-follow-on-extraction-start"]
                    # Add a rate-of-seconds-per-paper for the follow-on papers
                    if (self.num_papers_follow_on > 0):
                        self.time_components["paper-extraction-rate-seconds-per-paper-follow-on-rate"] = self.time_components["paper-extraction-follow-on-extraction-total"] / self.num_papers_follow_on
                    completed = True

                else:
                    print("TheorizerNewTheoryGeneration: ERROR in `paper-extraction`: Invalid extraction stage index: " + str(self.extraction_stage_idx))
                    # If we reach here, then the extraction stage index is invalid.
                    self.errors.append("TheorizerNewTheoryGeneration: ERROR in `paper-extraction`: Invalid extraction stage index: " + str(self.extraction_stage_idx))
                    return False


                if (completed):
                    # Extraction completed -- move on to next step
                    # End time (paper extraction)
                    self.time_components["paper-extraction-total"] = time.time() - self.time_components["paper-extraction-start"]

                    return True
                else:
                    # Extraction not completed -- keep waiting
                    return False


            # Theory formation
            case "theory-formation":
                self.current_status = "Submitting theory formation request."

                # Start time
                if (self.time_components.get("theory-formation-start", None) is None):
                    # If the time component is not set, set it to the current time
                    self.time_components["theory-formation-start"] = time.time()

                # Step 1: Collect all the extraction results (evidence)
                self.completed_extraction_results = {}
                for extraction_result_id in self.completed_extraction_result_ids:
                    # Get the extraction result from the theory store
                    extraction_result = self.theorizer.theory_store.get_extraction_result(id=extraction_result_id)
                    if (extraction_result is not None):
                        # Add the extraction result to the completed extraction results
                        self.completed_extraction_results[extraction_result_id] = extraction_result

                # Step 1A: Collect the extracted evidence from the completed extraction results
                self.extracted_data_raw = {}
                total_cost_extraction = 0.0
                for extraction_result_id, extraction_result in self.completed_extraction_results.items():
                    # Get the evidence from the extraction result
                    extracted_data = extraction_result.extracted_data
                    if (extracted_data is not None):
                        # Add the evidence to the evidence raw dictionary
                        self.extracted_data_raw[extraction_result_id] = extracted_data

                    # Also (while we're here) add the cost of the extraction result to the total cost
                    total_cost_extraction += extraction_result.cost

                self.costs["extraction_from_papers"] = total_cost_extraction

                # Step 2: Split the extracted data into two groups: That before the cutoff date (i.e. 'training'), and that after the cutoff date (i.e. 'evaluation')
                self.extracted_data_training = {}
                self.extracted_data_evaluation = {}

                for extraction_result_id, extracted_data in self.extracted_data_raw.items():
                    # Get the paper ID from the extraction result, so we can look up the publication date
                    extraction_result = self.completed_extraction_results[extraction_result_id]
                    if (extraction_result is None):
                        print("TheorizerNewTheoryGeneration: ERROR: Extraction result with ID '" + str(extraction_result_id) + "' is None. Skipping this extraction result.")
                        continue
                    paper_id = extraction_result.paper_id
                    paper_record = self.theorizer.paperstore.get_paper_by_key(key=paper_id)
                    publication_year = paper_record.get("publication_year", None)
                    publication_month = paper_record.get("publication_month", None)
                    paper_title = paper_record.get("title", None)

                    # If the publiction year/month are None, then skip this paper
                    if (publication_year is None) or (publication_month is None):
                        print("TheorizerNewTheoryGeneration: WARNING: Paper with ID '" + paper_id + "' has no publication year/month. Skipping this paper.")
                        continue

                    # Append the source info to each element in the directory
                    extracted_data_with_paper_info = extracted_data.copy()
                    for evidence in extracted_data_with_paper_info:
                        # Add the paper ID and title to the evidence
                        evidence["source_info"] = {
                            "paper_title": paper_title,
                            "publication_date_yy_mm": str(publication_year) + "-" + str(publication_month).zfill(2),  # Format as "YYYY-MM"
                        }

                    # Check if the publication date is before or after the cutoff date
                    if (publication_year < self.extraction_evaluation_year) or (publication_year == self.extraction_evaluation_year and publication_month < self.extraction_evaluation_month):
                        # If the publication date is before the cutoff date, add it to the training data
                        self.extracted_data_training[extraction_result_id] = extracted_data_with_paper_info
                    else:
                        # If the publication date is after the cutoff date, add it to the evaluation data
                        self.extracted_data_evaluation[extraction_result_id] = extracted_data_with_paper_info

                # Print the number of extracted data points in each group
                print(f"TheorizerNewTheoryGeneration: Divided into train and evaluation. Training: {len(self.extracted_data_training)} papers.  Evaluation: {len(self.extracted_data_evaluation)} papers.")

                # Flatten the extracted data into a list of results.
                consolidated_results_before_cutoff = []
                consolidated_results_after_cutoff = []

                for extraction_result_id, extracted_data in self.extracted_data_training.items():
                    for idx, evidence in enumerate(extracted_data):
                        consolidated_results_before_cutoff.append(evidence)

                for extraction_result_id, extracted_data in self.extracted_data_evaluation.items():
                    for idx, evidence in enumerate(extracted_data):
                        consolidated_results_after_cutoff.append(evidence)

                # Step 2A: Subsample the training data (if required) to fit within the token limits.
                MAX_TOKENS_DATA = 100000       # Generic limit: Maximum 100000 tokens of extraction results.
                if ("gpt-oss" in self.model_str_build_new_request):
                    MAX_TOKENS_DATA = 80000    # Can add model-specific token limits here

                consolidated_results_before_cutoff_subsampled = []
                consolidated_subsampled_results = consolodate_results_with_subsampling(extracted_results_in=consolidated_results_before_cutoff, max_tokens=MAX_TOKENS_DATA)
                if (consolidated_subsampled_results is None):
                    # Some terrible error has happened in the subsampling -- the results are too large to fit in the model.
                    self.errors.append("TheorizerNewTheoryGeneration: ERROR in `theory-formation`: The training data is too large to fit in the model. Please reduce the number of papers or the amount of evidence extracted from each paper.")
                    return False
                # If we reach here, then the subsampling was successful.
                consolidated_results_before_cutoff_subsampled = consolidated_subsampled_results.get("data_subsampled", [])
                self.subsampling_rate_training = consolidated_subsampled_results.get("subsampling_rate", None)

                # Step 3: Build the theory from the consolidated results
                self.status_str = "Running theory formation request."
                use_reflection_theory_building = True
                MAX_TOKENS_OUT = 32000        # Generic limit: Maximum 32000 tokens for theory generation output.
                if ("gpt-oss" in self.model_str_build_new_request):
                    MAX_TOKENS_OUT = 16000    # GPT-OSS models have lower token limits

                known_generation_objectives = [GENERATION_OBJECTIVE_ACCURACY_FOCUS, GENERATION_OBJECTIVE_NOVELTY_FOCUS]
                if (self.generation_objective not in known_generation_objectives):
                    print("Warning: Unknown generation objective '" + str(self.generation_objective) + "'. Defaulting to 'accuracy-focused' theory generation.")
                    self.generation_objective = GENERATION_OBJECTIVE_ACCURACY_FOCUS

                theory_result = None
                self.original_generation_objective = str(self.generation_objective)
                # Accuracy-focused theory generation objective
                if (self.generation_objective == GENERATION_OBJECTIVE_ACCURACY_FOCUS):
                    theory_result = build_theory_from_results_single_theory_reflection3(query=self.theory_request_normalized, results=consolidated_results_before_cutoff_subsampled, model_str=self.model_str_build_new_request, max_tokens=MAX_TOKENS_OUT, temperature=1.0, use_reflection=use_reflection_theory_building, include_query_in_generation=True)

                # Novelty-focused theory generation objective ("no safe basin")
                elif (self.generation_objective == GENERATION_OBJECTIVE_NOVELTY_FOCUS):
                    theory_result = build_theory_from_results_single_theory_reflection4_nonsafebasin(query=self.theory_request_normalized, results=consolidated_results_before_cutoff_subsampled, model_str=self.model_str_build_new_request, max_tokens=MAX_TOKENS_OUT, temperature=1.0, use_reflection=use_reflection_theory_building, include_query_in_generation=True)

                else:
                    # Default to 'accuracy-focused'
                    print("ERROR: Unknown generation objective '" + str(self.generation_objective) + "'. This is a critical error.")


                # DEBUG OUTPUT: Enable this if you'd like to see the raw results of the theory formation step
                DEBUG_OUTPUT = False
                if (DEBUG_OUTPUT == True):
                    filename_out_debug = "theory2-qualitative-quantitative-build." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
                    print("DEBUG: Saving file: " + str(filename_out_debug))
                    with open(filename_out_debug, "w") as f:
                        json.dump(theory_result, f, indent=4)


                    ### DEBUG: SAVE INPUT AND OUTPUT FROM THEORY FORMATION
                    debug_packed = {
                        "query": self.theory_request_normalized,
                        "model": self.model_str_build_new_request,
                        "original_generation_objective": self.original_generation_objective,
                        "generation_objective": self.generation_objective,
                        "use_reflection": True,
                        "num_papers_initial": self.num_papers_initial,
                        "num_papers_follow_on": self.num_papers_follow_on,
                        "num_completed_extraction_results": len(self.completed_extraction_results),
                        "num_extracted_data_raw": len(self.extracted_data_raw),
                        "num_extracted_data_training": len(self.extracted_data_training),
                        "num_extracted_data_evaluation": len(self.extracted_data_evaluation),
                        "num_consolidated_results_subsampled": len(consolidated_results_before_cutoff_subsampled),
                        "num_consolidated_results_before_cutoff": len(consolidated_results_before_cutoff),
                        "extraction_request_ids": self.extraction_request_ids,
                        "completed_extraction_result_ids": self.completed_extraction_result_ids,
                        "completed_extraction_result_ids_after_follow_on": self.completed_extraction_result_ids_after_follow_on,
                        #"completed_extraction_results": self.completed_extraction_results,
                        #"extracted_data_raw": self.extracted_data_raw,
                        "consolidated_results_before_cutoff_subsampled": consolidated_results_before_cutoff_subsampled,
                        "consolidated_results_before_cutoff": consolidated_results_before_cutoff,
                        "theory_result": theory_result
                    }
                    # Save the debug packed data to a file
                    filename_out_debug = "debug_theory_formation." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
                    print("Saving debug file...")
                    with open(filename_out_debug, "w") as f:
                        json.dump(debug_packed, f, indent=4)


                # Get the total cost of the theory formation request
                self.costs["theory_formation"] = theory_result.get("total_cost", 0.0)
                # Get the theory response
                theory_response = theory_result.get("theory_response", None)
                if (theory_response is None):
                    print("TheorizerNewTheoryGeneration: ERROR: Theory formation failed. No theory response returned.")
                    self.errors.append("TheorizerNewTheoryGeneration: ERROR: Theory formation failed. No theory response returned.")
                    return True

                # The output should be a dictionary with two main keys: `theories_general` and `theories_specific`.  Each of these should be a list of theories.
                # Convert to a flat list (with a `type` field to indicate whether it's a general or specific theory)
                theories_flat = []
                for theory in theory_response.get("theories_general", []):
                    theory["type"] = "general"
                    theories_flat.append(theory)
                for theory in theory_response.get("theories_specific", []):
                    theory["type"] = "specific"
                    theories_flat.append(theory)


                # Now convert these into Theory storage classes, that we can store in the Theory Store
                self.theories = []
                for theory in theories_flat:
                    # def __init__(self, name:str, description:str, type:str, derived_from:list, components:dict, supporting_evidence_ids:list[str], id:str=None):
                    theory_name = theory.get("theory_name", None)
                    theory_description = theory.get("theory_description", None)
                    theory_type = theory.get("type", None)  # "general" or "specific"
                    theory_derived_from = theory.get("derived_from", [])    # Not derived from any other theories (new theory)
                    theory_components = theory.get("components", {})  # Empty dictionary
                    theory_supporting_evidence_ids = theory.get("supporting_evidence", [])  # List of evidence IDs (will be populated later)
                    # Leave the ID blank, it will be generated when we add the theory to the Theory Store

                    # Place the current theory dict into the 'components' section
                    theory_components = theory
                    # Populate the 'supporting_evidence_ids' with the IDs of the supporting evidence in the training set
                    theory_supporting_evidence_ids = []
                    ## TODO: This should take into account any subsampling
                    for extraction_result_id, extracted_data in self.extracted_data_training.items():
                        # Get the extraction result from the theory store
                        extraction_result = self.theorizer.theory_store.get_extraction_result(id=extraction_result_id)
                        if (extraction_result is not None):
                            # Add the evidence ID to the list of supporting evidence IDs
                            theory_supporting_evidence_ids.append(extraction_result.id)

                    knowledge_cutoff_year = self.extraction_evaluation_year
                    knowledge_cutoff_month = self.extraction_evaluation_month
                    # Should be one month behind the extraction month
                    if (knowledge_cutoff_month == 1):
                        knowledge_cutoff_year -= 1
                        knowledge_cutoff_month = 12
                    else:
                        knowledge_cutoff_month -= 1

                    # Add the generation objective to the theory components
                    theory_components["generation_objective"] = self.generation_objective
                    theory_components["original_generation_objective"] = self.original_generation_objective     # This is added, just in case the original theory objective was invalid, to see what was requested and what it actually used.

                    # Create the Theory object
                    theory_obj = Theory(name=theory_name,
                                        description=theory_description,
                                        type=theory_type,
                                        theory_query=self.theory_request_normalized,
                                        derived_from=theory_derived_from,
                                        theory_evaluation_ids=[],  # No evaluations yet
                                        components=theory_components,
                                        supporting_evidence_ids=theory_supporting_evidence_ids,
                                        knowledge_cutoff_year=knowledge_cutoff_year,
                                        knowledge_cutoff_month=knowledge_cutoff_month,
                                        model_str=self.model_str_build_new_request)
                    # Add the theory to the list of theories
                    self.theories.append(theory_obj)

                # Add the theories to the theory store
                self.theory_ids = []
                for theory in self.theories:
                    new_theory_id = self.theorizer.theory_store.add_theory(theory=theory)
                    if (new_theory_id != None):
                        self.theory_ids.append(new_theory_id)

                print("TheorizerNewTheoryGeneration: Theory formation completed. Theories generated: " + str(len(self.theories)) + ". Theory IDs: " + str(self.theory_ids))
                self.current_status = "Theory formation completed. Theories generated: " + str(len(self.theories)) + ". Theory IDs: " + str(self.theory_ids)

                # End time
                self.time_components["theory-formation-total"] = time.time() - self.time_components["theory-formation-start"]

                return True


            # Default case to catch undefined steps
            case _:
                self.current_status = "ERROR: Current step '" + current_step_name + "' is not defined in the workflow."
                # For all non-defined steps, write an error message.
                error_message = "TheorizerNewTheoryGeneration.start_work_for_current_step(): Error: Current step '" + current_step_name + "' is not defined in the workflow."
                self.errors.append(error_message)
                print(error_message)
                return False



#
#   Parametric-only Theory Generation Workflow
#

# `extraction_evaluation_year` and `extraction_evaluation_month` are the cutoff date used for dividing extracted results into 'train' (for building the theory) and 'eval' (for evaluating the theory).
# They are inclusive for the evaluation set (i.e. if the evaluation year is 2025 and the evaluation month is 1, then all papers extracted in January 2025 or afterwards are included in the evaluation set).
class TheorizerNewTheoryGenerationParametricOnly(TheorizerWorkflow):
    def __init__(self, theorizer:Theorizer, name:str, theory_query:str, theory_id:str=None, model_str_build_new_request:str=MODEL_STR_TO_USE, generation_objective:str=GENERATION_OBJECTIVE_ACCURACY_FOCUS):
        super().__init__(theorizer=theorizer, name=name)
        self.step_names = ["not started", "theory-formation", "completed"]
        self.theory_query = theory_query
        self.theory_id = theory_id if theory_id is not None else None ### NOTE: Not currently used, may not make sense to include (since each request usually generates multiple theories)

        # Which model to use
        self.model_str_build_new_request = model_str_build_new_request

        # Generation objective (i.e. accuracy-focused, novelty-focused, etc.)
        self.generation_objective = generation_objective



    # 'True' means the work is done.  'False' means remain on the current step.
    # Adding to the errors list will cause the workflow to be marked as completed (and effectively stopped).
    def _perform_work_for_current_step(self):
        current_step_name = self.get_current_step_name()
        match current_step_name:
            # Start
            case "not started":
                # This is the first step, and always has no work.
                self.status_str = "Work has not started"
                return True

            # Complete
            case "completed":
                # This is the last step, and always has no work.
                self.status_str = "Work has completed"
                self.end_time = time.time()  # Set the end time
                return True

            # Theory formation
            case "theory-formation":
                self.current_status = "Submitting theory formation request."

                # Start time
                if (self.time_components.get("theory-formation-start", None) is None):
                    # If the time component is not set, set it to the current time
                    self.time_components["theory-formation-start"] = time.time()

                # Step 3: Build the theory (using parametric knowledge alone)
                self.status_str = "Running theory formation request."
                use_reflection_theory_building = True
                MAX_TOKENS_OUT = 32000        # Generic limit: Maximum 32000 tokens for theory generation output.
                if ("gpt-oss" in self.model_str_build_new_request):
                    MAX_TOKENS_OUT = 16000    # GPT-OSS models have lower token limits

                known_generation_objectives = [GENERATION_OBJECTIVE_ACCURACY_FOCUS, GENERATION_OBJECTIVE_NOVELTY_FOCUS]
                if (self.generation_objective not in known_generation_objectives):
                    print("Warning: Unknown generation objective '" + str(self.generation_objective) + "'. Defaulting to 'accuracy-focused' theory generation.")
                    self.generation_objective = GENERATION_OBJECTIVE_ACCURACY_FOCUS

                theory_result = None
                self.original_generation_objective = str(self.generation_objective)
                # Accuracy-focused theory generation objective
                if (self.generation_objective == GENERATION_OBJECTIVE_ACCURACY_FOCUS):
                    theory_result = build_theory_from_results_single_theory_reflection3_llm_baseline_no_evidence(query=self.theory_query, original_theory_id=None, original_theory_name=None, provide_matched_control_thery_name=False, model_str=self.model_str_build_new_request, max_tokens=MAX_TOKENS_OUT, temperature=1.0, use_reflection=use_reflection_theory_building, include_query_in_generation=True)

                # Novelty-focused theory generation objective ("no safe basin")
                elif (self.generation_objective == GENERATION_OBJECTIVE_NOVELTY_FOCUS):
                    theory_result = build_theory_from_results_single_theory_reflection4_llm_baseline_no_evidence_nonsafebasin(query=self.theory_query, original_theory_id=None, original_theory_name=None, provide_matched_control_thery_name=False, model_str=self.model_str_build_new_request, max_tokens=MAX_TOKENS_OUT, temperature=1.0, use_reflection=use_reflection_theory_building, include_query_in_generation=True)

                else:
                    # Default to 'accuracy-focused'
                    print("ERROR: Unknown generation objective '" + str(self.generation_objective) + "'. This is a critical error.")


                # DEBUG OUTPUT: Enable this if you'd like to see the raw results of the theory formation step
                DEBUG_OUTPUT = False
                if (DEBUG_OUTPUT == True):
                    filename_out_debug = "theory2-qualitative-quantitative-build." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
                    print("DEBUG: Saving file: " + str(filename_out_debug))
                    with open(filename_out_debug, "w") as f:
                        json.dump(theory_result, f, indent=4)


                    ### DEBUG: SAVE INPUT AND OUTPUT FROM THEORY FORMATION
                    debug_packed = {
                        "query": self.theory_query,
                        "model": self.model_str_build_new_request,
                        "original_generation_objective": self.original_generation_objective,
                        "generation_objective": self.generation_objective,
                        "use_reflection": True,
                        "theory_result": theory_result
                    }
                    # Save the debug packed data to a file
                    filename_out_debug = "debug_theory_formation." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
                    print("Saving debug file...")
                    with open(filename_out_debug, "w") as f:
                        json.dump(debug_packed, f, indent=4)


                # Get the total cost of the theory formation request
                self.costs["theory_formation"] = theory_result.get("total_cost", 0.0)
                # Get the theory response
                theory_response = theory_result.get("theory_response", None)
                if (theory_response is None):
                    print("TheorizerNewTheoryGeneration: ERROR: Theory formation failed. No theory response returned.")
                    self.errors.append("TheorizerNewTheoryGeneration: ERROR: Theory formation failed. No theory response returned.")
                    return True

                # The output should be a dictionary with two main keys: `theories_general` and `theories_specific`.  Each of these should be a list of theories.
                # Convert to a flat list (with a `type` field to indicate whether it's a general or specific theory)
                theories_flat = []
                for theory in theory_response.get("theories_general", []):
                    theory["type"] = "general"
                    theories_flat.append(theory)
                for theory in theory_response.get("theories_specific", []):
                    theory["type"] = "specific"
                    theories_flat.append(theory)


                # Now convert these into Theory storage classes, that we can store in the Theory Store
                self.theories = []
                for theory in theories_flat:
                    # def __init__(self, name:str, description:str, type:str, derived_from:list, components:dict, supporting_evidence_ids:list[str], id:str=None):
                    theory_name = theory.get("theory_name", None)
                    theory_description = theory.get("theory_description", None)
                    theory_type = theory.get("type", None)  # "general" or "specific"
                    theory_derived_from = theory.get("derived_from", [])    # Not derived from any other theories (new theory)
                    theory_components = theory.get("components", {})  # Empty dictionary
                    theory_supporting_evidence_ids = theory.get("supporting_evidence", [])  # List of evidence IDs (will be populated later)
                    # Leave the ID blank, it will be generated when we add the theory to the Theory Store

                    # Place the current theory dict into the 'components' section
                    theory_components = theory
                    # Populate the 'supporting_evidence_ids' with the IDs of the supporting evidence in the training set
                    theory_supporting_evidence_ids = []

                    knowledge_cutoff_year = None
                    knowledge_cutoff_month = None

                    # Add the generation objective to the theory components
                    theory_components["generation_objective"] = self.generation_objective
                    theory_components["original_generation_objective"] = self.original_generation_objective     # This is added, just in case the original theory objective was invalid, to see what was requested and what it actually used.

                    # Create the Theory object
                    theory_obj = Theory(name=theory_name,
                                        description=theory_description,
                                        type=theory_type,
                                        theory_query=self.theory_query,
                                        derived_from=theory_derived_from,
                                        theory_evaluation_ids=[],  # No evaluations yet
                                        components=theory_components,
                                        supporting_evidence_ids=theory_supporting_evidence_ids,
                                        knowledge_cutoff_year=knowledge_cutoff_year,
                                        knowledge_cutoff_month=knowledge_cutoff_month,
                                        model_str=self.model_str_build_new_request)
                    # Add the theory to the list of theories
                    self.theories.append(theory_obj)

                # Add the theories to the theory store
                self.theory_ids = []
                for theory in self.theories:
                    new_theory_id = self.theorizer.theory_store.add_theory(theory=theory)
                    if (new_theory_id != None):
                        self.theory_ids.append(new_theory_id)

                print("TheorizerNewTheoryGeneration: Theory formation completed. Theories generated: " + str(len(self.theories)) + ". Theory IDs: " + str(self.theory_ids))
                self.current_status = "Theory formation completed. Theories generated: " + str(len(self.theories)) + ". Theory IDs: " + str(self.theory_ids)

                # End time
                self.time_components["theory-formation-total"] = time.time() - self.time_components["theory-formation-start"]

                return True


            # Default case to catch undefined steps
            case _:
                self.current_status = "ERROR: Current step '" + current_step_name + "' is not defined in the workflow."
                # For all non-defined steps, write an error message.
                error_message = "TheorizerNewTheoryGeneration.start_work_for_current_step(): Error: Current step '" + current_step_name + "' is not defined in the workflow."
                self.errors.append(error_message)
                print(error_message)
                return False



#
#   Main Theorizer class
#
class Theorizer:
    # Constructor
    def __init__(self, filename_theory_store_in:str=None):
        # External stores
        self.paperstore = PaperStore()
        self.theory_store = TheoryStore(paperstore=self.paperstore, filename_in=filename_theory_store_in)
        self.paperfinder_requests = PaperFinderRequests()
        self.schema_extraction_queue = SchemaExtractionQueue(theorystore=self.theory_store, paperstore=self.paperstore)

        # Queues
        self.workflow_queue = []
        self.workflows_completed = []
        self.finished_work = {}
        self.next_request_id = 1

        # Update message: Show the user the Theorizer heartbeat every 10 seconds
        self.heartbeat_interval = 5
        self.heartbeat_last_time = 0

        # Thread lock
        self.THREAD_LOCK_THEORIZER_REQUESTS = threading.Lock()
        self.THREAD_ACTIVE = False

        # Start a background thread to process the queue
        self.worker_thread = None
        self.actively_processing_work = False
        self.start_thread()


    #
    #   Work Queueing/Threading
    #

    # Submit a request to the PaperFinder API, which will add it to the request queue.
    # This is the normal (literature-supported) theory generation workflow.
    def submit_theory_query(self, query:str, max_papers_to_retrieve:int=10, extraction_evaluation_year:int=2025, extraction_evaluation_month:int=1, model_str_build_new_request:str=MODEL_STR_TO_USE, model_str_extraction:str="gpt-5-mini", generation_objective:str=GENERATION_OBJECTIVE_ACCURACY_FOCUS):
        # Make a workflow for this query
        new_workflow = TheorizerNewTheoryGeneration(theorizer=self, name="Theory Query: " + query, theory_query=query, theory_id=None, max_papers_to_retrieve=max_papers_to_retrieve, extraction_evaluation_year=extraction_evaluation_year, extraction_evaluation_month=extraction_evaluation_month, model_str_build_new_request=model_str_build_new_request, model_str_extraction=model_str_extraction, generation_objective=generation_objective)
        with self.THREAD_LOCK_THEORIZER_REQUESTS:
            self.workflow_queue.append(new_workflow)

    # TODO: Make a function to submit a Parametric-only theory workflow.
    def submit_theory_query_parametric_only(self, query:str, model_str_build_new_request:str=MODEL_STR_TO_USE, generation_objective:str=GENERATION_OBJECTIVE_ACCURACY_FOCUS):
        # Make a workflow for this query
        new_workflow = TheorizerNewTheoryGenerationParametricOnly(theorizer=self, name="Parametric-Only Theory Query: " + query, theory_query=query, theory_id=None, model_str_build_new_request=model_str_build_new_request, generation_objective=generation_objective)
        with self.THREAD_LOCK_THEORIZER_REQUESTS:
            self.workflow_queue.append(new_workflow)


    # Get the number of requests in the queue
    def get_num_active_workflows(self):
        with self.THREAD_LOCK_THEORIZER_REQUESTS:
            return len(self.workflow_queue)

    # Get the status of all active workflows in the system
    def get_workflow_statuses(self):
        # Not using the thread lock here, since we're just reading, and it's generally OK if this information is slightly out of date.
        out_active = []
        for workflow in self.workflow_queue:
            has_errors, errors = workflow.has_errors()
            out_active.append({"name": workflow.get_name(), "current_step": workflow.get_current_step_name(), "is_completed": workflow.is_completed, "has_errors": has_errors, "errors": errors, "status_str": workflow.get_status_str(), "cost": workflow.get_cost(), "runtime_sec": workflow.get_runtime(), "runtime_steps": workflow.get_time_components()})

        out_completed = []
        for workflow in self.workflows_completed:
            has_errors, errors = workflow.has_errors()
            out_completed.append({"name": workflow.get_name(), "current_step": workflow.get_current_step_name(), "is_completed": workflow.is_completed, "has_errors": has_errors, "errors": errors, "status_str": workflow.get_status_str(), "cost": workflow.get_cost(), "runtime_sec": workflow.get_runtime(), "runtime_steps": workflow.get_time_components()})

        packed = {
            "active_workflows": out_active,
            "completed_workflows": out_completed,
        }
        # Return the active and completed workflow statuses
        return packed

    # Returns true if there is anything in the queue, or anything currently being processed
    def is_busy(self):
        # Check 1: Are there any requests in the queue?
        if (self.get_num_active_workflows() > 0):
            return True
        # Check 2: Are we actively processing work?
        with self.THREAD_LOCK_THEORIZER_REQUESTS:
            if (self.actively_processing_work):
                return True

        # If we reach here, then we're not busy
        return False


    # Stop the worker thread
    def stop_thread(self):
        print("Theorizer.start_thread(): Stopping worker thread...")
        self.THREAD_ACTIVE = False
        if (self.worker_thread is not None):
            self.worker_thread.join()
        self.worker_thread = None

    def start_thread(self):
        with self.THREAD_LOCK_THEORIZER_REQUESTS:
            if (self.worker_thread is not None) and (self.worker_thread.is_alive()):
                print("Theorizer.start_thread(): WARNING: Worker thread is already running -- not starting a new one.")
                return
            print("Theorizer.start_thread(): Starting worker thread...")

            self.THREAD_ACTIVE = True
            # Start the worker thread
            self.worker_thread = threading.Thread(target=self.process_work_monitor, daemon=True)
            self.worker_thread.start()

            print("Theorizer.start_thread(): Worker thread started.")


    # Main thread: Process the request queue
    # This should be running in the background.
    def process_work_monitor(self):
        print("Theorizer.process_work_monitor(): Worker thread started")
        while (self.THREAD_ACTIVE):
            # Step 0: Sleep for 1 second to avoid busy waiting
            time.sleep(1)

            # Step 1: Check if the heartbeat interval has passed
            time_since_last_heartbeat = time.time() - self.heartbeat_last_time
            if (time_since_last_heartbeat >= self.heartbeat_interval):
                # Print a heartbeat message
                print("Theorizer.process_work_monitor(): Heartbeat: Worker thread is alive. (active workflows: " + str(len(self.workflow_queue)) + ", completed workflows: " + str(len(self.workflows_completed)) + ")")
                self.heartbeat_last_time = time.time()

            # Step 2: Look for work to perform.

            # Poll each function in the workflow.  This will handle the workflow progressing (or being marked as finished).
            MAX_SIMULTANEOUS_WORKFLOWS = 5      # Up to 5 workflows can be processed, though these will go round-robin style.
            num_processed_workflows = 0
            for w_idx, workflow in enumerate(self.workflow_queue):
                try:
                    start_stage_name = workflow.get_current_step_name()
                    workflow.poll()
                    end_stage_name = workflow.get_current_step_name()
                    num_processed_workflows += 1
                    print("Theorizer.process_work_monitor(): Polled workflow " + str(w_idx) + ": '" + start_stage_name + "' -> '" + end_stage_name + "'")
                    if (num_processed_workflows >= MAX_SIMULTANEOUS_WORKFLOWS):
                        # If we've processed enough workflows, break out of the loop
                        print("Theorizer.process_work_monitor(): Reached maximum number of workflows processed in this iteration (" + str(MAX_SIMULTANEOUS_WORKFLOWS) + "). Breaking out of the loop.")
                        break
                except Exception as e:
                    # Make an error message
                    import traceback
                    error_message = f"Theorizer.process_work_monitor(): Error while polling workflow {w_idx}: {str(e)} \n{traceback.format_exc()}"
                    workflow.add_external_error(error_message)
                    print(error_message)


            # Check if any of the workflows have completed
            with self.THREAD_LOCK_THEORIZER_REQUESTS:
                w_idx = 0
                while (w_idx < len(self.workflow_queue)):
                    workflow = self.workflow_queue[w_idx]
                    if (workflow.is_completed):
                        # Pop the workflow from the queue
                        completed_workflow = self.workflow_queue.pop(w_idx)
                        # Add it to the completed workflows
                        self.workflows_completed.append(completed_workflow)

                        # Save the finished work to the finished work directory
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename_prefix = f"theorizer-state-autosave-{timestamp}"

                        # Save the theorizer state
                        self.save(filename_prefix=filename_prefix, only_theorystore=True)   # Only save the theorystore since saving the paperstore was filling up the drive.

                    else:
                        w_idx += 1

        # If we reach here, then the thread is being stopped
        print("Theorizer.process_work_monitor(): Worker thread has completed.")




    #
    #   Loading/Saving
    #

    # Save/load the contents of the Theorizer and Paperstore to/from a folder
    def save(self, filename_prefix:str, only_theorystore:bool=False):
        print("Theorizer.save(): Saving state... (" + filename_prefix + ")")

        # Save the theory store
        self.theory_store.save(filename_prefix + ".theorystore.json")

        # Save the paperstore
        if (not only_theorystore):
            self.paperstore.save(filename_prefix + ".paperstore.json")

        print(f"Theorizer: Saved state to {filename_prefix}.")


    def load(self, filename_prefix:str):
        print("Theorizer.load(): Loading state... (" + filename_prefix + ")")

        # Load the theory store
        self.theory_store.load(filename_prefix + ".theorystore.json")

        # Load the paperstore
        self.paperstore.load(filename_prefix + ".paperstore.json")

        print(f"Theorizer: Loaded state from {filename_prefix}.")


    # Load only a paperstore
    def load_paperstore(self, filename_paperstore:str):
        print("Theorizer.load_paperstore(): Loading paperstore from file... (" + filename_paperstore + ")")
        self.paperstore.load(filename_paperstore)
        print("Theorizer.load_paperstore(): Paperstore loaded.")