# TheorizerModalAPI.py
# A quick example of using the Theorizer API on Modal.
import os
import json
import time
import requests

# Default constants
GENERATION_OBJECTIVE_ACCURACY_FOCUS = "accuracy-focused"
GENERATION_INPUT_LITERATURE_SUPPORT = "literature-support"

DEFAULT_MODEL_STR_PRIMARY = "gpt-5.2"
DEFAULT_MODEL_STR_EXTRACTION = "gpt-5-mini"
DEFAULT_GENERATION_OBJECTIVE = GENERATION_OBJECTIVE_ACCURACY_FOCUS
DEFAULT_GENERATION_INPUT = GENERATION_INPUT_LITERATURE_SUPPORT
DEFAULT_MAX_TOKENS = 250000
DEFAULT_KNOWLEDGE_CUTOFF_YEAR = 2050
DEFAULT_KNOWLEDGE_CUTOFF_MONTH = 1
DEFAULT_NUM_PAPERS_TO_RETRIEVE = 100


class TheorizerModalAPI():
    def __init__(self, server_url:str, bearer_token:str=None, local_save_folder:str="theory-cache/"):
        self.server_url = server_url
        self.bearer_token = bearer_token
        self.local_save_folder = local_save_folder

        # A set of (currently active) generation_ids
        self.currently_active_requests = set()
        self.completed_requests = set()

        # A dictionary keeping track of how many retries we've attempted for retrieving the TheoryStore from completed workflows. Key: generation_id, Value: int (number of attempts)
        self.data_retrieval_attempts = {}

        # Create the local save folder if it doesn't exist
        if (self.local_save_folder is not None):
            if (not os.path.exists(self.local_save_folder)):
                os.makedirs(self.local_save_folder)

    #
    #   Submit a new theory request
    #
    def submit_theory_request(self,
            user_id:str,
            theory_query:str,
            mission_statement:str,
            max_papers_to_retrieve:int=DEFAULT_NUM_PAPERS_TO_RETRIEVE,
            knowledge_cutoff_year:int=DEFAULT_KNOWLEDGE_CUTOFF_YEAR,
            knowledge_cutoff_month:int=DEFAULT_KNOWLEDGE_CUTOFF_MONTH,
            model_str_primary:str=DEFAULT_MODEL_STR_PRIMARY,
            model_str_extraction:str=DEFAULT_MODEL_STR_EXTRACTION,
            generation_objective:str=DEFAULT_GENERATION_OBJECTIVE,
            generation_input:str=DEFAULT_GENERATION_INPUT,
            max_data_tokens:int=DEFAULT_MAX_TOKENS,
            do_qualified_novelty_evaluation:bool=False
            ):

        # Submit the theory request.  (Include the bearer token in the header, if not None)
        headers = {}
        if (self.bearer_token is not None):
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        packed_request = {
            "user_id": user_id,
            "theory_query": theory_query,
            "mission_statement": mission_statement,
            "max_papers_to_retrieve": max_papers_to_retrieve,
            "knowledge_cutoff_year": knowledge_cutoff_year,
            "knowledge_cutoff_month": knowledge_cutoff_month,
            "model_str_primary": model_str_primary,
            "model_str_extraction": model_str_extraction,
            "generation_objective": generation_objective,
            "generation_input": generation_input,
            "max_data_tokens": max_data_tokens,
            "do_qualified_novelty_evaluation": do_qualified_novelty_evaluation
        }

        # Wrap in a timeout
        TIMEOUT_SEC = 30
        print("Submitting theory request to server. Theory Query: `" + theory_query + "`. Timeout for this request: " + str(TIMEOUT_SEC) + " seconds.")
        url = self.server_url + "/submit_theory_query"
        try:
            response = requests.post(url, json=packed_request, headers=headers, timeout=TIMEOUT_SEC)
        except requests.exceptions.Timeout:
            print("Failed to submit theory request: Request timed out after " + str(TIMEOUT_SEC) + " seconds.")
            return {
                "generation_id": None,
                "error": f"Request timed out after {TIMEOUT_SEC} seconds when trying to submit theory request."
            }

        print("Response text:", response.text)

        # Parse the response
        if (response.status_code == 202):
            # Success
            submission_result = response.json()
            # Add this generation_id to the list of currently active requests
            generation_id = submission_result.get("generation_id", None)
            if (generation_id is not None):
                print(f"Successfully submitted theory request.  Theory Query: `{theory_query}`. Generation ID: {generation_id}")
                self.currently_active_requests.add(generation_id)
            return submission_result
        else:
            print("Failed to submit theory request. Status code: " + str(response.status_code) + ". Response: " + response.text)
            return {
                "generation_id": None,
                "error": f"Failed to submit theory request. Status code: {response.status_code}. Response: {response.text}"
            }

    #
    #   (Internal) Helpers
    #

    # Get the workflow status for a given generation ID
    # Internal method: Directly polls the server
    def _get_workflow_status(self, generation_id:str):
        headers = {}
        if (self.bearer_token is not None):
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        TIMEOUT_SEC = 30
        try:
            response = requests.post(self.server_url + "/get_workflow_status_by_id", json={"generation_id": generation_id}, headers=headers, timeout=TIMEOUT_SEC)
        except requests.exceptions.Timeout:
            print("Failed to get workflow status for generation ID " + generation_id + ": Request timed out after " + str(TIMEOUT_SEC) + " seconds.")
            return None

        if (response.status_code == 200):
            data = response.json()
            self.save_workflow(generation_id=generation_id, workflow_data=data)
            return data
        else:
            print("Failed to get workflow status for generation ID " + generation_id + ". Status code: " + str(response.status_code) + ". Response: " + response.text)
            return None

    # Get the theorystore for a given generation ID
    # (Internal method, directly polls the server)
    def _get_theorystore(self, generation_id:str):
        headers = {}
        if (self.bearer_token is not None):
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        TIMEOUT_SEC = 30
        try:
            response = requests.post(self.server_url + "/get_theorystore_by_id", json={"generation_id": generation_id}, headers=headers, timeout=TIMEOUT_SEC)
        except requests.exceptions.Timeout:
            print("Failed to get theorystore for generation ID " + generation_id + ": Request timed out after " + str(TIMEOUT_SEC) + " seconds.")
            return None

        if (response.status_code == 200):
            data = response.json()
            self.save_theorystore(generation_id=generation_id, theorystore_data=data)
            return data
        else:
            print("Failed to get theorystore for generation ID " + generation_id + ". Status code: " + str(response.status_code) + ". Response: " + response.text)
            return None


    #
    #   (Internal) File I/O for caching
    #

    def load_workflow(self, generation_id:str):
        if (self.local_save_folder is None):
            print("No local save folder specified. Cannot load workflow from file.")
            return None

        filename = os.path.join(self.local_save_folder, f"workflow_status_{generation_id}.json")
        if (not os.path.exists(filename)):
            return None

        try:
            with open(filename, "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"Error loading workflow status from file for generation ID {generation_id}: {str(e)}")
            return None

    def save_workflow(self, generation_id:str, workflow_data:dict):
        if (self.local_save_folder is None):
            print("No local save folder specified. Cannot save workflow to file.")
            return False

        filename = os.path.join(self.local_save_folder, f"workflow_status_{generation_id}.json")
        try:
            with open(filename, "w") as f:
                json.dump(workflow_data, f, indent=4)
                return True
        except Exception as e:
            print(f"Error saving workflow status to file for generation ID {generation_id}: {str(e)}")
            return False

    def load_theorystore(self, generation_id:str):
        if (self.local_save_folder is None):
            print("No local save folder specified. Cannot load theorystore from file.")
            return None

        filename = os.path.join(self.local_save_folder, f"theorystore_{generation_id}.json")
        if (not os.path.exists(filename)):
            return None

        try:
            with open(filename, "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"Error loading theorystore from file for generation ID {generation_id}: {str(e)}")
            return None

    def save_theorystore(self, generation_id:str, theorystore_data:dict):
        if (self.local_save_folder is None):
            print("No local save folder specified. Cannot save theorystore to file.")
            return False

        filename = os.path.join(self.local_save_folder, f"theorystore_{generation_id}.json")
        try:
            with open(filename, "w") as f:
                json.dump(theorystore_data, f, indent=4)
                return True
        except Exception as e:
            print(f"Error saving theorystore to file for generation ID {generation_id}: {str(e)}")
            return False


    #
    #   (User-facing) Polling: Check the status of all active theory requests, and retrieve results for any that have completed.
    #
    def poll(self):
        WAIT_TIME_BETWEEN_REQUESTS = 2  # seconds

        # For each active request
        for generation_id in list(self.currently_active_requests):
            # Check the status of this request
            print("Polling for status of generation ID: " + generation_id)
            workflow_status = self._get_workflow_status(generation_id=generation_id)
            if (workflow_status is None):
                continue
            else:
                # Check if this workflow has finished successfully
                is_finished = workflow_status.get("is_finished", False)
                was_successful = workflow_status.get("was_successful", False)

                if (is_finished == True) and (was_successful == True):
                    # If so, then retrieve the theorystore for this request
                    theorystore = self._get_theorystore(generation_id=generation_id)
                    if (theorystore is None):
                        self.data_retrieval_attempts[generation_id] = self.data_retrieval_attempts.get(generation_id, 0) + 1
                        MAX_ATTEMPTS = 3
                        if (self.data_retrieval_attempts[generation_id] >= MAX_ATTEMPTS):
                            # If we've failed to retrieve the theorystore after 3 attempts, then mark this request as finished with failure (since we can't get the results).
                            if ("errors" not in workflow_status):
                                workflow_status["errors"] = []
                            if (not isinstance(workflow_status["errors"], list)):
                                workflow_status["errors"] = [str(workflow_status["errors"])]
                            workflow_status["errors"].append(f"Failed to retrieve theorystore after {MAX_ATTEMPTS} attempts. Last retrieval attempt returned None.")

                            self.save_workflow(generation_id=generation_id, workflow_data=workflow_status)

                            # Remove this generation ID from the list of active requests (since it's finished, even though we failed to get the theorystore)
                            self.currently_active_requests.remove(generation_id)
                            self.completed_requests.add(generation_id)

                    # Remove this generation ID from the list of active requests (since it's finished)
                    self.currently_active_requests.remove(generation_id)
                    self.completed_requests.add(generation_id)

                elif (is_finished == True) and (was_successful == False):
                    # If the workflow has finished but was not successful, then mark this request as finished with failure (since the workflow has completed but was not successful).
                    if ("errors" not in workflow_status):
                        workflow_status["errors"] = []
                    if (not isinstance(workflow_status["errors"], list)):
                        workflow_status["errors"] = [str(workflow_status["errors"])]
                    workflow_status["errors"].append(f"Workflow has finished but was not successful.")

                    self.save_workflow(generation_id=generation_id, workflow_data=workflow_status)
                    # Remove this generation ID from the list of active requests (since it's finished, even though it was not successful)
                    self.currently_active_requests.remove(generation_id)
                    self.completed_requests.add(generation_id)

            # Wait a bit between requests to avoid overwhelming the server
            time.sleep(WAIT_TIME_BETWEEN_REQUESTS)


    # User-facing: Get the workflow status for a given generation ID
    # Returns None if not available.
    def get_workflow_status(self, generation_id:str):
        # First check if we have this workflow status saved locally
        local_workflow_status = self.load_workflow(generation_id=generation_id)
        return local_workflow_status

    # User-facing: Get the theorystore for a given generation ID.
    # Returns None if not available.
    def get_theorystore(self, generation_id:str):
        # First check if we have this theorystore saved locally
        local_theorystore = self.load_theorystore(generation_id=generation_id)
        return local_theorystore

    # User-facing: Get a list of the currently-active generation IDs
    def get_active_generation_ids(self):
        return list(self.currently_active_requests)

    # User-facing: Get a list of completed generation IDs
    def get_completed_generation_ids(self):
        return list(self.completed_requests)





#
#   Show a notes/status display
#
def display_status(workflow_status:dict, generation_id:str=None):
    print("-"*80)

    # Check to see if the workflow is in the queue
    queue_position = workflow_status.get("queue_position", None)
    if (queue_position is not None):
        print(f"Generation ID: {generation_id} | Status: In Queue | Queue Position: {queue_position}")
        print("-"*80)
        return

    # Check to see if any data is available
    generation_id_from_workflow = workflow_status.get("generation_id", None)
    if (generation_id_from_workflow is None):
        print("Generation ID: " + str(generation_id))
        print("No data available yet.")
        print("-"*80)
        return

    print("Generation ID: " + str(generation_id_from_workflow))
    print("Theory Query: " + workflow_status.get("theory_query", "N/A"))
    is_finished = workflow_status.get("is_finished", False)
    was_successful = workflow_status.get("was_successful", False)
    print("Status: " + ("Finished" if is_finished else "In Progress")) #+ " | " + ("Successful" if was_successful else "Not Successful"))
    if (is_finished):
        print("Success: " + ("Successful" if was_successful else "Not Successful"))

    print("-"*80)
    print("Notes:")
    notes = workflow_status.get("notes", [])
    for idx, note in enumerate(notes):
        print(f"{idx+1}. {note}")
    print("Errors:")
    errors = workflow_status.get("errors", [])
    for idx, error in enumerate(errors):
        print(f"{idx+1}. {error}")
    print("-"*80)
    print("")



if __name__ == "__main__":
    # Step 1: Submit a theory request and get the generation ID
    SERVER_URL = os.environ["THEORIZER_SERVER_URL"]   # URL for the deployed Modal server
    #SERVER_URL = "http://localhost:5002"        # Local server URL

    # Get the bearer token from the environment variable (if set)
    bearer_token = os.environ.get("THEORIZER_BEARER_TOKEN", None)


    #NUM_PAPERS = 5          # Just for testing
    NUM_PAPERS = 100       # A normal number of papers
    model_str_primary = "gpt-5.4"               # Faster model
    #model_str_primary = "claude-sonnet-4-6"    # Slower model (sometimes so slow it times out)

    model_str_extraction = "gpt-5-mini"         # Model for paper extraction. Should be fast and inexpensive. `gpt-5-mini` recommended.


    theorizer_api = TheorizerModalAPI(server_url=SERVER_URL, bearer_token=bearer_token, local_save_folder="theory-cache/")

    #
    #   Step 1: Submit some sample theory requests (each returns a unique `generation_id` that we can use to track the status and retrieve results).
    #
    stress_test_copies = 1

    for i in range(stress_test_copies):
        response1 = theorizer_api.submit_theory_request(
            user_id="test_user",
            theory_query="Build a theory about how language model agents can best use memory to solve text games.",
            mission_statement="I want to understand how language model agents can best use memory to solve text games, so that I can build better agents in the future.",
            max_papers_to_retrieve=NUM_PAPERS,
            do_qualified_novelty_evaluation=True,
        )
        request1_generation_id = response1.get("generation_id", None)

        response2 = theorizer_api.submit_theory_request(
            user_id="test_user",
            theory_query="Build a theory of the key factors that contribute to the development of Alzheimer's disease.",
            mission_statement="I want to understand the key factors that contribute to the development of Alzheimer's disease, so that I can contribute to research in this area.",
            max_papers_to_retrieve=NUM_PAPERS,
            do_qualified_novelty_evaluation=False,
        )
        request2_generation_id = response2.get("generation_id", None)


    #
    #   Step 2: Poll, waiting for the theory requests to complete.
    #
    POLL_TIME_SEC = 30      # Poll every 60 seconds.  Too often could overload the server.
    # Wait a short time before the first poll
    print("Pausing briefly before starting status polling...")
    time.sleep(10)
    while True:
        print("Polling for workflow status of active requests...")
        theorizer_api.poll()

        # Print the status of all active requests
        active_generation_ids = theorizer_api.get_active_generation_ids()
        print(f"Currently active generation IDs: {active_generation_ids}")

        completed_generation_ids = theorizer_api.get_completed_generation_ids()
        print(f"Completed generation IDs: {completed_generation_ids}")

        # For each active and compelted generation ID, print the most recent status notes and errors (if any)
        for generation_id in active_generation_ids + completed_generation_ids:
            workflow_status = theorizer_api.get_workflow_status(generation_id=generation_id)
            if (workflow_status is not None):
                display_status(workflow_status=workflow_status, generation_id=generation_id)
            else:
                print(f"No workflow status available for generation ID {generation_id}.")

            # Show an example of how to access the theorystore data for a completed generation ID (if available)
            if (generation_id in completed_generation_ids):
                theorystore_data = theorizer_api.get_theorystore(generation_id=generation_id)
                if (theorystore_data is not None):
                    print(f"Theorystore data for generation ID {generation_id} is available. Example keys in theorystore data: {list(theorystore_data.keys())}")
                else:
                    print(f"Theorystore data for generation ID {generation_id} is not yet available.")

        # Show a message about the cache location
        print(f"Workflow statuses and theorystore data (when available) are being saved to the local folder: {theorizer_api.local_save_folder}")

        # If there are no more active requests, then we can stop polling
        if (len(active_generation_ids) == 0):
            print("No more active generation IDs. Stopping polling.")
            break

        # Wait before polling again
        print(f"Waiting for {POLL_TIME_SEC} seconds before polling again...")
        time.sleep(POLL_TIME_SEC)
