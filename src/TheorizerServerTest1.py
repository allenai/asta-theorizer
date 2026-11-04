# TheorizerServerTest1.py
# A quick test of the Theorizer Server API.
import os
import json
import requests

from Theorizer1 import *



if __name__ == "__main__":
    # Step 1: Submit a theory request and get the generation ID
    SERVER_URL = "http://localhost:5002"

    submissions = []

    NUM_PAPERS = 5
    #NUM_PAPERS = 100
    #model_str_primary = "gpt-5.2"
    model_str_primary = "gpt-5.4"
    #model_str_primary = "claude-sonnet-4-6"

    model_str_extraction = "gpt-5-mini"

    #qualified_novelty_evaluation_enabled = True
    qualified_novelty_evaluation_enabled = False

    # Do first submission
    response1 = requests.post(SERVER_URL + "/submit_theory_query", json={
        "user_id": "test_user",
        "theory_query": "Build a theory about how language model agents can best use memory to solve text games.",
        "mission_statement": "I want to understand how language model agents can best use memory to solve text games, so that I can build better agents in the future.",
        "max_papers_to_retrieve": NUM_PAPERS,
        "knowledge_cutoff_year": 2050,
        "knowledge_cutoff_month": 1,
        #"model_str_primary": "gpt-5.2",
        "model_str_primary": model_str_primary,
        "model_str_extraction": model_str_extraction,
        "generation_objective": GENERATION_OBJECTIVE_ACCURACY_FOCUS,
        "generation_input": GENERATION_INPUT_LITERATURE_SUPPORT,
        "max_data_tokens": 250000,
        "do_qualified_novelty_evaluation": qualified_novelty_evaluation_enabled
    })

    if (response1.status_code == 202):
        submission_result1 = response1.json()
        print("Submission 1 successful. Result:")
        print(json.dumps(submission_result1, indent=2))
        submissions.append(submission_result1)
    else:
        print(f"Submission 1 failed with status code {response1.status_code}. Response:")
        print(response1.text)

    # # Do second submission
    # response2 = requests.post(SERVER_URL + "/submit_theory_query", json={
    #     "user_id": "test_user",
    #     "theory_query": "Build a theory of the key factors that contribute to the development of Alzheimer's disease.",
    #     "mission_statement": "I want to understand the key factors that contribute to the development of Alzheimer's disease, so that I can contribute to research in this area.",
    #     "max_papers_to_retrieve": NUM_PAPERS,
    #     "knowledge_cutoff_year": 2050,
    #     "knowledge_cutoff_month": 1,
    #     "model_str_primary": model_str_primary,
    #     "model_str_extraction": model_str_extraction,
    #     "generation_objective": GENERATION_OBJECTIVE_ACCURACY_FOCUS,
    #     "generation_input": GENERATION_INPUT_LITERATURE_SUPPORT,
    #     "max_data_tokens": 250000,
    #     "do_qualified_novelty_evaluation": True
    # })

    # if (response2.status_code == 202):
    #     submission_result2 = response2.json()
    #     print("Submission 2 successful. Result:")
    #     print(json.dumps(submission_result2, indent=2))
    #     submissions.append(submission_result2)
    # else:
    #     print(f"Submission 2 failed with status code {response2.status_code}. Response:")
    #     print(response2.text)



    # Wait for submission status -- periodically poll every 30 seconds.
    POLL_TIME_SEC = 30
    OUTPUT_FOLDER = "output-debug/"
    if (not os.path.exists(OUTPUT_FOLDER)):
        os.makedirs(OUTPUT_FOLDER)

    num_unfinished_submissions = len(submissions)
    while (num_unfinished_submissions > 0):
        print("Waiting for " + str(POLL_TIME_SEC) + " seconds before polling for status of submissions...")
        time.sleep(POLL_TIME_SEC)

        num_unfinished_submissions = 0

        for submission in submissions:
            time.sleep(2) # Sleep a bit between status requests to avoid overwhelming the server

            generation_id = submission.get("generation_id", None)
            if (generation_id == None):
                print("No generation ID found in submission result. Skipping status polling for this submission.")
                continue

            # If this submission is already finished, then skip it.
            if (submission.get("finished", False) == True):
                print(f"Submission with generation ID {generation_id} is already marked as finished. Skipping status polling for this submission.")
                continue
            else:
                num_unfinished_submissions += 1

            print(f"Polling for status of generation ID: {generation_id} ...")
            status_response = requests.post(SERVER_URL + "/get_workflow_status_by_id", json={"generation_id": generation_id})
            if (status_response.status_code == 200):
                status_result = status_response.json()
                print(f"Status for generation ID {generation_id}:")
                print(json.dumps(status_result, indent=4))
                # Write to file
                filename_out = os.path.join(OUTPUT_FOLDER, f"status_{generation_id}.json")
                print("Saving status to file: " + filename_out)
                with open(filename_out, "w") as f:
                    json.dump(status_result, f, indent=4)

                # If the theorystore is available, then also retrieve it.
                is_finished = False
                if ("is_finished" in status_result):
                    is_finished = status_result.get("is_finished", False)

                was_successful = False
                if ("was_successful" in status_result):
                    was_successful = status_result["was_successful"]

                if (was_successful == True):
                    # Get the theory store and save it
                    theorystore_response = requests.post(SERVER_URL + "/get_theorystore_by_id", json={"generation_id": generation_id})
                    if (theorystore_response.status_code == 200):
                        theorystore_result = theorystore_response.json()
                        print(f"Retrieving Theorystore for generation ID {generation_id}:")
                        # Write to file
                        filename_out = os.path.join(OUTPUT_FOLDER, f"theorystore_{generation_id}.json")
                        print("Saving theorystore to file: " + filename_out)
                        with open(filename_out, "w") as f:
                            json.dump(theorystore_result, f, indent=4)
                        submission["finished"] = True

                    else:
                        print(f"Failed to get theorystore for generation ID {generation_id}. Status code: {theorystore_response.status_code}. Response:")
                        print(theorystore_response.text)
                        MAX_ATTEMPTS = 3
                        submission["failed_to_get_theorystore_count"] = submission.get("failed_to_get_theorystore_count", 0) + 1
                        if (submission["failed_to_get_theorystore_count"] >= MAX_ATTEMPTS):
                            print(f"Failed to get theorystore for generation ID {generation_id} after 3 attempts. Marking submission as finished with failure.")
                            submission["finished"] = True
                            submission["errors"] = submission.get("errors", []) + [f"Failed to get theorystore after {MAX_ATTEMPTS} attempts. Last status code: {theorystore_response.status_code}. Last response: {theorystore_response.text}"]

                # If 'is_finished' is True but 'was_successful' is False, then mark this submission as finished with failure (since the workflow has completed but was not successful).
                elif (is_finished == True) and (was_successful == False):
                    print(f"Workflow for generation ID {generation_id} has finished but was not successful. Marking submission as finished with failure.")
                    submission["finished"] = True
                    submission["errors"] = submission.get("errors", []) + [f"Workflow has finished but was not successful. Last status code: {status_response.status_code}. Last response: {status_response.text}"]


            else:
                print(f"Failed to get status for generation ID {generation_id}. Status code: {status_response.status_code}. Response:")
                print(status_response.text)

    print("All submissions have finished processing.")
