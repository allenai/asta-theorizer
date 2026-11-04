# (COMPLETED PASS)

# TheorizerServer.py
# This is the main web server for Theorizer.

from __future__ import annotations

from flask import Flask, request, jsonify

import os
import json
import time

from ExtractionUtils import *

from Struct import *
from PaperStore import *
from PaperFinderRequests import *
from Theorizer1 import *

from Theorizer1 import GENERATION_OBJECTIVE_ACCURACY_FOCUS, GENERATION_OBJECTIVE_NOVELTY_FOCUS, GENERATION_INPUT_LITERATURE_SUPPORT, GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY

# Globals
theorizer = Theorizer()

# Flask App
app = Flask(__name__)


#
#   Server Endpoints
#

@app.route("/submit_theory_query", methods=["POST"])
def submit_theory_query():
    # Submit a new theory query to the server.  Return the generation ID and other relevant information about the submission.

    # Step 1: Unpack the request data
    data = request.get_json()
    if (not data):
        return jsonify({"error": "No JSON data provided"}), 400

    theory_query = data.get("theory_query", None)
    user_id = data.get("user_id", "unknown_user")
    mission_statement = data.get("mission_statement", None)
    max_papers_to_retrieve = data.get("max_papers_to_retrieve", 100)
    knowledge_cutoff_year = data.get("knowledge_cutoff_year", 2050)
    knowledge_cutoff_month = data.get("knowledge_cutoff_month", 1)
    model_str_primary = data.get("model_str_primary", DEFAULT_MODEL_PRIMARY)
    model_str_extraction = data.get("model_str_extraction", DEFAULT_MODEL_EXTRACTION)
    generation_objective = data.get("generation_objective", GENERATION_OBJECTIVE_ACCURACY_FOCUS)
    generation_input = data.get("generation_input", GENERATION_INPUT_LITERATURE_SUPPORT)
    max_data_tokens = data.get("max_data_tokens", 100000)
    do_qualified_novelty_evaluation = data.get("do_qualified_novelty_evaluation", False)

    # Validate required fields
    errors = []
    if (theory_query == None) or (not isinstance(theory_query, str)):
        errors.append("Missing or invalid required field: `theory_query` (string)")
    if (mission_statement == None) or (not isinstance(mission_statement, str)):
        errors.append("Missing or invalid required field: `mission_statement` (string)")
    if (not isinstance(max_papers_to_retrieve, int)) or (max_papers_to_retrieve < 0):
        errors.append("Invalid value for field: `max_papers_to_retrieve` must be a non-negative integer")
    if (not isinstance(knowledge_cutoff_year, int)):
        errors.append("Invalid value for field: `knowledge_cutoff_year` must be an integer.")
    if (not isinstance(knowledge_cutoff_month, int)) or (knowledge_cutoff_month < 1) or (knowledge_cutoff_month > 12):
        errors.append("Invalid value for field: `knowledge_cutoff_month` must be an integer between 1 and 12.")
    if (not isinstance(model_str_primary, str)):
        errors.append("Invalid value for field: `model_str_primary` must be a string.")
    elif (len(model_str_primary.strip()) == 0):
        errors.append("Invalid value for field: `model_str_primary` cannot be an empty string.")
    if (not isinstance(model_str_extraction, str)):
        errors.append("Invalid value for field: `model_str_extraction` must be a string.")
    elif (len(model_str_extraction.strip()) == 0):
        errors.append("Invalid value for field: `model_str_extraction` cannot be an empty string.")

    known_generation_objectives = [GENERATION_OBJECTIVE_ACCURACY_FOCUS, GENERATION_OBJECTIVE_NOVELTY_FOCUS]
    if (generation_objective not in known_generation_objectives):
        errors.append("Invalid value for field: `generation_objective` must be one of: " + str(known_generation_objectives))

    known_generation_inputs = [GENERATION_INPUT_LITERATURE_SUPPORT, GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY]
    if (generation_input not in known_generation_inputs):
        errors.append("Invalid value for field: `generation_input` must be one of: " + str(known_generation_inputs))

    # Check User ID
    if (not isinstance(user_id, str)):
        errors.append("Invalid value for field: `user_id` must be a string.")
    elif (len(user_id.strip()) == 0):
        errors.append("Invalid value for field: `user_id` cannot be an empty string.")

    # Check max data tokens (should be at least 50k)
    if (not isinstance(max_data_tokens, int)) or (max_data_tokens < 50000):
        errors.append("Invalid value for field: `max_data_tokens` must be an integer greater than or equal to 50,000.")

    # Qualified novelty evaluation
    if (not isinstance(do_qualified_novelty_evaluation, bool)):
        errors.append("Invalid value for field: `do_qualified_novelty_evaluation` must be a boolean (true or false).")

    if (len(errors) > 0):
        return jsonify({"error": "Invalid request data", "details": errors}), 400

    # Step 2: Submit the theory query to the Theorizer
    submission_result = theorizer.submit_theory_query(
        user_id=user_id,
        theory_query=theory_query,
        mission_statement=mission_statement,
        max_papers_to_retrieve=max_papers_to_retrieve,
        knowledge_cutoff_year=knowledge_cutoff_year,
        knowledge_cutoff_month=knowledge_cutoff_month,
        model_str_primary=model_str_primary,
        model_str_extraction=model_str_extraction,
        generation_objective=generation_objective,
        generation_input=generation_input,
        max_data_tokens=max_data_tokens,
        do_qualified_novelty_evaluation=do_qualified_novelty_evaluation
    )

    # Step 3: Return the submission result
    return jsonify(submission_result), 202



#
#   Get workflow status by generation ID
#
# get_workflow_status_by_generation_id
@app.route("/get_workflow_status_by_id", methods=["POST"])
def workflow_status_by_id():
    # Get the generation ID from the request data
    data = request.get_json()
    if (not data):
        return jsonify({"error": "No JSON data provided"}), 400

    generation_id = data.get("generation_id", None)
    if (generation_id == None) or (not isinstance(generation_id, str)):
        return jsonify({"error": "Missing or invalid required field: `generation_id` (string)"}), 400
    elif (len(generation_id.strip()) == 0):
        return jsonify({"error": "Invalid value for field: `generation_id` cannot be an empty string."}), 400

    # Get the workflow status from the Theorizer
    workflow_status = theorizer.get_workflow_status_by_generation_id(generation_id=generation_id)

    if (workflow_status == None):
        return jsonify({"error": f"No workflow found with generation ID: {generation_id}"}), 404

    # Return the workflow status
    return jsonify(workflow_status), 200


#     def get_theorystore_by_generation_id(self, generation_id:str):
@app.route("/get_theorystore_by_id", methods=["POST"])
def theorystore_by_id():
    # Get the generation ID from the request data
    data = request.get_json()
    if (not data):
        return jsonify({"error": "No JSON data provided"}), 400

    generation_id = data.get("generation_id", None)
    if (generation_id == None) or (not isinstance(generation_id, str)):
        return jsonify({"error": "Missing or invalid required field: `generation_id` (string)"}), 400
    elif (len(generation_id.strip()) == 0):
        return jsonify({"error": "Invalid value for field: `generation_id` cannot be an empty string."}), 400

    # Get the theorystore from the Theorizer
    theorystore = theorizer.get_theorystore_by_generation_id(generation_id=generation_id)

    if (theorystore == None):
        return jsonify({"error": f"No theorystore found for generation ID: {generation_id}"}), 404

    # Return the theorystore
    return jsonify(theorystore), 200



#
#   Status
#
@app.route("/status", methods=["GET", "POST"])
def get_server_status():
    # Return a packet that shows summary statistics about which workflows have completed, are currently active, or are queued.

    status = None
    status = theorizer.get_server_status()

    return jsonify(status), 200


#
#   Default route for the server
#
@app.route("/")
def index():
    return jsonify({"message": "Welcome to the Theorizer Server!"})


#
#   Main Entry Point
#
if __name__ == "__main__":
    # Load the API keys
    loadAPIKeys()

    # Step 1: Run the server
    port = int(os.environ.get("THEORIZER_PORT", 5002))
    print("Initializing Theorizer server...")
    # use_reloader (true) causes the server to restart on code changes, which is useful during development.
    app.run(port=port, debug=False, use_reloader=False)
