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
from Theorizer import *

from Theorizer import GENERATION_OBJECTIVE_ACCURACY_FOCUS, GENERATION_OBJECTIVE_NOVELTY_FOCUS, GENERATION_INPUT_LITERATURE_SUPPORT, GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY

# Globals
theorizer = Theorizer()

# Flask App
app = Flask(__name__)


#
#   Server Endpoints
#

# Get a list of the theories
@app.route("/theorylist", methods=["GET"])
def server_theory_list():
    theory_list = theorizer.theory_store.get_all_theories_as_dict()

    # Return
    packed = {
        "theories": theory_list,
        "count": len(theory_list)
    }
    return jsonify(packed), 200


# Submit a new theory request (manual)
@app.route("/theoryrequestmanual", methods=["POST"])
def server_theory_request_manual():
    # Check that there's data
    data = request.get_json()
    if (not data):
        return jsonify({"error": "No JSON data provided"}), 400

    # Get the theory query from the request data
    theory_query = data.get("theory_query", None)
    num_papers = data.get("num_papers", 0)
    extraction_evaluation_year = data.get("extraction_evaluation_year", 2025)
    extraction_evaluation_month = data.get("extraction_evaluation_month", 1)
    submission_mode = data.get("submission_mode", None)
    model_str = data.get("model_str", None)
    model_str_extraction = data.get("model_str_extraction", "gpt-5-mini")
    generation_objective = data.get("generation_objective", GENERATION_OBJECTIVE_ACCURACY_FOCUS)
    generation_input_type = data.get("generation_input_type", GENERATION_INPUT_LITERATURE_SUPPORT)

    if (theory_query == None):
        return jsonify({"error": "Missing expected field: `theory_query`"}), 400
    if (model_str == None):
        return jsonify({"error": "Missing expected field: `model_str`"}), 400

    # Validate the generation_objective
    known_objectives = [GENERATION_OBJECTIVE_ACCURACY_FOCUS, GENERATION_OBJECTIVE_NOVELTY_FOCUS]
    if (generation_objective not in known_objectives):
        return jsonify({"error": "Invalid generation_objective: " + str(generation_objective) + ". Known options are: " + str(known_objectives)}), 400

    # Validate the generation_input_type
    known_input_types = [GENERATION_INPUT_LITERATURE_SUPPORT, GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY]
    if (generation_input_type not in known_input_types):
        return jsonify({"error": "Invalid generation_input_type: " + str(generation_input_type) + ". Known options are: " + str(known_input_types)}), 400

    # Submit the theory query to the Theorizer API
    if (generation_input_type == GENERATION_INPUT_LITERATURE_SUPPORT):
        theorizer.submit_theory_query(query=theory_query, max_papers_to_retrieve=num_papers, extraction_evaluation_year=extraction_evaluation_year, extraction_evaluation_month=extraction_evaluation_month, model_str_build_new_request=model_str, model_str_extraction=model_str_extraction, generation_objective=generation_objective)
    elif (generation_input_type == GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY):
        theorizer.submit_theory_query_parametric_only(query=theory_query, model_str_build_new_request=model_str, generation_objective=generation_objective)
    else:
        return jsonify({"error": "Invalid generation_input_type: " + str(generation_input_type)}), 400

    time.sleep(1)

    # Signify a successful response
    response = {
        "status": "success",
        "message": f"Theory request submitted successfully with query: {theory_query}",
    }
    return jsonify(response), 202


# Get a specific theory
@app.route("/theory/<theory_id>", methods=["GET"])
def server_theory_get(theory_id):
    # Check that the theory exists
    theory = theorizer.theory_store.get_theory(theory_id)
    if (theory == None):
        return jsonify({"error": "Theory not found"}), 404

    # Return the theory
    packed = {
        "theory": theory.to_dict()
    }
    return jsonify(packed), 200

# Get a specific paper by its paper ID
@app.route("/paper/<paper_id>", methods=["GET"])
def server_paper_get(paper_id):
    # Check that the paper exists
    paper = theorizer.paperstore.get_paper_by_key(paper_id)
    if (paper == None):
        return jsonify({"error": "Paper not found"}), 404

    # Return the paper
    packed = {
        "paper": paper
    }
    return jsonify(packed), 200

# Get a specific extraction schema by ID
@app.route("/extraction-schema/<schema_id>", methods=["GET"])
def server_extraction_schema_get(schema_id):
    # Check that the schema exists
    schema = theorizer.theory_store.get_extraction_schema(schema_id)
    if (schema == None):
        return jsonify({"error": "Extraction schema not found with id" + str(schema_id) + "'"}), 404

    # Return the schema
    packed = {
        "extraction_schema": schema.to_dict()
    }
    return jsonify(packed), 200

# Get a specific theory evaluation result
@app.route("/theory-evaluation/<evaluation_id>", methods=["GET"])
def server_theory_evaluation_get(evaluation_id):
    # Check that the evaluation exists
    evaluation = theorizer.theory_store.get_theory_evaluation(evaluation_id)
    if (evaluation == None):
        return jsonify({"error": "Theory evaluation not found with id '" + str(evaluation_id) + "'"}), 404

    # Return the evaluation
    packed = {
        "theory_evaluation": evaluation.to_dict()
    }
    return jsonify(packed), 200

# Get a specific piece of evidence
@app.route("/extraction/<extraction_id>", methods=["GET"])
def server_evidence_get(extraction_id):
    # There are two possible references for evidence: extraction-result-<num>, or e<num>.<fact_index>.  We should support both.
    # Regardless of which one is used, the first step is to retrieve the full "extraction-result-" record.
    supporting_evidence_id = None
    if (extraction_id.startswith("extraction-result-")):
        supporting_evidence_id = extraction_id
    if (extraction_id.startswith("e")):
        # e<num>.<fact_index>
        # Remove the leading 'e'
        just_numbers = extraction_id[1:]
        fields = just_numbers.split(".")
        if (len(fields) != 2):
            return jsonify({"error": "Invalid evidence ID format"}), 400
        # Reconstruct the supporting evidence ID
        supporting_evidence_id = f"extraction-result-{fields[0]}"
    else:
        return jsonify({"error": "Invalid evidence ID format"}), 400

    # Try to retrieve the evidence record
    supporting_evidence = theorizer.theory_store.get_extraction_result(supporting_evidence_id)
    if (supporting_evidence == None):
        return jsonify({"error": "Supporting evidence not found"}), 404

    # Return the whole evidence record
    supporting_evidence_dict = supporting_evidence.to_dict()
    packed = {
        "id": supporting_evidence_id,
        "extraction": supporting_evidence_dict
    }
    return jsonify(packed), 200


#
#   Save current state
#

# When this is called, it will save the current state of the theorizer to disk.
@app.route("/save", methods=["POST"])
def server_save():
    # Place a date/timestamp on the filename prefix
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename_prefix = f"theorizer-state-{timestamp}"

    # Save the theorizer state
    theorizer.save(filename_prefix=filename_prefix)

    # Return a success message
    return jsonify({"message": "Theorizer state saved successfully", "filename_prefix": filename_prefix}), 200

#
#   Status
#
@app.route("/status", methods=["GET", "POST"])
def get_server_status():
    # Return a packet that shows summary statistics, queue length, etc.

    # Get the workflow statuses
    workflow_statuses = theorizer.get_workflow_statuses()

    # TODO: Should wrap these in timeouts just in case
    # Paperfinder status
    paperfinder_queue_length = theorizer.paperfinder_requests.get_num_requests()
    paperfinder_completed_requests = theorizer.paperfinder_requests.get_num_completed_requests()
    # Paperstore status
    paperstore_queue_length = theorizer.theory_store.paperstore.get_num_requests()
    # Extraction status
    extraction_queue_length = theorizer.schema_extraction_queue.get_num_requests()
    extraction_completed_requests = theorizer.schema_extraction_queue.get_num_completed_requests()

    # Pack status information
    status = {
        "paperfinder_queue_length": paperfinder_queue_length,
        "paperfinder_completed_requests": paperfinder_completed_requests,
        "paperstore_queue_length": paperstore_queue_length,
        "extraction_queue_length": extraction_queue_length,
        "extraction_completed_requests": extraction_completed_requests,
        "num_theories": theorizer.theory_store.num_theories(),
        "num_papers": theorizer.theory_store.paperstore.get_num_papers(),
        "num_extraction_schemas": theorizer.theory_store.num_extraction_schemas(),
        "num_extraction_results": theorizer.theory_store.num_extraction_results(),
        "num_theory_evaluations": theorizer.theory_store.num_theory_evaluations(),
        "workflow_statuses": workflow_statuses,
    }

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

    # Step 1: Define a Theorizer state to load from diskTry to load the theorizer state from disk
    filename_prefix = None                                  # Start fresh
    #filename_prefix = "theorizer-state-20251120-010203     # Load a previous state from disk
    if (filename_prefix != None):
        print(f"Loading theorizer state from '{filename_prefix}'...")
        theorizer.load(filename_prefix=filename_prefix)

    # Step 2: Run the server
    port = 5002
    print("Initializing...")
    # use_reloader (true) causes the server to restart on code changes, which is useful during development.
    app.run(port=port, debug=False, use_reloader=False)
