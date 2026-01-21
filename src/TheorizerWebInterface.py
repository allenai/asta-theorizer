# (COMPLETED PASS)

# TheorizerWebInterface.py
# This is the main way of interacting with the TheorizerServer.py web server, via a web-based user interface (instead of an API).
from __future__ import annotations

import os
import json
import time
import argparse
import traceback

import requests

import pywebio
from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *
from pywebio.session import run_js

from Theorizer import GENERATION_OBJECTIVE_ACCURACY_FOCUS, GENERATION_OBJECTIVE_NOVELTY_FOCUS, GENERATION_INPUT_LITERATURE_SUPPORT, GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY


from flask import Flask, send_from_directory, abort, current_app, make_response
from flask import make_response, Response

from pywebio.platform.flask import webio_view

app = Flask(__name__)
pywebio.config(title="Theorizer")

SERVER_URL = "http://localhost:5002"


# Options for main Theorizer models (i.e. the "expensive" models)
MODEL_OPTIONS = ["claude-sonnet-4-5-20250929", "openrouter/openai/gpt-oss-120b", "openai/gpt-5", "openai/gpt-4.1-2025-04-14", "openai/gpt-5-mini"]
# Options for the paper extraction models (i.e. the "inexpensive" models)
MODEL_OPTIONS_EXTRACTION = ["gpt-5-mini", "openrouter/openai/gpt-oss-120b"]


# Additional generation options
OBJECTIVE_OPTIONS = [GENERATION_OBJECTIVE_ACCURACY_FOCUS, GENERATION_OBJECTIVE_NOVELTY_FOCUS]
INPUT_OPTIONS = [GENERATION_INPUT_LITERATURE_SUPPORT, GENERATION_INPUT_PARAMETRIC_KNOWLEDGE_ONLY]


#
#   Header
#
# Show main header for all pages
def showHeader():
    # Include FontAwesome
    put_html('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">')

    # Show a small Theorizer logo
    import base64

    FILENAME_LOGO = "images/logo-500.png"

    try:
        # Read and encode the images as base64
        with open(FILENAME_LOGO, "rb") as img_file:
            img_content_logo = base64.b64encode(img_file.read()).decode('utf-8')

        htmlStr  = f"""<div style="display: flex; align-items: center; width: 100%; height: 50px;">"""
        htmlStr += f"""<div style="width: 100%; text-align: left;">"""
        htmlStr += f"""<a href="/" style="text-decoration: none; color: black;"><img src="data:image/png;base64,{img_content_logo}" alt="CodeScientist Logo" style="max-width: 100%; height: 50px;"></a>"""
        htmlStr += f"""</div>"""
        htmlStr += f"""</div>"""
        put_html(htmlStr)


    except Exception as e:
        put_markdown("ERROR: " + str(e))


#
#   Manually add a theory
#
def showTheoryRequestManual():
    run_js('window.location.href="/theoryrequestmanual";')

# Manually add a theory
@app.route('/theoryrequestmanual', methods=['GET', 'POST'])
def createTheoryRequestManual():
    def pywebio_show(message=None):
        # Clear the output
        clear()

        # Header
        showHeader()

        put_markdown("# Theory Request (Manual Entry)")

        put_markdown("*Note: For `Parametric Knowledge Only` input type, the literature-only parameters (e.g. number of papers, extraction model, knowledge cutoff date) are ignored.*")

        # Get user input
        user_input = input_group("Automatically Generate Ideas", [
            # add a selection box for the model to use
            textarea("Theory Query", name='theory_query', rows=1, placeholder="Build a theory of how language models perform arithmetic/mathematical reasoning.", required=True),
            # Base model to use
            select("Select the base model to use for theory generation", options=MODEL_OPTIONS, name='model_str', value=MODEL_OPTIONS[0], required=True),
            # Add the objective (default accuracy-focused)
            select("Select the objective for theory generation", options=OBJECTIVE_OPTIONS, name='objective', value=OBJECTIVE_OPTIONS[0], required=True),
            # Add the input type (default literature-support)
            select("Select the input type for theory generation", options=INPUT_OPTIONS, name='input_type', value=INPUT_OPTIONS[0], required=True),

            # Extraction model to use
            select("(Literature-only) Select the model to use for knowledge extraction", options=MODEL_OPTIONS_EXTRACTION, name='model_str_extraction', value=MODEL_OPTIONS_EXTRACTION[0], required=True),
            # A selection box for the number of papers to use (10, 25, 50, 100, 200)
            select("(Literature-only) Select the number of papers to use", options=[10, 25, 50, 100, 200], name='num_papers', value=100, required=True),
            # Knowledge extraction cutoff date (year). Selection box. Automatically generate options from 1980 to 2030
            select("(Literature-only) Select the literature knowledge cutoff year", options=list(range(1980, 2031)), name='extraction_evaluation_year', value=2025, required=True),
            # Knowledge extraction cutoff date (month). Selection box. Automatically generate options from 1 to 12
            select("(Literature-only) Select the literature knowledge cutoff month", options=list(range(1, 13)), name='extraction_evaluation_month', value=1, required=True),
            # Base model to use
        ])


        # Prepare JSON payload
        payload = {
            'theory_query': user_input['theory_query'],
            'num_papers': user_input['num_papers'],
            'extraction_evaluation_year': user_input['extraction_evaluation_year'],
            'extraction_evaluation_month': user_input['extraction_evaluation_month'],
            'submission_mode': 'manual',
            'model_str': user_input['model_str'],
            'model_str_extraction': user_input['model_str_extraction'],
            'generation_objective': user_input['objective'],
            'generation_input_type': user_input['input_type'],
        }

        # Send POST request to the server
        try:
            response = requests.post(f'{SERVER_URL}/theoryrequestmanual', json=payload)
            if (response.status_code == 202):
                response_data = response.json()
                put_markdown("## Server Response")
                put_text(json.dumps(response_data, indent=4))
            else:
                put_text(f"Server returned an error: {response.status_code}")
        except Exception as e:
            put_text(f"Error communicating with the server: {str(e)}")

    # Get the response from webio_view
    response = webio_view(pywebio_show)()
    # The code below forces the page to not cache, and to reload the page every time, so that the back button in the browser works correctly.
    # Check if the response is a Flask Response object
    if not isinstance(response, Response):
        response = make_response(response)
    # Set cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


#
#   Manually add a batch of theories
#
def showTheoryRequestManualBatch():
    run_js('window.location.href="/theoryrequestmanualbatch";')

# Manually add a theory
@app.route('/theoryrequestmanualbatch', methods=['GET', 'POST'])
def createTheoryRequestManualBatch():
    def pywebio_show(message=None):
        # Clear the output
        clear()

        # Header
        showHeader()

        put_markdown("# Theory Request (Manual Entry, Batch Mode)")

        put_markdown("*Note: For `Parametric Knowledge Only` input type, the literature-only parameters (e.g. number of papers, extraction model, knowledge cutoff date) are ignored.*")

        # Get user input
        user_input = input_group("Automatically Generate Ideas", [
            # add a selection box for the model to use
            textarea("Enter your list of theory queries, one per line", name='theory_query', rows=20, placeholder="Build a theory of how language models perform arithmetic/mathematical reasoning.\nBuild a theory of...", required=True),
            # Base model to use
            select("Select the base model to use for theory generation", options=MODEL_OPTIONS, name='model_str', value=MODEL_OPTIONS[0], required=True),
            # Add the objective (default accuracy-focused)
            select("Select the objective for theory generation", options=OBJECTIVE_OPTIONS, name='objective', value=OBJECTIVE_OPTIONS[0], required=True),
            # Add the input type (default literature-support)
            select("Select the input type for theory generation", options=INPUT_OPTIONS, name='input_type', value=INPUT_OPTIONS[0], required=True),

            # Extraction model to use
            select("(Literature-only) Select the model to use for knowledge extraction", options=MODEL_OPTIONS_EXTRACTION, name='model_str_extraction', value=MODEL_OPTIONS_EXTRACTION[0], required=True),
            # A selection box for the number of papers to use (10, 25, 50, 100, 200)
            select("(Literature-only) Select the number of papers to use", options=[10, 25, 50, 100, 200], name='num_papers', value=100, required=True),
            # Knowledge extraction cutoff date (year). Selection box. Automatically generate options from 1980 to 2030
            select("(Literature-only) Select the literature knowledge cutoff year", options=list(range(1980, 2031)), name='extraction_evaluation_year', value=2025, required=True),
            # Knowledge extraction cutoff date (month). Selection box. Automatically generate options from 1 to 12
            select("(Literature-only) Select the literature knowledge cutoff month", options=list(range(1, 13)), name='extraction_evaluation_month', value=1, required=True),
            # Base model to use
        ])

        # Split the theory queries by newlines and strip whitespace
        theory_queries = [query.strip() for query in user_input['theory_query'].split('\n') if query.strip()]

        # Send a verification dialog that these are the N theories that should be processed.
        if (len(theory_queries) == 0):
            put_text("No theory queries provided. Please enter at least one theory query.")
            return

        put_markdown("## Theories to be Processed")
        put_markdown("The following theories will be processed:")
        for idx, query in enumerate(theory_queries):
            put_markdown(f"{idx + 1}. {query}")
        put_markdown("Please confirm that these are the theories you want to process.")

        confirmation = actions("Confirm Theories", [
            {'label': 'Confirm', 'value': 'confirm', 'color': 'primary'},
            {'label': 'Cancel', 'value': 'cancel', 'color': 'danger'}
        ])
        if (confirmation != 'confirm'):
            put_text("Theory processing cancelled.")
            return

        # If confirmed, proceed with processing each theory query
        put_markdown("## Processing Theories")
        put_markdown(f"Submitting {len(theory_queries)} theories... (this may take a moment, please do not close this window)")
        # Loop through each theory query and send a request for each
        for idx, theory_query in enumerate(theory_queries):
            # Prepare JSON payload
            payload = {
                'theory_query': theory_query,
                'num_papers': user_input['num_papers'],
                'extraction_evaluation_year': user_input['extraction_evaluation_year'],
                'extraction_evaluation_month': user_input['extraction_evaluation_month'],
                'submission_mode': 'manual',
                'model_str': user_input['model_str'],
                'model_str_extraction': user_input['model_str_extraction'],
                'generation_objective': user_input['objective'],
                'generation_input_type': user_input['input_type'],
            }

            # Send POST request to the server
            try:
                response = requests.post(f'{SERVER_URL}/theoryrequestmanual', json=payload)
                if (response.status_code == 202):
                    response_data = response.json()
                    put_markdown("## Server Response")
                    put_text(json.dumps(response_data, indent=4))
                else:
                    put_text(f"Server returned an error: {response.status_code}")
            except Exception as e:
                put_text(f"Error communicating with the server: {str(e)}")

        put_markdown("All theories have been submitted for processing.")

    # Get the response from webio_view
    response = webio_view(pywebio_show)()
    # The code below forces the page to not cache, and to reload the page every time, so that the back button in the browser works correctly.
    # Check if the response is a Flask Response object
    if not isinstance(response, Response):
        response = make_response(response)
    # Set cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


#
#   List of Theories
#

# Retrieve a list of theories from the server
def get_theory_list_from_server():
    try:
        response = requests.get(f'{SERVER_URL}/theorylist')
        if response.status_code == 200:
            return response.json()
        else:
            put_text(f"Server returned an error: {response.status_code}")
            return []
    except Exception as e:
        put_text(f"Error communicating with the server: {str(e)}")
        return []


@app.route('/theorylist', methods=['GET', 'POST'])
def theory_list():
    def pywebio_show():
        # Clear the output
        clear()

        # Header
        showHeader()

        put_markdown("# List of Theories")

        theory_list = []
        # Fetch the theories from the server
        theory_dict = get_theory_list_from_server()
        # Convert from a dict to a list
        theories = theory_dict.get('theories', {})
        for value in theories.values():
            # Add the theory to the list
            theory_list.append(value)

        print(json.dumps(theory_list, indent=3))
        # Sort the list by theory ID, descending.  BUT, theory ID is a string of the form "theory-<id>", so we need to extract the ID and convert it to an integer for sorting.
        for theory in theory_list:
            try:
                id = theory['id']
                id_numerical = int(id.split('-')[1])  # Extract the numerical part of the ID
                theory['id_numerical'] = id_numerical  # Add a new field for sorting
            except Exception as e:
                # Use a default numerical ID of -1
                theory['id_numerical'] = -1
        # Sort the theory list by the numerical ID in descending order
        theory_list.sort(key=lambda x: x['id_numerical'], reverse=True)


        # Show the theories in a table.
        table_rows = []
        for theory in theory_list:
            theory_description_html = "<b>Name:</b> " + theory["name"] + "<br><b>Description:</b> " + theory["description"]
            derived_from = theory.get('derived_from', [])
            if (len(derived_from) > 0):
                theory_description_html += "<br><b>Derived From:</b> " + ", ".join([f"[{d}]" for d in derived_from])
            # LLM baseline
            original_theory_id = theory.get('components', {}).get('original_theory_id', None)
            if (original_theory_id is not None):
                theory_description_html += f"<br><b>LLM Baseline of Theory ID:</b> {original_theory_id}"

            # Show the base model used to build this theory
            model_str = theory.get('model_str', None)
            if (model_str is not None):
                theory_description_html += f"<br><b>Model:</b> {model_str}"

            theory_url = f"/theory/{theory['id']}"

            table_rows.append([
                theory["id"],
                put_html(theory_description_html),
                put_button("Details", onclick=lambda url=theory_url: run_js(f'window.open("{url}", "_blank");'))
            ])

        # Show the table
        clear('theory_list_table')
        with use_scope('theory_list_table'):
            put_table([
                ["Theory ID", "Theory Description", ""],  # Header row
                *table_rows  # Unpack the list of rows
            ])


    # Get the response from webio_view
    response = webio_view(pywebio_show)()

    # The code below forces the page to not cache, and to reload the page every time, so that the back button in the browser works correctly.
    # Check if the response is a Flask Response object
    if not isinstance(response, Response):
        response = make_response(response)

    # Set cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response



#
#   Show specific theory details
#
@app.route('/theory/<theory_id>', methods=['GET', 'POST'])
def theory_detail(theory_id):
    print("STARTED! Theory Detail for ID:", theory_id)
    def pywebio_show():
        # Clear the output
        clear()

        # Header
        showHeader()

        put_markdown(f"# Theory Details for {theory_id}")

        # Fetch the theory from the server
        theory = None
        try:
            response = requests.get(f'{SERVER_URL}/theory/{theory_id}')
            if response.status_code == 200:
                theory_data = response.json()
                theory = theory_data.get('theory', {})
            else:
                put_text(f"Server returned an error: {response.status_code}")
        except Exception as e:
            put_text(f"Error communicating with the server: {str(e)}")

        if (theory is None):
            put_text("No theory found with that ID.")
            return

        # Show the theory details
        put_markdown("## Theory (General Information)")

        # Unpack
        id = theory.get('id', None)
        name = theory.get('name', None)
        type = theory.get('type', None)
        description = theory.get('description', None)
        theory_query = theory.get('theory_query', None)
        knowledge_cutoff_year = theory.get('knowledge_cutoff_year', None)
        knowledge_cutoff_month = theory.get('knowledge_cutoff_month', None)
        components = theory.get('components', {})

        derived_from = theory.get('derived_from', [])
        change_log = theory.get('change_log', [])

        theory_evaluation_ids = theory.get('theory_evaluation_ids', [])

        theory_statements = components.get('theory_statements', [])
        supporting_evidence = components.get('supporting_evidence', [])
        new_predictions_likely = components.get('new_predictions_likely', [])
        new_predictions_unknown = components.get('new_predictions_unknown', [])
        negative_experiments = components.get('negative_experiments', [])
        unaccounted_for_evidence = components.get('unaccounted_for', [])
        conflicting_evidence = components.get('conflicting_evidence', [])
        special_cases = components.get('special_cases', [])

        version = components.get('version', None)
        model_str = components.get('model_str', None)
        model_str_extraction = components.get('model_str_extraction', None)


        # Display (General)
        put_markdown(f"**ID:** {id}")
        put_markdown(f"**Name:** {name}")
        put_markdown(f"**Type:** {type}")
        if (theory_query is not None):
            put_markdown(f"**Theory Query:** {theory_query}")
        put_markdown(f"**Description:** {description}")

        # Put down a light horizontal rule
        put_markdown("### Generation Parameters")
        put_markdown(f"**Knowledge Cutoff Year:** {knowledge_cutoff_year}")
        put_markdown(f"**Knowledge Cutoff Year:** {knowledge_cutoff_year}")
        put_markdown(f"**Knowledge Cutoff Month:** {knowledge_cutoff_month}")
        # Display Generation Models
        put_markdown(f"**Prompt Workflow Version:** {version}")
        put_markdown(f"**Generation Model:** {model_str}")
        put_markdown(f"**Extraction Model:** {model_str_extraction}")

        # Unused in this interface
        # # Display (Derived from)
        # put_markdown("## Theory (Derived From)")
        # # Links to the theory IDs that this theory is derived from
        # if (derived_from is None or len(derived_from) == 0):
        #     put_markdown("**Derived From:** <i>None</i>")
        # else:
        #     dervied_from_str = "**Derived From:** " + " ".join(["[[" + str(derived_from_id) + f"]](/theory/{derived_from_id})" for derived_from_id in derived_from])
        #     put_markdown(dervied_from_str)

        # # Display (Change Log)
        # if (len(change_log) > 0):
        #     put_markdown("**Change Log:**")
        #     # Unordered list of change log entries in Markdown
        #     for entry in change_log:
        #         put_markdown(f"- {entry}")
        # else:
        #     put_markdown("**Change Log:** <i>No change log entries.</i>")

        # Unused in this interface
        # # Display (evaluations of this theory)
        # put_markdown("## Evaluations of this Theory")
        # if (len(theory_evaluation_ids) > 0):
        #     for idx, evaluation_id in enumerate(theory_evaluation_ids):
        #         # Make this a link to the theory evaluation details page
        #         put_markdown(f"{idx + 1}. [{evaluation_id}](../theory-evaluation/{evaluation_id})")
        # else:
        #     put_markdown("<i>No evaluations of this theory.</i>")

        # Display (details)
        put_markdown("## Theory (Details)")

        # Check to see the format of the theory -- are the theory statements a list of strings, or dicts?
        theory_format = None
        if (len(theory_statements) > 0):
            first_statement = theory_statements[0]
            if isinstance(first_statement, str):
                theory_format = 'string_list'
            elif isinstance(first_statement, dict):
                theory_format = 'dict_list'

        if (theory_format == 'string_list') or (theory_format is None):
            # Theory statements
            put_markdown("### Theory Statements")
            if (len(theory_statements) > 0):
                for idx, statement in enumerate(theory_statements):
                    put_markdown(f"{idx + 1}. {statement}")
            else:
                put_markdown("<i>No theory statements provided.</i>")

            # Supporting evidence
            put_markdown("### Supporting Evidence")
            if (len(supporting_evidence) > 0):
                for idx, evidence in enumerate(supporting_evidence):
                    evidence_text = evidence.get('text', None)
                    uuids = evidence.get('uuids', [])
                    markdown_str = f"{idx + 1}. {evidence_text} "           # Evidence text
                    for uuid in uuids:
                        markdown_str += f"[[{uuid}]](../extraction/{uuid}) "    # Link to the individual pieces of supporting evidence (by their UUIDs)
                    put_markdown(markdown_str)
            else:
                put_markdown("<i>No supporting evidence provided.</i>")

        elif (theory_format == 'dict_list'):
            # List of dicts with 3 keys: `if`, `then`, and `supporting_evidence`.  `if` and `then` are lists of triples, and `supporting_evidence` is a list of evidence dicts (`text` and `uuid`, as above).
            put_markdown("### Theory Statements")
            format_type = "if-then"
            format_type = "regular"

            if (len(theory_statements) > 0):
                if (format_type == "regular"):
                    for idx, statement in enumerate(theory_statements):
                        statement_name = statement.get('statement_name', None)
                        theory_statement = statement.get('theory_statement', None)
                        domain_scope = statement.get('domain_scope', None)
                        special_cases = statement.get('special_cases', [])
                        supporting_evidence_st = statement.get('supporting_evidence', [])
                        qual_or_quant = statement.get('qual_or_quant', None)
                        novelty_evaluation = statement.get('novelty_evaluation', {})

                        put_markdown(f"#### Statement {idx + 1}: {statement_name}")
                        put_markdown(f"**Theory Statement:** {theory_statement}")
                        put_markdown(f"**Scope/Domain:** {domain_scope}")
                        put_markdown(f"**Qualitative or Quantitative:** {qual_or_quant}")

                        # Special cases (a list)
                        put_markdown("**Special Cases:**")
                        if (len(special_cases) > 0):
                            for jdx, case in enumerate(special_cases):
                                put_markdown(f"{jdx + 1}. {case}")
                        else:
                            put_markdown("<i>No special cases provided.</i>")


                        put_markdown("**Supporting Evidence:**")
                        if (len(supporting_evidence_st) > 0):
                            for jdx, evidence in enumerate(supporting_evidence_st):
                                evidence_text = evidence.get('text', None)
                                uuids = evidence.get('uuids', [])
                                markdown_str = f"{jdx + 1}. {evidence_text} "           # Evidence text
                                for uuid in uuids:
                                    markdown_str += f"[[{uuid}]](../extraction/{uuid}) "    # Link to the individual pieces of supporting evidence (by their UUIDs)
                                put_markdown(markdown_str)
                        else:
                            put_markdown("<i>No supporting evidence provided.</i>")

                        put_markdown("**Novelty Evaluation:**")
                        what_already_exists = novelty_evaluation.get('what_already_exists', None)
                        what_is_novel = novelty_evaluation.get('what_is_novel', None)
                        classification_explanation = novelty_evaluation.get('classification_explanation', None)
                        likely_classification = novelty_evaluation.get('likely_classification', None)
                        references = novelty_evaluation.get('references', [])

                        put_markdown(f"- **What Already Exists:** {what_already_exists}")
                        put_markdown(f"- **What is Novel:** {what_is_novel}")
                        put_markdown(f"- **Classification Explanation:** {classification_explanation}")
                        put_markdown(f"- **Likely Classification:** {likely_classification}")
                        if (len(references) > 0):
                            put_markdown("- **References:**")
                            for jdx, reference in enumerate(references):
                                put_markdown(f"{jdx + 1}. {reference}")
                                print("REFERENCE:", reference)
                        else:
                            put_markdown(" **References:** <i>No references provided.</i>")

                        # Put down a light horizontal rule
                        put_markdown("---")

        else:
            put_markdown("<i>Unknown theory statement format.</i>")


        # New predictions (that are likely)
        put_markdown("### New Predictions (Likely outcome)")
        if (len(new_predictions_likely) > 0):
            for idx, prediction in enumerate(new_predictions_likely):
                put_markdown(f"{idx + 1}. {prediction}")
        else:
            put_markdown("<i>No new predictions (likely) provided.</i>")

        # New predictions (unknown)
        put_markdown("### New Predictions (Unknown outcome/high-entropy)")
        if (len(new_predictions_unknown) > 0):
            for idx, prediction in enumerate(new_predictions_unknown):
                put_markdown(f"{idx + 1}. {prediction}")
        else:
            put_markdown("<i>No new predictions (unknown) provided.</i>")

        # Negative experiments
        put_markdown("### Negative Experiments")
        if (len(negative_experiments) > 0):
            for idx, experiment in enumerate(negative_experiments):
                put_markdown(f"{idx + 1}. {experiment}")
        else:
            put_markdown("<i>No negative experiments provided.</i>")

        # Unaccounted for evidence
        put_markdown("### Unaccounted for Evidence")
        if (len(unaccounted_for_evidence) > 0):
            for idx, evidence in enumerate(unaccounted_for_evidence):
                evidence_text = evidence.get('text', None)
                uuids = evidence.get('uuids', [])
                markdown_str = f"{idx + 1}. {evidence_text} "           # Evidence text
                for uuid in uuids:
                    markdown_str += f"[[{uuid}]](../extraction/{uuid}) "
                put_markdown(markdown_str)
        else:
            put_markdown("<i>No unaccounted for evidence provided.</i>")

        # Conflicting Evidence
        put_markdown("### Conflicting Evidence")
        if (len(conflicting_evidence) > 0):
            for idx, evidence in enumerate(conflicting_evidence):
                evidence_text = evidence.get('text', None)
                uuids = evidence.get('uuids', [])
                markdown_str = f"{idx + 1}. {evidence_text} "           # Evidence text
                for uuid in uuids:
                    markdown_str += f"[[{uuid}]](../extraction/{uuid}) "
                put_markdown(markdown_str)
        else:
            put_markdown("<i>No conflicting evidence provided.</i>")

        # Special Cases
        put_markdown("### Special Cases")
        if (len(special_cases) > 0):
            for idx, case in enumerate(special_cases):
                put_markdown(f"{idx + 1}. {case}")
        else:
            put_markdown("<i>No special cases provided.</i>")


        # Existing Theory
        put_markdown("### Existing Theory")
        existing_theory = components.get('existing_theory', None)
        if (existing_theory is not None):
            likely_classification = existing_theory.get('likely_classification', None)
            references = existing_theory.get('references', [])
            put_markdown(f"**Likely Classification:** {likely_classification}")
            put_markdown("**References:**")
            if (len(references) > 0):
                for idx, reference in enumerate(references):
                    put_markdown(f"{idx + 1}. {reference}")
            else:
                put_markdown("<i>No references provided.</i>")
        else:
            put_markdown("<i>No existing theory information provided.</i>")


        # Display (debug)
        put_markdown("## Theory Components (Debug)")
        put_markdown("```json\n" + json.dumps(components, indent=4) + "\n```")


    # Get the response from webio_view
    response = webio_view(pywebio_show)()
    # The code below forces the page to not cache, and to reload the page every time, so that the back button in the browser works correctly.
    # Check if the response is a Flask Response object
    if not isinstance(response, Response):
        response = make_response(response)
    # Set cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# Show the details of a specific piece of extracted supporting evidence
@app.route('/extraction/<uuid>', methods=['GET'])
def evidence_detail(uuid):
    def pywebio_show():
        # Clear the output
        clear()

        # Header
        showHeader()

        put_markdown(f"# Extracted Data from Paper: Details for {uuid}")

        # Fetch the evidence from the server
        extraction_id = None
        extraction = None
        try:
            response = requests.get(f'{SERVER_URL}/extraction/{uuid}')
            if response.status_code == 200:
                evidence_data = response.json()
                extraction_id = evidence_data.get('id', None)
                extraction = evidence_data.get('extraction', {})
            else:
                put_text(f"Server returned an error: {response.status_code}")
        except Exception as e:
            put_text(f"Error communicating with the server: {str(e)}")

        if (extraction is None):
            put_text("No extracted data found with that UUID.")
            return


        # Unpack
        extraction_result_id = extraction.get("id", None)
        extraction_schema_id = extraction.get("extraction_schema_id", None)
        paper_id = extraction.get("paper_id", None)
        cost = extraction.get("cost", None)
        extracted_data = extraction.get("extracted_data", [])
        potentially_relevant_new_papers = extraction.get("potentially_relevant_new_papers", [])

        # Also try to get the paper's data from the server (using the paper_id)
        paper_record = {}
        if (paper_id is not None):
            try:
                response = requests.get(f'{SERVER_URL}/paper/{paper_id}')
                if response.status_code == 200:
                    paper_data = response.json()
                    paper_record = paper_data.get('paper', {})
                else:
                    put_text(f"Server returned an error: {response.status_code}")
            except Exception as e:
                put_text(f"Error communicating with the server: {str(e)}")

        paper_title = paper_record.get('title', None)
        s2_metadata = paper_record.get('s2_metadata', {})
        paper_venue = s2_metadata.get('venue', None)
        paper_year = s2_metadata.get('year', None)
        paper_url = s2_metadata.get('url', None)
        paper_tldr1 = s2_metadata.get('tldr', None)
        paper_tldr = None
        if (paper_tldr1 is not None and isinstance(paper_tldr1, dict)):
            paper_tldr = paper_tldr1.get('text', None)
        paper_abstract = s2_metadata.get("abstract", None)
        paper_markdown = paper_record.get('paper_markdown', None)

        # Also try to get the extraction schema, and extract a single field from it (the 'extraction_query')
        extraction_schema = None
        extraction_query = None
        if (extraction_schema_id is not None):
            try:
                response = requests.get(f'{SERVER_URL}/extraction-schema/{extraction_schema_id}')
                if response.status_code == 200:
                    schema_data = response.json()
                    extraction_schema = schema_data.get('extraction_schema', {})
                else:
                    put_text(f"Server returned an error: {response.status_code}")
            except Exception as e:
                put_text(f"Error communicating with the server: {str(e)}")
        if (extraction_schema is not None):
            extraction_query = extraction_schema.get('extraction_query', None)



        # Show the evidence details
        put_markdown("## Extracted Data (Header)")

        # Display (General)
        put_markdown(f"**Extraction ID:** {extraction_id}")
        if (extraction_schema_id is not None):
            put_markdown(f"**Extraction Schema Used (ID):** [extraction-schema-{extraction_schema_id}](../extraction-schema/{extraction_schema_id})")
        if (extraction_query is not None):
            put_markdown(f"**Extraction Query:** {extraction_query}")

        put_markdown(f"**Paper ID:** {paper_id}")
        if (paper_title is not None):
            # Make this a link to the paper URL, if it's not none
            if (paper_url is not None):
                put_markdown(f"**Paper Title:** [{paper_title}]({paper_url})")
            else:
                put_markdown(f"**Paper Title:** {paper_title}")
        if (paper_year is not None):
            put_markdown(f"**Paper Year:** {paper_year}")
        if (paper_venue is not None):
            put_markdown(f"**Paper Venue:** {paper_venue}")
        if (paper_tldr is not None):
            put_markdown(f"**Paper TL;DR:** {paper_tldr}")
        if (paper_abstract is not None):
            put_markdown(f"**Paper Abstract:** {paper_abstract}")

        cost_rounded = round(cost, 3) if cost is not None else "--"
        put_markdown(f"**Cost:** {cost_rounded}")

        # Display (extracted data)
        put_markdown("## Extracted Data (Details)")
        if (len(extracted_data) > 0):
            for idx, data in enumerate(extracted_data):
                # Get the UUID first
                data_uuid = data.get('uuid', None)
                put_markdown("### Extracted Data Instance " + str(idx) + " (" + str(data_uuid) + ")")
                # Show the data (in a table).  Column 1 is the field name (dictionary key), column 2 is the value (dictionary value).
                data_table = []
                # Header
                data_table.append(["Field", "Value"])
                for key, value in data.items():
                    if (key == 'uuid'):
                        # Skip the UUID field, it's already shown in the header
                        continue
                    # Add the row to the table
                    key_str = "<span style='font-weight: bold;'>" + key + "</span>"
                    data_table.append([put_html(key_str), str(value)])
                # Show the table
                put_table(data_table)

        # Note the potentially relevant papers also extracted
        put_markdown("## Potentially Relevant New Papers (mentioned by this paper)")
        if (len(potentially_relevant_new_papers) > 0):
            for idx, paper in enumerate(potentially_relevant_new_papers):
                paper_title = paper.get('paper_title', None)
                rating = paper.get('rating', None)
                put_markdown(f"{idx + 1}. {paper_title} *(Rating: {rating})*")
        else:
            put_markdown("<i>No potentially relevant new papers extracted.</i>")

        put_markdown("## Extracted Data (debug)")
        put_markdown("```json\n" + json.dumps(extraction, indent=4) + "\n```")

        # Paper
        put_markdown("## Paper")
        if (paper_markdown is not None):
            put_markdown(paper_markdown)
        else:
            put_markdown("<i>No paper full-text available.</i>")



    # Get the response from webio_view
    response = webio_view(pywebio_show)()
    # The code below forces the page to not cache, and to reload the page every time, so that the back button in the browser works correctly.
    # Check if the response is a Flask Response object
    if not isinstance(response, Response):
        response = make_response(response)
    # Set cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# Show the details of a specific extraction schema
@app.route('/extraction-schema/<schema_id>', methods=['GET'])
def extraction_schema_detail(schema_id):
    print("STARTED! Extraction Schema Detail for ID:", schema_id)
    def pywebio_show():
        # Clear the output
        clear()

        # Header
        showHeader()

        put_markdown(f"# Extraction Schema Details for {schema_id}")

        # Fetch the schema from the server
        schema = None
        try:
            response = requests.get(f'{SERVER_URL}/extraction-schema/{schema_id}')
            if response.status_code == 200:
                schema_data = response.json()
                schema = schema_data.get('extraction_schema', {})
            else:
                put_text(f"Server returned an error: {response.status_code}")
        except Exception as e:
            put_text(f"Error communicating with the server: {str(e)}")

        if (schema is None):
            put_text("No extraction schema found with that ID.")
            return

        # Unpack
        schema_id_ = schema.get('id', None)
        extraction_query = schema.get('extraction_query', None)
        schema = schema.get('schema', [])


        # Show the schema details
        put_markdown("## Extraction Schema (General Information)")
        put_markdown(f"**Schema ID:** {schema_id_}")
        put_markdown(f"**Extraction Query:** {extraction_query}")

        # Display (details)
        put_markdown("## Extraction Schema (Details)")
        if (len(schema) > 0):
            # Display as a table: 3 fields ('name', 'type', 'description').
            schema_table = []
            # Header
            schema_table.append(["Field Name", "Type", "Description"])
            for field in schema:
                field_name = field.get('name', 'None')
                field_type = field.get('type', 'None')
                field_description = field.get('description', 'No description provided.')
                field_name_str = "<span style='font-weight: bold;'>" + field_name + "</span>"
                # Add the row to the table
                schema_table.append([put_html(field_name), field_type, field_description])
            # Show the table
            put_table(schema_table)
        else:
            put_markdown("<i>No fields in this extraction schema.</i>")


        # Debug
        put_markdown("## Extraction Schema (Debug)")
        put_markdown("```json\n" + json.dumps(schema, indent=4) + "\n```")

    # Show this page, no caching so the back button works
    #return webio_view(pywebio_show)()
    # Get the response from webio_view
    response = webio_view(pywebio_show)()
    # The code below forces the page to not cache, and to reload the page every time, so that the back button in the browser works correctly.
    # Check if the response is a Flask Response object
    if not isinstance(response, Response):
        response = make_response(response)
    # Set cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


#
#   Status
#

# Show the queue status from the server
def showStatus():
    # Request the status from the server using the `/status` endpoint.  It will respond with a JSON object.
    endpoint = f"{SERVER_URL}/status"
    response_data = None
    try:
        payload = {}         # Empty payload
        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            response_data = response.json()

            htmlOut = ""
            # Show the new server status output, using appropriate fontawesome icons
            htmlOut += "<b>Theory Store:</b> "
            htmlOut += "<ul>"
            htmlOut += f"<li> <i class='fas fa-file'></i> {response_data['num_papers']} papers </li>"
            htmlOut += f"<li> <i class='fas fa-lightbulb'></i> {response_data['num_theories']} theories </li>"
            htmlOut += f"<li> <i class='fas fa-cogs'></i> {response_data['num_extraction_schemas']} extraction schemas </li> "
            htmlOut += f"<li> <i class='fas fa-search'></i> {response_data['num_extraction_results']} extraction results </li> "
            htmlOut += f"<li> <i class='fas fa-chart-bar'></i> {response_data['num_theory_evaluations']} theory evaluations </li>"
            htmlOut += "</ul>"
            htmlOut += "<br><br>"

            htmlOut += "<b>Queue Lengths:</b><br>"
            htmlOut += "<ul>"
            # Paperfinder
            htmlOut += "<li><i class='fas fa-tasks'></i> PaperFinder: " + str(response_data['paperfinder_queue_length']) + " queued (" + str(response_data['paperfinder_completed_requests']) + " completed requests) </li>"
            # PaperStore
            htmlOut += "<li><i class='fas fa-database'></i> PaperStore: " + str(response_data['paperstore_queue_length']) + " queued </li>"
            # Schema Extraction
            htmlOut += "<li><i class='fas fa-file-alt'></i> Schema Extraction: " + str(response_data['extraction_queue_length']) + " queued (" + str(response_data['extraction_completed_requests']) + " completed requests) </li>"

            htmlOut += "</ul>"
            htmlOut += "<br><br>"

            active_workflows = response_data.get('workflow_statuses', {}).get('active_workflows', [])
            completed_workflows = response_data.get('workflow_statuses', {}).get('completed_workflows', [])

            htmlOut += "<b>Workflow Statuses:</b><br>"
            htmlOut += "<i>Workflows in process</i><br>"

            if (len(active_workflows) == 0):
                htmlOut += "<i style='color: grey;'>No active workflows</i><br>"
            else:
                htmlOut += "<ol>"
                for workflow in active_workflows:
                    workflow_name = workflow.get("name", "None")
                    workflow_current_step = workflow.get("current_step", "None")
                    workflow_is_completed = workflow.get("is_completed", False)
                    workflow_has_errors = workflow.get("has_errors", False)
                    workflow_errors = workflow.get("errors", [])
                    workflow_status_str = workflow.get("status_str", "")
                    workflow_cost = workflow.get("cost", 0.0)
                    workflow_runtime_seconds = workflow.get("runtime_sec", 0.0)
                    workflow_runtime_steps = workflow.get("runtime_steps", {})
                    # Make a workflow runtime string of the form "00h00m00s".  Truncate to "00m00s" if no hours, and "00s" if no minutes.
                    workflow_runtime_str = ""
                    if (workflow_runtime_seconds >= 3600):
                        workflow_runtime_str = f"{int(workflow_runtime_seconds // 3600):02d}h{int((workflow_runtime_seconds % 3600) // 60):02d}m{int(workflow_runtime_seconds % 60):02d}s"
                    elif (workflow_runtime_seconds >= 60):
                        workflow_runtime_str = f"{int(workflow_runtime_seconds // 60):02d}m{int(workflow_runtime_seconds % 60):02d}s"
                    else:
                        workflow_runtime_str = f"{int(workflow_runtime_seconds):02d}s"

                    htmlOut += f"<li>{workflow_name} (Current Step: {workflow_current_step}: {workflow_status_str}) [Cost: {workflow_cost:.2f}, Runtime: {workflow_runtime_str}]"
                    # Runtimes of steps (one per line)
                    if (len(workflow_runtime_steps) > 0):
                        htmlOut += "<ul>"
                        for step_name, step_runtime in workflow_runtime_steps.items():
                            # Do not include anything that ends with "start"
                            if (step_name.endswith("start")):
                                continue
                            # Show the runtime in minutes and seconds (no hours).
                            # step_runtime is in seconds, so convert to minutes and seconds.
                            step_runtime_minutes = step_runtime / 60
                            # Round down for minutes, and get the remainder for seconds
                            step_runtime_minutes = int(step_runtime_minutes)
                            step_runtime_seconds = step_runtime % 60
                            step_runtime_str = f"{step_runtime_minutes:02d}m{int(step_runtime_seconds):02d}s  ({step_runtime:.2f} seconds)"
                            htmlOut += f"<li>{step_name}: {step_runtime_str}</li>"
                        htmlOut += "</ul>"
                    if (workflow_has_errors):
                        # Signal these in red
                        htmlOut += "<span style='color: red;'> - Errors: " + str(workflow_has_errors) + "</span>"

                    htmlOut += "</li>"

                htmlOut += "</ol>"
            htmlOut += "<br><br>"

            htmlOut += "<i>Workflows completed</i><br>"
            if (len(completed_workflows) == 0):
                htmlOut += "<i style='color: grey;'>No completed workflows</i><br>"
            else:
                htmlOut += "<ol>"
                for workflow in completed_workflows:
                    workflow_name = workflow.get("name", "None")
                    workflow_is_completed = workflow.get("is_completed", False)
                    workflow_has_errors = workflow.get("has_errors", False)
                    workflow_errors = workflow.get("errors", [])
                    workflow_status_str = workflow.get("status_str", "")
                    workflow_cost = workflow.get("cost", 0.0)
                    workflow_runtime_seconds = workflow.get("runtime_sec", 0.0)
                    workflow_runtime_steps = workflow.get("runtime_steps", {})
                    # Make a workflow runtime string of the form "00h00m00s".  Truncate to "00m00s" if no hours, and "00s" if no minutes.
                    workflow_runtime_str = ""
                    if (workflow_runtime_seconds >= 3600):
                        workflow_runtime_str = f"{int(workflow_runtime_seconds // 3600):02d}h{int((workflow_runtime_seconds % 3600) // 60):02d}m{int(workflow_runtime_seconds % 60):02d}s"
                    elif (workflow_runtime_seconds >= 60):
                        workflow_runtime_str = f"{int(workflow_runtime_seconds // 60):02d}m{int(workflow_runtime_seconds % 60):02d}s"
                    else:
                        workflow_runtime_str = f"{int(workflow_runtime_seconds):02d}s"

                    htmlOut += f"<li>{workflow_name} [Cost: {workflow_cost:.2f}, Runtime: {workflow_runtime_str}]"
                    # Runtimes of steps (one per line)
                    if (len(workflow_runtime_steps) > 0):
                        htmlOut += "<ul>"
                        for step_name, step_runtime in workflow_runtime_steps.items():
                            # Do not include anything that ends with "start"
                            if (step_name.endswith("start")):
                                continue
                            # Show the runtime in minutes and seconds (no hours).
                            # step_runtime is in seconds, so convert to minutes and seconds.
                            step_runtime_minutes = step_runtime / 60
                            # Round down for minutes, and get the remainder for seconds
                            step_runtime_minutes = int(step_runtime_minutes)
                            step_runtime_seconds = step_runtime % 60
                            step_runtime_str = f"{step_runtime_minutes:02d}m{int(step_runtime_seconds):02d}s  ({step_runtime:.2f} seconds)"
                            htmlOut += f"<li>{step_name}: {step_runtime_str}</li>"
                        htmlOut += "</ul>"

                    if (workflow_has_errors):
                        # Signal these in red
                        htmlOut += "<span style='color: red;'> - Errors: " + str(workflow_has_errors) + "</span>"
                    htmlOut += "</li>"

                htmlOut += "</ol>"
            htmlOut += "<br><br>"


            # Show the HTML
            put_html(htmlOut)

        else:
            put_text(f"Server returned an error: {response.status_code}")
    except Exception as e:
        put_text(f"Error communicating with the server: {str(e)}")

    # Also save the most recent status to a file (in the /status directory)
    pathOut = "status/"
    if (not os.path.exists(pathOut)):
        os.makedirs(pathOut)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(pathOut, f"server-status-{timestamp}.json")
    try:
        with open(filename, 'w') as f:
            json.dump(response_data, f, indent=4)
    except Exception as e:
        print("Error saving server status to file:", str(e))

    pass


#
#   Save the server state
#
@app.route('/save', methods=['GET'])
def save_server_state():
    def pywebio_show():
        # Clear the output
        clear()

        # Header
        showHeader()

        put_markdown("# Export Theorizer State")

        # Request the server to save its state
        endpoint = f"{SERVER_URL}/save"
        try:
            payload = {}  # Empty payload
            response = requests.post(endpoint, json=payload)

            if response.status_code == 200:
                # Keys are 'message' and 'filename_prefix'
                message = response.json().get('message', 'No message provided')
                filename_prefix = response.json().get('filename_prefix', 'No filename prefix provided')

                put_markdown("## Export State Response")
                put_text(f"Message: {message}")
                put_text(f"Filename Prefix: {filename_prefix}")

            else:
                put_text(f"Server returned an error: {response.status_code}")
        except Exception as e:
            put_text(f"Error communicating with the server: {str(e)}")

    # Show this page, no caching so the back button works
    # Get the response from webio_view
    response = webio_view(pywebio_show)()
    # The code below forces the page to not cache, and to reload the page every time, so that the back button in the browser works correctly.
    # Check if the response is a Flask Response object
    if not isinstance(response, Response):
        response = make_response(response)
    # Set cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


#
#   Show server status
#
@app.route('/status', methods=['GET', 'POST'])
def _showStatus():
    def pywebio_show():
        # Inject JavaScript to reload the page when navigating back (otherwise table doesn't populate)
        run_js(""" window.onpageshow = function(event) { if (event.persisted) { window.location.reload() } }; """)

        # Clear the output
        clear()

        # Header
        showHeader()

        # Show the status
        put_markdown("# Server Status")
        showStatus()

    # Get the response from webio_view
    response = webio_view(pywebio_show)()

    # The code below forces the page to not cache, and to reload the page every time, so that the back button in the browser works correctly.
    # Check if the response is a Flask Response object
    if not isinstance(response, Response):
        response = make_response(response)

    # Set cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response



#
#   Web server main
#
@app.route('/', methods=['GET', 'POST'])
def pywebio_app():
    def pywebio_show():
        # Clear the output
        clear()

        # Header
        showHeader()

        put_markdown("# Main Menu")

        # Show the main dashboard functions
        put_table([
            #["", "Function"],
            ["", ""],
            [put_button("Theory Request (Manual)", onclick=lambda:run_js('window.location.href="/theoryrequestmanual"')), "Manually enter a theory request."],
            [put_button("Theory Request (Manual, Batch)", onclick=lambda:run_js('window.location.href="/theoryrequestmanualbatch"')), "Manually enter a batch of theory requests."],
            [put_button("List of Theories", onclick=lambda:run_js('window.location.href="/theorylist"')), "List all known theories."],
            [put_button("Export", onclick=lambda:run_js('window.location.href="/save"')), "Export the current Theorizer knowledge."],
            [put_button("Status", onclick=lambda:run_js('window.location.href="/status"')), "Show the system status."],
        ])

    # Show the app, no caching so the back button works
    # Get the response from webio_view
    response = webio_view(pywebio_show)()

    # The code below forces the page to not cache, and to reload the page every time, so that the back button in the browser works correctly.
    # Check if the response is a Flask Response object
    if not isinstance(response, Response):
        response = make_response(response)

    # Set cache control headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response



# Parse command line arguments
def parse_args():
    desc = "Launch a webserver to interact with CodeScientist from your browser."
    parser = argparse.ArgumentParser(desc)
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to use for the webserver.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind the webserver to. 0.0.0.0 for all interfaces.")
    return parser.parse_args()


#
#   Main entry point (web interface)
#

app.add_url_rule('/', 'webio_view', webio_view(pywebio_app), methods=['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
    args = parse_args()
    app.run(port=args.port, host=args.host, debug=False, use_reloader=True)
