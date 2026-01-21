# ExportTheoryStoreToHTML.py
# Takes a theory store as input, and exports it as a set of static HTML pages for offline viewing.
# Note: Much of this styling/HTML export is LLM-generated.

import os
import json
import markdown


PATH_THEORIES = "theories/"
PATH_PAPERS = "papers/"
PATH_SCHEMAS = "schemas/"
PATH_RESULTS = "results/"
PATH_EVALUATIONS = "evaluations/"
PATH_PREDACC = "predictiveaccuracy/"
PATH_QUALNOV = "qualifiednovelty/"

# Common CSS styles for all pages
COMMON_STYLES = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f5f5f5;
}

.header {
    display: flex;
    align-items: center;
    width: 100%;
    height: 50px;
    margin-bottom: 30px;
    background-color: white;
    padding: 10px 20px;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header a {
    text-decoration: none;
    color: #333;
    font-size: 24px;
    font-weight: bold;
}

.content {
    background-color: white;
    padding: 30px;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

h1 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    margin-top: 0;
}

h2 {
    color: #34495e;
    margin-top: 30px;
    margin-bottom: 15px;
    font-size: 1.5em;
}

h3 {
    color: #7f8c8d;
    margin-top: 20px;
    margin-bottom: 10px;
    font-size: 1.2em;
}

h3.extraction-instance {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 2px solid #e0e0e0;
}

h3.extraction-instance:first-of-type {
    margin-top: 20px;
    border-top: none;
    padding-top: 0;
}

.btn {
    display: inline-block;
    padding: 8px 16px;
    background-color: #007bff;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    font-size: 14px;
    transition: background-color 0.3s;
}

.btn:hover {
    background-color: #0056b3;
    text-decoration: none;
}

strong {
    color: #2c3e50;
}

p {
    margin: 10px 0;
}

ul, ol {
    margin: 10px 0;
    padding-left: 30px;
}

li {
    margin: 8px 0;
}

a {
    color: #3498db;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

.evidence-link {
    display: inline-block;
    margin: 0 2px;
    padding: 2px 6px;
    background-color: #e3f2fd;
    border-radius: 3px;
    font-size: 0.9em;
}

.info-section {
    background-color: #f8f9fa;
    padding: 15px;
    border-left: 4px solid #3498db;
    margin: 15px 0;
}

.empty-note {
    font-style: italic;
    color: #7f8c8d;
}

pre {
    background-color: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 15px;
    overflow-x: auto;
    font-size: 0.9em;
}

code {
    font-family: 'Courier New', monospace;
}

.section {
    margin-bottom: 30px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #ddd;
    vertical-align: top;
}

th {
    background-color: #f8f9fa;
    font-weight: bold;
    color: #2c3e50;
}

tr:hover {
    background-color: #f8f9fa;
}

.extraction-instance-container {
    margin-bottom: 30px;
    padding: 15px;
    border-radius: 5px;
    transition: all 0.3s ease;
}

.extraction-instance-container:target {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    box-shadow: 0 0 0 3px rgba(255, 193, 7, 0.2);
}

.extraction-query {
    background-color: #f0f8ff;
    padding: 10px;
    border-left: 4px solid #3498db;
    margin: 10px 0;
    font-style: italic;
}

.paper-content {
    line-height: 1.8;
    font-size: 0.95em;
}

.paper-content h1 {
    font-size: 1.8em;
    margin-top: 30px;
}

.paper-content h2 {
    font-size: 1.5em;
    margin-top: 25px;
}

.paper-content h3 {
    font-size: 1.3em;
    margin-top: 20px;
}

.paper-content p {
    margin: 15px 0;
}

.paper-content code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
}
"""

def create_css_file(path_out):
    """Create a separate CSS file"""
    css_path = os.path.join(path_out, "style.css")
    with open(css_path, 'w', encoding='utf-8') as f:
        f.write(COMMON_STYLES)
    print(f"Created CSS file: {css_path}")

def create_html_page(title, content, css_path="style.css", back_to_index="../index.html"):
    """Create a complete HTML page with common styling"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Theorizer</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
    <link rel="stylesheet" href="{css_path}">
</head>
<body>
    <div class="header">
        <a href="{back_to_index}"><i class="fas fa-flask"></i> Theorizer</a>
    </div>

    <div class="content">
{content}
    </div>
</body>
</html>"""


def format_evidence_list(evidence_list, results_path="../results/", uuid_to_extraction_id=None):
    """Format a list of evidence with UUID links"""
    if not evidence_list or len(evidence_list) == 0:
        return '<p class="empty-note">No evidence provided.</p>'

    html = "<ol>\n"
    for evidence in evidence_list:
        evidence_text = evidence.get('text', '')
        uuids = evidence.get('uuids', [])
        html += f"    <li>{evidence_text} "
        for uuid in uuids:
            # Map UUID to extraction ID for the filename
            extraction_id = uuid_to_extraction_id.get(uuid, uuid) if uuid_to_extraction_id else uuid
            html += f'<a href="{results_path}{extraction_id}.html#{uuid}" class="evidence-link">[{uuid}]</a> '
        html += "</li>\n"
    html += "</ol>"
    return html


def export_one_theory(theory, filename_out, uuid_to_extraction_id=None, predictive_accuracy_evaluations_by_statement_lut={}, qualified_novelty_evaluations_by_statement_lut={}):
    """Export a single theory to HTML"""
    theory_id = theory.get('id', 'Unknown')
    name = theory.get('name', 'Unknown')
    theory_type = theory.get('type', 'Unknown')
    theory_query = theory.get('theory_query', 'Not available.')
    description = theory.get('description', 'No description provided.')
    knowledge_cutoff_year = theory.get('knowledge_cutoff_year', 'Unknown')
    knowledge_cutoff_month = theory.get('knowledge_cutoff_month', 'Unknown')
    model_str = theory.get('model_str', 'Unknown')

    derived_from = theory.get('derived_from', [])
    change_log = theory.get('change_log', [])
    theory_evaluation_ids = theory.get('theory_evaluation_ids', [])

    components = theory.get('components', {})
    theory_statements = components.get('theory_statements', [])
    supporting_evidence = components.get('supporting_evidence', [])
    new_predictions_likely = components.get('new_predictions_likely', [])
    new_predictions_unknown = components.get('new_predictions_unknown', [])
    negative_experiments = components.get('negative_experiments', [])
    unaccounted_for_evidence = components.get('unaccounted_for', [])
    existing_theory = components.get('existing_theory', {})

    content = f"""        <h1>Theory Details for {theory_id}</h1>

        <div class="section">
            <h2>Theory (General Information)</h2>
            <div class="info-section">
                <p><strong>ID:</strong> {theory_id}</p>
                <p><strong>Name:</strong> {name}</p>
                <p><strong>Type:</strong> {theory_type}</p>
                <p><strong>Theory Query:</strong> {theory_query}</p>
                <p><strong>Description:</strong> {description}</p>
                <p><strong>Knowledge Cutoff Year:</strong> {knowledge_cutoff_year}</p>
                <p><strong>Knowledge Cutoff Month:</strong> {knowledge_cutoff_month}</p>
                <p><strong>Base Model:</strong> {model_str}</p>
            </div>
        </div>
"""

    content += """        </div>

        <div class="section">
            <h2>Theory (Statement/Laws)</h2>

            <h3>Theory Statements/Laws</h3>
"""
    # Add a horizontal rule
    content += "            <hr/>\n"

    if (theory_statements) and (len(theory_statements) > 0):

        for idx, law in enumerate(theory_statements):

            statement = law.get('theory_statement', None)
            law_name = law.get('statement_name', f'Law {idx}')
            domain_scope = law.get('domain_scope', None)
            special_cases = law.get('special_cases', [])
            supporting_evidence_law = law.get('supporting_evidence', [])
            qual_or_quant = law.get('qual_or_quant', None)
            existing_law = law.get('novelty_evaluation', {})

            content += f'            <h3>Statement {idx}: {law_name}</h3>\n'
            content += f'            <p><strong>Statement:</strong> {statement}</p>\n'
            content += f'            <p><strong>Domain/Scope:</strong> {domain_scope if domain_scope else "<span class=\"empty-note\">Not specified.</span>"}</p>\n'

            # List of special cases (numbered list, each is a string).
            content += "            <h4>Special Cases</h4>\n"
            if (special_cases) and (len(special_cases) > 0):
                content += "            <ol>\n"
                for case in special_cases:
                    content += f"                <li>{case}</li>\n"
                content += "            </ol>\n"

            content += "            <h4>Supporting Evidence for this Law</h4>\n"
            content += format_evidence_list(supporting_evidence_law, f"../{PATH_RESULTS}", uuid_to_extraction_id)

            # Existing law comparison (i.e. the model that generated the theory's self evaluation of novelty for this law)
            content += "            <h4>Self-Evaluation of Law Novelty (produced by the generation model)</h4>\n"
            if (existing_law != None):

                likely_classification = existing_law.get('likely_classification', 'unknown')
                # Change the color of the likely classification based on its value
                likely_classification_str = likely_classification
                if (likely_classification.lower() == 'new'):
                    likely_classification_str = f'<span style="color: green; font-weight: bold;">{likely_classification}</span>'
                elif (likely_classification.lower() == 'somewhat-related-to-existing'):
                    likely_classification_str = f'<span style="color: orange; font-weight: bold;">{likely_classification}</span>'

                classification_explanation = existing_law.get('classification_explanation', 'No explanation provided.')
                what_already_exists = existing_law.get('what_already_exists', None)
                what_is_novel = existing_law.get('what_is_novel', None)

                references = existing_law.get('references', [])
                content += f"""            <p><strong>Likely Classification:</strong> {likely_classification_str}</p>
        <p><strong>Explanation:</strong> {classification_explanation}</p>"""
                if (what_already_exists != None):
                    content += f"""            <p><strong>What Already Exists:</strong> {what_already_exists}</p>"""
                if (what_is_novel != None):
                    content += f"""            <p><strong>What is Novel:</strong> {what_is_novel}</p>"""
                content += f"""
        <p><strong>References:</strong> """
                if references and len(references) > 0:
                    content += "<ul>\n"
                    for ref in references:
                        content += f"    <li>{ref}</li>\n"
                    content += "</ul>\n"
                else:
                    content += '<span class="empty-note">No references provided.</span>'

            else:
                content += '            <p class="empty-note">No existing law comparison provided.</p>\n'

            content += "            <h4>External Evaluations of this Law</h4>\n"
            # Predictive Accuracy Evaluation, if available.
            predictive_accuracy_link = None
            # Check in the LUT to see if there are any predictive accuracy evaluations for this statement/law
            if (statement in predictive_accuracy_evaluations_by_statement_lut):
                predictive_accuracy_link = predictive_accuracy_evaluations_by_statement_lut[statement]

            if predictive_accuracy_link:
                short_name = os.path.basename(predictive_accuracy_link).replace('.html', '')
                content += f'            <p><strong>Predictive Accuracy Evaluation:</strong> <a href="../{predictive_accuracy_link}" class="evidence-link">[predictive accuracy evaluation ' + str(short_name) + ']</a></p>\n'
            else:
                content += f'            <p><strong>Predictive Accuracy Evaluation:</strong> <span class="empty-note">Not available.</span></p>\n'

            # Qualified Novelty evaluation, if available.
            novelty_evaluation_link = None
            # Check in the LUT to see if there are any novelty evaluations for this statement/law
            if (statement in qualified_novelty_evaluations_by_statement_lut):
                novelty_evaluation_link = qualified_novelty_evaluations_by_statement_lut[statement]
            if novelty_evaluation_link:
                short_name = os.path.basename(novelty_evaluation_link).replace('.html', '')
                content += f'            <p><strong>Novelty Evaluation:</strong> <a href="../{novelty_evaluation_link}" class="evidence-link">[qualified novelty evaluation ' + str(short_name) + ']</a></p>\n'
            else:
                content += f'            <p><strong>Novelty Evaluation:</strong> <span class="empty-note">Not available (available only for a randomly selected subset of 100 laws, due to cost).</span></p>\n'

            # Add a horizontal rule between laws
            content += "            <hr/>\n"
    else:
        content += '            <p class="empty-note">No theory statements provided.</p>\n'

    content += """        </div>
        <div class="section">
            <h2>Theory (Additional Details)</h2>
            """


    content += "            <h3>New Predictions (Likely outcome)</h3>\n"
    if new_predictions_likely and len(new_predictions_likely) > 0:
        content += "            <ol>\n"
        for pred in new_predictions_likely:
            content += f"                <li>{pred}</li>\n"
        content += "            </ol>\n"
    else:
        content += '            <p class="empty-note">No new predictions (likely) provided.</p>\n'

    content += "            <h3>New Predictions (Unknown outcome/high-entropy)</h3>\n"
    if new_predictions_unknown and len(new_predictions_unknown) > 0:
        content += "            <ol>\n"
        for pred in new_predictions_unknown:
            content += f"                <li>{pred}</li>\n"
        content += "            </ol>\n"
    else:
        content += '            <p class="empty-note">No new predictions (unknown) provided.</p>\n'

    content += "            <h3>Negative Experiments</h3>\n"
    if negative_experiments and len(negative_experiments) > 0:
        content += "            <ol>\n"
        for exp in negative_experiments:
            content += f"                <li>{exp}</li>\n"
        content += "            </ol>\n"
    else:
        content += '            <p class="empty-note">No negative experiments provided.</p>\n'

    content += "            <h3>Unaccounted for Evidence</h3>\n"
    content += format_evidence_list(unaccounted_for_evidence, f"../{PATH_RESULTS}", uuid_to_extraction_id)
    content += """        </div>"""

    # Add a little buffer space at the bottom
    content += """        <div style="height: 30px;"></div>"""

    html = create_html_page(name, content, css_path="../style.css")
    with open(filename_out, 'w', encoding='utf-8') as f:
        f.write(html)


def export_one_extraction_schema(schema, filename_out):
    """Export a single extraction schema to HTML"""
    schema_id = schema.get('id', 'Unknown')
    extraction_query = schema.get('extraction_query', 'No query provided.')
    schema_fields = schema.get('schema', [])

    content = f"""        <h1>Extraction Schema Details for {schema_id}</h1>

        <div class="section">
            <h2>Extraction Schema (General Information)</h2>
            <div class="info-section">
                <p><strong>Schema ID:</strong> {schema_id}</p>
                <p><strong>Extraction Query:</strong> {extraction_query}</p>
            </div>
        </div>

        <div class="section">
            <h2>Extraction Schema (Details)</h2>
"""

    if schema_fields and len(schema_fields) > 0:
        content += """            <table>
                <thead>
                    <tr>
                        <th style="width: 20%;">Field Name</th>
                        <th style="width: 15%;">Type</th>
                        <th style="width: 65%;">Description</th>
                    </tr>
                </thead>
                <tbody>
"""
        for field in schema_fields:
            field_name = field.get('name', 'None')
            field_type = field.get('type', 'None')
            field_description = field.get('description', 'No description provided.')
            content += f"""                    <tr>
                        <td><strong>{field_name}</strong></td>
                        <td>{field_type}</td>
                        <td>{field_description}</td>
                    </tr>
"""
        content += """                </tbody>
            </table>
"""
    else:
        content += '            <p class="empty-note">No fields in this extraction schema.</p>\n'

    content += """        </div>
"""

    html = create_html_page(f"Extraction Schema {schema_id}", content, css_path="../style.css")
    with open(filename_out, 'w', encoding='utf-8') as f:
        f.write(html)


# Look-up-table for pre-computed paper markdown
paper_markdown_cache = {}



def export_one_extraction_result(extraction, filename_out, paperstore=None, paperstore_lut=None, extraction_schemas=None):
    """Export a single extraction result to HTML (all instances in one file)"""
    global paper_markdown_cache

    extraction_id = extraction.get("id", "Unknown")
    extraction_schema_id = extraction.get("extraction_schema_id", None)
    paper_id = extraction.get("paper_id", None)
    cost = extraction.get("cost", None)
    extracted_data = extraction.get("extracted_data", [])
    potentially_relevant_new_papers = extraction.get("potentially_relevant_new_papers", [])

    # Get extraction query from schema
    extraction_query = None
    if extraction_schemas and extraction_schema_id and extraction_schema_id in extraction_schemas:
        extraction_query = extraction_schemas[extraction_schema_id].get('extraction_query', None)

    # Try to get paper data
    paper_record = {}
    if (paperstore_lut != None) and (paper_id != None) and (paper_id in paperstore_lut):
        canonical_paper_id = paperstore_lut[paper_id]
        if (paperstore != None) and (canonical_paper_id in paperstore):
            paper_record = paperstore[canonical_paper_id]

    paper_title = paper_record.get('title', None)
    s2_metadata = paper_record.get('s2_metadata', {})
    paper_venue = s2_metadata.get('venue', None)
    paper_year = paper_record.get('year', None)

    paper_url = s2_metadata.get('url', None)
    paper_urls = paper_record.get("paper_urls", [])
    if (paper_url == None) and (paper_urls and len(paper_urls) > 0):
        paper_url = paper_urls[0]
    paper_tldr1 = s2_metadata.get('tldr', None)
    paper_tldr = None
    if paper_tldr1 and isinstance(paper_tldr1, dict):
        paper_tldr = paper_tldr1.get('text', None)

    paper_abstract = s2_metadata.get("abstract", None)
    if (paper_abstract == None):
        more_metadata = s2_metadata.get("metadata", {})
        paper_abstract = more_metadata.get("abstract", None)

    paper_markdown = paper_record.get('paper_markdown', None)

    content = f"""        <h1>Extracted Data Details for {extraction_id}</h1>

        <div class="section">
            <h2>Extracted Data (Header)</h2>
            <div class="info-section">
                <p><strong>Extraction ID:</strong> {extraction_id}</p>
"""

    if extraction_schema_id:
        content += f'                <p><strong>Extraction Schema Used (ID):</strong> <a href="../{PATH_SCHEMAS}{extraction_schema_id}.html">{extraction_schema_id}</a></p>\n'

    if extraction_query:
        content += f'                <div class="extraction-query"><strong>Extraction Query:</strong> {extraction_query}</div>\n'

    content += f"                <p><strong>Paper ID:</strong> {paper_id}</p>\n"

    if paper_title:
        if paper_url:
            content += f'                <p><strong>Paper Title:</strong> <a href="{paper_url}" target="_blank">{paper_title}</a></p>\n'
        else:
            content += f"                <p><strong>Paper Title:</strong> {paper_title}</p>\n"
    if paper_year:
        content += f"                <p><strong>Paper Year:</strong> {paper_year}</p>\n"
    if paper_venue:
        content += f"                <p><strong>Paper Venue:</strong> {paper_venue}</p>\n"
    if paper_tldr:
        content += f"                <p><strong>Paper TL;DR:</strong> {paper_tldr}</p>\n"
    if paper_abstract:
        content += f"                <p><strong>Paper Abstract:</strong> {paper_abstract}</p>\n"

    cost_rounded = round(cost, 3) if cost else "--"
    content += f"""                <p><strong>Cost:</strong> {cost_rounded}</p>
            </div>
        </div>

        <div class="section">
            <h2>Extracted Data (Details)</h2>
"""

    if extracted_data and len(extracted_data) > 0:
        for idx, data in enumerate(extracted_data):
            data_uuid = data.get('uuid', 'Unknown')

            # Wrap each instance in a container div with ID for targeting
            content += f'            <div class="extraction-instance-container" id="{data_uuid}">\n'
            content += f'                <h3 class="extraction-instance">Extracted Data Instance {idx} ({data_uuid})</h3>\n'
            content += f'                <div class="extraction-query"><strong>Extraction Query:</strong> {extraction_query}</div>\n'

            content += """                <table>
                    <thead>
                        <tr>
                            <th style="width: 30%;">Field</th>
                            <th style="width: 70%;">Value</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            for key, value in data.items():
                if key == 'uuid':
                    continue
                # Format the value properly
                if isinstance(value, bool):
                    value_str = str(value)
                elif value is None:
                    value_str = '<span class="empty-note">null</span>'
                else:
                    value_str = str(value)
                content += f"""                        <tr>
                            <td><strong>{key}</strong></td>
                            <td>{value_str}</td>
                        </tr>
"""
            content += """                    </tbody>
                </table>
            </div>
"""
    else:
        content += '            <p class="empty-note">No extracted data.</p>\n'

    content += """        </div>

        <div class="section">
            <h2>Potentially Relevant New Papers (mentioned by this paper)</h2>
"""

    if potentially_relevant_new_papers and len(potentially_relevant_new_papers) > 0:
        content += "            <ol>\n"
        for paper in potentially_relevant_new_papers:
            paper_title_ref = paper.get('paper_title', 'Unknown')
            rating = paper.get('rating', 'N/A')
            content += f"                <li>{paper_title_ref} <em>(Rating: {rating})</em></li>\n"
        content += "            </ol>\n"
    else:
        content += '            <p class="empty-note">No potentially relevant new papers extracted.</p>\n'

    content += """        </div>
"""

    if paper_markdown:
        # Check the cache
        converted_markdown = paper_markdown_cache.get(paper_id, None)
        if not converted_markdown:
            # Convert markdown to HTML
            try:
                converted_markdown = markdown.markdown(paper_markdown, extensions=['extra', 'codehilite'])
                paper_markdown_cache[paper_id] = converted_markdown
            except Exception as e:
                converted_markdown = f"<p class='empty-note'>Error converting markdown to HTML: {str(e)}</p>"

        content += """
        </div>
"""

    html = create_html_page(f"Extraction {extraction_id}", content, css_path="../style.css")
    with open(filename_out, 'w', encoding='utf-8') as f:
        f.write(html)


def export_one_theory_evaluation(evaluation, filename_out, uuid_to_extraction_id=None):
    """Export a single theory evaluation to HTML"""
    eval_id = evaluation.get('id', 'Unknown')
    theory_id = evaluation.get('theory_id', None)
    revised_theory_ids = evaluation.get('revised_theory_ids', [])
    overall_support = evaluation.get('overall_support_or_contradict', None)
    overall_explanation = evaluation.get('overall_support_or_contradict_explanation', None)

    fully_supporting = evaluation.get('fully_supporting_evidence', [])
    partially_supporting = evaluation.get('partially_supporting_evidence', [])
    fully_contradicting = evaluation.get('fully_contradicting_evidence', [])
    partially_contradicting = evaluation.get('partially_contradicting_evidence', [])
    potentially_modifying = evaluation.get('potentially_modifying_evidence', [])
    suggested_revisions = evaluation.get('suggested_revisions', [])

    content = f"""        <h1>Theory Evaluation Details for {eval_id}</h1>

        <div class="section">
            <h2>Theory Evaluation (General Information)</h2>
            <div class="info-section">
                <p><strong>Evaluation ID:</strong> {eval_id}</p>
"""

    if theory_id:
        content += f'                <p><strong>This evaluation is for theory ID:</strong> <a href="../{PATH_THEORIES}{theory_id}.html">{theory_id}</a></p>\n'

    if revised_theory_ids and len(revised_theory_ids) > 0:
        revised_links = [f'<a href="../{PATH_THEORIES}{rid}.html">{rid}</a>' for rid in revised_theory_ids]
        content += f"                <p><strong>This evaluation resulted in these revised theories:</strong> {' '.join(revised_links)}</p>\n"
    else:
        content += '                <p><strong>This evaluation resulted in these revised theories:</strong> <span class="empty-note">None</span></p>\n'

    content += f"""                <p><strong>Overall Support or Contradict:</strong> {overall_support}</p>
                <p><strong>Overall Support or Contradict Explanation:</strong> {overall_explanation}</p>
            </div>
        </div>

        <div class="section">
            <h2>Theory Evaluation (Components)</h2>

            <h3>Fully Supporting Evidence</h3>
"""
    content += format_evidence_list(fully_supporting, f"../{PATH_RESULTS}", uuid_to_extraction_id)

    content += "            <h3>Partially Supporting Evidence</h3>\n"
    content += format_evidence_list(partially_supporting, f"../{PATH_RESULTS}", uuid_to_extraction_id)

    content += "            <h3>Fully Contradicting Evidence</h3>\n"
    content += format_evidence_list(fully_contradicting, f"../{PATH_RESULTS}", uuid_to_extraction_id)

    content += "            <h3>Partially Contradicting Evidence</h3>\n"
    content += format_evidence_list(partially_contradicting, f"../{PATH_RESULTS}", uuid_to_extraction_id)

    content += "            <h3>Potentially Modifying Evidence</h3>\n"
    content += format_evidence_list(potentially_modifying, f"../{PATH_RESULTS}", uuid_to_extraction_id)

    content += "            <h3>Suggested Revisions</h3>\n"
    if suggested_revisions and len(suggested_revisions) > 0:
        content += "            <ol>\n"
        for rev in suggested_revisions:
            content += f"                <li>{rev}</li>\n"
        content += "            </ol>\n"
    else:
        content += '            <p class="empty-note">No suggested revisions provided.</p>\n'

    content += """        </div>

        <div class="section">
            <h2>Theory Evaluation (Debug)</h2>
            <pre><code>"""
    content += json.dumps(evaluation, indent=4, ensure_ascii=False).replace('<', '&lt;').replace('>', '&gt;')
    content += """</code></pre>
        </div>"""

    html = create_html_page(f"Theory Evaluation {eval_id}", content, css_path="../style.css")
    with open(filename_out, 'w', encoding='utf-8') as f:
        f.write(html)


def create_index_page(theories, path_out):
    """Create an index page listing all theories"""
    theory_list = []
    for theory_id, theory in theories.items():
        try:
            id_numerical = int(theory_id.split('-')[1])
            theory['id_numerical'] = id_numerical
        except:
            theory['id_numerical'] = -1
        theory_list.append(theory)

    theory_list.sort(key=lambda x: x['id_numerical'], reverse=True)

    content = """        <h1>List of Theories</h1>

        <table>
            <thead>
                <tr>
                    <th style="width: 15%;">Theory ID</th>
                    <th style="width: 70%;">Theory Description</th>
                    <th style="width: 15%;">Actions</th>
                </tr>
            </thead>
            <tbody>
"""

    last_theory_query = None
    for theory in theory_list:
        theory_id = theory.get('id', 'Unknown')
        name = theory.get('name', 'Unknown')
        theory_query = theory.get('theory_query', 'Not available.')
        if (len(theory_query.strip()) == 0):
            theory_query = 'Not available.'
        description = theory.get('description', 'No description')
        derived_from = theory.get('derived_from', [])

        desc_html = f"<b>Name:</b> {name}<br><b>Description:</b> {description}"
        if derived_from and len(derived_from) > 0:
            derived_str = ", ".join([f"[{d}]" for d in derived_from])
            derived_links = ", ".join([f'<a href="{PATH_THEORIES}{d}.html">[{d}]</a>' for d in derived_from])
            desc_html += f"<br><b>Derived From:</b> {derived_links}"

        model = theory.get("model_str", None)
        generation_method = theory.get('components', {}).get('generation_mode', None)
        original_theory_id = theory.get('components', {}).get('original_theory_id', None)
        version = theory.get('components', {}).get('version', None)
        # Also get it's `likely_classification` from `existing_theory`
        likely_classification = theory.get('components', {}).get('existing_theory', {}).get('likely_classification', None)

        if (likely_classification != None):
            # If the likely classification is `new`, then make it green; otherwise, do not color it.
            if (likely_classification.lower() == 'new'):
                desc_html += f"<br><b>Self-classification:</b> <span style='color: green; font-weight: bold;'>{likely_classification}</span>"
            elif (likely_classification.lower() == 'somewhat-related-to-existing'):
                desc_html += f"<br><b>Self-classification:</b> <span style='color: orange; font-weight: bold;'>{likely_classification}</span>"
            else:
                desc_html += f"<br><b>Self-classification:</b> {likely_classification}"
        if (generation_method != None):
            desc_html += f"<br><b>Generation Method:</b> {generation_method}"
        if (version != None):
            desc_html += f"<br><b>Version:</b> {version}"
        if (model != None):
            desc_html += f"<br><b>Model:</b> {model}"
        if (original_theory_id != None):
            desc_html += f'<br><b>Original Theory ID:</b> <a href="{PATH_THEORIES}{original_theory_id}.html">{original_theory_id}</a>'



        if (theory_query != last_theory_query):
            content += f"""                <tr>
                    <td colspan="3" style="background-color: #f0f8ff;"><strong>Theory Query:</strong> {theory_query}</td>
                </tr>
"""
            last_theory_query = theory_query

        content += f"""                <tr>
                    <td>{theory_id}</td>
                    <td>{desc_html}</td>
                    <td><a href="{PATH_THEORIES}{theory_id}.html" class="btn">Details</a></td>
                </tr>
"""

    content += """            </tbody>
        </table>"""

    html = create_html_page("Theory List", content, css_path="style.css", back_to_index="index.html")

    with open(os.path.join(path_out, "index.html"), 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Created index page: {os.path.join(path_out, 'index.html')}")


def build_extraction_maps(extraction_results):
    """Build mappings for UUIDs to extraction results and IDs"""
    uuid_to_extraction_id = {}
    for result_id, result in extraction_results.items():
        extracted_data = result.get("extracted_data", [])
        for data in extracted_data:
            uuid = data.get('uuid')
            if uuid:
                uuid_to_extraction_id[uuid] = result_id
    return uuid_to_extraction_id


def export_predictive_accuracy_evaluations(path_in:str, path_out:str):
    # Make the output folder
    if (not os.path.exists(path_out)):
        os.makedirs(path_out)

    # Load all predictive accuracy evaluations from the given directory
    predictive_accuracy_evaluations_by_statement = {}

    pe_idx = 0

    # Load all JSON files in the given directory
    for filename in os.listdir(path_in):
        if filename.endswith(".json"):
            filepath = os.path.join(path_in, filename)
            data = None
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            num_papers_evaluated = data.get('num_papers_evaluated', 0)
            rubric_elems_with_some_evidence = data.get('rubric_elems_with_some_evidence', 0)

            rubric = data.get('rubric', None)
            rubric_evaluation = data.get('rubric_evaluation', None)
            theory_statement_law = data.get('theory_statement_law', None)
            if (theory_statement_law == None):
                print(f"Warning: No theory_statement_law found in file {filename}, skipping.")
                continue
            theory_id = theory_statement_law.get("theory_id", None)
            statement = theory_statement_law.get("law", {}).get("theory_statement", None)

            individual_paper_evaluations = data.get('individual_paper_evaluations', [])

            content = ""
            # Header
            content += f"        <h1>Predictive Accuracy Evaluation for Theory {theory_id}</h1>\n"
            content += '        <div class="section">\n'
            content += '            <h2>Theory Statement Being Evaluated</h2>\n'
            content += f'            <p><strong>Theory ID:</strong> {theory_id}</p>\n'
            content += f'            <p><strong>Theory Statement:</strong> {statement}</p>\n'
            content += f'            <p><strong>Number of Papers Evaluated:</strong> {num_papers_evaluated}</p>\n'
            content += f'            <p><strong>Number of Rubric Elements with Some Evidence:</strong> {rubric_elems_with_some_evidence}</p>\n'
            content += '        </div>\n'

            # Horizontal rule
            content += '        <hr>\n'
            # Rubric Summary
            content += '        <div class="section">\n'
            content += '            <h2>Predictions / Rubric Summary</h2>\n'

            if (rubric_evaluation) and (len(rubric_evaluation) > 0):
                for idx, elem in enumerate(rubric_evaluation):
                    prediction_short_name = elem.get('prediction_short_name', 'Unknown')
                    rubric_elem = elem.get('rubric_elem', {})
                    specific_prediction = rubric_elem.get('specific_prediction', 'No prediction provided.')
                    counts = elem.get('counts', {})
                    proportions = elem.get('proportions', {})

                    content += f'            <h3>Prediction ' + str(idx+1) + f': {prediction_short_name}</h3>\n'
                    content += f'            <p><strong>Specific Prediction:</strong> {specific_prediction}</p>\n'
                    content += '            <table>\n'
                    content += '                <thead>\n'
                    content += '                    <tr>\n'
                    content += '                        <th style="width: 33%;">Evaluation</th>\n'
                    content += '                        <th style="width: 22%;">Support</th>\n'
                    content += '                        <th style="width: 22%;">Contradict</th>\n'
                    content += '                        <th style="width: 22%;">No Evidence</th>\n'
                    content += '                    </tr>\n'
                    content += '                </thead>\n'
                    content += '                <tbody>\n'
                    content += '                    <tr>\n'
                    content += '                        <td><strong>Counts (# of papers)</strong></td>\n'
                    content += f'                        <td>{counts.get("support", 0)}</td>\n'
                    content += f'                        <td>{counts.get("contradict", 0)}</td>\n'
                    content += f'                        <td>{counts.get("no_evidence", 0)}</td>\n'
                    content += '                    </tr>\n'
                    content += '                    <tr>\n'
                    content += '                        <td><strong>Proportions (%)</strong></td>\n'
                    content += f'                        <td>{proportions.get("support", 0)}</td>\n'
                    content += f'                        <td>{proportions.get("contradict", 0)}</td>\n'
                    content += f'                        <td>--</td>\n'
                    content += '                    </tr>\n'
                    content += '                </tbody>\n'
                    content += '            </table>\n'

            else:
                content += '            <p class="empty-note">No rubric evaluation data available.</p>\n'

            # Add a bit of blank space, to separate from next prediction
            content += '            <div style="height: 40px;"></div>\n'

            content += '        </div>\n'

            # Individual Paper Evaluations
            content += '        <div class="section">\n'
            content += '            <h2>Individual Paper Evaluations</h2>\n'
            content += '            <hr>\n'

            paper_evals_with_evidence = []
            paper_evals_without_evidence = []

            if (individual_paper_evaluations) and (len(individual_paper_evaluations) > 0):
                for paper_eval in individual_paper_evaluations:

                    original_paper_dict = paper_eval.get('original_paper_dict', {})
                    title = original_paper_dict.get('title', 'Unknown Title')
                    publication_year = original_paper_dict.get('publication_year', 'Unknown Year')
                    publication_month = original_paper_dict.get('publication_month', 'Unknown Month')
                    s2_metadata = original_paper_dict.get('s2_metadata', {})
                    metadata = s2_metadata.get('metadata', {})
                    authors = metadata.get('authors', [])
                    external_ids = original_paper_dict.get('external_ids', {})
                    arxiv_id = external_ids.get('arxiv_id', None)

                    response = paper_eval.get('response', {})
                    if (not response) or (not isinstance(response, dict)):
                        continue
                    predictive_evaluation = response.get('predictive_evaluation', [])
                    elements_with_evidence = []
                    for elem in predictive_evaluation:
                        overall_evaluation = elem.get('overall_evaluation', "")
                        if (overall_evaluation in ["support", "contradict"]):
                            elements_with_evidence.append(elem)
                    content_paper = ""
                    content_paper += f'            <h3>Paper: {title} ({publication_year}-{publication_month})</h3>\n'
                    if authors and len(authors) > 0:
                        authors_str = []
                        for author in authors:
                            print(json.dumps(author, indent=4))
                            # first, middle, last
                            first = author.get('first', '')
                            middle = author.get('middle', '')
                            if (isinstance(middle, list)):
                                middle = " ".join(middle)
                            last = author.get('last', '')
                            full_name = str(first) + " " + str(middle) + " " + str(last)
                            # Replace multiple spaces with single space, and strip leading/trailing spaces
                            import re
                            full_name = re.sub(' +', ' ', full_name).strip()
                            authors_str.append(full_name)

                        authors_str = ", ".join(authors_str)
                        content_paper += f'            <p><strong>Authors:</strong> {authors_str}</p>\n'
                    if arxiv_id:
                        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
                        content_paper += f'            <p><strong>ArXiv ID:</strong> <a href="{arxiv_url}" target="_blank">{arxiv_id}</a></p>\n'

                    if (len(elements_with_evidence) == 0):
                        content_paper += '            <p class="empty-note">Paper did not contain any evidence rated as strongly testing any specific predictions in the rubric.</p>\n'
                    else:
                        content_paper += '            <ul>\n'
                        for elem in elements_with_evidence:

                            prediction_short_name = elem.get('prediction_short_name', 'Unknown')
                            overall_evaluation = elem.get('overall_evaluation', 'no_evidence')
                            evidence_quote_or_locator = elem.get('evidence_quote_or_locator', 'No locator provided.')
                            evidence_in_support = elem.get('evidence_in_support', 'No supporting evidence provided.')
                            evidence_in_contradiction = elem.get('evidence_in_contradiction', 'No contradicting evidence provided.')

                            content_paper += f'                <li>\n'
                            content_paper += f'                    <p><strong>Prediction:</strong> {prediction_short_name}</p>\n'
                            content_paper += f'                    <p><strong>Overall Evaluation:</strong> {overall_evaluation}</p>\n'
                            content_paper += f'                    <p><strong>Evidence Locator:</strong> {evidence_quote_or_locator}</p>\n'
                            content_paper += f'                    <p><strong>Evidence in Support:</strong> {evidence_in_support}</p>\n'
                            content_paper += f'                    <p><strong>Evidence in Contradiction:</strong> {evidence_in_contradiction}</p>\n'
                            content_paper += f'                </li>\n'
                        content_paper += '            </ul>\n'
                    content_paper += '            <hr>\n'

                    if (len(elements_with_evidence) > 0):
                        paper_evals_with_evidence.append(content_paper)
                    else:
                        paper_evals_without_evidence.append(content_paper)
            else:
                content += '            <p class="empty-note">No individual paper evaluations available.</p>\n'


            paper_count_idx = 1

            if (len(paper_evals_with_evidence) > 0):
                for pe in paper_evals_with_evidence:
                    pe = pe.replace("Paper:", f"Paper {paper_count_idx}:", 1)
                    content += pe
                    paper_count_idx += 1
            else:
                pass

            if (len(paper_evals_without_evidence) > 0):
                for pe in paper_evals_without_evidence:
                    pe = pe.replace("Paper:", f"Paper {paper_count_idx}:", 1)
                    content += pe
                    paper_count_idx += 1
            else:
                pass

            content += '        </div>\n'


            # Write to HTML file
            filename_out = "pe-" + str(pe_idx).zfill(5) + f"-{theory_id}.html"

            html = create_html_page(f"Predictive Accuracy Evaluation for Theory {theory_id}", content, css_path="../style.css")
            with open(os.path.join(path_out, filename_out), 'w', encoding='utf-8') as f:
                f.write(html)

            pe_idx += 1

            predictive_accuracy_evaluations_by_statement[statement] = os.path.join(PATH_PREDACC, filename_out)


    return predictive_accuracy_evaluations_by_statement



def export_qualified_novelty_evaluations(path_in:str, path_out:str):
    # Make the output folder
    if (not os.path.exists(path_out)):
        os.makedirs(path_out)

    # Load all predictive accuracy evaluations from the given directory
    qualified_novelty_evaluations_by_statement = {}

    eval_idx = 0

    # Load all JSON files in the given directory
    for filename in os.listdir(path_in):
        if filename.endswith(".json"):
            filepath = os.path.join(path_in, filename)
            data = None
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            theory_id = data.get('theory_id', None)
            theory_statement_law = data.get('theory_statement_law', {})
            theory_statement = theory_statement_law.get("theory_statement", "Unknown Statement")
            domain_scope = theory_statement_law.get("domain_scope", "Unknown Domain/Scope")
            special_cases = theory_statement_law.get("special_cases", [])

            num_papers_evaluated = data.get('num_papers_evaluated', 0)

            final_aggregated_novelty_evaluation = data.get('final_aggregated_novelty_evaluation', {})
            all_novelty_evaluations = data.get('all_novelty_evaluations', [])
            all_novelty_evaluations_best_responses = []
            for nov_eval in all_novelty_evaluations:
                novelty_evaluation = nov_eval.get('novelty_evaluation', {})
                if (novelty_evaluation == None):
                    continue
                best_response = novelty_evaluation.get('best_response', {})
                if (best_response == None):
                    continue
                novelty_evaluation = best_response.get('novelty_evaluation', {})
                if (novelty_evaluation == None):
                    continue
                all_novelty_evaluations_best_responses.append(novelty_evaluation)

            content = ""
            content += f"        <h1>Qualified Novelty Evaluation</h1>\n"
            content += '        <div class="section">\n'
            content += '            <h2>Theory Statement Being Evaluated</h2>\n'
            content += f'            <p><strong>Theory ID:</strong> {theory_id}</p>\n'
            content += f'            <p><strong>Theory Statement:</strong> {theory_statement}</p>\n'
            content += f'            <p><strong>Domain/Scope:</strong> {domain_scope}</p>\n'
            if (special_cases) and (len(special_cases) > 0):
                content += '            <p><strong>Special Cases:</strong></p>\n'
                content += '            <ul>\n'
                for sc in special_cases:
                    content += f'                <li>{sc}</li>\n'
                content += '            </ul>\n'
            else:
                content += '            <p><strong>Special Cases:</strong> <span class="empty-note">None</span></p>\n'
            content += f'            <p><strong>Number of Papers Evaluated:</strong> {num_papers_evaluated}</p>\n'
            content += '        </div>\n'

            content += '        <hr>\n'

            # Overall Aggregated Qualified Novelty Evaluation
            content += '        <div class="section">\n'
            content += '            <h2>Overall Aggregated Qualified Novelty Evaluation</h2>\n'
            content += '            <hr>\n'

            dimensions = final_aggregated_novelty_evaluation.keys()
            print(json.dumps(final_aggregated_novelty_evaluation, indent=4))
            print(f"Dimensions: {dimensions}")
            print("\n\n\n")
            if (final_aggregated_novelty_evaluation == None):
                continue

            for dim in dimensions:
                if (dim == None):
                    continue
                dim_eval = final_aggregated_novelty_evaluation.get(dim, {})
                if (dim_eval == None):
                    continue
                what_is_known = dim_eval.get('what_is_known', 'Not provided.')
                what_introduced = dim_eval.get('what_introduced', 'Not provided.')
                what_novel = dim_eval.get('what_novel', 'Not provided.')
                degree_of_novelty = dim_eval.get('degree_of_novelty', 'Not provided.')

                content += f'            <h3>Dimension: {dim}</h3>\n'
                content += f'            <p><strong>What is Known:</strong> {what_is_known}</p>\n'
                content += f'            <p><strong>What is Introduced by the Theory:</strong> {what_introduced}</p>\n'
                content += f'            <p><strong>What is Novel:</strong> {what_novel}</p>\n'
                content += f'            <p><strong>Degree of Novelty:</strong> {degree_of_novelty}</p>\n'
                content += '            <hr>\n'

            content += '        </div>\n'

            content += '        <div class="section">\n'
            content += '            <h2>Raw Novelty Evaluations (per paper)</h2>\n'
            content += '            <hr>\n'

            # Sort the novelty evaluations by those with/without content
            novelty_evals_with_content = []
            novelty_evals_without_content = []
            for nov_eval in all_novelty_evaluations_best_responses:
                has_content = False
                for key in nov_eval.keys():
                    val = nov_eval.get(key, "")
                    if (isinstance(val, str)) and (len(val.strip()) > 0):
                        if (key != 'paper_title'):
                            has_content = True
                if (has_content):
                    novelty_evals_with_content.append(nov_eval)
                else:
                    novelty_evals_without_content.append(nov_eval)

            papers_in_order_of_content = novelty_evals_with_content + novelty_evals_without_content
            for idx, nov_eval in enumerate(papers_in_order_of_content):
                content += f'            <h3>Paper {idx+1}: {nov_eval.get("paper_title", "Unknown Title")}</h3>\n'
                dimensions = ['phenomenon_effect', 'explanatory_mechanistic', 'unification', 'generalization_scope_expansion', 'constraint_limitation', 'conceptual_reframing_abstraction', 'empirical_synthesis_meta_regulariry']
                num_dims_with_content = 0
                for dim in dimensions:
                    dim_eval = nov_eval.get(dim, {})
                    if (dim_eval == None) or (not isinstance(dim_eval, dict)) or (len(dim_eval.keys()) == 0):
                        continue
                    what_is_known = dim_eval.get('what_is_known', 'Not provided.')
                    what_introduced = dim_eval.get('what_introduced', 'Not provided.')
                    what_novel = dim_eval.get('what_novel', 'Not provided.')
                    degree_of_novelty = dim_eval.get('degree_of_novelty', 'Not provided.')

                    content += f'            <h4>Dimension: {dim}</h4>\n'
                    content += f'            <p><strong>What is Known:</strong> {what_is_known}</p>\n'
                    content += f'            <p><strong>What is Introduced by the Theory:</strong> {what_introduced}</p>\n'
                    content += f'            <p><strong>What is Novel:</strong> {what_novel}</p>\n'
                    content += f'            <p><strong>Degree of Novelty:</strong> {degree_of_novelty}</p>\n'
                    # Add a small amount of space
                    content += '            <div style="height: 40px;"></div>\n'
                    num_dims_with_content += 1

                if (num_dims_with_content == 0):
                    content += '            <p class="empty-note">Paper did not contain information relevant for evaluating novelty.</p>\n'

                content += '            <hr>\n'
            content += '        </div>\n'

            # Write to HTML file
            filename_out = "qn-" + str(eval_idx).zfill(5) + f"-{theory_id}.html"
            html = create_html_page(f"Qualified Novelty Evaluation for Theory {theory_id}", content, css_path="../style.css")
            with open(os.path.join(path_out, filename_out), 'w', encoding='utf-8') as f:
                f.write(html)

            eval_idx += 1

            qualified_novelty_evaluations_by_statement[theory_statement] = os.path.join(PATH_QUALNOV, filename_out)
    return qualified_novelty_evaluations_by_statement




def export_all(theories, extraction_schemas, extraction_results, theory_evaluations, paperstore, paperstore_lut, path_out, path_predictive_accuracy_evals, path_qualified_novelty_evals, DEBUG_LIMIT:int=None):
    """Export all data to HTML files"""
    os.makedirs(path_out, exist_ok=True)

    # Create CSS file first
    print("Creating CSS file...")
    create_css_file(path_out)

    # Create subdirectories
    os.makedirs(os.path.join(path_out, PATH_THEORIES), exist_ok=True)
    os.makedirs(os.path.join(path_out, PATH_SCHEMAS), exist_ok=True)
    os.makedirs(os.path.join(path_out, PATH_RESULTS), exist_ok=True)
    os.makedirs(os.path.join(path_out, PATH_EVALUATIONS), exist_ok=True)

    # Export a list of all the theory queries
    print("Exporting theory queries list...")
    theory_queries = set()
    for theory_id, theory in theories.items():
        theory_query = theory.get('theory_query', None)
        if theory_query:
            theory_queries.add(theory_query)
    theory_queries = list(theory_queries)
    theory_queries.sort()
    filename_queries = os.path.join(path_out, "theory_queries.txt")
    with open(filename_queries, 'w', encoding='utf-8') as f:
        for tq in theory_queries:
            f.write(tq + "\n")
    print(f"  ✓ Exported {len(theory_queries)} unique theory queries to {filename_queries}")

    # Build UUID to extraction ID mapping
    print("Building UUID mappings...")
    uuid_to_extraction_id = build_extraction_maps(extraction_results)

    # Export predictive accuracy evaluations
    print("Exporting predictive accuracy evaluations...")
    predictive_accuracy_evaluations_by_statement_lut = {}
    if (path_predictive_accuracy_evals != None):
        predictive_accuracy_evaluations_by_statement_lut = export_predictive_accuracy_evaluations(path_predictive_accuracy_evals, os.path.join(path_out, PATH_PREDACC))

    # Export qualified novelty evaluations
    print("Exporting qualified novelty evaluations...")
    qualified_novelty_evaluations_by_statement_lut = {}
    if (path_qualified_novelty_evals != None):
        qualified_novelty_evaluations_by_statement_lut = export_qualified_novelty_evaluations(path_qualified_novelty_evals, os.path.join(path_out, PATH_QUALNOV))

    # Export theories
    print("Exporting theories...")
    for theory_id, theory in theories.items():
        filename = os.path.join(path_out, PATH_THEORIES, f"{theory_id}.html")
        export_one_theory(theory, filename, uuid_to_extraction_id, predictive_accuracy_evaluations_by_statement_lut, qualified_novelty_evaluations_by_statement_lut)
    print(f"  ✓ Exported {len(theories)} theories")

    # Export extraction schemas
    print("Exporting extraction schemas...")
    for schema_id, schema in extraction_schemas.items():
        filename = os.path.join(path_out, PATH_SCHEMAS, f"{schema_id}.html")
        export_one_extraction_schema(schema, filename)
    print(f"  ✓ Exported {len(extraction_schemas)} extraction schemas")

    # Export extraction results (one file per extraction, not per UUID)
    print("Exporting extraction results...")
    count = 0
    from tqdm import tqdm
    for extraction_id, extraction in tqdm(extraction_results.items(), desc="  Exporting extraction results", unit="result"):
        filename = os.path.join(path_out, PATH_RESULTS, f"{extraction_id}.html")
        export_one_extraction_result(extraction, filename, paperstore, paperstore_lut, extraction_schemas)
        count += 1
        if (DEBUG_LIMIT != None) and (count >= DEBUG_LIMIT):
            print(" NOTE: DEBUG LIMIT REACHED FOR EXTRACTION RESULTS -- STOPPING EARLY")
            break

    print(f"  ✓ Exported {len(extraction_results)} extraction results")

    # Export theory evaluations
    print("Exporting theory evaluations...")
    for eval_id, evaluation in theory_evaluations.items():
        filename = os.path.join(path_out, PATH_EVALUATIONS, f"{eval_id}.html")
        export_one_theory_evaluation(evaluation, filename, uuid_to_extraction_id)
    print(f"  ✓ Exported {len(theory_evaluations)} theory evaluations")



    # Create index page
    print("Creating index page...")
    create_index_page(theories, path_out)


#
#   Main Entry Point for exporting the theory store as HTML
#
if __name__ == "__main__":
    # Filename of the input theory store and paper store
    filename_in_theory_store = "theorystore-example2-literaturesupported.json"
    filename_in_paper_store = "paperstore-example2-literaturesupported.json"

    # Paths to predictive accuracy evaluations and qualified novelty evaluations (if available).  If not available, set to None.
    path_predictive_accuracy_evals = "theorystore-example2-predictive-evaluation/predictive-accuracy-evaluation/claude_sonnet_4_5_20250929/"
    path_qualified_novelty_evals = "qualified-novelty-evaluations/literature-supported/"

    # Path to export HTML files to
    path_out_html = "html-output-example2/"

    # Debug limit (set to None for no limit)
    DEBUG_LIMIT = None
    #DEBUG_LIMIT = 100

    # Load the theory store
    print("Loading theory store...")
    with open(filename_in_theory_store, "r") as f:
        theory_store = json.load(f)
    theories = theory_store["theories"]
    extraction_schemas = theory_store["extraction_schemas"]
    extraction_results = theory_store["extraction_results"]
    theory_evaluations = theory_store["theory_evaluations"]

    # Load paper store
    print("Loading paper store...")
    with open(filename_in_paper_store, "r") as f:
        paper_store = json.load(f)
    paperstore = paper_store["paperstore"]
    paperstore_lut = paper_store["paperstore_lut"]

    # Export all
    print("\nExporting all data to HTML...")
    export_all(theories, extraction_schemas, extraction_results, theory_evaluations, paperstore, paperstore_lut, path_out_html, path_predictive_accuracy_evals, path_qualified_novelty_evals, DEBUG_LIMIT=DEBUG_LIMIT)

    print("\n✓ Export complete!")
    print(f"Open {os.path.join(path_out_html, 'index.html')} in your browser to view the theories.")