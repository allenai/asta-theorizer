# Struct.py
# Data structures for the Theorizer system

from __future__ import annotations

import os
import json
import time

# Imports for the data structures
from typing import List, Dict

from PaperStore import *


class TheoryStore():
    #
    #   Constructor
    #
    def __init__(self, paperstore:PaperStore=None, filename_in:str=None):
        self.theories = {}
        #self.papers = {}
        self.paperstore = paperstore if paperstore is not None else PaperStore()
        self.extraction_schemas = {}
        self.extraction_results = {}
        self.theory_evaluations = {}

        # For generating UUIDs
        self.all_ids = set()  # To keep track of all unique IDs
        self.next_ids = {"theory": 0, "extraction-schema": 0, "extraction-result": 0, "theory-evaluation": 0}

        # Load the store from a file (if provided)
        if (filename_in is not None):
            self.load(filename_in)



    #
    #   Getters/Setters
    #

    # Theory
    def add_theory(self, theory:Theory):
        if (not isinstance(theory, Theory)):
            raise ValueError("Expected a Theory object")

        # Check to see if the theory already exists
        existing_id = theory.id
        if (existing_id is not None):
            # Check to see if the theory already exists in the store
            if (existing_id in self.theories):
                print("WARNING: TheoryStore.add_theory: Adding a theory, but a theory with that ID already exists in the store: " + str(existing_id))
                return existing_id
            else:
                # Add the theory to the store
                self.theories[existing_id] = theory
                return existing_id
        else:
            # No ID -- generate a new one
            theory.id = self.generate_id("theory")
            self.theories[theory.id] = theory           # Can safely assume the generated ID is unique (since it's guaranteed by the function)
            return theory.id


    def get_theory(self, id:str):
        if (id not in self.theories):
            return None
        return self.theories[id]

    def num_theories(self):
        return len(self.theories)



    # Extraction Schema
    def add_extraction_schema(self, schema:ExtractionQuerySchema):
        if (not isinstance(schema, ExtractionQuerySchema)):
            raise ValueError("Expected an ExtractionQuerySchema object")

        # Check to see if the schema already exists
        existing_id = schema.id
        if (existing_id is not None):
            # Check to see if the schema already exists in the store
            if (existing_id in self.extraction_schemas):
                print("WARNING: TheoryStore.add_extraction_schema: Adding a schema, but a schema with that ID already exists in the store: " + str(existing_id))
                return existing_id
            else:
                # Add the schema to the store
                self.extraction_schemas[existing_id] = schema
                return existing_id
        else:
            # No ID -- generate a new one
            schema.id = self.generate_id("extraction-schema")
            self.extraction_schemas[schema.id] = schema           # Can safely assume the generated ID is unique (since it's guaranteed by the function)
            return schema.id

    def get_extraction_schema(self, id:str):
        if (id not in self.extraction_schemas):
            return None
        return self.extraction_schemas[id]

    def num_extraction_schemas(self):
        return len(self.extraction_schemas)



    # Extraction Result
    def add_extraction_result(self, result:ExtractionQueryResult):
        if (not isinstance(result, ExtractionQueryResult)):
            raise ValueError("Expected an ExtractionQueryResult object")

        # Check to see if the result already exists
        existing_id = result.id
        if (existing_id is not None):
            # Check to see if the result already exists in the store
            if (existing_id in self.extraction_results):
                print("WARNING: TheoryStore.add_extraction_result: Adding a result, but a result with that ID already exists in the store: " + str(existing_id))
                return existing_id
            else:
                # Add the result to the store
                self.extraction_results[existing_id] = result
                return existing_id
        else:
            # No ID -- generate a new one
            # Generate a new ID for the result
            new_id = self.generate_id("extraction-result")
            result.set_id(new_id)  # Set the ID in the result object.  Must use this method to correctly set the ID and add UUIDs to the extracted data.
            self.extraction_results[result.id] = result           # Can safely assume the generated ID is unique (since it's guaranteed by the function)
            return result.id


    def get_extraction_result(self, id:str):
        if (id not in self.extraction_results):
            return None
        return self.extraction_results[id]

    def num_extraction_results(self):
        return len(self.extraction_results)



    # Theory Evaluation
    def add_theory_evaluation(self, evaluation:TheoryEvaluation):
        if (not isinstance(evaluation, TheoryEvaluation)):
            raise ValueError("Expected a TheoryEvaluation object")

        # Check to see if the evaluation already exists
        existing_id = evaluation.id
        if (existing_id is not None):
            # Check to see if the evaluation already exists in the store
            if (existing_id in self.theory_evaluations):
                print("WARNING: TheoryStore.add_theory_evaluation: Adding an evaluation, but an evaluation with that ID already exists in the store: " + str(existing_id))
                return existing_id
            else:
                # Add the evaluation to the store
                self.theory_evaluations[existing_id] = evaluation
                return existing_id
        else:
            # No ID -- generate a new one
            evaluation.id = self.generate_id("theory-evaluation")
            self.theory_evaluations[evaluation.id] = evaluation           # Can safely assume the generated ID is unique (since it's guaranteed by the function)
            return evaluation.id

    def get_theory_evaluation(self, id:str):
        if (id not in self.theory_evaluations):
            return None
        return self.theory_evaluations[id]

    def num_theory_evaluations(self):
        return len(self.theory_evaluations)


    # Convert to lists
    def get_all_theories_as_dict(self):
        return {k: v.to_dict() for k, v in self.theories.items()}
    # def get_all_papers_as_dict(self):
    #     return {k: v.to_dict() for k, v in self.papers.items()}
    def get_all_extraction_schemas_as_dict(self):
        return {k: v.to_dict() for k, v in self.extraction_schemas.items()}
    def get_all_extraction_results_as_dict(self):
        return {k: v.to_dict() for k, v in self.extraction_results.items()}
    def get_all_theory_evaluations_as_dict(self):
        return {k: v.to_dict() for k, v in self.theory_evaluations.items()}


    #
    #   Loading/Saving
    #
    def save(self, filename:str):
        print("TheoryStore.save(): Saving store to file: " + filename)
        # Convert the whole store to a dictionary
        store_dict = {
            'theories': self.get_all_theories_as_dict(),
            'extraction_schemas': self.get_all_extraction_schemas_as_dict(),
            'extraction_results': self.get_all_extraction_results_as_dict(),
            'theory_evaluations': self.get_all_theory_evaluations_as_dict(),
            'all_ids': list(self.all_ids),
            'next_ids': self.next_ids
        }
        # Write to a JSON file
        with open(filename, 'w') as f:
            json.dump(store_dict, f, indent=4)


    def load(self, filename:str):
        print("TheoryStore.load(): Loading store from file: " + filename)
        if (not os.path.exists(filename)):
            raise FileNotFoundError(f"TheoryStore.load(): ERROR: File '{filename}' does not exist.")

        # Read from the JSON file
        with open(filename, 'r') as f:
            store_dict = json.load(f)

        # Clear the current store
        self.theories.clear()
        #self.papers.clear()
        self.extraction_schemas.clear()
        self.extraction_results.clear()
        self.theory_evaluations.clear()

        # Load theories
        for k, v in store_dict['theories'].items():
            theory = Theory.from_dict(v)
            theory.id = k
            self.theories[k] = theory

        # Load extraction schemas
        for k, v in store_dict['extraction_schemas'].items():
            schema = ExtractionQuerySchema.from_dict(v)
            schema.id = k
            self.extraction_schemas[k] = schema

        # Load extraction results
        for k, v in store_dict['extraction_results'].items():
            result = ExtractionQueryResult.from_dict(v)
            result.id = k
            self.extraction_results[k] = result

        # Load theory evaluations
        if ("theory_evaluations" in store_dict):
            # DEBUG: This is a new field, so we need to check if it exists
            for k, v in store_dict['theory_evaluations'].items():
                evaluation = TheoryEvaluation.from_dict(v)
                evaluation.id = k
                self.theory_evaluations[k] = evaluation

        # Load the IDs
        self.all_ids = set(store_dict['all_ids'])
        all_id_keys = self.next_ids.keys()
        self.next_ids = store_dict['next_ids']
        for key in all_id_keys:
            if (key not in self.next_ids):
                # If the key is not in the next_ids, set it to 0
                self.next_ids[key] = 0

        # Print some summary statistics
        print("TheoryStore.load(): Loaded store with " + str(len(self.theories)) + " theories, " + str(len(self.extraction_schemas)) + " extraction schemas, and " + str(len(self.extraction_results)) + " extraction results.")


    #
    #   Generating UUIDs
    #
    def generate_id(self, entity_type:str):
        next_id_str = None
        MAX_ITER = 100000

        # Check that the entity_type is valid
        if (entity_type not in self.next_ids):
            raise ValueError(f"Invalid entity type: {entity_type}. Valid types are: {list(self.next_ids.keys())}")

        iter_count = 0
        while (iter_count < MAX_ITER):
            self.next_ids[entity_type] += 1
            next_id = self.next_ids[entity_type]
            next_id_str = f"{entity_type}-{next_id}"
            # Ensure the ID is unique
            if (next_id_str not in self.all_ids):
                self.all_ids.add(next_id_str)
                return next_id_str

            iter_count += 1

        # If we reach here, we failed to generate a unique ID
        raise Exception(f"Failed to generate a unique ID for {entity_type} after {MAX_ITER} iterations.")






#
#   Storage Class: Theory
#
class Theory:
    # TODO: Might want to modify this to have a "knowledge cutoff date", to determine what knowledge the theory knows up to.
    def __init__(self, name:str, description:str, type:str, theory_query:str, derived_from:list, theory_evaluation_ids:list, components:dict, supporting_evidence_ids:list[str], knowledge_cutoff_year:int, knowledge_cutoff_month:int, change_log:list=None, id:str=None, model_str:str=None):
        self.id = id  # This will be set when added to the store
        self.name = name
        self.description = description
        self.type = type
        self.theory_query = theory_query
        self.derived_from = derived_from
        self.theory_evaluation_ids = theory_evaluation_ids  # A list of theory evaluation IDs that support this theory
        self.components = components  # A dictionary with all the (LLM-generated) theory components, that are of expected types, but may change -- so here it's being kept fluid as a dict.
        self.supporting_evidence_ids = supporting_evidence_ids
        self.knowledge_cutoff_year = knowledge_cutoff_year
        self.knowledge_cutoff_month = knowledge_cutoff_month
        self.change_log = change_log if change_log is not None else []
        self.model_str = model_str  # Model used for theory generation


    # Get the supporting data
    def get_supporting_data(self, theory_store:TheoryStore):
        out = []
        for evidence_id in self.supporting_evidence_ids:
            evidence = theory_store.get_extraction_result(evidence_id)
            if (evidence is not None):
                out.append(evidence.extracted_data)
        return out

    # Get the extraction schema(s) used for this theory
    def get_extraction_schemas(self, theory_store:TheoryStore):
        extraction_schemas = []
        schema_ids = set()
        # Step 1: Get a list of all the unique extraction schema ids
        for evidence_id in self.supporting_evidence_ids:
            evidence = theory_store.get_extraction_result(evidence_id)
            if (evidence is not None):
                schema_ids.add(evidence.extraction_schema_id)

        # Step 2: Get the actual schemas from the store
        for schema_id in schema_ids:
            schema = theory_store.get_extraction_schema(schema_id)
            if (schema is not None):
                extraction_schemas.append(schema)

        return extraction_schemas

    def add_theory_evaluation_id(self, theory_evaluation_id:str):
        if (theory_evaluation_id not in self.theory_evaluation_ids):
            self.theory_evaluation_ids.append(theory_evaluation_id)

    def add_derived_from_theory_id(self, theory_id:str):
        if (theory_id not in self.derived_from):
            self.derived_from.append(theory_id)

    #
    # Serialize the theory to a dictionary
    #
    def to_dict(self):
        return {
            'id': self.id,  # Include ID for serialization
            'name': self.name,
            'description': self.description,
            'type': self.type,
            'theory_query': self.theory_query,
            'derived_from': self.derived_from,
            'theory_evaluation_ids': self.theory_evaluation_ids,  # List of theory evaluation IDs
            'components': self.components,
            'supporting_evidence_ids': self.supporting_evidence_ids,
            'knowledge_cutoff_year': self.knowledge_cutoff_year,
            'knowledge_cutoff_month': self.knowledge_cutoff_month,
            'change_log': self.change_log,
            'model_str': self.model_str
        }

    # Deserialize a dictionary to a Theory object
    @staticmethod
    def from_dict(data:dict):
        return Theory(
            name=data['name'],
            description=data['description'],
            type=data['type'],
            theory_query=data.get('theory_query', ""),
            derived_from=data.get('derived_from', []),
            theory_evaluation_ids=data.get('theory_evaluation_ids', []),  # Default to empty list if not provided
            components=data['components'],
            supporting_evidence_ids=data['supporting_evidence_ids'],
            knowledge_cutoff_year=data['knowledge_cutoff_year'],
            knowledge_cutoff_month=data['knowledge_cutoff_month'],
            change_log=data.get('change_log', []),  # Default to empty list if not provided
            id=data.get('id', None),
            model_str=data.get('model_str', None)
        )




#
#   Storage Class: ExtractionQuerySchema
#
class ExtractionQuerySchema:
    def __init__(self, schema:list[dict], extraction_query:str, supporting_theory_ids:list[str], id:str=None, model_str:str=None):
        self.id = id
        self.schema = schema
        self.extraction_query = extraction_query
        self.supporting_theory_ids = supporting_theory_ids
        self.model_str = model_str  # Model used for schema generation

    #
    #   Serialization/Deserialization methods
    #
    def to_dict(self):
        return {
            'id': self.id,
            'schema': self.schema,
            'extraction_query': self.extraction_query,
            'supporting_theory_ids': self.supporting_theory_ids,
            'model_str': self.model_str
        }

    @staticmethod
    def from_dict(data:dict):
        return ExtractionQuerySchema(
            schema=data['schema'],
            extraction_query=data['extraction_query'],
            supporting_theory_ids=data['supporting_theory_ids'],
            id=data.get('id', None),
            model_str=data.get('model_str', None)
        )


#
#   Storage Class: ExtractionQueryResult
#
class ExtractionQueryResult:
    def __init__(self, paper_id:str, extraction_schema_id:int, extracted_data:list, potentially_relevant_new_papers:list[dict], cost:float=0.0, id:str=None, model_str:str=None):
        # NOTE: `potentially_relevant_new_papers` is a list of dictionaries, each representing a new paper (cited in this paper) that might be relevant to perform the extraction process on.
        # Each dictionary has two keys: `paper_title`, and `rating` (an integer, where higher means more relevant).
        self.id = id
        self.paper_id = paper_id
        self.extraction_schema_id = extraction_schema_id
        self.extracted_data = extracted_data
        self.potentially_relevant_new_papers = potentially_relevant_new_papers
        self.cost = cost
        self.model_str = model_str  # Model used for extraction

    def set_id(self, id:str):
        self.id = id
        # Also assign a UUID to each extraction result
        base_uuid = str(self.id).replace("extraction-result-", "e")  # Remove the prefix to get a clean base UUID (and just replace it with an "e"). This compactness is because these will be referenced/generated by LLMs.
        for idx in range(len(self.extracted_data)):
            data_uuid = base_uuid + "." + str(idx)
            if (isinstance(self.extracted_data[idx], dict)):
                self.extracted_data[idx]['uuid'] = data_uuid  # Add a UUID to each extracted data item

    #
    #   Serialization/Deserialization methods
    #
    def to_dict(self):
        return {
            'id': self.id,
            'paper_id': self.paper_id,
            'extraction_schema_id': self.extraction_schema_id,
            'extracted_data': self.extracted_data,
            'potentially_relevant_new_papers': self.potentially_relevant_new_papers,
            'cost': self.cost,
            'model_str': self.model_str
        }

    @staticmethod
    def from_dict(data:dict):
        return ExtractionQueryResult(
            paper_id=data['paper_id'],
            extraction_schema_id=data['extraction_schema_id'],
            extracted_data=data['extracted_data'],
            potentially_relevant_new_papers=data.get('potentially_relevant_new_papers', []),
            cost=data.get('cost', 0.0),
            id=data.get('id', None),
            model_str=data.get('model_str', None)
        )


#
#   Storage Class: Theory Evalation
#
class TheoryEvaluation:
    def __init__(self, theory_id:str, fully_supporting_evidence:list,
                 partially_supporting_evidence:list, fully_contradicting_evidence:list,
                 partially_contradicting_evidence:list, potentially_modifying_evidence:list,
                 suggested_revisions:list, overall_support_or_contradict:str,
                 overall_support_or_contradict_explanation:str, revised_theory_ids:list=None, id:str=None, model_str:str=None):
        # NOTE: lists of evidence are lists of dictionaries with two keys: `text` (text description) and `uuids` (a list of supporting evidence uuids)

        self.id = id
        self.theory_id = theory_id
        self.fully_supporting_evidence = fully_supporting_evidence
        self.partially_supporting_evidence = partially_supporting_evidence
        self.fully_contradicting_evidence = fully_contradicting_evidence
        self.partially_contradicting_evidence = partially_contradicting_evidence
        self.potentially_modifying_evidence = potentially_modifying_evidence
        self.suggested_revisions = suggested_revisions
        self.overall_support_or_contradict = overall_support_or_contradict
        self.overall_support_or_contradict_explanation = overall_support_or_contradict_explanation
        self.revised_theory_ids = revised_theory_ids if revised_theory_ids is not None else []
        self.model_str = model_str  # Model used for evaluation

    def add_revised_theory_id(self, theory_id:str):
        if (theory_id not in self.revised_theory_ids):
            self.revised_theory_ids.append(theory_id)

    #
    #  Serialization/Deserialization methods
    #
    def to_dict(self):
        return {
            'id': self.id,
            'theory_id': self.theory_id,
            'fully_supporting_evidence': self.fully_supporting_evidence,
            'partially_supporting_evidence': self.partially_supporting_evidence,
            'fully_contradicting_evidence': self.fully_contradicting_evidence,
            'partially_contradicting_evidence': self.partially_contradicting_evidence,
            'potentially_modifying_evidence': self.potentially_modifying_evidence,
            'suggested_revisions': self.suggested_revisions,
            'overall_support_or_contradict': self.overall_support_or_contradict,
            'overall_support_or_contradict_explanation': self.overall_support_or_contradict_explanation,
            'revised_theory_ids': self.revised_theory_ids,
            'model_str': self.model_str
        }

    @staticmethod
    def from_dict(data:dict):
        return TheoryEvaluation(
            theory_id=data['theory_id'],
            fully_supporting_evidence=data['fully_supporting_evidence'],
            partially_supporting_evidence=data['partially_supporting_evidence'],
            fully_contradicting_evidence=data['fully_contradicting_evidence'],
            partially_contradicting_evidence=data['partially_contradicting_evidence'],
            potentially_modifying_evidence=data['potentially_modifying_evidence'],
            suggested_revisions=data['suggested_revisions'],
            overall_support_or_contradict=data['overall_support_or_contradict'],
            overall_support_or_contradict_explanation=data['overall_support_or_contradict_explanation'],
            revised_theory_ids=data.get('revised_theory_ids', []),  # Default to empty list if not provided
            id=data.get('id', None),
            model_str=data.get('model_str', None)
        )