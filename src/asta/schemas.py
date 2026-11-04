"""Pydantic input shapes and Annotated aliases used in @asta.task signatures."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Theorizer1 import (
    DEFAULT_MODEL_PRIMARY,
    DEFAULT_MODEL_EXTRACTION,
    GENERATION_OBJECTIVE_ACCURACY_FOCUS,
)


PaperStatus = Literal["full-text", "metadata-only", "failed", "timeout"]
GenerationObjective = Literal["accuracy-focused", "novelty-focused"]


# extra="allow" lets Theorizer1 add fields without breaking validation; protected_namespaces=() because it uses the reserved model_str prefix.
_PERMISSIVE = ConfigDict(extra="allow", populate_by_name=True, protected_namespaces=())


class PaperEntry(BaseModel):
    model_config = _PERMISSIVE

    status: Optional[PaperStatus] = None
    s2_id: Optional[str] = None
    corpus_id: Optional[Union[str, int]] = None
    title: Optional[str] = None
    publication_year: Optional[int] = None
    publication_month: Optional[int] = None
    s2_metadata: Optional[dict[str, Any]] = None
    paper_urls: list[str] = Field(default_factory=list)
    paper_markdown: Optional[str] = None
    # asta-papers provenance (populated when USE_ASTA_PAPERS fetch succeeds; None otherwise)
    fetch_channel: Optional[str] = None
    license_class: Optional[str] = None
    license_url: Optional[str] = None
    source_type: Optional[str] = None
    # True/False when asta-papers license-gated the text for LLM/TDM use; None on the legacy path
    llm_usable: Optional[bool] = None


class PaperStore(BaseModel):
    model_config = _PERMISSIVE

    paperstore: dict[str, PaperEntry]
    paperstore_lut: dict[str, str] = Field(default_factory=dict)


class ExtractionSchemaField(BaseModel):
    model_config = _PERMISSIVE

    name: str = ""
    type: str = ""
    description: str = ""


class ExtractionQuerySchema(BaseModel):
    model_config = _PERMISSIVE

    id: Optional[str] = None
    schema_: list[ExtractionSchemaField] = Field(default_factory=list, alias="schema")
    extraction_query: str = ""
    supporting_theory_ids: list[str] = Field(default_factory=list)
    model_str: Optional[str] = None


class ExtractionQueryResult(BaseModel):
    model_config = _PERMISSIVE

    id: Optional[str] = None
    paper_id: str
    extraction_schema_id: Optional[str] = None
    extracted_data: list[dict[str, Any]] = Field(default_factory=list)
    potentially_relevant_new_papers: list[dict[str, Any]] = Field(default_factory=list)
    cost: float = 0.0
    model_str: Optional[str] = None


class EvidenceItem(BaseModel):
    model_config = _PERMISSIVE

    text: str = ""
    uuids: list[str] = Field(default_factory=list)


class TheoryStatementLaw(BaseModel):
    model_config = _PERMISSIVE

    statement_name: Optional[str] = None
    theory_statement: Optional[str] = None
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    conflicting_evidence: list[EvidenceItem] = Field(default_factory=list)


class TheoryStatement(BaseModel):
    # Theorizer1 emits both nested {"law": {...}} and flat-law shapes.
    model_config = _PERMISSIVE

    law: Optional[TheoryStatementLaw] = None
    statement_name: Optional[str] = None
    theory_statement: Optional[str] = None
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    conflicting_evidence: list[EvidenceItem] = Field(default_factory=list)


class TheoryComponents(BaseModel):
    model_config = _PERMISSIVE

    theory_statements: Optional[list[TheoryStatement]] = None
    new_predictions_likely: Optional[list[str]] = None
    new_predictions_unknown: Optional[list[str]] = None
    unaccounted_for: Optional[list[EvidenceItem]] = None
    negative_experiments: Optional[list[str]] = None
    supporting_evidence: Optional[list[EvidenceItem]] = None


class Theory(BaseModel):
    model_config = _PERMISSIVE

    id: Optional[str] = None
    name: str
    description: str
    type: Optional[str] = None
    theory_query: Optional[str] = ""
    derived_from: list[str] = Field(default_factory=list)
    theory_evaluation_ids: list[str] = Field(default_factory=list)
    components: TheoryComponents = Field(default_factory=TheoryComponents)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    knowledge_cutoff_year: Optional[int] = None
    knowledge_cutoff_month: Optional[int] = None
    change_log: list[dict[str, Any]] = Field(default_factory=list)
    model_str: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_lists(cls, data: Any) -> Any:
        # Older Theorizer1 snapshots persist these as null.
        if isinstance(data, dict):
            for f in ("derived_from", "theory_evaluation_ids",
                      "supporting_evidence_ids", "change_log"):
                if f in data and data[f] is None:
                    data[f] = []
        return data


class TheoryEvaluation(BaseModel):
    model_config = _PERMISSIVE

    id: Optional[str] = None
    theory_id: str
    fully_supporting_evidence: list[dict[str, Any]] = Field(default_factory=list)
    partially_supporting_evidence: list[dict[str, Any]] = Field(default_factory=list)
    fully_contradicting_evidence: list[dict[str, Any]] = Field(default_factory=list)
    partially_contradicting_evidence: list[dict[str, Any]] = Field(default_factory=list)
    potentially_modifying_evidence: list[dict[str, Any]] = Field(default_factory=list)
    suggested_revisions: list[dict[str, Any]] = Field(default_factory=list)
    overall_support_or_contradict: Optional[str] = None
    overall_support_or_contradict_explanation: Optional[str] = None
    revised_theory_ids: list[str] = Field(default_factory=list)
    model_str: Optional[str] = None


class TheoryStore(BaseModel):
    model_config = _PERMISSIVE

    theories: dict[str, Theory]
    extraction_schemas: dict[str, ExtractionQuerySchema]
    extraction_results: dict[str, ExtractionQueryResult]
    theory_evaluations: dict[str, TheoryEvaluation] = Field(default_factory=dict)
    all_ids: list[str] = Field(default_factory=list)
    next_ids: dict[str, int] = Field(default_factory=dict)


class ExtractionSchemaInput(BaseModel):
    model_config = _PERMISSIVE

    extraction_schemas: dict[str, ExtractionQuerySchema]


class ExtractionResultsInput(BaseModel):
    model_config = _PERMISSIVE

    extraction_schemas: dict[str, ExtractionQuerySchema]
    extraction_results: dict[str, ExtractionQueryResult]


# Annotated aliases for @asta.task signatures; Field(description=...) surfaces in the agent card's per-skill input schemas.

Query = Annotated[str, Field(
    description="Research question.",
    examples=["What factors govern transformer arithmetic accuracy?"],
)]
Mission = Annotated[str, Field(description="Optional focus or constraints to scope theory generation.")]
PaperStoreInput = Annotated[Optional[PaperStore], Field(
    description=(
        "Pre-collected paper store to use as input. Entries with `paper_markdown` "
        "are used directly. Entries without `paper_markdown` but with at least one "
        "of `corpus_id`, `s2_id`, or `title` are hydrated server-side via the same "
        "S2 + MistralOCR pipeline used for PaperFinder discoveries (~30-60s per "
        "paper). Hydration is performed during the PaperFinder step, so it "
        "requires `search_additional_papers=True`; with PaperFinder disabled, "
        "identifier-only entries are dropped. Duplicates between injected seeds "
        "and PaperFinder discoveries dedupe by corpus_id."
    ),
    default=None,
)]
RequiredPaperStore = Annotated[PaperStore, Field(
    description=(
        "Additional papers to merge into the prior run's paper store. Pass "
        "entries with `paper_markdown` populated; identifier-only entries are "
        "not hydrated by this task (use a fresh Literature Theory Generation "
        "run if you need server-side hydration)."
    ),
)]
ExtractionSchemaIn = Annotated[Optional[ExtractionSchemaInput], Field(
    description="Pre-built extraction schema. Skips schema construction.",
    default=None,
)]
ExtractionResultsIn = Annotated[Optional[ExtractionResultsInput], Field(
    description="Pre-extracted results. Skips paper finding and extraction.",
    default=None,
)]
TheoryStoreIn = Annotated[Optional[TheoryStore], Field(
    description="Full theory store. Skips paper finding, extraction, and theory generation.",
    default=None,
)]

MaxPapers = Annotated[int, Field(description="Max papers to find and extract from.", ge=1)]
CutoffYear = Annotated[int, Field(description="Only use papers published before this year.")]
CutoffMonth = Annotated[int, Field(description="Only use papers published before this month.", ge=1, le=12)]
PrimaryModel = Annotated[str, Field(description="LiteLLM model id for theory generation and schema construction.")]
ExtractionModel = Annotated[str, Field(description="LiteLLM model id for per-paper extraction.")]
Objective = Annotated[GenerationObjective, Field(description="Theory-generation objective.")]
MaxTokens = Annotated[int, Field(description="Max tokens of extraction data sent to the generation model.", ge=1)]
Temperature = Annotated[float, Field(
    description="LLM sampling temperature (0.0 = deterministic; 1.0 = model default).",
    ge=0.0, le=2.0,
)]
RunId = Annotated[Optional[str], Field(
    description="Run ID returned by a prior task (shape: 'gen-YYYYMMDD-<8hex>').",
    examples=["gen-20260428-bc46f181"],
)]
DoNovelty = Annotated[bool, Field(description="Run novelty evaluation after theory generation.")]
SearchPapers = Annotated[bool, Field(description="Also search open-access literature with PaperFinder.")]


DEFAULT_MAX_PAPERS = 50
DEFAULT_KNOWLEDGE_CUTOFF_YEAR = 2050
DEFAULT_KNOWLEDGE_CUTOFF_MONTH = 1
DEFAULT_MAX_DATA_TOKENS = 100000
DEFAULT_GENERATION_TEMPERATURE = 1.0


__all__ = [
    "PaperEntry", "PaperStore",
    "ExtractionQuerySchema", "ExtractionQueryResult",
    "Theory", "TheoryStatement", "TheoryComponents", "TheoryEvaluation",
    "TheoryStore", "ExtractionSchemaInput", "ExtractionResultsInput",
    "Query", "Mission",
    "PaperStoreInput", "RequiredPaperStore",
    "ExtractionSchemaIn", "ExtractionResultsIn", "TheoryStoreIn",
    "MaxPapers", "CutoffYear", "CutoffMonth",
    "PrimaryModel", "ExtractionModel",
    "Objective", "MaxTokens", "Temperature",
    "RunId", "DoNovelty", "SearchPapers",
    "DEFAULT_MAX_PAPERS", "DEFAULT_KNOWLEDGE_CUTOFF_YEAR",
    "DEFAULT_KNOWLEDGE_CUTOFF_MONTH", "DEFAULT_MAX_DATA_TOKENS",
    "DEFAULT_GENERATION_TEMPERATURE",
    "DEFAULT_MODEL_PRIMARY", "DEFAULT_MODEL_EXTRACTION",
    "GENERATION_OBJECTIVE_ACCURACY_FOCUS",
]
