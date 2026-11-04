"""Per-skill completion messages: callables(workflow, credits_charged) -> str, where
credits_charged is None when no charge landed."""

from __future__ import annotations

from asta_agent import ArtifactRef


def theory_message(workflow, credits_charged: int | None) -> str:
    # Per-theory artifacts still stream during the run (and appear in status updates);
    # the final message only summarizes the count and links the Theory Store artifact
    # (appended in AstaWorkflow._succeed).
    n = len(workflow.theory_store.theories)
    if not n:
        return "No theories generated."
    theories = "theory" if n == 1 else "theories"
    message = f"Generated {n} {theories}"
    if credits_charged:
        credits = "credit" if credits_charged == 1 else "credits"
        message += f" using {credits_charged} {credits}"
    return message + ":"


def schema_message(workflow, _credits_charged: int | None) -> str:
    refs = [ArtifactRef(str(s.id), "Extraction Schema")
            for s in workflow.theory_store.extraction_schemas.values()]
    if not refs:
        return "No extraction schema generated."
    return "Built extraction schema:\n\n" + "\n".join(str(r) for r in refs)
