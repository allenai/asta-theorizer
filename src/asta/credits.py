"""Credit pricing and estimation for the Theorizer's A2A skills.

Pricing model: per unit of work, where the unit is the **number of papers** the
run processes (``N × number of papers``). ``N`` (``CREDITS_PER_PAPER``) and the
per-run ``BASE_CREDITS`` are placeholders — the final rates are not yet decided,
so tune them here without touching any call site.
"""

from __future__ import annotations

from Theorizer1 import Step

# The nora-credits bucket these skills draw from. Same string for check + consume.
CREDIT_TYPE = "general"

# Pricing constants (PLACEHOLDERS — final rates TBD; see CREDITS.md "Open items").
# Cost is dominated by per-paper work (OCR + extraction LLM calls), so the unit is
# the number of papers a run processes.
CREDITS_PER_PAPER = 1  # the "N" in "N × number of papers"
# A small per-run base so paperless skills (e.g. Build Extraction Schema, which
# processes zero papers) still charge something — needed to honor "gate all six".
# Set to 0 to make paperless runs free.
BASE_CREDITS = 1


def calc_credits(num_papers: int) -> int:
    """Credits owed for processing ``num_papers`` papers: ``BASE + N × papers``."""
    return BASE_CREDITS + CREDITS_PER_PAPER * max(0, int(num_papers))


def _papers_in_store(workflow) -> int:
    """Papers currently present in the workflow's paper store (injected or resumed)."""
    try:
        return len(workflow.theory_store.paperstore.paperstore or {})
    except AttributeError:
        return 0


def _extraction_results(workflow) -> int:
    """Number of per-paper extraction results in the workflow's store."""
    try:
        return len(workflow.theory_store.extraction_results or {})
    except AttributeError:
        return 0


def estimate_paper_count(workflow, steps) -> int:
    """Generous pre-run upper bound on the papers this run will process.

    Used by the up-front gate. Over-estimating only asks the caller to hold a
    little more; the actual charge (see :func:`actual_paper_count`) is what they
    really pay.

    - If the run will search PaperFinder, it can pull up to
      ``max_papers_to_retrieve`` new papers on top of any already injected.
    - Otherwise (extract-only on injected papers, or form-theory / novelty over
      existing evidence) it works the papers already in the store.
    """
    wf = workflow.theory_workflow_structure
    existing = _papers_in_store(workflow)
    max_papers = int(wf.get("max_papers_to_retrieve") or 0)

    if Step.FIND_PAPERS_WITH_PAPERFINDER in steps:
        return existing + max_papers
    if Step.EXTRACT_FROM_PAPERS in steps:
        # Extracting the injected set, no new retrieval.
        return existing
    # No paper processing (e.g. Form Theory, Evaluate Novelty, Build Schema).
    return existing


def actual_paper_count(workflow) -> int:
    """Papers actually processed by a finished run.

    Prefers extraction results (the real per-paper work); falls back to papers
    present in the store for skills that don't extract (e.g. Evaluate Novelty or
    Form Theory over injected evidence). Returns 0 for paperless runs (e.g.
    schema-only), which then price at ``BASE_CREDITS``.
    """
    return _extraction_results(workflow) or _papers_in_store(workflow)
