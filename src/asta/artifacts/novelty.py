"""Aggregated novelty evaluation -> Artifact."""

from __future__ import annotations

from asta_artifact import Artifact


NOVELTY_DIMENSION_LABELS = {
    "phenomenon_effect": "Phenomenon / Effect",
    "explanatory_mechanistic": "Explanatory / Mechanistic",
    "unification": "Unification",
    "generalization_scope_expansion": "Generalization / Scope Expansion",
    "constraint_limitation": "Constraint / Limitation",
    "conceptual_reframing_abstraction": "Conceptual Reframing / Abstraction",
    "empirical_synthesis_meta_regulariry": "Empirical Synthesis / Meta-Regularity",
}

NOVELTY_DIMENSION_KEYS = list(NOVELTY_DIMENSION_LABELS.keys())

NOVELTY_DEGREE_LABELS = {
    "explicit_established": "Explicit Established",
    "explicit_peripheral": "Explicit Peripheral",
    "derivable_unstated": "Derivable Unstated",
    "genuinely_new": "Genuinely New",
}


NOVELTY_DEGREE_ORDER = list(NOVELTY_DEGREE_LABELS.keys())


def novelty_to_artifact(
    theory_id: str,
    statement_index: int,
    statement_name: str | None,
    agg_eval: dict,
) -> Artifact:
    name = statement_name or f"Statement {statement_index + 1}"
    artifact_id = f"novelty-{theory_id}-{statement_index}"

    highest_idx = -1
    novel_parts = []
    for dim_key in NOVELTY_DIMENSION_KEYS:
        dim = agg_eval.get(dim_key)
        if not isinstance(dim, dict) or not dim.get("degree_of_novelty"):
            continue
        try:
            idx = NOVELTY_DEGREE_ORDER.index(dim["degree_of_novelty"])
            if idx > highest_idx:
                highest_idx = idx
        except ValueError:
            pass
        what_novel = dim.get("what_novel")
        if what_novel and what_novel.strip():
            novel_parts.append(what_novel.strip())

    highest = NOVELTY_DEGREE_ORDER[highest_idx] if highest_idx >= 0 else None
    verdict = NOVELTY_DEGREE_LABELS.get(highest, "Not assessed") if highest else "Not assessed"
    description = (f"{verdict}. " + " ".join(novel_parts)) if novel_parts else verdict

    a = Artifact(
        artifact_id=artifact_id,
        name=f"Novelty: {name}",
        description=description,
        subtype="novelty",
    )
    a.markdown(f'Assessed theory: <artifact id="{theory_id}">{theory_id}</artifact>')

    with a.sections("Dimensions") as ss:
        for dim_key in NOVELTY_DIMENSION_KEYS:
            dim_label = NOVELTY_DIMENSION_LABELS[dim_key]
            dim_eval = agg_eval.get(dim_key)

            if not isinstance(dim_eval, dict) or not dim_eval:
                with ss.section(f"{dim_label} — Not assessed") as s:
                    s.markdown("This dimension was not assessed.")
                continue

            degree = dim_eval.get("degree_of_novelty", "Not assessed")
            degree_display = NOVELTY_DEGREE_LABELS.get(degree, degree)
            what_known = dim_eval.get("what_is_known", "Not provided.")
            what_introduced = dim_eval.get("what_introduced", "Not provided.")
            what_novel = dim_eval.get("what_novel", "Not provided.")

            with ss.section(f"{dim_label} — {degree_display}") as s:
                s.markdown(f"**What is Known:** {what_known}")
                s.markdown(f"\n\n**What is Introduced:** {what_introduced}")
                s.markdown(f"\n\n**What is Novel:** {what_novel}")

    return a
