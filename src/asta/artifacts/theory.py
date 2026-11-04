"""Theory -> Artifact."""

from __future__ import annotations

from asta_artifact import Artifact, paper_tag

from .common import get_s2, s2_corpus_id, s2_citation, datum_content_id


def theory_to_artifact(theory, theory_store, paperstore) -> Artifact:
    components = getattr(theory, "components", {}) or {}
    display_name = components.get("theory_name_plain_language") or theory.name
    display_description = components.get("theory_description_plain_language") or theory.description
    a = Artifact(artifact_id=str(theory.id), name=display_name,
                 description=display_description, subtype="theory")
    paper_meta = _add_paper_entities(theory, theory_store, paperstore, a)

    statements = components.get("theory_statements") or []
    if statements:
        with a.sections(f"{len(statements)} Theory Statements") as ss:
            for stmt in statements:
                if not isinstance(stmt, dict):
                    continue
                law = stmt.get("law") or stmt  # `or`, not default: a stmt with explicit "law": null must fall back to stmt
                with ss.section(law.get("statement_name", "Statement")) as s:
                    statement_text = law.get("theory_statement_plain_language") or law.get("theory_statement")
                    if statement_text:
                        s.markdown(statement_text)
                    if law.get("supporting_evidence"):
                        s.markdown("\n\n**Supporting evidence:**")
                        _add_evidence_lines(law["supporting_evidence"], s, a, theory_store, paper_meta)
                    if law.get("conflicting_evidence"):
                        s.markdown("\n\n**Conflicting evidence:**")
                        _add_evidence_lines(law["conflicting_evidence"], s, a, theory_store, paper_meta)

    pred_likely = components.get("new_predictions_likely") or []
    pred_unknown = components.get("new_predictions_unknown") or []
    if pred_likely or pred_unknown:
        with a.sections("Predictions") as ss:
            if pred_likely:
                with ss.section(f"{len(pred_likely)} Likely outcomes") as s:
                    s.bullet_list(pred_likely)
            if pred_unknown:
                with ss.section(f"{len(pred_unknown)} Unknown outcomes") as s:
                    s.bullet_list(pred_unknown)

    unaccounted = components.get("unaccounted_for") or []
    negative = components.get("negative_experiments") or []
    if unaccounted or negative:
        with a.sections("Conflicting & Unaccounted Evidence") as ss:
            if negative:
                with ss.section(f"{len(negative)} Negative experiments") as s:
                    s.bullet_list(negative)
            if unaccounted:
                with ss.section(f"{len(unaccounted)} Unaccounted for") as s:
                    _add_evidence_lines(unaccounted, s, a, theory_store, paper_meta)

    return a


def _add_paper_entities(theory, theory_store, paperstore, artifact):
    # Returns paper_key -> (s2_dict, title) for use when rendering evidence lines.
    paper_meta: dict[str, tuple[dict, str]] = {}
    for evidence_id in getattr(theory, "supporting_evidence_ids", []) or []:
        result = theory_store.get_extraction_result(evidence_id)
        if not result:
            continue
        pk = result.paper_id
        if pk not in paper_meta:
            resolved = get_s2(pk, paperstore)
            if resolved:
                s2, title = resolved
                artifact.add_paper_entity(display_label=title, s2_metadata=s2, entity_id=pk)
                paper_meta[pk] = (s2, title)
    return paper_meta


def _add_evidence_lines(evidence_list, section, artifact, theory_store, paper_meta):
    for ev in evidence_list:
        text = ev.get("text", str(ev)) if isinstance(ev, dict) else str(ev)
        if not text:
            continue

        uuids = ev.get("uuids", []) if isinstance(ev, dict) else []
        tags, ann_ids = [], []
        seen: set[str] = set()
        for u in uuids:
            entry = theory_store.uuid_index.get(u)
            if not entry:
                continue
            pk, result_id = entry
            if pk in seen:
                continue
            seen.add(pk)
            s2, _ = paper_meta.get(pk, (None, None))
            if not s2:
                continue
            deep_link = f"{result_id}#{datum_content_id(u)}"
            ann_id = artifact.add_snippet(entity_id=pk, text="")
            ann_ids.append(ann_id)
            tags.append(paper_tag(pk, ann_id, s2_corpus_id(s2), s2_citation(s2), artifact_id=deep_link))

        line = f"- {text}"
        if tags:
            line += "  " + " ".join(tags)
        md = section.markdown(line)
        md.annotation_ids.extend(ann_ids)
