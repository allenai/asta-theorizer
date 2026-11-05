"""ExtractionQueryResult -> Artifact."""

from __future__ import annotations

from asta_artifact import Artifact

from .common import get_s2, datum_content_id

_S2_METADATA_KEYS = {
    "paperId", "corpusId", "externalIds", "url", "title", "abstract",
    "venue", "year", "authors", "journal", "publicationDate", "tldr",
}


def extraction_to_artifact(result, paperstore, extraction_query: str | None = None) -> Artifact | None:
    data = [item for item in (result.extracted_data or []) if isinstance(item, dict)]
    if not data:
        return None

    resolved = get_s2(result.paper_id, paperstore)
    s2, title = resolved if resolved else (None, result.paper_id)

    headings = []
    for item in data:
        heading = item.get("name_full") or item.get("name_short")
        if heading:
            headings.append(heading.strip())
    description = "; ".join(headings) if headings else f"Extracted data from {title}"

    metadata = {k: v for k, v in s2.items() if k in _S2_METADATA_KEYS} if s2 else None

    a = Artifact(
        artifact_id=result.id,
        name=f"Extraction: {title}",
        description=description,
        subtype="extraction",
        metadata=metadata,
    )

    for idx, item in enumerate(data):
        heading = item.get("name_full") or item.get("name_short") or f"Instance {idx + 1}"
        section_id = datum_content_id(item["uuid"]) if item.get("uuid") else None
        rows = [[k, str(v)] for k, v in item.items() if k != "uuid"]
        with a.section(heading, section_id=section_id) as s:
            s.table(["Field", "Value"], rows)

    return a
