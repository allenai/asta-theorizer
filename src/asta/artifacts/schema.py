"""ExtractionQuerySchema -> Artifact."""

from __future__ import annotations

from asta_artifact import Artifact


def schema_to_artifact(schema) -> Artifact:
    a = Artifact(
        artifact_id=schema.id,
        name="Extraction Schema",
        description=schema.extraction_query,
        subtype="extraction-schema",
    )
    if schema.schema:
        with a.section("Schema") as s:
            s.table(
                ["Field", "Type", "Description"],
                [[f.get("name", ""), f.get("type", ""), f.get("description", "")] for f in schema.schema],
            )
    return a
