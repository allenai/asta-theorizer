"""Shared helpers for artifact builders."""

from __future__ import annotations


def get_s2(paper_key: str, paperstore) -> tuple[dict, str] | None:
    paper = paperstore.get_paper_by_key(paper_key)
    if not paper:
        return None
    s2 = paper.get("s2_metadata")
    if not s2:
        return None
    title = paper.get("title") or s2.get("title", paper_key)
    return s2, title


def s2_corpus_id(s2: dict) -> str | None:
    cid = s2.get("corpusId") or s2.get("corpus_id")
    return str(cid) if cid else None


def s2_citation(s2: dict) -> str:
    authors = s2.get("authors") or []
    parts = (authors[0].get("name") or "").split() if authors else []
    first = parts[-1] if parts else ""
    suffix = " et al." if len(authors) > 1 else ""
    year = s2.get("year", "")
    if first and year:
        return f"{first}{suffix}, {year}"
    return f"{first}{suffix}{year}"


def s2_url(s2: dict) -> str | None:
    cid = s2_corpus_id(s2)
    return f"https://www.semanticscholar.org/paper/{cid}" if cid else None


def datum_content_id(uuid: str) -> str:
    return f"uuid-{uuid}"
