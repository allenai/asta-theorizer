"""Public installations do not contact deployment-specific services by default."""

from unittest.mock import patch

import pytest
import NoraRetrievalFullText as retrieval
import SemanticScholar as scholar
from PaperStore import PaperStore


def test_retrieval_requires_explicit_service_address(monkeypatch):
    monkeypatch.setenv("NORA_RETRIEVAL_ENV", "prod")
    monkeypatch.delenv("NORA_RETRIEVAL_BASE_URL_TEMPLATE", raising=False)
    assert retrieval.base_url() is None


def test_paperfinder_reports_missing_service_address(monkeypatch):
    monkeypatch.setattr(scholar, "_PAPER_FINDER_BASE_URL", "")
    with patch.object(scholar.requests, "post") as post:
        with pytest.raises(scholar.PaperFinderError, match="PAPER_FINDER_BASE_URL"):
            scholar._get_paperfinder_results_internal("test")
        post.assert_not_called()


def test_paper_cache_is_disabled_without_address(monkeypatch):
    monkeypatch.delenv("PAPER_CACHE_BASE_URL", raising=False)
    assert PaperStore._add_paper_internal(None, corpus_id="123") is None
