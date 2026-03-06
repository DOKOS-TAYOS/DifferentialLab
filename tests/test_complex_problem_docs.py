"""Tests for shared complex-problem documentation metadata."""

from __future__ import annotations

from complex_problems.problem_docs import get_all_problem_docs, get_problem_doc
from complex_problems.problem_registry import get_problem_descriptors


def test_problem_docs_cover_all_registered_problems() -> None:
    descriptors = get_problem_descriptors()
    docs = get_all_problem_docs()
    assert set(docs) == set(descriptors)


def test_problem_doc_fields_are_non_empty() -> None:
    for problem_id in get_problem_descriptors():
        doc = get_problem_doc(problem_id)
        assert doc.problem_type.strip()
        assert doc.extended_description.strip()
        assert doc.config_options_summary
        assert doc.visualizations_summary
        for line in doc.config_options_summary:
            assert line.strip()
        for line in doc.visualizations_summary:
            assert line.strip()
