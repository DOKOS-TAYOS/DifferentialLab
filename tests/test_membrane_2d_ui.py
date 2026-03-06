"""Tests for membrane 2D UI term-selection policy."""

from __future__ import annotations

import pytest

from complex_problems.membrane_2d.ui import (
    _TERM_ALPHA,
    _TERM_BETA,
    _TERM_HIGH,
    resolve_optional_membrane_terms,
)


def test_membrane_optional_terms_default_to_zero_when_not_selected() -> None:
    alpha, beta, high_coeff, high_power = resolve_optional_membrane_terms(
        set(),
        alpha_text="not-used",
        beta_text="not-used",
        high_coeff_text="not-used",
        high_power_text="not-used",
    )
    assert alpha == 0.0
    assert beta == 0.0
    assert high_coeff == 0.0
    assert high_power == 5


def test_membrane_optional_terms_parse_selected_values() -> None:
    alpha, beta, high_coeff, high_power = resolve_optional_membrane_terms(
        {_TERM_ALPHA, _TERM_BETA, _TERM_HIGH},
        alpha_text="0.25",
        beta_text="-0.5",
        high_coeff_text="2.0",
        high_power_text="7",
    )
    assert alpha == pytest.approx(0.25)
    assert beta == pytest.approx(-0.5)
    assert high_coeff == pytest.approx(2.0)
    assert high_power == 7


def test_membrane_optional_terms_reject_invalid_power_when_selected() -> None:
    with pytest.raises(ValueError):
        resolve_optional_membrane_terms(
            {_TERM_HIGH},
            alpha_text="0.0",
            beta_text="0.0",
            high_coeff_text="1.0",
            high_power_text="1",
        )
