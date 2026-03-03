"""Notation translation between user-facing f[...] and internal flat y[j] arrays.

The user writes expressions using ``f[i,k]`` (vector ODE), ``f[k]`` (scalar ODE /
difference), or ``f[a,b,...]`` (PDE derivatives).  Internally, scipy's solve_ivp
operates on a flat state vector ``y``.  This module bridges the two representations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

EquationKind = Literal["ode", "vector_ode", "difference", "pde"]


@dataclass(frozen=True)
class FNotation:
    """Describes the notation context for a particular equation.

    Attributes:
        kind: One of ``"ode"``, ``"vector_ode"``, ``"difference"``, ``"pde"``.
        n_components: Number of components (1 for scalar ODE / PDE / difference).
        order: ODE order per component, or max derivative order for PDE.
        n_independent_vars: Number of independent variables (1 for ODE, >=1 for PDE).
    """

    kind: EquationKind
    n_components: int = 1
    order: int = 1
    n_independent_vars: int = 1
    component_orders: tuple[int, ...] = ()

    def state_size(self) -> int:
        """Total number of entries in the flat state vector.

        Returns:
            Number of entries in the flat ``y`` array.
        """
        if self.component_orders:
            return sum(self.component_orders)
        return self.n_components * self.order


# ---------------------------------------------------------------------------
# f[...] -> y[...] rewriting
# ---------------------------------------------------------------------------

# Matches:  f  |  f[...]  (greedy bracket content)
_F_TOKEN = re.compile(
    r"""
    \bf                  # word-boundary then literal 'f'
    (?:\[([^\]]*)\])?    # optional bracketed content (group 1)
    (?![(\w])            # NOT followed by '(' or word char (avoid matching func names)
    """,
    re.VERBOSE,
)


def _rewrite_match_ode_scalar(m: re.Match, order: int) -> str:
    """Rewrite a single f-token for scalar ODE / difference."""
    bracket = m.group(1)
    if bracket is None:
        # bare 'f' -> y[0]
        return "y[0]"
    bracket = bracket.strip()
    if not bracket:
        return "y[0]"
    # f[k] -> y[k]
    return f"y[{bracket}]"


def _rewrite_match_vector_ode(m: re.Match, notation: FNotation) -> str:
    """Rewrite a single f-token for vector ODE."""
    bracket = m.group(1)
    if bracket is None:
        return "y[0]"
    bracket = bracket.strip()
    if not bracket:
        return "y[0]"

    parts = [p.strip() for p in bracket.split(",")]
    if len(parts) == 1:
        # f[k] -> y[0*order + k]  (implicit component 0)
        k_expr = parts[0]
        if notation.component_orders:
            # component 0 always starts at offset 0
            return f"y[{k_expr}]"
        return f"y[{k_expr}]"
    if len(parts) == 2:
        i_expr, k_expr = parts
        # f[i,k] -> y[i*order + k]
        # If the indices are literal integers, compute directly for clarity.
        try:
            i_val = int(i_expr)
            k_val = int(k_expr)
            if notation.component_orders:
                offset = sum(notation.component_orders[:i_val])
                return f"y[{offset + k_val}]"
            flat = i_val * notation.order + k_val
            return f"y[{flat}]"
        except ValueError:
            # Symbolic indices (e.g. loop variable i)
            if notation.component_orders:
                # Fallback: cannot compute with heterogeneous orders symbolically.
                # Assume uniform order for symbolic case.
                return f"y[({i_expr})*{notation.order}+({k_expr})]"
            return f"y[({i_expr})*{notation.order}+({k_expr})]"
    # More than 2 indices is invalid for vector ODE
    return m.group(0)  # leave unchanged


def rewrite_f_expression(expression: str, notation: FNotation) -> str:
    """Rewrite user-facing ``f[...]`` tokens to internal ``y[...]`` form.

    Args:
        expression: Python expression using ``f``, ``f[k]``, ``f[i,k]``, etc.
        notation: Context describing the equation type and dimensions.

    Returns:
        Equivalent expression with ``y[...]`` indexing suitable for the solver.
    """
    if notation.kind in ("ode", "difference"):
        return _F_TOKEN.sub(lambda m: _rewrite_match_ode_scalar(m, notation.order), expression)

    if notation.kind == "vector_ode":
        return _F_TOKEN.sub(lambda m: _rewrite_match_vector_ode(m, notation), expression)

    if notation.kind == "pde":
        # PDE: f alone -> y[0] (the solution value).
        # f[a,b,...] derivative references are handled by the PDE parser (Phase 9).
        # For now, only rewrite bare f -> y[0].
        return _F_TOKEN.sub(lambda m: _rewrite_match_ode_scalar(m, notation.order), expression)

    return expression


# ---------------------------------------------------------------------------
# Flat index -> label (for plots, CSV headers)
# ---------------------------------------------------------------------------

_PRIME_SYMBOLS = ["", "\u2032", "\u2033", "\u2034"]  # '', ′, ″, ‴
_SUBSCRIPT_DIGITS = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"


def _prime_str(k: int) -> str:
    """Return a prime string for derivative order k."""
    if k < len(_PRIME_SYMBOLS):
        return _PRIME_SYMBOLS[k]
    return f"({k})"


def _subscript(n: int) -> str:
    """Return subscript digits for integer n (e.g. 12 → '₁₂')."""
    if 0 <= n < len(_SUBSCRIPT_DIGITS):
        return _SUBSCRIPT_DIGITS[n]
    return "".join(
        _SUBSCRIPT_DIGITS[int(d)] if d.isdigit() else d for d in str(n)
    )


def flat_index_to_label(j: int, notation: FNotation) -> str:
    """Convert flat state-vector index *j* to a human-readable label.

    Args:
        j: Index into the flat ``y`` array.
        notation: Equation context.

    Returns:
        A string like ``"f"``, ``"f\u2032"``, ``"f\u2032\u2081"``, etc.
    """
    if notation.kind in ("ode", "difference"):
        if notation.order == 1:
            return "f"
        return f"f{_prime_str(j)}"

    if notation.kind == "vector_ode":
        if notation.component_orders:
            # Heterogeneous orders: find component
            cumsum = 0
            for comp, comp_order in enumerate(notation.component_orders):
                if j < cumsum + comp_order:
                    k = j - cumsum
                    return f"f{_prime_str(k)}{_subscript(comp)}"
                cumsum += comp_order
            return f"y[{j}]"
        comp = j // notation.order
        k = j % notation.order
        return f"f{_prime_str(k)}{_subscript(comp)}"

    if notation.kind == "pde":
        if j == 0:
            return "f"
        return f"y[{j}]"

    return f"y[{j}]"


def generate_derivative_labels(notation: FNotation) -> list[str]:
    """Generate labels for every entry in the flat state vector.

    Args:
        notation: Equation context.

    Returns:
        List of human-readable labels, length == ``notation.state_size()``.
    """
    return [flat_index_to_label(j, notation) for j in range(notation.state_size())]


def generate_phase_space_options(notation: FNotation) -> list[tuple[str, int | None]]:
    """Generate options for phase-space axis selectors.

    Each option is ``(label, flat_index)`` where flat_index is ``None`` for the
    independent variable ``x``.

    Args:
        notation: Equation context.

    Returns:
        List of ``(label, flat_index_or_None)`` pairs.
    """
    x_label = "n" if notation.kind == "difference" else "x"
    options: list[tuple[str, int | None]] = [(x_label, None)]
    for j in range(notation.state_size()):
        options.append((flat_index_to_label(j, notation), j))
    return options
