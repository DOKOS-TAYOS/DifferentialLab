"""Tests for the f-notation translation layer."""


from solver.notation import (
    FNotation,
    _flat_index_to_label,
    _rewrite_f_expression,
    generate_derivative_labels,
    generate_phase_space_options,
)

# ── _rewrite_f_expression: ODE scalar ──────────────────────────────────


class TestRewriteODEScalar:
    """ODE scalar: f[k] -> y[k], bare f -> y[0]."""

    nota = FNotation(kind="ode", n_components=1, order=2)

    def test_bare_f(self):
        assert _rewrite_f_expression("f", self.nota) == "y[0]"

    def test_f_zero(self):
        assert _rewrite_f_expression("f[0]", self.nota) == "y[0]"

    def test_f_one(self):
        assert _rewrite_f_expression("f[1]", self.nota) == "y[1]"

    def test_expression(self):
        result = _rewrite_f_expression("-omega**2 * f[0]", self.nota)
        assert result == "-omega**2 * y[0]"

    def test_multiple_refs(self):
        result = _rewrite_f_expression("-2*gamma*f[1] - omega**2*f[0]", self.nota)
        assert result == "-2*gamma*y[1] - omega**2*y[0]"

    def test_bare_f_in_expression(self):
        result = _rewrite_f_expression("k * f", self.nota)
        assert result == "k * y[0]"

    def test_no_match_function_names(self):
        """Should not rewrite 'floor', 'float', etc."""
        result = _rewrite_f_expression("floor(f[0])", self.nota)
        assert result == "floor(y[0])"

    def test_no_match_function_call(self):
        """f followed by ( is a function call, not our f variable."""
        result = _rewrite_f_expression("func(x)", self.nota)
        assert result == "func(x)"

    def test_empty_brackets(self):
        result = _rewrite_f_expression("f[]", self.nota)
        assert result == "y[0]"


# ── _rewrite_f_expression: Difference ──────────────────────────────────


class TestRewriteDifference:
    nota = FNotation(kind="difference", n_components=1, order=2)

    def test_f_zero(self):
        assert _rewrite_f_expression("f[0]", self.nota) == "y[0]"

    def test_f_one(self):
        assert _rewrite_f_expression("f[1]", self.nota) == "y[1]"

    def test_expression(self):
        result = _rewrite_f_expression("f[1] + f[0]", self.nota)
        assert result == "y[1] + y[0]"


# ── _rewrite_f_expression: Vector ODE ─────────────────────────────────


class TestRewriteVectorODE:
    """Vector ODE: f[i,k] -> y[i*order + k]."""

    nota = FNotation(kind="vector_ode", n_components=3, order=2)

    def test_f_ik_literal(self):
        assert _rewrite_f_expression("f[0,0]", self.nota) == "y[0]"
        assert _rewrite_f_expression("f[0,1]", self.nota) == "y[1]"
        assert _rewrite_f_expression("f[1,0]", self.nota) == "y[2]"
        assert _rewrite_f_expression("f[1,1]", self.nota) == "y[3]"
        assert _rewrite_f_expression("f[2,0]", self.nota) == "y[4]"
        assert _rewrite_f_expression("f[2,1]", self.nota) == "y[5]"

    def test_single_index_implicit_component(self):
        """f[k] with single index -> y[k] (component 0 implicit)."""
        result = _rewrite_f_expression("f[1]", self.nota)
        assert result == "y[1]"

    def test_bare_f(self):
        assert _rewrite_f_expression("f", self.nota) == "y[0]"

    def test_expression(self):
        result = _rewrite_f_expression(
            "-omega**2 * f[0,0] + k * (f[1,0] - f[0,0])", self.nota
        )
        assert result == "-omega**2 * y[0] + k * (y[2] - y[0])"

    def test_symbolic_indices(self):
        """f[i,0] with symbolic i -> y[(i)*2+(0)]."""
        result = _rewrite_f_expression("f[i,0]", self.nota)
        assert result == "y[(i)*2+(0)]"

    def test_modular_symbolic(self):
        """f[(i+1)%n,0] -> y[((i+1)%n)*2+(0)]."""
        result = _rewrite_f_expression("f[(i+1)%n,0]", self.nota)
        assert result == "y[((i+1)%n)*2+(0)]"


# ── _rewrite_f_expression: Heterogeneous orders ───────────────────────


class TestRewriteHeterogeneousOrders:
    """Vector ODE where each component has a different order."""

    nota = FNotation(
        kind="vector_ode",
        n_components=2,
        order=2,
        component_orders=(3, 2),  # comp 0: order 3, comp 1: order 2
    )

    def test_f00(self):
        # Component 0 starts at offset 0
        assert _rewrite_f_expression("f[0,0]", self.nota) == "y[0]"

    def test_f02(self):
        assert _rewrite_f_expression("f[0,2]", self.nota) == "y[2]"

    def test_f10(self):
        # Component 1 starts at offset 3 (sum of component_orders[:1])
        assert _rewrite_f_expression("f[1,0]", self.nota) == "y[3]"

    def test_f11(self):
        assert _rewrite_f_expression("f[1,1]", self.nota) == "y[4]"

    def test_state_size(self):
        assert self.nota.state_size() == 5


# ── _flat_index_to_label ──────────────────────────────────────────────


class TestFlatIndexToLabel:
    def test_ode_scalar_order1(self):
        nota = FNotation(kind="ode", order=1)
        assert _flat_index_to_label(0, nota) == "f"

    def test_ode_scalar_order2(self):
        nota = FNotation(kind="ode", order=2)
        assert _flat_index_to_label(0, nota) == "f"
        assert _flat_index_to_label(1, nota) == "f\u2032"

    def test_ode_scalar_order3(self):
        nota = FNotation(kind="ode", order=3)
        assert _flat_index_to_label(0, nota) == "f"
        assert _flat_index_to_label(1, nota) == "f\u2032"
        assert _flat_index_to_label(2, nota) == "f\u2033"

    def test_vector_ode(self):
        nota = FNotation(kind="vector_ode", n_components=2, order=2)
        assert _flat_index_to_label(0, nota) == "f\u2080"
        assert _flat_index_to_label(1, nota) == "f\u2032\u2080"
        assert _flat_index_to_label(2, nota) == "f\u2081"
        assert _flat_index_to_label(3, nota) == "f\u2032\u2081"

    def test_difference(self):
        nota = FNotation(kind="difference", order=2)
        assert _flat_index_to_label(0, nota) == "f"
        assert _flat_index_to_label(1, nota) == "f\u2032"


# ── generate_derivative_labels ───────────────────────────────────────


class TestGenerateDerivativeLabels:
    def test_scalar_order2(self):
        nota = FNotation(kind="ode", order=2)
        labels = generate_derivative_labels(nota)
        assert labels == ["f", "f\u2032"]

    def test_vector_2comp_order2(self):
        nota = FNotation(kind="vector_ode", n_components=2, order=2)
        labels = generate_derivative_labels(nota)
        assert labels == ["f\u2080", "f\u2032\u2080", "f\u2081", "f\u2032\u2081"]


# ── generate_phase_space_options ─────────────────────────────────────


class TestPhaseSpaceOptions:
    def test_ode_scalar(self):
        nota = FNotation(kind="ode", order=2)
        options = generate_phase_space_options(nota)
        assert options[0] == ("x", None)
        assert options[1] == ("f", 0)
        assert options[2] == ("f\u2032", 1)

    def test_difference(self):
        nota = FNotation(kind="difference", order=1)
        options = generate_phase_space_options(nota)
        assert options[0] == ("n", None)
