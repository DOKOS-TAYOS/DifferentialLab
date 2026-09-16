"""Typed data structures for scalar and vector finite-difference PDE solvers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

import numpy as np

BC_DIRICHLET = "dirichlet"
BC_NEUMANN = "neumann"
BC_ROBIN = "robin"

BoundaryKind: TypeAlias = Literal["dirichlet", "neumann", "robin"]
BoundaryValueFunction: TypeAlias = Callable[[float, float], float]
BoundaryData: TypeAlias = float | np.ndarray | BoundaryValueFunction

# (u_xx, u_xy, u_yy, u_x, u_y, u, affine constant)
PDECoefficients: TypeAlias = tuple[float, float, float, float, float, float, float]
PDECoefficientProvider: TypeAlias = Callable[[float, float, dict[str, float]], PDECoefficients]
VectorPDEResidual: TypeAlias = Callable[..., np.ndarray]


@dataclass(frozen=True)
class PDEBoundaryCondition:
    """One scalar boundary condition.

    ``value`` is the prescribed solution for Dirichlet, ``du/dn`` for
    Neumann, and ``gamma`` for Robin. Robin follows
    ``alpha*u + beta*du/dn = gamma``. Boundary data may be a scalar, a full
    ``(ny, nx)`` array, or a callable ``value(x, y)``.
    """

    kind: BoundaryKind
    value: BoundaryData = 0.0
    alpha: BoundaryData = 0.0
    beta: BoundaryData = 0.0

    @classmethod
    def dirichlet(cls, value: BoundaryData = 0.0) -> PDEBoundaryCondition:
        """Create a Dirichlet condition ``u = value``."""
        return cls(BC_DIRICHLET, value)

    @classmethod
    def neumann(cls, value: BoundaryData = 0.0) -> PDEBoundaryCondition:
        """Create an outward-normal Neumann condition ``du/dn = value``."""
        return cls(BC_NEUMANN, value)

    @classmethod
    def robin(
        cls,
        alpha: BoundaryData,
        beta: BoundaryData,
        gamma: BoundaryData,
    ) -> PDEBoundaryCondition:
        """Create a Robin condition ``alpha*u + beta*du/dn = gamma``."""
        return cls(BC_ROBIN, gamma, alpha, beta)


@dataclass(frozen=True)
class PDEBoundaryConditions:
    """Structured scalar-2D boundary configuration.

    Edge conditions apply to rectangular domains. At a mixed corner,
    Dirichlet fixes the corner while Neumann/Robin apply to the open edge.
    Two Dirichlet values at a corner must agree. A corner reached by two
    non-Dirichlet edges is rejected because this pointwise representation has
    no unique grid normal.

    For an arbitrary mask, set ``contour`` and leave the four edges unset.
    Periodic axes are available only without a mask and use non-duplicated
    upper endpoints.
    """

    left: PDEBoundaryCondition | None = None
    right: PDEBoundaryCondition | None = None
    bottom: PDEBoundaryCondition | None = None
    top: PDEBoundaryCondition | None = None
    contour: PDEBoundaryCondition | None = None
    periodic_x: bool = False
    periodic_y: bool = False


ComponentBoundaryCondition: TypeAlias = PDEBoundaryCondition | Sequence[PDEBoundaryCondition | None]


@dataclass(frozen=True)
class VectorPDEBoundaryConditions:
    """Boundary configuration for a vector PDE system.

    Each edge or contour accepts either one :class:`PDEBoundaryCondition`,
    which is explicitly shared by every component, or a sequence whose length
    is exactly the system component count. A length-one sequence is not
    broadcast. On rectangular edges, use ``None`` in a component sequence to
    request the same default zero-Dirichlet behavior as an omitted scalar
    edge. A masked contour must provide a condition for every component.

    Periodicity is shared by the system because all components use one grid
    and one domain mask.
    """

    left: ComponentBoundaryCondition | None = None
    right: ComponentBoundaryCondition | None = None
    bottom: ComponentBoundaryCondition | None = None
    top: ComponentBoundaryCondition | None = None
    contour: ComponentBoundaryCondition | None = None
    periodic_x: bool = False
    periodic_y: bool = False


@dataclass(frozen=True)
class VectorPDECoefficients:
    """Coefficients of an ``m``-component linear PDE system.

    The six operator fields have shape ``(m, m)``. Rows select the residual
    equation and columns select the differentiated solution component.
    ``constant`` has shape ``(m,)``.
    """

    fxx: np.ndarray
    fxy: np.ndarray
    fyy: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    f: np.ndarray
    constant: np.ndarray


VectorPDECoefficientProvider: TypeAlias = Callable[
    [float, float, dict[str, float]], VectorPDECoefficients
]


@dataclass(frozen=True)
class PDEDiagnostics:
    """Numerical evidence for an assembled scalar PDE solve.

    Residual values describe the unknown-point sparse system ``A @ u = b``.
    The condition estimate is populated only for systems within a fixed small
    dense-diagnostic bound.
    """

    discrete_residual_l2: float
    discrete_residual_linf: float
    relative_residual_l2: float
    matrix_shape: tuple[int, int]
    nnz: int
    condition_estimate: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VectorPDEDiagnostics:
    """Global and per-equation evidence for a vector PDE sparse solve."""

    discrete_residual_l2: float
    discrete_residual_linf: float
    relative_residual_l2: float
    component_residual_l2: tuple[float, ...]
    component_residual_linf: tuple[float, ...]
    component_relative_residual_l2: tuple[float, ...]
    matrix_shape: tuple[int, int]
    nnz: int
    condition_estimate: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass
class PDESolution:
    """Container for a scalar PDE solution and its optional diagnostics.

    ``u`` has shape ``(ny, nx)`` and is NaN outside an arbitrary mask. ``grid``
    contains ``(x, y)``; a periodic axis omits its duplicated upper endpoint.
    """

    grid: tuple[np.ndarray, ...]
    u: np.ndarray
    success: bool
    message: str
    n_eval: int = 0
    mask: np.ndarray | None = None
    diagnostics: PDEDiagnostics | None = None


@dataclass
class VectorPDESolution:
    """Solution of a linear vector PDE system on one shared 2D grid.

    ``u`` always has public shape ``(m, ny, nx)``. Sparse unknowns use stable
    component-major indexing: ``component * n_unknown_points + point_index``.
    Values outside an arbitrary shared mask are NaN.
    """

    grid: tuple[np.ndarray, np.ndarray]
    u: np.ndarray
    success: bool
    message: str
    n_eval: int = 0
    mask: np.ndarray | None = None
    diagnostics: VectorPDEDiagnostics | None = None
