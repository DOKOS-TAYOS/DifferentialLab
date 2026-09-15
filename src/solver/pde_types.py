"""Typed data structures for scalar finite-difference PDE solvers."""

from __future__ import annotations

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
