"""Registry and lazy loader for complex problem plugins."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from complex_problems.base import ComplexProblem, ProblemDescriptor

if TYPE_CHECKING:
    from tkinter import Tk, Toplevel


@dataclass(frozen=True, slots=True)
class ProblemRegistration:
    """Module path and symbol name for a problem plugin."""

    module_path: str
    symbol_name: str = "PROBLEM"


class ProblemRegistry:
    """Lazy plugin registry for complex problems."""

    def __init__(self, registrations: tuple[ProblemRegistration, ...]) -> None:
        self._registrations = registrations
        self._plugins: dict[str, ComplexProblem] | None = None

    def _load_plugin(self, registration: ProblemRegistration) -> ComplexProblem:
        module = importlib.import_module(registration.module_path)
        plugin = getattr(module, registration.symbol_name, None)
        if plugin is None:
            raise RuntimeError(
                f"Problem module '{registration.module_path}' does not define "
                f"'{registration.symbol_name}'."
            )
        descriptor = getattr(plugin, "descriptor", None)
        open_dialog = getattr(plugin, "open_dialog", None)
        if not isinstance(descriptor, ProblemDescriptor) or not callable(open_dialog):
            raise RuntimeError(
                f"Plugin '{registration.module_path}.{registration.symbol_name}' "
                "does not implement the complex problem interface."
            )
        return plugin

    def _ensure_loaded(self) -> None:
        if self._plugins is not None:
            return
        plugins: dict[str, ComplexProblem] = {}
        for registration in self._registrations:
            plugin = self._load_plugin(registration)
            problem_id = plugin.descriptor.id
            if problem_id in plugins:
                raise RuntimeError(f"Duplicate complex problem id: '{problem_id}'")
            plugins[problem_id] = plugin
        self._plugins = plugins

    def get_descriptors(self) -> dict[str, ProblemDescriptor]:
        """Return descriptors keyed by problem id."""
        self._ensure_loaded()
        assert self._plugins is not None
        return {pid: plugin.descriptor for pid, plugin in self._plugins.items()}

    def open_problem_dialog(self, problem_id: str, parent: "Tk | Toplevel") -> None:
        """Open the dialog for a registered problem id."""
        self._ensure_loaded()
        assert self._plugins is not None
        plugin = self._plugins.get(problem_id)
        if plugin is None:
            raise KeyError(f"Unknown complex problem id: '{problem_id}'")
        plugin.open_dialog(parent)


class _LazyDescriptorMapping(Mapping[str, ProblemDescriptor]):
    """Mapping-like read-only view that defers plugin loading."""

    def __init__(self, registry: ProblemRegistry) -> None:
        self._registry = registry

    def __getitem__(self, key: str) -> ProblemDescriptor:
        return self._registry.get_descriptors()[key]

    def __iter__(self) -> "Iterator[str]":
        return iter(self._registry.get_descriptors())

    def __len__(self) -> int:
        return len(self._registry.get_descriptors())


_REGISTRATIONS: tuple[ProblemRegistration, ...] = (
    ProblemRegistration(module_path="complex_problems.coupled_oscillators.problem"),
    ProblemRegistration(module_path="complex_problems.membrane_2d.problem"),
    ProblemRegistration(module_path="complex_problems.nonlinear_waves.problem"),
    ProblemRegistration(module_path="complex_problems.schrodinger_td.problem"),
    ProblemRegistration(module_path="complex_problems.antenna_radiation.problem"),
    ProblemRegistration(module_path="complex_problems.aerodynamics_2d.problem"),
    ProblemRegistration(module_path="complex_problems.pipe_flow.problem"),
)

_REGISTRY = ProblemRegistry(_REGISTRATIONS)
PROBLEM_REGISTRY: "Mapping[str, ProblemDescriptor]" = _LazyDescriptorMapping(_REGISTRY)


def open_problem_dialog(problem_id: str, parent: "Tk | Toplevel") -> None:
    """Open a registered complex problem dialog by id."""
    _REGISTRY.open_problem_dialog(problem_id, parent)


def get_problem_descriptors() -> dict[str, ProblemDescriptor]:
    """Get all problem descriptors."""
    return _REGISTRY.get_descriptors()
