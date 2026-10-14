"""Operation-family discovery used by HTTP, workers and recipe execution."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Type, Tuple, Optional

from pydantic import BaseModel

from measure_api.v2.models import CalculationRequest, ServiceSettings, OutputSpec, FileReference
from measure_api.v2.files import SharedFiles


@dataclass
class CalculationContext:
    """Services available to one isolated calculation, with no HTTP dependency."""

    settings: ServiceSettings
    operation_id: str
    trace_id: str
    commit_guard: Any = None
    _files: Any = field(default=None, init=False, repr=False)

    @property
    def files(self) -> SharedFiles:
        """Return shared-file access governed by this deployment's settings.

        Returns:
            Shared-file accessor governed by this operation's deployment and commit guard.
        """
        if self._files is None:
            self._files = SharedFiles(self.settings, self.commit_guard)
        return self._files

    def stage(self, stage_id: str) -> Tuple[CalculationContext, OutputSpec]:
        """Create cancellable temporary stage output, never a parent business commit.

        Args:
            stage_id: Unique safe identifier within the parent computation.

        Returns:
            A child context and its temporary output specification. Copy any needed
            payloads into the parent's final commit; stage descriptors expire at termination.
        """
        if not stage_id or not stage_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Stage identifier must be alphanumeric with hyphens or underscores")
        if self.commit_guard is None or self.commit_guard.staging_directory is None:
            raise ValueError("Temporary stages require an operation runtime context")
        roots = dict(self.settings.shared_roots)
        roots["__stages__"] = Path(self.commit_guard.staging_directory).resolve()
        writable = self.settings.writable_roots
        child_settings = self.settings.model_copy(update={
            "shared_roots": roots,
            "writable_roots": None if writable is None else [*writable, "__stages__"],
        })
        guard = replace(self.commit_guard, finalize=False)
        context = CalculationContext(child_settings, stage_id, self.trace_id, guard)
        output = OutputSpec(directory=FileReference(root="__stages__", path="."))
        return context, output


@dataclass(frozen=True)
class Operation:
    """One typed public calculation and its directly callable implementation."""

    path: str
    request_model: Type[CalculationRequest]
    result_model: Type[BaseModel]
    execute: Callable[[Any, CalculationContext], Any]
    summary: str
    family: str
    output_ports: Dict[str, str] = field(default_factory=dict)

    prepare: Optional[Callable[[Any, SharedFiles], Any]] = None

    @property
    def operation_id(self) -> str:
        """Stable semantic identifier derived from its public route.

        Returns:
            Stable semantic calculation identifier derived from the registered route.
        """
        return self.path.strip("/").replace("/", ".").replace(":", "_")


@lru_cache(maxsize=1)
def operations() -> Dict[str, Operation]:
    """Discover family modules; adding a family requires no central router edits.

    Returns:
        Registered semantic identifiers mapped to typed calculation implementations.
    """
    from measure_api.v2 import families

    result = {}
    for item in sorted(pkgutil.iter_modules(families.__path__), key=lambda item: item.name):
        module = importlib.import_module(f"{families.__name__}.{item.name}")
        for operation in getattr(module, "OPERATIONS", []):
            if operation.operation_id in result:
                raise ValueError(f"Duplicate operation {operation.operation_id}")
            result[operation.operation_id] = operation
    return result
