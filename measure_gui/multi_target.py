"""Backwards-compatibility shim.

The multi-target engine used to live here as a second, divergent copy.  It was
promoted to :mod:`measure.multi_target_workflow` so that the GUI, API v1 and the
v2 engine all share a single implementation (and so no API layer needs to import
from a GUI package).

Import from :mod:`measure.multi_target_workflow` directly in new code; this
module only re-exports for existing callers.
"""

from __future__ import annotations

# Engine
from measure.multi_target_workflow import (  # noqa: F401
    MultiTargetWorkflow,
    TargetResult,
    _VirtualPoint,
    _build_profile_image,
    _compute_target_box_corners,
    _draw_geometric_result,
    _draw_result_on_image,
    _map_result_to_original,
    _references_label,
    _result_to_dict,
)

# Object-type registry (previously a private table in this module)
from measure.object_specs import (  # noqa: F401
    OBJECT_SPECS,
    ObjectSpec,
    factory_table,
    get_spec,
)
from measure.object_specs import defaults_for as _get_defaults

_OBJECT_FACTORIES = factory_table()

# Result formatting (previously defined in this module)
from measure.result_format import (  # noqa: F401
    _fmt_length,
    _format_geometric_result,
    _format_result_dict,
    _result_dict_line_length,
)

__all__ = [
    "MultiTargetWorkflow",
    "TargetResult",
    "_OBJECT_FACTORIES",
    "_get_defaults",
]