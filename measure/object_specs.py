"""Single source of truth for measurement object types.

Every layer derives its object-type metadata from :data:`OBJECT_SPECS`:

* the GUI / v1 API build objects with ``spec.factory`` and validate with
  ``spec.required`` + ``spec.deps`` + ``spec.defaults``;
* the v2 engine derives its ``_OBJECT_CONTRACTS`` (required / defaults /
  input_types / result_type);
* ``measure_api.schemas`` derives ``PRIMITIVE_TYPES`` / ``COMPOSED_TYPES`` /
  ``COMPOSED_DEPS``;
* ``measure.measure_workflow._OBJECT_TYPE_REGISTRY`` is cross-checked against
  the same type set by the test suite.

Adding a new measurement object therefore means adding one entry here plus its
``MeasureObject`` subclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Mapping, Tuple

import numpy as np

from measure.measure_workflow import (
    EdgePairObject,
    EdgePointObject,
    FitCircleObject,
    FitLineObject,
    PointCircleDistanceObject,
    PointLineDistanceObject,
    TemplateMatchPointObject,
    TwoLinesAngleObject,
    TwoPointsDistanceObject,
    TwoPointsLineObject,
)


# ===========================================================================
# Factories — (label, **params) -> MeasureObject
# ===========================================================================

def _make_edge_point(label: str, **p) -> EdgePointObject:
    return EdgePointObject(
        label=label,
        row=p["row"],
        col=p["col"],
        angle=p["angle"],
        length1=p["length1"],
        length2=p["length2"],
        sigma=p.get("sigma", 1.0),
        threshold=p.get("threshold", 30.0),
        transition="all",
        select="first",
        interpolation=p.get("interpolation", "linear"),
    )


def _make_edge_pair(label: str, **p) -> EdgePairObject:
    return EdgePairObject(
        label=label,
        row=p["row"],
        col=p["col"],
        angle=p["angle"],
        length1=p["length1"],
        length2=p["length2"],
        sigma=p.get("sigma", 1.0),
        threshold=p.get("threshold", 30.0),
        transition="all",
        select="endpoints",
        interpolation=p.get("interpolation", "linear"),
    )


def _make_fit_line(label: str, **p) -> FitLineObject:
    return FitLineObject(
        label=label,
        start=p["start"],
        end=p["end"],
        measure_length1=p["measure_length1"],
        measure_length2=p["measure_length2"],
        num_measures=p.get("num_measures", 10),
        sigma=p.get("sigma", 1.0),
        threshold=p.get("threshold", 30.0),
        transition=p.get("transition", "all"),
    )


def _make_fit_circle(label: str, **p) -> FitCircleObject:
    return FitCircleObject(
        label=label,
        center=p["center"],
        radius=p["radius"],
        measure_length1=p["measure_length1"],
        measure_length2=p["measure_length2"],
        radius_min=p.get("radius_min"),
        radius_max=p.get("radius_max"),
        num_measures=p.get("num_measures", 12),
        sigma=p.get("sigma", 1.0),
        threshold=p.get("threshold", 30.0),
        transition=p.get("transition", "all"),
        start_phi=p.get("start_phi", 0.0),
        end_phi=p.get("end_phi", 2 * np.pi),
    )


def _make_template_match_point(label: str, **p) -> TemplateMatchPointObject:
    return TemplateMatchPointObject(
        label=label,
        row=p["row"],
        col=p["col"],
        template_size=p.get("template_size", 40),
        preprocessor_type=p.get("preprocessor_type", "raw"),
        match_score_threshold=p.get("match_score_threshold", 0.5),
        angle_range_half=p.get("angle_range_half", 15.0),
        angle_step=p.get("angle_step", 1.0),
        use_subpixel=p.get("use_subpixel", True),
        _template_point=p.get("_template_point", None),
    )


def _make_two_points_line(label: str, **p) -> TwoPointsLineObject:
    return TwoPointsLineObject(
        label=label,
        point_a_label=p["point_a_label"],
        point_b_label=p["point_b_label"],
    )


def _make_two_points_distance(label: str, **p) -> TwoPointsDistanceObject:
    return TwoPointsDistanceObject(
        label=label,
        point_a_label=p["point_a_label"],
        point_b_label=p["point_b_label"],
    )


def _make_point_line_distance(label: str, **p) -> PointLineDistanceObject:
    return PointLineDistanceObject(
        label=label,
        point_label=p["point_label"],
        line_label=p["line_label"],
    )


def _make_two_lines_angle(label: str, **p) -> TwoLinesAngleObject:
    return TwoLinesAngleObject(
        label=label,
        line_a_label=p["line_a_label"],
        line_b_label=p["line_b_label"],
    )


def _make_point_circle_distance(label: str, **p) -> PointCircleDistanceObject:
    return PointCircleDistanceObject(
        label=label,
        point_label=p["point_label"],
        circle_label=p["circle_label"],
    )


# ===========================================================================
# Object specs
# ===========================================================================


@dataclass(frozen=True)
class ObjectSpec:
    """Metadata describing one measurement object type.

    Args:
        object_type: Public type name used by every API.
        factory: ``(label, **params) -> MeasureObject`` constructor.
        required: Required non-dependency parameters.
        defaults: Optional parameters and their default values.
        deps: ``(param_name, expected_result_type)`` pairs describing composed
            inputs in label form.  Empty for primitives.
        result_type: One of ``point``/``line``/``circle``/``distance``/``angle``.
    """

    object_type: str
    factory: Callable[..., Any]
    required: Tuple[str, ...]
    defaults: Mapping[str, Any]
    deps: Tuple[Tuple[str, str], ...]
    result_type: str

    @property
    def is_composed(self) -> bool:
        """True when this object is resolved from other objects' results."""
        return bool(self.deps)

    @property
    def input_types(self) -> Tuple[str, ...]:
        """Result types required from the composed inputs, in dependency order."""
        return tuple(expected for _, expected in self.deps)

    @property
    def dep_param_names(self) -> Tuple[str, ...]:
        """Parameter names carrying composed inputs in label form."""
        return tuple(name for name, _ in self.deps)

    @property
    def param_keys(self) -> Tuple[str, ...]:
        """All accepted parameters, in declaration order."""
        return self.required + self.dep_param_names + tuple(self.defaults)


_SPECS: Tuple[ObjectSpec, ...] = (
    ObjectSpec(
        "EdgePoint", _make_edge_point,
        ("row", "col", "angle", "length1", "length2"),
        {"sigma": 1.0, "threshold": 30.0, "interpolation": "linear"},
        (), "point",
    ),
    ObjectSpec(
        "EdgePair", _make_edge_pair,
        ("row", "col", "angle", "length1", "length2"),
        {"sigma": 1.0, "threshold": 30.0, "interpolation": "linear"},
        (), "point",
    ),
    ObjectSpec(
        "FitLine", _make_fit_line,
        ("start", "end", "measure_length1", "measure_length2"),
        {"num_measures": 10, "sigma": 1.0, "threshold": 30.0,
         "transition": "all"},
        (), "line",
    ),
    ObjectSpec(
        "FitCircle", _make_fit_circle,
        ("center", "radius", "measure_length1", "measure_length2"),
        {"radius_min": None, "radius_max": None, "num_measures": 12,
         "sigma": 1.0, "threshold": 30.0, "transition": "all",
         "start_phi": 0.0, "end_phi": 2 * np.pi},
        (), "circle",
    ),
    ObjectSpec(
        "TemplateMatchPoint", _make_template_match_point,
        ("row", "col"),
        {"template_size": 40, "preprocessor_type": "raw",
         "match_score_threshold": 0.5, "angle_range_half": 15.0,
         "angle_step": 1.0, "use_subpixel": True,
         "pyramid_decimate": 0, "pyramid_max_template_size": 400},
        (), "point",
    ),
    ObjectSpec(
        "TwoPointsLine", _make_two_points_line, (),
        {}, (("point_a_label", "point"), ("point_b_label", "point")), "line",
    ),
    ObjectSpec(
        "TwoPointsDistance", _make_two_points_distance, (),
        {}, (("point_a_label", "point"), ("point_b_label", "point")),
        "distance",
    ),
    ObjectSpec(
        "PointLineDistance", _make_point_line_distance, (),
        {}, (("point_label", "point"), ("line_label", "line")), "distance",
    ),
    ObjectSpec(
        "TwoLinesAngle", _make_two_lines_angle, (),
        {}, (("line_a_label", "line"), ("line_b_label", "line")), "angle",
    ),
    ObjectSpec(
        "PointCircleDistance", _make_point_circle_distance, (),
        {}, (("point_label", "point"), ("circle_label", "circle")), "distance",
    ),
)

OBJECT_SPECS: Mapping[str, ObjectSpec] = {s.object_type: s for s in _SPECS}


# ===========================================================================
# Derived views — the only supported way to read the registry
# ===========================================================================


def get_spec(object_type: str) -> ObjectSpec:
    """Return the spec for ``object_type``.

    Raises:
        KeyError: When the type is unknown (message lists known types).
    """
    try:
        return OBJECT_SPECS[object_type]
    except KeyError:
        raise KeyError(
            f"Unknown object_type: {object_type!r}. "
            f"Known types: {list(OBJECT_SPECS)}"
        ) from None


def object_types() -> Tuple[str, ...]:
    """All known object type names, in declaration order."""
    return tuple(OBJECT_SPECS)


def primitive_types() -> FrozenSet[str]:
    """Types measured directly on the patch."""
    return frozenset(t for t, s in OBJECT_SPECS.items() if not s.is_composed)


def composed_types() -> FrozenSet[str]:
    """Types resolved from other objects' results."""
    return frozenset(t for t, s in OBJECT_SPECS.items() if s.is_composed)


def defaults_for(object_type: str) -> Dict[str, Any]:
    """A mutable copy of the default parameters for ``object_type``."""
    return dict(get_spec(object_type).defaults)


def composed_deps() -> Dict[str, list]:
    """``{type: [dependency parameter name, ...]}`` (schemas.py compatibility)."""
    return {t: list(s.dep_param_names)
            for t, s in OBJECT_SPECS.items() if s.is_composed}


def factory_table() -> Dict[str, Tuple[Callable[..., Any], Tuple[str, ...]]]:
    """``{type: (factory, param_keys)}`` (GUI ``_OBJECT_FACTORIES`` shape)."""
    return {t: (s.factory, s.param_keys) for t, s in OBJECT_SPECS.items()}
