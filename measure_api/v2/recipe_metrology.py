"""Map selected samples before fitting and composing pixel/physical observations."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from measure_api.v2.calibration_assets import (
    CameraCalibration,
    ScalarCalibration,
    calibration_digest,
    check_applicability,
    load_calibration,
)
from measure_api.v2.errors import ServiceError
from measure_api.v2.families.geometry import PointFitRequest, fit_circle, fit_line
from measure_api.v2.families.plane_stereo import PlaneCalibration, perspective
from measure_api.v2.models import Point, CoordinateMapping
from measure_api.v2.recipe_models import Recipe
from measure_api.v2.recipe_graph import check_metric_frame
from measure_api.v2.registry import CalculationContext


class SampleMapper:
    """A target patch to source and source to physical sample transformation chain."""

    def __init__(
        self,
        recipe: Recipe,
        context: CalculationContext,
        patch_to_image: Optional[np.ndarray] = None,
        image_to_source: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        canvas: bool = False,
    ) -> None:
        self.recipe = recipe
        self.context = context
        self.patch_to_image = np.eye(3) if patch_to_image is None else patch_to_image
        self.image_to_source = image_to_source or (lambda points: points)
        self.canvas = canvas
        self.asset = None
        if recipe.metric is not None:
            self.asset = load_calibration(recipe.metric.calibration, context)
            check_applicability(self.asset, recipe.metric.applicability)
            try:
                check_metric_frame(recipe, self.asset)
            except ValueError as error:
                raise ServiceError("recipe.metric_frame_mismatch", str(error)) from error
            if isinstance(self.asset, CameraCalibration):
                raise ServiceError(
                    "recipe.metric_unsupported",
                    "Camera intrinsics alone cannot establish physical metrology",
                )

    def physical_mapping(self) -> Optional[CoordinateMapping]:
        """Expose the exact source pixel to physical transform for upstream result drawing.

        Returns:
            Source-to-physical transform, or None when no metric asset was supplied.
        """
        if self.asset is None:
            return None
        source, destination = self.recipe.source_frame, self.recipe.metric.physical_frame
        unit = self.asset.unit
        if isinstance(self.asset, ScalarCalibration):
            scale = self.asset.parameters.scale_per_px
            matrix = np.diag([scale, scale, 1.0])
            kind, details = "similarity", {}
        elif self.canvas:
            matrix = np.array(self.asset.parameters.canvas.canvas_to_plane)
            kind, details = "projective", {}
        else:
            view = next(
                v
                for v in self.asset.parameters.views
                if v.camera.applicability.camera_id == self.recipe.metric.applicability.camera_id
            )
            camera = view.camera.parameters
            if np.max(np.abs(camera.distortion)) <= 1e-14:
                matrix = np.array(view.camera_to_plane)
                kind, details = "projective", {}
            else:
                return CoordinateMapping(
                    source_frame=source,
                    destination_frame=destination,
                    type="nonlinear",
                    unit=unit,
                    details={
                        "model": "opencv_pinhole_plane",
                        "camera_matrix": camera.camera_matrix,
                        "distortion": camera.distortion,
                        "corrected_to_physical": view.camera_to_plane,
                        "physical_to_corrected": view.plane_to_camera,
                        "calibration_sha256": calibration_digest(self.asset),
                    },
                )
        details["calibration_sha256"] = calibration_digest(self.asset)
        return CoordinateMapping(
            source_frame=source,
            destination_frame=destination,
            type=kind,
            matrix=matrix.tolist(),
            inverse_matrix=np.linalg.inv(matrix).tolist(),
            unit=unit,
            details=details,
        )

    def map_points(self, values: List[dict], physical: bool = False) -> List[dict]:
        if not values:
            return []
        points = np.array([[value["x"], value["y"]] for value in values], np.float64)
        points = perspective(points, self.patch_to_image)
        points = self.image_to_source(points)
        if physical:
            if self.asset is None:
                raise ServiceError(
                    "recipe.metric_missing", "Physical observations require calibration"
                )
            if isinstance(self.asset, ScalarCalibration):
                points = points * self.asset.parameters.scale_per_px
            elif isinstance(self.asset, PlaneCalibration):
                if self.canvas:
                    points = perspective(
                        points, np.array(self.asset.parameters.canvas.canvas_to_plane)
                    )
                else:
                    camera_id = self.recipe.metric.applicability.camera_id
                    view = next(
                        (
                            v
                            for v in self.asset.parameters.views
                            if v.camera.applicability.camera_id == camera_id
                        ),
                        None,
                    )
                    if view is None:
                        raise ServiceError(
                            "recipe.metric_camera", "Metric plane lacks the source camera"
                        )
                    camera = view.camera.parameters
                    points = cv2.undistortPoints(
                        points.reshape(-1, 1, 2),
                        np.array(camera.camera_matrix),
                        np.array(camera.distortion),
                        P=np.array(camera.camera_matrix),
                    ).reshape(-1, 2)
                    points = perspective(points, np.array(view.camera_to_plane))
            else:
                raise ServiceError(
                    "recipe.metric_unsupported", "Unsupported sample metric calibration"
                )
        if not np.isfinite(points).all():
            raise ServiceError("recipe.metric_nonfinite", "Mapped samples are non-finite")
        return [{"x": float(x), "y": float(y)} for x, y in points]

    def convert(self, data: Dict[str, Any], operation: str, physical: bool = False) -> dict:
        """Recompute measured geometry from mapped observations and actual selected samples.

        Args:
            data: Original atomic observation with selected samples in the aligned patch frame.
            operation: Semantic atomic identifier string, such as measurements.edge-pairs,
                selecting the appropriate sample mapping and refitting behavior.
            physical: Whether to map recovered source samples into calibrated physical units.

        Returns:
            Observation refitted from mapped samples in the selected frame and length unit.
        """
        result = copy.deepcopy(data)
        frame = self.recipe.metric.physical_frame if physical else self.recipe.source_frame
        unit = self.asset.unit if physical else "px"
        result.update(frame=frame, unit=unit)

        def point(value: Optional[dict]) -> Optional[dict]:
            return self.map_points([value], physical)[0] if value is not None else None

        def observed(value: Optional[dict]) -> Optional[dict]:
            if value is None:
                return None
            value.update(frame=frame, unit=unit)
            if "point" in value:
                value["point"] = point(value["point"])
            return value

        if operation in {"measurements.lines", "measurements.circles"}:
            samples = self.map_points(data.get("samples", []), physical)
            request = PointFitRequest(points=[Point(**p) for p in samples], frame=frame, unit=unit)
            fitted = (
                fit_line(request, self.context)
                if operation.endswith("lines")
                else fit_circle(request, self.context)
            )
            mapped = fitted.model_dump(mode="json")
            mapped["diagnostics"]["requested_probe_count"] = data["diagnostics"][
                "requested_probe_count"
            ]
            mapped["diagnostics"]["successful_probe_count"] = data["diagnostics"][
                "successful_probe_count"
            ]
            result.update(mapped)
        if operation == "edge-profiles":
            result["sample_points"] = self.map_points(data["sample_points"], physical)
        if operation == "measurements.template-points":
            match = result.pop("match", None)
            if match is not None:
                result["localization_diagnostics"] = {
                    "score": match["score"],
                    "angle_degrees": match["angle_degrees"],
                    "scale": match["scale"],
                    "sampling_frame": "aligned",
                    "match": match,
                }
        if "edges" in result:
            for edge in result["edges"]:
                edge["point"] = point(edge["point"])
                edge.update(frame=frame, unit=unit)
        if operation in {"measurements.edge-points", "measurements.template-points"}:
            result["point"] = observed(result.get("point"))
            if operation == "measurements.edge-points":
                points = [edge["point"] for edge in result["edges"]]
                result["distances"] = [
                    float(np.hypot(b["x"] - a["x"], b["y"] - a["y"]))
                    for a, b in zip(points, points[1:])
                ]
        if operation == "measurements.edge-pairs":
            centers = []
            for pair in result["pairs"]:
                a, b = pair["edge_a"], pair["edge_b"]
                a["point"], b["point"] = point(a["point"]), point(b["point"])
                a.update(frame=frame, unit=unit)
                b.update(frame=frame, unit=unit)
                ap, bp = a["point"], b["point"]
                center = {"x": (ap["x"] + bp["x"]) / 2, "y": (ap["y"] + bp["y"]) / 2}
                pair["center"].update(frame=frame, unit=unit, point=center)
                pair["width"].update(
                    frame=frame,
                    unit=unit,
                    value=float(np.hypot(bp["x"] - ap["x"], bp["y"] - ap["y"])),
                )
                centers.append(center)
            single = result["pairs"][0] if len(result["pairs"]) == 1 else None
            result["edge_a"] = (
                dict(valid=True, frame=frame, unit=unit, point=single["edge_a"]["point"])
                if single
                else None
            )
            result["edge_b"] = (
                dict(valid=True, frame=frame, unit=unit, point=single["edge_b"]["point"])
                if single
                else None
            )
            result["center"] = single["center"] if single else None
            result["width"] = single["width"] if single else None
            result["center_distances"] = [
                float(np.hypot(b["x"] - a["x"], b["y"] - a["y"]))
                for a, b in zip(centers, centers[1:])
            ]
        if physical:
            result["calibration_sha256"] = calibration_digest(self.asset)
        return result
