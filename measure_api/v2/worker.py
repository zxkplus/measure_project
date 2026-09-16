"""Pickle-safe CPU tasks for the headless v2 engine."""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from measure.measure_template import (
    CannyPreprocessor,
    CLAHEPreprocessor,
    RawPreprocessor,
    SobelPreprocessor,
    TemplatePoint,
    ThresholdPreprocessor,
)
from measure.measure_workflow import (
    EdgePairObject,
    EdgePointObject,
    FitCircleObject,
    FitLineObject,
    MeasurementWorkflow,
    PointCircleDistanceObject,
    PointLineDistanceObject,
    TemplateMatchPointObject,
    TwoLinesAngleObject,
    TwoPointsDistanceObject,
    TwoPointsLineObject,
)
from measure_api.v2.application import EngineError

MODEL_CONTRACT_VERSION = "2.0"
PACKAGE_CONTRACT_VERSION = "2.0"


def run_worker_task(kind: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Dispatch an operation payload inside the CPU subprocess.

    Args:
        kind: Registered worker task kind.
        payload: Pickle-safe operation input.

    Returns:
        JSON-compatible task result.
    """
    tasks = {
        "preview": _preview,
        "publish": _publish,
        "measure": _measure,
        "export": _export,
        "import": _import,
    }
    task = tasks.get(kind)
    if task is None:
        raise EngineError(
            "operation.kind_unsupported",
            "operation",
            "Unsupported engine operation",
        )
    return task(payload)


def _build_workflow(
    reference_image: np.ndarray,
    definitions: Sequence[Mapping[str, Any]],
) -> MeasurementWorkflow:
    workflow = MeasurementWorkflow()
    for definition in definitions:
        obj: Any
        object_id = str(definition["object_id"])
        object_type = str(definition["object_type"])
        parameters = dict(definition["parameters"])
        inputs = list(definition.get("input_object_ids", []))
        if object_type == "EdgePoint":
            obj = EdgePointObject(
                label=object_id,
                row=parameters["row"],
                col=parameters["col"],
                angle=parameters["angle"],
                length1=parameters["length1"],
                length2=parameters["length2"],
                sigma=parameters["sigma"],
                threshold=parameters["threshold"],
                transition="all",
                select="first",
                interpolation=parameters["interpolation"],
            )
        elif object_type == "EdgePair":
            obj = EdgePairObject(
                label=object_id,
                row=parameters["row"],
                col=parameters["col"],
                angle=parameters["angle"],
                length1=parameters["length1"],
                length2=parameters["length2"],
                sigma=parameters["sigma"],
                threshold=parameters["threshold"],
                transition="all",
                select="endpoints",
                interpolation=parameters["interpolation"],
            )
        elif object_type == "FitLine":
            obj = FitLineObject(
                label=object_id,
                start=tuple(parameters["start"]),
                end=tuple(parameters["end"]),
                measure_length1=parameters["measure_length1"],
                measure_length2=parameters["measure_length2"],
                num_measures=parameters["num_measures"],
                sigma=parameters["sigma"],
                threshold=parameters["threshold"],
                transition=parameters["transition"],
            )
        elif object_type == "FitCircle":
            obj = FitCircleObject(
                label=object_id,
                center=tuple(parameters["center"]),
                radius=parameters["radius"],
                radius_min=parameters.get("radius_min"),
                radius_max=parameters.get("radius_max"),
                measure_length1=parameters["measure_length1"],
                measure_length2=parameters["measure_length2"],
                num_measures=parameters["num_measures"],
                sigma=parameters["sigma"],
                threshold=parameters["threshold"],
                transition=parameters["transition"],
                start_phi=parameters["start_phi"],
                end_phi=parameters["end_phi"],
            )
        elif object_type == "TemplateMatchPoint":
            preprocessors: Dict[str, Any] = {
                "raw": RawPreprocessor(),
                "canny": CannyPreprocessor(),
                "sobel": SobelPreprocessor(),
                "clahe": CLAHEPreprocessor(),
                "threshold": ThresholdPreprocessor(),
            }
            preprocessor_type = parameters["preprocessor_type"]
            preprocessor = preprocessors.get(preprocessor_type)
            if preprocessor is None:
                raise EngineError(
                    "object.parameters_invalid",
                    "request",
                    "Unsupported template preprocessor",
                    details={"preprocessor_type": preprocessor_type},
                )
            angle_range_half = float(parameters["angle_range_half"])
            template_point = TemplatePoint(
                reference_image,
                click_row=float(parameters["row"]),
                click_col=float(parameters["col"]),
                template_size=int(parameters["template_size"]),
                preprocessor=preprocessor,
                match_score_threshold=float(parameters["match_score_threshold"]),
                use_subpixel=bool(parameters["use_subpixel"]),
                rotation_invariant=angle_range_half > 0,
                angle_range=(-angle_range_half, angle_range_half),
                angle_step=float(parameters["angle_step"]),
            )
            obj = TemplateMatchPointObject(
                label=object_id,
                row=parameters["row"],
                col=parameters["col"],
                template_size=parameters["template_size"],
                preprocessor_type=parameters["preprocessor_type"],
                match_score_threshold=parameters["match_score_threshold"],
                angle_range_half=parameters["angle_range_half"],
                angle_step=parameters["angle_step"],
                use_subpixel=parameters["use_subpixel"],
                _template_point=template_point,
            )
        elif object_type == "TwoPointsLine":
            obj = TwoPointsLineObject(object_id, inputs[0], inputs[1])
        elif object_type == "TwoPointsDistance":
            obj = TwoPointsDistanceObject(object_id, inputs[0], inputs[1])
        elif object_type == "PointLineDistance":
            obj = PointLineDistanceObject(object_id, inputs[0], inputs[1])
        elif object_type == "TwoLinesAngle":
            obj = TwoLinesAngleObject(object_id, inputs[0], inputs[1])
        elif object_type == "PointCircleDistance":
            obj = PointCircleDistanceObject(object_id, inputs[0], inputs[1])
        else:  # pragma: no cover - validated by the application
            raise EngineError("object.type_unsupported", "request", "Unsupported object type")
        workflow.add(obj)
    try:
        workflow.resolve()
    except ValueError as exc:
        raise EngineError(
            "object.dependencies_invalid",
            "measurement",
            "Measurement object dependency graph is invalid",
            details={"diagnostic": str(exc)},
        ) from exc
    return workflow


def _preview(payload: Mapping[str, Any]) -> Dict[str, Any]:
    definitions = payload["objects"]
    workflow = _build_workflow(payload["reference_image"], definitions)
    image = np.ascontiguousarray(payload["image"], dtype=np.uint8)
    results = workflow.measure(image)
    return {
        "objects": _object_results(
            definitions,
            results,
            trace_id=str(payload["operation_id"]),
        )
    }


def _publish(payload: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        definitions = payload["objects"]
        reference_image = np.ascontiguousarray(payload["reference_image"], dtype=np.uint8)
        workflow = _build_workflow(reference_image, definitions)
        asset_dir = Path(str(payload["asset_dir"]))
        asset_dir.mkdir(parents=True, exist_ok=True)

        workflow_path = asset_dir / "workflow.npz"
        reference_path = asset_dir / "reference.npy"
        manifest_path = asset_dir / "manifest.json"
        workflow.save(str(workflow_path))
        np.save(str(reference_path), reference_image, allow_pickle=False)
        manifest = {
            "contract_version": MODEL_CONTRACT_VERSION,
            "calibration_model_id": payload["calibration_model_id"],
            "asset_ref": payload["asset_ref"],
            "objects": definitions,
            "files": {
                "workflow.npz": _file_sha256(workflow_path),
                "reference.npy": _file_sha256(reference_path),
            },
        }
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=True, sort_keys=True)
    except EngineError:
        raise
    except Exception as exc:
        raise EngineError(
            "model.asset_unavailable",
            "model",
            "Model asset could not be published",
            details={"diagnostic": str(exc)},
        ) from exc
    return {
        "calibration_model_id": payload["calibration_model_id"],
        "asset_ref": payload["asset_ref"],
    }


def _measure(payload: Mapping[str, Any]) -> Dict[str, Any]:
    asset_dir = Path(str(payload["asset_dir"]))
    manifest = _load_asset(
        asset_dir,
        str(payload["calibration_model_id"]),
        str(payload["asset_ref"]),
    )
    try:
        reference_image = np.load(str(asset_dir / "reference.npy"), allow_pickle=False)
        workflow = _build_workflow(reference_image, manifest["objects"])
    except Exception as exc:
        raise EngineError(
            "model.asset_unavailable",
            "model",
            "Model asset is unavailable",
            details={"diagnostic": str(exc)},
        ) from exc
    image = np.ascontiguousarray(payload["image"], dtype=np.uint8)
    results = workflow.measure(image)
    return {
        "calibration_model_id": payload["calibration_model_id"],
        "asset_ref": payload["asset_ref"],
        "objects": _object_results(
            manifest["objects"],
            results,
            trace_id=str(payload["operation_id"]),
        ),
    }


def _export(payload: Mapping[str, Any]) -> Dict[str, Any]:
    asset_dir = Path(str(payload["asset_dir"]))
    model_manifest = _load_asset(
        asset_dir,
        str(payload["calibration_model_id"]),
        str(payload["asset_ref"]),
    )
    export_dir = Path(str(payload["export_dir"]))
    export_dir.mkdir(parents=True, exist_ok=True)
    package_path = export_dir / f"{payload['operation_id']}.zip"
    package_manifest = {
        "contract_version": PACKAGE_CONTRACT_VERSION,
        "model_contract_version": model_manifest["contract_version"],
        "calibration_model_id": model_manifest["calibration_model_id"],
        "objects": model_manifest["objects"],
        "files": model_manifest["files"],
    }
    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(package_manifest, ensure_ascii=True, sort_keys=True),
        )
        archive.write(asset_dir / "workflow.npz", "workflow.npz")
        archive.write(asset_dir / "reference.npy", "reference.npy")
    digest = _file_sha256(package_path)
    return {
        "download_ref": f"v2-export:{payload['operation_id']}",
        "sha256": digest,
        "size": package_path.stat().st_size,
        "package_contract_version": PACKAGE_CONTRACT_VERSION,
    }


def _import(payload: Mapping[str, Any]) -> Dict[str, Any]:
    package = bytes(payload["package"])
    try:
        with zipfile.ZipFile(io.BytesIO(package), "r") as archive:
            names = set(archive.namelist())
            expected = {"manifest.json", "workflow.npz", "reference.npy"}
            if names != expected:
                raise ValueError("model package contains unexpected files")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("contract_version") != PACKAGE_CONTRACT_VERSION:
                raise ValueError("unsupported package contract version")
            workflow_bytes = archive.read("workflow.npz")
            reference_bytes = archive.read("reference.npy")
            if hashlib.sha256(workflow_bytes).hexdigest() != manifest["files"]["workflow.npz"]:
                raise ValueError("workflow digest mismatch")
            if hashlib.sha256(reference_bytes).hexdigest() != manifest["files"]["reference.npy"]:
                raise ValueError("reference digest mismatch")
            reference_image = np.load(io.BytesIO(reference_bytes), allow_pickle=False)
    except (KeyError, TypeError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise EngineError(
            "model.package_invalid",
            "model",
            "Model package is invalid",
            details={"diagnostic": str(exc)},
        ) from exc
    return {
        "reference_image": np.ascontiguousarray(reference_image, dtype=np.uint8),
        "objects": manifest["objects"],
    }


def _load_asset(
    asset_dir: Path,
    calibration_model_id: str,
    asset_ref: str,
) -> Dict[str, Any]:
    try:
        with (asset_dir / "manifest.json").open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest["contract_version"] != MODEL_CONTRACT_VERSION:
            raise ValueError("unsupported model contract version")
        if manifest["calibration_model_id"] != calibration_model_id:
            raise ValueError("calibration model identity mismatch")
        if manifest["asset_ref"] != asset_ref:
            raise ValueError("asset reference mismatch")
        for name, digest in manifest["files"].items():
            path = asset_dir / name
            if not path.is_file() or _file_sha256(path) != digest:
                raise ValueError(f"asset file is missing or corrupt: {name}")
        return manifest
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EngineError(
            "model.asset_unavailable",
            "model",
            "Model asset is unavailable",
            details={"diagnostic": str(exc)},
        ) from exc


def _object_results(
    definitions: Sequence[Mapping[str, Any]],
    results: Mapping[str, Any],
    *,
    trace_id: str,
) -> List[Dict[str, Any]]:
    output = []
    for definition in definitions:
        object_id = str(definition["object_id"])
        result = results.get(object_id)
        valid = bool(result is not None and getattr(result, "valid", False))
        item = {
            "object_id": object_id,
            "object_type": definition["object_type"],
            "label": definition["label"],
            "valid": valid,
            "observation": _observation(result) if valid else None,
            "quality": _json_value(getattr(result, "meta", {})) if result is not None else {},
            "error": None,
        }
        if valid and not _finite_value(item["observation"]):
            valid = False
            item["valid"] = False
            item["observation"] = None
            item["error"] = {
                "code": "measurement.non_finite",
                "category": "measurement",
                "message": "Measurement object produced a non-finite observation",
                "retryable": False,
                "details": {},
                "trace_id": trace_id,
            }
        if not valid:
            diagnostic = None
            if result is not None:
                diagnostic = getattr(result, "meta", {}).get("error")
            if item["error"] is None:
                item["error"] = {
                    "code": "measurement.object_invalid",
                    "category": "measurement",
                    "message": "Measurement object did not produce a valid observation",
                    "retryable": False,
                    "details": {"diagnostic": diagnostic} if diagnostic else {},
                    "trace_id": trace_id,
                }
        output.append(item)
    return output


def _observation(result: Any) -> Dict[str, Any]:
    result_type = str(result.type)
    if result_type == "point":
        return {"type": "point", "row": float(result.row), "col": float(result.col)}
    if result_type == "line":
        return {
            "type": "line",
            "a": float(result.a),
            "b": float(result.b),
            "c": float(result.c),
            "start_row": float(result.start_row),
            "start_col": float(result.start_col),
            "end_row": float(result.end_row),
            "end_col": float(result.end_col),
        }
    if result_type == "circle":
        return {
            "type": "circle",
            "center_row": float(result.center_row),
            "center_col": float(result.center_col),
            "radius": float(result.radius),
        }
    if result_type == "angle":
        return {"type": "scalar", "value": float(result.value_deg), "unit": "degree"}
    return {"type": "scalar", "value": float(result.value), "unit": "pixel"}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_value(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
