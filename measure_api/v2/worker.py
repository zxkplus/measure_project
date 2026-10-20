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

from measure.alignment import strategy_from_roi_state
from measure.measure_template import (
    CannyPreprocessor,
    CLAHEPreprocessor,
    RawPreprocessor,
    SobelPreprocessor,
    ThresholdPreprocessor,
    _deserialize_preprocessor,
)
from measure.multi_target_workflow import (
    MANIFEST_CONTRACT_VERSION,
    MultiTargetWorkflow,
)
from measure.object_specs import get_spec
from measure_api.v2.application import EngineError

# The v2 engine now stores models in the shared manifest format.
MODEL_CONTRACT_VERSION = MANIFEST_CONTRACT_VERSION
PACKAGE_CONTRACT_VERSION = "2.0"

_PREPROCESSOR_FACTORIES = {
    "raw": RawPreprocessor,
    "canny": CannyPreprocessor,
    "sobel": SobelPreprocessor,
    "clahe": CLAHEPreprocessor,
    "threshold": ThresholdPreprocessor,
}


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


def _resolve_preprocessor(roi: Mapping[str, Any]):
    """Build the ROI preprocessor named in the ROI definition."""
    data = dict(roi.get("preprocessor") or {"type": "raw"})
    if "preprocessor_type" in roi and "type" not in data:
        data["type"] = roi["preprocessor_type"]
    ptype = str(data.get("type", "raw"))
    if ptype not in _PREPROCESSOR_FACTORIES:
        raise EngineError(
            "object.parameters_invalid",
            "request",
            "Unsupported template preprocessor",
            details={"preprocessor_type": ptype},
        )
    if "type" not in data:
        return _PREPROCESSOR_FACTORIES[ptype]()
    try:
        return _deserialize_preprocessor(data)
    except Exception as exc:  # noqa: BLE001 - report as a request error
        raise EngineError(
            "object.parameters_invalid",
            "request",
            "Invalid ROI preprocessor definition",
            details={"diagnostic": str(exc)},
        ) from exc


def _dependency_params(
    definition: Mapping[str, Any],
    label_by_id: Mapping[str, str],
) -> Dict[str, Any]:
    """Turn an object's id-based dependencies into the label form the engine uses."""
    object_type = str(definition["object_type"])
    params = dict(definition["parameters"])
    dep_names = get_spec(object_type).dep_param_names
    if "input_labels" in definition:
        labels = list(definition["input_labels"])
    else:
        labels = [
            label_by_id.get(str(i), str(i))
            for i in definition.get("input_object_ids", [])
        ]
    for name, target_label in zip(dep_names, labels):
        params[name] = target_label
    return params


def _build_engine(
    reference_image: np.ndarray,
    roi: Mapping[str, Any],
    definitions: Sequence[Mapping[str, Any]],
) -> MultiTargetWorkflow:
    """Build the shared multi-target engine for a modelling session or asset.

    Args:
        reference_image: Reference image the ROI was taught on.
        roi: ROI definition (box geometry, matching params, preprocessor).
        definitions: Object definitions in **patch** coordinates.

    Returns:
        A taught MultiTargetWorkflow with every object registered.
    """
    engine = MultiTargetWorkflow()
    engine._alignment = strategy_from_roi_state(
        dict(roi.get("roi_state") or {}), reference_image
    )
    try:
        engine.teach_template(
            reference_image,
            center=tuple(roi["center"]),
            size=tuple(roi["size"]),
            angle_deg=float(roi.get("angle_deg", 0.0)),
            preprocessor=_resolve_preprocessor(roi),
            match_score_threshold=float(roi.get("match_score_threshold", 0.5)),
            angle_range=tuple(roi.get("angle_range", (-30.0, 30.0))),
            angle_step=float(roi.get("angle_step", 1.0)),
            max_matches=int(roi.get("max_matches", 0)),
            overlap=float(roi.get("overlap", 0.3)),
            coarse_fine=bool(roi.get("coarse_fine", True)),
            coarse_angle_step=float(roi.get("coarse_angle_step", 5.0)),
            pyramid_decimate=int(roi.get("pyramid_decimate", 0)),
            pyramid_max_template_size=int(roi.get("pyramid_max_template_size", 400)),
        )
    except EngineError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EngineError(
            "session.roi_invalid",
            "request",
            "ROI definition could not be taught on the reference image",
            details={"diagnostic": str(exc)},
        ) from exc

    label_by_id = {
        str(d["object_id"]): str(d["label"]) for d in definitions
    }
    for definition in definitions:
        object_type = str(definition["object_type"])
        label = str(definition["label"])
        params = _dependency_params(definition, label_by_id)
        try:
            engine.add_measurement(object_type, label, **params)
        except KeyError as exc:
            raise EngineError(
                "object.type_unsupported",
                "request",
                "Unsupported measurement object type",
                details={"object_type": object_type},
            ) from exc
        except ValueError as exc:
            raise EngineError(
                "object.parameters_invalid",
                "request",
                "Measurement object parameters are invalid",
                details={"label": label, "diagnostic": str(exc)},
            ) from exc
    return engine


def _preview(payload: Mapping[str, Any]) -> Dict[str, Any]:
    definitions = payload["objects"]
    image = np.ascontiguousarray(payload["image"], dtype=np.uint8)
    if not definitions:
        # Nothing to measure, so no ROI is required to answer.
        return {"instances": [], "objects": []}
    engine = _build_engine(
        np.ascontiguousarray(payload["reference_image"], dtype=np.uint8),
        payload["roi"],
        definitions,
    )
    return _instance_payload(
        engine.measure(image),
        definitions,
        trace_id=str(payload["operation_id"]),
    )


def _publish(payload: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        reference_image = np.ascontiguousarray(payload["reference_image"], dtype=np.uint8)
        engine = _build_engine(reference_image, payload["roi"], payload["objects"])
        asset_dir = Path(str(payload["asset_dir"]))
        asset_dir.mkdir(parents=True, exist_ok=True)

        # The shared engine owns the asset layout (manifest.json +
        # reference.npy + derived workflow.npz).
        engine.save(
            str(asset_dir),
            model_id=str(payload["calibration_model_id"]),
            object_ids={
                str(d["label"]): str(d["object_id"]) for d in payload["objects"]
            },
        )

        manifest_path = asset_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["calibration_model_id"] = payload["calibration_model_id"]
        manifest["asset_ref"] = payload["asset_ref"]
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
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
        engine = MultiTargetWorkflow.load(str(asset_dir))
    except Exception as exc:
        raise EngineError(
            "model.asset_unavailable",
            "model",
            "Model asset is unavailable",
            details={"diagnostic": str(exc)},
        ) from exc
    image = np.ascontiguousarray(payload["image"], dtype=np.uint8)
    definitions = manifest.get("objects", [])
    result = _instance_payload(
        engine.measure(image),
        definitions,
        trace_id=str(payload["operation_id"]),
    )
    result["calibration_model_id"] = payload["calibration_model_id"]
    result["asset_ref"] = payload["asset_ref"]
    return result


def _roi_from_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Rebuild a modelling-session ROI definition from a model manifest."""
    alignment = dict(manifest.get("alignment") or {})
    matching = dict(manifest.get("matching") or {})
    return {
        "center": list(alignment.get("box_center", [0.0, 0.0])),
        "size": list(alignment.get("box_size", [1.0, 1.0])),
        "angle_deg": float(alignment.get("box_angle_deg", 0.0)),
        "alignment_mode": alignment.get("mode", "single_box"),
        "preprocessor": dict(alignment.get("preprocessor") or {"type": "raw"}),
        "roi_state": dict(alignment.get("roi_state") or {}),
        "match_score_threshold": matching.get("match_score_threshold", 0.5),
        "angle_range": list(matching.get("angle_range", (-30.0, 30.0))),
        "angle_step": matching.get("angle_step", 1.0),
        "max_matches": matching.get("max_matches", 0),
        "overlap": matching.get("overlap", 0.3),
        "coarse_fine": matching.get("coarse_fine", True),
        "coarse_angle_step": matching.get("coarse_angle_step", 5.0),
        "pyramid_decimate": matching.get("pyramid_decimate", 0),
        "pyramid_max_template_size": matching.get("pyramid_max_template_size", 400),
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
        # The ROI and matching parameters are what make the package
        # republishable: without them the imported session could not rebuild
        # the patch coordinate system its objects are defined in.
        "alignment": model_manifest.get("alignment", {}),
        "matching": model_manifest.get("matching", {}),
        "physical_scale_mm": model_manifest.get("physical_scale_mm"),
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
        "roi": _roi_from_manifest(manifest),
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


def _instance_payload(
    target_results: Sequence[Any],
    definitions: Sequence[Mapping[str, Any]],
    *,
    trace_id: str,
) -> Dict[str, Any]:
    """Shape engine results for the v2 response.

    The engine returns **one result per detected target** (top-k).  The response
    exposes them as ``instances``; ``objects`` keeps its historical
    one-entry-per-object shape by reporting the primary (highest scoring)
    instance, so existing consumers are unaffected.

    Args:
        target_results: ``TargetResult`` list from ``MultiTargetWorkflow.measure``.
        definitions: Object definitions in modelling order.
        trace_id: Operation id used for error correlation.

    Returns:
        Mapping with ``instances`` and ``objects``.
    """
    instances = [
        {
            "index": int(target.id),
            "score": float(target.score),
            "rotation_deg": float(target.rotation_deg),
            "scale": float(target.scale),
            "center_row": float(target.center_row),
            "center_col": float(target.center_col),
            "valid": bool(target.valid),
            "measurements": _object_results(
                definitions, target.measurements, trace_id=trace_id
            ),
        }
        for target in target_results
    ]

    primary = target_results[0].measurements if target_results else {}
    return {
        "instances": instances,
        "objects": _object_results(definitions, primary, trace_id=trace_id),
    }


def _object_results(
    definitions: Sequence[Mapping[str, Any]],
    measurements: Mapping[str, Any],
    *,
    trace_id: str,
) -> List[Dict[str, Any]]:
    """Map engine measurement results onto the stable per-object response shape.

    ``measurements`` maps label -> result dict (the engine maps coordinates back
    to the inspection image and returns JSON-friendly dicts).
    """
    output = []
    for definition in definitions:
        object_id = str(definition["object_id"])
        label = str(definition["label"])
        raw = measurements.get(label)
        valid = bool(isinstance(raw, Mapping) and raw.get("valid", False))
        item = {
            "object_id": object_id,
            "object_type": definition["object_type"],
            "label": label,
            "valid": valid,
            "observation": None,
            "quality": _json_value(raw.get("meta", {})) if isinstance(raw, Mapping) else {},
            "error": None,
        }
        if valid:
            item["observation"] = _observation(raw)
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
            if isinstance(raw, Mapping):
                diagnostic = raw.get("meta", {}).get("error") or raw.get("reason")
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


def _observation(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert an engine result dict into the v2 observation union."""
    result_type = str(raw.get("type", ""))
    if result_type == "point":
        return {"type": "point", "row": float(raw["row"]), "col": float(raw["col"])}
    if result_type == "line":
        return {
            "type": "line",
            "a": float(raw["a"]),
            "b": float(raw["b"]),
            "c": float(raw["c"]),
            "start_row": float(raw["start_row"]),
            "start_col": float(raw["start_col"]),
            "end_row": float(raw["end_row"]),
            "end_col": float(raw["end_col"]),
        }
    if result_type == "circle":
        return {
            "type": "circle",
            "center_row": float(raw["center_row"]),
            "center_col": float(raw["center_col"]),
            "radius": float(raw["radius"]),
        }
    if result_type == "angle":
        value = raw.get("value_deg")
        if value is None:
            value = math.degrees(float(raw.get("value") or 0.0))
        return {"type": "scalar", "value": float(value), "unit": "degree"}
    return {"type": "scalar", "value": float(raw.get("value") or 0.0), "unit": "pixel"}


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
