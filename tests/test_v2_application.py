"""Behavior tests for the headless v2 engine application."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest


def _edge_point(row: float, col: float) -> dict:
    return {
        "object_type": "EdgePoint",
        "label": f"edge-{row}-{col}",
        "parameters": {
            "row": row,
            "col": col,
            "angle": 0.0,
            "length1": 10.0,
            "length2": 3.0,
        },
    }


def test_modeling_session_uses_stable_object_ids_for_dependencies(tmp_path) -> None:
    """Renaming an input must not break a composed object's dependency."""
    from measure_api.v2.application import EngineApplication, EngineError

    current = [datetime(2026, 8, 9, tzinfo=timezone.utc)]
    application = EngineApplication(tmp_path, now=lambda: current[0])
    session = application.create_modeling_session(
        np.zeros((40, 40), dtype=np.uint8)
    )

    first = application.create_object(session["session_id"], _edge_point(10, 10))
    second = application.create_object(session["session_id"], _edge_point(20, 20))
    distance = application.create_object(
        session["session_id"],
        {
            "object_type": "TwoPointsDistance",
            "label": "gap",
            "parameters": {},
            "input_object_ids": [first["object_id"], second["object_id"]],
        },
    )

    renamed = application.update_object(
        session["session_id"], first["object_id"], {"label": "renamed-edge"}
    )
    objects = application.list_objects(session["session_id"])

    assert renamed["object_id"] == first["object_id"]
    assert objects[-1]["object_id"] == distance["object_id"]
    assert objects[-1]["input_object_ids"] == [first["object_id"], second["object_id"]]
    current[0] += timedelta(hours=8, microseconds=1)
    with pytest.raises(EngineError, match="expired") as error:
        application.get_modeling_session(session["session_id"])
    assert error.value.error["code"] == "session.expired"
    application.close()


def test_all_first_release_object_contracts_accept_representative_requests(
    tmp_path,
) -> None:
    """Every OpenAPI discriminator variant has a matching application behavior."""
    from measure_api.v2.application import EngineApplication

    application = EngineApplication(tmp_path)
    try:
        session = application.create_modeling_session(np.zeros((80, 80), dtype=np.uint8))
        session_id = session["session_id"]
        point_a = application.create_object(session_id, _edge_point(20, 20))
        point_b = application.create_object(session_id, _edge_point(30, 30))
        edge_pair = application.create_object(
            session_id,
            {
                **_edge_point(40, 40),
                "object_type": "EdgePair",
                "label": "edge-pair",
            },
        )
        fit_line = application.create_object(
            session_id,
            {
                "object_type": "FitLine",
                "label": "fit-line",
                "parameters": {
                    "start": [10, 10],
                    "end": [60, 60],
                    "measure_length1": 8,
                    "measure_length2": 3,
                },
            },
        )
        fit_circle = application.create_object(
            session_id,
            {
                "object_type": "FitCircle",
                "label": "fit-circle",
                "parameters": {
                    "center": [40, 40],
                    "radius": 20,
                    "measure_length1": 8,
                    "measure_length2": 3,
                },
            },
        )
        template_point = application.create_object(
            session_id,
            {
                "object_type": "TemplateMatchPoint",
                "label": "template-point",
                "parameters": {"row": 40, "col": 40},
            },
        )
        two_points_line = application.create_object(
            session_id,
            {
                "object_type": "TwoPointsLine",
                "label": "two-points-line",
                "parameters": {},
                "input_object_ids": [point_a["object_id"], point_b["object_id"]],
            },
        )
        composites = [
            {
                "object_type": "TwoPointsDistance",
                "label": "two-points-distance",
                "parameters": {},
                "input_object_ids": [point_a["object_id"], point_b["object_id"]],
            },
            {
                "object_type": "PointLineDistance",
                "label": "point-line-distance",
                "parameters": {},
                "input_object_ids": [point_a["object_id"], fit_line["object_id"]],
            },
            {
                "object_type": "TwoLinesAngle",
                "label": "two-lines-angle",
                "parameters": {},
                "input_object_ids": [
                    fit_line["object_id"],
                    two_points_line["object_id"],
                ],
            },
            {
                "object_type": "PointCircleDistance",
                "label": "point-circle-distance",
                "parameters": {},
                "input_object_ids": [point_a["object_id"], fit_circle["object_id"]],
            },
        ]
        created_composites = [
            application.create_object(session_id, request) for request in composites
        ]

        object_types = {
            item["object_type"] for item in application.list_objects(session_id)
        }
        assert object_types == {
            "EdgePoint",
            "EdgePair",
            "FitLine",
            "FitCircle",
            "TemplateMatchPoint",
            "TwoPointsLine",
            "TwoPointsDistance",
            "PointLineDistance",
            "TwoLinesAngle",
            "PointCircleDistance",
        }
        assert edge_pair["parameters"]["threshold"] == 30.0
        assert template_point["parameters"]["use_subpixel"] is True
        assert len(created_composites) == 4
    finally:
        application.close()


def test_production_import_graph_does_not_load_gui_modules() -> None:
    """The production application and HTTP adapter import without GUI code."""
    code = (
        "import sys; "
        "import measure_api.v2.http; "
        "assert not any(name == 'measure_gui' or name.startswith('measure_gui.') "
        "for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_publish_then_measure_uses_persistent_opaque_asset(tmp_path) -> None:
    """Publishing closes the draft and production measurement uses only the asset."""
    from measure_api.v2.application import EngineApplication, EngineError

    image = np.zeros((60, 80), dtype=np.uint8)
    image[:, 40:] = 255
    application = EngineApplication(tmp_path)
    try:
        session = application.create_modeling_session(image)
        created = application.create_object(
            session["session_id"],
            {
                "object_type": "EdgePoint",
                "label": "edge",
                "parameters": {
                    "row": 30.0,
                    "col": 40.0,
                    "angle": 0.0,
                    "length1": 20.0,
                    "length2": 5.0,
                    "threshold": 5.0,
                },
            },
        )

        publishing = application.publish(session["session_id"], "cal-model-17")
        published = application.wait_operation(publishing["operation_id"], 15_000)

        assert published["status"] == "succeeded", published
        asset_ref = published["result"]["asset_ref"]
        assert asset_ref.startswith("v2-model:")
        with pytest.raises(EngineError):
            application.get_modeling_session(session["session_id"])

        measuring = application.measure("cal-model-17", asset_ref, image)
        measured = application.wait_operation(measuring["operation_id"], 15_000)

        assert measured["status"] == "succeeded", measured
        assert measured["result"]["objects"][0]["object_id"] == created["object_id"]
        assert measured["result"]["objects"][0]["observation"]["type"] == "point"
        assert "decision" not in measured["result"]["objects"][0]
    finally:
        application.close()


def test_operation_deadline_includes_process_startup_and_repeated_posts_are_distinct(
    tmp_path,
) -> None:
    """Each POST creates a new operation and an expired deadline is terminal."""
    from measure_api.v2.application import EngineApplication

    application = EngineApplication(tmp_path)
    try:
        session = application.create_modeling_session(np.zeros((20, 20), dtype=np.uint8))
        first = application.preview(
            session["session_id"], timeout_ms=1, client_request_id="request-7"
        )
        second = application.preview(
            session["session_id"], timeout_ms=1, client_request_id="request-7"
        )

        first_done = application.wait_operation(first["operation_id"], 5_000)
        second_done = application.wait_operation(second["operation_id"], 5_000)

        assert first["operation_id"] != second["operation_id"]
        assert first_done["status"] == "timed_out"
        assert second_done["status"] == "timed_out"
        assert len(application.list_operations("request-7")) == 2

        cancellable = application.preview(session["session_id"], timeout_ms=30_000)
        application.cancel_operation(cancellable["operation_id"])
        cancelled = application.wait_operation(cancellable["operation_id"], 5_000)
        assert cancelled["status"] == "cancelled"
    finally:
        application.close()


def test_operation_deadline_can_start_before_application_submission(tmp_path) -> None:
    """Adapters can include request decoding in the application operation budget."""
    from measure_api.v2.application import EngineApplication

    application = EngineApplication(tmp_path)
    try:
        session = application.create_modeling_session(np.zeros((20, 20), dtype=np.uint8))
        operation = application.preview(
            session["session_id"],
            timeout_ms=1_000,
            accepted_monotonic=time.monotonic() - 2.0,
        )

        completed = application.wait_operation(operation["operation_id"], 5_000)

        assert completed["status"] == "timed_out"
    finally:
        application.close()


def test_completion_handler_obeys_deadline_and_rolls_back() -> None:
    """Parent-process import materialization remains inside the operation deadline."""
    from measure_api.v2.operations import OperationManager

    rolled_back = []
    manager = OperationManager(multiprocessing_context="fork")

    def materialize(_result: dict) -> dict:
        time.sleep(0.5)
        return {"session_id": "temporary-session"}

    def rollback(result: dict) -> None:
        rolled_back.append(result["session_id"])

    try:
        operation = manager.submit(
            "preview",
            {
                "reference_image": np.zeros((2, 2), dtype=np.uint8),
                "image": np.zeros((2, 2), dtype=np.uint8),
                "objects": [],
            },
            200,
            on_success=materialize,
            on_abort=rollback,
        )
        completed = manager.wait(operation["operation_id"], 5_000)

        assert completed["status"] == "timed_out"
        assert rolled_back == ["temporary-session"]
    finally:
        manager.close()


def test_object_failure_is_structured_without_failing_preview_operation(tmp_path) -> None:
    """A completed multi-object orchestration keeps failures at object scope."""
    from measure_api.v2.application import EngineApplication

    application = EngineApplication(tmp_path)
    try:
        session = application.create_modeling_session(np.zeros((40, 40), dtype=np.uint8))
        application.create_object(session["session_id"], _edge_point(20, 20))

        operation = application.preview(session["session_id"])
        completed = application.wait_operation(operation["operation_id"], 15_000)

        assert completed["status"] == "succeeded", completed
        result = completed["result"]["objects"][0]
        assert result["valid"] is False
        assert result["observation"] is None
        assert set(result["error"]) == {
            "code",
            "category",
            "message",
            "retryable",
            "details",
            "trace_id",
        }
        assert result["error"]["category"] == "measurement"
    finally:
        application.close()


def test_export_survives_restart_and_import_creates_editable_session(tmp_path) -> None:
    """Portable packages are integrity checked and never replace assets on import."""
    from measure_api.v2.application import EngineApplication

    image = np.zeros((40, 40), dtype=np.uint8)
    first_application = EngineApplication(tmp_path)
    session = first_application.create_modeling_session(image)
    original = first_application.create_object(
        session["session_id"], _edge_point(20, 20)
    )
    publication = first_application.publish(session["session_id"], "portable-model")
    published = first_application.wait_operation(publication["operation_id"], 15_000)
    assert published["status"] == "succeeded", published
    asset_ref = published["result"]["asset_ref"]
    first_application.close()

    second_application = EngineApplication(tmp_path)
    try:
        exporting = second_application.export_model("portable-model", asset_ref)
        exported = second_application.wait_operation(exporting["operation_id"], 15_000)
        assert exported["status"] == "succeeded", exported
        package = second_application.read_export(exported["result"]["download_ref"])

        importing = second_application.import_model(package)
        imported = second_application.wait_operation(importing["operation_id"], 15_000)
        assert imported["status"] == "succeeded", imported
        imported_objects = second_application.list_objects(
            imported["result"]["session_id"]
        )
        assert imported_objects[0]["object_id"] == original["object_id"]
        assert imported_objects[0]["label"] == original["label"]

        measured = second_application.measure("portable-model", asset_ref, image)
        completed = second_application.wait_operation(measured["operation_id"], 15_000)
        assert completed["status"] == "succeeded", completed
    finally:
        second_application.close()


def test_invalid_import_is_reported_as_model_operation_error(tmp_path) -> None:
    """Semantic package failures use the stable operation error structure."""
    from measure_api.v2.application import EngineApplication

    application = EngineApplication(tmp_path)
    try:
        operation = application.import_model(b"not-a-model-package")
        completed = application.wait_operation(operation["operation_id"], 15_000)

        assert completed["status"] == "failed"
        assert completed["error"]["code"] == "model.package_invalid"
        assert completed["error"]["category"] == "model"
    finally:
        application.close()


def test_publish_write_failure_is_a_stable_model_error(tmp_path) -> None:
    """A direct-overwrite filesystem failure must not leak as an internal error."""
    from measure_api.v2.application import EngineApplication

    application = EngineApplication(tmp_path)
    try:
        session = application.create_modeling_session(
            np.zeros((20, 20), dtype=np.uint8)
        )
        application.create_object(session["session_id"], _edge_point(10, 10))
        (tmp_path / "models").write_text("blocks the model directory", encoding="utf-8")

        operation = application.publish(session["session_id"], "unwritable-model")
        completed = application.wait_operation(operation["operation_id"], 15_000)

        assert completed["status"] == "failed"
        assert completed["error"]["code"] == "model.asset_unavailable"
        assert completed["error"]["category"] == "model"
        assert application.get_modeling_session(session["session_id"])
    finally:
        application.close()


def test_corrupt_asset_fails_without_engine_or_model_fallback(tmp_path) -> None:
    from measure_api.v2.application import EngineApplication

    image = np.zeros((40, 40), dtype=np.uint8)
    application = EngineApplication(tmp_path)
    try:
        session = application.create_modeling_session(image)
        application.create_object(session["session_id"], _edge_point(20, 20))
        publication = application.publish(session["session_id"], "corrupt-model")
        published = application.wait_operation(publication["operation_id"], 15_000)
        reference_path = next((tmp_path / "models").glob("*/reference.npy"))
        reference_path.write_bytes(b"corrupt")

        operation = application.measure(
            "corrupt-model", published["result"]["asset_ref"], image
        )
        completed = application.wait_operation(operation["operation_id"], 15_000)

        assert completed["status"] == "failed"
        assert completed["error"]["code"] == "model.asset_unavailable"
    finally:
        application.close()


def test_multi_object_measurement_preserves_partial_results(tmp_path) -> None:
    from measure_api.v2.application import EngineApplication

    image = np.zeros((60, 80), dtype=np.uint8)
    image[:, 40:] = 255
    application = EngineApplication(tmp_path)
    try:
        session = application.create_modeling_session(image)
        application.create_object(
            session["session_id"],
            {
                **_edge_point(30, 40),
                "parameters": {
                    **_edge_point(30, 40)["parameters"],
                    "length1": 20.0,
                    "length2": 5.0,
                    "threshold": 5.0,
                },
            },
        )
        application.create_object(session["session_id"], _edge_point(10, 10))
        publication = application.publish(session["session_id"], "partial-model")
        published = application.wait_operation(publication["operation_id"], 15_000)

        operation = application.measure(
            "partial-model", published["result"]["asset_ref"], image
        )
        completed = application.wait_operation(operation["operation_id"], 15_000)

        assert completed["status"] == "succeeded"
        assert [item["valid"] for item in completed["result"]["objects"]] == [
            True,
            False,
        ]
        assert completed["result"]["objects"][1]["error"]["category"] == "measurement"
    finally:
        application.close()


def test_export_digest_and_import_require_explicit_publication(tmp_path) -> None:
    from measure_api.v2.application import EngineApplication, EngineError

    image = np.zeros((40, 40), dtype=np.uint8)
    application = EngineApplication(tmp_path)
    try:
        session = application.create_modeling_session(image)
        application.create_object(session["session_id"], _edge_point(20, 20))
        publication = application.publish(session["session_id"], "source-model")
        published = application.wait_operation(publication["operation_id"], 15_000)
        exporting = application.export_model(
            "source-model", published["result"]["asset_ref"]
        )
        exported = application.wait_operation(exporting["operation_id"], 15_000)
        package = application.read_export(exported["result"]["download_ref"])

        assert hashlib.sha256(package).hexdigest() == exported["result"]["sha256"]
        assert len(package) == exported["result"]["size"]
        with zipfile.ZipFile(io.BytesIO(package), "r") as archive:
            manifest = json.loads(archive.read("manifest.json"))
        assert manifest["contract_version"] == exported["result"][
            "package_contract_version"
        ]

        importing = application.import_model(package)
        imported = application.wait_operation(importing["operation_id"], 15_000)
        with pytest.raises(EngineError) as unavailable:
            application.measure("imported-model", "v2-model:missing", image)
        assert unavailable.value.error["code"] == "model.asset_unavailable"

        republishing = application.publish(
            imported["result"]["session_id"], "imported-model"
        )
        republished = application.wait_operation(republishing["operation_id"], 15_000)
        measured = application.measure(
            "imported-model", republished["result"]["asset_ref"], image
        )
        completed = application.wait_operation(measured["operation_id"], 15_000)
        assert completed["status"] == "succeeded"
    finally:
        application.close()
