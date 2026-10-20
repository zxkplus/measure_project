"""Manifest model-asset format (v3.0): save/load, integrity, migration.

Covers the format switch from a single legacy ``workflow.npz`` blob to a
manifest directory, including the guarantees that:

  * a saved asset round-trips through disk with identical measurements
  * ``workflow.npz`` is a *derived cache* and is never needed to load
  * corrupt assets fail loudly instead of silently
  * legacy and v2-era single-file assets produce an actionable error
  * ``scripts/migrate_models_to_manifest.py`` preserves measurements exactly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from measure.multi_target_workflow import (  # noqa: E402
    DERIVED_FILENAME,
    MANIFEST_CONTRACT_VERSION,
    MANIFEST_FILENAME,
    REFERENCE_FILENAME,
    MultiTargetWorkflow,
)

from tests.test_unified_multi_target import (  # noqa: E402
    MARKER_IMG,
    MARKER_PATCH,
    PATCH_ORIGIN_COL,
    PATCH_ORIGIN_ROW,
    make_inspection,
    make_reference,
    taught_workflow,
)


def _reference_placement():
    return (int(PATCH_ORIGIN_ROW), int(PATCH_ORIGIN_COL))


def _measure_summary(wf, image):
    return [
        {
            "id": tr.id,
            "valid": tr.valid,
            "center_row": round(float(tr.center_row), 9),
            "center_col": round(float(tr.center_col), 9),
            "measurements": json.loads(
                json.dumps(tr.measurements, default=str, sort_keys=True)
            ),
        }
        for tr in wf.measure(image)
    ]


# ---------------------------------------------------------------------------
# Asset layout
# ---------------------------------------------------------------------------


class TestAssetLayout:
    def test_save_writes_manifest_reference_and_cache(self, tmp_path: Path):
        wf = taught_workflow()
        wf.save(str(tmp_path))

        assert (tmp_path / MANIFEST_FILENAME).is_file()
        assert (tmp_path / REFERENCE_FILENAME).is_file()
        assert (tmp_path / DERIVED_FILENAME).is_file()

    def test_manifest_is_self_describing(self, tmp_path: Path):
        wf = taught_workflow()
        wf.save(str(tmp_path), model_id="unit-test-model")
        manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())

        assert manifest["contract_version"] == MANIFEST_CONTRACT_VERSION
        assert manifest["model_id"] == "unit-test-model"
        assert set(manifest) >= {
            "contract_version", "model_id", "created_at", "reference_image",
            "alignment", "matching", "physical_scale_mm", "objects", "files",
            "derived_fingerprint",
        }
        assert manifest["alignment"]["preprocessor"]["type"] == "raw"
        assert manifest["reference_image"]["file"] == REFERENCE_FILENAME
        assert set(manifest["files"]) == {REFERENCE_FILENAME, DERIVED_FILENAME}

    def test_save_accepts_npz_path_and_uses_its_parent(self, tmp_path: Path):
        """Existing callers pass <model_dir>/workflow.npz — keep that working."""
        wf = taught_workflow()
        wf.save(str(tmp_path / DERIVED_FILENAME))

        assert (tmp_path / MANIFEST_FILENAME).is_file()
        assert (tmp_path / REFERENCE_FILENAME).is_file()

    def test_objects_carry_stable_ids_and_label_dependencies(self, tmp_path: Path):
        wf = taught_workflow(add_tools=False)
        wf.add_measurement("EdgePoint", "a", row=10.0, col=10.0, angle=0.0,
                           length1=5.0, length2=2.0)
        wf.add_measurement("EdgePoint", "b", row=60.0, col=10.0, angle=0.0,
                           length1=5.0, length2=2.0)
        wf.add_measurement("TwoPointsDistance", "gap",
                           point_a_label="a", point_b_label="b")
        wf.save(str(tmp_path))

        manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
        gap = next(o for o in manifest["objects"] if o["label"] == "gap")
        assert gap["object_type"] == "TwoPointsDistance"
        assert gap["input_labels"] == ["a", "b"]
        # dependency labels must NOT leak into the plain parameter bag
        assert "point_a_label" not in gap["parameters"]

        # ids are deterministic -> re-saving yields the same ids
        wf.save(str(tmp_path))
        again = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
        assert ([o["object_id"] for o in manifest["objects"]]
                == [o["object_id"] for o in again["objects"]])


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_load_reproduces_measurements(self, tmp_path: Path):
        wf = taught_workflow()
        wf.save(str(tmp_path))
        reloaded = MultiTargetWorkflow.load(str(tmp_path))

        insp = make_inspection(make_reference(), [_reference_placement()])
        assert _measure_summary(wf, insp) == _measure_summary(reloaded, insp)

    def test_load_accepts_dir_manifest_and_npz(self, tmp_path: Path):
        taught_workflow().save(str(tmp_path))
        for path in (tmp_path,
                     tmp_path / MANIFEST_FILENAME,
                     tmp_path / DERIVED_FILENAME):
            wf = MultiTargetWorkflow.load(str(path))
            assert wf.template_image is not None

    def test_derived_cache_is_optional(self, tmp_path: Path):
        """Deleting workflow.npz must not affect loading at all."""
        wf = taught_workflow()
        wf.save(str(tmp_path))
        (tmp_path / DERIVED_FILENAME).unlink()

        reloaded = MultiTargetWorkflow.load(str(tmp_path))
        insp = make_inspection(make_reference(), [_reference_placement()])
        assert _measure_summary(wf, insp) == _measure_summary(reloaded, insp)

    def test_physical_scale_survives_round_trip(self, tmp_path: Path):
        wf = taught_workflow()
        wf.physical_scale_mm = 0.037
        wf.save(str(tmp_path))
        assert MultiTargetWorkflow.load(str(tmp_path)).physical_scale_mm == 0.037

    def test_matching_params_survive_round_trip(self, tmp_path: Path):
        wf = MultiTargetWorkflow()
        wf.teach_template(make_reference(), center=(120.0, 160.0),
                          size=(100.0, 120.0), angle_deg=7.5,
                          match_score_threshold=0.66,
                          angle_range=(-22.0, 22.0), angle_step=2.5,
                          max_matches=4, overlap=0.15)
        wf.save(str(tmp_path))
        reloaded = MultiTargetWorkflow.load(str(tmp_path))

        assert reloaded.alignment.box_angle_deg == pytest.approx(7.5)
        assert tuple(reloaded.alignment.box_center) == (120.0, 160.0)

    def test_virtual_edgepair_endpoint_dependency_round_trips(self, tmp_path: Path):
        """Composed tools may reference '<edgepair>_A'/'_B' virtual endpoints."""
        wf = taught_workflow(add_tools=False)
        wf.add_measurement("EdgePair", "pair", row=30.0, col=60.0, angle=0.0,
                           length1=40.0, length2=6.0)
        wf.add_measurement("TwoPointsDistance", "width",
                           point_a_label="pair_A", point_b_label="pair_B")
        wf.save(str(tmp_path))

        reloaded = MultiTargetWorkflow.load(str(tmp_path))
        width = next(d for d in reloaded.measurement_defs if d["label"] == "width")
        assert width["params"]["point_a_label"] == "pair_A"
        assert width["params"]["point_b_label"] == "pair_B"


# ---------------------------------------------------------------------------
# Integrity / failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_missing_manifest_reports_legacy_format(self, tmp_path: Path):
        """A legacy single-file .npz must produce an actionable error."""
        legacy = tmp_path / DERIVED_FILENAME
        meta = {"version": 1, "box_center": [0, 0], "box_size": [1, 1]}
        np.savez_compressed(
            str(legacy),
            workflow_meta=np.array([json.dumps(meta)], dtype=object),
        )
        with pytest.raises(ValueError, match="migrate_models_to_manifest"):
            MultiTargetWorkflow.load(str(tmp_path))

    def test_corrupt_reference_is_rejected(self, tmp_path: Path):
        taught_workflow().save(str(tmp_path))
        ref_path = tmp_path / REFERENCE_FILENAME
        data = np.load(str(ref_path), allow_pickle=False)
        np.save(str(ref_path), np.zeros_like(data), allow_pickle=False)

        with pytest.raises(ValueError, match="corrupt|sha256"):
            MultiTargetWorkflow.load(str(tmp_path))

    def test_unsupported_contract_version_is_rejected(self, tmp_path: Path):
        taught_workflow().save(str(tmp_path))
        manifest_path = tmp_path / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        manifest["contract_version"] = "0.1"
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="contract_version"):
            MultiTargetWorkflow.load(str(tmp_path))

    def test_missing_reference_is_reported(self, tmp_path: Path):
        taught_workflow().save(str(tmp_path))
        (tmp_path / REFERENCE_FILENAME).unlink()
        with pytest.raises(ValueError, match="incomplete"):
            MultiTargetWorkflow.load(str(tmp_path))

    def test_save_without_template_raises(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="No template defined"):
            MultiTargetWorkflow().save(str(tmp_path))


# ---------------------------------------------------------------------------
# Migration script
# ---------------------------------------------------------------------------


class TestMigrationScript:
    def _write_legacy_asset(self, model_dir: Path) -> MultiTargetWorkflow:
        """Create a legacy-format model by reusing the legacy writer path."""
        wf = taught_workflow()
        reference = wf._alignment._reference_image
        tp = wf._alignment.template_point
        meta = {
            "version": 1,
            "box_center": list(wf._alignment.box_center),
            "box_size": list(wf._alignment.box_size),
            "box_angle_deg": wf._alignment.box_angle_deg,
            "template_size": wf._alignment._template_size,
            "match_score_threshold": wf._match_score_threshold,
            "angle_range": list(wf._angle_range),
            "angle_step": wf._angle_step,
            "max_matches": wf._max_matches,
            "overlap": wf._overlap,
            "coarse_fine": wf._coarse_fine,
            "coarse_angle_step": wf._coarse_angle_step,
            "pyramid_decimate": tp.pyramid_decimate,
            "pyramid_max_template_size": tp.pyramid_max_template_size,
            "preprocessor_data": tp.preprocessor.serialize(),
            "click_row": tp.click_row,
            "click_col": tp.click_col,
            "crop_center_row": tp._crop_center_row,
            "crop_center_col": tp._crop_center_col,
            "crop_h": tp._crop_h,
            "crop_w": tp._crop_w,
            "measurement_defs": wf._measurement_defs,
            "alignment_mode": wf._alignment.mode_name,
            "alignment_roi_state": wf._alignment.get_roi_state(),
        }
        model_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(model_dir / DERIVED_FILENAME),
            workflow_meta=np.array([json.dumps(meta)], dtype=object),
            edge_template=tp.edge_template,
            template_image=wf._alignment.template_image,
            reference_image=reference,
        )
        return wf

    def test_migrate_preserves_measurements_and_keeps_backup(self, tmp_path: Path):
        migrate = pytest.importorskip("migrate_models_to_manifest")

        model_dir = tmp_path / "model"
        wf = self._write_legacy_asset(model_dir)
        insp = make_inspection(make_reference(), [_reference_placement()])
        before = _measure_summary(wf, insp)

        assert migrate.main([str(model_dir)]) == 0

        assert (model_dir / MANIFEST_FILENAME).is_file()
        assert (model_dir / "workflow.npz.legacy.npz").is_file()

        reloaded = MultiTargetWorkflow.load(str(model_dir))
        assert _measure_summary(reloaded, insp) == before

    def test_dry_run_writes_nothing(self, tmp_path: Path):
        migrate = pytest.importorskip("migrate_models_to_manifest")

        model_dir = tmp_path / "model"
        self._write_legacy_asset(model_dir)

        assert migrate.main([str(model_dir), "--dry-run"]) == 0
        assert not (model_dir / MANIFEST_FILENAME).exists()