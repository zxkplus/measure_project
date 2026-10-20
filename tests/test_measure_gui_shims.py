"""The measure_gui shim modules must keep re-exporting the shared engine.

These modules used to hold second copies of the engine; they are now thin
re-exports so that external callers keep working while there is still only one
implementation.
"""

from __future__ import annotations


def test_multi_target_shim_exports_the_shared_engine():
    import measure.multi_target_workflow as engine
    import measure_gui.multi_target as shim

    assert shim.MultiTargetWorkflow is engine.MultiTargetWorkflow
    assert shim.TargetResult is engine.TargetResult
    assert shim._map_result_to_original is engine._map_result_to_original


def test_multi_target_shim_exposes_object_registry():
    import measure_gui.multi_target as shim
    from measure.object_specs import OBJECT_SPECS, factory_table

    assert set(shim._OBJECT_FACTORIES) == set(OBJECT_SPECS)
    assert shim._OBJECT_FACTORIES == factory_table()
    assert shim._get_defaults("FitCircle") == dict(OBJECT_SPECS["FitCircle"].defaults)


def test_multi_target_shim_exposes_formatters():
    import measure.result_format as rf
    import measure_gui.multi_target as shim

    assert shim._format_geometric_result is rf._format_geometric_result
    assert shim._format_result_dict is rf._format_result_dict


def test_alignment_shim_exports_shared_strategies():
    import measure.alignment as engine
    import measure_gui.alignment as shim

    assert shim.AlignmentStrategy is engine.AlignmentStrategy
    assert shim.SingleBoxAlignment is engine.SingleBoxAlignment
    assert shim.MultiPointAlignment is engine.MultiPointAlignment
    assert shim.strategy_from_roi_state is engine.strategy_from_roi_state


def test_utils_shim_reexports_geometry_and_keeps_gui_helpers():
    import measure.rectify as rectify
    import measure_gui.utils as shim

    # geometry now lives in measure.rectify
    assert shim.crop_and_straighten is rectify.crop_and_straighten
    assert shim.map_point_via_affine is rectify.map_point_via_affine
    assert shim.compute_rotated_box_corners is rectify.compute_rotated_box_corners

    # the two genuinely GUI-specific helpers stay here
    assert callable(shim.cv2_to_pil)
    assert callable(shim.cv2_to_tk)


def test_gui_package_has_no_duplicate_engine_class():
    """There must be exactly one MultiTargetWorkflow implementation in the repo."""
    import inspect

    import measure_gui.multi_target as shim

    assert inspect.getmodule(shim.MultiTargetWorkflow).__name__ == (
        "measure.multi_target_workflow"
    )