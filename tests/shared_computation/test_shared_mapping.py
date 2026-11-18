"""Exact transforms and sample metrology preserve geometry and measurement identity."""

import numpy as np
import pytest


def test_d02_translation_updates_line_coefficients_and_endpoints(api, space, observed):
    line = observed(api("computation_types", "LineValue")(
        (0.0, 1.0, -10.0), start=(2.0, 10.0), end=(8.0, 10.0),
    ), frame="patch")
    mapping = api("computation_types", "make_mapping")(
        np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 200.0], [0.0, 0.0, 1.0]]),
        source=space("patch"), destination=space(), kind="similarity",
    )
    result = api("geometry_mapping", "transform_geometry")(line, mapping)
    assert result.valid and result.value.coefficients == pytest.approx((0.0, 1.0, -210.0))
    assert {result.value.start, result.value.end} == {(102.0, 210.0), (108.0, 210.0)}


def test_d02_similarity_scales_circle_radius_as_well_as_center(api, space, observed):
    circle = observed(api("computation_types", "CircleValue")((3.0, 4.0), 5.0), frame="patch")
    mapping = api("computation_types", "make_mapping")(
        np.diag([2.0, 2.0, 1.0]), source=space("patch"), destination=space(),
        kind="similarity",
    )
    result = api("geometry_mapping", "transform_geometry")(circle, mapping)
    assert result.valid and result.value.center == pytest.approx((6.0, 8.0))
    assert result.value.radius == pytest.approx(10.0)


def test_affine_circle_cannot_be_misreported_as_an_exact_circle(api, space, observed):
    circle = observed(api("computation_types", "CircleValue")((3.0, 4.0), 5.0))
    mapping = api("computation_types", "make_mapping")(
        np.diag([2.0, 3.0, 1.0]), source=space(), destination=space("display"), kind="affine",
    )
    result = api("geometry_mapping", "transform_geometry")(circle, mapping)
    assert not result.valid and result.value is None
    assert result.reason == "unsupported_geometry_mapping"


def test_circle_display_approximation_does_not_claim_to_be_measurement_samples(
    api, space, observed,
):
    module = api("geometry_mapping")
    circle = observed(api("computation_types", "CircleValue")((0.0, 0.0), 5.0))
    mapping = api("computation_types", "make_mapping")(
        np.diag([2.0, 3.0, 1.0]), source=space(), destination=space("display"), kind="affine",
    )
    chain = module.MappingChain(steps=(module.MatrixStep(mapping=mapping),))
    result = module.map_geometry_for_display(circle, chain, curve_samples=16)
    assert result.valid and result.value.closed
    np.testing.assert_allclose(result.value.points_xy[0], [10.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(result.value.points_xy[4], [0.0, 15.0], atol=1e-10)
    assert result.diagnostics["approximation"] == "display_curve"


def test_mapping_wrong_source_frame_is_rejected(api, space, observed):
    mapping = api("computation_types", "make_mapping")(
        np.eye(3), source=space("patch"), destination=space(), kind="similarity",
    )
    with pytest.raises(api("computation_types", "ComputationContractError")):
        api("geometry_mapping", "transform_geometry")(observed((3.0, 4.0)), mapping)


def test_rotation_maps_a_box_center_dimensions_and_direction(api, space, observed):
    box = observed(api("computation_types", "RotatedBox")((3.0, 4.0), (8.0, 6.0), 0.0))
    mapping = api("computation_types", "make_mapping")(
        np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        source=space(), destination=space("rotated"), kind="similarity",
    )
    result = api("geometry_mapping", "transform_geometry")(box, mapping)
    assert result.valid and result.value.center == pytest.approx((-8.0, 6.0))
    assert result.value.size_wh == pytest.approx((16.0, 12.0))
    assert result.value.angle_rad == pytest.approx(np.pi / 2)


def test_sample_chain_keeps_order_and_converts_pixel_lengths_to_millimeters(api, space):
    module = api("geometry_mapping")
    matrix = np.array([[0.5, 0.0, 1.0], [0.0, 0.5, 2.0], [0.0, 0.0, 1.0]])
    mapping = api("computation_types", "make_mapping")(
        matrix, source=space(), destination=space("plane", "mm"), kind="similarity",
    )
    chain = module.MappingChain(steps=(module.MatrixStep(mapping=mapping),))
    original = np.array([[8.0, 4.0], [0.0, 0.0], [2.0, 6.0]])
    result = module.map_sample_points(original, chain)
    assert result.valid and result.space == space("plane", "mm")
    np.testing.assert_allclose(result.value, [[5.0, 4.0], [1.0, 2.0], [2.0, 5.0]])
    np.testing.assert_array_equal(original, [[8.0, 4.0], [0.0, 0.0], [2.0, 6.0]])


def test_empty_point_mapping_is_valid_and_retains_shape(api, space):
    module = api("geometry_mapping")
    mapping = api("computation_types", "make_mapping")(
        np.eye(3), source=space(), destination=space("plane", "mm"), kind="similarity",
    )
    result = module.map_sample_points(
        np.empty((0, 2)), module.MappingChain((module.MatrixStep(mapping),)),
    )
    assert result.valid and result.value.shape == (0, 2)


def test_chain_cannot_join_incompatible_intermediate_spaces(api, space):
    module = api("geometry_mapping")
    first = api("computation_types", "make_mapping")(
        np.eye(3), source=space("patch"), destination=space("source"), kind="similarity",
    )
    second = api("computation_types", "make_mapping")(
        np.eye(3), source=space("other-camera"), destination=space("plane", "mm"),
        kind="similarity",
    )
    with pytest.raises(api("computation_types", "ComputationContractError")):
        module.MappingChain((module.MatrixStep(first), module.MatrixStep(second)))


def test_projective_horizon_does_not_silently_drop_one_sample(api, space):
    module = api("geometry_mapping")
    mapping = api("computation_types", "make_mapping")(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]]),
        source=space(), destination=space("plane", "mm"), kind="projective",
    )
    result = module.map_sample_points(
        np.array([[0.0, 2.0], [1.0, 2.0]]), module.MappingChain((module.MatrixStep(mapping),)),
    )
    assert not result.valid and result.value is None


def test_zero_distortion_is_an_identity_in_pixel_coordinates(api, space, camera_model):
    module = api("geometry_mapping")
    camera = camera_model()
    step = module.LensStep(
        source=space(), destination=space("left:pinhole"), camera_matrix=camera.matrix,
        distortion=camera.distortion, direction="undistort",
    )
    result = module.map_sample_points(np.array([[10.0, 20.0], [160.0, 120.0]]),
                                     module.MappingChain((step,)))
    assert result.valid
    np.testing.assert_allclose(result.value, [[10.0, 20.0], [160.0, 120.0]], atol=1e-9)


@pytest.mark.parametrize("direction,input_xy,expected_xy", [
    ("undistort", [[192.16, 180.3]], [[192.0, 180.0]]),
    ("distort", [[192.0, 180.0]], [[192.16, 180.3]]),
])
def test_radial_lens_mapping_obeys_direction_in_pixel_coordinates(
    api, space, camera_model, direction, input_xy, expected_xy,
):
    module = api("geometry_mapping")
    camera = camera_model()
    step = module.LensStep(
        source=space("input"), destination=space("output"), camera_matrix=camera.matrix,
        distortion=np.array([0.1, 0.0, 0.0, 0.0, 0.0]), direction=direction,
    )
    result = module.map_sample_points(np.array(input_xy), module.MappingChain((step,)))
    assert result.valid
    np.testing.assert_allclose(result.value, expected_xy, atol=1e-5)


def test_multiple_mapping_steps_run_in_declared_order(api, space):
    module = api("geometry_mapping")
    make = api("computation_types", "make_mapping")
    first = make(
        np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]]),
        source=space("source"), destination=space("middle"), kind="similarity",
    )
    second = make(
        np.diag([2.0, 2.0, 1.0]), source=space("middle"), destination=space("target"),
        kind="similarity",
    )
    result = module.map_sample_points(
        np.array([[2.0, 3.0]]),
        module.MappingChain((module.MatrixStep(first), module.MatrixStep(second))),
    )
    assert result.valid and result.space == space("target")
    np.testing.assert_allclose(result.value, [[10.0, 14.0]])


def test_metric_circle_is_refitted_from_selected_samples(api, space, observed):
    module = api("geometry_mapping")
    mapping = api("computation_types", "make_mapping")(
        np.diag([2.0, 2.0, 1.0]), source=space(), destination=space("plane", "mm"),
        kind="similarity",
    )
    result = module.map_metric_samples(
        observed(np.array([[8.0, 4.0], [3.0, 9.0], [-2.0, 4.0], [3.0, -1.0]])),
        module.MappingChain((module.MatrixStep(mapping),)), target_model="circle",
    )
    assert result.valid and result.value.geometry.center == pytest.approx((6.0, 8.0))
    assert result.value.geometry.radius == pytest.approx(10.0)
    assert result.diagnostics["fit"].residual_space == space("plane", "mm")


def test_metric_line_residual_is_recomputed_in_target_units(api, space, observed):
    module = api("geometry_mapping")
    mapping = api("computation_types", "make_mapping")(
        np.diag([0.5, 0.5, 1.0]), source=space(), destination=space("plane", "mm"),
        kind="similarity",
    )
    result = module.map_metric_samples(
        observed(np.array([[0.0, 9.0], [0.0, 11.0], [20.0, 9.0], [20.0, 11.0]])),
        module.MappingChain((module.MatrixStep(mapping),)), target_model="line",
    )
    assert result.valid and result.diagnostics["fit"].rms_error == pytest.approx(0.5)


def test_metric_points_mode_does_not_invent_a_fitted_model(api, space, observed):
    module = api("geometry_mapping")
    mapping = api("computation_types", "make_mapping")(
        np.eye(3), source=space(), destination=space("plane", "mm"), kind="similarity",
    )
    result = module.map_metric_samples(
        observed(np.array([[2.0, 3.0]])), module.MappingChain((module.MatrixStep(mapping),)),
        target_model="points",
    )
    assert result.valid and result.value.model == "points" and result.value.geometry is None
    np.testing.assert_allclose(result.value.samples_xy, [[2.0, 3.0]])
