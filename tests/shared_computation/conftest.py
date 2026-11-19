"""Lazy public-interface access and deterministic in-memory contract data."""

import importlib

import cv2
import numpy as np
import pytest


@pytest.fixture
def api():
    """Resolve proposed interfaces during execution so missing APIs fail individual tests."""
    def load(module, name=None):
        qualified = "measure." + module
        try:
            value = importlib.import_module(qualified)
        except ModuleNotFoundError as error:
            if error.name != qualified:
                raise
            pytest.fail("Unimplemented contract module: " + qualified, pytrace=False)
        if name is None:
            return value
        if not hasattr(value, name):
            pytest.fail("Unimplemented contract interface: " + qualified + "." + name,
                        pytrace=False)
        return getattr(value, name)

    return load


@pytest.fixture
def space(api):
    def make(frame="camera:left/source", unit="px"):
        return api("computation_types", "Space")(frame=frame, unit=unit)

    return make


@pytest.fixture
def observed(api, space):
    def make(value=None, *, frame="camera:left/source", unit="px", reason=None):
        return api("computation_types", "Observation")(
            space=space(frame, unit), valid=reason is None, value=value, reason=reason,
        )

    return make


@pytest.fixture
def camera_model(api, space):
    def make(camera_id="left", width=320, height=240):
        return api("measure_calibration", "CameraModel")(
            camera_id=camera_id,
            image_space=space("camera:" + camera_id + "/source"),
            image_size_wh=(width, height),
            matrix=np.array([[320.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]),
            distortion=np.zeros(5),
        )

    return make


@pytest.fixture
def board_views(api, space):
    """Generate perspective observations from known physical camera parameters."""
    def make(camera, *, camera_shift_x=0.0):
        objects = np.array([(x * 4.0, y * 4.0, 0.0)
                            for y in range(5) for x in range(6)])
        rotations = [(0.0, 0.0, 0.0), (0.25, -0.2, 0.1),
                     (-0.2, 0.3, -0.1), (0.35, 0.15, 0.2), (-0.3, -0.25, 0.15)]
        views = []
        for index, vector in enumerate(rotations):
            rotation = cv2.Rodrigues(np.array(vector, dtype=float))[0]
            translation = np.array([-10.0 + index + camera_shift_x,
                                    -8.0 + index * 0.5, 100.0 + index * 10])
            points = objects @ rotation.T + translation
            homogeneous = points @ camera.matrix.T
            pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
            views.append(api("measure_calibration", "BoardObservation")(
                view_id="capture-" + str(index), image_space=camera.image_space,
                object_space=space("board", "mm"), image_size_wh=camera.image_size_wh,
                image_points_xy=pixels, object_points_xyz=objects.copy(),
                ordering="row_major",
            ))
        return tuple(views)

    return make


@pytest.fixture
def ramp_image():
    return np.tile(np.arange(100, dtype=np.uint8), (80, 1))


@pytest.fixture
def band_image():
    image = np.zeros((80, 100), np.uint8)
    for left in (10, 30, 50, 70):
        image[:, left:left + 10] = 255
    return image


@pytest.fixture
def textured_image():
    image = np.zeros((160, 200), np.uint8)
    rng = np.random.default_rng(20261118)
    image[50:90, 60:100] = rng.integers(20, 240, (40, 40), dtype=np.uint8)
    return image
