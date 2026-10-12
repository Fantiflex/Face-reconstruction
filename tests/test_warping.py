import cv2
import numpy as np

from face_reconstruction.geometry.warping import (
    affine_for_triangle,
    signed_triangle_area,
    warp_piecewise_affine,
)


def test_signed_triangle_area_positive_for_ccw_triangle():
    triangle = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ]
    )

    area = signed_triangle_area(triangle)

    assert area > 0


def test_affine_for_triangle_maps_valid_triangle():
    source = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ]
    )

    target = np.array(
        [
            [5.0, 5.0],
            [15.0, 5.0],
            [5.0, 15.0],
        ]
    )

    matrix, reason = affine_for_triangle(
        source,
        target,
        minimum_absolute_area=0.01,
        relative_area_tolerance=1e-4,
        maximum_condition_number=1e4,
    )

    assert matrix is not None
    assert reason is None
    assert matrix.shape == (2, 3)


def test_affine_for_triangle_rejects_orientation_flip():
    source = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ]
    )

    target = np.array(
        [
            [0.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
        ]
    )

    matrix, reason = affine_for_triangle(
        source,
        target,
        minimum_absolute_area=0.01,
        relative_area_tolerance=1e-4,
        maximum_condition_number=1e4,
    )

    assert matrix is None
    assert reason == "orientation_flip"


def test_piecewise_affine_identity_warp():
    image = np.full((20, 20, 3), 128, dtype=np.uint8)

    points = np.array(
        [
            [0.0, 0.0],
            [19.0, 0.0],
            [19.0, 19.0],
            [0.0, 19.0],
        ]
    )

    triangles = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int32,
    )

    warped, covered, skipped = warp_piecewise_affine(
        image,
        points,
        points,
        triangles,
        skipped_triangle_indices=set(),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        minimum_absolute_area=0.01,
        relative_area_tolerance=1e-4,
        maximum_condition_number=1e4,
    )

    assert covered.all()
    assert skipped == []
    assert np.allclose(warped, image)
    