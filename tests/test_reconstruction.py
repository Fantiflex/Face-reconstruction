import numpy as np
import pytest
from pathlib import Path

from face_reconstruction.reconstruction.morph import (
    accumulate_weighted_warp,
    finalize_weighted_reconstruction,
)


def test_accumulate_weighted_warp():
    numerator = np.zeros((2, 2, 3))
    denominator = np.zeros((2, 2))

    warped = np.full(
        (2, 2, 3),
        10.0,
    )

    covered = np.array(
        [
            [True, False],
            [True, True],
        ]
    )

    accumulate_weighted_warp(
        numerator,
        denominator,
        warped,
        covered,
        0.5,
    )

    assert np.allclose(
        denominator,
        [
            [0.5, 0.0],
            [0.5, 0.5],
        ],
    )

    assert np.allclose(
        numerator[0, 0],
        [5.0, 5.0, 5.0],
    )


def test_finalize_weighted_reconstruction():
    numerator = np.array(
        [
            [
                [10.0, 10.0, 10.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )

    denominator = np.array(
        [
            [0.5, 0.0],
        ]
    )

    fallback = np.array(
        [
            [
                [100.0, 100.0, 100.0],
                [50.0, 50.0, 50.0],
            ]
        ]
    )

    reconstruction, covered, uncovered_fraction = (
        finalize_weighted_reconstruction(
            numerator,
            denominator,
            fallback,
            max_uncovered_fraction=0.5,
        )
    )

    assert np.allclose(
        reconstruction[0, 0],
        [20.0, 20.0, 20.0],
    )

    assert np.allclose(
        reconstruction[0, 1],
        [50.0, 50.0, 50.0],
    )

    assert covered.tolist() == [
        [True, False]
    ]

    assert uncovered_fraction == 0.5


def test_finalize_rejects_too_many_uncovered_pixels():
    numerator = np.zeros((2, 2, 3))
    denominator = np.zeros((2, 2))
    denominator[0, 0] = 1.0

    fallback = np.zeros((2, 2, 3))

    with pytest.raises(
        RuntimeError,
        match="Too many reconstruction pixels",
    ):
        finalize_weighted_reconstruction(
            numerator,
            denominator,
            fallback,
            max_uncovered_fraction=0.5,
        )


def test_reconstruct_weighted_face_identity():
    import cv2

    from face_reconstruction.reconstruction.morph import (
        reconstruct_weighted_face,
    )

    image_a = np.full(
        (20, 20, 3),
        100,
        dtype=np.uint8,
    )

    image_b = np.full(
        (20, 20, 3),
        200,
        dtype=np.uint8,
    )

    landmarks = np.array(
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

    anchors = np.empty(
        (0, 2),
        dtype=np.float64,
    )

    loaded = [
        (
            "face_a",
            Path("face_a.png"),
            Path("face_a.npy"),
            image_a,
            landmarks,
            set(),
        ),
        (
            "face_b",
            Path("face_b.png"),
            Path("face_b.npy"),
            image_b,
            landmarks,
            set(),
        ),
    ]

    (
        reconstruction,
        covered,
        denominator,
        uncovered_fraction,
        weight_rows,
        skip_rows,
    ) = reconstruct_weighted_face(
        loaded,
        np.array([0.25, 0.75]),
        landmarks,
        triangles,
        anchors,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        minimum_absolute_area=0.01,
        relative_area_tolerance=1e-4,
        maximum_condition_number=1e4,
        max_uncovered_fraction=0.0,
    )

    assert covered.all()
    assert uncovered_fraction == 0.0
    assert skip_rows == []
    assert len(weight_rows) == 2

    assert np.allclose(
        reconstruction,
        175.0,
    )

    assert np.allclose(
        denominator,
        1.0,
    )