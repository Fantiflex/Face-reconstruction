import numpy as np
import pytest

from face_reconstruction.reconstruction.morph import (
    build_target_points,
    weighted_mean_landmarks,
)


def test_weighted_mean_landmarks():
    landmarks = np.array(
        [
            [
                [0.0, 0.0],
                [10.0, 10.0],
            ],
            [
                [10.0, 20.0],
                [20.0, 30.0],
            ],
        ]
    )

    weights = np.array([0.25, 0.75])

    result = weighted_mean_landmarks(
        landmarks,
        weights,
    )

    expected = np.array(
        [
            [7.5, 15.0],
            [17.5, 25.0],
        ]
    )

    assert np.allclose(result, expected)


def test_weighted_mean_landmarks_normalizes_weights():
    landmarks = np.array(
        [
            [[0.0, 0.0]],
            [[10.0, 20.0]],
        ]
    )

    result = weighted_mean_landmarks(
        landmarks,
        np.array([1.0, 3.0]),
    )

    assert np.allclose(
        result,
        [[7.5, 15.0]],
    )


def test_weighted_mean_rejects_mismatched_weights():
    landmarks = np.zeros((3, 10, 2))

    with pytest.raises(
        ValueError,
        match="Number of faces",
    ):
        weighted_mean_landmarks(
            landmarks,
            [0.5, 0.5],
        )


def test_build_target_points_appends_anchors():
    landmarks = np.array(
        [
            [
                [10.0, 10.0],
                [20.0, 20.0],
            ],
            [
                [30.0, 30.0],
                [40.0, 40.0],
            ],
        ]
    )

    anchors = np.array(
        [
            [0.0, 0.0],
            [99.0, 99.0],
        ]
    )

    target_landmarks, target_points = build_target_points(
        landmarks,
        [0.5, 0.5],
        anchors,
    )

    assert target_landmarks.shape == (2, 2)
    assert target_points.shape == (4, 2)
    assert np.allclose(
        target_points[-2:],
        anchors,
    )