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


def test_load_reconstruction_inputs(tmp_path):
    import cv2
    import pandas as pd

    from face_reconstruction.reconstruction.morph import (
        load_reconstruction_inputs,
    )

    aligned_dir = tmp_path / "aligned"
    aligned_dir.mkdir()

    image = np.full(
        (20, 20, 3),
        128,
        dtype=np.uint8,
    )

    image_path = aligned_dir / "face.png"

    assert cv2.imwrite(
        str(image_path),
        image,
    )

    landmarks = np.array(
        [
            [5.0, 5.0],
            [10.0, 10.0],
            [15.0, 15.0],
        ],
        dtype=np.float64,
    )

    landmark_path = (
        aligned_dir / "face_landmarks.npy"
    )

    np.save(
        landmark_path,
        landmarks,
        allow_pickle=False,
    )

    eligible = pd.DataFrame(
        {
            "face_id": ["face_1"],
            "aligned_image_path": [
                str(image_path)
            ],
            "aligned_landmark_path": [
                str(landmark_path)
            ],
            "unstable_triangle_indices_parsed": [
                set()
            ],
        }
    )

    loaded = load_reconstruction_inputs(
        eligible,
        aligned_dir=aligned_dir,
        selected_indices=np.array(
            [0, 2],
            dtype=np.int32,
        ),
        n_detected=3,
        n_selected=2,
        width=20,
        height=20,
    )

    assert len(loaded) == 1

    (
        face_id,
        _,
        _,
        loaded_image,
        selected_landmarks,
        unstable,
    ) = loaded[0]

    assert face_id == "face_1"
    assert loaded_image.shape == (20, 20, 3)

    assert np.allclose(
        selected_landmarks,
        [
            [5.0, 5.0],
            [15.0, 15.0],
        ],
    )

    assert unstable == set()