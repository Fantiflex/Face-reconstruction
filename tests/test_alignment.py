import numpy as np

from face_reconstruction.geometry.alignment import (
    eye_centres,
    similarity_transform_from_eyes,
    transform_landmarks,
)


def test_eye_centres_are_ordered_left_to_right():
    landmarks = np.zeros((478, 2), dtype=float)

    landmarks[[33, 133, 159, 145]] = [70.0, 40.0]
    landmarks[[362, 263, 386, 374]] = [30.0, 40.0]

    left, right = eye_centres(landmarks)

    assert left[0] < right[0]


def test_similarity_transform_maps_eye_centres_to_targets():
    source_left = np.array([30.0, 40.0])
    source_right = np.array([70.0, 40.0])

    target_left = np.array([35.0, 38.0])
    target_right = np.array([65.0, 38.0])

    matrix, scale, rotation = similarity_transform_from_eyes(
        source_left,
        source_right,
        target_left,
        target_right,
    )

    transformed = transform_landmarks(
        np.array([source_left, source_right]),
        matrix,
    )

    assert np.allclose(transformed[0], target_left)
    assert np.allclose(transformed[1], target_right)
    assert scale > 0
    assert np.isfinite(rotation)