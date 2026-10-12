import numpy as np

from face_reconstruction.geometry.landmarks import (
    load_landmarks,
    save_landmarks,
    validate_landmarks,
)


def test_validate_landmarks_accepts_valid_array():
    landmarks = np.full((478, 2), [50.0, 50.0], dtype=float)

    # Give the synthetic face a realistic bounding box.
    landmarks[0] = [20.0, 50.0]
    landmarks[17] = [80.0, 50.0]
    landmarks[10] = [50.0, 20.0]   # forehead
    landmarks[152] = [50.0, 85.0]  # chin

    # Eyes: horizontally separated and approximately level.
    landmarks[[33, 133, 159, 145]] = [35.0, 40.0]
    landmarks[[362, 263, 386, 374]] = [65.0, 40.0]

    # Vertical facial anatomy.
    landmarks[1] = [50.0, 55.0]          # nose
    landmarks[[13, 14]] = [50.0, 70.0]   # mouth

    valid, reason = validate_landmarks(
        landmarks,
        image_shape=(100, 100, 3),
    )

    assert valid is True, reason
    assert reason == "ok"


def test_validate_landmarks_rejects_wrong_shape():
    landmarks = np.zeros((67, 2))

    valid, reason = validate_landmarks(
        landmarks,
        image_shape=(100, 100, 3),
    )

    assert valid is False
    assert reason == "unexpected_shape"


def test_validate_landmarks_rejects_points_outside_image():
    landmarks = np.zeros((478, 2))
    landmarks[:, 0] = 150
    landmarks[:, 1] = 50

    valid, reason = validate_landmarks(
        landmarks,
        image_shape=(100, 100, 3),
    )

    assert valid is False
    assert reason == "x_far_outside_image"


def test_save_and_load_landmarks(tmp_path):
    landmarks = np.random.default_rng(42).random((478, 2))
    output_path = tmp_path / "face.npy"

    save_landmarks(landmarks, output_path)
    loaded = load_landmarks(output_path)

    assert output_path.exists()
    assert loaded.shape == (478, 2)
    assert np.allclose(loaded, landmarks)