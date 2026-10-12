from pathlib import Path

import numpy as np
LANDMARK_COUNT = 478

def validate_landmarks(
    landmarks: np.ndarray,
    image_shape: tuple[int, ...],
    expected_n: int = LANDMARK_COUNT,
) -> tuple[bool, str]:
    landmarks = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    if landmarks.shape != (expected_n, 2):
        return False, "unexpected_shape"

    if not np.isfinite(landmarks).all():
        return False, "non_finite_coordinates"

    image_height, image_width = image_shape[:2]
    x = landmarks[:, 0]
    y = landmarks[:, 1]

    margin_x = 0.02 * image_width
    margin_y = 0.02 * image_height

    if (x < -margin_x).any() or (x > image_width - 1 + margin_x).any():
        return False, "x_far_outside_image"

    if (y < -margin_y).any() or (y > image_height - 1 + margin_y).any():
        return False, "y_far_outside_image"

    face_width = float(np.ptp(x))
    face_height = float(np.ptp(y))

    if face_width < 0.12 * image_width:
        return False, "face_bbox_too_narrow"

    if face_height < 0.12 * image_height:
        return False, "face_bbox_too_short"

    if face_width > 0.95 * image_width:
        return False, "face_bbox_too_wide"

    if face_height > 0.98 * image_height:
        return False, "face_bbox_too_tall"

    left_eye = landmarks[[33, 133, 159, 145]].mean(axis=0)
    right_eye = landmarks[[362, 263, 386, 374]].mean(axis=0)

    eyes_y = float((left_eye[1] + right_eye[1]) / 2.0)

    interocular = float(
        np.linalg.norm(right_eye - left_eye)
    )

    if interocular < 0.08 * image_width:
        return False, "interocular_distance_too_small"

    if interocular > 0.55 * image_width:
        return False, "interocular_distance_too_large"

    if abs(float(left_eye[1] - right_eye[1])) > 0.20 * interocular:
        return False, "eyes_excessively_tilted"

    forehead_y = float(landmarks[10, 1])
    nose_y = float(landmarks[1, 1])
    mouth_y = float(landmarks[[13, 14], 1].mean())
    chin_y = float(landmarks[152, 1])

    if not forehead_y < eyes_y < nose_y < mouth_y < chin_y:
        return False, "implausible_vertical_anatomy"

    return True, "ok"

def save_landmarks(
    landmarks: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        np.asarray(landmarks, dtype=np.float32),
    )


def load_landmarks(
    landmark_path: Path,
) -> np.ndarray:
    landmarks = np.load(landmark_path)

    if landmarks.ndim != 2 or landmarks.shape[1] != 2:
        raise ValueError(
            f"Invalid landmark array shape: {landmarks.shape}"
        )

    return landmarks.astype(np.float32)