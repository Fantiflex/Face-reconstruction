import numpy as np

EYE_GROUP_A = np.array([33, 133, 159, 145], dtype=np.int32)
EYE_GROUP_B = np.array([362, 263, 386, 374], dtype=np.int32)


def eye_centres(
    landmarks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eye centres ordered from image-left to image-right."""
    centre_a = landmarks[EYE_GROUP_A].mean(axis=0)
    centre_b = landmarks[EYE_GROUP_B].mean(axis=0)

    if float(centre_a[0]) <= float(centre_b[0]):
        return centre_a, centre_b

    return centre_b, centre_a


def similarity_transform_from_eyes(
    source_left: np.ndarray,
    source_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    source_vector = np.asarray(
        source_right - source_left,
        dtype=np.float64,
    )
    target_vector = np.asarray(
        target_right - target_left,
        dtype=np.float64,
    )

    source_distance = float(np.linalg.norm(source_vector))
    target_distance = float(np.linalg.norm(target_vector))

    if not np.isfinite(source_distance) or source_distance <= 1.0:
        raise ValueError("Source interocular distance is degenerate.")

    if not np.isfinite(target_distance) or target_distance <= 1.0:
        raise ValueError("Target interocular distance is degenerate.")

    source_angle = float(
        np.arctan2(source_vector[1], source_vector[0])
    )
    target_angle = float(
        np.arctan2(target_vector[1], target_vector[0])
    )

    rotation_radians = target_angle - source_angle
    scale = target_distance / source_distance

    cosine = float(np.cos(rotation_radians))
    sine = float(np.sin(rotation_radians))

    linear = scale * np.array(
        [
            [cosine, -sine],
            [sine, cosine],
        ],
        dtype=np.float64,
    )

    translation = (
        target_left
        - linear @ np.asarray(source_left, dtype=np.float64)
    )

    matrix = np.column_stack(
        [linear, translation]
    ).astype(np.float64)

    return (
        matrix,
        scale,
        float(np.degrees(rotation_radians)),
    )


def transform_landmarks(
    landmarks: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    homogeneous = np.column_stack(
        [
            np.asarray(landmarks, dtype=np.float64),
            np.ones(len(landmarks)),
        ]
    )

    transformed = homogeneous @ matrix.T

    return np.asarray(
        transformed,
        dtype=np.float32,
    )


def in_bounds_fraction(
    landmarks: np.ndarray,
    *,
    width: int,
    height: int,
) -> float:
    inside = (
        (landmarks[:, 0] >= 0.0)
        & (landmarks[:, 0] <= width - 1)
        & (landmarks[:, 1] >= 0.0)
        & (landmarks[:, 1] <= height - 1)
    )

    return float(inside.mean())