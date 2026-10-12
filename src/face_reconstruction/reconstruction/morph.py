import numpy as np

from .weighting import normalize_weights


def weighted_mean_landmarks(
    landmarks: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute the weighted mean facial shape.

    Parameters
    ----------
    landmarks:
        Array of shape (n_faces, n_landmarks, 2).
    weights:
        One non-negative weight per face.

    Returns
    -------
    np.ndarray
        Weighted mean landmarks of shape (n_landmarks, 2).
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float64,
    )

    if landmarks.ndim != 3 or landmarks.shape[2] != 2:
        raise ValueError(
            "landmarks must have shape "
            "(n_faces, n_landmarks, 2)."
        )

    if not np.isfinite(landmarks).all():
        raise ValueError(
            "landmarks must contain only finite coordinates."
        )

    normalized_weights = normalize_weights(weights)

    if landmarks.shape[0] != len(normalized_weights):
        raise ValueError(
            "Number of faces and number of weights must match."
        )

    return np.tensordot(
        normalized_weights,
        landmarks,
        axes=(0, 0),
    )


def build_target_points(
    landmarks: np.ndarray,
    weights: np.ndarray,
    boundary_anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build weighted target landmarks and topology points."""

    target_landmarks = weighted_mean_landmarks(
        landmarks,
        weights,
    )

    boundary_anchors = np.asarray(
        boundary_anchors,
        dtype=np.float64,
    )

    if (
        boundary_anchors.ndim != 2
        or boundary_anchors.shape[1] != 2
    ):
        raise ValueError(
            "boundary_anchors must have shape "
            "(n_anchors, 2)."
        )

    if not np.isfinite(boundary_anchors).all():
        raise ValueError(
            "boundary_anchors must contain finite coordinates."
        )

    target_points = np.vstack(
        [
            target_landmarks,
            boundary_anchors,
        ]
    )

    return target_landmarks, target_points