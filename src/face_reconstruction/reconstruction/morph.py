import numpy as np

from .weighting import normalize_weights
from pathlib import Path

import cv2
import pandas as pd

from face_reconstruction.utils.io import resolve_path

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


def accumulate_weighted_warp(
    numerator: np.ndarray,
    denominator: np.ndarray,
    warped: np.ndarray,
    covered: np.ndarray,
    weight: float,
) -> None:
    """Accumulate one warped face into a locally weighted reconstruction."""

    if numerator.ndim != 3:
        raise ValueError("numerator must be a 3D image array.")

    if denominator.shape != numerator.shape[:2]:
        raise ValueError(
            "denominator shape must match numerator spatial dimensions."
        )

    if warped.shape != numerator.shape:
        raise ValueError(
            "warped image shape must match numerator shape."
        )

    if covered.shape != numerator.shape[:2]:
        raise ValueError(
            "covered mask shape must match image dimensions."
        )

    if not np.isfinite(weight) or weight < 0:
        raise ValueError(
            "weight must be finite and non-negative."
        )

    numerator[covered] += (
        weight * warped[covered].astype(np.float64)
    )

    denominator[covered] += weight



def finalize_weighted_reconstruction(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fallback_texture: np.ndarray,
    *,
    max_uncovered_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize accumulated warped textures pixel by pixel."""

    numerator = np.asarray(
        numerator,
        dtype=np.float64,
    )
    denominator = np.asarray(
        denominator,
        dtype=np.float64,
    )
    fallback_texture = np.asarray(
        fallback_texture,
        dtype=np.float64,
    )

    if numerator.ndim != 3:
        raise ValueError(
            "numerator must be a 3D image array."
        )

    if denominator.shape != numerator.shape[:2]:
        raise ValueError(
            "denominator shape must match numerator."
        )

    if fallback_texture.shape != numerator.shape:
        raise ValueError(
            "fallback_texture must match numerator shape."
        )

    if not 0.0 <= max_uncovered_fraction <= 1.0:
        raise ValueError(
            "max_uncovered_fraction must be between 0 and 1."
        )

    covered = denominator > 0.0

    if not covered.any():
        raise RuntimeError(
            "No output pixel is covered by a positive-weight face."
        )

    uncovered_fraction = float(
        1.0 - covered.mean()
    )

    if uncovered_fraction > max_uncovered_fraction:
        raise RuntimeError(
            "Too many reconstruction pixels are uncovered: "
            f"{uncovered_fraction:.6f}."
        )

    reconstruction = np.zeros_like(
        numerator,
        dtype=np.float64,
    )

    reconstruction[covered] = (
        numerator[covered]
        / denominator[covered, None]
    )

    if (~covered).any():
        reconstruction[~covered] = (
            fallback_texture[~covered]
        )

    return (
        reconstruction,
        covered,
        uncovered_fraction,
    )





def load_reconstruction_inputs(
    eligible: pd.DataFrame,
    *,
    aligned_dir: Path,
    selected_indices: np.ndarray,
    n_detected: int,
    n_selected: int,
    width: int,
    height: int,
) -> list[
    tuple[
        str,
        Path,
        Path,
        np.ndarray,
        np.ndarray,
        set[int],
    ]
]:
    """Load aligned images and selected landmarks for reconstruction."""

    alignment_manifest_path = (
        aligned_dir / "alignment_manifest.csv"
    )

    loaded = []

    for row in eligible.itertuples(index=False):
        image_path = resolve_path(
            str(row.aligned_image_path),
            alignment_manifest_path,
        )

        landmark_path = resolve_path(
            str(row.aligned_landmark_path),
            alignment_manifest_path,
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                f"OpenCV could not read aligned image: "
                f"{image_path}"
            )

        if image.shape[:2] != (height, width):
            raise ValueError(
                f"Image {row.face_id} has shape "
                f"{image.shape[:2]}, expected "
                f"{(height, width)}."
            )

        landmarks = np.asarray(
            np.load(
                landmark_path,
                allow_pickle=False,
            ),
            dtype=np.float64,
        )

        if landmarks.shape != (n_detected, 2):
            raise ValueError(
                f"Landmarks for {row.face_id} "
                f"have shape {landmarks.shape}; "
                f"expected {(n_detected, 2)}."
            )

        if not np.isfinite(landmarks).all():
            raise ValueError(
                f"Non-finite landmarks for "
                f"{row.face_id}."
            )

        selected_landmarks = landmarks[
            selected_indices
        ]

        if selected_landmarks.shape != (
            n_selected,
            2,
        ):
            raise ValueError(
                f"Landmark selection failed for "
                f"{row.face_id}: "
                f"{selected_landmarks.shape}."
            )

        unstable = getattr(
            row,
            "unstable_triangle_indices_parsed",
            set(),
        )

        loaded.append(
            (
                str(row.face_id),
                image_path,
                landmark_path,
                image,
                selected_landmarks,
                set(unstable),
            )
        )

    if not loaded:
        raise ValueError(
            "No eligible faces available for reconstruction."
        )

    return loaded