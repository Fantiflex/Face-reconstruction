import json
from typing import Any

import numpy as np
from scipy.spatial import Delaunay, QhullError

# The 478-point Face Landmarker output remains untouched in steps 06-08.
# For piecewise-affine morphing, we exclude points whose geometry varies for
# reasons that are not stable identity cues:
#   - the inner lip contour can become almost collinear when the mouth closes;
#   - iris positions vary with gaze and create very thin local triangles.
# The retained 444 points are selected identically for every face.
INNER_LIP_INDICES = frozenset(
    {
        13, 14, 78, 80, 81, 82, 87, 88, 95, 178,
        191, 308, 310, 311, 312, 317, 318, 324, 402, 415,
    }
)
IRIS_INDICES = frozenset(range(468, 478))
ADDITIONAL_UNSTABLE_INDICES = {
    323,
    361,
    454, 
    358,
}



EXCLUDED_LANDMARK_INDICES = (
    INNER_LIP_INDICES
    | IRIS_INDICES
    | ADDITIONAL_UNSTABLE_INDICES
)



def build_selected_landmark_indices(expected_landmarks: int) -> np.ndarray:
    """Return stable MediaPipe indices, preserving their original order."""
    invalid = sorted(
        index
        for index in EXCLUDED_LANDMARK_INDICES
        if index < 0 or index >= expected_landmarks
    )
    if invalid:
        raise ValueError(
            "The configured exclusion set is incompatible with "
            f"--expected-landmarks={expected_landmarks}: {invalid}"
        )
    selected = np.asarray(
        [
            index
            for index in range(expected_landmarks)
            if index not in EXCLUDED_LANDMARK_INDICES
        ],
        dtype=np.int32,
    )
    if selected.size == 0:
        raise ValueError("Landmark selection is empty.")
    return selected



def boundary_points(width: int, height: int, margin: float) -> np.ndarray:
    left, top = margin, margin
    right, bottom = width - 1.0 - margin, height - 1.0 - margin
    if not (0.0 <= left < right and 0.0 <= top < bottom):
        raise ValueError("--boundary-margin leaves no valid canvas interior.")
    mid_x = (left + right) / 2.0
    mid_y = (top + bottom) / 2.0
    return np.array(
        [
            [left, top], [mid_x, top], [right, top], [right, mid_y],
            [right, bottom], [mid_x, bottom], [left, bottom], [left, mid_y],
        ],
        dtype=np.float64,
    )



def signed_double_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (
        b[:, 1] - a[:, 1]
    ) * (c[:, 0] - a[:, 0])


def orient_triangles_ccw(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    result = np.asarray(triangles, dtype=np.int32).copy()
    negative = signed_double_areas(points, result) < 0.0
    result[negative, 1], result[negative, 2] = (
        result[negative, 2].copy(), result[negative, 1].copy()
    )
    return result



def triangle_geometry(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = points[triangles]

    ab = vertices[:, 1] - vertices[:, 0]
    ac = vertices[:, 2] - vertices[:, 0]
    bc = vertices[:, 2] - vertices[:, 1]

    signed_area = (
        ab[:, 0] * ac[:, 1]
        - ab[:, 1] * ac[:, 0]
    ) / 2.0

    squared_lengths = np.stack(
        [
            np.sum(ab * ab, axis=1),
            np.sum(ac * ac, axis=1),
            np.sum(bc * bc, axis=1),
        ],
        axis=1,
    )

    max_edge_squared = squared_lengths.max(axis=1)

    # 1 = triangle équilatéral ; 0 = triangle aplati
    quality = (
        4.0
        * np.sqrt(3.0)
        * np.abs(signed_area)
        / np.maximum(squared_lengths.sum(axis=1), 1e-12)
    )

    return signed_area, max_edge_squared, quality



def validate_face_topology(
    points: np.ndarray,
    reference_points: np.ndarray,
    triangles: np.ndarray,
    selected_indices: np.ndarray,
    *,
    min_absolute_area: float,
    relative_area_tolerance: float,
    min_triangle_quality: float,
    max_flipped_fraction: float,
    max_degenerate_fraction: float,
) -> dict[str, Any]:
    face_area, max_edge_squared, quality = triangle_geometry(
        points,
        triangles,
    )
    reference_area, _, _ = triangle_geometry(
        reference_points,
        triangles,
    )

    area_tolerance = np.maximum(
        min_absolute_area,
        relative_area_tolerance * max_edge_squared,
    )

    degenerate = (
        (np.abs(face_area) <= area_tolerance)
        | (quality < min_triangle_quality)
    )

    # On ne classe pas comme inversé un triangle déjà quasi dégénéré.
    flipped = (
        (face_area * reference_area < 0.0)
        & ~degenerate
    )

    unstable = degenerate | flipped
    problem_triangle_indices = np.flatnonzero(unstable)

    if len(problem_triangle_indices) > 0:
        problem_point_indices = np.unique(
            triangles[problem_triangle_indices].ravel()
        )
    else:
        problem_point_indices = np.array([], dtype=int)

    n_landmarks = len(selected_indices)

    problem_mediapipe_indices: list[int | str] = []

    for point_index in problem_point_indices:
        point_index = int(point_index)

        if point_index < n_landmarks:
            problem_mediapipe_indices.append(
                int(selected_indices[point_index])
            )
        else:
            anchor_index = point_index - n_landmarks
            problem_mediapipe_indices.append(
                f"anchor_{anchor_index}"
            )

    flipped_fraction = float(flipped.mean())
    degenerate_fraction = float(degenerate.mean())

    reasons: list[str] = []
    if flipped_fraction > max_flipped_fraction:
        reasons.append("flipped_triangles")
    if degenerate_fraction > max_degenerate_fraction:
        reasons.append("degenerate_triangles")

    return {
        "n_triangles": len(triangles),
        "n_flipped_triangles": int(flipped.sum()),
        "flipped_triangle_fraction": flipped_fraction,
        "flipped_triangle_indices": json.dumps(
            np.flatnonzero(flipped).tolist()
        ),
        "n_degenerate_triangles": int(degenerate.sum()),
        "degenerate_triangle_fraction": degenerate_fraction,
        "degenerate_triangle_indices": json.dumps(
            np.flatnonzero(degenerate).tolist()
        ),
        "unstable_triangle_indices": json.dumps(
            np.flatnonzero(unstable).tolist()
        ),
        "min_absolute_triangle_area_px2": float(
            np.abs(face_area).min()
        ),
        "min_triangle_quality": float(quality.min()),
        "median_triangle_quality": float(np.median(quality)),
        "topology_qc_pass": not reasons,
        "topology_qc_reason": (
            "ok" if not reasons else ";".join(reasons)
        ),
        
        "problem_point_indices": json.dumps(
            problem_point_indices.tolist()
        ),
        "problem_mediapipe_indices": json.dumps(
            problem_mediapipe_indices
        ),
        "problem_triangle_indices": json.dumps(
            problem_triangle_indices.tolist()
        ),
    }




def build_delaunay_triangulation(
    points: np.ndarray,
    *,
    min_triangle_area: float = 0.05,
) -> np.ndarray:
    """Build a counter-clockwise Delaunay triangulation."""

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_points, 2).")

    if len(points) < 3:
        raise ValueError("At least three points are required.")

    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates.")

    if min_triangle_area < 0:
        raise ValueError("min_triangle_area cannot be negative.")

    if len(np.unique(np.round(points, decimals=6), axis=0)) != len(points):
        raise ValueError(
            "points contain duplicates at 1e-6 precision."
        )

    try:
        triangles = Delaunay(points).simplices.astype(np.int32)
    except QhullError as error:
        raise ValueError(
            "Delaunay triangulation failed."
        ) from error

    triangles = orient_triangles_ccw(
        points,
        triangles,
    )

    areas = signed_double_areas(
        points,
        triangles,
    ) / 2.0

    if np.any(areas <= min_triangle_area):
        count = int(
            np.sum(areas <= min_triangle_area)
        )

        raise ValueError(
            f"Triangulation contains {count} triangles "
            f"at or below {min_triangle_area} px²."
        )

    return triangles