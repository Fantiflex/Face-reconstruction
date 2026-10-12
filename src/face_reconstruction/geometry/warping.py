from typing import Any

import cv2
import numpy as np


def signed_triangle_area(triangle: np.ndarray) -> float:
    triangle = np.asarray(triangle, dtype=np.float64)
    if triangle.shape != (3, 2):
        raise ValueError(f"Expected a (3, 2) triangle, got {triangle.shape}.")
    ab = triangle[1] - triangle[0]
    ac = triangle[2] - triangle[0]
    return float(0.5 * (ab[0] * ac[1] - ab[1] * ac[0]))


def triangle_area_tolerance(
    triangle: np.ndarray,
    *,
    minimum_absolute_area: float,
    relative_tolerance: float,
) -> float:
    triangle = np.asarray(triangle, dtype=np.float64)
    edges = np.asarray(
        [
            triangle[1] - triangle[0],
            triangle[2] - triangle[0],
            triangle[2] - triangle[1],
        ],
        dtype=np.float64,
    )
    maximum_squared_length = float(
        np.max(np.sum(edges * edges, axis=1))
    )
    return max(
        float(minimum_absolute_area),
        float(relative_tolerance) * maximum_squared_length,
    )


def affine_for_triangle(
    source: np.ndarray,
    target: np.ndarray,
    *,
    minimum_absolute_area: float,
    relative_area_tolerance: float,
    maximum_condition_number: float,
) -> tuple[np.ndarray | None, str | None]:
    source_area = signed_triangle_area(source)
    target_area = signed_triangle_area(target)
    source_tolerance = triangle_area_tolerance(
        source,
        minimum_absolute_area=minimum_absolute_area,
        relative_tolerance=relative_area_tolerance,
    )
    target_tolerance = triangle_area_tolerance(
        target,
        minimum_absolute_area=minimum_absolute_area,
        relative_tolerance=relative_area_tolerance,
    )

    if abs(source_area) <= source_tolerance:
        return None, "source_degenerate"
    if abs(target_area) <= target_tolerance:
        return None, "target_degenerate"
    if source_area * target_area < 0.0:
        return None, "orientation_flip"

    try:
        matrix = cv2.getAffineTransform(
            np.asarray(source, dtype=np.float32),
            np.asarray(target, dtype=np.float32),
        )
    except cv2.error:
        return None, "opencv_affine_error"

    if not np.isfinite(matrix).all():
        return None, "non_finite_affine"
    condition_number = float(np.linalg.cond(matrix[:, :2]))
    if (
        not np.isfinite(condition_number)
        or condition_number > maximum_condition_number
    ):
        return None, "ill_conditioned_affine"
    return matrix, None


def warp_piecewise_affine(
    image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    triangles: np.ndarray,
    *,
    skipped_triangle_indices: set[int],
    interpolation: int,
    border_mode: int,
    minimum_absolute_area: float,
    relative_area_tolerance: float,
    maximum_condition_number: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    height, width = image.shape[:2]
    accumulation = np.zeros((height, width, 3), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    if source_points.shape != target_points.shape:
        raise ValueError(
            "Source and target point arrays must have the same shape; got "
            f"{source_points.shape} and {target_points.shape}."
        )

    skipped: list[dict[str, Any]] = []
    for triangle_index, triangle in enumerate(triangles):
        if triangle_index in skipped_triangle_indices:
            skipped.append(
                {
                    "triangle_index": int(triangle_index),
                    "reason": "step09_unstable",
                }
            )
            continue

        source_triangle = source_points[triangle]
        target_triangle = target_points[triangle]
        matrix, failure_reason = affine_for_triangle(
            source_triangle,
            target_triangle,
            minimum_absolute_area=minimum_absolute_area,
            relative_area_tolerance=relative_area_tolerance,
            maximum_condition_number=maximum_condition_number,
        )
        if matrix is None:
            skipped.append(
                {
                    "triangle_index": int(triangle_index),
                    "reason": str(failure_reason),
                }
            )
            continue

        x, y, box_width, box_height = cv2.boundingRect(
            np.asarray(target_triangle, dtype=np.float32)
        )
        x0, y0 = max(0, x), max(0, y)
        x1 = min(width, x + box_width)
        y1 = min(height, y + box_height)
        if x0 >= x1 or y0 >= y1:
            skipped.append(
                {
                    "triangle_index": int(triangle_index),
                    "reason": "target_outside_canvas",
                }
            )
            continue

        local_matrix = matrix.copy()
        local_matrix[0, 2] -= x0
        local_matrix[1, 2] -= y0
        warped_patch = cv2.warpAffine(
            image,
            local_matrix,
            (x1 - x0, y1 - y0),
            flags=interpolation,
            borderMode=border_mode,
            borderValue=(0, 0, 0),
        ).astype(np.float32)

        local_triangle = np.rint(
            target_triangle - np.array([x0, y0], dtype=np.float64)
        ).astype(np.int32)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        cv2.fillConvexPoly(mask, local_triangle, 1.0, lineType=cv2.LINE_8)
        if not np.any(mask > 0.0):
            skipped.append(
                {
                    "triangle_index": int(triangle_index),
                    "reason": "empty_raster_mask",
                }
            )
            continue
        accumulation[y0:y1, x0:x1] += warped_patch * mask[..., None]
        coverage[y0:y1, x0:x1] += mask

    covered = coverage > 0.0
    output = np.zeros_like(accumulation, dtype=np.float32)
    output[covered] = accumulation[covered] / coverage[covered, None]
    return output, covered, skipped

