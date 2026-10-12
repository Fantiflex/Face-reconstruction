import numpy as np
import pytest

from face_reconstruction.geometry.triangulation import (
    build_delaunay_triangulation,
    signed_double_areas,
)


def test_build_delaunay_triangulation():
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [5.0, 5.0],
        ]
    )

    triangles = build_delaunay_triangulation(
        points,
        min_triangle_area=0.01,
    )

    assert triangles.ndim == 2
    assert triangles.shape[1] == 3

    areas = (
        signed_double_areas(
            points,
            triangles,
        )
        / 2.0
    )

    assert np.all(areas > 0)


def test_build_delaunay_rejects_duplicate_points():
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [0.0, 10.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        build_delaunay_triangulation(points)


def test_build_delaunay_rejects_too_few_points():
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="At least three",
    ):
        build_delaunay_triangulation(points)