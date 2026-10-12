import numpy as np
import pytest

from face_reconstruction.reconstruction.morph import (
    accumulate_weighted_warp,
    finalize_weighted_reconstruction,
)


def test_accumulate_weighted_warp():
    numerator = np.zeros((2, 2, 3))
    denominator = np.zeros((2, 2))

    warped = np.full(
        (2, 2, 3),
        10.0,
    )

    covered = np.array(
        [
            [True, False],
            [True, True],
        ]
    )

    accumulate_weighted_warp(
        numerator,
        denominator,
        warped,
        covered,
        0.5,
    )

    assert np.allclose(
        denominator,
        [
            [0.5, 0.0],
            [0.5, 0.5],
        ],
    )

    assert np.allclose(
        numerator[0, 0],
        [5.0, 5.0, 5.0],
    )


def test_finalize_weighted_reconstruction():
    numerator = np.array(
        [
            [
                [10.0, 10.0, 10.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    )

    denominator = np.array(
        [
            [0.5, 0.0],
        ]
    )

    fallback = np.array(
        [
            [
                [100.0, 100.0, 100.0],
                [50.0, 50.0, 50.0],
            ]
        ]
    )

    reconstruction, covered, uncovered_fraction = (
        finalize_weighted_reconstruction(
            numerator,
            denominator,
            fallback,
            max_uncovered_fraction=0.5,
        )
    )

    assert np.allclose(
        reconstruction[0, 0],
        [20.0, 20.0, 20.0],
    )

    assert np.allclose(
        reconstruction[0, 1],
        [50.0, 50.0, 50.0],
    )

    assert covered.tolist() == [
        [True, False]
    ]

    assert uncovered_fraction == 0.5


def test_finalize_rejects_too_many_uncovered_pixels():
    numerator = np.zeros((2, 2, 3))
    denominator = np.zeros((2, 2))
    denominator[0, 0] = 1.0

    fallback = np.zeros((2, 2, 3))

    with pytest.raises(
        RuntimeError,
        match="Too many reconstruction pixels",
    ):
        finalize_weighted_reconstruction(
            numerator,
            denominator,
            fallback,
            max_uncovered_fraction=0.5,
        )