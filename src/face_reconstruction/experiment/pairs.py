"""Pair-generation utilities for pairwise face-comparison experiments."""

from itertools import combinations

import numpy as np
import pandas as pd

from ..schemas import require_columns


def _validate_face_ids(face_ids: list[str]) -> list[str]:
    cleaned = [str(face_id).strip() for face_id in face_ids]
    if len(cleaned) < 2:
        raise ValueError("At least two face IDs are required.")
    if any(not face_id for face_id in cleaned):
        raise ValueError("Face IDs cannot be empty.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Face IDs must be unique.")
    return cleaned


def generate_all_pairs(face_ids: list[str]) -> pd.DataFrame:
    """Generate every unique unordered pair of face identifiers."""

    cleaned = _validate_face_ids(face_ids)
    rows = [
        {"model_a": left, "model_b": right}
        for left, right in combinations(cleaned, 2)
    ]
    return pd.DataFrame(rows)


def sample_pairs(
    face_ids: list[str],
    n_pairs: int,
    random_state: int,
) -> pd.DataFrame:
    """Sample a reproducible subset of unique unordered pairs."""

    all_pairs = generate_all_pairs(face_ids)
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive.")
    if n_pairs > len(all_pairs):
        raise ValueError(
            f"Requested {n_pairs} pairs but only {len(all_pairs)} exist."
        )
    return all_pairs.sample(n=n_pairs, random_state=random_state).reset_index(
        drop=True
    )


def generate_balanced_pairs(
    face_ids: list[str],
    n_pairs: int,
    random_state: int,
) -> pd.DataFrame:
    """Generate a unique pair set with approximately balanced exposure."""

    cleaned = _validate_face_ids(face_ids)
    all_pairs = list(combinations(cleaned, 2))
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive.")
    if n_pairs > len(all_pairs):
        raise ValueError(
            f"Requested {n_pairs} pairs but only {len(all_pairs)} exist."
        )

    rng = np.random.default_rng(random_state)
    rng.shuffle(all_pairs)
    exposure = {face_id: 0 for face_id in cleaned}
    selected: list[tuple[str, str]] = []
    remaining = all_pairs.copy()

    while len(selected) < n_pairs:
        minimum_cost = min(
            exposure[left] + exposure[right] for left, right in remaining
        )
        candidates = [
            pair
            for pair in remaining
            if exposure[pair[0]] + exposure[pair[1]] == minimum_cost
        ]
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        remaining.remove(chosen)
        selected.append(chosen)
        exposure[chosen[0]] += 1
        exposure[chosen[1]] += 1

    return pd.DataFrame(selected, columns=["model_a", "model_b"])


def randomise_pair_sides(
    pairs: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    """Randomly assign each pair member to the left or right position."""

    require_columns(pairs, ["model_a", "model_b"], "pair table")
    rng = np.random.default_rng(random_state)
    swap = rng.random(len(pairs)) < 0.5

    result = pairs.copy().reset_index(drop=True)
    result["model_left"] = np.where(
        swap, result["model_b"], result["model_a"]
    )
    result["model_right"] = np.where(
        swap, result["model_a"], result["model_b"]
    )
    return result.drop(columns=["model_a", "model_b"])


def add_pair_ids(pairs: pd.DataFrame) -> pd.DataFrame:
    """Assign stable sequential identifiers and presentation order."""

    require_columns(pairs, ["model_left", "model_right"], "pair table")
    result = pairs.copy().reset_index(drop=True)
    result["pair_order"] = np.arange(1, len(result) + 1)
    result["pair_id"] = result["pair_order"].map(lambda x: f"PAIR_{x:05d}")
    return result[["pair_id", "model_left", "model_right", "pair_order"]]


def compute_appearance_counts(pairs: pd.DataFrame) -> pd.Series:
    """Count how often each face appears across all comparisons."""

    require_columns(pairs, ["model_left", "model_right"], "pair table")
    return pd.concat(
        [pairs["model_left"], pairs["model_right"]],
        ignore_index=True,
    ).value_counts().sort_index()


def generate_stratified_pairs(
    stimuli: pd.DataFrame,
    n_pairs: int,
    group_column: str,
    target_pair_type_proportions: dict[tuple[str, str], float],
    random_state: int,
) -> pd.DataFrame:
    """Generate unique pairs with controlled group-pair proportions.

    Pair types are specified as tuples such as
    ("Black", "Black"), ("Black", "White"), and ("White", "White").

    Within each requested pair type, faces are selected greedily to keep
    individual stimulus exposure as balanced as possible.
    """

    require_columns(
        stimuli,
        ["face_id", group_column],
        "stimulus table",
    )

    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive.")

    if not target_pair_type_proportions:
        raise ValueError(
            "target_pair_type_proportions cannot be empty."
        )

    table = stimuli[["face_id", group_column]].copy()

    table["face_id"] = (
        table["face_id"]
        .astype(str)
        .str.strip()
    )

    table[group_column] = (
        table[group_column]
        .astype(str)
        .str.strip()
    )

    if table["face_id"].eq("").any():
        raise ValueError("Face IDs cannot be empty.")

    if table[group_column].eq("").any():
        raise ValueError(
            f"Column {group_column!r} contains empty group values."
        )

    if table["face_id"].duplicated().any():
        raise ValueError(
            "Stimulus face IDs must be unique."
        )

    groups = set(
        table[group_column].unique()
    )

    canonical_proportions: dict[
        tuple[str, str],
        float,
    ] = {}

    for pair_type, proportion in (
        target_pair_type_proportions.items()
    ):
        if len(pair_type) != 2:
            raise ValueError(
                "Each pair type must contain exactly two groups."
            )

        group_a = str(pair_type[0]).strip()
        group_b = str(pair_type[1]).strip()

        if group_a not in groups or group_b not in groups:
            raise ValueError(
                "Unknown group in pair type: "
                f"{pair_type}"
            )

        if not np.isfinite(proportion) or proportion < 0:
            raise ValueError(
                "Pair-type proportions must be finite "
                "and non-negative."
            )

        canonical_pair = tuple(
            sorted((group_a, group_b))
        )

        canonical_proportions[canonical_pair] = (
            canonical_proportions.get(
                canonical_pair,
                0.0,
            )
            + float(proportion)
        )

    total_proportion = sum(
        canonical_proportions.values()
    )

    if not np.isclose(
        total_proportion,
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Pair-type proportions must sum to 1. "
            f"Observed: {total_proportion:.6f}"
        )

    # Convert proportions into exact integer pair counts using
    # largest-remainder rounding.
    raw_counts = {
        pair_type: proportion * n_pairs
        for pair_type, proportion
        in canonical_proportions.items()
    }

    target_counts = {
        pair_type: int(np.floor(count))
        for pair_type, count
        in raw_counts.items()
    }

    remaining = (
        n_pairs
        - sum(target_counts.values())
    )

    remainder_order = sorted(
        raw_counts,
        key=lambda pair_type: (
            raw_counts[pair_type]
            - target_counts[pair_type]
        ),
        reverse=True,
    )

    for pair_type in remainder_order[:remaining]:
        target_counts[pair_type] += 1

    faces_by_group = {
        group: table.loc[
            table[group_column] == group,
            "face_id",
        ].tolist()
        for group in groups
    }

    rng = np.random.default_rng(
        random_state
    )

    exposure = {
        face_id: 0
        for face_id in table["face_id"]
    }

    selected_pairs: list[
        tuple[str, str]
    ] = []

    for pair_type, target_count in (
        target_counts.items()
    ):
        if target_count == 0:
            continue

        group_a, group_b = pair_type

        if group_a == group_b:
            candidates = list(
                combinations(
                    faces_by_group[group_a],
                    2,
                )
            )
        else:
            candidates = [
                (face_a, face_b)
                for face_a in faces_by_group[group_a]
                for face_b in faces_by_group[group_b]
            ]

        if target_count > len(candidates):
            raise ValueError(
                f"Requested {target_count} pairs of type "
                f"{group_a} × {group_b}, but only "
                f"{len(candidates)} unique pairs are available."
            )

        rng.shuffle(candidates)

        for _ in range(target_count):
            minimum_cost = min(
                exposure[left]
                + exposure[right]
                for left, right in candidates
            )

            best_candidates = [
                pair
                for pair in candidates
                if (
                    exposure[pair[0]]
                    + exposure[pair[1]]
                    == minimum_cost
                )
            ]

            chosen = best_candidates[
                int(
                    rng.integers(
                        0,
                        len(best_candidates),
                    )
                )
            ]

            candidates.remove(chosen)

            selected_pairs.append(
                chosen
            )

            exposure[chosen[0]] += 1
            exposure[chosen[1]] += 1

    rng.shuffle(selected_pairs)

    return pd.DataFrame(
        selected_pairs,
        columns=[
            "model_a",
            "model_b",
        ],
    )
