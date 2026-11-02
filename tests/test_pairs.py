import pandas as pd

from face_reconstruction.experiment.pairs import (
    generate_all_pairs,
    generate_stratified_pairs,
)


def test_generate_all_pairs():
    pairs = generate_all_pairs(
        ["A", "B", "C"]
    )

    assert len(pairs) == 3



def test_generate_stratified_pairs_controls_pair_types():
    stimuli = pd.DataFrame(
        {
            "face_id": [
                "B1",
                "B2",
                "B3",
                "B4",
                "W1",
                "W2",
                "W3",
                "W4",
            ],
            "ethnicity_perceived": [
                "Black",
                "Black",
                "Black",
                "Black",
                "White",
                "White",
                "White",
                "White",
            ],
        }
    )

    pairs = generate_stratified_pairs(
        stimuli=stimuli,
        n_pairs=12,
        group_column="ethnicity_perceived",
        target_pair_type_proportions={
            ("Black", "Black"): 0.25,
            ("Black", "White"): 0.50,
            ("White", "White"): 0.25,
        },
        random_state=42,
    )

    group_map = stimuli.set_index(
        "face_id"
    )["ethnicity_perceived"].to_dict()

    pair_types = []

    for row in pairs.itertuples(
        index=False
    ):
        pair_types.append(
            tuple(
                sorted(
                    (
                        group_map[row.model_a],
                        group_map[row.model_b],
                    )
                )
            )
        )

    counts = pd.Series(
        pair_types
    ).value_counts()

    assert len(pairs) == 12
    assert counts[("Black", "Black")] == 3
    assert counts[("Black", "White")] == 6
    assert counts[("White", "White")] == 3


def test_generate_stratified_pairs_has_no_duplicates():
    stimuli = pd.DataFrame(
        {
            "face_id": [
                "B1",
                "B2",
                "B3",
                "W1",
                "W2",
                "W3",
            ],
            "ethnicity_perceived": [
                "Black",
                "Black",
                "Black",
                "White",
                "White",
                "White",
            ],
        }
    )

    pairs = generate_stratified_pairs(
        stimuli=stimuli,
        n_pairs=8,
        group_column="ethnicity_perceived",
        target_pair_type_proportions={
            ("Black", "Black"): 0.25,
            ("Black", "White"): 0.50,
            ("White", "White"): 0.25,
        },
        random_state=42,
    )

    canonical_pairs = {
        tuple(
            sorted(
                (
                    row.model_a,
                    row.model_b,
                )
            )
        )
        for row in pairs.itertuples(
            index=False
        )
    }

    assert len(canonical_pairs) == len(pairs)