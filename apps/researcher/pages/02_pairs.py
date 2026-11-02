from __future__ import annotations

import json
from itertools import combinations_with_replacement
from pathlib import Path

import pandas as pd
import streamlit as st

from face_reconstruction.experiment.pairs import (
    add_pair_ids,
    generate_balanced_pairs,
    generate_stratified_pairs,
    randomise_pair_sides,
    sample_pairs,
)

STIMULI_PATH = Path("data/processed/stimuli.csv")
PAIRS_OUTPUT_PATH = Path("data/processed/pairs.csv")
CONFIG_OUTPUT_PATH = Path(
    "data/processed/experiment_config.json"
)


st.set_page_config(
    page_title="Pairs",
    page_icon="🔀",
    layout="wide",
)


@st.cache_data
def load_stimuli() -> pd.DataFrame:
    if not STIMULI_PATH.exists():
        raise FileNotFoundError(
            f"Stimulus table not found: {STIMULI_PATH}"
        )

    stimuli = pd.read_csv(STIMULI_PATH).copy()

    if "face_id" not in stimuli.columns:
        raise ValueError(
            "Stimulus table must contain a face_id column."
        )

    stimuli["face_id"] = (
        stimuli["face_id"]
        .astype(str)
        .str.strip()
    )

    return stimuli

def build_group_counts(
    stimuli: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    if column not in stimuli.columns:
        return pd.DataFrame()

    return (
        stimuli[column]
        .fillna("missing")
        .astype(str)
        .value_counts()
        .rename_axis(column)
        .reset_index(name="n_faces")
    )

def build_exposure_table(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    appearances = pd.concat(
        [
            pairs["model_left"],
            pairs["model_right"],
        ],
        ignore_index=True,
    )

    return (
        appearances.value_counts()
        .rename_axis("face_id")
        .reset_index(name="n_appearances")
        .sort_values(
            ["n_appearances", "face_id"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def build_group_exposure_table(
    pairs: pd.DataFrame,
    stimuli: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    group_map = stimuli.set_index(
        "face_id"
    )[group_column].astype(str).to_dict()

    appearances = pd.concat(
        [
            pairs["model_left"],
            pairs["model_right"],
        ],
        ignore_index=True,
    )

    groups = appearances.map(group_map)

    counts = (
        groups.value_counts()
        .rename_axis("group")
        .reset_index(name="n_appearances")
    )

    total = counts["n_appearances"].sum()

    counts["proportion"] = (
        counts["n_appearances"] / total
    )

    return counts.sort_values(
        "group"
    ).reset_index(drop=True)


def build_pair_type_table(
    pairs: pd.DataFrame,
    stimuli: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    group_map = stimuli.set_index(
        "face_id"
    )[group_column].astype(str).to_dict()

    pair_types: list[str] = []

    for row in pairs.itertuples(
        index=False
    ):
        left_group = group_map[
            row.model_left
        ]
        right_group = group_map[
            row.model_right
        ]

        group_a, group_b = sorted(
            (
                left_group,
                right_group,
            )
        )

        pair_types.append(
            f"{group_a} × {group_b}"
        )

    counts = (
        pd.Series(
            pair_types,
            name="pair_type",
        )
        .value_counts()
        .rename_axis("pair_type")
        .reset_index(name="n_pairs")
    )

    counts["proportion"] = (
        counts["n_pairs"]
        / len(pairs)
    )

    return counts.sort_values(
        "pair_type"
    ).reset_index(drop=True)


st.title("Pair design")

st.caption(
    "Generate pairwise comparisons and control how "
    "demographic or perceived groups are represented."
)

try:
    stimuli = load_stimuli()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()


face_ids = (
    stimuli["face_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

n_faces = len(face_ids)

maximum_pairs = (
    n_faces * (n_faces - 1) // 2
)

st.info(
    "This page uses the active stimulus set saved from "
    "the Stimuli page."
)

st.caption(
    f"Source: {STIMULI_PATH}"
)


st.subheader("Active stimulus pool")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Faces",
        n_faces,
    )

with col2:
    st.metric(
        "Maximum unique pairs",
        maximum_pairs,
    )

with st.expander(
    "Show active pool composition",
    expanded=False,
):
    available_summary_columns = [
        column
        for column in [
            "ethnicity_perceived",
            "ethnicity_self",
            "gender_self",
        ]
        if column in stimuli.columns
    ]

    if not available_summary_columns:
        st.info(
            "No demographic summary columns found."
        )
    else:
        summary_columns = st.columns(
            len(available_summary_columns)
        )

        for container, column in zip(
            summary_columns,
            available_summary_columns,
        ):
            with container:
                st.markdown(
                    f"**{column}**"
                )

                summary = build_group_counts(
                    stimuli,
                    column,
                )

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True,
                )


st.divider()

st.subheader("Experimental design")

method = st.selectbox(
    "Pair generation strategy",
    options=[
        "stratified",
        "balanced",
        "random",
    ],
    index=0,
    help=(
        "Stratified controls group composition. "
        "Balanced controls individual face exposure. "
        "Random samples unique pairs without balancing."
    ),
)

col1, col2 = st.columns(2)

with col1:
    n_pairs = st.number_input(
        "Number of pairs",
        min_value=1,
        max_value=maximum_pairs,
        value=min(
            300,
            maximum_pairs,
        ),
        step=1,
    )

with col2:
    random_state = st.number_input(
        "Random seed",
        min_value=0,
        value=42,
        step=1,
    )


target_pair_type_proportions = None
group_column = None


if method == "stratified":
    st.divider()

    st.subheader("Group control")

    available_group_columns = [
        column
        for column in [
            "ethnicity_perceived",
            "ethnicity_self",
            "gender_self",
        ]
        if column in stimuli.columns
    ]

    group_column = st.selectbox(
        "Variable used to define groups",
        options=available_group_columns,
        index=0,
    )

    groups = sorted(
        stimuli[group_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    st.markdown(
        "**Available faces by group**"
    )

    group_counts = (
        stimuli[group_column]
        .astype(str)
        .value_counts()
        .rename_axis("group")
        .reset_index(name="n_faces")
    )

    st.dataframe(
        group_counts,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "**Target pair-type proportions**"
    )

    st.caption(
        "Choose the proportion of each pair type. "
        "The final category is calculated automatically "
        "so that the total is exactly 100%."
    )

    pair_types = list(
        combinations_with_replacement(
            groups,
            2,
        )
    )

    target_pair_type_proportions = {}

    remaining_percentage = 100

    for index, pair_type in enumerate(
        pair_types
    ):
        group_a, group_b = pair_type

        label = (
            f"{group_a} × {group_b}"
        )

        is_last = (
            index
            == len(pair_types) - 1
        )

        if is_last:
            percentage = max(
                0,
                remaining_percentage,
            )

            st.number_input(
                label,
                min_value=0,
                max_value=100,
                value=int(
                    percentage
                ),
                step=1,
                disabled=True,
                key=(
                    f"pair_type_"
                    f"{group_a}_"
                    f"{group_b}"
                ),
            )

        else:
            default_value = int(
                remaining_percentage
                / (
                    len(pair_types)
                    - index
                )
            )

            percentage = st.number_input(
                label,
                min_value=0,
                max_value=max(
                    0,
                    remaining_percentage,
                ),
                value=default_value,
                step=1,
                key=(
                    f"pair_type_"
                    f"{group_a}_"
                    f"{group_b}"
                ),
            )

            remaining_percentage -= int(
                percentage
            )

        target_pair_type_proportions[
            pair_type
        ] = (
            float(percentage)
            / 100.0
        )
        total_percentage = round(
            sum(
                target_pair_type_proportions.values()
            )
            * 100
        )

        st.success(
            f"Total pair allocation: {total_percentage}%"
        )

st.divider()


if st.button(
    "Generate pair design",
    type="primary",
):
    try:
        if method == "stratified":
            if (
                target_pair_type_proportions
                is None
                or group_column is None
            ):
                raise ValueError(
                    "Stratified configuration is incomplete."
                )

            

            pairs = generate_stratified_pairs(
                stimuli=stimuli,
                n_pairs=int(n_pairs),
                group_column=group_column,
                target_pair_type_proportions=(
                    target_pair_type_proportions
                ),
                random_state=int(
                    random_state
                ),
            )

        elif method == "balanced":
            pairs = generate_balanced_pairs(
                face_ids=face_ids,
                n_pairs=int(n_pairs),
                random_state=int(
                    random_state
                ),
            )

        else:
            pairs = sample_pairs(
                face_ids=face_ids,
                n_pairs=int(n_pairs),
                random_state=int(
                    random_state
                ),
            )

        pairs = randomise_pair_sides(
            pairs,
            random_state=int(
                random_state
            ),
        )

        pairs = add_pair_ids(
            pairs
        )

        st.session_state[
            "generated_pairs"
        ] = pairs

        st.session_state[
            "pair_group_column"
        ] = group_column

    except ValueError as error:
        st.error(str(error))


if "generated_pairs" in st.session_state:
    pairs = st.session_state[
        "generated_pairs"
    ]

    exposure = build_exposure_table(
        pairs
    )

    st.divider()

    st.subheader("Design diagnostics")

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:
        st.metric(
            "Pairs",
            len(pairs),
        )

    with col2:
        st.metric(
            "Min face exposure",
            int(
                exposure[
                    "n_appearances"
                ].min()
            ),
        )

    with col3:
        st.metric(
            "Max face exposure",
            int(
                exposure[
                    "n_appearances"
                ].max()
            ),
        )

    with col4:
        st.metric(
            "Mean face exposure",
            f"{exposure['n_appearances'].mean():.2f}",
        )


    diagnostics_group_column = (
        st.session_state.get(
            "pair_group_column"
        )
    )

    if diagnostics_group_column:
        left, right = st.columns(2)

        group_exposure = (
            build_group_exposure_table(
                pairs,
                stimuli,
                diagnostics_group_column,
            )
        )

        pair_type_table = (
            build_pair_type_table(
                pairs,
                stimuli,
                diagnostics_group_column,
            )
        )

        with left:
            st.markdown(
                "**Observed group exposure**"
            )

            display_group_exposure = (
                group_exposure.copy()
            )

            display_group_exposure[
                "proportion"
            ] = (
                display_group_exposure[
                    "proportion"
                ]
                * 100
            ).round(2)

            st.dataframe(
                display_group_exposure,
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.markdown(
                "**Observed pair types**"
            )

            display_pair_types = (
                pair_type_table.copy()
            )

            display_pair_types[
                "proportion"
            ] = (
                display_pair_types[
                    "proportion"
                ]
                * 100
            ).round(2)

            st.dataframe(
                display_pair_types,
                use_container_width=True,
                hide_index=True,
            )


    st.markdown(
        "**Individual face exposure**"
    )

    st.dataframe(
        exposure,
        use_container_width=True,
        hide_index=True,
    )


    with st.expander(
        "Generated pair table"
    ):
        st.dataframe(
            pairs,
            use_container_width=True,
            hide_index=True,
        )


    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        csv_data = pairs.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download pairs.csv",
            data=csv_data,
            file_name="pairs.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        config = {
            "stimuli_path": str(STIMULI_PATH),
            "pairs_path": str(PAIRS_OUTPUT_PATH),
            "n_faces": n_faces,
            "n_pairs": int(n_pairs),
            "generation_method": method,
            "random_state": int(random_state),
            "group_column": diagnostics_group_column,
        }

        if (
            method == "stratified"
            and target_pair_type_proportions is not None
        ):
            config["target_pair_type_proportions"] = {
                f"{group_a}__{group_b}": proportion
                for (group_a, group_b), proportion
                in target_pair_type_proportions.items()
            }
        if st.button(
            "Save as active pair design",
            use_container_width=True,
        ):
            PAIRS_OUTPUT_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            pairs.to_csv(
                PAIRS_OUTPUT_PATH,
                index=False,
            )

            with CONFIG_OUTPUT_PATH.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    config,
                    file,
                    indent=2,
                )

            st.success(
                "Active pair design saved."
            )

            st.caption(
                f"Pairs: {PAIRS_OUTPUT_PATH}"
            )

            st.caption(
                f"Configuration: {CONFIG_OUTPUT_PATH}"
            )