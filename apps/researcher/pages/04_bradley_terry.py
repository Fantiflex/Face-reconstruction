from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from face_reconstruction.models.bradley_terry import (
    add_softmax_weights,
    estimate_standard_errors,
    estimate_theta,
)

OUTPUT_DIRECTORY = Path(
    "data/results/bradley_terry"
)

REQUIRED_COLUMNS = {
    "participant_id",
    "pair_id",
    "model_left",
    "model_right",
    "selected_model",
}


def validate_response_table(
    responses: pd.DataFrame,
) -> None:
    missing_columns = (
        REQUIRED_COLUMNS
        - set(responses.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if responses.empty:
        raise ValueError(
            "The uploaded response table is empty."
        )


def fit_participant(
    responses: pd.DataFrame,
    *,
    participant_id: str,
    regularization: float,
    softmax_temperature: float,
) -> pd.DataFrame:
    participant_responses = responses.loc[
        responses["participant_id"].astype(str)
        == participant_id
    ].copy()

    if participant_responses.empty:
        raise ValueError(
            f"No responses found for participant "
            f"{participant_id}."
        )

    # Timeout / missing-choice trials should not enter
    # the Bradley-Terry likelihood.
    participant_responses = (
        participant_responses.loc[
            participant_responses[
                "selected_model"
            ].notna()
        ].copy()
    )

    if participant_responses.empty:
        raise ValueError(
            "This participant has no valid choices."
        )

    scores = estimate_theta(
        responses=participant_responses,
        regularization=regularization,
        participant_id=participant_id,
    )

    scores = estimate_standard_errors(
        responses=participant_responses,
        scores=scores,
        regularization=regularization,
    )

    scores = add_softmax_weights(
        scores=scores,
        temperature=softmax_temperature,
    )

    return scores.sort_values(
        "rank"
    ).reset_index(drop=True)


st.set_page_config(
    page_title="Bradley-Terry",
    page_icon="📊",
    layout="wide",
)

st.title(
    "Bradley–Terry analysis"
)

st.caption(
    "Upload participant pairwise-choice responses "
    "and estimate latent face-preference scores."
)


uploaded_file = st.file_uploader(
    "Upload participant responses JSON",
    type=["json"],
)

if uploaded_file is None:
    st.info(
        "Upload a participant response CSV to begin."
    )
    st.stop()


try:
    raw_data = json.load(
        uploaded_file
    )

    if not isinstance(raw_data, list):
        raise TypeError(
            "Expected a JSON list of participant responses."
        )

    responses = pd.DataFrame(
        raw_data
    )

    validate_response_table(
        responses
    )

except (
    ValueError,
    TypeError,
    json.JSONDecodeError,
) as error:
    st.error(
        str(error)
    )
    st.stop()


responses["participant_id"] = (
    responses["participant_id"]
    .astype(str)
)


participant_ids = sorted(
    responses["participant_id"]
    .dropna()
    .unique()
    .tolist()
)


st.success(
    f"Loaded {len(responses)} responses "
    f"from {len(participant_ids)} participant(s)."
)


with st.expander(
    "Show uploaded response table",
    expanded=False,
):
    st.dataframe(
        responses,
        use_container_width=True,
        hide_index=True,
    )


st.divider()

st.subheader(
    "Model configuration"
)


col1, col2, col3 = st.columns(3)


with col1:
    participant_id = st.selectbox(
        "Participant",
        options=participant_ids,
    )


with col2:
    regularization = st.number_input(
        "L2 regularization",
        min_value=0.0,
        value=1.0,
        step=0.1,
        help=(
            "Stabilizes Bradley-Terry estimation, "
            "especially for sparse comparison designs."
        ),
    )


with col3:
    softmax_temperature = st.number_input(
        "Softmax temperature",
        min_value=0.01,
        value=1.0,
        step=0.1,
        help=(
            "Controls how concentrated the final "
            "reconstruction weights are."
        ),
    )


participant_responses = responses.loc[
    responses["participant_id"]
    == participant_id
].copy()


valid_responses = participant_responses.loc[
    participant_responses[
        "selected_model"
    ].notna()
]


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Trials",
        len(participant_responses),
    )

with col2:
    st.metric(
        "Valid choices",
        len(valid_responses),
    )

with col3:
    st.metric(
        "Missing / timed out",
        (
            len(participant_responses)
            - len(valid_responses)
        ),
    )


st.divider()


if st.button(
    "Fit Bradley–Terry model",
    type="primary",
    use_container_width=True,
):
    try:
        scores = fit_participant(
            responses,
            participant_id=participant_id,
            regularization=float(
                regularization
            ),
            softmax_temperature=float(
                softmax_temperature
            ),
        )

    except (
        ValueError,
        RuntimeError,
    ) as error:
        st.error(
            str(error)
        )

    else:
        st.session_state[
            "bradley_terry_scores"
        ] = scores


if "bradley_terry_scores" in st.session_state:
    scores = st.session_state[
        "bradley_terry_scores"
    ]


    st.divider()

    st.subheader(
        "Bradley–Terry scores"
    )


    display_scores = scores.copy()

    if "theta" in display_scores.columns:
        display_scores["theta"] = (
            display_scores["theta"]
            .round(4)
        )

    if (
        "standard_error"
        in display_scores.columns
    ):
        display_scores[
            "standard_error"
        ] = (
            display_scores[
                "standard_error"
            ]
            .round(4)
        )

    if "weight" in display_scores.columns:
        display_scores["weight"] = (
            display_scores["weight"]
            .round(6)
        )


    st.dataframe(
        display_scores,
        use_container_width=True,
        hide_index=True,
    )


    st.subheader(
        "Latent preference scores"
    )

    theta_chart = (
        scores[
            [
                "face_id",
                "theta",
            ]
        ]
        .set_index(
            "face_id"
        )
        .sort_values(
            "theta",
            ascending=False,
        )
    )

    st.bar_chart(
        theta_chart
    )


    st.subheader(
        "Reconstruction weights"
    )

    weight_chart = (
        scores[
            [
                "face_id",
                "weight",
            ]
        ]
        .set_index(
            "face_id"
        )
        .sort_values(
            "weight",
            ascending=False,
        )
    )

    st.bar_chart(
        weight_chart
    )


    with st.expander(
        "Show top-ranked faces",
        expanded=False,
    ):
        top_n = st.slider(
            "Number of faces",
            min_value=1,
            max_value=len(scores),
            value=min(
                10,
                len(scores),
            ),
        )

        st.dataframe(
            display_scores.head(
                top_n
            ),
            use_container_width=True,
            hide_index=True,
        )


    st.divider()

    download_col, save_col = st.columns(
        2
    )


    with download_col:
        csv_data = scores.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "Download scores CSV",
            data=csv_data,
            file_name=(
                f"bradley_terry_"
                f"{participant_id}.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )


    with save_col:
        if st.button(
            "Save scores for reconstruction",
            use_container_width=True,
        ):
            OUTPUT_DIRECTORY.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                OUTPUT_DIRECTORY
                / (
                    f"scores_"
                    f"{participant_id}.csv"
                )
            )

            scores.to_csv(
                output_path,
                index=False,
            )

            st.success(
                f"Saved to {output_path}"
            )