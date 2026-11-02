from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from face_reconstruction.models.bradley_terry import (
    add_softmax_weights,
    estimate_standard_errors,
    estimate_theta,
)


PAIRS_PATH = Path("data/processed/pairs.csv")
STIMULI_PATH = Path("data/processed/stimuli.csv")


st.set_page_config(
    page_title="Face Reconstruction",
    layout="wide",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(PAIRS_PATH)
    stimuli = pd.read_csv(STIMULI_PATH)

    stimuli["face_id"] = stimuli["face_id"].astype(str)

    return pairs, stimuli


def get_image_path(
    face_id: str,
    stimuli: pd.DataFrame,
) -> Path:
    row = stimuli.loc[
        stimuli["face_id"] == face_id
    ]

    if row.empty:
        raise ValueError(
            f"Unknown face_id: {face_id}"
        )

    return Path(
        row.iloc[0]["image_path"]
    )


def fit_bradley_terry(
    responses: pd.DataFrame,
    participant_id: str,
) -> pd.DataFrame:
    scores = estimate_theta(
        responses=responses,
        regularization=1.0,
        participant_id=participant_id,
    )

    scores = estimate_standard_errors(
        responses=responses,
        scores=scores,
        regularization=1.0,
    )

    scores = add_softmax_weights(
        scores=scores,
        temperature=1.0,
    )

    return scores


pairs, stimuli = load_data()

if "trial_index" not in st.session_state:
    st.session_state.trial_index = 0

if "responses" not in st.session_state:
    st.session_state.responses = []

if "trial_start" not in st.session_state:
    st.session_state.trial_start = time.perf_counter()


st.title("Face Reconstruction Test")

participant_id = st.sidebar.text_input(
    "Participant ID",
    value="TEST_001",
)

n_trials = st.sidebar.slider(
    "Number of pairwise choices",
    min_value=5,
    max_value=min(100, len(pairs)),
    value=min(20, len(pairs)),
    step=5,
)

if st.sidebar.button("Reset experiment"):
    st.session_state.trial_index = 0
    st.session_state.responses = []
    st.session_state.trial_start = time.perf_counter()
    st.rerun()


trial_index = st.session_state.trial_index

if trial_index < n_trials:
    pair = pairs.iloc[trial_index]

    left_id = str(pair["model_left"])
    right_id = str(pair["model_right"])

    left_path = get_image_path(
        left_id,
        stimuli,
    )

    right_path = get_image_path(
        right_id,
        stimuli,
    )

    st.progress(
        trial_index / n_trials
    )

    st.write(
        f"Choice {trial_index + 1} / {n_trials}"
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.image(
            str(left_path),
            use_container_width=True,
        )

        choose_left = st.button(
            "Choose left",
            use_container_width=True,
            key=f"left_{trial_index}",
        )

    with col_right:
        st.image(
            str(right_path),
            use_container_width=True,
        )

        choose_right = st.button(
            "Choose right",
            use_container_width=True,
            key=f"right_{trial_index}",
        )

    selected_model = None

    if choose_left:
        selected_model = left_id

    if choose_right:
        selected_model = right_id

    if selected_model is not None:
        response_time_ms = (
            time.perf_counter()
            - st.session_state.trial_start
        ) * 1000

        st.session_state.responses.append(
            {
                "participant_id": participant_id,
                "trial_index": trial_index,
                "pair_id": pair["pair_id"],
                "model_left": left_id,
                "model_right": right_id,
                "selected_model": selected_model,
                "response_time_ms": response_time_ms,
            }
        )

        st.session_state.trial_index += 1
        st.session_state.trial_start = time.perf_counter()

        st.rerun()

else:
    st.success(
        "Pairwise choices completed."
    )

    responses = pd.DataFrame(
        st.session_state.responses
    )

    st.subheader("Responses")
    st.dataframe(
        responses,
        use_container_width=True,
    )

    if st.button(
        "Estimate Bradley-Terry weights",
        type="primary",
    ):
        scores = fit_bradley_terry(
            responses,
            participant_id,
        )

        st.session_state.scores = scores


if "scores" in st.session_state:
    scores = st.session_state.scores

    st.subheader(
        "Bradley-Terry scores"
    )

    st.dataframe(
        scores.sort_values(
            "weight",
            ascending=False,
        ),
        use_container_width=True,
    )

    st.subheader(
        "Top weighted faces"
    )

    top_faces = scores.nlargest(
        5,
        "weight",
    )

    cols = st.columns(
        len(top_faces)
    )

    for col, row in zip(
        cols,
        top_faces.itertuples(),
    ):
        path = get_image_path(
            row.face_id,
            stimuli,
        )

        with col:
            st.image(
                str(path),
                use_container_width=True,
            )
            st.caption(
                f"{row.face_id}\n"
                f"w={row.weight:.3f}"
            )