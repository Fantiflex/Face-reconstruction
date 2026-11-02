from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

STIMULI_PATH = Path(
    "data/processed/stimuli.csv"
)

PAIRS_PATH = Path(
    "data/processed/pairs.csv"
)

PARTICIPANT_DIR = Path(
    "apps/participant"
)

PARTICIPANT_STIMULI_DIR = (
    PARTICIPANT_DIR / "stimuli"
)

EXPERIMENT_CONFIG_PATH = (
    PARTICIPANT_DIR / "experiment.json"
)



def load_stimuli() -> pd.DataFrame:
    if not STIMULI_PATH.exists():
        raise FileNotFoundError(
            f"Active stimulus set not found: {STIMULI_PATH}"
        )

    return pd.read_csv(
        STIMULI_PATH
    )



def load_pairs() -> pd.DataFrame:
    if not PAIRS_PATH.exists():
        raise FileNotFoundError(
            f"Active pair design not found: {PAIRS_PATH}"
        )

    return pd.read_csv(
        PAIRS_PATH
    )


def build_participant_package(
    stimuli: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    randomize_trials: bool,
    response_timeout_ms: int | None,
) -> dict:
    PARTICIPANT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PARTICIPANT_STIMULI_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove images from a previous experiment package.
    for existing_file in (
        PARTICIPANT_STIMULI_DIR.iterdir()
    ):
        if existing_file.is_file():
            existing_file.unlink()

    image_paths: dict[str, str] = {}

    for row in stimuli.itertuples(
        index=False
    ):
        face_id = str(
            row.face_id
        )

        source_path = Path(
            row.image_path
        )

        if not source_path.is_absolute():
            source_path = (
                Path.cwd()
                / source_path
            )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Image not found for {face_id}: "
                f"{source_path}"
            )

        extension = (
            source_path.suffix.lower()
        )

        destination_name = (
            f"{face_id}{extension}"
        )

        destination_path = (
            PARTICIPANT_STIMULI_DIR
            / destination_name
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        image_paths[face_id] = (
            f"stimuli/{destination_name}"
        )

    trials = []

    for row in pairs.itertuples(
        index=False
    ):
        left_id = str(
            row.model_left
        )

        right_id = str(
            row.model_right
        )

        if left_id not in image_paths:
            raise ValueError(
                f"Pair references missing stimulus: {left_id}"
            )

        if right_id not in image_paths:
            raise ValueError(
                f"Pair references missing stimulus: {right_id}"
            )

        trials.append(
            {
                "pair_id": str(
                    row.pair_id
                ),
                "pair_order": int(
                    row.pair_order
                ),
                "model_left": left_id,
                "model_right": right_id,
                "image_left": (
                    image_paths[left_id]
                ),
                "image_right": (
                    image_paths[right_id]
                ),
            }
        )

    config = {
        "version": 1,
        "n_stimuli": len(stimuli),
        "n_trials": len(trials),
        "randomize_trials": (
            randomize_trials
        ),
        "response_timeout_ms": (
            response_timeout_ms
        ),
        "trials": trials,
    }

    with EXPERIMENT_CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    return config


st.set_page_config(
    page_title="Experiment",
    page_icon="🧪",
    layout="wide",
)

st.title(
    "Participant experiment"
)

st.caption(
    "Build the participant-facing experiment from "
    "the active stimulus pool and pair design."
)


try:
    stimuli = load_stimuli()
    pairs = load_pairs()

except FileNotFoundError as error:
    st.error(
        str(error)
    )

    st.stop()


col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Active stimuli",
        len(stimuli),
    )

with col2:
    st.metric(
        "Trials",
        len(pairs),
    )


st.divider()

st.subheader(
    "Participant task settings"
)


randomize_trials = st.checkbox(
    "Randomize trial order for each participant",
    value=True,
)


use_timeout = st.checkbox(
    "Use response timeout",
    value=False,
)


response_timeout_ms = None

if use_timeout:
    timeout_seconds = st.number_input(
        "Response timeout (seconds)",
        min_value=1,
        max_value=60,
        value=5,
        step=1,
    )

    response_timeout_ms = int(
        timeout_seconds * 1000
    )


st.divider()


if st.button(
    "Build participant experiment",
    type="primary",
    use_container_width=True,
):
    try:
        config = build_participant_package(
            stimuli,
            pairs,
            randomize_trials=(
                randomize_trials
            ),
            response_timeout_ms=(
                response_timeout_ms
            ),
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        st.error(
            str(error)
        )

    else:
        st.success(
            "Participant experiment built successfully."
        )

        st.write(
            f"Trials: {config['n_trials']}"
        )

        st.write(
            f"Stimuli: {config['n_stimuli']}"
        )

        st.code(
            str(
                EXPERIMENT_CONFIG_PATH
            )
        )

        st.code(
            str(
                PARTICIPANT_STIMULI_DIR
            )
        )