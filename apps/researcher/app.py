from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

STIMULI_PATH = Path("data/processed/stimuli.csv")
PAIRS_PATH = Path("data/processed/pairs.csv")
SCORES_PATH = Path("data/results/bradley_terry/scores.csv")
ALIGNMENT_MANIFEST_PATH = Path(
    "data/interim/aligned_faces/alignment_manifest.csv"
)
TRIANGULATION_MANIFEST_PATH = Path(
    "data/interim/common_triangulation/triangulation_manifest.csv"
)


st.set_page_config(
    page_title="Face Reconstruction Researcher",
    page_icon="🧠",
    layout="wide",
)


def csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None

    return len(
        pd.read_csv(path)
    )


def qc_pass_count(
    path: Path,
    column: str,
) -> tuple[int, int] | None:
    if not path.exists():
        return None

    table = pd.read_csv(path)

    if column not in table.columns:
        return None

    values = table[column].astype(bool)

    return int(values.sum()), len(table)


st.title("Face Reconstruction")
st.caption(
    "Researcher dashboard for experiment preparation, "
    "Bradley–Terry analysis, and facial reconstruction."
)

st.divider()

st.subheader("Pipeline status")

stimuli_count = csv_row_count(STIMULI_PATH)
pairs_count = csv_row_count(PAIRS_PATH)

alignment_status = qc_pass_count(
    ALIGNMENT_MANIFEST_PATH,
    "alignment_qc_pass",
)

topology_status = qc_pass_count(
    TRIANGULATION_MANIFEST_PATH,
    "topology_qc_pass",
)

if SCORES_PATH.exists():
    scores = pd.read_csv(SCORES_PATH)

    participant_count = (
        scores["participant_id"].nunique()
        if "participant_id" in scores.columns
        else None
    )
else:
    participant_count = None


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Stimuli",
        stimuli_count if stimuli_count is not None else "—",
    )

with col2:
    st.metric(
        "Pairs",
        pairs_count if pairs_count is not None else "—",
    )

with col3:
    st.metric(
        "Participants fitted",
        participant_count if participant_count is not None else "—",
    )

with col4:
    if alignment_status is None:
        alignment_label = "—"
    else:
        passed, total = alignment_status
        alignment_label = f"{passed}/{total}"

    st.metric(
        "Aligned faces",
        alignment_label,
    )


st.divider()

st.subheader("Geometry")

col1, col2 = st.columns(2)

with col1:
    if alignment_status is None:
        st.info(
            "No alignment manifest found."
        )
    else:
        passed, total = alignment_status

        st.write(
            f"Alignment QC: **{passed}/{total} passed**"
        )

with col2:
    if topology_status is None:
        st.info(
            "No triangulation manifest found."
        )
    else:
        passed, total = topology_status

        st.write(
            f"Topology QC: **{passed}/{total} passed**"
        )


st.divider()

st.subheader("Research workflow")

st.markdown(
    """
    Use this dashboard to:

    1. prepare and inspect the stimulus set;
    2. generate pairwise comparisons;
    3. import participant responses;
    4. fit participant-specific Bradley–Terry models;
    5. inspect reconstruction weights;
    6. reconstruct participant-specific facial prototypes.
    """
)

st.info(
    "The scientific computations remain in "
    "`src/face_reconstruction/`. "
    "This application only provides an interface to them."
)