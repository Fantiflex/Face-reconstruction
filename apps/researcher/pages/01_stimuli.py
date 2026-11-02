from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

HARMONISED_MANIFEST_PATH = Path(
    "data/interim/cfd_audit/cfd_manifest_harmonised.csv"
)

IMAGE_DIRECTORY = Path(
    "data/raw/cfd/images"
)

DEFAULT_SAVE_PATH = Path("data/processed/stimuli_selected.csv")

from face_reconstruction.data.stimuli import build_stimuli_table

st.set_page_config(
    page_title="Stimuli",
    page_icon="🖼️",
    layout="wide",
)


@st.cache_data
def load_manifest() -> pd.DataFrame:
    if not HARMONISED_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Harmonised CFD manifest not found: "
            f"{HARMONISED_MANIFEST_PATH}"
        )

    return pd.read_csv(
        HARMONISED_MANIFEST_PATH
    )


def resolve_image_path(raw_path: str) -> Path:
    path = Path(raw_path)

    if path.is_absolute():
        return path

    return Path.cwd() / path


def optional_str_options(
    table: pd.DataFrame,
    column: str,
) -> list[str]:
    if column not in table.columns:
        return []

    return sorted(
        table[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def filter_stimuli(
    stimuli: pd.DataFrame,
    *,
    selected_genders: list[str],
    selected_self_ethnicities: list[str],
    selected_perceived_ethnicities: list[str],
    selection_mode: str,
    minimum_perceived_ethnicity: float,
    minimum_perceived_gender: float,
) -> pd.DataFrame:
    filtered = stimuli.copy()

    if selected_genders and "gender_self" in filtered.columns:
        filtered = filtered.loc[
            filtered["gender_self"]
            .astype(str)
            .isin(selected_genders)
        ]

    if selection_mode in {"self", "both"} and (
        selected_self_ethnicities
        and "ethnicity_self" in filtered.columns
    ):
        filtered = filtered.loc[
            filtered["ethnicity_self"]
            .astype(str)
            .isin(selected_self_ethnicities)
        ]

    if selection_mode in {"perceived", "both"}:
        if (
            selected_perceived_ethnicities
            and "ethnicity_perceived" in filtered.columns
        ):
            filtered = filtered.loc[
                filtered["ethnicity_perceived"]
                .astype(str)
                .isin(selected_perceived_ethnicities)
            ]

        if (
            "ethnicity_perceived_probability"
            in filtered.columns
        ):
            filtered = filtered.loc[
                pd.to_numeric(
                    filtered[
                        "ethnicity_perceived_probability"
                    ],
                    errors="coerce",
                )
                >= minimum_perceived_ethnicity
            ]

    if "gender_perceived_probability" in filtered.columns:
        filtered = filtered.loc[
            pd.to_numeric(
                filtered[
                    "gender_perceived_probability"
                ],
                errors="coerce",
            )
            >= minimum_perceived_gender
        ]

    return filtered.reset_index(drop=True)


def render_distribution(
    table: pd.DataFrame,
    column: str,
    title: str,
) -> None:
    st.markdown(f"**{title}**")

    if column not in table.columns:
        st.info(
            f"Column `{column}` not found."
        )
        return

    counts = (
        table[column]
        .fillna("missing")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
    )

    st.dataframe(
        counts,
        use_container_width=True,
        hide_index=True,
    )


def save_selection_to_disk(
    filtered: pd.DataFrame,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    filtered.to_csv(save_path, index=False)


st.title("Stimuli")
st.caption(
    "Configure, inspect, and export the current face stimulus set."
)

try:
    manifest = load_manifest()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

st.divider()

st.subheader("Current dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total faces",
        len(manifest),
    )

with col2:
    value = (
        manifest["gender_self"].nunique()
        if "gender_self" in manifest.columns
        else "—"
    )

    st.metric(
        "Gender groups",
        value,
    )

with col3:
    value = (
        manifest["ethnicity_self"].nunique()
        if "ethnicity_self" in manifest.columns
        else "—"
    )

    st.metric(
        "Self ethnicity groups",
        value,
    )

with col4:
    value = (
        manifest["ethnicity_perceived"].nunique()
        if "ethnicity_perceived" in manifest.columns
        else "—"
    )

    st.metric(
        "Perceived ethnicity groups",
        value,
    )

st.divider()

st.subheader("Stimulus configuration")

gender_options = sorted(
    manifest["gender_self"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

self_ethnicity_options = sorted(
    manifest["ethnicity_self"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

perceived_ethnicity_options = sorted(
    manifest["ethnicity_perceived"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

left, right = st.columns(2)

with left:
    selection_mode = st.selectbox(
        "Ethnicity selection mode",
        options=[
            "perceived",
            "self",
            "both",
        ],
        index=0,
        help=(
            "Choose whether stimulus selection uses "
            "perceived ethnicity, self-reported ethnicity, "
            "or both simultaneously."
        ),
    )

    selected_genders = st.multiselect(
        "Gender",
        options=gender_options,
        default=gender_options,
        help=(
            "Filter by self-reported gender."
        ),
    )

    selected_self_ethnicities = st.multiselect(
        "Self-reported ethnicities",
        options=self_ethnicity_options,
        default=self_ethnicity_options,
        help=(
            "Used when selection mode includes "
            "`self`."
        ),
    )

with right:
    selected_perceived_ethnicities = st.multiselect(
        "Perceived ethnicities",
        options=perceived_ethnicity_options,
        default=perceived_ethnicity_options,
        help=(
            "Used when selection mode includes "
            "`perceived`."
        ),
    )

    minimum_perceived_ethnicity = st.slider(
        "Minimum perceived ethnicity probability",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.01,
        help=(
            "Lower bound on the dominant perceived "
            "ethnicity probability."
        ),
    )

    minimum_perceived_gender = st.slider(
        "Minimum perceived gender probability",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.01,
        help=(
            "Lower bound on perceived gender "
            "probability."
        ),
    )

filtered = build_stimuli_table(
    manifest=manifest,
    image_directory=IMAGE_DIRECTORY,
    genders=(
        selected_genders
        if selected_genders
        else None
    ),
    minimum_perceived_gender=minimum_perceived_gender,
    ethnicities_self=(
        selected_self_ethnicities
        if selected_self_ethnicities
        else None
    ),
    ethnicities_perceived=(
        selected_perceived_ethnicities
        if selected_perceived_ethnicities
        else None
    ),
    minimum_perceived_probability=(
        minimum_perceived_ethnicity
    ),
    ethnicity_selection_mode=selection_mode,
)

st.divider()

st.subheader("Selection summary")

col1, col2 = st.columns([1, 2])

with col1:
    st.metric(
        "Selected faces",
        len(filtered),
    )

    st.write("**Current configuration**")
    st.write(
        {
            "selection_mode": selection_mode,
            "genders": selected_genders,
            "ethnicities_self": selected_self_ethnicities,
            "ethnicities_perceived": selected_perceived_ethnicities,
            "minimum_perceived_ethnicity": minimum_perceived_ethnicity,
            "minimum_perceived_gender": minimum_perceived_gender,
        }
    )

    if filtered.empty:
        st.error(
            "No stimulus matches the current configuration."
        )
    else:
        csv_data = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download selected stimuli CSV",
            data=csv_data,
            file_name="stimuli_selected.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button(
            "Save selection to data/processed/stimuli_selected.csv",
            use_container_width=True,
        ):
            save_selection_to_disk(
                filtered,
                DEFAULT_SAVE_PATH,
            )
            st.success(
                f"Saved selection to: {DEFAULT_SAVE_PATH}"
            )

with col2:
    dist1, dist2, dist3 = st.columns(3)

    with dist1:
        render_distribution(
            filtered,
            "gender_self",
            "Gender",
        )

    with dist2:
        render_distribution(
            filtered,
            "ethnicity_self",
            "Self-reported ethnicity",
        )

    with dist3:
        render_distribution(
            filtered,
            "ethnicity_perceived",
            "Perceived ethnicity",
        )

st.divider()

st.subheader("Stimulus gallery")

if filtered.empty:
    st.info(
        "No image to display for the current selection."
    )
else:
    columns_per_row = st.slider(
        "Images per row",
        min_value=3,
        max_value=8,
        value=6,
        step=1,
    )

    records = filtered.to_dict(
        orient="records"
    )

    for start in range(
        0,
        len(records),
        columns_per_row,
    ):
        row_records = records[
            start : start + columns_per_row
        ]

        columns = st.columns(
            columns_per_row
        )

        for column, record in zip(
            columns,
            row_records,
        ):
            with column:
                image_path_value = record.get(
                    "image_path",
                    "",
                )
                image_path = resolve_image_path(
                    str(image_path_value)
                )

                if image_path.exists():
                    st.image(
                        str(image_path),
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "Image not found"
                    )

                face_id = str(
                    record.get("face_id", "unknown")
                )
                st.caption(face_id)

                if "ethnicity_perceived" in record:
                    st.write(
                        f"Perceived: {record['ethnicity_perceived']}"
                    )

                if "ethnicity_self" in record:
                    st.write(
                        f"Self: {record['ethnicity_self']}"
                    )

st.divider()

with st.expander("Raw selected stimulus table"):
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )