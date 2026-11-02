from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from face_reconstruction.data.stimuli import build_stimuli_table

HARMONISED_MANIFEST_PATH = Path(
    "data/interim/cfd_audit/cfd_manifest_harmonised.csv"
)

IMAGE_DIRECTORY = Path(
    "data/raw/cfd/images"
)

ACTIVE_STIMULI_PATH = Path(
    "data/processed/stimuli.csv"
)


st.set_page_config(
    page_title="Stimuli",
    page_icon="🖼️",
    layout="wide",
)


@st.cache_data
def load_manifest() -> pd.DataFrame:
    """Load the full harmonised CFD manifest."""

    if not HARMONISED_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Harmonised CFD manifest not found: "
            f"{HARMONISED_MANIFEST_PATH}"
        )

    return pd.read_csv(
        HARMONISED_MANIFEST_PATH
    )


def resolve_image_path(
    raw_path: str,
) -> Path:
    """Resolve an image path relative to the repository root."""

    path = Path(raw_path)

    if path.is_absolute():
        return path

    return Path.cwd() / path


def get_options(
    table: pd.DataFrame,
    column: str,
) -> list[str]:
    """Return sorted non-null values available in a column."""

    if column not in table.columns:
        return []

    return sorted(
        table[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def render_distribution(
    table: pd.DataFrame,
    column: str,
    title: str,
) -> None:
    """Display counts for a categorical variable."""

    st.markdown(
        f"**{title}**"
    )

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


def sample_stimulus_pool(
    stimuli: pd.DataFrame,
    *,
    group_column: str,
    target_counts: dict[str, int],
    random_state: int,
) -> pd.DataFrame:
    """Sample a controlled number of stimuli from each group."""

    if group_column not in stimuli.columns:
        raise ValueError(
            f"Column not found: {group_column}"
        )

    selected_parts: list[pd.DataFrame] = []

    for group, target_count in target_counts.items():
        available = stimuli.loc[
            stimuli[group_column]
            .astype(str)
            .eq(group)
        ]

        if target_count > len(available):
            raise ValueError(
                f"Requested {target_count} faces for {group}, "
                f"but only {len(available)} are available."
            )

        if target_count == 0:
            continue

        sampled = available.sample(
            n=target_count,
            random_state=random_state,
        )

        selected_parts.append(
            sampled
        )

    if not selected_parts:
        return stimuli.iloc[
            0:0
        ].copy()

    result = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    return (
        result.sample(
            frac=1.0,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )


def save_active_stimuli(
    stimuli: pd.DataFrame,
) -> None:
    """Save the final experimental stimulus pool."""

    ACTIVE_STIMULI_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stimuli.to_csv(
        ACTIVE_STIMULI_PATH,
        index=False,
    )


st.title("Stimuli")

st.caption(
    "Configure the eligible face set and define the final "
    "experimental stimulus pool."
)


try:
    manifest = load_manifest()

except FileNotFoundError as error:
    st.error(
        str(error)
    )
    st.stop()


# ---------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------

st.divider()

st.subheader(
    "Source dataset"
)

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
        manifest[
            "ethnicity_perceived"
        ].nunique()
        if "ethnicity_perceived"
        in manifest.columns
        else "—"
    )

    st.metric(
        "Perceived ethnicity groups",
        value,
    )


# ---------------------------------------------------------------------
# Eligibility filters
# ---------------------------------------------------------------------

st.divider()

st.subheader(
    "1. Eligibility filters"
)

st.caption(
    "Choose which faces are eligible before defining the "
    "final experimental pool."
)


gender_options = get_options(
    manifest,
    "gender_self",
)

self_ethnicity_options = get_options(
    manifest,
    "ethnicity_self",
)

perceived_ethnicity_options = get_options(
    manifest,
    "ethnicity_perceived",
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
            "Choose whether inclusion criteria use perceived "
            "ethnicity, self-reported ethnicity, or both."
        ),
    )

    selected_genders = st.multiselect(
        "Gender",
        options=gender_options,
        default=gender_options,
        help=(
            "Filter using self-reported gender."
        ),
    )

    selected_self_ethnicities = st.multiselect(
        "Self-reported ethnicities",
        options=self_ethnicity_options,
        default=self_ethnicity_options,
        disabled=selection_mode == "perceived",
    )


with right:
    selected_perceived_ethnicities = st.multiselect(
        "Perceived ethnicities",
        options=perceived_ethnicity_options,
        default=perceived_ethnicity_options,
        disabled=selection_mode == "self",
    )

    minimum_perceived_ethnicity = st.slider(
        "Minimum perceived ethnicity probability",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.01,
        disabled=selection_mode == "self",
        help=(
            "Minimum confidence required for the dominant "
            "perceived ethnicity."
        ),
    )

    minimum_perceived_gender = st.slider(
        "Minimum perceived gender probability",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.01,
        help=(
            "Minimum confidence required for perceived gender."
        ),
    )


try:
    filtered = build_stimuli_table(
        manifest=manifest,
        image_directory=IMAGE_DIRECTORY,
        genders=(
            selected_genders
            if selected_genders
            else None
        ),
        minimum_perceived_gender=(
            minimum_perceived_gender
        ),
        ethnicities_self=(
            selected_self_ethnicities
            if (
                selection_mode
                in {"self", "both"}
                and selected_self_ethnicities
            )
            else None
        ),
        ethnicities_perceived=(
            selected_perceived_ethnicities
            if (
                selection_mode
                in {"perceived", "both"}
                and selected_perceived_ethnicities
            )
            else None
        ),
        minimum_perceived_probability=(
            minimum_perceived_ethnicity
            if selection_mode
            in {"perceived", "both"}
            else None
        ),
        ethnicity_selection_mode=(
            selection_mode
        ),
    )

except (ValueError, FileNotFoundError) as error:
    st.error(
        str(error)
    )
    st.stop()


st.metric(
    "Eligible faces",
    len(filtered),
)


# ---------------------------------------------------------------------
# Final pool
# ---------------------------------------------------------------------

st.divider()

st.subheader(
    "2. Final stimulus pool"
)

st.caption(
    "Keep all eligible faces or draw a smaller controlled "
    "experimental pool."
)


sampling_mode = st.radio(
    "Pool selection",
    options=[
        "Keep all eligible faces",
        "Sample a controlled subset",
    ],
    horizontal=True,
)


active_selection = (
    filtered.copy()
)


sampling_group_column = None
sampling_seed = None
target_counts: dict[str, int] = {}


if sampling_mode == "Sample a controlled subset":
    available_sampling_columns = [
        column
        for column in [
            "ethnicity_perceived",
            "ethnicity_self",
            "gender_self",
        ]
        if column in filtered.columns
    ]

    if not available_sampling_columns:
        st.error(
            "No categorical variable is available "
            "for controlled sampling."
        )

        active_selection = (
            filtered.iloc[0:0].copy()
        )

    else:
        col1, col2 = st.columns(2)

        with col1:
            sampling_group_column = st.selectbox(
                "Variable used to control pool composition",
                options=available_sampling_columns,
                index=0,
            )

        with col2:
            sampling_seed = st.number_input(
                "Sampling seed",
                min_value=0,
                value=42,
                step=1,
            )

        groups = sorted(
            filtered[
                sampling_group_column
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        st.markdown(
            "**Number of faces to keep per group**"
        )

        if not groups:
            st.warning(
                "No groups are available after filtering."
            )

            active_selection = (
                filtered.iloc[0:0].copy()
            )

        else:
            columns = st.columns(
                min(
                    3,
                    len(groups),
                )
            )

            for index, group in enumerate(
                groups
            ):
                available_count = int(
                    (
                        filtered[
                            sampling_group_column
                        ]
                        .astype(str)
                        .eq(group)
                    ).sum()
                )

                with columns[
                    index % len(columns)
                ]:
                    target_count = st.number_input(
                        f"{group}",
                        min_value=0,
                        max_value=available_count,
                        value=available_count,
                        step=1,
                        key=(
                            f"pool_count_"
                            f"{sampling_group_column}_"
                            f"{group}"
                        ),
                        help=(
                            f"{available_count} eligible "
                            f"faces available."
                        ),
                    )

                target_counts[group] = int(
                    target_count
                )

            try:
                active_selection = sample_stimulus_pool(
                    filtered,
                    group_column=(
                        sampling_group_column
                    ),
                    target_counts=target_counts,
                    random_state=int(
                        sampling_seed
                    ),
                )

            except ValueError as error:
                st.error(
                    str(error)
                )

                active_selection = (
                    filtered.iloc[
                        0:0
                    ].copy()
                )


st.metric(
    "Final pool size",
    len(active_selection),
)


# ---------------------------------------------------------------------
# Save active stimulus set
# ---------------------------------------------------------------------

st.markdown(
    "### Active stimulus set"
)

st.caption(
    "Saving this pool makes it the input used automatically "
    "by the Pair Design page."
)


save_col, download_col = st.columns(2)


with save_col:
    if st.button(
        "Save as active stimulus set",
        type="primary",
        use_container_width=True,
        disabled=active_selection.empty,
    ):
        save_active_stimuli(
            active_selection
        )

        st.success(
            "Active stimulus set saved to "
            f"{ACTIVE_STIMULI_PATH}"
        )


with download_col:
    csv_data = (
        active_selection.to_csv(
            index=False
        )
        .encode("utf-8")
    )

    st.download_button(
        "Download stimulus pool CSV",
        data=csv_data,
        file_name="stimuli.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=active_selection.empty,
    )


# ---------------------------------------------------------------------
# Selection summary
# ---------------------------------------------------------------------

st.divider()


with st.expander(
    "Selection summary",
    expanded=False,
):
    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:
        st.metric(
            "Eligible faces",
            len(filtered),
        )

        st.metric(
            "Final pool",
            len(active_selection),
        )

        st.markdown(
            "**Eligibility configuration**"
        )

        st.write(
            {
                "selection_mode": (
                    selection_mode
                ),
                "genders": (
                    selected_genders
                ),
                "ethnicities_self": (
                    selected_self_ethnicities
                ),
                "ethnicities_perceived": (
                    selected_perceived_ethnicities
                ),
                "minimum_perceived_ethnicity": (
                    minimum_perceived_ethnicity
                ),
                "minimum_perceived_gender": (
                    minimum_perceived_gender
                ),
            }
        )

        if (
            sampling_mode
            == "Sample a controlled subset"
        ):
            st.markdown(
                "**Pool sampling configuration**"
            )

            st.write(
                {
                    "group_column": (
                        sampling_group_column
                    ),
                    "target_counts": (
                        target_counts
                    ),
                    "sampling_seed": (
                        sampling_seed
                    ),
                }
            )

    with col2:
        dist1, dist2, dist3 = st.columns(
            3
        )

        with dist1:
            render_distribution(
                active_selection,
                "gender_self",
                "Gender",
            )

        with dist2:
            render_distribution(
                active_selection,
                "ethnicity_self",
                "Self-reported ethnicity",
            )

        with dist3:
            render_distribution(
                active_selection,
                "ethnicity_perceived",
                "Perceived ethnicity",
            )


# ---------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------

st.divider()

st.subheader(
    "Stimulus gallery"
)


if active_selection.empty:
    st.info(
        "No image to display for the current selection."
    )

else:
    with st.expander(
        f"Show selected faces ({len(active_selection)})",
        expanded=False,
    ):
        columns_per_row = st.slider(
            "Images per row",
            min_value=3,
            max_value=8,
            value=6,
            step=1,
        )

        records = (
            active_selection.to_dict(
                orient="records"
            )
        )

        for start in range(
            0,
            len(records),
            columns_per_row,
        ):
            row_records = records[
                start : start
                + columns_per_row
            ]

            columns = st.columns(
                columns_per_row
            )

            for column, record in zip(
                columns,
                row_records,
            ):
                with column:
                    image_path_value = (
                        record.get(
                            "image_path",
                            "",
                        )
                    )

                    image_path = (
                        resolve_image_path(
                            str(
                                image_path_value
                            )
                        )
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
                        record.get(
                            "face_id",
                            "unknown",
                        )
                    )

                    st.caption(
                        face_id
                    )

                    if (
                        "ethnicity_perceived"
                        in record
                    ):
                        st.write(
                            "Perceived: "
                            f"{record['ethnicity_perceived']}"
                        )

                    if (
                        "ethnicity_self"
                        in record
                    ):
                        st.write(
                            "Self: "
                            f"{record['ethnicity_self']}"
                        )


# ---------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------

st.divider()


with st.expander(
    "Show raw final stimulus table",
    expanded=False,
):
    st.dataframe(
        active_selection,
        use_container_width=True,
        hide_index=True,
    )