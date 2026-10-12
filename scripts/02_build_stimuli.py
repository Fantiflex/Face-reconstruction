"""
Build the study-specific CFD stimulus table.

This script loads the harmonised CFD metadata, matches metadata rows with
local images, applies demographic and perceptual inclusion criteria, validates
the selected stimuli, and saves the resulting study manifest.
"""

import argparse
from pathlib import Path

import pandas as pd

from face_reconstruction.data.cfd_manifest import (
    harmonize_cfd_manifest,
    load_cfd_data,
)
from face_reconstruction.data.stimuli import (
    build_stimuli_table,
)
from face_reconstruction.validation import (
    validate_stimuli,
)

WORKBOOK_PATH = Path(
    "data/raw/cfd/CFD_codebook.xlsx"
)

HARMONISED_MANIFEST_PATH = Path(
    "data/interim/cfd_audit/cfd_manifest_harmonised.csv"
)

IMAGE_DIRECTORY = Path(
    "data/raw/cfd/images"
)

OUTPUT_PATH = Path(
    "data/processed/stimuli.csv"
)

GENDERS = ["M"] #options are M, F, or None for all self-reported

ETHNICITIES_SELF = None # options are B, W, H, A, O, or None for all self-reported

ETHNICITIES_PERCEIVED = ["Black", "White"] # options are Black, White, Hispanic, Asian, Other, or None for all perceived

ETHNICITY_SELECTION_MODE = "perceived"

MINIMUM_PERCEIVED_PROBABILITY = 0.9

MINIMUM_PERCEIVED_GENDER = 0.9


def normalise_cli_selection(
    values: list[str] | None,
) -> list[str] | None:
    if values is None:
        return None

    if any(value.lower() == "all" for value in values):
        return None

    return values

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the standardized CFD stimulus table."
    )

    parser.add_argument(
        "--workbook-path",
        type=Path,
        default=WORKBOOK_PATH,
        help=f"CFD workbook path. Default: {WORKBOOK_PATH}",
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=IMAGE_DIRECTORY,
        help=f"CFD image directory. Default: {IMAGE_DIRECTORY}",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Stimulus CSV output path. Default: {OUTPUT_PATH}",
    )

    parser.add_argument(
    "--genders",
    nargs="+",
    default=GENDERS,
    metavar="GENDER",
    help=(
        "Self-reported genders to include, e.g. "
        "'--genders M' or '--genders M F'. "
        "Use 'all' for no gender filter. "
        f"Default: {GENDERS}"
    ),
)

    parser.add_argument(
        "--ethnicities-self",
        nargs="+",
        default=ETHNICITIES_SELF,
        metavar="ETHNICITY",
        help=(
            "Self-reported ethnicities to include. "
            "Use 'all' for no self-reported ethnicity filter. "
            f"Default: {ETHNICITIES_SELF}"
        ),
    )

    parser.add_argument(
        "--ethnicities-perceived",
        nargs="+",
        default=ETHNICITIES_PERCEIVED,
        metavar="ETHNICITY",
        help=(
            "Perceived ethnicities to include, e.g. "
            "'--ethnicities-perceived Black White'. "
            "Use 'all' for no perceived ethnicity filter. "
            f"Default: {ETHNICITIES_PERCEIVED}"
        ),
    )

    parser.add_argument(
        "--ethnicity-selection-mode",
        choices=["self", "perceived", "both"],
        default=ETHNICITY_SELECTION_MODE,
        help=(
            "Which ethnicity criterion to apply. "
            f"Default: {ETHNICITY_SELECTION_MODE}"
        ),
    )

    parser.add_argument(
        "--min-perceived-probability",
        type=float,
        default=MINIMUM_PERCEIVED_PROBABILITY,
        help=(
            "Minimum probability for the dominant perceived ethnicity. "
            f"Default: {MINIMUM_PERCEIVED_PROBABILITY}"
        ),
    )

    parser.add_argument(
        "--min-perceived-gender",
        type=float,
        default=MINIMUM_PERCEIVED_GENDER,
        help=(
            "Minimum perceived-gender probability. "
            f"Default: {MINIMUM_PERCEIVED_GENDER}"
        ),
    )
    return parser.parse_args()

def build_selection_summary(
    stimuli: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count selected stimuli by demographic and perceived categories.

    Parameters
    ----------
    stimuli:
        Final study-specific stimulus table.

    Returns
    -------
    pd.DataFrame
        Group counts for self-reported gender, self-reported ethnicity, and
        dominant perceived ethnicity.
    """

    grouping_columns = [
        "gender_self",
        "ethnicity_self",
        "ethnicity_perceived",
    ]

    available_grouping_columns = [
        column
        for column in grouping_columns
        if column in stimuli.columns
    ]

    summary = (
        stimuli.groupby(
            available_grouping_columns,
            dropna=False,
        )
        .size()
        .rename("n_stimuli")
        .reset_index()
        .sort_values(
            by=available_grouping_columns
        )
    )

    return summary.reset_index(drop=True)


def save_stimuli_outputs(
    stimuli: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save the selected stimuli and their demographic summary.

    Parameters
    ----------
    stimuli:
        Final validated stimulus table.

    output_path:
        CSV path for the final stimulus table.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stimuli.to_csv(
        output_path,
        index=False,
    )

    selection_summary = (
        build_selection_summary(
            stimuli
        )
    )

    summary_path = (
        output_path.parent
        / "stimuli_selection_summary.csv"
    )

    selection_summary.to_csv(
        summary_path,
        index=False,
    )


def print_stimuli_summary(
    stimuli: pd.DataFrame,
    *,
    genders: list[str] | None,
    ethnicities_self: list[str] | None,
    ethnicities_perceived: list[str] | None,
    ethnicity_selection_mode: str,
    minimum_perceived_probability: float | None,
    minimum_perceived_gender: float | None,
) -> None:
    """
    Print a summary of the selected experimental stimuli.

    Parameters
    ----------
    stimuli:
        Final study-specific stimulus table.
    """

    print("\nStimulus selection")
    print("------------------")
    print("\nSelection criteria:")
    print(f"- genders: {genders or 'all'}")
    print(
        "- self-reported ethnicities: "
        f"{ethnicities_self or 'all'}"
    )
    print(
        "- perceived ethnicities: "
        f"{ethnicities_perceived or 'all'}"
    )
    print(
        "- ethnicity selection mode: "
        f"{ethnicity_selection_mode}"
    )
    print(
        "- minimum perceived ethnicity probability: "
        f"{minimum_perceived_probability}"
    )
    print(
        "- minimum perceived gender probability: "
        f"{minimum_perceived_gender}"
    )
    print(
        f"Selected stimuli: {len(stimuli)}"
    )

    if "gender_self" in stimuli.columns:
        print("\nGender:")
        print(
            stimuli["gender_self"]
            .value_counts(dropna=False)
        )

    if "ethnicity_self" in stimuli.columns:
        print("\nSelf-reported ethnicity:")
        print(
            stimuli["ethnicity_self"]
            .value_counts(dropna=False)
        )

    if (
        "ethnicity_perceived"
        in stimuli.columns
    ):
        print("\nPerceived ethnicity:")
        print(
            stimuli["ethnicity_perceived"]
            .value_counts(dropna=False)
        )


def main() -> None:
    """
    Build, validate, and save the study-specific stimulus table.
    """
    args = parse_arguments()
    if HARMONISED_MANIFEST_PATH.exists():
        manifest = pd.read_csv(HARMONISED_MANIFEST_PATH)
    else:
        raw_cfd_data = load_cfd_data(
            workbook_path=args.workbook_path
        )

        manifest = harmonize_cfd_manifest(
            raw_cfd_data
        )

        HARMONISED_MANIFEST_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        manifest.to_csv(
            HARMONISED_MANIFEST_PATH,
            index=False,
        )

    genders = normalise_cli_selection(
        args.genders
    )

    ethnicities_self = normalise_cli_selection(
        args.ethnicities_self
    )

    ethnicities_perceived = normalise_cli_selection(
        args.ethnicities_perceived
    )


    stimuli = build_stimuli_table(
        manifest=manifest,
        image_directory=args.image_dir,
        genders=genders,
        minimum_perceived_gender=args.min_perceived_gender,
        ethnicities_self=ethnicities_self,
        ethnicities_perceived=ethnicities_perceived,
        minimum_perceived_probability=(
            args.min_perceived_probability
        ),
        ethnicity_selection_mode=(
            args.ethnicity_selection_mode
        ),
    )
    validate_stimuli(
        stimuli
    )

    save_stimuli_outputs(
        stimuli=stimuli,
        output_path=args.output_path,
    )

    print_stimuli_summary(
        stimuli,
        genders=genders,
        ethnicities_self=ethnicities_self,
        ethnicities_perceived=ethnicities_perceived,
        ethnicity_selection_mode=(
            args.ethnicity_selection_mode
        ),
        minimum_perceived_probability=(
            args.min_perceived_probability
        ),
        minimum_perceived_gender=(
            args.min_perceived_gender
        ),
    )

    print(
        "\nStimulus table saved to: "
        f"{args.output_path}"
    )


if __name__ == "__main__":
    main()