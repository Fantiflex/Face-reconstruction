from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from face_reconstruction.geometry.warping import (
    warp_piecewise_affine,
)

from face_reconstruction.reconstruction.morph import (
    build_target_points,
)

from face_reconstruction.reconstruction.weighting import (
    softmax_weights,
)

from face_reconstruction.reconstruction.morph import (
    accumulate_weighted_warp,
    build_target_points,
    finalize_weighted_reconstruction,
    load_reconstruction_inputs,
)


from face_reconstruction.utils.io import (
    parse_bool,
    resolve_path,
    sha256_arrays,
)



DEFAULT_ALIGNED_DIR = Path("data/interim/aligned_faces")
DEFAULT_TRIANGULATION_DIR = Path("data/interim/common_triangulation")
DEFAULT_BT_OUTPUT_DIR = Path("data/results/morphing/bradley_terry")
DEFAULT_BT_OUTPUT_NAME = "bradley_terry_morph.png"
DEFAULT_UNIFORM_OUTPUT_DIR = Path("data/results/morphing/uniform")
DEFAULT_UNIFORM_OUTPUT_NAME = "uniform_morph.png"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Delaunay morph from QC-passed aligned faces using either "
            "uniform weights or Bradley-Terry weights, and the stable landmark "
            "subset saved by step 09."
        )
    )
    parser.add_argument(
        "--weight-mode",
        choices=("bradley-terry", "uniform"),
        default="bradley-terry",
        help=(
            "Weighting scheme. 'bradley-terry' requires --weights-path; "
            "'uniform' reproduces the baseline. Default: bradley-terry."
        ),
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help=(
            "CSV containing one row per face. Required in Bradley-Terry mode. "
            "Provide either --weight-column or --score-column."
        ),
    )
    parser.add_argument(
        "--face-id-column",
        type=str,
        default="face_id",
        help="Face identifier column in the Bradley-Terry CSV. Default: face_id.",
    )
    parser.add_argument(
        "--participant-id-column",
        type=str,
        default="participant_id",
        help=(
            "Participant identifier column in the Bradley-Terry CSV. The CSV "
            "must contain exactly one participant. Default: participant_id."
        ),
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="theta",
        help=(
            "Bradley-Terry utility column transformed with softmax when "
            "--weight-column is omitted. Default: theta."
        ),
    )
    parser.add_argument(
        "--weight-column",
        type=str,
        default=None,
        help=(
            "Optional column containing precomputed non-negative weights. "
            "These values are renormalized to sum to one."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax temperature applied to Bradley-Terry utilities. Used only "
            "when --weight-column is omitted. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Optionally retain only the k highest-scoring/highest-weight faces "
            "before normalization. Default: use every eligible face."
        ),
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=DEFAULT_ALIGNED_DIR,
        help=f"Directory produced by step 08. Default: {DEFAULT_ALIGNED_DIR}",
    )
    parser.add_argument(
        "--triangulation-dir",
        type=Path,
        default=DEFAULT_TRIANGULATION_DIR,
        help=(
            "Directory produced by step 09. "
            f"Default: {DEFAULT_TRIANGULATION_DIR}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Base output directory. In Bradley-Terry mode, a participant "
            "subdirectory is added automatically. Defaults depend on "
            f"--weight-mode: {DEFAULT_BT_OUTPUT_DIR} or "
            f"{DEFAULT_UNIFORM_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Output PNG filename. Defaults depend on --weight-mode: "
            f"{DEFAULT_BT_OUTPUT_NAME} or {DEFAULT_UNIFORM_OUTPUT_NAME}."
        ),
    )
    parser.add_argument(
        "--interpolation",
        choices=("linear", "cubic"),
        default="linear",
        help="Piecewise-affine interpolation. Default: linear.",
    )
    parser.add_argument(
        "--border-mode",
        choices=("reflect101", "constant"),
        default="reflect101",
        help="Sampling outside source images. Default: reflect101.",
    )
    parser.add_argument(
        "--save-individual-warps",
        action="store_true",
        help="Save every included face after warping to the weighted target shape.",
    )
    parser.add_argument(
        "--allow-topology-failures",
        action="store_true",
        help=(
            "Keep aligned faces that failed topology QC, but skip only their "
            "per-face unstable triangles listed by step 09. By default, any "
            "topology failure stops the run."
        ),
    )
    parser.add_argument(
        "--min-triangle-area",
        type=float,
        default=0.01,
        help=(
            "Minimum absolute triangle area in pixels squared for the runtime "
            "geometric check. Default: 0.01."
        ),
    )
    parser.add_argument(
        "--relative-area-tolerance",
        type=float,
        default=1e-4,
        help=(
            "Relative degeneracy tolerance based on the longest squared edge. "
            "Default: 1e-4."
        ),
    )
    parser.add_argument(
        "--max-affine-condition-number",
        type=float,
        default=1e4,
        help=(
            "Maximum accepted condition number for the affine linear part. "
            "Default: 1e4."
        ),
    )
    parser.add_argument(
        "--max-uncovered-fraction",
        type=float,
        default=1e-4,
        help=(
            "Maximum output-pixel fraction with no valid contributing face. "
            "Default: 1e-4."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing reconstruction artifacts.",
    )
    return parser.parse_args()



def parse_index_list(value: Any) -> set[int]:
    """Parse a JSON list of non-negative triangle indices from a CSV cell."""
    if pd.isna(value) or str(value).strip() in {"", "[]"}:
        return set()

    parsed = json.loads(str(value))

    if not isinstance(parsed, list):
        raise ValueError(
            f"Expected a JSON list of triangle indices, got: {value!r}"
        )

    indices = {int(index) for index in parsed}

    if any(index < 0 for index in indices):
        raise ValueError(f"Negative triangle index in: {value!r}")

    return indices




def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "unnamed_face"



def require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    table_name: str,
) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{table_name} is missing columns: {sorted(missing)}")




def load_bradley_terry_weights(
    eligible_face_ids: list[str],
    *,
    weights_path: Path,
    face_id_column: str,
    participant_id_column: str,
    score_column: str,
    weight_column: str | None,
    temperature: float,
    top_k: int | None,
) -> tuple[list[str], np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Load, validate, subset, and normalize Bradley-Terry weights."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Bradley-Terry CSV not found: {weights_path}")
    table = pd.read_csv(
        weights_path,
        dtype={
            face_id_column: "string",
            participant_id_column: "string",
        },
    )
    value_column = weight_column if weight_column is not None else score_column
    require_columns(
        table,
        {participant_id_column, face_id_column, value_column},
        table_name="Bradley-Terry weights table",
    )

    table = table.copy()
    table[participant_id_column] = (
        table[participant_id_column].astype("string").str.strip()
    )
    if (
        table[participant_id_column].isna().any()
        or table[participant_id_column].eq("").any()
    ):
        raise ValueError(
            "Empty participant identifier in Bradley-Terry weights table."
        )
    participant_ids = table[participant_id_column].unique().tolist()
    if len(participant_ids) != 1:
        raise ValueError(
            "Bradley-Terry weights table must contain exactly one "
            f"participant; found: {sorted(map(str, participant_ids))}. "
            "Filter the scores CSV to one participant before morphing."
        )
    participant_id = str(participant_ids[0])

    table[face_id_column] = table[face_id_column].astype("string").str.strip()
    if table[face_id_column].isna().any() or table[face_id_column].eq("").any():
        raise ValueError("Empty face identifier in Bradley-Terry weights table.")
    if table[face_id_column].duplicated().any():
        duplicates = sorted(
            table.loc[
                table[face_id_column].duplicated(keep=False), face_id_column
            ].unique()
        )
        raise ValueError(
            "Duplicate face identifiers in Bradley-Terry weights table: "
            f"{duplicates}"
        )

    table[value_column] = pd.to_numeric(table[value_column], errors="coerce")
    invalid = ~np.isfinite(table[value_column].to_numpy(dtype=np.float64))
    if invalid.any():
        invalid_ids = table.loc[invalid, face_id_column].tolist()
        raise ValueError(
            f"Non-numeric or non-finite values in {value_column!r} for: "
            f"{invalid_ids}"
        )

    eligible_set = set(eligible_face_ids)
    available_set = set(table[face_id_column])
    missing_ids = sorted(eligible_set.difference(available_set))
    extra_ids = sorted(available_set.difference(eligible_set))
    if missing_ids:
        raise ValueError(
            "Eligible faces missing from the Bradley-Terry table: "
            f"{missing_ids}"
        )

    # Reorder exactly like the QC-passed morphing corpus. Extra CSV rows are
    # documented but never used silently.
    indexed = table.set_index(face_id_column, drop=False)
    matched = indexed.loc[eligible_face_ids].reset_index(drop=True)
    raw_values = matched[value_column].to_numpy(dtype=np.float64)

    if weight_column is not None:
        if np.any(raw_values < 0.0):
            bad_ids = matched.loc[raw_values < 0.0, face_id_column].tolist()
            raise ValueError(f"Precomputed weights are negative for: {bad_ids}")
        ranking_values = raw_values
        conversion = "precomputed_weights_renormalized"
    else:
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("--temperature must be finite and strictly positive.")
        ranking_values = raw_values
        conversion = "softmax_of_bradley_terry_scores"

    n_available = len(matched)
    all_raw_values = raw_values.copy()
    descending_order = np.argsort(-ranking_values, kind="stable")
    ranks = np.empty(n_available, dtype=np.int32)
    ranks[descending_order] = np.arange(1, n_available + 1, dtype=np.int32)
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("--top-k must be a strictly positive integer.")
        if top_k > n_available:
            raise ValueError(
                f"--top-k={top_k} exceeds the {n_available} eligible faces."
            )
        # Stable sort plus the pre-existing face_id order makes ties
        # deterministic and reproducible.
        selected_positions = descending_order[:top_k]
        selected_positions = np.sort(selected_positions)
        raw_values = raw_values[selected_positions]
    else:
        selected_positions = np.arange(n_available, dtype=np.int32)

    if weight_column is not None:
        total = float(raw_values.sum(dtype=np.float64))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Precomputed Bradley-Terry weights sum to zero.")
        normalized = raw_values / total
    else:
        normalized = softmax_weights(raw_values, temperature)

    if not np.isfinite(normalized).all() or np.any(normalized < 0.0):
        raise ValueError("Normalized Bradley-Terry weights are invalid.")
    if not np.isclose(normalized.sum(dtype=np.float64), 1.0, atol=1e-12):
        raise ValueError("Normalized Bradley-Terry weights do not sum to one.")

    diagnostic = matched.copy()
    diagnostic["weight_input_value"] = all_raw_values
    diagnostic["rank_by_input_value"] = ranks
    diagnostic["included_in_morph"] = False
    diagnostic["weight"] = 0.0
    diagnostic.loc[selected_positions, "included_in_morph"] = True
    diagnostic.loc[selected_positions, "weight"] = normalized
    selected_face_ids = diagnostic.loc[
        selected_positions, face_id_column
    ].astype(str).tolist()
    metadata = {
        "weights_path": weights_path.as_posix(),
        "participant_id": participant_id,
        "participant_id_column": participant_id_column,
        "face_id_column": face_id_column,
        "score_column": None if weight_column is not None else score_column,
        "weight_column": weight_column,
        "weight_conversion": conversion,
        "temperature": None if weight_column is not None else float(temperature),
        "top_k": top_k,
        "n_eligible_before_top_k": n_available,
        "n_used_after_top_k": len(selected_face_ids),
        "extra_face_ids_ignored": extra_ids,
    }
    return selected_face_ids, normalized, diagnostic, metadata


def load_pipeline_artifacts(
    aligned_dir: Path,
    triangulation_dir: Path,
    *,
    allow_topology_failures: bool,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    alignment_manifest_path = aligned_dir / "alignment_manifest.csv"
    topology_manifest_path = triangulation_dir / "triangulation_manifest.csv"
    common_points_path = triangulation_dir / "common_points.npy"
    triangles_path = triangulation_dir / "common_triangles.npy"
    selected_indices_path = triangulation_dir / "selected_landmark_indices.npy"
    config_path = triangulation_dir / "triangulation_config.json"

    required_paths = (
        alignment_manifest_path,
        topology_manifest_path,
        common_points_path,
        triangles_path,
        selected_indices_path,
        config_path,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required pipeline artifact not found: {path}")

    alignment = pd.read_csv(alignment_manifest_path)
    topology = pd.read_csv(topology_manifest_path)
    require_columns(
        alignment,
        {
            "face_id",
            "aligned_image_path",
            "aligned_landmark_path",
            "alignment_qc_pass",
        },
        table_name="Alignment manifest",
    )
    require_columns(
        topology,
        {
        "face_id",
        "topology_qc_pass",
        "topology_qc_reason",
        "n_triangles",
        "unstable_triangle_indices",
        },
        table_name="Triangulation manifest",
    )

    for frame, qc_column, table_name in (
        (alignment, "alignment_qc_pass", "alignment manifest"),
        (topology, "topology_qc_pass", "triangulation manifest"),
    ):
        frame["face_id"] = frame["face_id"].astype(str).str.strip()
        frame[qc_column] = parse_bool(frame[qc_column])
        if frame["face_id"].eq("").any():
            raise ValueError(f"Empty face_id in {table_name}.")
        if frame["face_id"].duplicated().any():
            duplicates = frame.loc[
                frame["face_id"].duplicated(keep=False), "face_id"
            ].tolist()
            raise ValueError(f"Duplicate face_id in {table_name}: {duplicates}")

    topology = topology.copy()
    topology["unstable_triangle_indices_parsed"] = topology[
        "unstable_triangle_indices"
    ].apply(parse_index_list)
    topology["n_triangles"] = pd.to_numeric(
        topology["n_triangles"], errors="coerce"
    )
    invalid_triangle_counts = (
        topology["n_triangles"].isna()
        | (topology["n_triangles"] < 1)
        | (topology["n_triangles"] % 1 != 0)
    )
    if invalid_triangle_counts.any():
        invalid_ids = topology.loc[invalid_triangle_counts, "face_id"].tolist()
        raise ValueError(
            "Invalid n_triangles in triangulation manifest for: "
            f"{invalid_ids}"
        )
    topology["n_triangles"] = topology["n_triangles"].astype(np.int64)

    aligned_pass = alignment.loc[alignment["alignment_qc_pass"]].copy()
    if aligned_pass.empty:
        raise ValueError("No face passed alignment QC in step 08.")

    topology_for_aligned = aligned_pass[["face_id"]].merge(
        topology[
            [
                "face_id",
                "topology_qc_pass",
                "topology_qc_reason",
                "n_triangles",
                "unstable_triangle_indices",
                "unstable_triangle_indices_parsed",
            ]
        ],
        on="face_id",
        how="left",
        validate="one_to_one",
    )
    missing_topology = topology_for_aligned["topology_qc_pass"].isna()
    if missing_topology.any():
        missing_ids = topology_for_aligned.loc[missing_topology, "face_id"].tolist()
        raise ValueError(
            "Aligned QC-passed faces are missing from the topology manifest: "
            f"{missing_ids}"
        )

    failed_topology = topology_for_aligned.loc[
        ~topology_for_aligned["topology_qc_pass"].astype(bool)
    ]
    if not failed_topology.empty and not allow_topology_failures:
        summary = ", ".join(
            f"{row.face_id} ({row.topology_qc_reason})"
            for row in failed_topology.itertuples(index=False)
        )
        raise ValueError(
            "Some alignment-QC-passed faces failed topology QC. Rebuild/fix step "
            f"09 before morphing: {summary}. To retain the faces while skipping "
            "only their unstable triangles, use --allow-topology-failures."
        )

    eligible = aligned_pass.merge(
        topology_for_aligned[
            [
                "face_id",
                "topology_qc_pass",
                "topology_qc_reason",
                "n_triangles",
                "unstable_triangle_indices",
                "unstable_triangle_indices_parsed",
            ]
        ],
        on="face_id",
        how="inner",
        validate="one_to_one",
    )
    eligible = eligible.sort_values("face_id", kind="stable").reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No alignment-QC-passed face is eligible for morphing.")

    common_points = np.asarray(
        np.load(common_points_path, allow_pickle=False), dtype=np.float64
    )
    triangles = np.asarray(
        np.load(triangles_path, allow_pickle=False), dtype=np.int32
    )
    selected_indices = np.asarray(
        np.load(selected_indices_path, allow_pickle=False), dtype=np.int32
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if common_points.ndim != 2 or common_points.shape[1] != 2:
        raise ValueError(f"Invalid common_points shape: {common_points.shape}")
    if not np.isfinite(common_points).all():
        raise ValueError("common_points contains non-finite coordinates.")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError(f"Invalid common_triangles shape: {triangles.shape}")
    if triangles.min() < 0 or triangles.max() >= len(common_points):
        raise ValueError("Triangle indices fall outside common_points.")
    manifest_triangle_counts = eligible["n_triangles"].to_numpy(dtype=np.int64)
    if not np.all(manifest_triangle_counts == len(triangles)):
        invalid_ids = eligible.loc[
            manifest_triangle_counts != len(triangles), "face_id"
        ].tolist()
        raise ValueError(
            "Triangulation-manifest counts do not match common_triangles.npy "
            f"for: {invalid_ids}"
        )
    invalid_unstable_indices = {
        str(row.face_id): sorted(
            index
            for index in row.unstable_triangle_indices_parsed
            if index >= len(triangles)
        )
        for row in eligible.itertuples(index=False)
    }
    invalid_unstable_indices = {
        face_id: indices
        for face_id, indices in invalid_unstable_indices.items()
        if indices
    }
    if invalid_unstable_indices:
        raise ValueError(
            "Unstable triangle indices fall outside common_triangles.npy: "
            f"{invalid_unstable_indices}"
        )
    if selected_indices.ndim != 1 or selected_indices.size == 0:
        raise ValueError(
            f"Invalid selected_landmark_indices shape: {selected_indices.shape}"
        )
    if len(np.unique(selected_indices)) != len(selected_indices):
        raise ValueError("selected_landmark_indices contains duplicates.")
    if np.any(selected_indices < 0):
        raise ValueError("selected_landmark_indices contains negative values.")

    n_detected = int(config["n_detected_landmarks"])
    n_selected = int(config["n_selected_landmarks"])
    n_anchors = int(config["n_boundary_anchors"])
    if len(selected_indices) != n_selected:
        raise ValueError(
            "selected_landmark_indices does not match triangulation_config.json."
        )
    if selected_indices.max() >= n_detected:
        raise ValueError(
            "selected_landmark_indices falls outside the detected landmark range."
        )
    if len(common_points) != n_selected + n_anchors:
        raise ValueError(
            "common_points count does not equal selected landmarks plus anchors."
        )
    if config.get("selected_landmark_indices_file") != selected_indices_path.name:
        raise ValueError(
            "triangulation_config.json references a different landmark-index file."
        )

    observed_hash = sha256_arrays(common_points, triangles)
    if observed_hash != config.get("topology_sha256"):
        raise ValueError(
            "Triangulation arrays do not match triangulation_config.json."
        )

    return eligible, common_points, triangles, selected_indices, config



def ensure_writable(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Morphing outputs already exist. Use --overwrite: "
            + ", ".join(existing)
        )


def main() -> int:
    args = parse_arguments()
    if not np.isfinite(args.min_triangle_area) or args.min_triangle_area < 0.0:
        raise ValueError("--min-triangle-area must be finite and non-negative.")
    if (
        not np.isfinite(args.relative_area_tolerance)
        or args.relative_area_tolerance < 0.0
    ):
        raise ValueError(
            "--relative-area-tolerance must be finite and non-negative."
        )
    if (
        not np.isfinite(args.max_affine_condition_number)
        or args.max_affine_condition_number <= 1.0
    ):
        raise ValueError(
            "--max-affine-condition-number must be finite and greater than 1."
        )
    if (
        not np.isfinite(args.max_uncovered_fraction)
        or not 0.0 <= args.max_uncovered_fraction <= 1.0
    ):
        raise ValueError("--max-uncovered-fraction must be between 0 and 1.")

    aligned_dir = args.aligned_dir.expanduser()
    triangulation_dir = args.triangulation_dir.expanduser()
    default_output_dir = (
        DEFAULT_BT_OUTPUT_DIR
        if args.weight_mode == "bradley-terry"
        else DEFAULT_UNIFORM_OUTPUT_DIR
    )
    default_output_name = (
        DEFAULT_BT_OUTPUT_NAME
        if args.weight_mode == "bradley-terry"
        else DEFAULT_UNIFORM_OUTPUT_NAME
    )
    base_output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else default_output_dir
    )
    face_id_column = args.face_id_column.strip()
    participant_id_column = args.participant_id_column.strip()
    score_column = args.score_column.strip()
    weight_column = (
        args.weight_column.strip() if args.weight_column is not None else None
    )
    if not face_id_column:
        raise ValueError("--face-id-column cannot be empty.")
    if not participant_id_column:
        raise ValueError("--participant-id-column cannot be empty.")
    if args.weight_mode == "bradley-terry" and weight_column is None:
        if not score_column:
            raise ValueError("--score-column cannot be empty.")
    if weight_column == "":
        raise ValueError("--weight-column cannot be empty.")

    (
        eligible,
        common_points,
        triangles,
        selected_indices,
        topology_config,
    ) = load_pipeline_artifacts(
        aligned_dir,
        triangulation_dir,
        allow_topology_failures=args.allow_topology_failures,
    )

    eligible_face_ids = eligible["face_id"].astype(str).tolist()
    if args.weight_mode == "bradley-terry":
        if args.weights_path is None:
            raise ValueError(
                "--weights-path is required when --weight-mode=bradley-terry."
            )
        selected_face_ids, weights, weight_table, weight_metadata = (
            load_bradley_terry_weights(
                eligible_face_ids,
                weights_path=args.weights_path.expanduser(),
                face_id_column=face_id_column,
                participant_id_column=participant_id_column,
                score_column=score_column,
                weight_column=weight_column,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        )
        eligible = eligible.set_index("face_id", drop=False).loc[
            selected_face_ids
        ].reset_index(drop=True)
        participant_id = str(weight_metadata["participant_id"])
        participant_path_name = safe_filename(participant_id)
        output_dir = base_output_dir / participant_path_name
        default_output_name = (
            f"bradley_terry_morph_{participant_path_name}.png"
        )
    else:
        if args.weights_path is not None:
            raise ValueError(
                "--weights-path is incompatible with --weight-mode=uniform."
            )
        if args.weight_column is not None:
            raise ValueError(
                "--weight-column is incompatible with --weight-mode=uniform."
            )
        if args.top_k is not None:
            raise ValueError("--top-k is incompatible with --weight-mode=uniform.")
        selected_face_ids = eligible_face_ids
        weights = np.full(
            len(selected_face_ids),
            1.0 / len(selected_face_ids),
            dtype=np.float64,
        )
        weight_table = pd.DataFrame(
            {
                face_id_column: selected_face_ids,
                "weight_input_value": weights,
                "rank_by_input_value": np.ones(
                    len(selected_face_ids), dtype=np.int32
                ),
                "weight": weights,
                "included_in_morph": True,
            }
        )
        weight_metadata = {
            "weights_path": None,
            "participant_id": None,
            "participant_id_column": None,
            "face_id_column": face_id_column,
            "score_column": None,
            "weight_column": None,
            "weight_conversion": "uniform",
            "temperature": None,
            "top_k": None,
            "n_eligible_before_top_k": len(selected_face_ids),
            "n_used_after_top_k": len(selected_face_ids),
            "extra_face_ids_ignored": [],
        }
        participant_id = None
        output_dir = base_output_dir

    output_name = (
        args.output_name.strip()
        if args.output_name is not None
        else default_output_name
    )
    if Path(output_name).name != output_name or not output_name.lower().endswith(
        ".png"
    ):
        raise ValueError("--output-name must be a plain .png filename.")

    n_detected = int(topology_config["n_detected_landmarks"])
    n_selected = int(topology_config["n_selected_landmarks"])
    n_anchors = int(topology_config["n_boundary_anchors"])
    width = int(topology_config["canvas_width"])
    height = int(topology_config["canvas_height"])
    anchors = common_points[n_selected:]
    if anchors.shape != (n_anchors, 2):
        raise ValueError(f"Invalid boundary-anchor shape: {anchors.shape}")

    loaded = load_reconstruction_inputs(
        eligible,
        aligned_dir=aligned_dir,
        selected_indices=selected_indices,
        n_detected=n_detected,
        n_selected=n_selected,
        width=width,
        height=height,
    )
    n_faces = len(loaded)
    if weights.shape != (n_faces,):
        raise ValueError(
            f"Weight vector has shape {weights.shape}; expected {(n_faces,)}."
        )
    landmark_stack = np.stack(
        [item[4] for item in loaded],
        axis=0,
    )

    target_landmarks, target_points = build_target_points(
        landmark_stack,
        weights,
        anchors,
    )
    if target_points.shape != common_points.shape:
        raise ValueError(
            f"Target-point shape {target_points.shape} does not match common "
            f"topology shape {common_points.shape}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(output_name).stem
    reconstruction_path = output_dir / output_name
    target_landmarks_path = output_dir / f"{stem}_selected_landmarks.npy"
    target_points_path = output_dir / f"{stem}_points_with_anchors.npy"
    weights_path = output_dir / f"{stem}_weights.csv"
    triangle_qc_path = output_dir / f"{stem}_triangle_qc.csv"
    config_path = output_dir / f"{stem}_config.json"
    warp_dir = output_dir / f"{stem}_individual_warps"
    output_paths = [
        reconstruction_path,
        target_landmarks_path,
        target_points_path,
        weights_path,
        triangle_qc_path,
        config_path,
    ]
    if args.save_individual_warps:
        output_paths.extend(
            warp_dir / f"{safe_filename(item[0])}.png" for item in loaded
        )
    ensure_writable(output_paths, args.overwrite)

    interpolation = (
        cv2.INTER_LINEAR
        if args.interpolation == "linear"
        else cv2.INTER_CUBIC
    )
    border_mode = (
        cv2.BORDER_REFLECT_101
        if args.border_mode == "reflect101"
        else cv2.BORDER_CONSTANT
    )
    reconstruction_numerator = np.zeros(
        (height, width, 3), dtype=np.float64
    )
    reconstruction_denominator = np.zeros(
        (height, width), dtype=np.float64
    )
    fallback_texture = np.zeros((height, width, 3), dtype=np.float64)
    if args.save_individual_warps:
        warp_dir.mkdir(parents=True, exist_ok=True)

    weight_details = weight_table.set_index(
        face_id_column, drop=False
    )
    weight_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []
    for position, (
        face_id,
        image_path,
        landmark_path,
        image,
        landmarks,
        manifest_bad_triangles,
    ) in enumerate(loaded, start=1):
        weight = float(weights[position - 1])
        print(
            f"[{position:03d}/{n_faces:03d}] Warping {face_id} "
            f"({args.weight_mode} weight={weight:.8f})"
        )
        source_points = np.vstack([landmarks, anchors])
        warped, covered, skipped = warp_piecewise_affine(
            image,
            source_points,
            target_points,
            triangles,
            skipped_triangle_indices=manifest_bad_triangles,
            interpolation=interpolation,
            border_mode=border_mode,
            minimum_absolute_area=args.min_triangle_area,
            relative_area_tolerance=args.relative_area_tolerance,
            maximum_condition_number=args.max_affine_condition_number,
        )
        accumulate_weighted_warp(
            reconstruction_numerator,
            reconstruction_denominator,
            warped,
            covered,
            weight,
        )
        fallback_texture += weight * image.astype(np.float64)

        for item in skipped:
            skip_rows.append(
                {
                    "face_id": face_id,
                    "triangle_index": int(item["triangle_index"]),
                    "reason": str(item["reason"]),
                    "original_face_weight": weight,
                }
            )

        if args.save_individual_warps:
            saved_warp = np.clip(np.rint(warped), 0, 255).astype(np.uint8)
            warp_path = warp_dir / f"{safe_filename(face_id)}.png"
            if not cv2.imwrite(str(warp_path), saved_warp):
                raise OSError(f"OpenCV could not save individual warp: {warp_path}")

        weight_rows.append(
            {
                "face_id": face_id,
                "weight_input_value": float(
                    weight_details.loc[face_id, "weight_input_value"]
                ),
                "weight": weight,
                "n_manifest_unstable_triangles": len(manifest_bad_triangles),
                "n_runtime_skipped_triangles": len(skipped),
                "covered_pixel_fraction": float(covered.mean()),
                "aligned_image_path": image_path.as_posix(),
                "aligned_landmark_path": landmark_path.as_posix(),
            }
        )

    (
    reconstruction,
        covered_globally,
        uncovered_fraction,
    ) = finalize_weighted_reconstruction(
        reconstruction_numerator,
        reconstruction_denominator,
        fallback_texture,
        max_uncovered_fraction=args.max_uncovered_fraction,
    )

    result = np.clip(np.rint(reconstruction), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(reconstruction_path), result):
        raise OSError(
            f"OpenCV could not save reconstruction: {reconstruction_path}"
        )
    np.save(
        target_landmarks_path,
        target_landmarks.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        target_points_path,
        target_points.astype(np.float32),
        allow_pickle=False,
    )
    used_paths = pd.DataFrame(weight_rows).drop(
        columns=["weight_input_value", "weight"]
    )
    weight_diagnostics = weight_table[
        [
            face_id_column,
            "weight_input_value",
            "rank_by_input_value",
            "included_in_morph",
            "weight",
        ]
    ].rename(columns={face_id_column: "face_id"})
    weight_diagnostics = weight_diagnostics.merge(
        used_paths,
        on="face_id",
        how="left",
        validate="one_to_one",
    )
    weight_diagnostics.to_csv(weights_path, index=False)

    triangle_qc = pd.DataFrame(
        skip_rows,
        columns=[
            "face_id",
            "triangle_index",
            "reason",
            "original_face_weight",
        ],
    )
    triangle_qc.to_csv(triangle_qc_path, index=False)

    positive_weights = weights[weights > 0.0]
    entropy = float(
        -np.sum(positive_weights * np.log(positive_weights), dtype=np.float64)
    )
    effective_n_entropy = float(np.exp(entropy))
    effective_n_kish = float(
        1.0 / np.sum(np.square(weights), dtype=np.float64)
    )
    weight_mode_detail = str(weight_metadata["weight_conversion"])
    config = {
        "algorithm": "common_delaunay_piecewise_affine_weighted_average",
        "weight_mode": args.weight_mode,
        "weight_mode_detail": weight_mode_detail,
        "participant_id": participant_id,
        "output_directory": output_dir.as_posix(),
        "uniform_weight": (
            float(1.0 / n_faces) if args.weight_mode == "uniform" else None
        ),
        "bradley_terry": weight_metadata,
        "n_faces": n_faces,
        "weight_sum": float(weights.sum(dtype=np.float64)),
        "minimum_weight": float(weights.min()),
        "maximum_weight": float(weights.max()),
        "weight_entropy_nats": entropy,
        "effective_number_of_faces_entropy": effective_n_entropy,
        "effective_number_of_faces_kish": effective_n_kish,
        "canvas_width": width,
        "canvas_height": height,
        "n_detected_landmarks_per_face": n_detected,
        "n_selected_landmarks_per_face": n_selected,
        "n_excluded_landmarks_per_face": n_detected - n_selected,
        "n_boundary_anchors": n_anchors,
        "n_topology_points": len(common_points),
        "n_triangles": len(triangles),
        "selected_landmark_indices_file": str(
            triangulation_dir / "selected_landmark_indices.npy"
        ),
        "topology_sha256": topology_config["topology_sha256"],
        "interpolation": args.interpolation,
        "border_mode": args.border_mode,
        "allow_topology_failures": bool(args.allow_topology_failures),
        "triangle_failure_policy": (
            "skip_per_face_and_locally_renormalize_weights"
        ),
        "runtime_triangle_checks": {
            "minimum_absolute_area_px2": float(args.min_triangle_area),
            "relative_area_tolerance": float(args.relative_area_tolerance),
            "maximum_affine_condition_number": float(
                args.max_affine_condition_number
            ),
        },
        "n_skipped_face_triangles": len(skip_rows),
        "n_faces_with_skipped_triangles": len(
            {row["face_id"] for row in skip_rows}
        ),
        "globally_uncovered_pixel_fraction": uncovered_fraction,
        "maximum_allowed_uncovered_pixel_fraction": float(
            args.max_uncovered_fraction
        ),
        "globally_uncovered_pixel_fallback": (
            "weighted_mean_of_unwarped_aligned_images"
        ),
        "minimum_local_weight_sum": float(
            reconstruction_denominator[covered_globally].min()
        ),
        "fraction_pixels_with_local_weight_renormalization": float(
            np.mean(
                covered_globally
                & (reconstruction_denominator < 1.0 - 1e-8)
            )
        ),
        "rounding": "numpy_rint_then_clip_uint8",
        "target_shape": "weighted_mean_of_selected_aligned_landmarks",
        "texture_average": "weighted_mean_of_piecewise_affine_warps",
    }
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\n{args.weight_mode} Delaunay morph completed")
    print(f"Faces used: {n_faces}")
    print(f"Weight conversion: {weight_mode_detail}")
    print(f"Weight range: {weights.min():.8f} to {weights.max():.8f}")
    print(f"Effective N (entropy): {effective_n_entropy:.3f}")
    print(f"Effective N (Kish): {effective_n_kish:.3f}")
    print(
        f"Landmarks: {n_detected} loaded -> {n_selected} selected -> "
        f"{n_detected - n_selected} excluded"
    )
    print(f"Topology points: {len(common_points)} ({n_selected} + {n_anchors})")
    print(f"Triangles: {len(triangles)}")
    print(f"Skipped face-triangles: {len(skip_rows)}")
    print(f"Globally uncovered pixels: {uncovered_fraction:.8f}")
    print(f"Reconstruction: {reconstruction_path}")
    print(f"Weights: {weights_path}")
    print(f"Triangle QC: {triangle_qc_path}")
    print(f"Target shape: {target_landmarks_path}")
    print(f"Configuration: {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
