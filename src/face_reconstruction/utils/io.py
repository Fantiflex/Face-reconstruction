from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_bool(series: pd.Series) -> pd.Series:
    """Convert common boolean encodings to a boolean Series."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).eq(1)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "oui"})
    )


def resolve_path(
    raw: str,
    reference_file: Path,
) -> Path:
    """Resolve an absolute or relative artifact path."""

    path = Path(raw).expanduser()

    if path.is_absolute():
        return path

    candidates = (
        Path.cwd() / path,
        reference_file.resolve().parent / path,
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def sha256_arrays(
    points: np.ndarray,
    triangles: np.ndarray,
) -> str:
    """Return a deterministic hash for triangulation arrays."""

    digest = hashlib.sha256()

    digest.update(
        np.asarray(
            points,
            dtype="<f4",
        ).tobytes(order="C")
    )

    digest.update(
        np.asarray(
            triangles,
            dtype="<i4",
        ).tobytes(order="C")
    )

    return digest.hexdigest()


def load_alignment_artifacts(
    aligned_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Load and validate step-08 alignment artifacts."""

    aligned_dir = Path(aligned_dir)

    manifest_path = aligned_dir / "alignment_manifest.csv"
    mean_path = aligned_dir / "mean_aligned_landmarks.npy"
    config_path = aligned_dir / "alignment_config.json"

    for path in (
        manifest_path,
        mean_path,
        config_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required alignment artifact not found: {path}"
            )

    manifest = pd.read_csv(manifest_path)

    required_columns = {
        "face_id",
        "aligned_image_path",
        "aligned_landmark_path",
        "alignment_qc_pass",
    }

    missing = required_columns.difference(
        manifest.columns
    )

    if missing:
        raise ValueError(
            "Alignment manifest is missing columns: "
            f"{sorted(missing)}"
        )

    manifest["face_id"] = (
        manifest["face_id"]
        .astype(str)
        .str.strip()
    )

    manifest["alignment_qc_pass"] = parse_bool(
        manifest["alignment_qc_pass"]
    )

    manifest = (
        manifest.loc[
            manifest["alignment_qc_pass"]
        ]
        .sort_values(
            "face_id",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if manifest.empty:
        raise ValueError(
            "No alignment passed QC."
        )

    if manifest["face_id"].duplicated().any():
        raise ValueError(
            "Alignment manifest contains duplicate face_id values."
        )

    mean_landmarks = np.asarray(
        np.load(
            mean_path,
            allow_pickle=False,
        ),
        dtype=np.float64,
    )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    return (
        manifest,
        mean_landmarks,
        config,
    )