from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


from face_reconstruction.geometry.triangulation import (
    EXCLUDED_LANDMARK_INDICES,
    INNER_LIP_INDICES,
    IRIS_INDICES,
    boundary_points,
    build_selected_landmark_indices,
    validate_face_topology,
    build_delaunay_triangulation,
    signed_double_areas,
)


DEFAULT_ALIGNED_DIR = Path("data/interim/aligned_faces")
DEFAULT_OUTPUT_DIR = Path("data/interim/common_triangulation")
EXPECTED_LANDMARK_COUNT = 478





def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one Delaunay triangulation on the mean aligned MediaPipe "
            "shape and validate the same topology on every aligned face."
        )
    )
    parser.add_argument(
        "--aligned-dir", type=Path, default=DEFAULT_ALIGNED_DIR,
        help=f"Directory produced by step 08. Default: {DEFAULT_ALIGNED_DIR}",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Triangulation output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--expected-landmarks", type=int, default=EXPECTED_LANDMARK_COUNT,
        help=f"Expected MediaPipe landmark count. Default: {EXPECTED_LANDMARK_COUNT}.",
    )
    parser.add_argument(
        "--boundary-margin", type=float, default=0.0,
        help="Inset in pixels for the eight canvas anchors. Default: 0.",
    )
    parser.add_argument(
        "--min-mean-triangle-area", type=float, default=0.05,
        help="Reject mean-shape triangles below this area in px². Default: 0.05.",
    )
    parser.add_argument(
        "--min-face-triangle-area", type=float, default=0.01,
        help="Per-face degeneracy threshold in px². Default: 0.01.",
    )
    parser.add_argument(
        "--max-flipped-fraction", type=float, default=0.0,
        help="Maximum fraction of triangles allowed to flip. Default: 0.",
    )
    parser.add_argument(
        "--max-degenerate-fraction", type=float, default=0.0,
        help="Maximum fraction of degenerate triangles. Default: 0.",
    )
    parser.add_argument(
        "--line-thickness", type=int, default=1,
        help="Thickness of the QC triangulation overlay. Default: 1.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing triangulation artifacts.",
    )
    parser.add_argument(
        "--fail-on-qc-error", action="store_true",
        help="Return status 1 if any aligned face fails topology QC.",
    )
    parser.add_argument(
    "--relative-area-tolerance",
    type=float,
    default=1e-4,
    help="Relative degeneracy tolerance based on squared edge length.",
    )
    parser.add_argument(
        "--min-triangle-quality",
        type=float,
        default=1e-3,
        help="Minimum dimensionless triangle quality.",
    )
    return parser.parse_args()


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).eq(1)
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "oui"}
    )


def resolve_path(raw: str, reference: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    candidates = (Path.cwd() / path, reference.resolve().parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_alignment(aligned_dir: Path) -> tuple[pd.DataFrame, np.ndarray, int, int]:
    manifest_path = aligned_dir / "alignment_manifest.csv"
    mean_path = aligned_dir / "mean_aligned_landmarks.npy"
    config_path = aligned_dir / "alignment_config.json"
    for path in (manifest_path, mean_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"Required step-08 artifact not found: {path}")

    manifest = pd.read_csv(manifest_path)
    required = {"face_id", "aligned_image_path", "aligned_landmark_path"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Alignment manifest is missing columns: {sorted(missing)}")
    if "alignment_qc_pass" not in manifest.columns:
        raise ValueError("Alignment manifest has no alignment_qc_pass column.")
    manifest["face_id"] = manifest["face_id"].astype(str).str.strip()
    manifest["alignment_qc_pass"] = parse_bool(manifest["alignment_qc_pass"])
    manifest = manifest.loc[manifest["alignment_qc_pass"]].copy()
    manifest = manifest.sort_values("face_id", kind="stable").reset_index(drop=True)
    if manifest.empty:
        raise ValueError("No step-08 alignment passed QC.")
    if manifest["face_id"].duplicated().any():
        raise ValueError("Alignment manifest contains duplicate face_id values.")

    mean_landmarks = np.asarray(np.load(mean_path, allow_pickle=False), dtype=np.float64)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    width = int(config["output_width"])
    height = int(config["output_height"])
    return manifest, mean_landmarks, width, height











def sha256_arrays(points: np.ndarray, triangles: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(points, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray(triangles, dtype="<i4").tobytes(order="C"))
    return digest.hexdigest()





def draw_overlay(
    image: np.ndarray,
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    thickness: int,
    landmark_count: int,
) -> np.ndarray:
    overlay = image.copy()
    for triangle in triangles:
        polygon = np.rint(points[triangle]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [polygon], True, (0, 220, 0), thickness, cv2.LINE_AA)
    for x, y in points[:landmark_count]:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 1, (0, 80, 255), -1)
    return overlay


def ensure_writable(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Triangulation outputs already exist. Use --overwrite: " + ", ".join(existing)
        )


def main() -> int:
    args = parse_arguments()
    if args.expected_landmarks <= 0:
        raise ValueError("--expected-landmarks must be positive.")
    if args.min_mean_triangle_area < 0 or args.min_face_triangle_area < 0:
        raise ValueError("Triangle-area thresholds cannot be negative.")
    if not 0 <= args.max_flipped_fraction <= 1:
        raise ValueError("--max-flipped-fraction must be between 0 and 1.")
    if not 0 <= args.max_degenerate_fraction <= 1:
        raise ValueError("--max-degenerate-fraction must be between 0 and 1.")
    if args.line_thickness <= 0:
        raise ValueError("--line-thickness must be positive.")

    aligned_dir = args.aligned_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    manifest, mean_landmarks, width, height = load_alignment(aligned_dir)
    expected_shape = (args.expected_landmarks, 2)
    if mean_landmarks.shape != expected_shape:
        raise ValueError(f"Expected mean shape {expected_shape}, got {mean_landmarks.shape}.")
    if not np.isfinite(mean_landmarks).all():
        raise ValueError("Mean landmarks contain non-finite coordinates.")

    selected_indices = build_selected_landmark_indices(args.expected_landmarks)
    excluded_indices = np.asarray(
        sorted(EXCLUDED_LANDMARK_INDICES), dtype=np.int32
    )
    selected_mean_landmarks = mean_landmarks[selected_indices]
    anchors = boundary_points(width, height, args.boundary_margin)
    common_points = np.vstack([selected_mean_landmarks, anchors])
    triangles = build_delaunay_triangulation(
        common_points,
        min_triangle_area=args.min_mean_triangle_area,
    )

    mean_areas = (
        signed_double_areas(
            common_points,
            triangles,
        )
        / 2.0
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / "common_points.npy"
    triangles_path = output_dir / "common_triangles.npy"
    selected_indices_path = output_dir / "selected_landmark_indices.npy"
    excluded_indices_path = output_dir / "excluded_landmark_indices.npy"
    manifest_path = output_dir / "triangulation_manifest.csv"
    config_path = output_dir / "triangulation_config.json"
    overlay_path = output_dir / "common_triangulation_overlay.png"
    ensure_writable(
        [
            points_path,
            triangles_path,
            selected_indices_path,
            excluded_indices_path,
            manifest_path,
            config_path,
            overlay_path,
        ],
        args.overwrite,
    )

    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        landmark_path = resolve_path(str(row.aligned_landmark_path), aligned_dir / "alignment_manifest.csv")
        result: dict[str, Any] = {
            "face_id": str(row.face_id),
            "aligned_landmark_path": landmark_path.as_posix(),
            "topology_qc_pass": False,
            "topology_qc_reason": "not_processed",
            "n_triangles": len(triangles),
            "n_flipped_triangles": np.nan,
            "flipped_triangle_fraction": np.nan,
            "n_degenerate_triangles": np.nan,
            "degenerate_triangle_fraction": np.nan,
            "min_absolute_triangle_area_px2": np.nan,
            "median_absolute_triangle_area_px2": np.nan,
            "error_type": "",
            "error_message": "",
        }
        try:
            landmarks = np.asarray(np.load(landmark_path, allow_pickle=False), dtype=np.float64)
            if landmarks.shape != expected_shape:
                raise ValueError(f"Expected {expected_shape}, got {landmarks.shape}.")
            if not np.isfinite(landmarks).all():
                raise ValueError("Non-finite aligned landmarks.")
            selected_landmarks = landmarks[selected_indices]
            face_points = np.vstack([selected_landmarks, anchors])
            result.update(
                validate_face_topology(
                    face_points,
                    common_points,
                    triangles,
                    selected_indices,
                    min_absolute_area=args.min_face_triangle_area,
                    relative_area_tolerance=args.relative_area_tolerance,
                    min_triangle_quality=args.min_triangle_quality,
                    max_flipped_fraction=args.max_flipped_fraction,
                    max_degenerate_fraction=args.max_degenerate_fraction,
                )
)
        except Exception as error:
            result.update(
                {
                    "topology_qc_reason": "processing_exception",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        rows.append(result)

    results = pd.DataFrame(rows)
    np.save(points_path, common_points.astype(np.float32), allow_pickle=False)
    np.save(triangles_path, triangles, allow_pickle=False)
    np.save(selected_indices_path, selected_indices, allow_pickle=False)
    np.save(excluded_indices_path, excluded_indices, allow_pickle=False)
    results.to_csv(manifest_path, index=False)

    mean_image_path = aligned_dir / "mean_aligned_face.png"
    mean_image = cv2.imread(str(mean_image_path), cv2.IMREAD_COLOR)
    if mean_image is None:
        raise ValueError(f"OpenCV could not read mean aligned face: {mean_image_path}")
    if mean_image.shape[:2] != (height, width):
        raise ValueError(
            f"Mean image shape {mean_image.shape[:2]} does not match {(height, width)}."
        )
    overlay = draw_overlay(
        mean_image,
        common_points,
        triangles,
        thickness=args.line_thickness,
        landmark_count=len(selected_indices),
    )
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"OpenCV could not save: {overlay_path}")

    config = {
        "algorithm": "scipy.spatial.Delaunay",
        "landmark_topology": "mediapipe_face_landmarker_478",
        "landmark_selection": "stable_448_without_inner_lips_or_irises",
        "triangulation_reference": "selected_mean_aligned_landmarks",
        "canvas_width": width,
        "canvas_height": height,
        "n_detected_landmarks": args.expected_landmarks,
        "n_selected_landmarks": int(len(selected_indices)),
        "n_excluded_landmarks": int(len(excluded_indices)),
        "selected_landmark_indices_file": selected_indices_path.name,
        "excluded_landmark_indices_file": excluded_indices_path.name,
        "inner_lip_indices_excluded": sorted(INNER_LIP_INDICES),
        "iris_indices_excluded": sorted(IRIS_INDICES),
        "n_boundary_anchors": len(anchors),
        "boundary_anchor_indices": list(
            range(len(selected_indices), len(common_points))
        ),
        "boundary_points": anchors.tolist(),
        "n_triangles": len(triangles),
        "min_mean_triangle_area_px2": float(mean_areas.min()),
        "median_mean_triangle_area_px2": float(np.median(mean_areas)),
        "min_face_triangle_area_threshold_px2": args.min_face_triangle_area,
        "max_flipped_fraction": args.max_flipped_fraction,
        "max_degenerate_fraction": args.max_degenerate_fraction,
        "topology_sha256": sha256_arrays(common_points, triangles),
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    passed = int(results["topology_qc_pass"].sum())
    print("\nCommon triangulation completed")
    print(
        f"Landmarks: {args.expected_landmarks} detected -> "
        f"{len(selected_indices)} selected -> {len(excluded_indices)} excluded"
    )
    print(f"Points: {len(common_points)} ({len(selected_indices)} landmarks + 8 anchors)")
    print(f"Triangles: {len(triangles)}")
    print(f"Face topology QC passed: {passed}/{len(results)}")
    print(f"QC overlay: {overlay_path}")
    print(f"Selected landmark indices: {selected_indices_path}")
    print(f"Manifest: {manifest_path}")
    failures = results.loc[~results["topology_qc_pass"]]
    if not failures.empty:
        print("\nTopology QC failures:")
        print(
            failures[
                ["face_id", "topology_qc_reason", "n_flipped_triangles",
                 "n_degenerate_triangles", "error_type", "error_message"]
            ].to_string(index=False)
        )
    return 1 if args.fail_on_qc_error and passed != len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
