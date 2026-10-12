# Face Reconstruction

A research pipeline for reconstructing participant-specific facial prototypes from pairwise preference judgments.

The project estimates latent preferences over a standardized set of faces using a Bradley–Terry model and transforms these estimates into reconstruction weights. Facial geometry is extracted with MediaPipe Face Landmarker, aligned to a common coordinate system, and reconstructed using a shared Delaunay triangulation with piecewise-affine warping.

## Overview

The pipeline is designed to answer the following question:

**Given a participant's repeated choices between pairs of faces, what facial prototype best represents their latent preference structure?**

The full pipeline is:

```text
Pairwise face judgments
        ↓
Bradley–Terry latent utilities
        ↓
Participant-specific reconstruction weights
        ↓
MediaPipe facial landmarks
        ↓
Similarity alignment
        ↓
Stable landmark selection
        ↓
Common Delaunay triangulation
        ↓
Piecewise-affine warping
        ↓
Weighted facial prototype
```

The implementation separates experimental design, statistical modeling, facial geometry, reconstruction, diagnostics, and validation into reusable Python modules.



## Method

The reconstruction pipeline is organized as a sequence of reproducible steps.

### 1. Audit the Chicago Face Database metadata

`scripts/01_audit_cfd_manifest.py`

Validates the available CFD metadata and image inventory before stimulus selection.

### 2. Build the stimulus set

`scripts/02_build_stimuli.py`

Constructs the standardized candidate face set used throughout the experiment and reconstruction pipeline.

### 3. Generate pairwise comparisons

`scripts/03_generate_pairs.py`

Creates balanced face pairs for participant judgments while controlling stimulus exposure across comparisons.

### 4. Simulate participant responses

`scripts/04_simulate_participants.py`

Generates synthetic pairwise-choice data for testing and validating the statistical pipeline before collecting real participant responses.

### 5. Estimate Bradley–Terry utilities

`scripts/05_fit_bradley_terry.py`

Fits a participant-specific Bradley–Terry model to pairwise choices and estimates one latent utility score per candidate face.

These utilities are converted into reconstruction weights used in the final morph.

### 6. Extract facial landmarks

`scripts/06_extract_landmarks.py`

Uses MediaPipe Face Landmarker to extract 478 facial landmarks from each stimulus image and performs landmark-level quality control.

### 7. Visualize landmark quality

`scripts/07_visualise_landmarks.py`

Produces diagnostic overlays for manually inspecting landmark placement and identifying failed detections.

### 8. Align faces

`scripts/08_align_faces.py`

Aligns faces to a shared coordinate system using eye-based similarity transforms so that geometry and texture can be meaningfully compared across stimuli.

### 9. Build a common Delaunay topology

`scripts/09_build_common_triangulation.py`

Selects a stable subset of facial landmarks, adds boundary anchors, and computes a common Delaunay triangulation on the mean aligned facial shape.

The same triangle topology is then validated across all aligned faces.

### 10. Reconstruct the participant-specific facial prototype

`scripts/10_morph_delaunay.py`

Builds a weighted target facial shape, piecewise-affine warps each aligned face to that target geometry, and combines the warped textures using participant-specific reconstruction weights.

The final output is a facial prototype representing the participant's latent preference structure.




## Repository Structure

```text
Face_Reconstruction/
├── scripts/
│   ├── 01_audit_cfd_manifest.py
│   ├── 02_build_stimuli.py
│   ├── 03_generate_pairs.py
│   ├── 04_simulate_participants.py
│   ├── 05_fit_bradley_terry.py
│   ├── 06_extract_landmarks.py
│   ├── 07_visualise_landmarks.py
│   ├── 08_align_faces.py
│   ├── 09_build_common_triangulation.py
│   └── 10_morph_delaunay.py
├── src/
│   └── face_reconstruction/
│       ├── data/
│       ├── diagnostics/
│       ├── experiment/
│       ├── geometry/
│       ├── models/
│       ├── reconstruction/
│       ├── utils/
│       ├── validation/
│       ├── pipeline.py
│       └── schemas.py
├── tests/
├── .github/
│   └── workflows/
├── pyproject.toml
└── README.md
```

The repository follows a `src/` layout. Reusable scientific logic lives in the `face_reconstruction` package, while `scripts/` contains the command-line entry points for each stage of the pipeline.

The main package responsibilities are:

- `data/`: CFD metadata handling and stimulus construction.
- `experiment/`: pair generation and response simulation.
- `models/`: Bradley–Terry estimation.
- `geometry/`: landmark validation, face alignment, triangulation, and piecewise-affine warping.
- `reconstruction/`: weighting and participant-specific facial morphing.
- `diagnostics/`: experimental and model diagnostics.
- `validation/`: validation utilities for experimental data.
- `utils/`: shared I/O and path utilities.

## Installation

Clone the repository:

```bash
git clone https://github.com/Fantiflex/Face-reconstruction.git
cd Face-reconstruction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the package in editable mode with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

The project currently supports Python 3.10–3.13.

Verify the installation with:

```bash
python -m pytest
python -m ruff check src scripts tests
```



## Quick Start

The scripts are designed to be run from the repository root and follow the numerical order of the pipeline.

### 1. Prepare the CFD stimulus set

The Chicago Face Database files are not distributed with this repository. Place the required CFD metadata and images under the expected `data/raw/` structure before running the preprocessing steps.

```bash
python scripts/01_audit_cfd_manifest.py
python scripts/02_build_stimuli.py
python scripts/03_generate_pairs.py
```

These steps audit the source data, construct the candidate stimulus set, and generate the pairwise-comparison design.

### 2. Validate the statistical pipeline with simulated participants

Before using experimental responses, the Bradley–Terry estimation pipeline can be tested on synthetic data:

```bash
python scripts/04_simulate_participants.py
python scripts/05_fit_bradley_terry.py
```

The simulation step generates known latent utilities and synthetic choices, allowing estimated Bradley–Terry scores to be compared against ground truth.

For real experiments, simulated responses are replaced by the participant response table collected during the pairwise-choice task.

### 3. Build the facial geometry

Extract and inspect MediaPipe landmarks:

```bash
python scripts/06_extract_landmarks.py
python scripts/07_visualise_landmarks.py
```

Align the valid faces to a common coordinate system:

```bash
python scripts/08_align_faces.py
```

Build and validate the common Delaunay topology:

```bash
python scripts/09_build_common_triangulation.py
```

### 4. Reconstruct a facial prototype

A uniform-weight reconstruction can be used as a baseline:

```bash
python scripts/10_morph_delaunay.py \
  --weight-mode uniform
```

For a participant-specific reconstruction, provide Bradley–Terry estimates:

```bash
python scripts/10_morph_delaunay.py \
  --weight-mode bradley-terry \
  --weights-path path/to/bradley_terry_scores.csv
```

The reconstruction combines two participant-specific components:

1. a **weighted facial shape**, obtained from the aligned landmark coordinates;
2. a **weighted facial texture**, obtained after piecewise-affine warping each source face to that target shape.

### Development checks

Before committing changes, run:

```bash
python -m pytest
python -m ruff check src scripts tests
```

The same checks are executed automatically by GitHub Actions on pushes and pull requests.



## Data

The repository does not distribute the Chicago Face Database (CFD) images or associated licensed source files.

Users must obtain the CFD separately and place the required source files under the local `data/raw/` directory before running the preprocessing pipeline.

The repository also does not version:

- raw participant response data;
- generated intermediate arrays and manifests;
- reconstructed images;
- downloaded MediaPipe model files;
- other large or derived artifacts.

These files are excluded from version control and are regenerated locally by the pipeline.

A typical local data layout is:

```text
data/
├── raw/
│   └── cfd/
├── interim/
│   ├── cfd_audit/
│   ├── aligned_faces/
│   └── common_triangulation/
├── processed/
│   ├── stimuli.csv
│   └── pairs.csv
├── simulated/
│   ├── true_latent_scores.csv
│   └── simulated_responses.csv
└── results/
    └── morphing/
```

Exact contents may vary depending on whether the pipeline is run with simulated or experimental participant data.

## Outputs

### Stimulus and experimental-design outputs

The preprocessing and experiment-design stages produce:

- a standardized stimulus table;
- balanced pairwise comparisons;
- exposure and design diagnostics;
- optional simulated participant responses.

### Bradley–Terry outputs

Participant responses are converted into participant-specific latent utilities.

Typical outputs include:

- one latent utility estimate per face;
- uncertainty estimates;
- face rankings;
- reconstruction weights;
- diagnostics for comparison-graph connectivity and score recovery.

### Geometry outputs

The facial-geometry stages produce:

- MediaPipe facial landmark arrays;
- landmark QC diagnostics and visualizations;
- aligned facial images and landmark coordinates;
- a common Delaunay topology;
- topology-QC manifests.

### Reconstruction outputs

The final reconstruction stage produces:

- the participant-specific reconstructed face;
- the weighted target landmark configuration;
- the target topology points including boundary anchors;
- the weights used for the reconstruction;
- triangle-level QC diagnostics;
- a JSON configuration describing the reconstruction run.

Optional individual warped faces can also be saved for quality control.

## Reproducibility

Derived files under `data/interim/`, `data/processed/`, `data/simulated/`, and `data/results/` are intended to be reproducible from the source data and pipeline configuration rather than maintained as source-controlled artifacts.

The codebase separates reusable scientific logic from executable scripts so that individual stages can be tested independently while preserving a reproducible end-to-end workflow.