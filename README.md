# Face Reconstruction

A research framework for reconstructing participant-specific facial prototypes from pairwise face judgments.

The project combines experimental design, Bradley–Terry preference modeling, facial landmark extraction, geometric alignment, Delaunay-based morphing, and participant-specific reconstruction in a reproducible pipeline.

The repository now provides two user-facing applications:

- a **researcher dashboard** for configuring experiments and analysing participant responses;
- a **participant application** for collecting pairwise face choices.

Reusable scientific methods are implemented independently in `src/face_reconstruction/` so that the applications, command-line tools, and tests all rely on the same core implementation.

---

## Overview

The project addresses the following question:

> Given a participant's repeated choices between pairs of faces, what facial prototype best represents their latent preference structure?

The main analysis pipeline is:

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

The experimental workflow is managed through the researcher and participant applications:

```text
Full CFD manifest
        ↓
Researcher app — Stimuli
        ↓
Active stimulus pool
        ↓
Researcher app — Pair design
        ↓
Participant experiment
        ↓
Pairwise participant responses
        ↓
Researcher app — Bradley–Terry analysis
        ↓
Reconstruction weights
        ↓
Facial reconstruction
```

---

## Applications

### Researcher application

The researcher dashboard is implemented with Streamlit:

```text
apps/researcher/
```

Run it from the repository root with:

```bash
streamlit run apps/researcher/app.py
```

The dashboard is intended to become the main operational interface for the pipeline, so routine experimental use does not require manually running the numbered scripts.

Current pages include:

```text
01 — Stimuli
02 — Pairs
03 — Experiment
04 — Bradley–Terry
```

### 01 — Stimuli

The Stimuli page starts from the full harmonised Chicago Face Database manifest:

```text
data/interim/cfd_audit/cfd_manifest_harmonised.csv
```

Researchers can configure:

- self-reported gender;
- self-reported ethnicity;
- perceived ethnicity;
- selection mode (`self`, `perceived`, or `both`);
- minimum perceived-ethnicity response proportion;
- minimum perceived-gender response proportion.

After applying the eligibility criteria, the resulting candidate pool can be further subsampled.

For example:

```text
Eligible faces: 275

Final experimental pool:
Black-perceived    40
White-perceived    10

Total              50
```

The subsampling stage supports reproducible random selection using an explicit seed.

The active experimental pool is saved to:

```text
data/processed/stimuli.csv
```

This file becomes the input for the Pair Design page.

---

### 02 — Pair Design

The Pairs page reads the active stimulus pool from:

```text
data/processed/stimuli.csv
```

The researcher can configure the number of comparisons and choose among multiple pair-generation strategies.

#### Balanced design

Attempts to distribute face appearances approximately evenly across comparisons.

#### Random design

Samples unique pairs without explicitly balancing individual exposure.

#### Stratified design

Controls the proportion of pair types according to a stimulus variable such as:

```text
ethnicity_perceived
ethnicity_self
gender_self
```

For example, with Black- and White-perceived faces:

```text
Black × Black      20%
Black × White      60%
White × White      20%
```

Within these constraints, the generator attempts to balance individual face exposure while avoiding duplicate pairs.

The active design is saved to:

```text
data/processed/pairs.csv
```

A pair design must be regenerated whenever the active stimulus pool changes.

---

### 03 — Participant Experiment

The Experiment page combines:

```text
data/processed/stimuli.csv
data/processed/pairs.csv
```

and builds the participant-facing experiment.

It validates that every face referenced by the pair design is still present in the active stimulus pool.

The researcher can configure participant-task settings such as:

- trial randomization;
- optional response deadlines.

The generated participant experiment consists of:

```text
apps/participant/experiment.json
apps/participant/stimuli/
```

These files are generated locally and are not committed to the repository.

---

### 04 — Bradley–Terry Analysis

The Bradley–Terry page allows researchers to upload participant-response JSON files and estimate participant-specific latent face preferences.

Expected response fields include:

```text
participant_id
trial_index
pair_id
model_left
model_right
selected_model
response_time_ms
```

For each participant, the application estimates one latent utility parameter per face:

\[
\theta_i
\]

with pairwise choice probability

\[
P(i \succ j)
=
\frac{\exp(\theta_i)}
{\exp(\theta_i)+\exp(\theta_j)}.
\]

The model uses L2 regularization for stability.

The resulting latent scores are centred, ranked, and converted into reconstruction weights through a temperature-controlled softmax:

\[
w_i
=
\frac{\exp(\theta_i/T)}
{\sum_j \exp(\theta_j/T)}.
\]

Typical outputs include:

```text
participant_id
face_id
theta
rank
standard_error
weight
```

The dashboard displays latent preference scores and reconstruction weights and allows the results to be downloaded or saved for later reconstruction.

---

## Participant application

The participant-facing application is located in:

```text
apps/participant/
```

It is intentionally lightweight and implemented using:

```text
HTML
CSS
JavaScript
```

Its responsibility is limited to experimental presentation and response collection.

The participant application:

1. loads the generated `experiment.json`;
2. presents one face pair at a time;
3. records the participant's selected face;
4. records response time and trial metadata;
5. exports or submits the resulting response data.

The participant application does **not** perform:

- stimulus selection;
- pair generation;
- Bradley–Terry fitting;
- landmark extraction;
- geometric processing;
- facial reconstruction.

Those operations belong to the researcher application and the reusable Python package.

### Response format

Each trial produces a response record such as:

```json
{
  "participant_id": "P001",
  "trial_index": 12,
  "pair_id": "PAIR_00012",
  "model_left": "BM-001",
  "model_right": "WM-012",
  "selected_model": "BM-001",
  "response_time_ms": 1342
}
```

These responses are directly compatible with the Bradley–Terry analysis.

The intended production flow is:

```text
Participant task
        ↓
JATOS
        ↓
Participant response JSON
        ↓
Researcher dashboard
        ↓
Bradley–Terry analysis
```

---

## Scientific architecture

Reusable scientific code lives in:

```text
src/face_reconstruction/
```

The applications call these modules rather than reimplementing the underlying methods.

```text
apps/researcher/
        │
        ▼
src/face_reconstruction/
        │
        ├── data/
        ├── experiment/
        ├── models/
        ├── geometry/
        ├── reconstruction/
        ├── diagnostics/
        ├── validation/
        └── utils/
        │
        ▼
apps/participant/
```

Main package responsibilities:

- `data/`: CFD metadata handling and stimulus construction;
- `experiment/`: pair generation and experimental-design utilities;
- `models/`: Bradley–Terry estimation and reconstruction weighting;
- `geometry/`: landmark extraction, alignment, triangulation, and warping;
- `reconstruction/`: weighted facial morphing;
- `diagnostics/`: experiment and model diagnostics;
- `validation/`: validation of experimental inputs and outputs;
- `utils/`: shared utilities and I/O.

---

## Repository structure

```text
Face_Reconstruction/
├── apps/
│   ├── researcher/
│   │   ├── app.py
│   │   ├── pages/
│   │   │   ├── 01_stimuli.py
│   │   │   ├── 02_pairs.py
│   │   │   ├── 03_experiment.py
│   │   │   └── 04_bradley_terry.py
│   │   └── README.md
│   │
│   └── participant/
│       ├── index.html
│       ├── script.js
│       ├── style.css
│       └── README.md
│
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
│
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
│
├── data/
├── tests/
├── .github/
│   └── workflows/
├── pyproject.toml
└── README.md
```

The `scripts/` directory is retained as a command-line interface for development, debugging, batch execution, and reproducibility.

The long-term goal is that normal experimental use can be completed entirely through the researcher and participant applications.

---

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

Install the researcher application dependencies if needed:

```bash
python -m pip install -e ".[app]"
```

The project supports Python 3.10–3.13.

---

## Running the researcher dashboard

From the repository root:

```bash
streamlit run apps/researcher/app.py
```

The current recommended workflow is:

```text
01 Stimuli
→ save active stimulus set

02 Pairs
→ generate and save active pair design

03 Experiment
→ build participant experiment

Participant task
→ collect response JSON

04 Bradley–Terry
→ upload responses and estimate reconstruction weights
```

---

## Facial reconstruction pipeline

The downstream geometry pipeline uses MediaPipe Face Landmarker to extract 478 facial landmarks from each selected face.

Valid faces are then aligned using similarity transforms.

A stable subset of landmarks is selected and a common Delaunay triangulation is constructed on the mean aligned face shape.

Participant-specific reconstruction combines:

1. a weighted target facial shape;
2. piecewise-affine warping of each source face toward that target;
3. weighted texture averaging using participant-specific Bradley–Terry weights.

The final output is a reconstructed facial prototype corresponding to the participant's estimated latent preference structure.

---

## Scripts

The numbered scripts remain available for reproducibility, development, debugging, and batch execution.

They correspond to the original modular pipeline:

```text
01 — Audit CFD metadata
02 — Build stimulus set
03 — Generate pair comparisons
04 — Simulate participant responses
05 — Fit Bradley–Terry model
06 — Extract facial landmarks
07 — Visualise landmark quality
08 — Align faces
09 — Build common triangulation
10 — Reconstruct facial prototype
```

They should increasingly be treated as command-line wrappers around reusable functionality in `src/face_reconstruction/`, rather than as the primary user interface.

---

## Data

The repository does not distribute the Chicago Face Database images or associated licensed source files.

Users must obtain the CFD separately and place the required files under:

```text
data/raw/cfd/
```

The repository does not version:

- CFD source images;
- generated participant experiment images;
- participant response data;
- generated intermediate arrays;
- reconstructed images;
- large derived artifacts.

In particular:

```text
apps/participant/experiment.json
apps/participant/stimuli/
```

are generated locally and should remain excluded from Git.

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
    ├── bradley_terry/
    └── morphing/
```

---

## Outputs

### Experimental design

Typical outputs include:

- active stimulus pool;
- reproducible pairwise comparison design;
- group-stratified pair allocations;
- participant experiment configuration.

### Bradley–Terry analysis

Typical outputs include:

- latent utility estimates;
- uncertainty estimates;
- face rankings;
- normalized reconstruction weights.

### Geometry

Typical outputs include:

- MediaPipe landmark arrays;
- alignment outputs;
- landmark QC diagnostics;
- common Delaunay topology;
- topology validation tables.

### Reconstruction

Typical outputs include:

- participant-specific reconstructed face;
- weighted target landmark configuration;
- reconstruction weights;
- triangle-level QC diagnostics;
- reconstruction configuration metadata.

---

## Reproducibility

The codebase separates:

```text
user interfaces
scientific logic
generated data
```

The scientific implementation lives in `src/face_reconstruction/` and is shared across:

- the researcher application;
- command-line scripts;
- tests;
- future batch or deployment interfaces.

Generated artifacts under `data/interim/`, `data/processed/`, `data/simulated/`, `data/results/`, and the generated participant experiment package are intended to be reproducible from source data and explicit experiment configuration rather than maintained as version-controlled source files.

---

## Development

Run the test suite:

```bash
python -m pytest
```

Run Ruff:

```bash
python -m ruff check src scripts apps/researcher tests
```

GitHub Actions runs automated checks on pushes and pull requests.