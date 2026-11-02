# Researcher Application

The researcher application is the main experiment-configuration interface for the Face Reconstruction project.

It is implemented with **Streamlit** and uses the scientific functions implemented in `src/face_reconstruction/`.

The goal is to allow researchers to configure and run the experiment without manually modifying or executing the numbered scripts.

## Run the application

From the repository root:

```bash
streamlit run apps/researcher/app.py
```

The application assumes that the project has been installed in editable mode and that the required CFD source files are available locally.

## Workflow

The researcher interface follows the experimental workflow:

```text
01 — Stimuli
      ↓
02 — Pairs
      ↓
03 — Experiment
      ↓
Participant task
      ↓
Participant responses
      ↓
Bradley–Terry model
      ↓
Face reconstruction
```

Later dashboard pages will progressively expose the analysis and reconstruction stages as well.

## 01 — Stimuli

The Stimuli page defines the face pool used by the experiment.

The page starts from the full harmonised CFD manifest rather than from a previously filtered stimulus table.

Source metadata:

```text
data/interim/cfd_audit/cfd_manifest_harmonised.csv
```

The researcher can configure:

- self-reported gender;
- self-reported ethnicity;
- perceived ethnicity;
- ethnicity selection mode:
  - `self`
  - `perceived`
  - `both`
- minimum perceived-ethnicity response proportion;
- minimum perceived-gender response proportion.

After applying these eligibility criteria, the researcher can either keep all eligible faces or create a smaller experimental pool.

### Controlled pool sampling

The final pool can be subsampled according to a categorical variable such as:

- `ethnicity_perceived`;
- `ethnicity_self`;
- `gender_self`.

For example:

```text
Eligible pool:
275 faces

Final experimental pool:
Black-perceived    40
White-perceived    10

Total              50
```

A random seed controls reproducibility.

The final selected pool becomes the active stimulus set when the researcher clicks:

```text
Save as active stimulus set
```

It is saved to:

```text
data/processed/stimuli.csv
```

This file is the input used automatically by the Pairs page.

The detailed selection summary, raw table, and stimulus gallery are collapsible to keep the interface compact.

## 02 — Pairs

The Pairs page loads:

```text
data/processed/stimuli.csv
```

and therefore always operates on the active pool saved from the Stimuli page.

The researcher can choose among different pair-generation strategies.

### Balanced design

Attempts to distribute individual face appearances approximately evenly across the experiment.

### Random design

Samples unique face pairs without explicitly controlling individual or group exposure.

### Stratified design

Controls the proportion of comparisons belonging to different group combinations.

The researcher first selects a grouping variable, for example:

```text
ethnicity_perceived
```

For Black- and White-perceived faces, possible pair types are:

```text
Black × Black
Black × White
White × White
```

A design can therefore specify, for example:

```text
Black × Black      20%
Black × White      60%
White × White      20%
```

The generator then selects unique pairs while attempting to balance individual stimulus exposure within the requested constraints.

The resulting active design is saved to:

```text
data/processed/pairs.csv
```

Pair designs must be regenerated whenever the active stimulus pool changes.

## 03 — Experiment

The Experiment page combines the active stimulus pool and active pair design:

```text
data/processed/stimuli.csv
data/processed/pairs.csv
```

It validates that every face referenced by the pair design is present in the active stimulus pool.

The researcher can configure participant-task parameters such as:

- trial randomization;
- optional response timeout.

The page then builds the participant experiment package.

Generated files:

```text
apps/participant/experiment.json
apps/participant/stimuli/
```

Only the faces actually required for the active experiment are copied into the generated participant stimulus directory.

These generated files are local experiment artifacts and are excluded from version control.

## Data flow

The key rule is that each page produces the active input for the next page:

```text
01 Stimuli
   ↓
data/processed/stimuli.csv

02 Pairs
   ↓
data/processed/pairs.csv

03 Experiment
   ↓
apps/participant/experiment.json
apps/participant/stimuli/
```

If the stimulus pool is changed, the pair design must be regenerated before rebuilding the participant experiment.

## Scientific code

The Streamlit application should contain interface and orchestration logic only.

Reusable scientific methods belong under:

```text
src/face_reconstruction/
```

For example:

```text
src/face_reconstruction/data/stimuli.py
src/face_reconstruction/experiment/pairs.py
src/face_reconstruction/models/bradley_terry.py
```

This ensures that the dashboard, command-line scripts, tests, and future applications use the same scientific implementation.

## Development checks

Before committing changes:

```bash
python -m ruff check apps/researcher src tests
python -m pytest
```

## Long-term goal

The researcher dashboard is intended to become the primary operational interface for the complete pipeline:

```text
stimulus selection
→ pair design
→ experiment generation
→ participant-data import
→ Bradley–Terry estimation
→ geometry processing
→ participant-specific reconstruction
→ export
```

The scripts remain useful for development, batch processing, debugging, and reproducibility, but should not be required for routine experimental use.