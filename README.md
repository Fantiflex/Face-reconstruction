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