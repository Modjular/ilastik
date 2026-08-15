# Standalone single-file Pixel Classification — rough draft plan

## Goal

A single `.py` file (pip-installable deps only — no conda, no vigra, no fastfilters,
no PyQt, no full ilastik/lazyflow install) that:

1. Loads a **real, already-trained** ilastik Pixel Classification `.ilp` project
   (produced by the desktop app).
2. Recomputes the same feature set on a new image.
3. Runs the same trained Random Forest and produces the same probability maps.

Training is explicitly **out of scope** for v1 — this is inference-only, matching
`ilastik.experimental.api.PixelClassificationPipeline` (see
`ilastik/experimental/api/_pipelines.py`), just without needing vigra/fastfilters/
lazyflow/PyQt installed.

## What already exists (found this session)

- `ilastik/experimental/api/_pipelines.py` — the existing in-repo Python API for
  loading a `.ilp` and predicting. This is the reference for *what* the standalone
  file needs to reproduce, but it still imports `vigra` and the full lazyflow
  operator graph (`OpFeatureSelection`, `OpClassifierPredict`, etc.) — not
  pip-friendly as-is.
- `notebooks/pixel_classification_api/` — a notebook + trained `pc.ilp` +
  test image, demonstrating the same API, still via conda (`environment.yml`
  pulls `ilastik-core` from the `ilastik-forge` channel).
- **`feat/webgpu` branch** (not on this branch, not merged) — this is the prior
  work you remembered. It contains, among other WebGPU browser-port material:
  - `run_pc.py` — a headless CLI pixel-classification runner, still vigra-based
    (uses `vigra.learning.RandomForest(ilp_path, "PixelClassification/ClassifierForests")`
    and `vigra.filters.*`). Useful as a *reference implementation / oracle*,
    not as the standalone file itself.
  - `20260327_ilp_stuff/generate_minimal_ilp.py` and a JS port
    `generate_minimal_ilp.js` (from PR #1, branch
    `claude/ilastik-js-ilp-generation-wfag77`, based on `feat/webgpu`) — hand-rolled
    HDF5 writers that build a minimal valid `.ilp` from scratch, with zero
    HDF5-library dependency. Good for generating tiny test fixtures.
  - `20260327_ilp_stuff/ILP_PixelClassification_Format.md` — a written spec of the
    `.ilp` HDF5 layout (Input Data / FeatureSelections / PixelClassification
    groups, axistags JSON format, label block format).
  - `20260602_webgpu/ff.js` (`fastfilters3`) — a from-scratch JS reimplementation
    of vigra/fastfilters' Gaussian-derivative-based feature filters (Gaussian
    smoothing, LoG, Gaussian gradient magnitude, difference of Gaussians,
    structure tensor eigenvalues, Hessian of Gaussian eigenvalues), including
    vigra's exact kernel-radius formula and normalization rules, written up
    for GPU compute shaders.
  - `20260602_webgpu/rf.js` — a from-scratch JS Random Forest **trainer**
    (`FlatRandomForest`) producing a flat array structure for GPU inference.
    Useful as a reference for a flat/array-based tree representation, but it
    trains its own forest rather than loading vigra's.

  I've copied the directly reusable reference files into
  `tools/standalone_pixel_classification/reference/` on this branch:
  `ILP_PixelClassification_Format.md`, `generate_minimal_ilp.py`,
  `generate_minimal_ilp.js`, `run_pc_vigra_baseline.py`, and
  `ff_feature_math_reference.js`. These are reference-only — none are pip-only,
  and `run_pc_vigra_baseline.py`/`ff_feature_math_reference.js` still assume
  vigra or a browser GPU context.

None of this prior work is a finished pip-only single file — it's either still
vigra-dependent (Python side) or a from-scratch training reimplementation aimed
at the browser (JS side), not loading a real vigra-trained forest. But it maps
out the whole problem well.

## The two real hurdles

**1. Feature filters (vigra/fastfilters → pure NumPy/SciPy)**

Solvable, and largely de-risked by `ff_feature_math_reference.js`, which already
worked out vigra's exact kernel construction:
- dynamic kernel radius `ceil((3 + 0.5*order) * scale)`
- strict per-order normalization (sum=1 for smoothing, 1st-moment=1 for the
  1st derivative, zero-mean + variance-normalized for the 2nd derivative)

`scipy.ndimage.correlate1d`/`gaussian_filter1d` with hand-built kernels (instead
of scipy's built-in Gaussian, which truncates/normalizes differently) should be
able to reproduce this bit-for-bit or very close. The six feature types
(`GaussianSmoothing`, `LaplacianOfGaussian`, `GaussianGradientMagnitude`,
`DifferenceOfGaussians`, `StructureTensorEigenvalues`,
`HessianOfGaussianEigenvalues`) are all separable-Gaussian-derivative based, so
this is one kernel builder + a handful of compositions (structure tensor and
Hessian eigenvalues additionally need a per-pixel 2x2/3x3 eigendecomposition —
`numpy.linalg.eigh` on a stacked array, with attention to vigra's eigenvalue
ordering convention: largest first).

**2. Random Forest inference (vigra's `RandomForest.writeHDF5` format)**

This is the actual unknown and the main risk in the plan. `ClassifierForests`
in the `.ilp` is written by vigra's own `forest.writeHDF5(...)` — a binary/array
HDF5 layout defined in vigra's C++ source (`rf_classnodes.hxx`), not documented
anywhere in this repo or in the prior `feat/webgpu` work (that work only trained
*new* forests in JS; it never had to read vigra's saved ones). To do inference
without vigra installed, we need to:
- dump the raw HDF5 group structure/arrays of `PixelClassification/ClassifierForests`
  from a real trained `.ilp` (using the vigra env you can access) and reverse
  engineer the topology/threshold/leaf-probability array encoding, or
- find/confirm a public description of vigra's RF HDF5 node encoding to
  cross-check against.
Once decoded, prediction is simple: walk each tree's flat array per pixel-feature
row and average per-tree class-probability leaves — structurally similar to the
node encoding `rf.js`'s `FlatRandomForest` already uses for its own trees
(`[feature_index, threshold, left_child, right_child]`-style flat arrays), just
reading vigra's layout instead of writing our own.

## Everything else is already solved

Parsing the rest of the `.ilp` (axis order, feature selection matrix, scales,
label names, input shape) is plain HDF5 + JSON, no C library needed — `h5py`
handles it, and there's already a working pydantic-based parser in
`ilastik/experimental/parser` plus the format writeup in
`ILP_PixelClassification_Format.md`. This part translates to the standalone file
close to verbatim.

## Proposed phases

1. **Forest format research** (blocking, do first): using your vigra-enabled
   env, dump `PixelClassification/ClassifierForests` from a couple of real
   trained `.ilp` files (small forest, e.g. 2-class/10-tree, plus the bundled
   `notebooks/pixel_classification_api/pc.ilp`) and reverse-engineer the array
   encoding well enough to reimplement `predictProbabilities` in NumPy.
2. **Feature filters**: implement the 6 filter types + kernel builder in NumPy/
   SciPy, validate per-filter against `vigra.filters.*` output on a test array
   (same env) within float tolerance.
3. **RF inference**: implement the decoded forest format as a flat-array NumPy
   walker, validate against `rf.predictProbabilities(...)` on real feature data.
4. **Assemble the single file**: HDF5 `.ilp` parsing (axistags, feature
   selection matrix, label names) + phase 2 + phase 3, wired together the same
   way `PixelClassificationPipeline` does it, exposed as both a small CLI
   (`python pixel_classification_standalone.py project.ilp image.tif -o out.h5`,
   modeled on `run_pc_vigra_baseline.py`'s CLI) and an importable function/class
   for notebook use. Target deps: `numpy`, `scipy`, `h5py`, `tifffile` — all
   pip-installable, no compiled non-pip deps.
5. **End-to-end validation**: run both the real ilastik
   `PixelClassificationPipeline` (conda env) and the standalone file on the
   same `.ilp` + test image(s) and diff the probability maps numerically
   (tolerance-based, not bit-exact, since Gaussian kernel truncation/float
   accumulation order can differ slightly).
6. **Packaging**: decide how it ships — single file people can just download
   and `pip install numpy scipy h5py tifffile` alongside, or also publish as
   a real pip package. Punt this decision until phases 1-3 prove the numerics
   actually match.

## Open questions for you

- Test fixtures: OK to use `notebooks/pixel_classification_api/pc.ilp` (already
  in this repo, small, has a trained forest) as the primary validation project,
  plus maybe one of the `tests/test_ilastik/data/inputdata/*PixelClassification*.ilp`
  files for a second data point?
- Should 3D (zyx) inputs be in scope for v1, or is 2D (yx/yxc) enough to start —
  the eigenvalue-based features get more involved in 3D (3x3 vs 2x2 tensors).
- Once phase 1 (forest format) is decoded, want it written up as its own doc
  (like `ILP_PixelClassification_Format.md`) so it's reusable/shareable, or is
  that overkill for what's just an internal implementation detail?
