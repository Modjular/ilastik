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

**2. Random Forest inference (vigra's `RandomForest.writeHDF5` format) — SOLVED**

~~This was the actual unknown and the main risk in the plan.~~ Decoded and
validated. `ClassifierForests` in the `.ilp` is written by vigra's own
`forest.writeHDF5(...)`. The format (from vigra's own C++ headers, which ship
inside the conda package at `<env>/include/vigra/random_forest/`) is:

- `PixelClassification/ClassifierForests` holds N `Forest000N` sub-forests
  (ilastik's `ParallelVigraRfLazyflowClassifier` splits training across cores;
  at inference time their outputs are just tree-count-weighted-averaged back
  together — see `lazyflow/classifiers/parallelVigraRfLazyflowClassifier.py:321-357`).
- Each `Forest000N/Tree_NN` group has exactly two flat arrays:
  `topology` (int32) and `parameters` (float64). `topology[0:2]` is
  `[featureCount, classCount]`; traversal starts at index 2.
- At each topology index: `typeID = topology[i]`. If
  `typeID & 0x40000000` (`LeafNodeTag`), it's a leaf (`e_ConstProbNode`):
  `parameters[addr+1 : addr+1+classCount]` are that leaf's per-class
  probabilities (`addr = topology[i+1]`). Otherwise it's an axis-aligned
  threshold split (`i_ThresholdNode`, the only split type ilastik's default RF
  training produces): `column = topology[i+4]`,
  `threshold = parameters[addr+1]`, next index is `topology[i+2]` if
  `feature[column] < threshold` else `topology[i+3]`.
- Per-forest probability = mean of all its trees' leaf probability vectors.
  Across sub-forests: weighted-average by each sub-forest's tree count
  (equivalent to averaging over every tree in every sub-forest).

Implemented as **`vigra_rf_reader.py`** (this directory) — pure NumPy + h5py,
zero vigra dependency — and validated against real
`vigra.learning.RandomForest.predictProbabilities()` on 3 independently
trained `.ilp` fixtures (`notebooks/pixel_classification_api/pc.ilp`,
`tests/test_ilastik/data/inputdata/mitocheckPixelClassification.ilp` [40
sub-forests], `tests/test_ilastik/data/inputdata/pc-oc-133.ilp` — 25 vs 49
features, 4 vs 40 sub-forests). Max abs diff ~2.9e-8 across all three (float64
accumulation-order noise, not a real discrepancy). See
`dev_validation/validate_rf_reader_against_vigra.py` to reproduce (needs a
vigra env — not needed to *use* `vigra_rf_reader.py` itself).

Not yet exercised: vigra's `HyperplaneNode`/`HypersphereNode` split types
(vigra supports them, but ilastik's default RF training only ever produces
axis-aligned `ThresholdNode` splits, which is all 3 fixtures used).

## Everything else is already solved

Parsing the rest of the `.ilp` (axis order, feature selection matrix, scales,
label names, input shape) is plain HDF5 + JSON, no C library needed — `h5py`
handles it, and there's already a working pydantic-based parser in
`ilastik/experimental/parser` plus the format writeup in
`ILP_PixelClassification_Format.md`. This part translates to the standalone file
close to verbatim.

## Proposed phases

1. **Forest format research** — DONE, see above. `vigra_rf_reader.py` is a
   working, validated, dependency-light (h5py+numpy only) reader for
   `ClassifierForests`.
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

## Vigra dev environment (for validation work)

No conda was preinstalled in this session's container. Set one up like this
(ephemeral — doesn't survive a container reset, ~5 min to rebuild):

```bash
curl -sS -L -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p <prefix>/miniconda
<prefix>/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
<prefix>/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
<prefix>/miniconda/bin/conda create -n vigra-env -c ilastik-forge -c conda-forge python=3.10 vigra fastfilters h5py numpy
```

Note: importing the *full* `lazyflow` package in this env fails (`ModuleNotFoundError: z5py`,
which isn't pip-installable for py3.10 and isn't in this minimal env) — that's
expected and fine, since the whole point is to avoid needing lazyflow. Talk to
`vigra`/`h5py` directly instead, as `dev_validation/validate_rf_reader_against_vigra.py`
does.

Useful reference, also bundled with the conda `vigra` package itself, no
internet needed: `<env>/include/vigra/random_forest/*.hxx` and
`<env>/include/vigra/random_forest_hdf5_impex.hxx` — the actual C++ source
defining the HDF5 format decoded above.

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
