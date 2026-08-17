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

**1. Feature filters (vigra/fastfilters → pure NumPy/SciPy) — SOLVED (2D), mostly solved (3D)**

Implemented as **`kernel1d.py`** + **`nd_filters.py`** (this directory) — pure
NumPy/SciPy, zero vigra/fastfilters dependency. Derived from vigra's actual
C++ source (`<env>/include/vigra/separableconvolution.hxx`, shipped inside the
conda package), not guessed:

- Kernel radius: `round(window_size * sigma)` (confirmed via `vigra.filters.Kernel1D`,
  not the `ceil((3+0.5*order)*scale)` guess from the old `ff.js` reference).
- Order 0 (smoothing): discretized Gaussian pdf, rescaled so it sums to 1.
- Order ≥ 1 (derivatives): discretized analytic Gaussian derivative, zero-meaned,
  then rescaled so its discrete moment exactly matches the analytic one
  (corrects for truncation error) — this exact recipe, including the zero-mean
  step for order 2, was reverse-engineered empirically against `vigra.filters.Kernel1D`
  output and now matches it to machine precision (`~1e-16`) across every
  (sigma, order, window_size) combination ilastik actually uses.
- Boundary handling: `scipy.ndimage` mode `'mirror'` == vigra's
  `BORDER_TREATMENT_REFLECT` (confirmed empirically).
- All 6 filter types (`GaussianSmoothing`, `LaplacianOfGaussian`,
  `GaussianGradientMagnitude`, `DifferenceOfGaussians`,
  `StructureTensorEigenvalues`, `HessianOfGaussianEigenvalues`) built from
  that one kernel builder + separable convolution + (for the last two)
  per-pixel eigendecomposition (`numpy.linalg.eigvalsh`, sorted descending —
  vigra's convention).

Validated against real `fastfilters` output (`dev_validation/validate_filters_against_fastfilters.py`,
needs the vigra/fastfilters conda env) across the full real ilastik scale set
(`0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0`):
- **2D: matches to ~1e-6/1e-7 for all 6 filter types, every scale.**
- **3D: gaussianSmoothing/laplacianOfGaussian/gaussianGradientMagnitude match
  the same way.** The two eigenvalue-based filters
  (`hessianOfGaussianEigenvalues`, `structureTensorEigenvalues`) show ~1e-4
  residual differences at small scales on adversarial random-noise volumes.
  **Root cause identified (not just "known gap" anymore):** the raw tensor
  components (before eigendecomposition) match `vigra.filters.hessianOfGaussian`
  to the same ~1e-6/1e-7 precision as every other filter — the convolution
  math is fine. The difference only appears *after* `eigvalsh`, and only at
  pixels where two eigenvalues are nearly degenerate (checked directly: at the
  single worst pixel in a test volume, two eigenvalues sat `0.0048` apart, and
  a ~1e-6 difference in the input matrix — inevitable between two
  independently-written convolution implementations that sum in a different
  order — got amplified by roughly `1/eigengap ≈ 200×` into the observed ~1e-4
  eigenvalue difference). **Confirmed this is not a dtype/precision issue**:
  recomputing the same pixel with a full float64 pipeline (vs. the normal
  float32-in/float32-out path scipy.ndimage uses by default) gave `0.00054293`
  vs `0.00054288` — essentially identical, if anything marginally worse. This
  only shows up on adversarial uncorrelated-noise test data (no dominant local
  orientation → near-isotropic Hessians are common there); real, spatially
  smooth microscopy images hit this condition far less often, and a 1e-4-scale
  wobble in one channel of a ~50-dimensional feature vector feeding an RF
  split threshold is unlikely to flip a prediction. Treating this as
  understood-and-accepted rather than open.

Found and had to work around one genuine **`fastfilters` bug**: its Python
`structureTensorEigenvalues(image, innerScale, outerScale, window_size)` has a
real parameter-order mismatch between its C header
(`fastfilters_fir_structure_tensor2d(in, sigma_outer, sigma_inner, ...)`) and
its C++/Python binding (`src/python/core.cxx`, `ConvolveST`), which passes its
own `sigma_inner` into the C function's `sigma_outer` slot and vice versa. Net
effect (confirmed empirically against real `svenpeter42/fastfilters` source
and output): the Python-level `innerScale` argument ends up used as the
tensor-smoothing scale, and `outerScale` ends up used as the gradient scale —
backwards from the parameter names, and from what `vigra.filters.structureTensorEigenvalues`
itself does. Since real trained `.ilp` projects are computed with `fastfilters`
(ilastik prefers it when installed), `nd_filters.py`'s
`structure_tensor_eigenvalues()` deliberately reproduces this swap — matching
real projects' actual numbers, not the "textbook"/vigra ordering.

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
features, 4 vs 40 sub-forests). **Bit-identical (`0.0` diff) on all three** —
the accumulation replicates vigra's exact float32-throughout arithmetic, not
just its math (see "Precision follow-up" below for how this was found and
closed; it wasn't bit-identical until that pass). See
`dev_validation/validate_rf_reader_against_vigra.py` to reproduce (needs a
vigra env — not needed to *use* `vigra_rf_reader.py` itself).

Not yet exercised: vigra's `HyperplaneNode`/`HypersphereNode` split types
(vigra supports them, but ilastik's default RF training only ever produces
axis-aligned `ThresholdNode` splits, which is all 3 fixtures used).

**The "presmoothing" scale composition — SOLVED**

`OpPixelFeaturesPresmoothed` doesn't call each feature filter at its nominal
scale directly — for performance it presmooths the source once per unique
scale column, then runs cheap small-window filters on top of that shared
presmoothed image, changing the actual numbers involved (not just a speed
trick):

```
temp_sigma, new_scale = (sqrt(s**2-1), 1.0) if s > 1 else (s, s)
presmoothed = gaussian_smoothing(source, temp_sigma, window_size=3.5)
feature_value = <feature filter>(presmoothed, new_scale, ..., window_size=2.0)
```

(For `s <= 1`, this really does mean smoothing twice at the same scale before
running the feature filter — confirmed against source, not a bug I introduced.)

Implemented as **`ilp_features.py`** (this directory), built on `nd_filters.py`.
Validated end-to-end against the same composition built from real `fastfilters`
calls (`dev_validation/validate_ilp_features_against_fastfilters.py`) across
all 7 real scales × all 6 feature types (2D) — matches to ~1e-7/1e-8
throughout.

## Everything else is already solved

Parsing the rest of the `.ilp` (axis order, feature selection matrix, scales,
label names, input shape) is plain HDF5 + JSON, no C library needed — `h5py`
handles it, and there's already a working pydantic-based parser in
`ilastik/experimental/parser` plus the format writeup in
`ILP_PixelClassification_Format.md`. This part translates to the standalone file
close to verbatim.

## Proposed phases

1. **Forest format research** — DONE. `vigra_rf_reader.py`, validated.
2. **Feature filters** — DONE (2D fully; 3D minor known-and-explained gap in
   eigenvalue filters, see above). `kernel1d.py` + `nd_filters.py`, validated.
3. **Presmoothing scale composition** — DONE. `ilp_features.py`, validated.
4. **Assemble the single file** — DONE. `ilp_project.py` (HDF5 `.ilp` parsing:
   axistags, feature selection matrix, scales, label names — plain h5py/JSON,
   mirrors `ilastik/experimental/parser`'s field derivation without needing
   pydantic/vigra) + `standalone_pipeline.py` (wires `ilp_project.py` +
   `ilp_features.py` + `vigra_rf_reader.py` together the same way
   `PixelClassificationPipeline` (`ilastik/experimental/api/_pipelines.py`)
   does conceptually: reorder input to canonical `zyxc` spatial order, compute
   every selected (feature, scale) block per input channel in
   FeatureIds-outer/Scales-inner/channel-innermost order exactly matching
   `OpPixelFeaturesPresmoothed`, flatten, run through the forest, reshape,
   reorder output back to the trained project's original axis order). Exposes
   the same `PixelClassificationPipeline.from_ilp_file(...).get_probabilities(...)`
   interface as the real API (accepts an `xarray.DataArray` if xarray happens
   to be installed, or a plain array + `dims=` otherwise), plus `ComputeIn2d`
   handling for 3D (per-z-slice, matching `fastfilters`' own branch for it).
   All four pieces then got concatenated into one literal file,
   **`pixel_classification_standalone.py`** — the actual deliverable, ~600
   lines, importable as a library or runnable as a CLI
   (`python pixel_classification_standalone.py project.ilp image.tif -o probs.npy`).
   Dependencies: `numpy`, `scipy`, `h5py` (required); `tifffile`/`imageio`
   (CLI-only, for reading non-`.npy` image files); `xarray` (optional, only
   changes how output is labeled). No compiled non-pip deps anywhere.
5. **End-to-end validation** — DONE. Built a real Python 3.11 conda env with
   the *actual* `ilastik.experimental.api.PixelClassificationPipeline`
   working end-to-end (see "Full reference-pipeline environment" below — this
   took real effort: `lazyflow` needs Python ≥ 3.11 for `enum.StrEnum`, pulls
   in `z5py` which isn't used by anything relevant here, and a handful of
   other packages). Ran both the real pipeline and
   `pixel_classification_standalone.py` on the same trained project
   (`notebooks/pixel_classification_api/pc.ilp`) and the same real test image
   (`2d_cells_apoptotic_1channel.png`, a 64×64 crop for speed — see the
   performance note below) and diffed the output:
   - **Raw probabilities: max abs diff 2.86e-8, mean abs diff 1.25e-10** — the
     same float64-accumulation-order noise floor seen in the RF-only
     validation, i.e. no real discrepancy at all.
   - **Binarized (argmax) predictions: 0 / 4096 pixels mismatched.**

   (Numbers as of this pass. The RF accumulation was later made bit-identical
   — see "Precision follow-up" below — and the harness itself was reworked
   to run the full image by default and gate on argmax-flip-vs-real-margin
   instead of raw diff; this snapshot is left as-is for the historical
   record of what phase 5 originally proved.)

   This is now a reusable, standing test harness, not a one-off check:
   **`dev_validation/compare_against_real_pipeline.py`**, runnable as
   `python compare_against_real_pipeline.py <project.ilp> <image> [--crop Y0 Y1 X0 X1] [--margin-threshold 0.1]`.
   It self-contains the `z5py`/`ilastik._version` workarounds (see below) so
   it doesn't need any manual setup beyond having the reference conda env
   active.
6. **Packaging**: `pixel_classification_standalone.py` already ships as the
   single file people can download and use directly, `pip install numpy
   scipy h5py` (+ `tifffile`/`imageio` for the CLI) alongside. Publishing it
   as a real installable pip package is still an open option, not yet done.
7. **Performance** — DONE. `vigra_rf_reader.py`'s (and
   `pixel_classification_standalone.py`'s embedded copy's) tree walker was a
   pure-Python per-pixel/per-tree loop — ~500s (measured) for the full
   1024×1344 `pc.ilp` test image (100 trees × ~1.4M pixels). Rewrote it to
   walk every pixel through a tree simultaneously, level by level: a NumPy
   array holds each pixel's current node index, updated in one vectorized
   batch per iteration, so the Python-level loop runs `O(tree depth)` times
   per tree instead of `O(n_pixels)` times. Confirmed bit-identical output to
   the old version (`dev_validation/validate_rf_reader_against_vigra.py`
   gives the exact same diff numbers, `2.8610229518832853e-08`, before and
   after — only the traversal order changed, not the algorithm). **Result:
   ~500s → ~20s for RF inference alone (~25×); ~47s measured for the full
   `pixel_classification_standalone.py` pipeline end-to-end on the full
   1024×1344 image (features + RF), down from what would have been ~500-600s+
   total.** Feature computation (not touched by this round) is now the
   larger share of total runtime; vectorizing/batching that further is a
   possible future follow-up but wasn't needed to hit "practical for a real
   image," which was the goal.

### Full-image validation: RF decision-boundary sensitivity (found during performance testing, not a bug)

Testing the *full* 1024×1344 `pc.ilp` image end-to-end (only practical after
the performance fix above) surfaced something the 64×64 crop never hit:
**12 / 1,376,256 pixels (0.00087%) with a different binarized (argmax)
prediction** than the real pipeline, and a handful of pixels with raw
probability differences up to ~0.09 (vs. the ~1e-8 noise floor seen
everywhere else). Traced to ground truth, not assumed:

- Several of the 12 "mismatches" are pixels where **both** implementations
  independently landed on an exact `[0.5, 0.5]` tie — `argmax` breaks ties by
  index, so a `~1e-8`-level float difference on either side of the tie flips
  which class "wins" a coin flip that was already 50/50. Not a real
  disagreement.
- For the larger ones (e.g. pixel `(469, 147)`: real `[0.35, 0.65]` vs.
  standalone `[0.26, 0.74]`), pulled the actual 49-value feature vector at
  that exact pixel from both the real `OpFeatureSelection` operator and
  `standalone_pipeline.py`'s `_compute_features` and diffed feature-by-feature.
  **The raw features match to ~1e-5/1e-6** (e.g.
  `HessianOfGaussianEigenvalues@0.7` differs by `0.000007`,
  `LaplacianOfGaussian@0.7` by `0.000009`) — the ordinary level of
  disagreement between two independently-written convolution
  implementations, consistent with everything validated elsewhere in this
  plan. There's no large per-feature error anywhere.
- Random forests are piecewise-constant (every split is an exact
  `feature[col] < threshold` test). At a pixel whose true feature vector
  happens to sit almost exactly on one of the ~100 trees' split thresholds,
  a `~1e-6`-level nudge is enough to flip that tree's branch — and if several
  of the 100 trees share a similar threshold near that same point (plausible,
  since they're bootstrap samples of the same training set), enough of them
  can flip in the same direction to move the ensemble average by something
  as large as `0.09`, even though the underlying computation is off by
  next to nothing.

This is inherent to comparing two independently-implemented floating-point
pipelines through an ensemble of exact-threshold decision trees — not a
defect introduced by this session's work, and not something the performance
change caused (the RF walker was proven bit-identical before/after
vectorizing, and this exact discrepancy was already latent in the original
row-by-row implementation; the 64×64 crop just never happened to contain one
of these rare boundary pixels). Rate observed: 12 boundary-sensitive pixels
out of 1,376,256 (0.00087%) on one real image. Follow-up below closed part of
this gap and reframed how it's measured.

### Precision follow-up: tested 3 concrete recommendations, 2 landed, 1 didn't (and why)

A second opinion (a separate Claude Opus review of this plan) proposed three
concrete, low-risk changes plus a reframed acceptance metric. Each was
implemented and *measured* against real vigra/fastfilters/ilastik output
rather than assumed to help - one of them didn't, and it's documented as a
negative result rather than silently dropped:

1. **"Float parity with fastfilters" isn't a fixed target in the first
   place.** fastfilters does runtime CPU dispatch (AVX2+FMA / AVX / SSE) -
   those paths don't produce identical floats, so even fastfilters doesn't
   bit-match itself across machines, and vigra/fastfilters don't bit-match
   each other either (ilastik uses whichever is installed). Bit-exactness
   with "the reference" was never a coherent target. This reframes what
   "done" means below.

2. **Exact RF accumulation order and dtype - DONE, and it's a complete win.**
   `vigra.learning.RandomForest.predictProbabilities()`'s Boost.Python
   signature is hard-typed to float32 output (`NumpyArray<2, float, ...>`,
   confirmed by triggering its `ArgumentError` with a float64 array) - so the
   *entire* accumulation in real ilastik happens in float32, not float64:
   vigra's `RandomForest::predictProbabilities` (random_forest.hxx)
   truncates every single tree's contribution to `T=float32` before adding
   it to the running per-pixel probability sum (only the internal
   `totalWeight` normalizer stays `double`); and
   `ParallelVigraRfLazyflowClassifier.predict_probabilities`
   (lazyflow/classifiers/parallelVigraRfLazyflowClassifier.py) then combines
   the sub-forests' already-float32 outputs with float32 arithmetic too.
   `vigra_rf_reader.py` was accumulating in float64 throughout - "more
   accurate" in isolation, but a mismatch from what real ilastik actually
   computes. Rewrote both `DecodedForest.predict_probabilities` (per-forest)
   and `DecodedParallelForest.predict_probabilities` (cross-forest) to
   replicate the exact float32 accumulation. Result:
   **`dev_validation/validate_rf_reader_against_vigra.py`'s diff against real
   vigra went from `2.86e-8` to exactly `0.0` on all 3 fixtures** - not
   "smaller," bit-identical. On the full 1024×1344 real-image end-to-end
   test, this reduced the argmax-flip count from 12/1,376,256 to
   **8/1,376,256**. (A `predict_weighted_` guard was added alongside this -
   the real accumulation formula branches on that flag and every fixture
   validated against uses `predict_weighted_=0`; the untested branch now
   raises instead of silently producing wrong output for a project that sets
   it.)

3. **Kernel radius rounding convention - checked, confirmed not an issue.**
   fastfilters computes its FIR kernel length as `floor(window_ratio*sigma +
   0.5)` (`src/library/fir_kernel.c`) - i.e. round-half-up, not Python's
   `round()` (round-half-to-even). These differ only exactly on a `.5` tie,
   and none of ilastik's real `(window_size, sigma)` products (`2.0`/`3.5`
   times any of `0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0` or their derived scales
   like `sqrt(s²-1)`, `s*0.66`, `s*0.5`) land on an exact tie - already
   implicitly confirmed by every prior filter validation matching to
   `~1e-6/1e-7` (a kernel-length-off-by-one would show up as a large,
   obvious error, not a small residual). No code change.

4. **Closed-form eigenvalues to match fastfilters' `linalg.c` -
   implemented, tested, and it does NOT help. Not adopted.** The hypothesis
   (from the second opinion) was that `numpy.linalg.eigvalsh` (LAPACK
   `?syevd`) amplifies near-degenerate-eigenvalue input noise differently
   than fastfilters' closed-form analytic solver
   (`src/library/linalg.c`: `_ev2d_default` for 2×2, an Eberly's-method
   trigonometric solve for 3×3), and that matching the algorithm would
   collapse the amplification gap. Transcribed fastfilters' exact C
   formulas (same expression grouping, same `min(aDiv3,0)`/`min(q,0)`
   clamps, same 3-element compare-swap sort network) into NumPy and tested
   against real fastfilters on the same adversarial 3D near-degenerate case
   characterized earlier: **the diff was identical to the last digit before
   and after** (`5.429e-04` both times at scale 0.7, `1.160e-04` both times
   at scale 1.6). Isolated why with a direct test: running *both*
   `eigvalsh` and the closed-form solver on the exact same (already
   computed) input matrix, they agree with **each other** to `1.5e-12` -
   both are highly accurate algorithms for this problem. That proves the
   `~1e-4` gap against fastfilters is driven entirely by the `~1e-6`
   *input*-matrix difference (from upstream convolution rounding, already
   characterized) passing through the eigenvalue problem's inherent
   conditioning near degeneracy - not by which accurate algorithm solves
   it. Swapping solvers can't fix sensitivity that lives in the math
   problem itself, only in a genuine algorithm defect (there wasn't one).
   Kept `numpy.linalg.eigvalsh` - it's simpler, already validated, and
   produces the same numbers.

5. **Reframed the acceptance metric, per the same feedback.** Raw
   max-abs-diff on probabilities was always going to plateau somewhere
   above zero for the reasons in point 1 - it doesn't distinguish "this
   pixel was already a coin flip in the real pipeline" from "we confidently
   disagree with a confident answer," which is the distinction that
   actually matters. `dev_validation/compare_against_real_pipeline.py` now
   gates pass/fail on: binarized (argmax) flip rate, and for every flipped
   pixel, the *real* pipeline's own decision margin (top1 - top2
   probability) there. A flip only fails the check if the real pipeline's
   own margin exceeds `--margin-threshold` (default `0.1`) - i.e. it was
   actually confident and got contradicted. Run against the full
   1024×1344 real image (default now, no `--crop` needed - see the
   performance section above): **8 flips, real-pipeline margin at every one
   of them between 0.0 and 0.08 (several exactly `[0.5, 0.5]`), 0 confident
   flips → `ALL OK`.** Raw probability diff stats are still printed
   (`max abs diff: 9.0e-2` on that same run) but are informational only, not
   part of the pass/fail decision anymore.

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

### Full reference-pipeline environment (for `compare_against_real_pipeline.py`)

Getting the *real* `ilastik.experimental.api.PixelClassificationPipeline`
importable (needed as ground truth for the end-to-end comparison harness) is
a step up from the vigra-only env above — `lazyflow`/`ilastik` pull in more
than just vigra/fastfilters:

```bash
<prefix>/miniconda/bin/conda create -n vigra-env311 -c ilastik-forge -c conda-forge \
    python=3.11 vigra fastfilters h5py numpy
<env>/bin/pip install xarray pydantic future psutil greenlet scikit-learn imageio
<prefix>/miniconda/bin/conda install -n vigra-env311 -c ilastik-forge -c conda-forge ndstructs
```

Notes:
- **Python must be ≥ 3.11** — `lazyflow/utility/data_semantics.py` uses
  `enum.StrEnum`, added in 3.11. The vigra-only env above used 3.10 (fine for
  everything that doesn't need full `lazyflow`); this one needs 3.11+.
  `ilastik-forge`/`conda-forge` do have `vigra`/`fastfilters` builds for 3.11.
- `sklearn` is not a real pip package name (installs a deprecated shim that
  errors) — install `scikit-learn` instead.
- `ndstructs` isn't on PyPI for this Python version; install it from
  `ilastik-forge` via conda instead of pip.
- Two more things aren't real missing dependencies, just artifacts of running
  from a plain source checkout instead of a proper install — both are handled
  automatically inside `compare_against_real_pipeline.py`, no manual step
  needed:
  - `lazyflow/__init__.py` unconditionally does `import z5py` and calls
    `z5py.set_json_encoder(...)` at import time, purely for N5 file I/O
    support, which nothing in this comparison needs and which isn't
    pip-installable for this Python version anyway. Stub it:
    `sys.modules["z5py"] = types.ModuleType("z5py")` with a no-op
    `set_json_encoder` attached, *before* importing `ilastik`.
  - `ilastik/__init__.py` imports `ilastik._version`, normally generated by
    `pip install -e .` (`setuptools_scm`). From a plain checkout it doesn't
    exist; write a throwaway one-line stub (`version = "0.0.0.dev0"`) if
    missing. Already gitignored (`ilastik/_version.py`), so this never ends
    up in a commit.

## Status

Every phase above is done and validated, end to end, against a real trained
`.ilp` project and the real `ilastik.experimental.api.PixelClassificationPipeline`.
The standalone deliverable is `pixel_classification_standalone.py`; the
standing regression/proof harness is `dev_validation/compare_against_real_pipeline.py`,
which now runs the **full** real test image by default (not just a crop) and
gates pass/fail on argmax-flip-rate vs. the real pipeline's own decision
margin, not raw probability diff (see "Precision follow-up" above for why).
Current result on the full 1024×1344 real image: RF inference is
bit-identical to real vigra; **8 argmax flips out of 1,376,256 pixels
(0.0006%), all at pixels where the real pipeline's own confidence margin was
already ≤0.08 → `ALL OK`.**

Remaining open items, none of them blocking correctness:

- **Feature-computation performance**: RF inference is now both fast (~20s
  for 1.37M pixels) and bit-identical to real vigra; feature computation is
  the larger remaining share of the ~47s full-image runtime. Not urgent
  (this is already a >10x improvement and "practical for a real image" was
  the bar), but a candidate for a future round if it needs to be faster
  still.
- **The residual ~1e-6 feature-level noise against fastfilters** (and the
  rare argmax flips it causes at already-ambiguous pixels) is understood in
  detail (see "Precision follow-up" above) and not closeable without
  vendoring fastfilters' actual C convolution code — deliberately not done,
  since it would destroy the "pip install numpy scipy h5py, one file"
  property that's the entire point of this. Current state (0.0006% flip
  rate, 100% confined to pixels the real pipeline was already unsure about)
  is treated as the acceptance bar, not a gap to keep closing.
- **3D eigenvalue-filter precision**: same root cause as above, same
  conclusion — not a bug, not chasing further (also tested and ruled out:
  switching to fastfilters' own closed-form eigenvalue algorithm, see
  "Precision follow-up" point 4).
- **Autocontext / other workflows**: this only handles the Pixel
  Classification workflow (`_PixelClassificationProject.from_ilp_file` raises
  on anything else). `AutocontextPipeline` in the real API is out of scope
  unless wanted later.
- **Packaging**: still just a single `.py` file someone downloads; not
  published as an installable package.
