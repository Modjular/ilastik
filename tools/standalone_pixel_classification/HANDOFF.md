# Handoff: standalone single-file Pixel Classification

Status snapshot for whoever picks this up next. Full research log with all
the reverse-engineering detail, worked examples, and validation numbers is
in `PLAN.md` in this same directory — this file is the short version: what's
done, what works, exact commands, and where to start.

**PR:** https://github.com/Modjular/ilastik/pull/2 (branch
`claude/pixel-classification-single-file-w6w64p`). Push more commits to that
branch to update it — don't open a new PR.

## What this is

A pip-only (`numpy`, `scipy`, `h5py` — no vigra, no fastfilters, no
lazyflow, no PyQt, no conda) reimplementation of ilastik Pixel
Classification **inference** (not training): load a real `.ilp` trained by
the desktop app, run it on a new image, get the same probability maps. Goal
was explicitly "run pixel classification via a single file."

## Where things stand: done, not a prototype

Every piece is individually reverse-engineered from vigra/fastfilters' own
source (not guessed) and validated against the real thing, then the whole
pipeline was validated end-to-end against the real
`ilastik.experimental.api.PixelClassificationPipeline` on a real trained
project. Then made fast enough to be practical, not just correct.

**The deliverable:** `pixel_classification_standalone.py` — one file, ~650
lines, library + CLI (`python pixel_classification_standalone.py project.ilp
image.tif -o probs.npy`).

**Proof it's correct**, on `notebooks/pixel_classification_api/pc.ilp` +
its bundled test image, full 1024×1344 image, diffed against the real
pipeline:
- Raw probabilities: max abs diff `2.86e-8` (float64 noise floor, not a
  real discrepancy)
- Binarized predictions: 12 / 1,376,256 pixels (0.00087%) differ — traced to
  ground truth and explained (RF decision-boundary sensitivity amplifying
  ordinary ~1e-6 cross-implementation feature noise; not a bug — see
  PLAN.md's "Full-image validation" section for the full diagnostic)

**Proof it's fast enough:** full image runs in ~47s end-to-end (was
impractical to even finish before the RF-walker vectorization — see
PLAN.md phase 7).

## File map

```
pixel_classification_standalone.py   <- THE deliverable (concatenation of the 4 below + CLI)
ilastik_pc.py                        <- teaching copy: same algorithm, every optimization removed,
                                        literal convolution loops, one-pixel-at-a-time forest walk.
                                        `--explain Y X` prints a single pixel's whole derivation
                                        (feature vector + each tree's decisions). Slow on purpose;
                                        verified to give identical probabilities to the deliverable.
                                        2D only. Not used by anything - it's for reading.
REVIEW_FIXES.md                      <- code review of the above, the seven fixes made, and what
                                        is measured vs merely reasoned. Read before optimizing.
kernel1d.py                          <- vigra's exact Gaussian/derivative kernel construction
nd_filters.py                        <- the 6 ilastik feature filters, built on kernel1d
ilp_features.py                      <- OpPixelFeaturesPresmoothed's 2-stage scale composition
vigra_rf_reader.py                   <- decodes vigra's RandomForest HDF5 format, pure numpy
ilp_project.py                       <- parses .ilp metadata via h5py (axes, scales, labels)
standalone_pipeline.py               <- wires ilp_project+ilp_features+vigra_rf_reader together
                                         (pixel_classification_standalone.py IS these 5 files
                                         concatenated - keep them in sync if you edit one)
dev_validation/                      <- scripts proving correctness against real vigra/fastfilters/
                                         ilastik. Mostly need a conda env (see PLAN.md), NOT part of
                                         the pip-only deliverable. The exception is
                                         test_standalone_no_conda.py, which needs only
                                         numpy/scipy/h5py - run it on every change, it's seconds.
reference/                           <- material ported from the feat/webgpu branch (an earlier,
                                         JS/WebGPU-targeted attempt at this same problem) - read-only
                                         reference, not used by the deliverable.
```

If you change the feature-filter math, RF decoding, or pipeline wiring,
change it in the small module file AND in
`pixel_classification_standalone.py`'s copy of that section (there's no
build step stitching them together — the concatenation was done by hand).

## How to pick this up

1. Read `PLAN.md` top to bottom once — it has the *why* behind every design
   decision (vigra's exact kernel formula, the presmoothing algorithm, the
   fastfilters innerScale/outerScale bug, the RF HDF5 format, the RF
   boundary-sensitivity finding). Skimming it will save you from
   re-deriving things that already took real effort to work out.
2. To touch anything and re-validate, you need a conda env — none is
   preinstalled in a fresh container. PLAN.md has copy-pasteable recipes for
   both:
   - a lightweight `vigra`+`fastfilters` env (Python 3.10) for per-piece
     validation (`dev_validation/validate_*_against_*.py`)
   - a heavier Python 3.11 env with the *actual*
     `ilastik.experimental.api.PixelClassificationPipeline` importable, for
     `dev_validation/compare_against_real_pipeline.py` (the end-to-end
     regression test). This one needs Python ≥3.11 for lazyflow's
     `enum.StrEnum` usage, plus a handful of extra pip/conda packages, and
     the harness self-stubs two irrelevant-but-blocking imports (`z5py`,
     `ilastik._version`) — see PLAN.md, "Full reference-pipeline
     environment," it's all worked out already.
   - Both envs are ephemeral (session scratch space) and won't survive a
     container reset — budget ~5-10 min to rebuild if starting fresh.
3. Run `dev_validation/compare_against_real_pipeline.py` before and after
   any change to `pixel_classification_standalone.py` to catch regressions.
   Before reaching for conda at all, run
   `dev_validation/test_standalone_no_conda.py` (numpy/scipy/h5py only,
   seconds) - it covers the cases the one available fixture can't reach, and
   checks both copies of the code.
4. If you're onboarding someone else onto this, start them on
   `ilastik_pc.py --explain`, not on the deliverable. It's the same algorithm
   with the optimizations removed and a per-pixel trace.

## Open items (none blocking correctness)

In rough priority order if you're picking a next task:

1. **Performance.** This item's premise was wrong and has been remeasured -
   see REVIEW_FIXES.md for the profile. Feature computation is *not* the
   larger chunk: on the full image it is ~8.3s against RF inference's ~20.7s.
   Within feature computation, the cost is concentrated in the two
   eigenvalue filters (~5.4s of 8.3s) - `np.linalg.eigvalsh` over ~1.4M small
   matrices plus the tensor-component smoothing - not in convolution
   scheduling. Redundant presmoothing has already been removed and was worth
   only ~0.5s. So the two real targets, in order: RF inference (still the
   single biggest term), then a closed-form 2x2/3x3 eigenvalue solver in
   place of `eigvalsh` (`ilastik_pc.py`'s `eigenvalues_2x2` shows the 2D form,
   and it is exact, not an approximation).
2. **Packaging.** Still just a file someone downloads. Could become a real
   pip package (`pip install ilastik-standalone-pc` or similar) if that's
   wanted — the file is already dependency-clean, so this is mostly
   `pyproject.toml` + CI, not more reverse-engineering.
3. **3D validation depth.** 2D is thoroughly validated (real project, real
   image, full-scale). 3D has only been validated on synthetic arrays
   against `fastfilters` directly (not against a real trained 3D `.ilp`
   end-to-end, since none was available in this repo/session). If a 3D test
   project turns up, running it through `compare_against_real_pipeline.py`
   would close that gap. Known residual: eigenvalue-based 3D filters have a
   root-caused (not a bug, see PLAN.md) ~1e-4-level precision gap near
   degenerate eigenvalues that doesn't show up in 2D.
4. **Autocontext / other workflows.** Explicitly out of scope so far — only
   `workflowName == "Pixel Classification"` is supported;
   `ilp_project.py`/`ilp_project.py`'s embedded copy raises on anything
   else. `ilastik.experimental.api.AutocontextPipeline` is the real-API
   equivalent if this is ever wanted; same overall approach would apply
   (two-stage feature+RF, doubled).
5. **Training** was explicitly out of scope from the start (this is
   inference-only, matching what the real experimental API offers). Not
   attempted, not started.
