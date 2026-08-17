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
pipeline. Acceptance metric is deliberately NOT raw max-abs-diff (see
PLAN.md's "Precision follow-up" section for why that's not a coherent
target — fastfilters doesn't even bit-match itself across CPUs); it's
argmax-flip-rate vs. the real pipeline's own decision-margin at each flip:
- RF inference: **bit-identical** (`0.0` diff) to real
  `vigra.learning.RandomForest.predictProbabilities()` — the accumulation
  replicates vigra's exact float32-throughout arithmetic, not just its math.
- End-to-end: **8 / 1,376,256 argmax flips (0.0006%)**, and every one of
  them is at a pixel where the real pipeline's own top1-top2 confidence
  margin was already ≤ 0.08 (several exactly `[0.5, 0.5]`) — i.e. genuinely
  ambiguous pixels, not confident disagreements. `dev_validation/compare_against_real_pipeline.py`
  reports `ALL OK` on this basis, run on the full image by default.
- Root cause of the residual ~1e-6-level feature noise that drives those
  rare flips is understood in detail (traced to a specific pixel, feature
  vector diffed term-by-term, confirmed as ordinary cross-implementation
  convolution rounding) and confirmed NOT closeable by algorithm changes —
  a second-opinion suggestion to swap `numpy.linalg.eigvalsh` for
  fastfilters' own closed-form eigenvalue solver was implemented and tested,
  and made no difference at all (proven why: eigvalsh and the closed-form
  solver agree with each other to `1.5e-12` on identical input — the gap is
  from *input* differences hitting inherent eigenvalue-problem conditioning,
  not algorithm choice).

**Proof it's fast enough:** full image runs in ~47s end-to-end (was
impractical to even finish before the RF-walker vectorization — see
PLAN.md phase 7).

## File map

```
pixel_classification_standalone.py   <- THE deliverable (concatenation of the 4 below + CLI)
kernel1d.py                          <- vigra's exact Gaussian/derivative kernel construction
nd_filters.py                        <- the 6 ilastik feature filters, built on kernel1d
ilp_features.py                      <- OpPixelFeaturesPresmoothed's 2-stage scale composition
vigra_rf_reader.py                   <- decodes vigra's RandomForest HDF5 format, pure numpy
ilp_project.py                       <- parses .ilp metadata via h5py (axes, scales, labels)
standalone_pipeline.py               <- wires ilp_project+ilp_features+vigra_rf_reader together
                                         (pixel_classification_standalone.py IS these 5 files
                                         concatenated - keep them in sync if you edit one)
dev_validation/                      <- scripts proving correctness against real vigra/fastfilters/
                                         ilastik. All need a conda env (see PLAN.md), NOT part of
                                         the pip-only deliverable.
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

## Open items (none blocking correctness)

In rough priority order if you're picking a next task:

1. **Feature-computation performance.** RF inference is fast now (~20s for
   1.37M pixels, and bit-identical to real vigra); feature computation (the
   `nd_filters.py`/`ilp_features.py` convolutions) is the larger remaining
   chunk of the ~47s full-image runtime. Same kind of vectorization
   opportunity as the RF walker had, or could lean harder on
   `scipy.ndimage`'s existing vectorization rather than the current per-axis
   Python loop over `(feature, scale, channel)` triples.
2. **Packaging.** Still just a file someone downloads. Could become a real
   pip package (`pip install ilastik-standalone-pc` or similar) if that's
   wanted — the file is already dependency-clean, so this is mostly
   `pyproject.toml` + CI, not more reverse-engineering.
3. **3D validation depth.** 2D is thoroughly validated (real project, real
   image, full-scale, and passes the argmax-flip acceptance metric). 3D has
   only been validated on synthetic arrays against `fastfilters` directly
   (not against a real trained 3D `.ilp` end-to-end, since none was
   available in this repo/session). If a 3D test project turns up, running
   it through `compare_against_real_pipeline.py` would close that gap. Known
   residual: eigenvalue-based 3D filters have a root-caused (not a bug, and
   confirmed not fixable by an algorithm swap — see PLAN.md's "Precision
   follow-up") ~1e-4-level precision gap near degenerate eigenvalues that
   doesn't meaningfully show up in 2D.
4. **Autocontext / other workflows.** Explicitly out of scope so far — only
   `workflowName == "Pixel Classification"` is supported;
   `ilp_project.py`/`ilp_project.py`'s embedded copy raises on anything
   else. `ilastik.experimental.api.AutocontextPipeline` is the real-API
   equivalent if this is ever wanted; same overall approach would apply
   (two-stage feature+RF, doubled).
5. **Training** was explicitly out of scope from the start (this is
   inference-only, matching what the real experimental API offers). Not
   attempted, not started.
