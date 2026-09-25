# Postmortem: the pure-Python single-file direction (2026-09-24)

**Status:** retired as the main direction. Its Python code (`.ilp` parsing, forest decoding and feature composition) is still used by the next direction. See `PLAN.md`, "Direction 2".
**Branch / PR:** `claude/pixel-classification-single-file-w6w64p`, https://github.com/Modjular/ilastik/pull/2

## Summary

For about a month we built `pixel_classification_standalone.py`. It is one file that runs ilastik Pixel Classification inference on a trained `.ilp` project, using only `numpy`, `scipy` and `h5py`.

It works and is almost exactly correct. It is also **25–50× slower than the ilastik binary** on the same image. That gap cannot be closed without compiled code, and compiled code is exactly what the single-file, pip-only constraint excluded. So the constraint and the use case contradicted each other, and we only found out once someone measured real ilastik's speed. That happened on 2026-09-24, after the implementation was finished.

## What we set out to do

We wanted a single `.py` file with pip-only dependencies (no conda, vigra, fastfilters, lazyflow or PyQt) that gives the same probability maps as `ilastik.experimental.api.PixelClassificationPipeline` for a real trained project.

## What worked

The correctness work was thorough, and all of it is still valid.

- **Kernels:** vigra's Gaussian and derivative kernel construction was reverse-engineered from the C++ source and matches it to machine precision.
- **Filters:** all 6 ilastik feature filters were reimplemented. In 2D they match fastfilters to about 1e-6/1e-7.
- **Scale composition:** ilastik's two-stage "presmoothing" scale composition (`OpPixelFeaturesPresmoothed`) was reproduced. So was the fastfilters structure-tensor parameter-swap bug that real output depends on.
- **Forest decoder:** a pure-NumPy decoder for vigra's RandomForest HDF5 format gives output **bit-identical** to `vigra.learning.RandomForest.predictProbabilities()`. That includes vigra's float32-throughout accumulation.
- **End to end:** on the full 1024×1344 test image, 8 of 1,376,256 pixels get a different predicted class than real ilastik. Every one of them is a pixel where real ilastik itself was close to a coin flip.

## What didn't work: speed

Measured on 2026-09-24 in one container (4 vCPU), full 1024×1344 `pc.ilp` test image:

| | Features | Random forest | Total |
|---|---|---|---|
| Pure-Python single file | 14.6s | 30.8s | **45.5s** |
| ilastik (`experimental.api`), default 4 threads | | | **0.9–1.75s** |
| ilastik, 1 thread | | | 3.1–4.1s |

The single file is about 25–50× slower than ilastik with its default settings. It is still 10–15× slower when ilastik is limited to one thread, so the gap comes mostly from compiled versus interpreted code, not from parallelism. Timings in this container vary by up to 2× from run to run, but the gap is far larger than that noise.

Our handoff notes had also drifted from reality. They said features were the larger share of the runtime. Measured, the random forest is about ⅔ of it.

## Why it can't be fixed within the constraint

1. **Random-forest inference is branchy pointer-chasing.**
   - Each pixel walks a different path through each tree, on data-dependent branches.
   - The best NumPy form walks all pixels through a tree one level at a time, using index arrays. That makes O(tree depth) full-array fancy-indexing passes for each of the 100 trees, each allocating temporary arrays and gathering from scattered memory.
   - It was already 25× faster than the per-pixel loop it replaced (about 500s down to about 20–30s). It is still about 100× slower than a plain C loop doing the same walk. The 40-line `rfwalk.c` in `native_prototype/` does it in about 0.4–1.0s.
   - NumPy has no fast primitive for "different branch per element, repeated until done."
2. **The feature filters are generic, while fastfilters is specialised.**
   - `scipy.ndimage.correlate1d` is a general-purpose, single-threaded convolution.
   - fastfilters uses hand-written AVX/FMA code for symmetric and antisymmetric Gaussian kernels. It also computes eigenvalues with a closed-form SIMD solver instead of `numpy.linalg.eigvalsh`'s general routine.
   - There is no scipy call that closes this gap.
3. **The only fixes are compiled code, which the constraint ruled out.** Numba, Cython, a C extension and vendoring fastfilters each break "pip install numpy scipy h5py, one file". Even the remaining 1e-6 feature differences (and the 8 changed pixels) came from not using fastfilters' own convolution code. `PLAN.md` noted this explicitly and deliberately left it unfixed for the same reason.

## How we got here: contributing causes

- **We never measured the reference speed at the start.** Speed first came up in phase 7, and the bar was "practical for a real image", not "comparable to ilastik". 47s met that bar, so the work was declared done. Timing ilastik on day one would have exposed the problem before any reverse-engineering started.
- **The constraint didn't match the deployment.**
  - "pip-only, no conda, single file" assumed an end user without ilastik installed.
  - In the actual deployment (the `tttk` Docker image), the ilastik binary is already in the container, and `tttk` already has a `build.py` that compiles vendored C code (OpenCV).
  - In that environment, compiled dependencies cost almost nothing, and ilastik's own compiled filter library is sitting on disk.
  - Nobody checked the constraint against the deployment until 2026-09-24.
- **Most of the effort went into precision.** We spent a lot on the last 1e-6 of agreement: kernel moment correction, float32 accumulation order, and comparing eigenvalue solvers. That work was valuable and is kept, but it was optimising a property that was already good enough while the actual blocker (speed) went unmeasured.

## What carries over

Direction 2 ("native kernels + thin Python harness", see `PLAN.md`) keeps everything above the two hot loops:

- `.ilp` parsing, axis handling and feature ordering (`ilp_project.py`, `standalone_pipeline.py`)
- the forest HDF5 decoder (`vigra_rf_reader.py`). The decoded arrays feed the C walker directly.
- the presmoothing and scale composition (`ilp_features.py`)
- all of `dev_validation/`, the acceptance metric, and the reference environment recipe

Only `kernel1d.py` and `nd_filters.py` (the NumPy filter reimplementations) and the NumPy tree walk are replaced. They can stay as a slow fallback.

On 2026-09-24 the prototype produced output identical to real ilastik to within 6e-8, with **0** pixels changing class (the pure-Python version changes 8). The full test image runs in about 1.2–2s, roughly on par with real ilastik's 0.9–1.75s.

## Lessons

1. **Time the thing you are replacing before you start**, and write the speed target as a ratio to it.
2. **Check constraints against the real deployment.** "No compiled dependencies" was a guess about users, not a requirement of the actual consumer.
3. **Keep status docs tied to measurements.** Put the date and machine next to any timing, and re-measure before handing off. The handoff's claim that features dominated the runtime was wrong and would have sent the next speed-up effort to the wrong place.
