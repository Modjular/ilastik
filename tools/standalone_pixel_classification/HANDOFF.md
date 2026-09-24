# Handoff: integrating native pixel classification into tttk

*Written 2026-09-24. For whoever (agent or person) moves this prototype into `tttk`.*

**PR:** https://github.com/Modjular/ilastik/pull/2, branch `claude/pixel-classification-single-file-w6w64p`, directory `tools/standalone_pixel_classification/`.

## What you're integrating

A way to run a trained ilastik **Pixel Classification** `.ilp` project on a new image, **inside tttk's own Python process**. There's no ilastik subprocess, lazyflow, vigra or conda, and it runs at about ilastik's speed. It is inference only; training stays in the ilastik desktop app.

The pipeline is plain Python except for two small native libraries loaded with `ctypes`:

```
.ilp (h5py) --> project metadata: axes, channels, selected features x scales, label count
            --> forest HDF5 --> decoded tree arrays --------------------------+
image --> per channel, per (feature, scale): presmooth + filter               |
          [fastfilters C library via ff_ctypes] --> feature stack (H,W,F)     |
          --> flatten to (N, F) float32 --> [rfwalk C library via rf_native] <-+
          --> probabilities (N, classes) float32 --> reshape to project axis order + c
```

- **fastfilters** is the exact filter library ilastik uses (MIT-licensed). We don't build it; prebuilt copies are taken from ilastik-forge's conda packages.
- **rfwalk** is our own 74-line C99 file. It walks vigra's random-forest trees with vigra's exact arithmetic.

## Status: what's proven and what isn't

**Proven (all on Linux x86_64, 2026-09-24):**
- `ff_ctypes` output is **bit-identical** to the `fastfilters` Python package for all 5 filters, in 2D and 3D, at several scales and window sizes.
- `rfwalk` output differs by **exactly 0.0** from real `vigra.learning.RandomForest` on 3 `.ilp` fixtures.
- **End to end** on `notebooks/pixel_classification_api/pc.ilp`, full 1024×1344 image, vs real `ilastik.experimental.api`: max abs diff **6e-8**, and **0 / 1,376,256** pixels with a different predicted class.
  - This ran from a plain system Python with only numpy/h5py/imageio, given only the fastfilters library file.
- **Speed:** ~1.2–2s end to end, vs 0.9–1.75s for real ilastik and 45s for the old pure-Python version. Treat these as rough: this container's timings vary up to ~2× run to run.
- **Cross-compiled but not run:** rfwalk.c cross-compiles cleanly for Linux x86_64/arm64, macOS x86_64/arm64 and Windows x86_64 (`zig cc`). The macOS and Windows fastfilters builds were inspected and export the needed API.

**Not yet proven:**
- **Nothing has run on real macOS or Windows.** This is the biggest gap. Close it before relying on those platforms.
- **3D projects end to end.** 3D filters match fastfilters, but no real trained 3D `.ilp` has been run through the whole pipeline.
- **Multi-channel images end to end.** The per-channel loop exists and follows ilastik's ordering, but only single-channel projects were tested.
- **`.ilp` files from other ilastik versions.** Tested fixtures: the three in this repo (`pc.ilp`, `mitocheckPixelClassification.ilp`, `pc-oc-133.ilp`).

## Settle these with the tttk owner first

1. **Does tttk run natively on Windows/macOS, or in Docker on those hosts?**
   - Docker means the container is always Linux. Then the only extra case is linux-aarch64 (ARM images on Apple Silicon), which has **no** prebuilt fastfilters (see below).
   - Native means all four targets in the table below.
2. **How does tttk call ilastik today, and how long does one image take including startup?** That's the baseline to beat and to test against.
3. **Which ilastik version is bundled, and which fastfilters version does it contain?**
   - Look for the library file's version suffix, e.g. `libfastfilters.so.0.3-5-ge484a99` = 0.3.post5.
   - For exact parity, tttk's fastfilters should be the same version as the ilastik that trained the projects. 0.3.post5 is what's validated.
4. **Should tttk keep calling the bundled ilastik as a fallback** (e.g. for Autocontext projects, which this doesn't support)?

## What to port into tttk

| Take | From | Notes |
|---|---|---|
| `ff_ctypes.py` | `native_prototype/` | As is. Loads the library and declares the C signatures. Copies fastfilters' own binding logic (see "Invariants"). |
| `rfwalk.c` | `native_prototype/` | As is. |
| `rf_native.py` | `native_prototype/` | As is. Packs decoded trees into flat arrays and threads over rows. Its `build()` is optional if you ship prebuilt libraries. |
| `.ilp` parsing + pipeline wiring | `pixel_classification_standalone.py` section 5 (`_PixelClassificationProject`, `PixelClassificationPipeline`, axis helpers) | Keep the logic; it's validated. |
| Scale composition | `pixel_classification_standalone.py` section 3 (`_presmoothing_sigma_and_new_scale`) + `feature_at_scale()` in `native_prototype/run_prototype.py` | `feature_at_scale` is the version that calls fastfilters. Use it instead of section 3's scipy-based `_compute_feature`. |
| Forest HDF5 decoding | `pixel_classification_standalone.py` section 4 (`_DecodedTree`, `_DecodedForest.from_h5_group`, `DecodedParallelForest.from_ilp`) | Only the *loading* is needed. The NumPy `predict_*` methods are replaced by `rf_native.NativeForest`. |
| **Don't take** | sections 1–2 (`kernel1d`, `nd_filters`), the CLI, the `scipy` import | These were the pure-Python filters. Dropping them removes **scipy** as a dependency: the single file imports it at module level, so the prototype currently pulls it in indirectly. |

The prototype harness (`run_prototype.py`) **monkeypatches** the single file: it rebinds the `compute_feature_at_scale` module global and replaces `pipeline._project.forest`. That's fine for a prototype, but don't carry it into tttk. Write the module so it calls `feature_at_scale` and `NativeForest` directly.

Suggested shape inside tttk (adapt to tttk conventions):

```
tttk/pixel_classification/
  __init__.py        # PixelClassifier.from_ilp(path) -> .predict(array, dims) -> probabilities
  _project.py        # .ilp parsing + forest HDF5 loading (from the single file, sections 4-5)
  _features.py       # scale composition + feature_at_scale over ff_ctypes
  _ff_ctypes.py
  _rf_native.py
  _native/           # filled by build.py, per platform:
    fastfilters lib  #   libfastfilters.so.* / libfastfilters.*.dylib / fastfilters.dll
    rfwalk lib       #   librfwalk.so / librfwalk.dylib / rfwalk.dll
```

Have the module find its bundled libraries relative to `__file__` and pass the paths explicitly:
- `ff_ctypes.load(path)`
- `NativeForest(forest, lib_path=...)`

Don't depend on `FASTFILTERS_LIB` or the system library search path in production.

## build.py

### 1. fastfilters: fetch, don't compile

Download the pinned ilastik-forge package for each target platform, check its sha256, and extract the one library file. The files below were downloaded and hashed on 2026-09-24.

| Platform | URL | sha256 of the `.conda` file | Library inside the package |
|---|---|---|---|
| linux-64 | https://conda.anaconda.org/ilastik-forge/linux-64/fastfilters-0.3.post5-py312ha43d40f_1.conda | `334955f197675dbc5827467e1d0e1adfa6c0157c0280986d4d28905c0bc35bbd` | `lib/libfastfilters.so.0.3-5-ge484a99` (`lib/libfastfilters.so` is a symlink to it) |
| osx-64 | https://conda.anaconda.org/ilastik-forge/osx-64/fastfilters-0.3.post5-py312hd8be5bc_1.conda | `4a5755476a39191c3c3578a3e7442c685f00d3749ca8c0ffe44563f3b875df2c` | `lib/libfastfilters.0.3-5-ge484a99.dylib` |
| osx-arm64 | https://conda.anaconda.org/ilastik-forge/osx-arm64/fastfilters-0.3.post5-py312hdbcb3e1_1.conda | `c578ad56b1ab3ae6ab2c4d76c68c6c21d445b47a3400e81772b7daf86ab4ffbe` | `lib/libfastfilters.0.3-5-ge484a99.dylib` (ad-hoc signed, min macOS 11) |
| win-64 | https://conda.anaconda.org/ilastik-forge/win-64/fastfilters-0.3.post5-py312h2ccd7c8_1.conda | `f8721f05f9f289c093ce116cd00cd5b1429d08abb4bd416edba4d98645cd2614` | `Library/bin/fastfilters.dll` |

- The `py312` in the package name only applies to the Python binding that's also in the package. The C library doesn't depend on Python, so it works from any Python version.
- **Extracting a `.conda` file:**
  - It's a zip containing `pkg-*.tar.zst`. Use `zipfile`, then `zstandard.ZstdDecompressor().stream_reader(...)`, then `tarfile`.
  - Extract the **real file**, not the symlink (on Linux, skip `lib/libfastfilters.so` and take the versioned `.so.0.3-5-ge484a99`).
  - `zstandard` is a build-time dependency only.
- **Each library needs only system libraries:** glibc / `libSystem` / `VCRUNTIME140` + UCRT.
- **No linux-aarch64 package exists.** If you need that target, either build fastfilters from source (CMake; it supports ARM through bundled SIMDe; source at https://github.com/ilastik/fastfilters, MIT) or run the amd64 image under emulation.

### 2. rfwalk: prebuild all targets from one machine with zig

`pip install ziglang` provides a C cross-compiler for every target, with no system compiler or MSVC needed. From `native_prototype/`:

```bash
ZIG="python -m ziglang cc -O3 -std=c99 -Wall -Wextra -shared -fPIC"
$ZIG -target x86_64-linux-gnu.2.17   -o linux-64/librfwalk.so     rfwalk.c
$ZIG -target aarch64-linux-gnu.2.17  -o linux-aarch64/librfwalk.so rfwalk.c
$ZIG -target x86_64-macos.11.0       -o osx-64/librfwalk.dylib    rfwalk.c
$ZIG -target aarch64-macos.11.0      -o osx-arm64/librfwalk.dylib rfwalk.c
$ZIG -target x86_64-windows-gnu      -o win-64/rfwalk.dll         rfwalk.c
```

- **Pin minimum OS versions** via the target triples, as above. Zig otherwise defaults to macOS 13 and to the build machine's glibc.
- **Verified 2026-09-24** (zig 0.16.0):
  - All five commands build with no warnings.
  - The outputs need glibc ≥ 2.2.5 (x86_64) / 2.17 (aarch64), and macOS ≥ 11.
  - The arm64 dylib is ad-hoc signed (required on Apple Silicon). The x86_64 one isn't, and doesn't need to be.
  - The Windows DLL exports `rf_predict`.
  - The linux-64 output gives results identical to the NumPy reference walker.
- **Don't add `-ffast-math`** or similar flags (see "Invariants").
- **Compiling on the user's machine instead:** `rf_native.build(out_path, cc="python -m ziglang cc")` (or plain `cc`) also works.

## Runtime notes per platform

- **Windows:**
  - `fastfilters.dll` needs `VCRUNTIME140.dll`. python.org and conda Pythons ship it next to `python.exe`, and it's usually in System32 too.
  - Load libraries by **absolute path**.
  - If you add a DLL that depends on another DLL in its own folder, call `os.add_dll_directory()` first. Python ≥ 3.8 no longer uses `PATH` for this.
- **macOS:** load by absolute path. Ship the libraries inside the package; files downloaded through a browser get quarantined by Gatekeeper, but files written by Python code don't.
- **All platforms:**
  - `rf_native` uses `os.cpu_count()` threads by default (`NativeForest(..., n_threads=)`). If tttk already parallelises across images, set this to 1–2 to avoid oversubscription.
  - Feature computation is single-threaded today (~0.6s on the test image). The same thread-pool approach works there, because ctypes releases the GIL.

## API contract (current behaviour to keep)

- **Input:** a numpy array plus `dims` (e.g. `("y","x")`, `("z","y","x","c")`), or an `xarray.DataArray`.
  - The channel count must equal the project's.
  - The number of spatial dims must equal the project's.
  - Both are checked and raise `ValueError`.
- **Output:** float32 probabilities with the **project's original spatial axis order + `c`** last, one channel per label class. It's an `xarray.DataArray` if xarray is installed, else a plain numpy array. Consider always returning numpy in tttk.
- **Supported:** `workflowName == "Pixel Classification"` only. Other workflows (Autocontext, object classification, …) raise `ValueError`.
- **Limits:**
  - ≤ 64 label classes (a fixed array size in `rfwalk.c`; raise it if needed).
  - Unweighted forests only (`predict_weighted_ == 0`, ilastik's default). Weighted forests raise.
- **2D-in-3D:** the project's per-scale "compute in 2D" flags are honoured (features computed slice by slice).

## Invariants: don't "clean these up"

Each looks like a bug or an inefficiency. Each is required to reproduce ilastik exactly:

1. **Structure tensor sigma swap.**
   - `structureTensorEigenvalues(img, inner, outer)` passes `(inner, outer)` into C parameters declared `(sigma_outer, sigma_inner)`.
   - That's fastfilters' own binding bug, and ilastik's output depends on it.
   - ilastik also calls it with `outer = 0.5 * inner`.
2. **3D eigenvalue argument order.** `ev3d(zz, yz, xz, yy, xy, xx, ...)`, as fastfilters' binding does, not the header's declared order.
3. **`fastfilters_init()`** must be called once before any filter call (`ff_ctypes.load()` does it).
4. **Presmoothing.**
   - Each feature at scale `s` is computed as: Gaussian presmooth with window 3.5 at `sqrt(s²-1)` (or at `s` if `s ≤ 1`), then the filter at scale 1.0 (or `s`) with window 2.0.
   - Difference of Gaussians uses `s` and `0.66·s`.
   - See `PLAN.md` phase 3 for the derivation.
5. **Forest arithmetic.**
   - Leaf weights are cast to float32 *before* accumulating.
   - `totalWeight` is accumulated in double, **one class at a time in class order**, then cast to float32 before the divide.
   - Sub-forests are combined in float32 as `sum(prob_f · ntrees_f) / total_trees`.
   - Compile without `-ffast-math`. Changing any of this breaks exact equality with vigra.
6. **Feature order** matters (the forest's columns are positional): feature id, then scale, then channel, as in `_compute_features`.

## Validating in tttk

**Acceptance metric:** don't gate on raw float equality across machines. fastfilters picks AVX/AVX2/FMA code at runtime on x86 and NEON (via SIMDe) on ARM, so the last bits can differ, just as they do for ilastik itself. Gate on:
- **argmax flips:** pixels whose predicted class differs from the reference;
- the reference's own **top1-top2 margin** at each flipped pixel.

A flip passes only where the reference's own margin is ≤ 0.1. `dev_validation/compare_against_real_pipeline.py` implements this.

**Reference outputs:** tttk already ships ilastik, so it can produce references with ilastik's headless mode:

```bash
<ilastik launcher> --headless --project=pc.ilp --export_source="Probabilities" \
    --output_format=numpy --output_filename_format=/tmp/{nickname}_probs.npy image.png
```

- The launcher is `run_ilastik.sh` on Linux, `ilastik.exe` on Windows, and the launcher inside the `.app` on macOS.
- The flags are ilastik's own (`ilastik/app.py`, `ilastik/applets/dataExport/dataExportApplet.py`), but this exact command wasn't run here.
- Check the exported axis order before comparing.

**Suggested tttk tests:**
1. **Per platform in CI** (ubuntu, macos-13 x86_64, macos-14 arm64, windows):
   - load both libraries;
   - run a small cropped image through `pc.ilp`;
   - compare with a committed reference (argmax + margin, a few KB) using the metric above.
2. **Once, where a conda env is available:** `dev_validation/validate_native_prototype.py` (exact-match checks against real vigra/fastfilters). The environment recipe is in `PLAN.md` ("Vigra dev environment"); the full-ilastik env needs `platformdirs` in addition.

## Try the prototype first

From the root of this repo, needs `numpy h5py imageio` and a C compiler (or a prebuilt rfwalk). `--libfastfilters` accepts a library file or an ilastik/conda root directory to search:

```bash
python tools/standalone_pixel_classification/native_prototype/run_prototype.py \
    notebooks/pixel_classification_api/pc.ilp \
    notebooks/pixel_classification_api/2d_cells_apoptotic_1channel.png \
    --libfastfilters /path/to/ilastik-or-conda-root \
    [--reference real_ilastik_probs.npy]
```

## Known rough edges

- **Occasional slow first run** (~3–5s instead of ~1.2s) in the multi-threaded runs here, not reproducible every time. Probably noise from the shared 4-vCPU container; re-measure on a normal machine.
- **`rf_native.NativeForest` compiles `rfwalk.c` on first use** if no library exists next to it. In tttk, ship prebuilt libraries and pass `lib_path`, so nothing compiles at runtime.

## Further reading

- `PLAN.md` → "Direction 2" and "Cross-platform": full design reasoning and measurements. Phases 1–7 cover the reverse-engineering of vigra/fastfilters/the `.ilp` format.
- `POSTMORTEM_2026-09-24_pure_python_single_file.md`: why the pure-Python version was retired.
- `HANDOFF_2026-08-17_pure_python.md`: the previous handoff (validation workflow, file map of the pure-Python modules).
