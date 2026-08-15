# Plan: fixing the code-review findings

Seven findings from a review of the `9e79f22..HEAD` diff. This is the
execution plan: what changes, why, in what order, and how each is verified.

**Every fix lands twice.** `pixel_classification_standalone.py` is a hand-made
concatenation of the six module files, with no build step. Each change below
must be applied to the module file *and* to the corresponding section of
`pixel_classification_standalone.py`. A final diff-check pass (step 8)
verifies the two stayed in sync.

## Order of work

Deliberately not in severity order: step 1 unblocks safe validation, and
step 2 is a cheap guard that will catch mistakes made in steps 3-7.

| # | Finding | Kind | Files |
|---|---------|------|-------|
| 1 | Version stub poisons the repo | dev tooling | `dev_validation/compare_against_real_pipeline.py` |
| 2 | No feature-count validation | wrong output | `vigra_rf_reader.py`, `standalone_pipeline.py` |
| 3 | `known_labels` ignored | wrong output | `vigra_rf_reader.py`, `ilp_project.py`, `standalone_pipeline.py` |
| 4 | Presmoothing recomputed per triple | performance | `standalone_pipeline.py`, `ilp_features.py` |
| 5 | `t` axis crashes in transpose | crash | `standalone_pipeline.py` |
| 6 | PNG/JPEG input aborts | crash | CLI section of `pixel_classification_standalone.py` |
| 7 | Thin-z `invalid_z` fallback missing | wrong output | `ilp_features.py`, `standalone_pipeline.py` |
| 8 | — | sync + validation | both copies |

---

## 1. Stop writing `ilastik/_version.py` into the repo

`compare_against_real_pipeline.py:44` writes a permanent
`ilastik/_version.py` containing `version = "0.0.0.dev0"`. It is gitignored
so it never gets committed, but it persists in the working tree, and
`isVersionCompatible` (`ilastik/__init__.py:125`) then computes
`v2 = (0, 0)`, which is not in `compatible_set` and never equals a real
project's `(1, 4)` — so every subsequent load of a real `.ilp` in that
checkout is rejected. It silently breaks the rest of the checkout for
anything other than this one script.

**Fix:** register a `sys.modules["ilastik._version"]` module object instead
of writing a file. `ilastik/__init__.py` does `from ._version import version`,
which resolves out of `sys.modules` when the submodule is pre-registered, so
nothing needs to touch disk. Use a 1.4-series version so `isVersionCompatible`
behaves like a real install, and leave a genuine `_version.py` alone if one
already exists (editable installs have one).

This checkout currently has no stale stub, so no cleanup is needed here — but
the script should also not leave one behind for the next person.

## 2. Validate the feature-column count against the forest

`feature_count` is decoded from the tree topology (`vigra_rf_reader.py:65`)
and then never used. If the computed feature stack has a different column
count than the forest was trained on, the RF walker indexes whatever columns
happen to exist and returns confident, plausible-looking, wrong probabilities.
This is the failure mode every other bug in this list produces, so it goes in
early as a net.

**Fix:**
- `DecodedForest` / `DecodedParallelForest` expose `feature_count`, taken from
  the first tree, asserting all trees agree.
- `get_probabilities` raises a `ValueError` when
  `features.shape[-1] != forest.feature_count`, naming both numbers and
  pointing at the usual causes (channel count, 2D-vs-3D, feature selection
  drift).

## 3. Honor `known_labels` (output channel count and class mapping)

The forest's class count is the number of labels that actually had painted
samples at training time, not the number of labels defined in the project.
ilastik reconciles the two in `classifierOperators.py:497-508`: it allocates
`len(LabelNames)` channels and scatters `probabilities[..., i]` into
`full[..., label - 1]` for each `label` in `classifier.known_classes`.

This code skips that entirely and returns forest-class-count channels. For a
project where every label was painted the two agree, which is why validation
passed — but for a project with an unpainted label the output has too few
channels *and* the surviving classes sit at the wrong indices.

`known_labels` is stored per-project at
`PixelClassification/ClassifierForests/known_labels`
(`parallelVigraRfLazyflowClassifier.py:401`), with the older-project fallback
`list(range(1, labelCount + 1))` (same file, :434-437).

**Fix:**
- `DecodedParallelForest.from_ilp` reads `known_labels` from the group, and
  falls back to `range(1, class_count + 1)` when the dataset is absent, exactly
  as the real deserializer does.
- `get_probabilities` expands the flat probability array to
  `project.label_count` channels via the `label - 1` scatter above, only when
  the counts differ (matching ilastik's "unusual case" branch).
- Guard against a `known_labels` entry that exceeds `label_count` rather than
  writing out of bounds.

## 4. Compute each presmoothed image once per (scale, channel)

`compute_feature_at_scale` (`ilp_features.py:86`) does presmoothing *and* the
feature filter in one call, and `_compute_features` calls it once per
`(feature, scale, channel)` triple. The presmoothing pass is the expensive
one — window size 3.5 versus 2.0 for the feature filters — and it is
identical for every feature in a given scale column. With six features
selected at a scale, it is done six times instead of once.

**Fix:** split the two stages at the call site. Restructure `_compute_features`
to iterate scale-column-outer, presmooth once per `(scale, channel)`, run every
feature selected in that column on the shared result, and stash the outputs in
a dict keyed `(feature_idx, scale_idx, channel)`. Emit blocks at the end in
ilastik's canonical order (feature-major, scale-inner, channel-innermost) via
an explicit index list.

Computing scale-outer but emitting feature-major keeps the output column order
bit-identical while bounding live presmoothed data to one scale column. Peak
memory is unchanged in practice — it is already dominated by the full feature
stack that `np.concatenate` builds.

`ilp_features.compute_feature_at_scale` stays as-is (it is the readable
reference and `dev_validation/validate_ilp_features_against_fastfilters.py`
uses it); the pipeline just stops routing through it.

Expected effect: this is the dominant term in the ~27s feature half of the
~47s runtime, so it is the cheapest large win available — and it should be
done before the `scipy.ndimage` rewrite floated as open item #1 in HANDOFF.md,
since it may make that rewrite unnecessary.

## 5. Reject or drop a `t` axis instead of dying in `np.transpose`

`get_probabilities` counts only `xyz` for the spatial check and `c` for the
channel check, so a `t` dim passes both. `target_dims` then omits it, `perm`
comes out shorter than the array rank, and `np.transpose` fails with
"axes don't match array" — an error that says nothing about the real problem.

**Fix:** after the existing checks, drop any non-`xyzc` axis of length 1
(a singleton `t` is meaningless here and dropping it cannot change results),
and raise a clear `ValueError` naming the offending axes if any non-`xyzc`
axis has length > 1. Time series would need per-frame iteration, which is out
of scope; the error should say so.

## 6. Fall back to imageio when tifffile can't read the file

In `_read_image`, `tifffile.imread(path)` sits inside a `try` whose `except`
only catches `ImportError`. When tifffile is installed but the file is a PNG
or JPEG, `imread` raises `TiffFileError`, which escapes — so the imageio
branch right below it, which would have read the file fine, is never reached.
The failure looks like a corrupt-file error rather than a wrong-backend one.

**Fix:** try each backend in turn, distinguishing "backend not installed"
(`ImportError`) from "backend can't read this file" (any other exception), and
continuing to the next backend in both cases. If all backends fail, raise a
single error listing what was tried and why each failed, keeping the existing
install hint.

## 7. Reproduce ilastik's thin-z 2D fallback

This is the most intricate one and goes last. ilastik degrades to 2D filtering
on volumes with a small z extent, in two separate places with *different*
conditions — the asymmetry is real and has to be preserved:

- **The feature filter** (`filterOperators.py:82-96`) computes
  `invalid_z = z_dim == 1 or any(ceil(s * 2.0) + 1 > z_dim for s in filter_scales)`
  over that filter's own scale arguments, and processes per-z-slice when
  `invalid_z or ComputeIn2d`. Since presmoothing caps `new_scale` at 1.0, the
  common case is `ceil(2.0) + 1 = 3`, i.e. 3D filtering needs `z >= 3`.
- **The presmoothing pass** (`opPixelFeaturesPresmoothed.py:425`) keys off
  `ComputeIn2d[j]` *only*, never `invalid_z`. So on a thin-z volume with
  `ComputeIn2d` false, ilastik presmooths in 3D and then filters in 2D.

There is a third piece: eigenvalue channel counts follow `_n_per_space_axis`
(`filterOperators.py:297`), which counts spatial axes with size > 1 over
`"yx"` in 2D mode and `"zyx"` in 3D mode. So dropping to 2D also drops
Hessian/structure-tensor eigenvalues from 3 channels to 2 — which changes the
feature vector width, which is why step 2's guard matters.

Finally, `opPixelFeaturesPresmoothed.py:109-113`: when the project stores no
`ComputeIn2d` list at all, ilastik fills it from the *input* z at inference
time — all-`True` if `z == 1`, else all-`False`. The current reader defaults
to all-`False` unconditionally, which diverges for `z == 1` input.

**Fix:**
- `ilp_project.py` records whether `ComputeIn2d` was present in the file
  rather than silently defaulting to zeros.
- When it was absent, `get_probabilities` resolves it from the actual input's
  z extent (all-`True` when `z == 1`), matching ilastik.
- `ilp_features.py` gains a helper returning each feature's scale arguments,
  and the pipeline computes per-filter `invalid_z` from them, applying it to
  the feature filter only — never to presmoothing.
- Eigenvalue channel count follows the 2D/3D decision and counts only spatial
  axes with extent > 1.

## 8. Sync check and validation

1. Diff each module file against its section of
   `pixel_classification_standalone.py` and confirm the only remaining
   differences are import style, naming, and docstrings — the same standard
   the review applied.
2. Rebuild the Python 3.11 reference environment per PLAN.md's "Full
   reference-pipeline environment" recipe (~5-10 min; nothing survives a
   container reset) and run
   `dev_validation/compare_against_real_pipeline.py` on
   `notebooks/pixel_classification_api/pc.ilp`. Expect the existing baseline:
   raw probabilities within ~2.9e-8, 12 / 1,376,256 binarized pixels differing.
   Anything worse is a regression introduced here.
3. Re-run `validate_ilp_features_against_fastfilters.py` and
   `validate_rf_reader_against_vigra.py` in the lightweight vigra env, since
   steps 4 and 7 touch feature computation.
4. Record the before/after wall-clock for the full-image run so the step 4
   speedup is a measured number rather than a claim.

### Coverage the current validation can't provide

The end-to-end harness only exercises one 2D, single-channel, all-labels-painted
project, which is precisely why findings 2, 3, 5 and 7 survived it. Add cheap
checks that need no conda env:

- `known_labels` expansion: synthesize the scatter against a fake
  `known_labels` (e.g. `[1, 3]` with 3 label names) and assert channel
  placement, since no unpainted-label `.ilp` is available here.
- Feature-count mismatch raises rather than returning an array.
- A `t`-axis input raises a clear error; a singleton `t` is dropped and gives
  the same answer as the same array without it.
- `_read_image` falls back to imageio for a PNG with tifffile installed.

Finding 7 stays partly unverified end-to-end for the same reason open item #3
in HANDOFF.md is open: no real 3D `.ilp` exists in this repo, and a thin-z one
would be needed to exercise it properly. The parity argument rests on reading
ilastik's source, and that limitation should be stated in the commit rather
than papered over.

## Commits

Roughly one commit per numbered step, each touching both copies, so a
regression can be bisected to a single finding. Push to
`claude/handoff-code-review-mhxgvb`.
