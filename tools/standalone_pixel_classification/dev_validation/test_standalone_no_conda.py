"""
Checks that need nothing beyond the deliverable's own dependencies (numpy,
scipy, h5py) - no conda, no vigra, no real ilastik. Run them on every change:

    python dev_validation/test_standalone_no_conda.py

They exist because compare_against_real_pipeline.py, the actual regression
test, can only be run against the one project fixture available here: 2D,
single channel, every label painted. That leaves whole branches unexercised -
label-space expansion, the feature-count guard, axis handling, the thin-z 2D
fallback - which is exactly where a batch of review findings turned up. These
are cheap stand-ins for the cases no fixture covers, not a replacement for
compare_against_real_pipeline.py.

Every check runs against BOTH copies of the code: the modules
(standalone_pipeline, ilp_features, ...) and the hand-assembled
pixel_classification_standalone.py. The two are kept in sync by hand with no
build step, so "it works in one of them" is a real failure mode.
"""
import os
import sys
import traceback

import numpy as np

TOOL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

import ilp_features  # noqa: E402
import pixel_classification_standalone as single  # noqa: E402
import standalone_pipeline  # noqa: E402
from ilp_project import FeatureMatrix, InputDataInfo, PixelClassificationProject  # noqa: E402

ALL_FEATURES = [
    "GaussianSmoothing",
    "LaplacianOfGaussian",
    "GaussianGradientMagnitude",
    "DifferenceOfGaussians",
    "StructureTensorEigenvalues",
    "HessianOfGaussianEigenvalues",
]


class FakeForest:
    """Stands in for DecodedParallelForest: enough surface for the pipeline."""

    def __init__(self, feature_count, known_labels, class_count=None):
        self.feature_count = feature_count
        self.known_labels = list(known_labels)
        self.class_count = class_count if class_count is not None else len(self.known_labels)

    def predict_probabilities(self, X):
        # Deterministic, distinguishable per column: column k gets value k+1.
        return np.tile(np.arange(1, self.class_count + 1, dtype=np.float64), (X.shape[0], 1))


def make_pipeline(module, *, feature_ids, scales, selections, label_names, forest, compute_in_2d=None,
                  axis_order="yxc", num_channels=1, compute_in_2d_from_project=True):
    """Build a pipeline of `module`'s flavour around a hand-made project."""
    n_scales = len(scales)
    if compute_in_2d is None:
        compute_in_2d = np.zeros(n_scales, dtype=bool)

    if module is single:
        fm = single._FeatureMatrix(
            feature_ids=feature_ids,
            scales=scales,
            selections=selections,
            compute_in_2d=compute_in_2d,
            compute_in_2d_from_project=compute_in_2d_from_project,
        )
        project = single._PixelClassificationProject(
            workflow_name="Pixel Classification",
            input_data=single._InputDataInfo(axis_order=axis_order, num_channels=num_channels),
            feature_matrix=fm,
            label_names=label_names,
            forest=forest,
        )
        return single.PixelClassificationPipeline(project)

    fm = FeatureMatrix(
        feature_ids=feature_ids,
        scales=scales,
        selections=selections,
        compute_in_2d=compute_in_2d,
        compute_in_2d_from_project=compute_in_2d_from_project,
    )
    project = PixelClassificationProject(
        workflow_name="Pixel Classification",
        input_data=InputDataInfo(axis_order=axis_order, num_channels=num_channels),
        feature_matrix=fm,
        label_names=label_names,
        forest=forest,
    )
    return standalone_pipeline.PixelClassificationPipeline(project)


###############################################################################
# The presmoothing refactor must not move a single number
###############################################################################


def reference_compute_features(feature_ids, scales, selections, data_zyxc):
    """
    The straightforward version: presmooth + filter together, once per
    (feature, scale, channel), looping feature-major. This is what the pipeline
    did before presmoothing was hoisted out to once per (scale, channel). The
    optimised path has to agree with it exactly, values and column order alike.
    """
    blocks = []
    for i, feature_id in enumerate(feature_ids):
        for j, scale in enumerate(scales):
            if not selections[i, j]:
                continue
            for c in range(data_zyxc.shape[-1]):
                channel_img = data_zyxc[..., c]
                feat = ilp_features.compute_feature_at_scale(channel_img, feature_id, scale)
                if feat.ndim == channel_img.ndim:
                    feat = feat[..., np.newaxis]
                blocks.append(feat)
    return np.concatenate(blocks, axis=-1)


def test_feature_computation_unchanged_by_presmoothing_reuse():
    rng = np.random.default_rng(0)
    scales = [0.3, 1.0, 1.6, 3.5]
    # Deliberately ragged: not every feature at every scale, so the column
    # ordering has gaps the optimised path has to reproduce.
    #
    # Column 0 (scale 0.3) is GaussianSmoothing only, because that is the only
    # filter ilastik offers at 0.3 - OpBaseFilter.minimum_scale is 0.7 and only
    # OpGaussianSmoothing lowers it to 0.3 (filterOperators.py). The others at
    # 0.3 would need an order-1 kernel of radius round(2 * 0.15) == 0, whose
    # moment is zero, so the kernel comes out NaN. Unreachable from a real
    # project, but don't put it in a fixture and call it coverage.
    selections = np.array(
        [
            [True, True, False, True],
            [False, True, True, False],
            [False, False, True, True],
            [False, False, True, True],
            [False, True, False, False],
            [False, True, True, True],
        ]
    )

    for num_channels in (1, 3):
        data = rng.random((24, 20, num_channels)).astype(np.float32)
        expected = reference_compute_features(ALL_FEATURES, scales, selections, data)

        for module in (standalone_pipeline, single):
            n_features = expected.shape[-1]
            pipeline = make_pipeline(
                module,
                feature_ids=ALL_FEATURES,
                scales=scales,
                selections=selections,
                label_names=["a", "b"],
                forest=FakeForest(n_features, [1, 2]),
                num_channels=num_channels,
            )
            got = pipeline._compute_features(data)
            assert got.shape == expected.shape, f"{module.__name__}: {got.shape} != {expected.shape}"
            # Bit-identical: the same convolutions in the same order, just not
            # recomputed. Anything else means the refactor changed the maths.
            assert np.array_equal(got, expected), (
                f"{module.__name__}: features differ from the reference "
                f"(max abs diff {np.abs(got - expected).max()})"
            )


def test_presmoothing_is_computed_once_per_scale_and_channel():
    """The point of the refactor: 6 features at one scale = 1 presmooth, not 6."""
    calls = []
    scales = [1.6, 3.5]
    selections = np.ones((len(ALL_FEATURES), len(scales)), dtype=bool)

    for module in (standalone_pipeline, single):
        presmooth_name = "presmooth" if module is standalone_pipeline else "_presmooth"
        target = ilp_features if module is standalone_pipeline else single
        original = getattr(target, presmooth_name)

        calls.clear()

        def counting(source, scale, in_2d=False, _orig=original):
            calls.append(scale)
            return _orig(source, scale, in_2d=in_2d)

        setattr(target, presmooth_name, counting)
        try:
            pipeline = make_pipeline(
                module,
                feature_ids=ALL_FEATURES,
                scales=scales,
                selections=selections,
                label_names=["a", "b"],
                forest=FakeForest(1, [1, 2]),
            )
            pipeline._compute_features(np.zeros((12, 10, 1), dtype=np.float32))
        finally:
            setattr(target, presmooth_name, original)

        assert len(calls) == len(scales), (
            f"{module.__name__}: presmoothed {len(calls)} times for "
            f"{len(scales)} scale columns x 1 channel; expected {len(scales)}"
        )


###############################################################################
# known_labels: a label that was never painted
###############################################################################


def test_unpainted_label_lands_in_the_right_channel():
    """
    3 labels defined, only 1 and 3 painted. The forest emits 2 columns; the
    output must have 3 channels, with label 3's column at index 2 and an
    all-zero channel where label 2 would be.
    """
    scales = [1.0]
    selections = np.zeros((len(ALL_FEATURES), 1), dtype=bool)
    selections[0, 0] = True  # GaussianSmoothing only -> 1 feature column

    for module in (standalone_pipeline, single):
        pipeline = make_pipeline(
            module,
            feature_ids=ALL_FEATURES,
            scales=scales,
            selections=selections,
            label_names=["one", "two", "three"],
            forest=FakeForest(1, known_labels=[1, 3]),
        )
        probs = pipeline.get_probabilities(np.zeros((6, 5), dtype=np.float32), dims=("y", "x"))
        probs = np.asarray(probs)

        assert probs.shape == (6, 5, 3), f"{module.__name__}: expected 3 label channels, got {probs.shape}"
        assert np.all(probs[..., 0] == 1.0), f"{module.__name__}: label 1 lost its column"
        assert np.all(probs[..., 1] == 0.0), f"{module.__name__}: unpainted label 2 should be all zeros"
        assert np.all(probs[..., 2] == 2.0), f"{module.__name__}: label 3 should be in channel 2, not channel 1"


def test_all_labels_painted_is_untouched():
    """The common case must not be perturbed by the expansion path."""
    scales = [1.0]
    selections = np.zeros((len(ALL_FEATURES), 1), dtype=bool)
    selections[0, 0] = True

    for module in (standalone_pipeline, single):
        pipeline = make_pipeline(
            module,
            feature_ids=ALL_FEATURES,
            scales=scales,
            selections=selections,
            label_names=["one", "two"],
            forest=FakeForest(1, known_labels=[1, 2]),
        )
        probs = np.asarray(pipeline.get_probabilities(np.zeros((4, 4), dtype=np.float32), dims=("y", "x")))
        assert probs.shape == (4, 4, 2), f"{module.__name__}: {probs.shape}"
        assert np.all(probs[..., 0] == 1.0) and np.all(probs[..., 1] == 2.0), module.__name__


###############################################################################
# Guards
###############################################################################


def expect_raises(fn, needle, what):
    try:
        fn()
    except ValueError as e:
        assert needle.lower() in str(e).lower(), f"{what}: raised, but message was {str(e)!r}"
        return
    raise AssertionError(f"{what}: expected a ValueError, got none")


def test_feature_count_mismatch_is_caught():
    scales = [1.0]
    selections = np.zeros((len(ALL_FEATURES), 1), dtype=bool)
    selections[0, 0] = True  # produces 1 feature column

    for module in (standalone_pipeline, single):
        pipeline = make_pipeline(
            module,
            feature_ids=ALL_FEATURES,
            scales=scales,
            selections=selections,
            label_names=["a", "b"],
            forest=FakeForest(49, [1, 2]),  # but the forest wants 49
        )
        expect_raises(
            lambda p=pipeline: p.get_probabilities(np.zeros((4, 4), dtype=np.float32), dims=("y", "x")),
            "feature columns",
            f"{module.__name__}: feature count mismatch",
        )


def test_singleton_t_axis_is_dropped_and_long_t_is_rejected():
    scales = [1.0]
    selections = np.zeros((len(ALL_FEATURES), 1), dtype=bool)
    selections[0, 0] = True

    for module in (standalone_pipeline, single):
        pipeline = make_pipeline(
            module,
            feature_ids=ALL_FEATURES,
            scales=scales,
            selections=selections,
            label_names=["a", "b"],
            forest=FakeForest(1, [1, 2]),
        )

        rng = np.random.default_rng(1)
        plain = rng.random((6, 5)).astype(np.float32)

        with_t = plain[np.newaxis, ...]
        got = np.asarray(pipeline.get_probabilities(with_t, dims=("t", "y", "x")))
        expected = np.asarray(pipeline.get_probabilities(plain, dims=("y", "x")))
        assert np.array_equal(got, expected), f"{module.__name__}: singleton t changed the result"

        expect_raises(
            lambda p=pipeline: p.get_probabilities(
                np.zeros((3, 6, 5), dtype=np.float32), dims=("t", "y", "x")
            ),
            "time series",
            f"{module.__name__}: multi-frame t",
        )


###############################################################################
# Thin-z 2D fallback (see REVIEW_FIXES.md - not verifiable end-to-end here)
###############################################################################


def test_thin_z_forces_2d_filtering():
    """
    A z=2 volume is too thin for a 3D filter at new_scale 1.0
    (ceil(1.0 * 2.0) + 1 = 3 > 2), so ilastik filters it slice by slice, which
    also means eigenvalue features yield 2 channels rather than 3.
    """
    assert ilp_features.filter_forced_to_2d("HessianOfGaussianEigenvalues", 1.0, 2) is True
    assert ilp_features.filter_forced_to_2d("HessianOfGaussianEigenvalues", 1.0, 3) is False
    assert ilp_features.filter_forced_to_2d("HessianOfGaussianEigenvalues", 1.0, 1) is True
    assert ilp_features.filter_forced_to_2d("HessianOfGaussianEigenvalues", 1.0, None) is False
    # DoG's second scale is smaller, but the largest one decides.
    assert ilp_features.filter_forced_to_2d("DifferenceOfGaussians", 0.3, 2) is False
    for z, expected in ((2, True), (3, False)):
        assert single._filter_forced_to_2d("HessianOfGaussianEigenvalues", 1.0, z) is expected, z

    scales = [1.6]
    selections = np.zeros((len(ALL_FEATURES), 1), dtype=bool)
    selections[5, 0] = True  # HessianOfGaussianEigenvalues

    for module in (standalone_pipeline, single):
        for z, expected_channels in ((2, 2), (4, 3)):
            pipeline = make_pipeline(
                module,
                feature_ids=ALL_FEATURES,
                scales=scales,
                selections=selections,
                label_names=["a", "b"],
                forest=FakeForest(expected_channels, [1, 2]),
                axis_order="zyxc",
            )
            feats = pipeline._compute_features(np.zeros((z, 8, 7, 1), dtype=np.float32))
            assert feats.shape[-1] == expected_channels, (
                f"{module.__name__}: z={z} gave {feats.shape[-1]} eigenvalue channels, "
                f"expected {expected_channels}"
            )


def test_unset_compute_in_2d_follows_input_z():
    """A project with no stored ComputeIn2d gets it from the input's z extent."""
    scales = [1.0, 3.5]
    selections = np.ones((len(ALL_FEATURES), 2), dtype=bool)

    for module in (standalone_pipeline, single):
        pipeline = make_pipeline(
            module,
            feature_ids=ALL_FEATURES,
            scales=scales,
            selections=selections,
            label_names=["a", "b"],
            forest=FakeForest(1, [1, 2]),
            axis_order="zyxc",
            compute_in_2d_from_project=False,
        )
        assert bool(pipeline._resolve_compute_in_2d(1).all()) is True, module.__name__
        assert bool(pipeline._resolve_compute_in_2d(10).any()) is False, module.__name__


###############################################################################
# CLI image reading
###############################################################################


def test_image_reader_falls_back_past_a_wrong_format_backend():
    """
    tifffile installed but handed a PNG must fall through to imageio, not blow
    up. Simulated with stand-in readers so the test needs neither package.
    """
    original_tif, original_iio = single._read_tifffile, single._read_imageio
    sentinel = np.zeros((2, 2), dtype=np.float32)

    def tif_cannot_read(path):
        raise ValueError("not a TIFF file")

    try:
        single._read_tifffile = tif_cannot_read
        single._read_imageio = lambda path: sentinel
        assert single._read_image("image.png") is sentinel, "did not fall back to imageio"

        # And when nothing can read it, the error names everything tried.
        def iio_missing(path):
            raise ImportError("no imageio")

        single._read_imageio = iio_missing
        try:
            single._read_image("image.png")
        except RuntimeError as e:
            msg = str(e)
            assert "tifffile" in msg and "imageio" in msg, f"unhelpful error: {msg}"
            assert "not a TIFF file" in msg, f"error dropped the underlying reason: {msg}"
        else:
            raise AssertionError("expected a RuntimeError when every backend fails")
    finally:
        single._read_tifffile, single._read_imageio = original_tif, original_iio


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = []
    for test in tests:
        try:
            test()
        except Exception:
            failures.append(test.__name__)
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
        else:
            print(f"ok   {test.__name__}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
