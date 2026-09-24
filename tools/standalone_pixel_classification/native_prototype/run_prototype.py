"""
Prototype of the "native kernels + thin Python harness" direction (see
PLAN.md, "Direction 2", and POSTMORTEM_2026-09-24_pure_python_single_file.md).

Keeps all of the existing Python wiring from pixel_classification_standalone.py
(.ilp parsing, forest HDF5 decoding, OpPixelFeaturesPresmoothed's two-stage
scale composition, feature ordering, axis handling) and swaps only the two
hot loops for compiled code:

  - feature filters -> fastfilters' C library via ctypes (ff_ctypes.py);
    in tttk, the libfastfilters.so that ships inside the ilastik binary
  - RF inference    -> rfwalk.c via ctypes (rf_native.py), compiled on first
    use (in tttk: by build.py)

Needs only numpy + h5py + imageio in Python, a C compiler, and a
libfastfilters.so. No vigra, lazyflow, pybind11 module or conda.

Usage (from repo root):
    python tools/standalone_pixel_classification/native_prototype/run_prototype.py \\
        notebooks/pixel_classification_api/pc.ilp \\
        notebooks/pixel_classification_api/2d_cells_apoptotic_1channel.png \\
        --libfastfilters /path/to/ilastik-release/lib/libfastfilters.so \\
        [--reference real_probs.npy] [--repeat 3] [-o out.npy]
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pixel_classification_standalone as pcs  # noqa: E402

import ff_ctypes  # noqa: E402
from rf_native import NativeForest  # noqa: E402

WS_FEATURE = 2.0
WS_PRESMOOTH = 3.5


def feature_at_scale(source: np.ndarray, feature_id: str, scale: float) -> np.ndarray:
    """OpPixelFeaturesPresmoothed's composition (same as ilp_features.compute_feature_at_scale),
    with every filter call going to fastfilters' C code."""
    temp_sigma, new_scale = pcs._presmoothing_sigma_and_new_scale(scale)
    presmoothed = ff_ctypes.gaussianSmoothing(source, temp_sigma, window_size=WS_PRESMOOTH)

    if feature_id == "GaussianSmoothing":
        return ff_ctypes.gaussianSmoothing(presmoothed, new_scale, window_size=WS_FEATURE)
    if feature_id == "LaplacianOfGaussian":
        return ff_ctypes.laplacianOfGaussian(presmoothed, new_scale, window_size=WS_FEATURE)
    if feature_id == "GaussianGradientMagnitude":
        return ff_ctypes.gaussianGradientMagnitude(presmoothed, new_scale, window_size=WS_FEATURE)
    if feature_id == "DifferenceOfGaussians":
        return ff_ctypes.gaussianSmoothing(
            presmoothed, new_scale, window_size=WS_FEATURE
        ) - ff_ctypes.gaussianSmoothing(presmoothed, new_scale * 0.66, window_size=WS_FEATURE)
    if feature_id == "StructureTensorEigenvalues":
        return ff_ctypes.structureTensorEigenvalues(presmoothed, new_scale, new_scale * 0.5, window_size=WS_FEATURE)
    if feature_id == "HessianOfGaussianEigenvalues":
        return ff_ctypes.hessianOfGaussianEigenvalues(presmoothed, new_scale, window_size=WS_FEATURE)
    raise ValueError(f"unsupported feature {feature_id!r}")


def load_native_pipeline(ilp_path: str) -> "pcs.PixelClassificationPipeline":
    # The pipeline looks compute_feature_at_scale up as a module global; rebinding
    # it routes every feature through fastfilters without touching the wiring.
    pcs.compute_feature_at_scale = feature_at_scale
    pipeline = pcs.PixelClassificationPipeline.from_ilp_file(ilp_path)
    pipeline._project.forest = NativeForest(pipeline._project.forest)
    return pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ilp_file")
    ap.add_argument("image_file")
    ap.add_argument("--dims", nargs="+", default=None, help="e.g. y x (default: y x for 2D, z y x for 3D)")
    ap.add_argument("--libfastfilters", default=None, help="path to libfastfilters.so (else $FASTFILTERS_LIB)")
    ap.add_argument("--reference", default=None, help=".npy of real ilastik probabilities to compare against")
    ap.add_argument("--repeat", type=int, default=2, help="timed runs (first includes warm-up)")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    ff_ctypes.load(args.libfastfilters)
    image = pcs._read_image(args.image_file)
    dims = tuple(args.dims) if args.dims else ("y", "x") if image.ndim == 2 else ("z", "y", "x")

    pipeline = load_native_pipeline(args.ilp_file)
    compute_features, forest = pipeline._compute_features, pipeline._project.forest
    predict = forest.predict_probabilities
    timings = {}

    def timed_features(*a):
        t = time.perf_counter()
        r = compute_features(*a)
        timings["features"] = time.perf_counter() - t
        return r

    def timed_predict(X):
        t = time.perf_counter()
        r = predict(X)
        timings["rf"] = time.perf_counter() - t
        return r

    pipeline._compute_features, forest.predict_probabilities = timed_features, timed_predict
    for i in range(args.repeat):
        t = time.perf_counter()
        probs = np.asarray(pipeline.get_probabilities(image, dims=dims))
        total = time.perf_counter() - t
        print(f"run {i}: total {total:.2f}s  (features {timings['features']:.2f}s, RF {timings['rf']:.2f}s)")

    if args.reference:
        ref = np.load(args.reference)
        flips = int((probs.argmax(-1) != ref.argmax(-1)).sum())
        print(f"vs reference: max abs diff {np.abs(probs - ref).max():.3e}, argmax flips {flips} / {ref[..., 0].size}")
    if args.output:
        np.save(args.output, probs)


if __name__ == "__main__":
    main()
