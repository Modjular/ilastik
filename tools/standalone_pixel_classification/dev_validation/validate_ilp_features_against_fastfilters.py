"""
Dev-only validation script. Requires real `fastfilters` (conda env) — NOT part
of the pip-only standalone deliverable.

Validates ilp_features.py's two-stage "presmoothing" composition (which is
what OpPixelFeaturesPresmoothed actually does per-scale, not just a single
filter call at the nominal scale) against the same composition built directly
from real fastfilters calls, mirroring
lazyflow/operators/opPixelFeaturesPresmoothed.py's algorithm:

    temp_sigma, new_scale = (sqrt(s**2-1), 1.0) if s > 1 else (s, s)
    presmoothed = fastfilters.gaussianSmoothing(source, temp_sigma, window_size=3.5)
    feature_value = <feature filter>(presmoothed, new_scale, ..., window_size=2.0)

Usage (from repo root, in a conda env with fastfilters):
    python tools/standalone_pixel_classification/dev_validation/validate_ilp_features_against_fastfilters.py
"""
import os
import sys

import numpy as np
import fastfilters

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ilp_features as ilpf

WS_FEATURE = 2.0
WS_PRESMOOTH = 3.5

FEATURE_IDS = [
    "GaussianSmoothing",
    "LaplacianOfGaussian",
    "GaussianGradientMagnitude",
    "DifferenceOfGaussians",
    "StructureTensorEigenvalues",
    "HessianOfGaussianEigenvalues",
]


def real_ff_feature_at_scale(source: np.ndarray, feature_id: str, scale: float) -> np.ndarray:
    """Same two-stage composition as ilp_features.compute_feature_at_scale, but
    calling real fastfilters at every step - the ground truth to compare against."""
    if scale > 1.0:
        temp_sigma, new_scale = (scale**2 - 1.0) ** 0.5, 1.0
    else:
        temp_sigma, new_scale = scale, scale

    presmoothed = fastfilters.gaussianSmoothing(source, temp_sigma, window_size=WS_PRESMOOTH)

    if feature_id == "GaussianSmoothing":
        return fastfilters.gaussianSmoothing(presmoothed, new_scale, window_size=WS_FEATURE)
    elif feature_id == "LaplacianOfGaussian":
        return fastfilters.laplacianOfGaussian(presmoothed, new_scale, window_size=WS_FEATURE)
    elif feature_id == "GaussianGradientMagnitude":
        return fastfilters.gaussianGradientMagnitude(presmoothed, new_scale, window_size=WS_FEATURE)
    elif feature_id == "DifferenceOfGaussians":
        return fastfilters.gaussianSmoothing(
            presmoothed, new_scale, window_size=WS_FEATURE
        ) - fastfilters.gaussianSmoothing(presmoothed, new_scale * 0.66, window_size=WS_FEATURE)
    elif feature_id == "StructureTensorEigenvalues":
        return fastfilters.structureTensorEigenvalues(
            presmoothed, new_scale, new_scale * 0.5, window_size=WS_FEATURE
        )
    elif feature_id == "HessianOfGaussianEigenvalues":
        return fastfilters.hessianOfGaussianEigenvalues(presmoothed, new_scale, window_size=WS_FEATURE)
    else:
        raise ValueError(feature_id)


def check(results, name, mine, ff_out, atol=2e-5, rtol=1e-4):
    mine = np.asarray(mine, dtype=np.float64)
    ff_out = np.asarray(ff_out, dtype=np.float64)
    ok = mine.shape == ff_out.shape and np.allclose(mine, ff_out, atol=atol, rtol=rtol)
    diff = np.max(np.abs(mine - ff_out)) if mine.shape == ff_out.shape else float("nan")
    print(f"{name:55s} maxdiff={diff:.3e} {'OK' if ok else 'MISMATCH'}")
    results.append(ok)


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    img = rng.standard_normal((70, 90)).astype(np.float32)

    results = []
    for scale in [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0]:
        for feature_id in FEATURE_IDS:
            if feature_id != "GaussianSmoothing" and scale < 0.7:
                continue  # not a real, GUI-selectable combination
            check(
                results,
                f"{feature_id} scale={scale}",
                ilpf.compute_feature_at_scale(img, feature_id, scale),
                real_ff_feature_at_scale(img, feature_id, scale),
            )

    print()
    print("ALL OK" if all(results) else f"{results.count(False)} / {len(results)} FAILED")
    sys.exit(0 if all(results) else 1)
