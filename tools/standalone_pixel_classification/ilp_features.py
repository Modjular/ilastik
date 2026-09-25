"""
Reproduces the exact feature-computation pipeline of
lazyflow/operators/opPixelFeaturesPresmoothed.py (OpPixelFeaturesPresmoothed),
using nd_filters.py's pure-NumPy/SciPy filters instead of vigra/fastfilters.

ilastik doesn't compute each selected (feature, scale) pair independently.
For performance, it "presmooths" the source image once per unique scale
column, then runs cheap small-window filters on top of that shared
presmoothed image. This changes the actual numbers (it's not just a speed
trick), so a standalone reimplementation has to replicate it exactly, not
just call the 6 filters directly at their nominal scale. Specifically, for
scale `s` in FeatureSelections/Scales:

    if s > 1.0:
        temp_sigma = sqrt(s**2 - 1.0)
        new_scale  = 1.0
    else:
        temp_sigma = s
        new_scale  = s

    presmoothed = gaussian_smoothing(source, temp_sigma, window_size=3.5)
    # then every feature selected at this scale column is computed on
    # `presmoothed` using `new_scale` (not `s`) and window_size=2.0.

Per-feature scale composition (from OpPixelFeaturesPresmoothed.setupOutputs):
    GaussianSmoothing:            sigma=new_scale
    LaplacianOfGaussian:          scale=new_scale
    GaussianGradientMagnitude:    sigma=new_scale
    DifferenceOfGaussians:        sigma0=new_scale, sigma1=new_scale*0.66
    StructureTensorEigenvalues:   innerScale=new_scale, outerScale=new_scale*0.5
    HessianOfGaussianEigenvalues: scale=new_scale

Each input channel is processed independently and results concatenated (see
run_pc_vigra_baseline.py in reference/ for the same convention). 2D-in-3D
(ComputeIn2d) handling: presmooth and filter per z-slice instead of as a full
3D volume when requested for that scale column - not yet implemented here
(only used for anisotropic 3D data with a thin z-extent).

WINDOW_SIZE constants match OpPixelFeaturesPresmoothed.WINDOW_SIZE (3.5, for
presmoothing) and OpBaseFilter.window_size_feature (2.0, for the feature
filters themselves).

Only dependency: numpy, scipy (via nd_filters/kernel1d).
"""
import numpy as np

import nd_filters as nf

PRESMOOTHING_WINDOW_SIZE = 3.5
FEATURE_WINDOW_SIZE = 2.0


def presmoothing_sigma_and_new_scale(scale: float):
    if scale > 1.0:
        temp_sigma = (scale**2 - 1.0) ** 0.5
        new_scale = 1.0
    else:
        temp_sigma = scale
        new_scale = scale
    return temp_sigma, new_scale


def compute_feature(presmoothed: np.ndarray, feature_id: str, new_scale: float) -> np.ndarray:
    """
    presmoothed: single-channel 2D or 3D array, already smoothed by
    presmoothing_sigma_and_new_scale(scale)[0] via nd_filters.gaussian_smoothing.
    Returns an array of shape presmoothed.shape (+ (k,) for multi-channel features).
    """
    ws = FEATURE_WINDOW_SIZE
    if feature_id == "GaussianSmoothing":
        return nf.gaussian_smoothing(presmoothed, new_scale, ws)
    elif feature_id == "LaplacianOfGaussian":
        return nf.laplacian_of_gaussian(presmoothed, new_scale, ws)
    elif feature_id == "GaussianGradientMagnitude":
        return nf.gaussian_gradient_magnitude(presmoothed, new_scale, ws)
    elif feature_id == "DifferenceOfGaussians":
        return nf.difference_of_gaussians(presmoothed, new_scale, new_scale * 0.66, ws)
    elif feature_id == "StructureTensorEigenvalues":
        return nf.structure_tensor_eigenvalues(presmoothed, new_scale, new_scale * 0.5, ws)
    elif feature_id == "HessianOfGaussianEigenvalues":
        return nf.hessian_of_gaussian_eigenvalues(presmoothed, new_scale, ws)
    else:
        raise ValueError(f"Unknown feature id {feature_id!r}")


def compute_feature_at_scale(source: np.ndarray, feature_id: str, scale: float) -> np.ndarray:
    """
    source: single-channel 2D or 3D array (one input channel).
    Full two-stage pipeline: presmooth at this scale's temp_sigma, then run
    the requested feature filter at new_scale on the presmoothed image.
    """
    temp_sigma, new_scale = presmoothing_sigma_and_new_scale(scale)
    presmoothed = nf.gaussian_smoothing(source, temp_sigma, PRESMOOTHING_WINDOW_SIZE)
    return compute_feature(presmoothed, feature_id, new_scale)
