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
run_pc_vigra_baseline.py in reference/ for the same convention).

2D-in-3D handling (presmooth and/or filter per z-slice rather than over the
whole volume) has two independent triggers, and ilastik applies them to
different stages:
  - ComputeIn2d[j], stored per scale column in the project, applies to BOTH
    the presmoothing pass and the feature filters.
  - invalid_z, recomputed per filter from the volume's z extent, applies to
    the feature filters ONLY (see filter_forced_to_2d).

WINDOW_SIZE constants match OpPixelFeaturesPresmoothed.WINDOW_SIZE (3.5, for
presmoothing) and OpBaseFilter.window_size_feature (2.0, for the feature
filters themselves).

Only dependency: numpy, scipy (via nd_filters/kernel1d).
"""
import math

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


def feature_scale_args(feature_id: str, new_scale: float):
    """
    The scale arguments OpBaseFilter passes to this filter's filter_fn, in the
    order they appear in compute_feature() below. Only used to decide whether
    the filter fits along z (see filter_forced_to_2d) - the actual values are
    applied in compute_feature().
    """
    if feature_id in ("GaussianSmoothing", "LaplacianOfGaussian", "GaussianGradientMagnitude"):
        return (new_scale,)
    elif feature_id == "DifferenceOfGaussians":
        return (new_scale, new_scale * 0.66)
    elif feature_id == "StructureTensorEigenvalues":
        return (new_scale, new_scale * 0.5)
    elif feature_id == "HessianOfGaussianEigenvalues":
        return (new_scale,)
    else:
        raise ValueError(f"Unknown feature id {feature_id!r}")


def filter_forced_to_2d(feature_id: str, new_scale: float, z_extent) -> bool:
    """
    Reproduces OpBaseFilter.setupOutputs' `invalid_z` (filterOperators.py):

        invalid_z = z_dim == 1 or any(ceil(s * window_size_feature) + 1 > z_dim
                                      for s in filter_scales)

    A volume too thin for the filter's own kernel along z is filtered slice by
    slice instead of failing. Since presmoothing caps new_scale at 1.0, the
    usual threshold is ceil(2.0) + 1 == 3, i.e. 3D filtering needs z >= 3.

    Note this applies to the feature filters ONLY, not to presmoothing, which
    keys off ComputeIn2d alone (opPixelFeaturesPresmoothed.py's
    _computeGaussianSmoothing passes in2d=self.ComputeIn2d.value[j] and never
    consults invalid_z). On a thin-z volume with ComputeIn2d unset, ilastik
    therefore presmooths in 3D and then filters in 2D. That asymmetry looks
    like an oversight, but it is what trained projects were computed with, so
    it has to be reproduced rather than tidied up.

    z_extent: length of the z axis, or None for genuinely 2D data.
    """
    if z_extent is None:
        return False
    if z_extent == 1:
        return True
    return any(math.ceil(s * FEATURE_WINDOW_SIZE) + 1 > z_extent for s in feature_scale_args(feature_id, new_scale))


def _per_slice(volume: np.ndarray, fn):
    """Apply fn to each z slice of a 3D volume and stack the results back up."""
    return np.stack([fn(volume[z]) for z in range(volume.shape[0])], axis=0)


def presmooth(source: np.ndarray, scale: float, in_2d: bool = False) -> np.ndarray:
    """
    Stage one of the two-stage pipeline: smooth the source image at this scale
    column's temp_sigma, with the wider (3.5) presmoothing window.

    Shared by every feature selected at this scale - it depends only on the
    scale column and the input channel, so it is computed once per (scale,
    channel) rather than once per (feature, scale, channel).
    """
    temp_sigma, _ = presmoothing_sigma_and_new_scale(scale)
    if in_2d and source.ndim == 3:
        return _per_slice(source, lambda sl: nf.gaussian_smoothing(sl, temp_sigma, PRESMOOTHING_WINDOW_SIZE))
    return nf.gaussian_smoothing(source, temp_sigma, PRESMOOTHING_WINDOW_SIZE)


def compute_feature_maybe_2d(
    presmoothed: np.ndarray, feature_id: str, new_scale: float, in_2d: bool = False
) -> np.ndarray:
    """
    Stage two, optionally slice by slice. Filtering a 3D volume per z slice
    also drops the eigenvalue-based features from 3 channels to 2, since each
    slice's eigendecomposition is 2x2 - which is exactly what ilastik's
    _n_per_space_axis does for its output channel count.
    """
    if in_2d and presmoothed.ndim == 3:
        return _per_slice(presmoothed, lambda sl: compute_feature(sl, feature_id, new_scale))
    return compute_feature(presmoothed, feature_id, new_scale)


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
