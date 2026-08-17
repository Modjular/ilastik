"""
Pure-NumPy/SciPy reimplementation of the 6 ilastik pixel-classification
feature filters (vigra.filters.* / fastfilters.*), N-dimensional (2D or 3D
spatial axes), single channel per call.

Validated (see dev_validation/validate_filters_against_fastfilters.py) against
real fastfilters/vigra.filters output. Boundary handling: scipy.ndimage mode
'mirror' == vigra's BORDER_TREATMENT_REFLECT (confirmed empirically, exact
match against vigra to float64 precision, ~1e-7 against fastfilters' float32
internals).

2D: matches fastfilters to ~1e-6/1e-7 across the full real ilastik scale set
(0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0) for all 6 filter types. 3D: gaussianSmoothing/
laplacianOfGaussian/gaussianGradientMagnitude match the same way; the
eigenvalue-based filters (hessianOfGaussianEigenvalues, structureTensorEigenvalues)
show ~1e-4 residual differences at small scales on adversarial random-noise
volumes (confirmed via multiset comparison to be genuine tiny numerical
differences, not an eigenvalue-ordering bug - likely float32-vs-float64
accumulation amplified by near-degenerate eigenvalues) - a known, not yet
closed, minor gap for the 3D case.

Only dependency: numpy, scipy.
"""
import numpy as np
from scipy.ndimage import correlate1d

from kernel1d import gaussian_kernel_1d

BORDER_MODE = "mirror"


def _conv_axis(img, sigma, order, window_size, axis):
    k = gaussian_kernel_1d(sigma, order, window_size)
    return correlate1d(img, k, axis=axis, mode=BORDER_MODE)


def _directional_derivative(img, sigma, window_size, orders):
    """Apply gaussian_kernel_1d(sigma, orders[axis]) along each axis, separably."""
    out = img
    for axis, order in enumerate(orders):
        out = _conv_axis(out, sigma, order, window_size, axis)
    return out


def gaussian_smoothing(img: np.ndarray, sigma: float, window_size: float = 2.0) -> np.ndarray:
    ndim = img.ndim
    return _directional_derivative(img, sigma, window_size, orders=(0,) * ndim)


def laplacian_of_gaussian(img: np.ndarray, scale: float, window_size: float = 2.0) -> np.ndarray:
    ndim = img.ndim
    out = np.zeros_like(img, dtype=np.float64)
    for target_axis in range(ndim):
        orders = [2 if a == target_axis else 0 for a in range(ndim)]
        out += _directional_derivative(img, scale, window_size, orders)
    return out


def gaussian_gradient_magnitude(img: np.ndarray, sigma: float, window_size: float = 2.0) -> np.ndarray:
    ndim = img.ndim
    acc = np.zeros_like(img, dtype=np.float64)
    for target_axis in range(ndim):
        orders = [1 if a == target_axis else 0 for a in range(ndim)]
        g = _directional_derivative(img, sigma, window_size, orders)
        acc += g**2
    return np.sqrt(acc)


def difference_of_gaussians(img: np.ndarray, sigma0: float, sigma1: float, window_size: float = 2.0) -> np.ndarray:
    return gaussian_smoothing(img, sigma0, window_size) - gaussian_smoothing(img, sigma1, window_size)


def _sorted_eigvalsh_descending(tensor_components: np.ndarray) -> np.ndarray:
    """
    tensor_components: (..., ndim, ndim) symmetric matrices per pixel.
    Returns (..., ndim) eigenvalues sorted largest-first (vigra convention).
    """
    eigvals = np.linalg.eigvalsh(tensor_components)  # ascending order
    return eigvals[..., ::-1]  # descending


def _structure_tensor_eigenvalues_literal(
    img: np.ndarray, gradient_scale: float, smoothing_scale: float, window_size: float = 2.0
) -> np.ndarray:
    """
    Textbook structure tensor: gradients at `gradient_scale`, tensor components
    smoothed at `smoothing_scale`. Matches vigra.filters.structureTensorEigenvalues
    bit-exactly when called as (image, innerScale, outerScale) i.e.
    gradient_scale=innerScale, smoothing_scale=outerScale.
    """
    ndim = img.ndim
    grads = []
    for axis in range(ndim):
        orders = [1 if a == axis else 0 for a in range(ndim)]
        grads.append(_directional_derivative(img, gradient_scale, window_size, orders))

    tensor = np.zeros(img.shape + (ndim, ndim), dtype=np.float64)
    for i in range(ndim):
        for j in range(ndim):
            comp = grads[i] * grads[j]
            comp = gaussian_smoothing(comp, smoothing_scale, window_size)
            tensor[..., i, j] = comp

    return _sorted_eigvalsh_descending(tensor)


def structure_tensor_eigenvalues(
    img: np.ndarray, inner_scale: float, outer_scale: float, window_size: float = 2.0
) -> np.ndarray:
    """
    Matches fastfilters.structureTensorEigenvalues(image, innerScale, outerScale,
    window_size) as actually called by ilastik (OpStructureTensorEigenvalues /
    OpPixelFeaturesPresmoothed).

    fastfilters has a real parameter-order mismatch between its C header
    (fastfilters_fir_structure_tensor2d(in, sigma_outer, sigma_inner, ...))
    and its C++/Python binding (src/python/core.cxx: ConvolveST calls
    fastfilters_fir_structure_tensor2d(&in, sigma_inner, sigma_outer, ...) —
    i.e. it passes its own `sigma_inner` into the C function's `sigma_outer`
    slot and vice versa). Net effect, confirmed empirically against real
    fastfilters output: the Python-level `innerScale` argument ends up used
    as the tensor-component *smoothing* scale, and `outerScale` ends up used
    as the *gradient* scale — the opposite of the parameter names and of
    vigra.filters.structureTensorEigenvalues' own behavior.

    Since real trained .ilp projects are computed with fastfilters (ilastik
    prefers it when installed - lazyflow/operators/filterOperators.py), this
    swapped behavior is what a standalone implementation must reproduce for
    numerical parity with real projects, not the "textbook"/vigra ordering.
    """
    return _structure_tensor_eigenvalues_literal(img, outer_scale, inner_scale, window_size)


def hessian_of_gaussian_eigenvalues(img: np.ndarray, scale: float, window_size: float = 2.0) -> np.ndarray:
    ndim = img.ndim
    hessian = np.zeros(img.shape + (ndim, ndim), dtype=np.float64)
    for i in range(ndim):
        for j in range(i, ndim):
            orders = [0] * ndim
            orders[i] += 1
            orders[j] += 1
            comp = _directional_derivative(img, scale, window_size, orders)
            hessian[..., i, j] = comp
            hessian[..., j, i] = comp

    return _sorted_eigvalsh_descending(hessian)
