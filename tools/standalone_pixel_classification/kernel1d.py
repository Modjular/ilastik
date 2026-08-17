"""
Pure-NumPy reimplementation of vigra.filters.Kernel1D.initGaussian /
initGaussianDerivative, derived from vigra source
(<env>/include/vigra/separableconvolution.hxx):

- radius = round(window_size * sigma)
- order 0: discretize the Gaussian pdf at integer offsets in [-radius, radius],
  then rescale so the values sum to 1 (norm=1).
- order >= 1: discretize the analytic Gaussian derivative at integer offsets,
  then rescale by norm / moment, where
  moment = sum_i [ (-i)^order * raw[i] ] / order!
  so that the corrected kernel exactly satisfies vigra's moment condition
  (norm=1), correcting for windowing/truncation error.
"""
import math
import numpy as np


def _gaussian_derivative_analytic(x, sigma, order):
    """vigra's Gaussian functor: n-th derivative of the Gaussian pdf."""
    g0 = np.exp(-0.5 * (x / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)
    if order == 0:
        return g0
    elif order == 1:
        return -(x / sigma**2) * g0
    elif order == 2:
        return ((x**2 - sigma**2) / sigma**4) * g0
    else:
        raise NotImplementedError(order)


def gaussian_kernel_1d(sigma: float, order: int, window_size: float) -> np.ndarray:
    radius = int(round(window_size * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    raw = _gaussian_derivative_analytic(x, sigma, order)
    if order == 0:
        return raw / raw.sum()
    # For order >= 1, vigra first zero-means the raw discretized derivative
    # (removes truncation-induced DC offset), then rescales so the discrete
    # moment matches the analytic derivative exactly (norm=1).
    raw = raw - raw.mean()
    moment = np.sum(((-x) ** order) * raw) / math.factorial(order)
    return raw / moment
