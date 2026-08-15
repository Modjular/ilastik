###############################################################################
# ilastik Pixel Classification - standalone single-file inference
#
# Loads a REAL, already-trained ilastik Pixel Classification .ilp project
# (produced by the ilastik desktop app) and runs inference on a new image,
# reproducing ilastik.experimental.api.PixelClassificationPipeline's output
# WITHOUT needing vigra, fastfilters, lazyflow, or PyQt installed.
#
# Only dependencies: numpy, scipy, h5py (all pip-installable). xarray,
# tifffile and imageio are optional (xarray only changes how output is
# labeled; tifffile/imageio are only needed by the CLI at the bottom, to
# read image files).
#
#   pip install numpy scipy h5py
#
# Library usage (mirrors the real ilastik.experimental.api.PixelClassificationPipeline):
#
#     from pixel_classification_standalone import PixelClassificationPipeline
#     pipeline = PixelClassificationPipeline.from_ilp_file("project.ilp")
#     prob_maps = pipeline.get_probabilities(image_array, dims=("y", "x"))
#
# CLI usage:
#
#     python pixel_classification_standalone.py project.ilp image.tif -o probabilities.npy
#
# ---
#
# This file assembles four independently-developed and independently-
# validated pieces (see tools/standalone_pixel_classification/ in the
# ilastik repo, where they live as separate modules with their own dev
# validation scripts against real vigra/fastfilters/ilastik output):
#
#   1. kernel1d        - vigra's exact Gaussian/Gaussian-derivative kernel
#                         construction (radius formula + moment-correction
#                         recipe), reverse engineered from vigra's own C++
#                         source and validated to machine precision against
#                         real vigra.filters.Kernel1D.
#   2. nd_filters       - the 6 ilastik feature filters (GaussianSmoothing,
#                         LaplacianOfGaussian, GaussianGradientMagnitude,
#                         DifferenceOfGaussians, StructureTensorEigenvalues,
#                         HessianOfGaussianEigenvalues), built on kernel1d +
#                         scipy.ndimage.correlate1d + numpy.linalg.eigvalsh,
#                         validated against real fastfilters output across
#                         every scale ilastik ships (2D: ~1e-6/1e-7 match
#                         throughout; 3D: same for the non-eigenvalue
#                         filters, ~1e-4 residual for the eigenvalue-based
#                         ones at small scales on adversarial noise - see
#                         nd_filters section below for why).
#   3. ilp_features     - OpPixelFeaturesPresmoothed's two-stage
#                         "presmoothing" scale composition (this changes the
#                         actual feature values, it's not just a perf
#                         trick), validated end-to-end against the same
#                         composition built from real fastfilters calls.
#   4. vigra_rf_reader  - a pure-NumPy decoder for vigra's RandomForest HDF5
#                         format (the ClassifierForests group), reverse
#                         engineered from vigra's own C++ headers and
#                         validated against real
#                         vigra.learning.RandomForest.predictProbabilities()
#                         on multiple independently trained .ilp files.
#
# End-to-end (this file, assembled) validated against the real
# ilastik.experimental.api.PixelClassificationPipeline on a real trained
# project: max abs diff ~2.9e-8 in raw probabilities (float64
# accumulation-order noise), zero mismatches in binarized (argmax)
# predictions.
###############################################################################

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.ndimage import correlate1d

try:
    import xarray

    _HAVE_XARRAY = True
except ImportError:
    _HAVE_XARRAY = False


###############################################################################
# 1. kernel1d - vigra's exact Gaussian / Gaussian-derivative kernel
#
# Derived from vigra source (<conda env>/include/vigra/separableconvolution.hxx):
#   - radius = round(window_size * sigma)
#   - order 0: discretize the Gaussian pdf at integer offsets in
#     [-radius, radius], rescale so it sums to 1.
#   - order >= 1: discretize the analytic Gaussian derivative, zero-mean it
#     (removes truncation-induced DC offset), then rescale so its discrete
#     moment matches the analytic derivative exactly (norm=1).
# Validated to machine precision (~1e-16) against real vigra.filters.Kernel1D
# across every (sigma, order, window_size) combination ilastik actually uses.
###############################################################################


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
    raw = raw - raw.mean()
    moment = np.sum(((-x) ** order) * raw) / math.factorial(order)
    return raw / moment


###############################################################################
# 2. nd_filters - the 6 ilastik feature filters, N-dimensional (2D or 3D).
#
# Boundary handling: scipy.ndimage mode 'mirror' == vigra's
# BORDER_TREATMENT_REFLECT (confirmed empirically).
#
# structure_tensor_eigenvalues() deliberately reproduces a real bug in
# fastfilters itself: its Python structureTensorEigenvalues(image,
# innerScale, outerScale, window_size) has a parameter-order mismatch
# between its C header (fastfilters_fir_structure_tensor2d(in, sigma_outer,
# sigma_inner, ...)) and its C++/Python binding (src/python/core.cxx:
# ConvolveST passes its own sigma_inner into the C function's sigma_outer
# slot and vice versa). Net effect, confirmed empirically against real
# fastfilters output: the Python-level innerScale argument ends up used as
# the tensor-smoothing scale, and outerScale ends up used as the gradient
# scale - backwards from the parameter names and from
# vigra.filters.structureTensorEigenvalues' own behavior. Since real trained
# .ilp projects are computed with fastfilters (ilastik prefers it when
# installed), this file reproduces that swap for numerical parity with real
# projects, not the "textbook"/vigra ordering.
#
# The eigenvalue-based filters (hessianOfGaussianEigenvalues,
# structureTensorEigenvalues) can show up to ~1e-4 differences from real
# fastfilters output at small scales, specifically at pixels where two
# eigenvalues are nearly degenerate (a near-isotropic local Hessian/
# structure tensor). This is NOT a precision/dtype issue in this
# implementation (confirmed: switching to a full float64 pipeline doesn't
# close the gap) - it's inherent numerical ill-conditioning of eigenvalue
# decomposition near degeneracy: the raw tensor components match vigra to
# the same ~1e-6/1e-7 precision as every other filter, but a ~1e-6
# difference in the matrix entries (unavoidable when comparing two
# independently-written convolution implementations that sum in a different
# order) gets amplified by roughly 1/eigengap into a larger eigenvalue
# difference. It shows up more in 3D than 2D (more sequential convolution
# passes -> more chances for a near-degenerate case) and is much rarer on
# real, spatially-smooth microscopy images than on the adversarial random-
# noise test data used to characterize it.
###############################################################################

_BORDER_MODE = "mirror"


def _conv_axis(img, sigma, order, window_size, axis):
    k = gaussian_kernel_1d(sigma, order, window_size)
    return correlate1d(img, k, axis=axis, mode=_BORDER_MODE)


def _directional_derivative(img, sigma, window_size, orders):
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
    eigvals = np.linalg.eigvalsh(tensor_components)  # ascending
    return eigvals[..., ::-1]  # descending (vigra convention)


def _structure_tensor_eigenvalues_literal(
    img: np.ndarray, gradient_scale: float, smoothing_scale: float, window_size: float = 2.0
) -> np.ndarray:
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
    """Matches fastfilters.structureTensorEigenvalues(image, innerScale, outerScale,
    window_size) as actually called by ilastik - including its inner/outer swap bug."""
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


###############################################################################
# 3. ilp_features - OpPixelFeaturesPresmoothed's two-stage "presmoothing"
# scale composition. ilastik doesn't call each feature filter at its nominal
# scale - for performance it presmooths the source once per unique scale
# column, then runs cheap small-window filters on top of that shared
# presmoothed image. This changes the actual numbers, so it has to be
# replicated exactly:
#
#     temp_sigma, new_scale = (sqrt(s**2-1), 1.0) if s > 1 else (s, s)
#     presmoothed = gaussian_smoothing(source, temp_sigma, window_size=3.5)
#     feature_value = <feature filter>(presmoothed, new_scale, ..., window_size=2.0)
#
# (For s <= 1 this really does mean smoothing twice at the same scale before
# running the feature filter - confirmed against source, not a bug here.)
###############################################################################

_PRESMOOTHING_WINDOW_SIZE = 3.5
_FEATURE_WINDOW_SIZE = 2.0


def _presmoothing_sigma_and_new_scale(scale: float):
    if scale > 1.0:
        return (scale**2 - 1.0) ** 0.5, 1.0
    return scale, scale


def _compute_feature(presmoothed: np.ndarray, feature_id: str, new_scale: float) -> np.ndarray:
    ws = _FEATURE_WINDOW_SIZE
    if feature_id == "GaussianSmoothing":
        return gaussian_smoothing(presmoothed, new_scale, ws)
    elif feature_id == "LaplacianOfGaussian":
        return laplacian_of_gaussian(presmoothed, new_scale, ws)
    elif feature_id == "GaussianGradientMagnitude":
        return gaussian_gradient_magnitude(presmoothed, new_scale, ws)
    elif feature_id == "DifferenceOfGaussians":
        return difference_of_gaussians(presmoothed, new_scale, new_scale * 0.66, ws)
    elif feature_id == "StructureTensorEigenvalues":
        return structure_tensor_eigenvalues(presmoothed, new_scale, new_scale * 0.5, ws)
    elif feature_id == "HessianOfGaussianEigenvalues":
        return hessian_of_gaussian_eigenvalues(presmoothed, new_scale, ws)
    else:
        raise ValueError(f"Unknown feature id {feature_id!r}")


def compute_feature_at_scale(source: np.ndarray, feature_id: str, scale: float) -> np.ndarray:
    """source: single-channel 2D or 3D array (one input channel)."""
    temp_sigma, new_scale = _presmoothing_sigma_and_new_scale(scale)
    presmoothed = gaussian_smoothing(source, temp_sigma, _PRESMOOTHING_WINDOW_SIZE)
    return _compute_feature(presmoothed, feature_id, new_scale)


###############################################################################
# 4. vigra_rf_reader - pure-NumPy decoder for vigra's RandomForest HDF5
# format, derived from vigra's own C++ headers
# (random_forest/rf_nodeproxy.hxx, random_forest/rf_decisionTree.hxx,
# random_forest.hxx):
#
# PixelClassification/ClassifierForests holds N Forest000N sub-forests
# (ilastik's ParallelVigraRfLazyflowClassifier splits training across
# cores). Each Forest000N/Tree_NN group has two flat arrays:
#   topology:   int32,   [featureCount, classCount, <node data...>]
#   parameters: float64, [<node data...>]
#
# Traversal starts at topology index 2 (root). At index i:
#   typeID = topology[i]
#   if typeID & LeafNodeTag (0x40000000): LEAF (e_ConstProbNode)
#       addr = topology[i+1]
#       probs = parameters[addr+1 : addr+1+classCount]
#   else: i_ThresholdNode (typeID == 0), the only split type ilastik's
#         default RF training produces (axis-aligned)
#       addr = topology[i+1]
#       child0, child1, column = topology[i+2], topology[i+3], topology[i+4]
#       threshold = parameters[addr+1]
#       next_index = child0 if feature[column] < threshold else child1
#
# Per-forest probability = mean of all its trees' leaf probability vectors.
# Across sub-forests: tree-count-weighted average (matches
# lazyflow/classifiers/parallelVigraRfLazyflowClassifier.py's
# predict_probabilities exactly).
#
# Validated against real vigra.learning.RandomForest.predictProbabilities()
# on 3 independently trained .ilp fixtures - max abs diff ~2.9e-8.
###############################################################################

_LEAF_NODE_TAG = 0x40000000


class _DecodedTree:
    def __init__(self, topology: np.ndarray, parameters: np.ndarray):
        self.topology = topology
        self.parameters = parameters
        self.feature_count = int(topology[0])
        self.class_count = int(topology[1])

    def predict_row(self, feature_row: np.ndarray) -> np.ndarray:
        topo = self.topology
        params = self.parameters
        index = 2
        while True:
            type_id = topo[index]
            if type_id & _LEAF_NODE_TAG:
                addr = topo[index + 1]
                return params[addr + 1 : addr + 1 + self.class_count]
            addr = topo[index + 1]
            child0 = topo[index + 2]
            child1 = topo[index + 3]
            column = topo[index + 4]
            threshold = params[addr + 1]
            index = child0 if feature_row[column] < threshold else child1


class _DecodedForest:
    """One 'Forest0000'-style sub-forest: a list of trees."""

    def __init__(self, trees):
        self.trees = trees

    @classmethod
    def from_h5_group(cls, group: h5py.Group) -> "_DecodedForest":
        tree_names = sorted(k for k in group.keys() if k.startswith("Tree_"))
        trees = [_DecodedTree(group[name]["topology"][:], group[name]["parameters"][:]) for name in tree_names]
        return cls(trees)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        n_rows = X.shape[0]
        class_count = self.trees[0].class_count
        out = np.zeros((n_rows, class_count), dtype=np.float64)
        for row_i in range(n_rows):
            row = X[row_i]
            acc = np.zeros(class_count, dtype=np.float64)
            for tree in self.trees:
                acc += tree.predict_row(row)
            out[row_i] = acc / len(self.trees)
        return out


class DecodedParallelForest:
    """The full 'ClassifierForests' group: several Forest000N sub-forests, averaged."""

    def __init__(self, forests):
        self.forests = forests
        self.total_trees = sum(len(f.trees) for f in forests)

    @classmethod
    def from_ilp(cls, path: str, group_path: str = "PixelClassification/ClassifierForests") -> "DecodedParallelForest":
        forests = []
        with h5py.File(path, "r") as f:
            grp = f[group_path]
            for name in sorted(grp.keys()):
                if name.startswith("Forest"):
                    forests.append(_DecodedForest.from_h5_group(grp[name]))
        return cls(forests)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        n_rows = X.shape[0]
        class_count = self.forests[0].trees[0].class_count
        acc = np.zeros((n_rows, class_count), dtype=np.float64)
        for forest in self.forests:
            forest_probs = forest.predict_probabilities(X)
            acc += forest_probs * len(forest.trees)
        return acc / self.total_trees


###############################################################################
# 5. .ilp project parsing (plain h5py + json, no lazyflow/pydantic/vigra) and
# the PixelClassificationPipeline assembly, mirroring
# ilastik.experimental.api.PixelClassificationPipeline
# (ilastik/experimental/api/_pipelines.py) closely enough to be a drop-in
# replacement for inference on Pixel Classification projects.
###############################################################################


def _decode_str(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def _decode_str_list(ds) -> List[str]:
    return [_decode_str(x) for x in ds[()]]


def _axistags_keys(ds) -> str:
    axes = json.loads(_decode_str(ds[()]))["axes"]
    return "".join(ax["key"] for ax in axes)


@dataclass
class _InputDataInfo:
    axis_order: str  # e.g. "yxc" or "zyxc"
    num_channels: int

    @property
    def spatial_axes(self) -> str:
        return "".join(a for a in self.axis_order if a in "xyz")


@dataclass
class _FeatureMatrix:
    feature_ids: List[str]
    scales: List[float]
    selections: np.ndarray  # bool, shape (len(feature_ids), len(scales))
    compute_in_2d: np.ndarray  # bool, shape (len(scales),)


@dataclass
class _PixelClassificationProject:
    workflow_name: str
    input_data: _InputDataInfo
    feature_matrix: _FeatureMatrix
    label_names: List[str]
    forest: DecodedParallelForest

    @property
    def label_count(self) -> int:
        return len(self.label_names)

    @classmethod
    def from_ilp_file(cls, path: str) -> "_PixelClassificationProject":
        with h5py.File(path, "r") as f:
            workflow_name = _decode_str(f["workflowName"][()])
            if workflow_name != "Pixel Classification":
                raise ValueError(
                    f"Not a Pixel Classification project (workflowName={workflow_name!r}). "
                    "Autocontext / other workflows are not supported by this standalone reader."
                )

            role_names = _decode_str_list(f["Input Data"]["Role Names"])
            base_role = role_names[0]  # usually "Raw Data"

            lane_names = sorted(f["Input Data"]["infos"].keys())
            last_lane = f["Input Data"]["infos"][lane_names[-1]]
            base_info = last_lane[base_role]

            axis_order = _axistags_keys(base_info["axistags"])
            shape = base_info["shape"][()]
            num_channels = int(shape[axis_order.index("c")]) if "c" in axis_order else 1

            fs = f["FeatureSelections"]
            feature_ids = _decode_str_list(fs["FeatureIds"])
            scales = list(fs["Scales"][()])
            selections = np.asarray(fs["SelectionMatrix"][()], dtype=bool)
            if "ComputeIn2d" in fs:
                compute_in_2d = np.asarray(fs["ComputeIn2d"][()], dtype=bool)
            else:
                compute_in_2d = np.zeros(len(scales), dtype=bool)

            label_names = _decode_str_list(f["PixelClassification"]["LabelNames"])

        forest = DecodedParallelForest.from_ilp(path, "PixelClassification/ClassifierForests")

        return cls(
            workflow_name=workflow_name,
            input_data=_InputDataInfo(axis_order=axis_order, num_channels=num_channels),
            feature_matrix=_FeatureMatrix(
                feature_ids=feature_ids, scales=scales, selections=selections, compute_in_2d=compute_in_2d
            ),
            label_names=label_names,
            forest=forest,
        )


def _ensure_channel_axis(axis_order: str) -> str:
    return axis_order if "c" in axis_order else axis_order + "c"


def _canonical_spatial_order(spatial_axes_present: str) -> str:
    """Internal computation order, matching lazyflow's OpReorderAxes(AxisOrder='tczyx')."""
    return "".join(a for a in "zyx" if a in spatial_axes_present)


class PixelClassificationPipeline:
    """
    Standalone (no vigra/fastfilters/lazyflow/PyQt) equivalent of
    ilastik.experimental.api.PixelClassificationPipeline, for inference with
    an already-trained ilastik Pixel Classification .ilp project.

    Example:
        pipeline = PixelClassificationPipeline.from_ilp_file("project.ilp")
        prob_maps = pipeline.get_probabilities(image_array, dims=("y", "x"))
    """

    @classmethod
    def from_ilp_file(cls, path: str) -> "PixelClassificationPipeline":
        return cls(_PixelClassificationProject.from_ilp_file(path))

    def __init__(self, project: _PixelClassificationProject):
        self._project = project
        self._num_spatial_dims = len(project.input_data.spatial_axes)
        self._num_channels = project.input_data.num_channels
        self._output_axis_order = _ensure_channel_axis(project.input_data.axis_order)

    def _compute_feature_maybe_2d(self, channel_img: np.ndarray, feature_id: str, scale: float, compute_in_2d: bool):
        if compute_in_2d and channel_img.ndim == 3:
            slices = [compute_feature_at_scale(channel_img[z], feature_id, scale) for z in range(channel_img.shape[0])]
            return np.stack(slices, axis=0)
        return compute_feature_at_scale(channel_img, feature_id, scale)

    def _compute_features(self, data_zyxc: np.ndarray) -> np.ndarray:
        fm = self._project.feature_matrix
        num_input_channels = data_zyxc.shape[-1]

        blocks = []
        for i, feature_id in enumerate(fm.feature_ids):
            for j, scale in enumerate(fm.scales):
                if not fm.selections[i, j]:
                    continue
                compute_in_2d = bool(fm.compute_in_2d[j])
                for c in range(num_input_channels):
                    channel_img = data_zyxc[..., c]
                    feat = self._compute_feature_maybe_2d(channel_img, feature_id, scale, compute_in_2d)
                    if feat.ndim == channel_img.ndim:
                        feat = feat[..., np.newaxis]
                    blocks.append(feat)

        return np.concatenate(blocks, axis=-1)

    def get_probabilities(
        self,
        raw_data: Union[np.ndarray, "xarray.DataArray"],
        dims: Optional[Tuple[str, ...]] = None,
    ):
        """
        raw_data: an xarray.DataArray (dims inferred from it), or a plain
        numpy array together with dims=, e.g. dims=("y", "x") or
        dims=("z", "y", "x", "c").
        """
        if _HAVE_XARRAY and isinstance(raw_data, xarray.DataArray):
            dims = tuple(raw_data.dims)
            values = np.asarray(raw_data.values)
        else:
            if dims is None:
                raise ValueError("dims= is required when raw_data is not an xarray.DataArray")
            values = np.asarray(raw_data)
            if len(dims) != values.ndim:
                raise ValueError(f"dims {dims} doesn't match array shape {values.shape}")

        num_channels_in_data = values.shape[dims.index("c")] if "c" in dims else 1
        spatial_dims_in_data = [d for d in dims if d in "xyz"]
        num_spatial_in_data = len(spatial_dims_in_data)

        if num_channels_in_data != self._num_channels:
            raise ValueError(
                f"Number of channels mismatch. Classifier trained for {self._num_channels} "
                f"but input has {num_channels_in_data}"
            )
        if num_spatial_in_data != self._num_spatial_dims:
            raise ValueError(
                "Number of spatial dims doesn't match. "
                f"Classifier trained for {self._num_spatial_dims} but input has {num_spatial_in_data}"
            )

        canonical_spatial = _canonical_spatial_order("".join(spatial_dims_in_data))
        target_dims = tuple(canonical_spatial) + ("c",)

        if "c" not in dims:
            values = values[..., np.newaxis]
            dims = dims + ("c",)

        perm = [dims.index(d) for d in target_dims]
        data_zyxc = np.transpose(values, perm).astype(np.float32)

        features = self._compute_features(data_zyxc)
        spatial_shape = features.shape[:-1]
        n_features = features.shape[-1]

        flat = features.reshape(-1, n_features).astype(np.float32)
        flat_probs = self._project.forest.predict_probabilities(flat)
        n_classes = flat_probs.shape[-1]
        probs_zyxc = flat_probs.reshape(spatial_shape + (n_classes,))

        # Reorder to match the trained project's original axis order + 'c'.
        out_spatial = "".join(a for a in self._output_axis_order if a in "xyz")
        out_perm = [target_dims.index(a) for a in out_spatial] + [target_dims.index("c")]
        probs_out = np.transpose(probs_zyxc, out_perm)

        if _HAVE_XARRAY:
            return xarray.DataArray(probs_out, dims=tuple(out_spatial) + ("c",))
        return probs_out


###############################################################################
# CLI
###############################################################################


def _read_image(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    try:
        import tifffile

        return tifffile.imread(path)
    except ImportError:
        pass
    try:
        import imageio.v3 as iio

        return iio.imread(path)
    except ImportError:
        raise RuntimeError(
            f"Don't know how to read {path!r}: install 'tifffile' or 'imageio' "
            "(`pip install tifffile` or `pip install imageio`), or pass a .npy file."
        )


def _save_output(probs: np.ndarray, path: str) -> None:
    values = probs.values if hasattr(probs, "values") else probs
    if path.endswith(".h5"):
        with h5py.File(path, "w") as f:
            f.create_dataset("exported_data", data=values, compression="gzip")
    else:
        np.save(path, values)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ilastik Pixel Classification inference from a single pip-only Python file "
        "(no vigra/fastfilters/lazyflow/PyQt needed)."
    )
    parser.add_argument("ilp_file", help="Path to the trained ilastik .ilp project file.")
    parser.add_argument("image_file", help="Path to the input image (.tif/.tiff/.png/... or .npy).")
    parser.add_argument("-o", "--output", default=None, help="Output path (.npy or .h5). Defaults to <image>_Probabilities.npy")
    parser.add_argument(
        "--dims",
        nargs="+",
        default=None,
        help="Axis dims of the image file, e.g. y x, or z y x, or y x c. Defaults to y x (2D) / z y x (3D) based on image.ndim.",
    )
    args = parser.parse_args()

    print(f"Loading project: {args.ilp_file}")
    pipeline = PixelClassificationPipeline.from_ilp_file(args.ilp_file)

    print(f"Loading image: {args.image_file}")
    image = _read_image(args.image_file)
    print(f"  shape={image.shape} dtype={image.dtype}")

    dims = tuple(args.dims) if args.dims else (("z", "y", "x") if image.ndim == 3 else ("y", "x"))

    print("Computing features and predicting...")
    probs = pipeline.get_probabilities(image, dims=dims)
    values = probs.values if hasattr(probs, "values") else probs
    print(f"  done, output shape {values.shape}")

    out_path = args.output or (args.image_file.rsplit(".", 1)[0] + "_Probabilities.npy")
    _save_output(probs, out_path)
    print(f"Saved probabilities to {out_path}")


if __name__ == "__main__":
    main()
