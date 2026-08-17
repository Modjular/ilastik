"""
Standalone (no vigra, no fastfilters, no lazyflow, no PyQt) equivalent of
ilastik.experimental.api.PixelClassificationPipeline
(ilastik/experimental/api/_pipelines.py), for running inference with an
already-trained ilastik Pixel Classification .ilp project.

Usage mirrors the real API:

    from standalone_pipeline import PixelClassificationPipeline
    pipeline = PixelClassificationPipeline.from_ilp_file("project.ilp")
    prob_maps = pipeline.get_probabilities(image, dims=("y", "x"))
    # or, if xarray is installed: pipeline.get_probabilities(xarray_data_array)

Only dependencies: numpy, scipy, h5py. xarray is optional (only used if the
input is already an xarray.DataArray, or to label the output's dims).

Feature computation and channel-concatenation order matches
lazyflow/operators/opPixelFeaturesPresmoothed.py exactly: outer loop over
FeatureIds, inner loop over Scales, and within each (feature, scale) block,
input channels concatenated in order.
"""
from typing import List, Optional, Tuple, Union

import numpy as np

import ilp_features as ilpf
from ilp_project import PixelClassificationProject

try:
    import xarray

    HAVE_XARRAY = True
except ImportError:
    HAVE_XARRAY = False


def ensure_channel_axis(axis_order: str) -> str:
    if "c" not in axis_order:
        return axis_order + "c"
    return axis_order


def _canonical_spatial_order(spatial_axes_present: str) -> str:
    """Internal computation order, matching lazyflow's OpReorderAxes(AxisOrder='tczyx')."""
    return "".join(a for a in "zyx" if a in spatial_axes_present)


class PixelClassificationPipeline:
    @classmethod
    def from_ilp_file(cls, path: str) -> "PixelClassificationPipeline":
        return cls(PixelClassificationProject.from_ilp_file(path))

    def __init__(self, project: PixelClassificationProject):
        self._project = project
        self._num_spatial_dims = len(project.input_data.spatial_axes)
        self._num_channels = project.input_data.num_channels
        self._output_axis_order = ensure_channel_axis(project.input_data.axis_order)

    def _compute_feature_maybe_2d(self, channel_img: np.ndarray, feature_id: str, scale: float, compute_in_2d: bool):
        if compute_in_2d and channel_img.ndim == 3:
            slices = [ilpf.compute_feature_at_scale(channel_img[z], feature_id, scale) for z in range(channel_img.shape[0])]
            return np.stack(slices, axis=0)
        return ilpf.compute_feature_at_scale(channel_img, feature_id, scale)

    def _compute_features(self, data_zyxc: np.ndarray) -> np.ndarray:
        """
        data_zyxc: array with spatial axes in canonical (z,y,x) order (only the
        axes actually present) followed by a channel axis, e.g. shape (y,x,c)
        or (z,y,x,c).
        Returns an array of the same spatial shape with a channel axis holding
        every selected (feature, scale) output, concatenated in ilastik's order.
        """
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
        numpy array together with `dims`, e.g. dims=("y", "x") or
        dims=("z", "y", "x", "c").
        """
        if HAVE_XARRAY and isinstance(raw_data, xarray.DataArray):
            dims = tuple(raw_data.dims)
            values = np.asarray(raw_data.values)
        else:
            if dims is None:
                raise ValueError("dims= is required when raw_data is not an xarray.DataArray")
            values = np.asarray(raw_data)
            if len(dims) != values.ndim:
                raise ValueError(f"dims {dims} doesn't match array shape {values.shape}")

        if "c" in dims:
            num_channels_in_data = values.shape[dims.index("c")]
        else:
            num_channels_in_data = 1

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

        # Reorder to match the trained project's original axis order + 'c',
        # so output layout matches ilastik.experimental.api.PixelClassificationPipeline.
        out_spatial = "".join(a for a in self._output_axis_order if a in "xyz")
        out_perm = [target_dims.index(a) for a in out_spatial] + [target_dims.index("c")]
        probs_out = np.transpose(probs_zyxc, out_perm)

        if HAVE_XARRAY:
            return xarray.DataArray(probs_out, dims=tuple(out_spatial) + ("c",))
        return probs_out
