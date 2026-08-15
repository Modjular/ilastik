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


def _drop_singleton_extra_axes(values: np.ndarray, dims: Tuple[str, ...]):
    """
    Drop length-1 axes that aren't spatial or channel - in practice a singleton
    't' from a time-series file that only holds one frame.

    Without this such an axis passes the channel and spatial-count checks
    (which only look at 'c' and 'xyz'), is then left out of the transpose's
    target order, and dies in np.transpose with "axes don't match array".
    """
    extra = [(d, values.shape[i]) for i, d in enumerate(dims) if d not in "xyzc"]
    if not extra:
        return values, dims

    too_long = [f"{d} (length {n})" for d, n in extra if n != 1]
    if too_long:
        raise ValueError(
            f"Unsupported non-spatial axes: {', '.join(too_long)}. Only x, y, z and c are "
            "supported; a time series has to be predicted one frame at a time."
        )

    index = tuple(slice(None) if d in "xyzc" else 0 for d in dims)
    return values[index], tuple(d for d in dims if d in "xyzc")


class PixelClassificationPipeline:
    @classmethod
    def from_ilp_file(cls, path: str) -> "PixelClassificationPipeline":
        return cls(PixelClassificationProject.from_ilp_file(path))

    def __init__(self, project: PixelClassificationProject):
        self._project = project
        self._num_spatial_dims = len(project.input_data.spatial_axes)
        self._num_channels = project.input_data.num_channels
        self._output_axis_order = ensure_channel_axis(project.input_data.axis_order)

    def _resolve_compute_in_2d(self, z_extent) -> np.ndarray:
        """
        A project that stores no ComputeIn2d gets one filled in from the input's
        z extent at prediction time, not from the file:

            if not self.ComputeIn2d.value:
                if self.Input.meta.shape[2] == 1:  # z
                    ComputeIn2d = [True] * len(scales)
                else:
                    ComputeIn2d = [False] * len(scales)

        (OpPixelFeaturesPresmoothed.setupOutputs). Defaulting to all-False here
        instead would diverge on single-slice input.
        """
        fm = self._project.feature_matrix
        if fm.compute_in_2d_from_project:
            return fm.compute_in_2d
        return np.full(len(fm.scales), z_extent == 1, dtype=bool)

    def _compute_features(self, data_zyxc: np.ndarray) -> np.ndarray:
        """
        data_zyxc: array with spatial axes in canonical (z,y,x) order (only the
        axes actually present) followed by a channel axis, e.g. shape (y,x,c)
        or (z,y,x,c).
        Returns an array of the same spatial shape with a channel axis holding
        every selected (feature, scale) output, concatenated in ilastik's order.

        Iterated scale-column-outer so that each presmoothed image is built
        once and shared by every feature selected at that scale (presmoothing
        depends only on the scale column and the input channel, and uses the
        wider 3.5 window, so it is the expensive half). The blocks are still
        emitted feature-major, which is the order ilastik's channel layout -
        and therefore the trained forest's feature columns - requires.
        """
        fm = self._project.feature_matrix
        num_input_channels = data_zyxc.shape[-1]
        num_spatial_dims = data_zyxc.ndim - 1
        z_extent = data_zyxc.shape[0] if num_spatial_dims == 3 else None
        compute_in_2d = self._resolve_compute_in_2d(z_extent)

        blocks = {}
        for j, scale in enumerate(fm.scales):
            selected = [i for i in range(len(fm.feature_ids)) if fm.selections[i, j]]
            if not selected:
                continue

            presmooth_in_2d = bool(compute_in_2d[j])
            _, new_scale = ilpf.presmoothing_sigma_and_new_scale(scale)

            for c in range(num_input_channels):
                presmoothed = ilpf.presmooth(data_zyxc[..., c], scale, in_2d=presmooth_in_2d)
                for i in selected:
                    feature_id = fm.feature_ids[i]
                    # invalid_z applies to the filter but not to the presmoothing
                    # above - see ilp_features.filter_forced_to_2d.
                    filter_in_2d = presmooth_in_2d or ilpf.filter_forced_to_2d(feature_id, new_scale, z_extent)
                    feat = ilpf.compute_feature_maybe_2d(presmoothed, feature_id, new_scale, in_2d=filter_in_2d)
                    if feat.ndim == presmoothed.ndim:
                        feat = feat[..., np.newaxis]
                    blocks[(i, j, c)] = feat

        ordered = [
            blocks[(i, j, c)]
            for i in range(len(fm.feature_ids))
            for j in range(len(fm.scales))
            for c in range(num_input_channels)
            if (i, j, c) in blocks
        ]
        return np.concatenate(ordered, axis=-1)

    def _expand_to_label_space(self, probs: np.ndarray) -> np.ndarray:
        """
        The forest only carries a column for each label that had training
        samples, so a label the user defined but never painted is absent from
        its output. ilastik reinstates it (classifierOperators.py):

            for i, label in enumerate(classifier.known_classes):
                full_probabilities[..., label - 1] = probabilities[..., i]

        Without this the output has too few channels AND the surviving classes
        sit at the wrong indices - e.g. known_labels [1, 3] of 3 labels puts
        label 3's probabilities in channel 1, where callers expect label 2's.
        """
        n_labels = self._project.label_count
        known = self._project.forest.known_labels

        if known == list(range(1, n_labels + 1)):
            return probs  # nothing was dropped; columns already line up

        if probs.shape[-1] != len(known):
            raise ValueError(
                f"Classifier emits {probs.shape[-1]} probability columns but its known_labels "
                f"lists {len(known)} entries ({known}); the project file is inconsistent."
            )
        out_of_range = [label for label in known if not 1 <= label <= n_labels]
        if out_of_range:
            raise ValueError(
                f"Classifier known_labels {out_of_range} fall outside the project's "
                f"{n_labels} label(s) ({self._project.label_names}); the project file is inconsistent."
            )

        full = np.zeros(probs.shape[:-1] + (n_labels,), dtype=probs.dtype)
        for i, label in enumerate(known):
            full[..., label - 1] = probs[..., i]
        return full

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

        values, dims = _drop_singleton_extra_axes(values, dims)

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

        # Without this the forest would happily index whatever columns exist and
        # return confident, plausible-looking, wrong probabilities.
        expected_features = self._project.forest.feature_count
        if n_features != expected_features:
            raise ValueError(
                f"Computed {n_features} feature columns but the trained forest expects "
                f"{expected_features}. The feature selection in the project file doesn't match "
                "what this input produces - check the channel count and whether the data is "
                "2D or 3D (filtering a thin-z volume slice by slice yields 2 eigenvalue "
                "channels per feature instead of 3)."
            )

        flat = features.reshape(-1, n_features).astype(np.float32)
        flat_probs = self._project.forest.predict_probabilities(flat)
        flat_probs = self._expand_to_label_space(flat_probs)
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
