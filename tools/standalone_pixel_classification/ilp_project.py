"""
Parses the metadata of a trained ilastik Pixel Classification .ilp project
directly via h5py + json - no lazyflow, no pydantic, no vigra.

Mirrors ilastik/experimental/parser/types.py's PixelClassificationProject
(field names, axis-order/channel-count derivation) closely enough to be a
drop-in equivalent for inference, but reads the HDF5 groups by hand instead
of going through vigra-dependent classifier deserialization - the trained
Random Forest is read via vigra_rf_reader.py instead of
ilastik.applets.base.appletSerializer.legacyClassifiers.deserialize_classifier
(which unpickles a ParallelVigraRfLazyflowClassifier and needs vigra to work).

Only dependency: h5py, numpy (both pip-installable).
"""
import json
from dataclasses import dataclass
from typing import List

import h5py
import numpy as np

from vigra_rf_reader import DecodedParallelForest


def _decode_str(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def _decode_str_list(ds) -> List[str]:
    return [_decode_str(x) for x in ds[()]]


def _axistags_keys(ds) -> str:
    axes = json.loads(_decode_str(ds[()]))["axes"]
    return "".join(ax["key"] for ax in axes)


@dataclass
class InputDataInfo:
    axis_order: str  # e.g. "yxc" or "zyxc"
    num_channels: int

    @property
    def spatial_axes(self) -> str:
        return "".join(a for a in self.axis_order if a in "xyz")


@dataclass
class FeatureMatrix:
    feature_ids: List[str]
    scales: List[float]
    selections: np.ndarray  # bool, shape (len(feature_ids), len(scales))
    compute_in_2d: np.ndarray  # bool, shape (len(scales),)


@dataclass
class PixelClassificationProject:
    workflow_name: str
    input_data: InputDataInfo
    feature_matrix: FeatureMatrix
    label_names: List[str]
    forest: DecodedParallelForest

    @property
    def label_count(self) -> int:
        return len(self.label_names)

    @classmethod
    def from_ilp_file(cls, path: str) -> "PixelClassificationProject":
        with h5py.File(path, "r") as f:
            workflow_name = _decode_str(f["workflowName"][()])
            if workflow_name != "Pixel Classification":
                raise ValueError(
                    f"Not a Pixel Classification project (workflowName={workflow_name!r}). "
                    "AutocontextProject / other workflows are not supported by this standalone reader."
                )

            role_names = _decode_str_list(f["Input Data"]["Role Names"])
            base_role = role_names[0]  # usually "Raw Data"

            lane_names = sorted(f["Input Data"]["infos"].keys())
            last_lane = f["Input Data"]["infos"][lane_names[-1]]
            base_info = last_lane[base_role]

            axis_order = _axistags_keys(base_info["axistags"])
            shape = base_info["shape"][()]
            if "c" in axis_order:
                num_channels = int(shape[axis_order.index("c")])
            else:
                num_channels = 1

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
            input_data=InputDataInfo(axis_order=axis_order, num_channels=num_channels),
            feature_matrix=FeatureMatrix(
                feature_ids=feature_ids, scales=scales, selections=selections, compute_in_2d=compute_in_2d
            ),
            label_names=label_names,
            forest=forest,
        )
