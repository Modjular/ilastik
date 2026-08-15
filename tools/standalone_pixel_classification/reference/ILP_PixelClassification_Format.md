# Ilastik .ilp Project Format (Pixel Classification)

This document describes the structure of an Ilastik `.ilp` project file for the **Pixel Classification** workflow. The format is based on HDF5.

## Root Level

At the root of the HDF5 file, the following datasets are required:

- `ilastikVersion`: (String) The version of Ilastik that created the project (e.g., `"1.4.0"`).
- `workflowName`: (String) Must be `"Pixel Classification"`.
- `time`: (String) A timestamp of when the project was created/saved (e.g., `Wed Aug 23 10:29:38 2023`).

## Input Data Group

**Path:** `/Input Data`

This group manages the input images (lanes).

### Datasets/Groups:
- `StorageVersion`: (String) `"0.2"`
- `Role Names`: (Dataset of Strings) Typically `[b'Raw Data', b'Prediction Mask']`.
- `infos`: (Group) Contains lane information.
  - `lane0000`: (Group)
    - `Raw Data`: (Group)
      - `__class__`: (String) `"FilesystemDatasetInfo"`
      - `filePath`: (String) Path to the image file (absolute or relative to `.ilp`).
      - `nickname`: (String) Display name for the dataset.
      - `datasetId`: (String) A unique UUID for this dataset.
      - `location`: (String) `"FileSystem"`
      - `allowLabels`: (Boolean) `True`
      - `display_mode`: (String) `"default"`
      - `normalizeDisplay`: (Boolean) `False`
      - `shape`: (Dataset of Int64) The shape of the image (e.g., `[1, 1104, 1225]` for ZYX).
      - `axistags`: (String, JSON) See "Axis Tags Format" below.

## Feature Selections Group

**Path:** `/FeatureSelections`

This group defines which features are calculated for classification.

### Datasets:
- `StorageVersion`: (String) `"0.1"`
- `FeatureIds`: (Dataset of Strings) List of feature internal names. Standard names:
  - `GaussianSmoothing`
  - `LaplacianOfGaussian`
  - `GaussianGradientMagnitude`
  - `DifferenceOfGaussians`
  - `StructureTensorEigenvalues`
  - `HessianOfGaussianEigenvalues`
- `Scales`: (Dataset of Float64) List of scales (sigmas) for feature calculation. Standard scales: `[0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0]`.
- `SelectionMatrix`: (Dataset of Boolean) Matrix of size `len(FeatureIds) x len(Scales)`. `True` indicates the feature is selected for that scale.
- `ComputeIn2d`: (Dataset of Boolean) Array of size `len(Scales)`. `True` indicates features for that scale should be computed in 2D.

## Pixel Classification Group

**Path:** `/PixelClassification`

This group contains labels and classification parameters.

### Datasets/Groups:
- `StorageVersion`: (String) `"0.1"`
- `LabelNames`: (Dataset of Strings) Names of the classes (e.g., `[b'Microglia', b'Background']`).
- `LabelColors`: (Dataset of Int32, shape `N_classes x 3`) RGB colors for labels.
- `PmapColors`: (Dataset of Int32, shape `N_classes x 3`) RGB colors for probability maps (usually same as `LabelColors`).
- `LabelSets`: (Group) Contains the actual labels.
  - `labels000`: (Group) Labels for lane 0.
    - `block0000`, `block0001`, ...: (Dataset of UInt8) Each dataset is a sparse block of labels.
      - Value `0` is unlabeled.
      - Values `1, 2, ...` correspond to the index in `LabelNames` + 1.
      - **Attribute:** `axistags` (String, JSON) describing the axes of this block (usually `yxc` or `zyxc`).
      - **Attribute:** `blockSlice` (String) describing the location of this block in the original image (e.g., `[y_start:y_stop, x_start:x_stop, c_start:c_stop]`).

- `ClassifierFactory`: (Pickled Object) Serialized `ParallelVigraRfLazyflowClassifierFactory`. If missing, Ilastik defaults to 100 trees.
- `ClassifierForests`: (Group, Optional) Contains the trained Random Forest. If missing, Ilastik will require retraining.

---

## Axis Tags Format

Axis tags are stored as a JSON string:

```json
{
  "axes": [
    {
      "key": "z",
      "typeFlags": 2,
      "resolution": 0,
      "description": ""
    },
    {
      "key": "y",
      "typeFlags": 2,
      "resolution": 0,
      "description": ""
    },
    {
      "key": "x",
      "typeFlags": 2,
      "resolution": 0,
      "description": ""
    }
  ]
}
```

### `typeFlags`:
- `1`: Channel (`c`)
- `2`: Space (`x`, `y`, `z`)
- `4`: Time (`t`)

## Block Slice Format

The `blockSlice` attribute is a string representation of a Python slice tuple:
`[start:stop,start:stop,start:stop]`

The number of slices and their order must match the `axistags` of the block. For example, if the block's axistags are `y, x, c`, a slice might be `[100:200,150:250,0:1]`.
