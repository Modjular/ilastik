import h5py
import numpy as np
import json
import uuid
import time

def make_axistags(keys):
    """
    Generate the JSON string for axistags based on a sequence of axis keys.
    """
    type_flags = {
        't': 4,
        'z': 2,
        'y': 2,
        'x': 2,
        'c': 1
    }
    axes = []
    for key in keys:
        axes.append({
            "key": key,
            "typeFlags": type_flags.get(key, 2),
            "resolution": 0.0,
            "description": ""
        })
    return json.dumps({"axes": axes}, indent=2)

def generate_minimal_ilp(output_path, image_path, image_shape):
    """
    Generate a minimal Ilastik .ilp file.
    """
    with h5py.File(output_path, 'w') as f:
        # Root attributes
        f.create_dataset("ilastikVersion", data=b"1.4.0")
        f.create_dataset("workflowName", data=b"Pixel Classification")
        f.create_dataset("time", data=time.ctime().encode('utf-8'))
        f.create_dataset("currentApplet", data=0)

        # Input Data Group
        input_data = f.create_group("Input Data")
        input_data.create_dataset("StorageVersion", data=b"0.2")
        input_data.create_dataset("Role Names", data=[b"Raw Data", b"Prediction Mask"])
        
        infos = input_data.create_group("infos")
        lane0 = infos.create_group("lane0000")
        raw_data = lane0.create_group("Raw Data")
        raw_data.create_dataset("__class__", data=b"FilesystemDatasetInfo")
        raw_data.create_dataset("allowLabels", data=True)
        raw_data.create_dataset("axistags", data=make_axistags("zyx").encode('utf-8'))
        raw_data.create_dataset("datasetId", data=str(uuid.uuid4()).encode('utf-8'))
        raw_data.create_dataset("display_mode", data=b"default")
        raw_data.create_dataset("filePath", data=image_path.encode('utf-8'))
        raw_data.create_dataset("location", data=b"FileSystem")
        raw_data.create_dataset("nickname", data=b"input_image")
        raw_data.create_dataset("normalizeDisplay", data=False)
        raw_data.create_dataset("shape", data=np.array(image_shape, dtype=np.int64))
        
        input_data.create_group("local_data")

        # Feature Selections Group
        fs = f.create_group("FeatureSelections")
        fs.create_dataset("StorageVersion", data=b"0.1")
        feature_ids = [
            b"GaussianSmoothing",
            b"LaplacianOfGaussian",
            b"GaussianGradientMagnitude",
            b"DifferenceOfGaussians",
            b"StructureTensorEigenvalues",
            b"HessianOfGaussianEigenvalues"
        ]
        fs.create_dataset("FeatureIds", data=feature_ids)
        scales = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0]
        fs.create_dataset("Scales", data=np.array(scales, dtype=np.float64))
        
        # Select first two features for the first two scales as an example
        selection = np.zeros((len(feature_ids), len(scales)), dtype=bool)
        selection[0:2, 0:2] = True
        fs.create_dataset("SelectionMatrix", data=selection)
        
        compute_in_2d = np.ones(len(scales), dtype=bool)
        fs.create_dataset("ComputeIn2d", data=compute_in_2d)

        # Pixel Classification Group
        pc = f.create_group("PixelClassification")
        pc.create_dataset("StorageVersion", data=b"0.1")
        pc.create_dataset("LabelNames", data=[b"Label 1", b"Label 2"])
        pc.create_dataset("LabelColors", data=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.int32))
        pc.create_dataset("PmapColors", data=np.array([[255, 0, 0], [0, 255, 0]], dtype=np.int32))
        
        # Bookmarks (empty but often present)
        pc.create_group("Bookmarks")

        # LabelSets
        label_sets = pc.create_group("LabelSets")
        labels0 = label_sets.create_group("labels000")
        
        # Example label block: a small 10x10 block at (10, 10)
        # Assuming 2D (y, x, c) for labels as seen in the example
        label_data = np.ones((10, 10, 1), dtype=np.uint8)
        block0 = labels0.create_dataset("block0000", data=label_data)
        block0.attrs["axistags"] = make_axistags("yxc")
        block0.attrs["blockSlice"] = b"[10:20,10:20,0:1]"

        # Default Classifier Factory (pickled)
        # This is a ParallelVigraRfLazyflowClassifierFactory with 100 trees
        factory_pickle = (
            b'ccopy_reg\n_reconstructor\np0\n(clazyflow.classifiers.parallelVigraRfLazyflowClassifier\n'
            b'ParallelVigraRfLazyflowClassifierFactory\np1\nc__builtin__\nobject\np2\nNtp3\nRp4\n(dp5\n'
            b'VVERSION\np6\nL2L\nsV_num_trees\np7\nL100L\nsV_label_proportion\np8\nNsV_variable_importance_path\n'
            b'p9\nNsV_variable_importance_enabled\np10\nI00\nsV_kwargs\np11\n(dp12\nsV_num_forests\np13\nL4L\nsb.'
        )
        pc.create_dataset("ClassifierFactory", data=factory_pickle)

if __name__ == "__main__":
    # Example usage
    generate_minimal_ilp(
        "20260327_ilp_stuff/minimal.ilp", 
        "dummy_test.tif", 
        [1, 100, 100] # ZYX
    )
    print("Minimal .ilp file generated at 20260327_ilp_stuff/minimal.ilp")
