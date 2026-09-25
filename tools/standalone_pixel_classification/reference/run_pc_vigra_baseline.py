import sys
import os
import argparse
import glob
import numpy as np
import h5py
import vigra
import tifffile
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Run Ilastik Pixel Classification headlessly.")
    parser.add_argument("ilp_file", help="Path to the Ilastik .ilp project file.")
    parser.add_argument("input_dir", help="Directory containing .tif images, or a single .tif file.")
    parser.add_argument("--output_dir", "-o", help="Directory to save output Probability maps (.h5). Defaults to input_dir.")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for prediction to save RAM. Defaults to 1024.")
    return parser.parse_args()

def load_ilp_config(ilp_path):
    print(f"Loading Ilastik project: {ilp_path}")
    with h5py.File(ilp_path, "r") as h5f:
        if "FeatureSelections" not in h5f:
            raise KeyError("FeatureSelections group not found in the .ilp file. Is this a Pixel Classification project?")
            
        scales = h5f["FeatureSelections/Scales"][:]
        feature_ids = [fid.decode('utf-8') for fid in h5f["FeatureSelections/FeatureIds"][:]]
        selection_matrix = h5f["FeatureSelections/SelectionMatrix"][:]
        
        # Check if computation was in 2D
        compute_in_2d = [False] * len(scales)
        if "ComputeIn2d" in h5f["FeatureSelections"]:
            compute_in_2d = h5f["FeatureSelections/ComputeIn2d"][:]

        if "PixelClassification/LabelNames" in h5f:
            label_names = [ln.decode('utf-8') for ln in h5f["PixelClassification/LabelNames"][:]]
        else:
            label_names = None

    print(f"Found {selection_matrix.sum()} selected features across {len(scales)} scales.")
    
    # Load the Random Forest using Vigra's HDF5 loader
    # vigra.learning.RandomForest loads from a group
    rf = vigra.learning.RandomForest(ilp_path, "PixelClassification/ClassifierForests")
    print(f"Loaded Random Forest with {rf.treeCount()} trees.")
    
    return {
        "scales": scales,
        "feature_ids": feature_ids,
        "selection_matrix": selection_matrix,
        "compute_in_2d": compute_in_2d,
        "label_names": label_names,
        "rf": rf
    }

def compute_features(image_c, scale, feature_id, window_size=3.5):
    """
    Computes a single feature for a single channel image.
    If image_c is 3D (Z,Y,X), some vigra filters will calculate in 3D.
    If image_c is 2D (Y,X), computes in 2D.
    """
    if feature_id == "GaussianSmoothing":
        feat = vigra.filters.gaussianSmoothing(image_c, scale, window_size=window_size)
    elif feature_id == "LaplacianOfGaussian":
        feat = vigra.filters.laplacianOfGaussian(image_c, scale=scale)
    elif feature_id == "GaussianGradientMagnitude":
        feat = vigra.filters.gaussianGradientMagnitude(image_c, sigma=scale)
    elif feature_id == "DifferenceOfGaussians":
        feat = vigra.filters.gaussianSmoothing(image_c, scale, window_size=window_size) - \
               vigra.filters.gaussianSmoothing(image_c, scale * 0.66, window_size=window_size)
    elif feature_id == "StructureTensorEigenvalues":
        feat = vigra.filters.structureTensorEigenvalues(image_c, innerScale=scale, outerScale=scale * 0.5)
    elif feature_id == "HessianOfGaussianEigenvalues":
        feat = vigra.filters.hessianOfGaussianEigenvalues(image_c, scale=scale)
    else:
        raise ValueError(f"Unknown feature {feature_id}")

    # Ensure output has a channel dimension at the end
    if feat.ndim == image_c.ndim:
        feat = feat[..., np.newaxis]
    return feat

def process_image(img_path, config, chunk_size):
    print(f"Processing image: {img_path}")
    image = tifffile.imread(img_path)
    
    # Tiff could be (Z, Y, X, C), (Y, X, C), (Y, X), etc.
    # We standardise to native vigra spacing.
    # Simple heuristic:
    if image.ndim == 2:
        image = image[..., np.newaxis] # add C
    elif image.ndim == 3 and image.shape[-1] > 32:
        image = image[..., np.newaxis] # Likely Z, Y, X. Add C
        
    num_channels = image.shape[-1]
    spatial_shape = image.shape[:-1]
    
    print(f"  Image shape identified as (Spatial: {spatial_shape}, Channels: {num_channels})")
    
    image = image.astype(np.float32)

    # Ilastik concatenates features in this exact order: 
    # outter loop over FeatureIds, inner loop over Scales
    all_features = []
    
    for i, feature_id in enumerate(config["feature_ids"]):
        for j, scale in enumerate(config["scales"]):
            if config["selection_matrix"][i, j]:
                # List to hold the feature computed per input channel
                channels_features = []
                for c in range(num_channels):
                    img_c = image[..., c]
                    
                    # ComputeIn2d handling
                    if len(spatial_shape) == 3 and config["compute_in_2d"][j]:
                        # 3D spatial but 2D feature requested: loop over Z
                        z_feats = []
                        for z in range(spatial_shape[0]):
                           f_2d = compute_features(img_c[z], scale, feature_id)
                           z_feats.append(f_2d)
                        feat = np.stack(z_feats, axis=0)
                    else:
                        # Native computation
                        feat = compute_features(img_c, scale, feature_id)
                        
                    channels_features.append(feat)
                
                # Concatenate the resulting channels for this Feature+Scale
                stacked_channels = np.concatenate(channels_features, axis=-1)
                all_features.append(stacked_channels)
                print(f"  Computed {feature_id} at scale {scale} -> shape {stacked_channels.shape}")
                
    # Concatenate all features
    full_feature_block = np.concatenate(all_features, axis=-1)
    print(f"  Total features extracted: {full_feature_block.shape}")

    # Reshape for prediction
    flat_features = full_feature_block.reshape(-1, full_feature_block.shape[-1])
    
    # Predict in chunks to save RAM
    n_pixels = flat_features.shape[0]
    preds = []
    
    rf = config["rf"]
    # Predict probabilities requires features as float32, which they already are.
    
    # Optional chunking
    if chunk_size > 0:
        chunk_pixels = chunk_size * chunk_size
        for start_idx in range(0, n_pixels, chunk_pixels):
            end_idx = min(start_idx + chunk_pixels, n_pixels)
            chunk = np.ascontiguousarray(flat_features[start_idx:end_idx], dtype=np.float32)
            chunk_pred = rf.predictProbabilities(chunk)
            preds.append(chunk_pred)
        
        flat_preds = np.vstack(preds)
    else:
        flat_features = np.ascontiguousarray(flat_features, dtype=np.float32)
        flat_preds = rf.predictProbabilities(flat_features)
        
    num_classes = flat_preds.shape[-1]
    final_shape = spatial_shape + (num_classes,)
    prob_map = flat_preds.reshape(final_shape)
    
    print(f"  Prediction finished. Prob map shape: {prob_map.shape}")
    return prob_map

def main():
    args = parse_args()
    
    config = load_ilp_config(args.ilp_file)
    
    if os.path.isdir(args.input_dir):
        image_files = glob.glob(os.path.join(args.input_dir, "*.tif")) + \
                      glob.glob(os.path.join(args.input_dir, "*.tiff"))
    else:
        image_files = [args.input_dir]
        
    out_dir = args.output_dir if args.output_dir else (args.input_dir if os.path.isdir(args.input_dir) else os.path.dirname(args.input_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    for img_file in image_files:
        prob_map = process_image(img_file, config, args.chunk_size)
        
        base_name = os.path.basename(img_file)
        name, _ = os.path.splitext(base_name)
        out_path = os.path.join(out_dir, f"{name}_Probabilities.h5")
        
        print(f"  Saving to {out_path}...")
        with h5py.File(out_path, "w") as f_out:
            ds = f_out.create_dataset("exported_data", data=prob_map, compression="gzip")
            if config["label_names"]:
                ds.attrs["label_names"] = [ln.encode('utf-8') for ln in config["label_names"]]
                
    print("Done!")

if __name__ == "__main__":
    main()
