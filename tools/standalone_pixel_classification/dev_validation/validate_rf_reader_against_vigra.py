"""
Dev-only validation script. Requires a real `vigra` install (conda env) —
NOT part of the pip-only standalone deliverable, just used to prove
`vigra_rf_reader.py`'s pure-NumPy decoder matches vigra's own
RandomForest.predictProbabilities().

Usage (from repo root, in a conda env with vigra):
    python tools/standalone_pixel_classification/dev_validation/validate_rf_reader_against_vigra.py
"""

import os
import shutil
import sys
import tempfile

import h5py
import numpy as np
import vigra

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vigra_rf_reader import DecodedParallelForest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def real_vigra_probabilities(ilp_path, group, X):
    """Ground truth: real vigra RandomForest objects, combined exactly the way
    lazyflow/classifiers/parallelVigraRfLazyflowClassifier.py does it."""
    tmpdir = tempfile.mkdtemp()
    try:
        cache_path = os.path.join(tmpdir, "cache.h5")
        with h5py.File(ilp_path, "r") as src, h5py.File(cache_path, "w") as dst:
            src.copy(src[group], dst, name="ClassifierForests")
        with h5py.File(cache_path, "r") as f:
            forest_names = sorted(k for k in f["ClassifierForests"].keys() if k.startswith("Forest"))

        real_forests = [vigra.learning.RandomForest(cache_path, f"ClassifierForests/{n}") for n in forest_names]
        tree_counts = [rf.treeCount() for rf in real_forests]
        acc = None
        for rf, tc in zip(real_forests, tree_counts):
            p = rf.predictProbabilities(X) * tc
            acc = p if acc is None else acc + p
        return acc / sum(tree_counts)
    finally:
        shutil.rmtree(tmpdir)


def check(ilp_path, group="PixelClassification/ClassifierForests"):
    print(f"\n=== {ilp_path} ===")
    decoded = DecodedParallelForest.from_ilp(ilp_path, group)
    first_tree = decoded.forests[0].trees[0]
    n_features, n_classes = first_tree.feature_count, first_tree.class_count
    print(
        f"sub-forests: {len(decoded.forests)}, total trees: {decoded.total_trees}, "
        f"features: {n_features}, classes: {n_classes}"
    )

    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, n_features)).astype(np.float32)
    ground_truth = real_vigra_probabilities(ilp_path, group, X)

    # Our pure-NumPy decoder, reading the same HDF5 arrays directly.
    ours = decoded.predict_probabilities(X)

    diff = np.abs(ground_truth - ours)
    print("max abs diff:", diff.max(), " mean abs diff:", diff.mean())
    ok = np.allclose(ground_truth, ours, atol=1e-6)
    print("MATCH" if ok else "MISMATCH!!!")
    return ok


if __name__ == "__main__":
    fixtures = [
        os.path.join(REPO_ROOT, "notebooks/pixel_classification_api/pc.ilp"),
        os.path.join(REPO_ROOT, "tests/test_ilastik/data/inputdata/mitocheckPixelClassification.ilp"),
        os.path.join(REPO_ROOT, "tests/test_ilastik/data/inputdata/pc-oc-133.ilp"),
    ]
    results = [check(f) for f in fixtures]
    print("\nALL PASS" if all(results) else "SOME FAILED")
    sys.exit(0 if all(results) else 1)
