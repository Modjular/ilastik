"""
Dev-only validation script. Requires real `vigra` and `fastfilters` (conda
env, see PLAN.md) plus a C compiler.

Proves the two native pieces of native_prototype/ are exact stand-ins for
what real ilastik runs:

  1. ff_ctypes (fastfilters' C API via ctypes) vs the `fastfilters` Python
     package: every filter ilastik uses, 2D and 3D, several scales. Both call
     the same C code, so the expected difference is exactly 0.
  2. rf_native.NativeForest (rfwalk.c) vs vigra.learning.RandomForest,
     combined the way ParallelVigraRfLazyflowClassifier does, on the same
     fixtures as validate_rf_reader_against_vigra.py. Expected difference: 0.

ff_ctypes loads the conda env's own libfastfilters.so by default, i.e. the
exact library the `fastfilters` package links against.

Usage (from repo root, in the conda env):
    python tools/standalone_pixel_classification/dev_validation/validate_native_prototype.py
"""

import os
import sys

import numpy as np

TOOL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [TOOL_DIR, os.path.join(TOOL_DIR, "native_prototype"), os.path.dirname(os.path.abspath(__file__))]

import fastfilters  # noqa: E402

import ff_ctypes  # noqa: E402
from rf_native import NativeForest  # noqa: E402
from validate_rf_reader_against_vigra import REPO_ROOT, real_vigra_probabilities  # noqa: E402
from vigra_rf_reader import DecodedParallelForest  # noqa: E402


def check_filters():
    ff_ctypes.load(os.environ.get("FASTFILTERS_LIB") or os.path.join(sys.prefix, "lib", "libfastfilters.so"))
    rng = np.random.default_rng(0)
    images = {"2d": rng.random((97, 131), dtype=np.float32), "3d": rng.random((23, 41, 37), dtype=np.float32)}
    ok = True
    for name, img in images.items():
        for s in (0.3, 0.7, 1.0, 1.6, 3.5):
            for ws in (2.0, 3.5):
                pairs = [
                    ("gaussianSmoothing", (s,)),
                    ("gaussianGradientMagnitude", (s,)),
                    ("laplacianOfGaussian", (s,)),
                    ("hessianOfGaussianEigenvalues", (s,)),
                    ("structureTensorEigenvalues", (s, s * 0.5)),
                ]
                for fn, a in pairs:
                    ref = getattr(fastfilters, fn)(img, *a, window_size=ws)
                    mine = getattr(ff_ctypes, fn)(img, *a, window_size=ws)
                    same = ref.shape == mine.shape and np.array_equal(ref, mine)
                    ok &= same
                    if not same:
                        d = np.abs(ref - mine).max() if ref.shape == mine.shape else "shape"
                        print(f"MISMATCH {name} {fn}{a} ws={ws}: {d}")
    print("filters: ff_ctypes == fastfilters bit-for-bit" if ok else "filters: MISMATCH")
    return ok


def check_forest(ilp_path, group="PixelClassification/ClassifierForests"):
    decoded = DecodedParallelForest.from_ilp(ilp_path, group)
    n_features = decoded.forests[0].trees[0].feature_count
    X = np.random.default_rng(42).standard_normal((20000, n_features)).astype(np.float32)
    ground_truth = real_vigra_probabilities(ilp_path, group, X)
    ours = NativeForest(decoded).predict_probabilities(X)
    diff = np.abs(ground_truth - ours).max()
    print(f"forest {os.path.basename(ilp_path)}: max abs diff vs vigra {diff}")
    return diff == 0.0


if __name__ == "__main__":
    fixtures = [
        "notebooks/pixel_classification_api/pc.ilp",
        "tests/test_ilastik/data/inputdata/mitocheckPixelClassification.ilp",
        "tests/test_ilastik/data/inputdata/pc-oc-133.ilp",
    ]
    results = [check_filters()] + [check_forest(os.path.join(REPO_ROOT, f)) for f in fixtures]
    print("\nALL PASS" if all(results) else "SOME FAILED")
    sys.exit(0 if all(results) else 1)
