"""
ctypes wrapper around rfwalk.c: vigra RandomForest inference over trees
already decoded by pixel_classification_standalone.DecodedParallelForest.

Output is bit-identical to DecodedParallelForest.predict_probabilities (and
therefore to vigra.learning.RandomForest.predictProbabilities combined the
way ilastik's ParallelVigraRfLazyflowClassifier does): same float32
accumulation, same double totalWeight, same cross-forest combine.

rfwalk.c is plain C99 with no OpenMP. Rows are split across a Python thread
pool instead; ctypes releases the GIL during each call, so the threads run
the C code truly in parallel. Rows are independent, so the output doesn't
depend on the thread count.

build() compiles rfwalk.c with a C compiler (gcc/clang, or `zig cc` to
cross-compile). Platforms without a compiler (typically Windows) use a
prebuilt library passed as lib_path, see PLAN.md "Cross-platform".
"""

import ctypes
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "rfwalk.c")

if sys.platform == "win32":
    LIB_NAME = "rfwalk.dll"
elif sys.platform == "darwin":
    LIB_NAME = "librfwalk.dylib"
else:
    LIB_NAME = "librfwalk.so"

# Below this many rows per thread, thread handoff costs more than it saves.
_MIN_ROWS_PER_TASK = 16384


def build(out_path=None, cc=None):
    # No -ffast-math: bit-exactness depends on strict IEEE float32/float64 ordering.
    out_path = out_path or os.path.join(HERE, LIB_NAME)
    cc = (cc or os.environ.get("CC", "cc")).split()
    subprocess.check_call(cc + ["-O3", "-std=c99", "-shared", "-fPIC", "-o", out_path, SRC])
    return out_path


class NativeForest:
    def __init__(self, parallel_forest, lib_path=None, n_threads=None):
        lib_path = lib_path or os.path.join(HERE, LIB_NAME)
        if not os.path.exists(lib_path):
            build(lib_path)
        self._lib = ctypes.CDLL(lib_path)
        P = np.ctypeslib.ndpointer
        self._lib.rf_predict.argtypes = [
            ctypes.c_int,
            P(np.int32, flags="C"),
            P(np.int32, flags="C"),
            P(np.int64, flags="C"),
            P(np.float64, flags="C"),
            P(np.int64, flags="C"),
            P(np.float32, flags="C"),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            P(np.float32, flags="C"),
        ]
        self._lib.rf_predict.restype = None
        self.n_threads = n_threads or os.cpu_count() or 1

        trees = [t for f in parallel_forest.forests for t in f.trees]
        self.class_count = trees[0].class_count
        if self.class_count > 64:
            raise ValueError("rfwalk.c supports at most 64 classes")
        self._trees_per_forest = np.array([len(f.trees) for f in parallel_forest.forests], np.int32)
        self._topo = np.concatenate([t.topology.astype(np.int32) for t in trees])
        self._params = np.concatenate([t.parameters.astype(np.float64) for t in trees])
        self._topo_off = np.cumsum([0] + [len(t.topology) for t in trees[:-1]]).astype(np.int64)
        self._param_off = np.cumsum([0] + [len(t.parameters) for t in trees[:-1]]).astype(np.int64)

    def _predict_rows(self, X, out):
        self._lib.rf_predict(
            len(self._trees_per_forest),
            self._trees_per_forest,
            self._topo,
            self._topo_off,
            self._params,
            self._param_off,
            X,
            X.shape[0],
            X.shape[1],
            self.class_count,
            out,
        )

    def predict_probabilities(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        out = np.empty((X.shape[0], self.class_count), np.float32)
        n_tasks = min(self.n_threads * 4, max(1, X.shape[0] // _MIN_ROWS_PER_TASK))
        if n_tasks <= 1:
            self._predict_rows(X, out)
            return out
        # Row slices of C-contiguous arrays are themselves C-contiguous views.
        bounds = np.linspace(0, X.shape[0], n_tasks + 1).astype(np.int64)
        with ThreadPoolExecutor(self.n_threads) as pool:
            list(
                pool.map(
                    lambda i: self._predict_rows(X[bounds[i] : bounds[i + 1]], out[bounds[i] : bounds[i + 1]]),
                    range(n_tasks),
                )
            )
        return out
