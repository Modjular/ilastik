"""
ctypes wrapper around rfwalk.c: vigra RandomForest inference over trees
already decoded by pixel_classification_standalone.DecodedParallelForest.

Output is bit-identical to DecodedParallelForest.predict_probabilities (and
therefore to vigra.learning.RandomForest.predictProbabilities combined the
way ilastik's ParallelVigraRfLazyflowClassifier does): same float32
accumulation, same double totalWeight, same cross-forest combine.

build() compiles rfwalk.c with the system C compiler; in tttk this would be
a step in build.py next to the other vendored C code.
"""

import ctypes
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "rfwalk.c")


def build(out_path=None, cc=os.environ.get("CC", "cc")):
    # No -ffast-math: bit-exactness depends on strict IEEE float32/float64 ordering.
    out_path = out_path or os.path.join(HERE, "librfwalk.so")
    subprocess.check_call([cc, "-O3", "-fopenmp", "-shared", "-fPIC", "-o", out_path, SRC])
    return out_path


class NativeForest:
    def __init__(self, parallel_forest, lib_path=None):
        lib_path = lib_path or os.path.join(HERE, "librfwalk.so")
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

        trees = [t for f in parallel_forest.forests for t in f.trees]
        self.class_count = trees[0].class_count
        if self.class_count > 64:
            raise ValueError("rfwalk.c supports at most 64 classes")
        self._trees_per_forest = np.array([len(f.trees) for f in parallel_forest.forests], np.int32)
        self._topo = np.concatenate([t.topology.astype(np.int32) for t in trees])
        self._params = np.concatenate([t.parameters.astype(np.float64) for t in trees])
        self._topo_off = np.cumsum([0] + [len(t.topology) for t in trees[:-1]]).astype(np.int64)
        self._param_off = np.cumsum([0] + [len(t.parameters) for t in trees[:-1]]).astype(np.int64)

    def predict_probabilities(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        out = np.empty((X.shape[0], self.class_count), np.float32)
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
        return out
