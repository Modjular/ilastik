"""
Pure-NumPy decoder for vigra's RandomForest HDF5 format (topology_/parameters_
arrays), validated against the real vigra.learning.RandomForest.predictProbabilities.

Format decoded from vigra source (random_forest/rf_nodeproxy.hxx,
random_forest/rf_decisionTree.hxx, random_forest.hxx):

Each Tree_NN group has two 1D arrays:
  topology:   int32,  [featureCount, classCount, <node data...>]
  parameters: float64, [<node data...>]

Traversal starts at topology index 2 (root). At index i:
  typeID = topology[i]
  if typeID & LeafNodeTag (0x40000000): LEAF (e_ConstProbNode, typeID==LeafNodeTag exactly here)
      parameter_addr = topology[i+1]
      weight = parameters[parameter_addr]
      probs  = parameters[parameter_addr+1 : parameter_addr+1+classCount]   # classCount doubles
  else: i_ThresholdNode (typeID == 0)
      parameter_addr = topology[i+1]
      child0 = topology[i+2]   # topology index of left child   (feature < threshold)
      child1 = topology[i+3]   # topology index of right child  (feature >= threshold)
      column = topology[i+4]
      threshold = parameters[parameter_addr+1]   # parameters[parameter_addr] is node weight
      next_index = child0 if feature[column] < threshold else child1

Per-forest prediction (RandomForest::predictProbabilities, options_.predict_weighted_ == 0):
  prob[row, :] = (1 / tree_count) * sum_over_trees( leaf_probs )
  (each tree's leaf_probs already sums to 1.0 across classes for a properly trained forest)

Across the 4 parallel sub-forests (ParallelVigraRfLazyflowClassifier.predict_probabilities):
  final_prob = (1 / total_tree_count) * sum_over_forests( forest_tree_count * forest_prob )
             = (1 / total_tree_count) * sum_over_all_trees_in_all_forests( leaf_probs )

Validated (see dev_validation/validate_rf_reader_against_vigra.py) against real
vigra.learning.RandomForest.predictProbabilities() on 3 independently trained
.ilp fixtures (49 and 25 features, 4 and 40 sub-forests, 100 trees total each) —
max abs diff ~2.9e-8 (float64 accumulation-order noise). Only covers
i_ThresholdNode (axis-aligned split) and e_ConstProbNode (leaf), which is what
ilastik's default RandomForest training produces; the HyperplaneNode/
HypersphereNode variants vigra also supports were not exercised.

Only dependencies: h5py, numpy. No vigra.
"""
import h5py
import numpy as np

LEAF_NODE_TAG = 0x40000000


class DecodedTree:
    def __init__(self, topology: np.ndarray, parameters: np.ndarray):
        self.topology = topology
        self.parameters = parameters
        self.feature_count = int(topology[0])
        self.class_count = int(topology[1])

    def predict_row(self, feature_row: np.ndarray) -> np.ndarray:
        topo = self.topology
        params = self.parameters
        index = 2
        while True:
            type_id = topo[index]
            if type_id & LEAF_NODE_TAG:
                addr = topo[index + 1]
                return params[addr + 1 : addr + 1 + self.class_count]
            # i_ThresholdNode
            addr = topo[index + 1]
            child0 = topo[index + 2]
            child1 = topo[index + 3]
            column = topo[index + 4]
            threshold = params[addr + 1]
            index = child0 if feature_row[column] < threshold else child1


class DecodedForest:
    """One 'Forest0000'-style sub-forest: a list of trees."""

    def __init__(self, trees):
        self.trees = trees

    @classmethod
    def from_h5_group(cls, group: h5py.Group) -> "DecodedForest":
        tree_names = sorted(k for k in group.keys() if k.startswith("Tree_"))
        trees = []
        for name in tree_names:
            t = group[name]
            trees.append(DecodedTree(t["topology"][:], t["parameters"][:]))
        return cls(trees)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        n_rows = X.shape[0]
        class_count = self.trees[0].class_count
        out = np.zeros((n_rows, class_count), dtype=np.float64)
        for row_i in range(n_rows):
            row = X[row_i]
            acc = np.zeros(class_count, dtype=np.float64)
            for tree in self.trees:
                acc += tree.predict_row(row)
            out[row_i] = acc / len(self.trees)
        return out


class DecodedParallelForest:
    """The full 'ClassifierForests' group: several Forest000N sub-forests, averaged."""

    def __init__(self, forests):
        self.forests = forests
        self.total_trees = sum(len(f.trees) for f in forests)

    @classmethod
    def from_ilp(cls, path: str, group_path: str = "PixelClassification/ClassifierForests") -> "DecodedParallelForest":
        forests = []
        with h5py.File(path, "r") as f:
            grp = f[group_path]
            for name in sorted(grp.keys()):
                if name.startswith("Forest"):
                    forests.append(DecodedForest.from_h5_group(grp[name]))
        return cls(forests)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        n_rows = X.shape[0]
        class_count = self.forests[0].trees[0].class_count
        acc = np.zeros((n_rows, class_count), dtype=np.float64)
        for forest in self.forests:
            forest_probs = forest.predict_probabilities(X)  # already normalized by that forest's tree count
            acc += forest_probs * len(forest.trees)
        return acc / self.total_trees
