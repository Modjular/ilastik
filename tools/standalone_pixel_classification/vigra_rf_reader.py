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

Performance: predict_probabilities() walks every pixel through every tree
level-by-level, vectorized across all pixels at once (a NumPy array holding
each pixel's current node index, updated in a batch each iteration), instead
of one Python-level tree descent per pixel. The Python-level loop only runs
O(tree depth) times per tree rather than O(n_pixels); each iteration does a
handful of NumPy gather/compare ops over however many pixels are still
active (haven't reached a leaf yet). Confirmed to produce bit-identical
results to the old one-pixel-at-a-time version (see
dev_validation/validate_rf_reader_against_vigra.py) - only the traversal
order/vectorization changed, not the algorithm.

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
        """Single-pixel reference implementation - kept for clarity/testing;
        predict_all(...) is the vectorized version actually used for bulk work."""
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

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized equivalent of calling predict_row(row) for every row in X.
        Walks all rows through the tree simultaneously, level by level: at
        each iteration, rows still at an internal (threshold) node advance to
        their child; rows that just reached a leaf have their probabilities
        gathered and are marked done. Runs O(tree depth) Python-level
        iterations total, not O(n_rows).
        """
        topo = self.topology
        params = self.parameters
        n_rows = X.shape[0]
        class_count = self.class_count

        node_index = np.full(n_rows, 2, dtype=np.intp)
        result = np.empty((n_rows, class_count), dtype=np.float64)
        done = np.zeros(n_rows, dtype=bool)
        active = np.arange(n_rows)
        class_offsets = np.arange(1, class_count + 1)

        while active.size:
            cur_nodes = node_index[active]
            type_ids = topo[cur_nodes]
            is_leaf = (type_ids & LEAF_NODE_TAG).astype(bool)

            leaf_rows = active[is_leaf]
            if leaf_rows.size:
                leaf_nodes = node_index[leaf_rows]
                addrs = topo[leaf_nodes + 1]
                result[leaf_rows] = params[addrs[:, None] + class_offsets[None, :]]
                done[leaf_rows] = True

            internal_rows = active[~is_leaf]
            if internal_rows.size:
                internal_nodes = node_index[internal_rows]
                addrs = topo[internal_nodes + 1]
                child0 = topo[internal_nodes + 2]
                child1 = topo[internal_nodes + 3]
                columns = topo[internal_nodes + 4]
                thresholds = params[addrs + 1]
                feature_values = X[internal_rows, columns]
                node_index[internal_rows] = np.where(feature_values < thresholds, child0, child1)

            active = internal_rows

        return result


class DecodedForest:
    """
    One 'Forest0000'-style sub-forest: a list of trees.

    predict_probabilities() replicates vigra's RandomForest::predictProbabilities
    (random_forest.hxx) accumulation exactly, including its dtype: the Python
    binding vigra.learning.RandomForest.predictProbabilities() is hard-typed to
    float32 output (confirmed via its Boost.Python signature - only
    NumpyArray<2, float, ...> is accepted), meaning the per-tree accumulator
    itself is float32, not float64:

        double totalWeight = 0.0;
        for each tree:
            cur_w = weights[l];                    // double, unweighted case (predict_weighted_==0)
            prob(row, l) += static_cast<T>(cur_w);  // T=float32: truncates EVERY tree's contribution
            totalWeight += cur_w;                   // stays double
        prob(row, l) /= static_cast<T>(totalWeight);

    i.e. totalWeight accumulates in double, but the running probability sum
    itself is truncated to float32 after every single tree, not just once at
    the end. Only the predict_weighted_==0 case (ilastik's default, and the
    only one seen in any fixture used to validate this file) is implemented;
    predict_weighted_==1 would need cur_w scaled by each leaf's node weight
    (parameters[addr]) and is deliberately unsupported (raises) rather than
    silently producing wrong output.
    """

    def __init__(self, trees, predict_weighted: bool = False):
        self.trees = trees
        if predict_weighted:
            raise NotImplementedError(
                "predict_weighted_=1 forests are not supported (ilastik's default, and every "
                "fixture this was validated against, uses predict_weighted_=0)"
            )

    @classmethod
    def from_h5_group(cls, group: h5py.Group) -> "DecodedForest":
        tree_names = sorted(k for k in group.keys() if k.startswith("Tree_"))
        trees = []
        for name in tree_names:
            t = group[name]
            trees.append(DecodedTree(t["topology"][:], t["parameters"][:]))
        predict_weighted = False
        if "_options" in group and "predict_weighted_" in group["_options"]:
            predict_weighted = bool(group["_options"]["predict_weighted_"][()][0])
        return cls(trees, predict_weighted=predict_weighted)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        n_rows = X.shape[0]
        class_count = self.trees[0].class_count
        prob = np.zeros((n_rows, class_count), dtype=np.float32)  # vigra's T=float32 accumulator
        total_weight = np.zeros(n_rows, dtype=np.float64)  # vigra's local `double totalWeight`
        for tree in self.trees:
            cur_w = tree.predict_all(X)  # float64, matches ArrayVector<double> leaf weights
            prob += cur_w.astype(np.float32)  # static_cast<T>(cur_w) happens before every +=
            total_weight += cur_w.sum(axis=-1)
        prob /= total_weight.astype(np.float32)[:, None]
        return prob


class DecodedParallelForest:
    """
    The full 'ClassifierForests' group: several Forest000N sub-forests, averaged.

    predict_probabilities() replicates
    lazyflow/classifiers/parallelVigraRfLazyflowClassifier.py's
    ParallelVigraRfLazyflowClassifier.predict_probabilities exactly, including
    its dtype: it operates directly on the float32 arrays returned by each
    sub-forest's predictProbabilities(), so the whole cross-forest combination
    (multiply by tree count, sum across forests, divide by total tree count)
    happens in float32, not float64:

        forest_predictions *= forest.treeCount()   # float32 *= python int -> stays float32
        total_predictions += forest_predictions    # float32 += float32
        total_predictions /= self._num_trees        # float32 /= python int -> stays float32

    Note: the real implementation runs each sub-forest's prediction in a
    separate thread (lazyflow.request.RequestPool) and accumulates results in
    whatever order those threads happen to finish - i.e. even the real
    pipeline's own output isn't perfectly run-to-run reproducible at the
    float32-rounding level (float32 addition isn't associative). This
    processes sub-forests in a fixed (sorted-by-name) order instead, which is
    a legitimate/equally-valid summation order, just not necessarily *the*
    order any specific real run used.
    """

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
        total = np.zeros((n_rows, class_count), dtype=np.float32)
        for forest in self.forests:
            forest_probs = forest.predict_probabilities(X) * np.float32(len(forest.trees))
            total += forest_probs
        total /= np.float32(self.total_trees)
        return total
