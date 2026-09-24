/*
 * vigra RandomForest inference over pre-decoded tree arrays (see rf_native.py).
 *
 * Replicates, bit for bit:
 *  - vigra RandomForest::predictProbabilities (random_forest.hxx) for
 *    predict_weighted_ == 0: each tree's leaf weights are truncated to float
 *    before accumulating, the totalWeight normalizer stays double;
 *  - ilastik's ParallelVigraRfLazyflowClassifier combine across sub-forests,
 *    done in float32.
 * Node layout is the one decoded in pixel_classification_standalone.py
 * (_DecodedTree): topology[node] = type id, +1 parameter address, +2/+3
 * children, +4 feature column; parameters[addr + 1] = threshold (internal)
 * or first class weight (leaf).
 *
 * Build without -ffast-math.
 */
#include <stdint.h>

#define LEAF_NODE_TAG 0x40000000
#define MAX_CLASSES 64

void rf_predict(int n_forests, const int32_t *trees_per_forest,
                const int32_t *topo, const int64_t *topo_off,
                const double *params, const int64_t *param_off,
                const float *X, int64_t n_rows, int n_feat, int n_classes, float *out)
{
    int total_trees = 0;
    for (int f = 0; f < n_forests; f++)
        total_trees += trees_per_forest[f];

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < n_rows; r++) {
        const float *x = X + r * n_feat;
        float total[MAX_CLASSES] = {0};
        float prob[MAX_CLASSES];
        int t = 0;
        for (int f = 0; f < n_forests; f++) {
            double total_weight = 0.0;
            for (int c = 0; c < n_classes; c++)
                prob[c] = 0.0f;
            for (int k = 0; k < trees_per_forest[f]; k++, t++) {
                const int32_t *tp = topo + topo_off[t];
                const double *pp = params + param_off[t];
                int64_t node = 2;
                while (!(tp[node] & LEAF_NODE_TAG)) {
                    int32_t addr = tp[node + 1];
                    /* float feature vs double threshold: compared in double, as numpy/vigra do */
                    node = ((double)x[tp[node + 4]] < pp[addr + 1]) ? tp[node + 2] : tp[node + 3];
                }
                const double *w = pp + tp[node + 1] + 1;
                for (int c = 0; c < n_classes; c++) {
                    prob[c] += (float)w[c];
                    total_weight += w[c]; /* per class, in class order, as vigra does */
                }
            }
            float norm = (float)total_weight; /* vigra casts totalWeight to T before dividing */
            float n_trees = (float)trees_per_forest[f];
            for (int c = 0; c < n_classes; c++)
                total[c] += (prob[c] / norm) * n_trees;
        }
        for (int c = 0; c < n_classes; c++)
            out[r * n_classes + c] = total[c] / (float)total_trees;
    }
}
