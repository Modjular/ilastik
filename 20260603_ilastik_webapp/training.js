// ── training.js ───────────────────────────────────────────────────────────────
// Random-forest training and inference pipeline.

import { buildTrainingDataset } from './utils.js';
import { MIN_LABELS_TO_TRAIN, TRAIN_DEBOUNCE_MS } from './constants.js';

// ── Debounced retrain ─────────────────────────────────────────────────────────
let _trainTimer = null;

/**
 * Debounce wrapper — safe to call on every brush stroke without hammering the
 * GPU.  Pass the same bound `trainAndPredictAll` each time.
 */
export function debounce(trainFn) {
    clearTimeout(_trainTimer);
    _trainTimer = setTimeout(trainFn, TRAIN_DEBOUNCE_MS);
}

// ── Core train + infer ────────────────────────────────────────────────────────
/**
 * @param {object} appState   Reactive Vue state (images, sigma)
 * @param {Map}    resourceMap  imgId → { backend, intensityArray, … }
 * @param {FlatRandomForest} rf
 */
export async function trainAndPredictAll(appState, resourceMap, rf) {
    const totalLabels = appState.images.reduce((s, i) => s + i.labels.length, 0);
    if (totalLabels < MIN_LABELS_TO_TRAIN) return;

    const { combinedX, yArray } = await buildTrainingDataset(
        appState.images, resourceMap, totalLabels, appState.sigma
    );
    rf.train(combinedX, yArray, 8); // 8 = number of GPU feature channels

    for (const meta of appState.images) {
        const res = resourceMap.get(meta.id);
        if (res) await res.backend.runInference(rf);
    }
}
