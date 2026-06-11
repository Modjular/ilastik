// ── constants.js ─────────────────────────────────────────────────────────────
// All tunable values in one place.  Import from here; never hard-code magic
// numbers in other modules.

export const RF_CONFIG = { numTrees: 8, maxDepth: 8 };
export const MIN_LABELS_TO_TRAIN = 5;

// Camera
export const CAMERA_ZOOM_MIN        = 0.1;
export const CAMERA_ZOOM_MAX        = 10;
export const CAMERA_ZOOM_SENSITIVITY = 0.01;

// Debounce interval for auto-retrain on brush-up
export const TRAIN_DEBOUNCE_MS = 300;

// Semi-transparent fill colours per class index
export const LABEL_COLORS = [
    'rgba(255,0,0,0.5)',
    'rgba(0,255,0,0.5)',
    'rgba(0,0,255,0.5)',
];
