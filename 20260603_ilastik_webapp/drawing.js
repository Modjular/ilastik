// ── drawing.js ────────────────────────────────────────────────────────────────
// Label painting, erasing, canvas redraw, and slice/sigma re-rendering.
// Receives appState, resourceMap, and callback refs as arguments so it stays
// free of module-level side effects.

import { eraseLabelsInRadius, pickSlice } from './utils.js';
import { LABEL_COLORS }                   from './constants.js';

// ── Slice re-render ───────────────────────────────────────────────────────────
// Called whenever the user moves a slice slider or changes the display axes.
export async function rerenderSlice(imgId, appState, resourceMap, scheduleRetrain) {
    const meta = appState.images.find(i => i.id === imgId);
    const res  = resourceMap.get(imgId);
    if (!meta || !res) return;

    const w = meta.shape[meta.axes.axisX];
    const h = meta.shape[meta.axes.axisY];

    // Resize canvas / container whenever axes change their displayed dimensions
    if (res.container.style.width  !== w + 'px') res.container.style.width  = w + 'px';
    if (res.container.style.height !== h + 'px') res.container.style.height = h + 'px';
    if (res.gpuCanvas.width  !== w) res.gpuCanvas.width  = w;
    if (res.gpuCanvas.height !== h) res.gpuCanvas.height = h;
    if (res.labelCanvas.width  !== w) res.labelCanvas.width  = w;
    if (res.labelCanvas.height !== h) res.labelCanvas.height = h;

    if (meta.shape.length <= 2) {
        redrawLabels(imgId, appState, resourceMap);
        return;
    }

    const slice = pickSlice(res.intensityArray, meta.shape, meta.axes, meta.sliceIndices);

    // Greyscale normalise → RGBA for the GPU texture
    const n = slice.length;
    const rgba = new Uint8ClampedArray(n * 4);
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < n; i++) {
        if (slice[i] < min) min = slice[i];
        if (slice[i] > max) max = slice[i];
    }
    const range = max - min || 1;
    for (let i = 0; i < n; i++) {
        const v = ((slice[i] - min) / range * 255) | 0;
        rgba[i*4] = rgba[i*4+1] = rgba[i*4+2] = v;
        rgba[i*4+3] = 255;
    }

    await res.backend.allocateImage(w, h, rgba);
    await res.backend.updateFeatures(slice, appState.sigma);

    redrawLabels(imgId, appState, resourceMap);

    if (appState.liveUpdate) scheduleRetrain();
}

// ── Sigma change ──────────────────────────────────────────────────────────────
export async function handleSigmaChange(sigma, appState, resourceMap, scheduleRetrain) {
    for (const meta of appState.images) {
        const res = resourceMap.get(meta.id);
        if (!res) continue;
        if (meta.shape.length <= 2) {
            await res.backend.updateFeatures(res.intensityArray, sigma);
        } else {
            const slice = pickSlice(res.intensityArray, meta.shape, meta.axes, meta.sliceIndices);
            await res.backend.updateFeatures(slice, sigma);
        }
    }
    if (appState.liveUpdate) scheduleRetrain();
}

// ── Label visibility helpers ──────────────────────────────────────────────────
export function isLabelVisible(label, meta) {
    let sliceIdx = 0;
    for (let d = 0; d < meta.shape.length; d++) {
        if (d !== meta.axes.axisY && d !== meta.axes.axisX) {
            if (label.coords[d] !== meta.sliceIndices[sliceIdx++]) return false;
        }
    }
    return true;
}

export function redrawLabels(imgId, appState, resourceMap) {
    const meta = appState.images.find(i => i.id === imgId);
    const res  = resourceMap.get(imgId);
    if (!meta || !res) return;

    const ctx = res.labelCanvas.getContext('2d');
    ctx.clearRect(0, 0, res.labelCanvas.width, res.labelCanvas.height);

    for (const l of meta.labels) {
        if (isLabelVisible(l, meta)) {
            const x = l.coords[meta.axes.axisX];
            const y = l.coords[meta.axes.axisY];
            ctx.fillStyle = LABEL_COLORS[l.cls] ?? LABEL_COLORS[0];
            ctx.beginPath();
            ctx.arc(x, y, l.radius, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}

// ── Brush painting ────────────────────────────────────────────────────────────
export function paint(e, imgMeta, appState, resourceMap) {
    const res = resourceMap.get(imgMeta.id);
    if (!res) return;

    const rect   = res._cachedRect || res.labelCanvas.getBoundingClientRect();
    const scaleX = imgMeta.shape[imgMeta.axes.axisX] / rect.width;
    const scaleY = imgMeta.shape[imgMeta.axes.axisY] / rect.height;
    const x      = Math.floor((e.clientX - rect.left) * scaleX);
    const y      = Math.floor((e.clientY - rect.top)  * scaleY);
    const radius = parseInt(document.getElementById('brushSize').value, 10);
    const w      = imgMeta.shape[imgMeta.axes.axisX];
    const h      = imgMeta.shape[imgMeta.axes.axisY];

    if (appState.isEraser) {
        imgMeta.labels = eraseLabelsInRadius(imgMeta.labels, x, y, radius, imgMeta.axes, imgMeta);
        redrawLabels(imgMeta.id, appState, resourceMap);
        return;
    }

    // Build full N-D coordinate for this label point
    const coords = new Array(imgMeta.shape.length);
    coords[imgMeta.axes.axisY] = y;
    coords[imgMeta.axes.axisX] = x;
    let sliceIdx = 0;
    for (let d = 0; d < imgMeta.shape.length; d++) {
        if (d !== imgMeta.axes.axisY && d !== imgMeta.axes.axisX) {
            coords[d] = imgMeta.sliceIndices[sliceIdx++];
        }
    }

    // Deduplicate consecutive identical strokes
    const last = imgMeta.labels[imgMeta.labels.length - 1];
    let isDup = false;
    if (last && last.cls === appState.currentClass) {
        isDup = coords.every((c, d) => last.coords[d] === c);
    }

    if (!isDup && x >= 0 && x < w && y >= 0 && y < h) {
        const ctx = res.labelCanvas.getContext('2d');
        ctx.fillStyle = LABEL_COLORS[appState.currentClass] ?? LABEL_COLORS[0];
        ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill();
        imgMeta.labels.push({ coords, cls: appState.currentClass, radius });
    }
}
