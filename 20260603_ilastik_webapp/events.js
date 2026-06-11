// ── events.js ─────────────────────────────────────────────────────────────────
// Imperative event handlers that bridge Vue component emits and vanilla-JS
// side-effects (appState mutations, resourceMap teardown, DOM reorder).
// Nothing in this file imports Vue directly.

import { reactive }            from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js';
import { WebGpuBackend }       from './backends/webgpu.js';
import { loadFileIntoArray, isDuplicateFile, inferAxes, exportImagesData, verifyPermission, writeFile } from './utils.js';
import { rerenderSlice }       from './drawing.js';
import { MIN_LABELS_TO_TRAIN } from './constants.js';

// ── Image ingestion ───────────────────────────────────────────────────────────

export async function handleFiles(files, appState, resourceMap, deps) {
    for (const file of files) {
        if (!file.name.match(/\.(tif|tiff|png|jpg|jpeg)$/i)) continue;
        if (isDuplicateFile(file, appState.images)) {
            console.info(`Skipping duplicate: ${file.name}`);
            continue;
        }
        await addImage(file, appState, resourceMap, deps);
    }
}

export async function addImage(file, appState, resourceMap, { isSpaceDownRef, paint, scheduleRetrain, updateStatus }) {
    const imgId = crypto.randomUUID();

    // 1. Push loading placeholder — Vue renders the spinner row immediately
    const imgMeta = reactive({
        id:      imgId,
        name:    file.name,
        fileSize: file.size,
        loading: true,
        shape:   [1, 1],
        axes:    { axisY: 0, axisX: 1 },
        sliceIndices:   [],
        nonDisplayDims: [],
        labels:  [],
    });
    appState.images.push(imgMeta);
    updateStatus();

    // 2. Load file
    let loaded;
    try {
        loaded = await loadFileIntoArray(file);
    } catch (err) {
        console.error(err);
        alert(`Failed to load ${file.name}: ${err.message}`);
        appState.images.splice(appState.images.indexOf(imgMeta), 1);
        updateStatus();
        return;
    }
    const { intensityArray, rgba, w, h, shape } = loaded;
    const imgShape = shape ?? [h, w];

    // 3. Infer display axes
    const axes = inferAxes(imgShape);
    const nonDisplayDims = imgShape
        .map((size, i) => ({ i, size }))
        .filter(({ i }) => i !== axes.axisY && i !== axes.axisX);
    const sliceIndices = nonDisplayDims.map(() => 0);

    // 4. Build canvas tile (vanilla DOM — Vue never touches this subtree)
    const container = document.createElement('div');
    container.className = 'image-container';
    container.style.width  = w + 'px';
    container.style.height = h + 'px';

    const gpuCanvas = document.createElement('canvas');
    gpuCanvas.className = 'gpu-canvas';
    gpuCanvas.width = w; gpuCanvas.height = h;

    const labelCanvas = document.createElement('canvas');
    labelCanvas.className = 'label-canvas';
    labelCanvas.width = w; labelCanvas.height = h;

    const tileLabel = document.createElement('div');
    tileLabel.className   = 'image-tile-label';
    tileLabel.textContent = file.name;

    container.appendChild(gpuCanvas);
    container.appendChild(labelCanvas);
    container.appendChild(tileLabel);
    document.getElementById('canvas-board').appendChild(container);

    // 5. Initialise WebGPU backend
    const backend = new WebGpuBackend();
    try {
        await backend.initialize(gpuCanvas);
    } catch (err) {
        console.error(err);
        alert('WebGPU not supported or initialization failed.');
        container.remove();
        appState.images.splice(appState.images.indexOf(imgMeta), 1);
        updateStatus();
        return;
    }
    await backend.allocateImage(w, h, rgba);
    await backend.updateFeatures(intensityArray, appState.sigma);

    // 6. Store GPU resources (inert — Vue never sees this map)
    resourceMap.set(imgId, {
        backend, gpuCanvas, labelCanvas, container, intensityArray,
        _cachedRect: null,
        _ro: null,
    });

    // 7. Finish reactive meta — one flush, Vue re-renders once
    imgMeta.shape          = imgShape;
    imgMeta.axes           = axes;
    imgMeta.nonDisplayDims = nonDisplayDims;
    imgMeta.sliceIndices   = sliceIndices;
    imgMeta.loading        = false;

    // 8. Wire drawing events onto the label canvas
    let isDrawing   = false;
    let activeImgId = null;

    labelCanvas.addEventListener('mousedown', (e) => {
        if (isSpaceDownRef()) return;
        resourceMap.get(imgId)._cachedRect = labelCanvas.getBoundingClientRect();
        isDrawing   = true;
        activeImgId = imgId;
        paint(e, imgMeta);
    });
    labelCanvas.addEventListener('mousemove', (e) => {
        if (isDrawing && activeImgId === imgId && !isSpaceDownRef()) {
            if (e.buttons !== 1) {
                isDrawing   = false;
                activeImgId = null;
                if (appState.liveUpdate) scheduleRetrain();
            } else {
                paint(e, imgMeta);
            }
        }
    });
    labelCanvas.addEventListener('mouseup', () => {
        if (activeImgId === imgId) {
            isDrawing   = false;
            activeImgId = null;
            if (appState.liveUpdate) scheduleRetrain();
        }
    });

    // Invalidate cached bounding rect on resize
    const ro = new ResizeObserver(() => {
        const res = resourceMap.get(imgId);
        if (res) res._cachedRect = null;
    });
    ro.observe(labelCanvas);
    resourceMap.get(imgId)._ro = ro;

    updateStatus();
}

// ── Reorder / delete ──────────────────────────────────────────────────────────

export function handleReorder(imgId, direction, appState) {
    const idx    = appState.images.findIndex(i => i.id === imgId);
    const newIdx = idx + direction;
    if (idx === -1 || newIdx < 0 || newIdx >= appState.images.length) return;

    // Swap reactive meta (Vue re-renders ImageList)
    const tmp = appState.images[idx];
    appState.images[idx]    = appState.images[newIdx];
    appState.images[newIdx] = tmp;

    // Mirror swap on canvas board
    const board = document.getElementById('canvas-board');
    const tiles = [...board.children];
    if (direction === -1) board.insertBefore(tiles[idx], tiles[newIdx]);
    else                  board.insertBefore(tiles[newIdx], tiles[idx]);
}

export function handleDelete(imgId, appState, resourceMap, scheduleRetrain, updateStatus) {
    const idx = appState.images.findIndex(i => i.id === imgId);
    if (idx === -1) return;

    const meta       = appState.images[idx];
    const labelCount = meta.labels.length;

    if (labelCount > 0) {
        const ok = confirm(
            `"${meta.name}" has ${labelCount} label${labelCount !== 1 ? 's' : ''}.\n` +
            `Deleting it will remove those labels and retrain the model. Continue?`
        );
        if (!ok) return;
    }

    const res = resourceMap.get(imgId);
    if (res) {
        res._ro?.disconnect();
        res.container.remove();
        resourceMap.delete(imgId);
    }

    appState.images.splice(idx, 1);

    if (labelCount > 0 && appState.liveUpdate) scheduleRetrain();
    updateStatus();
}

// ── Axis / slice handlers ─────────────────────────────────────────────────────

export function handleAxisChange(imgId, role, axisIdx, appState, resourceMap, scheduleRetrain) {
    const meta = appState.images.find(i => i.id === imgId);
    if (!meta) return;

    if (role === 'Y') meta.axes.axisY = axisIdx;
    else              meta.axes.axisX = axisIdx;

    meta.nonDisplayDims = meta.shape
        .map((size, i) => ({ i, size }))
        .filter(({ i }) => i !== meta.axes.axisY && i !== meta.axes.axisX);
    meta.sliceIndices = meta.nonDisplayDims.map(() => 0);

    rerenderSlice(imgId, appState, resourceMap, scheduleRetrain);
}

export function handleSliceChange(imgId, arrIdx, value, appState, resourceMap, scheduleRetrain) {
    const meta = appState.images.find(i => i.id === imgId);
    if (!meta) return;
    meta.sliceIndices[arrIdx] = value;
    rerenderSlice(imgId, appState, resourceMap, scheduleRetrain);
}

// ── Export ────────────────────────────────────────────────────────────────────

export async function handleExport(
    appState, resourceMap, rf, trainFn,
    { exportBusyRef, setExportBusy, totalLabels }
) {
    if (!appState.exportSeg && !appState.exportProb) {
        alert('Select at least one export option.');
        return;
    }
    if (totalLabels < MIN_LABELS_TO_TRAIN) {
        alert(`Paint at least ${MIN_LABELS_TO_TRAIN} labels before exporting.`);
        return;
    }
    setExportBusy(true);
    appState.exportStatus = 'Training…';
    try {
        await trainFn();
        appState.exportStatus = 'Exporting…';
        await exportImagesData(appState.images, rf, {
            exportSeg:  appState.exportSeg,
            exportProb: appState.exportProb,
            outputDirHandle: appState.outputDirHandle,
            verifyPermission,
            writeFile,
            resourceMap,
        });
        alert('All images exported successfully!');
    } catch (err) {
        console.error(err);
        alert(`Export failed: ${err.message}`);
    } finally {
        setExportBusy(false);
        appState.exportStatus = '';
    }
}
