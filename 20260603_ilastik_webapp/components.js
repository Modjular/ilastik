// ── components.js ────────────────────────────────────────────────────────────
// All Vue component definitions for the sidebar UI.
// These are pure declarative components: they emit events and never mutate
// appState directly.  All side-effects live in events.js / drawing.js.

import { ref, computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js';
import { MIN_LABELS_TO_TRAIN } from './constants.js';

// ── SliceSliders ──────────────────────────────────────────────────────────────
// Props:  nonDisplayDims [{ i, size }], sliceIndices [number]
// Emits:  slice-change { arrIdx, value }
export const SliceSliders = {
    props: ['nonDisplayDims', 'sliceIndices'],
    emits: ['slice-change'],
    template: `
        <div class="img-slice">
            <div
                v-for="(dim, arrIdx) in nonDisplayDims"
                :key="dim.i"
                class="img-slice-row"
            >
                <label>D{{ dim.i }}</label>
                <input
                    type="range" :min="0" :max="dim.size - 1"
                    :value="sliceIndices[arrIdx]"
                    @input="$emit('slice-change', { arrIdx, value: +$event.target.value })"
                />
                <span>{{ sliceIndices[arrIdx] }}/{{ dim.size - 1 }}</span>
            </div>
        </div>
    `,
};

// ── AxisSelector ──────────────────────────────────────────────────────────────
// Props:  shape [number], axes { axisY, axisX }
// Emits:  axis-change { role: 'Y'|'X', axisIdx }
export const AxisSelector = {
    props: ['shape', 'axes'],
    emits: ['axis-change'],
    computed: {
        dimOptions() {
            return this.shape.map((size, i) => ({ value: i, label: `Dim ${i} (${size})` }));
        },
    },
    template: `
        <div class="img-axes">
            <div class="img-axes-row" v-for="role in ['Y','X']" :key="role">
                <label>{{ role }}</label>
                <select
                    :value="role === 'Y' ? axes.axisY : axes.axisX"
                    @change="$emit('axis-change', { role, axisIdx: +$event.target.value })"
                >
                    <option v-for="opt in dimOptions" :key="opt.value" :value="opt.value">
                        {{ opt.label }}
                    </option>
                </select>
            </div>
        </div>
    `,
};

// ── ImageRow ──────────────────────────────────────────────────────────────────
// Props:  meta (imgMeta), isFirst, isLast, onAxisChange, onSliceChange
// Emits:  reorder { direction: -1|1 }, delete
export const ImageRow = {
    props: ['meta', 'isFirst', 'isLast', 'onAxisChange', 'onSliceChange'],
    emits: ['reorder', 'delete'],
    components: { AxisSelector, SliceSliders },
    computed: {
        labelText() {
            const n = this.meta.labels.length;
            return n === 0 ? 'no labels' : `${n} label${n !== 1 ? 's' : ''}`;
        },
        isMultiDim() {
            return this.meta.shape.length > 2;
        },
    },
    template: `
        <div class="img-row" :class="{ loading: meta.loading }">
            <div class="img-loading-bar" v-if="meta.loading"></div>
            <div class="img-row-header">
                <div class="img-reorder">
                    <button :disabled="isFirst" @click="$emit('reorder', { direction: -1 })" title="Move up">▲</button>
                    <button :disabled="isLast"  @click="$emit('reorder', { direction: +1 })" title="Move down">▼</button>
                </div>
                <div class="img-name" :title="meta.name">{{ meta.name }}</div>
                <div class="img-label-badge" :class="{ 'has-labels': meta.labels.length > 0 }">
                    {{ labelText }}
                </div>
                <button class="img-delete" :disabled="meta.loading"
                        @click="$emit('delete')" title="Remove image">✕</button>
            </div>
            <template v-if="!meta.loading && isMultiDim">
                <AxisSelector
                    :shape="meta.shape"
                    :axes="meta.axes"
                    @axis-change="e => onAxisChange(meta.id, e.role, e.axisIdx)"
                />
                <SliceSliders
                    :nonDisplayDims="meta.nonDisplayDims"
                    :sliceIndices="meta.sliceIndices"
                    @slice-change="e => onSliceChange(meta.id, e.arrIdx, e.value)"
                />
            </template>
        </div>
    `,
};

// ── ImageList ─────────────────────────────────────────────────────────────────
// Props:  images [], onAxisChange, onSliceChange
// Emits:  reorder { id, direction }, delete { id }
export const ImageList = {
    props: ['images', 'onAxisChange', 'onSliceChange'],
    emits: ['reorder', 'delete'],
    components: { ImageRow },
    template: `
        <div class="img-list">
            <div v-if="images.length === 0" class="img-empty">
                Drop images onto the canvas<br>or use the file picker above.
            </div>
            <ImageRow
                v-for="(meta, idx) in images"
                :key="meta.id"
                :meta="meta"
                :isFirst="idx === 0"
                :isLast="idx === images.length - 1"
                :onAxisChange="onAxisChange"
                :onSliceChange="onSliceChange"
                @reorder="e => $emit('reorder', { id: meta.id, direction: e.direction })"
                @delete="$emit('delete', { id: meta.id })"
            />
        </div>
    `,
};

// ── PhasePanel ────────────────────────────────────────────────────────────────
// Root sidebar component.  Owns activePhase locally; receives appState as prop.
// All user actions are delegated to handler callbacks passed in via props so
// that this file stays free of direct appState / resourceMap mutations.
export const PhasePanel = {
    props: [
        'state',
        // Handler callbacks — injected by index.html so components stay pure
        'onAxisChange',
        'onSliceChange',
        'onReorder',
        'onDelete',
        'onSigmaInput',
        'onToggleEraser',
        'onSelectClass',
        'onToggleLiveUpdate',
        'onSetDir',
        'onExport',
    ],
    components: { ImageList },
    setup(props) {
        const activePhase = ref('ingest');
        const exportBusy  = ref(false);

        const totalLabels = computed(() =>
            props.state.images.reduce((s, i) => s + i.labels.length, 0)
        );
        const canExport = computed(() =>
            props.state.images.length > 0 && !exportBusy.value
        );
        const exportCount = computed(() => props.state.images.length);

        return { activePhase, exportBusy, totalLabels, canExport, exportCount };
    },

    template: `
        <div class="phase-tabs">
            <button
                v-for="phase in ['ingest','train','export']"
                :key="phase"
                class="phase-tab"
                :class="{ active: activePhase === phase }"
                @click="activePhase = phase"
            >{{ { ingest: 'Images', train: 'Train', export: 'Export' }[phase] }}</button>
        </div>

        <!-- ── Images ──────────────────────────────────────────────── -->
        <div v-show="activePhase === 'ingest'" class="phase-panel">
            <ImageList
                :images="state.images"
                :onAxisChange="onAxisChange"
                :onSliceChange="onSliceChange"
                @reorder="e => onReorder(e.id, e.direction)"
                @delete="e => onDelete(e.id)"
            />
            <p class="hint">
                Drag files onto the canvas to load them.<br>
                Use ▲▼ to reorder. Delete removes labels &amp; retrains.
            </p>
        </div>

        <!-- ── Train ───────────────────────────────────────────────── -->
        <div v-show="activePhase === 'train'" class="phase-panel">
            <div class="control-group">
                <span class="control-label">Feature Scale (Sigma)</span>
                <div style="display:flex; gap:10px; align-items:center;">
                    <input type="range" min="0.3" max="10.0" step="0.1"
                           :value="state.sigma" @input="onSigmaInput" style="flex:1;">
                    <span style="min-width:30px; font-size:12px;">{{ state.sigma.toFixed(1) }}</span>
                </div>
            </div>

            <div class="control-group">
                <span class="control-label">Brush Size &amp; Eraser</span>
                <div style="display:flex; gap:10px; align-items:center;">
                    <input type="number" id="brushSize" value="5"
                           min="1" max="50" step="1" style="width:60px;">
                    <button
                        :class="state.isEraser ? 'btn-danger' : 'btn-subtle'"
                        style="flex:1;"
                        @click="onToggleEraser"
                    >Eraser: {{ state.isEraser ? 'ON' : 'OFF' }}</button>
                </div>
            </div>

            <div class="control-group">
                <span class="control-label">Classes</span>
                <div class="class-selector">
                    <div
                        v-for="(cls, idx) in [
                            { label: 'Class 1', color: 'red'   },
                            { label: 'Class 2', color: 'green' },
                            { label: 'Class 3', color: 'blue'  },
                        ]"
                        :key="idx"
                        class="class-item"
                        :class="{ active: state.currentClass === idx }"
                        @click="onSelectClass(idx)"
                    >
                        <div class="color-box" :style="{ background: cls.color }"></div>
                        <span>{{ cls.label }}</span>
                    </div>
                </div>
            </div>

            <hr class="divider">

            <button
                :disabled="state.images.length === 0"
                :style="{ background: state.liveUpdate ? '#4CAF50' : '' }"
                @click="onToggleLiveUpdate"
            >Live Update: {{ state.liveUpdate ? 'ON' : 'OFF' }}</button>

            <p class="hint">
                Paint labels on the canvas, then enable Live Update
                to see segmentation results in real time.
            </p>
        </div>

        <!-- ── Export ──────────────────────────────────────────────── -->
        <div v-show="activePhase === 'export'" class="phase-panel">
            <div class="control-group">
                <span class="control-label">Export</span>
                <label style="display:flex; align-items:center; gap:8px; font-size:13px; cursor:pointer;">
                    <input type="checkbox" v-model="state.exportSeg"> Segmentations
                </label>
                <label style="display:flex; align-items:center; gap:8px; font-size:13px; cursor:pointer;">
                    <input type="checkbox" v-model="state.exportProb"> Probabilities
                </label>
            </div>

            <hr class="divider">

            <div class="control-group">
                <span class="control-label">Destination</span>
                <div style="font-size:11px; color:#888; word-break:break-all;">
                    {{ state.dirLabel }}
                </div>
                <button v-if="state.hasDirectoryPicker"
                        class="btn-subtle" @click="onSetDir">
                    Set Output Folder…
                </button>
            </div>

            <button class="btn-success" :disabled="!canExport" @click="() => onExport(exportBusy, totalLabels)">
                <template v-if="exportBusy">{{ state.exportStatus }}</template>
                <template v-else>
                    Export Loaded Images{{ exportCount > 0 ? \` (\${exportCount})\` : '' }}
                </template>
            </button>

            <p class="hint">At least {{ $options.MIN_LABELS_TO_TRAIN }} labels needed to train before export.</p>
        </div>
    `,
    // Expose constant to template via $options
    MIN_LABELS_TO_TRAIN,
};
