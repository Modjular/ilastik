// ── camera.js ─────────────────────────────────────────────────────────────────
// Handles pan + zoom of the canvas board, and the space-bar grab cursor.
// Completely independent of Vue and the drawing layer.

import {
    CAMERA_ZOOM_MIN,
    CAMERA_ZOOM_MAX,
    CAMERA_ZOOM_SENSITIVITY,
} from './constants.js';

// Shared camera state — read by drawing.js for coordinate scaling
export const camera = { x: 0, y: 0, scale: 1 };

// Space-bar panning flags — read by drawing.js to suppress brush strokes
export let isSpaceDown = false;
export let isPanning   = false;

export function setupCamera() {
    const viewport = document.getElementById('viewport');
    const board    = document.getElementById('canvas-board');

    function updateBoard() {
        board.style.transform =
            `translate(${camera.x}px, ${camera.y}px) scale(${camera.scale})`;
    }

    // ── Wheel: pinch-to-zoom (ctrl/cmd) or pan ────────────────────────────────
    viewport.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (e.ctrlKey || e.metaKey) {
            const delta    = -e.deltaY * CAMERA_ZOOM_SENSITIVITY;
            const newScale = Math.min(
                Math.max(CAMERA_ZOOM_MIN, camera.scale * Math.exp(delta)),
                CAMERA_ZOOM_MAX
            );
            const rect   = viewport.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            const boardX = (mouseX - camera.x) / camera.scale;
            const boardY = (mouseY - camera.y) / camera.scale;
            camera.scale = newScale;
            camera.x = mouseX - boardX * camera.scale;
            camera.y = mouseY - boardY * camera.scale;
        } else {
            camera.x -= e.deltaX;
            camera.y -= e.deltaY;
        }
        updateBoard();
    }, { passive: false });

    // ── Space-bar: toggle grab cursor / panning mode ──────────────────────────
    window.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && !isSpaceDown) {
            isSpaceDown = true;
            document.body.classList.add('space-down');
        }
    });
    window.addEventListener('keyup', (e) => {
        if (e.code === 'Space') {
            isSpaceDown = false;
            isPanning   = false;
            document.body.classList.remove('space-down', 'panning');
        }
    });

    viewport.addEventListener('mousedown', () => {
        if (isSpaceDown) { isPanning = true; document.body.classList.add('panning'); }
    });
    window.addEventListener('mousemove', (e) => {
        if (isPanning) { camera.x += e.movementX; camera.y += e.movementY; updateBoard(); }
    });
    window.addEventListener('mouseup', () => {
        isPanning = false;
        document.body.classList.remove('panning');
    });
}
