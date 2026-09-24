"""
ctypes binding to fastfilters' plain C API (libfastfilters.so), with no
pybind11 module, vigra or conda involved.

Exposes the same five functions (and signatures) as the `fastfilters` Python
package that ilastik calls. Each one reproduces the glue in fastfilters'
own pybind layer (src/python/core.cxx + src/python/__init__.py, fastfilters
0.3) call-for-call, because that glue carries behavior the C library alone
doesn't:

  - structureTensorEigenvalues(img, innerScale, outerScale) passes
    (innerScale, outerScale) positionally into
    fastfilters_fir_structure_tensor2d/3d, whose C parameters are declared
    (sigma_outer, sigma_inner). This is the parameter-swap bug PLAN.md
    documents; real ilastik output depends on it, so it's kept.
  - The 3D eigenvalue solver is called as ev3d(zz, yz, xz, yy, xy, xx, ...),
    exactly as core.cxx does, not in the header's declared order.
  - Eigenvalue outputs are channel-first from C and rolled to channel-last,
    as __init__.py's np.rollaxis does.
  - window_size is passed straight through as fastfilters_options_t.window_ratio.

Only single-channel 2D (y, x) and 3D (z, y, x) float32 input is handled,
which is all ilastik's per-channel feature loop ever passes.

The library is located from, in order: the `path` argument to load(), the
FASTFILTERS_LIB environment variable, then the platform's default library
search path. In a tttk container, point it at the libfastfilters.so inside
the bundled ilastik binary's lib/ directory.
"""

import ctypes
import os

import numpy as np

_lib = None


class _Array2D(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.POINTER(ctypes.c_float)),
        ("n_x", ctypes.c_size_t),
        ("n_y", ctypes.c_size_t),
        ("stride_x", ctypes.c_size_t),
        ("stride_y", ctypes.c_size_t),
        ("n_channels", ctypes.c_size_t),
    ]


class _Array3D(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.POINTER(ctypes.c_float)),
        ("n_x", ctypes.c_size_t),
        ("n_y", ctypes.c_size_t),
        ("n_z", ctypes.c_size_t),
        ("stride_x", ctypes.c_size_t),
        ("stride_y", ctypes.c_size_t),
        ("stride_z", ctypes.c_size_t),
        ("n_channels", ctypes.c_size_t),
    ]


class _Options(ctypes.Structure):
    _fields_ = [("window_ratio", ctypes.c_float)]


def load(path=None):
    """Load libfastfilters.so and declare the C signatures used below."""
    global _lib
    if _lib is not None:
        return _lib
    path = path or os.environ.get("FASTFILTERS_LIB") or "libfastfilters.so"
    lib = ctypes.CDLL(path)

    A2, A3, O = ctypes.POINTER(_Array2D), ctypes.POINTER(_Array3D), ctypes.POINTER(_Options)
    F, d, u, sz = ctypes.POINTER(ctypes.c_float), ctypes.c_double, ctypes.c_uint, ctypes.c_size_t
    sigs = {
        "fastfilters_fir_gaussian2d": [A2, u, d, A2, O],
        "fastfilters_fir_gaussian3d": [A3, u, d, A3, O],
        "fastfilters_fir_gradmag2d": [A2, d, A2, O],
        "fastfilters_fir_gradmag3d": [A3, d, A3, O],
        "fastfilters_fir_laplacian2d": [A2, d, A2, O],
        "fastfilters_fir_laplacian3d": [A3, d, A3, O],
        "fastfilters_fir_hog2d": [A2, d, A2, A2, A2, O],
        "fastfilters_fir_hog3d": [A3, d, A3, A3, A3, A3, A3, A3, O],
        "fastfilters_fir_structure_tensor2d": [A2, d, d, A2, A2, A2, O],
        "fastfilters_fir_structure_tensor3d": [A3, d, d, A3, A3, A3, A3, A3, A3, O],
        "fastfilters_linalg_ev2d": [F, F, F, F, F, sz],
        "fastfilters_linalg_ev3d": [F, F, F, F, F, F, F, F, F, sz],
    }
    for name, argtypes in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_bool if "_fir_" in name else None
    lib.fastfilters_init.argtypes = []
    lib.fastfilters_init.restype = None
    lib.fastfilters_init()
    _lib = lib
    return lib


def _fptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _wrap(a):
    """numpy float32 C-contiguous (y,x) or (z,y,x) -> fastfilters array struct (strides in elements)."""
    s = [st // 4 for st in a.strides]
    if a.ndim == 2:
        return _Array2D(_fptr(a), a.shape[1], a.shape[0], s[1], s[0], 1)
    if a.ndim == 3:
        return _Array3D(_fptr(a), a.shape[2], a.shape[1], a.shape[0], s[2], s[1], s[0], 1)
    raise ValueError(f"expected a 2D or 3D single-channel array, got shape {a.shape}")


def _prep(image):
    image = np.ascontiguousarray(image, dtype=np.float32)
    if image.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D single-channel array, got shape {image.shape}")
    return image


def _call(name2d, image, *args):
    fn = getattr(load(), name2d if image.ndim == 2 else name2d.replace("2d", "3d"))
    if not fn(*args):
        raise RuntimeError(f"{fn.__name__} returned false")


def gaussianSmoothing(image, sigma, window_size=0.0):
    image = _prep(image)
    out = np.empty_like(image)
    _call("fastfilters_fir_gaussian2d", image, _wrap(image), 0, sigma, _wrap(out), _Options(window_size))
    return out


def gaussianGradientMagnitude(image, sigma, window_size=0.0):
    image = _prep(image)
    out = np.empty_like(image)
    _call("fastfilters_fir_gradmag2d", image, _wrap(image), sigma, _wrap(out), _Options(window_size))
    return out


def laplacianOfGaussian(image, scale=1.0, window_size=0.0):
    image = _prep(image)
    out = np.empty_like(image)
    _call("fastfilters_fir_laplacian2d", image, _wrap(image), scale, _wrap(out), _Options(window_size))
    return out


def _eigenvalues(image, tensor_fn_2d, sigmas, window_size):
    """Mirrors core.cxx's filter_ev_2d_binding / filter_ev_3d_binding, then __init__.py's rollaxis."""
    n_pixels = image.size
    opt = _Options(window_size)
    lib = load()
    if image.ndim == 2:
        xx, xy, yy = (np.empty_like(image) for _ in range(3))
        _call(tensor_fn_2d, image, _wrap(image), *sigmas, _wrap(xx), _wrap(xy), _wrap(yy), opt)
        ev = np.empty((2,) + image.shape, np.float32)
        lib.fastfilters_linalg_ev2d(_fptr(xx), _fptr(xy), _fptr(yy), _fptr(ev[0]), _fptr(ev[1]), n_pixels)
    else:
        xx, yy, zz, xy, xz, yz = (np.empty_like(image) for _ in range(6))
        _call(
            tensor_fn_2d,
            image,
            _wrap(image),
            *sigmas,
            _wrap(xx),
            _wrap(yy),
            _wrap(zz),
            _wrap(xy),
            _wrap(xz),
            _wrap(yz),
            opt,
        )
        ev = np.empty((3,) + image.shape, np.float32)
        lib.fastfilters_linalg_ev3d(
            _fptr(zz),
            _fptr(yz),
            _fptr(xz),
            _fptr(yy),
            _fptr(xy),
            _fptr(xx),
            _fptr(ev[0]),
            _fptr(ev[1]),
            _fptr(ev[2]),
            n_pixels,
        )
    return np.moveaxis(ev, 0, -1)


def hessianOfGaussianEigenvalues(image, scale, window_size=0.0):
    return _eigenvalues(_prep(image), "fastfilters_fir_hog2d", (scale,), window_size)


def structureTensorEigenvalues(image, innerScale, outerScale, window_size=0.0):
    # Positional (innerScale, outerScale) into C params declared (sigma_outer, sigma_inner): see module docstring.
    return _eigenvalues(_prep(image), "fastfilters_fir_structure_tensor2d", (innerScale, outerScale), window_size)
