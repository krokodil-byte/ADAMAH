"""
ADAMAH v3 - GPU Memory & Math Library
pip install adamah
"""

import numpy as np
import ctypes
import os
import subprocess
import tempfile
import sys

# Op codes
ADD, SUB, MUL, DIV = 0, 1, 2, 3
POW, ATAN2, MOD, MIN, MAX = 50, 51, 52, 53, 54
NEG, ABS, SQRT, EXP, LOG, TANH = 10, 11, 12, 13, 14, 15
RELU, GELU = 16, 17
SIN, COS, TAN = 18, 19, 23
ASIN, ACOS, ATAN = 24, 25, 26
SINH, COSH = 27, 28
LOG2, LOG10, EXP2 = 29, 30, 31
FLOOR, CEIL, ROUND, TRUNC, SIGN = 32, 33, 34, 35, 36
RECIP, SQR, COPY = 20, 21, 22
SUM, RMAX, RMIN, PROD, MEAN = 0, 1, 2, 3, 4

# Execution modes
MODE_CPU = 0
MODE_GPU = 1
MODE_AUTO = 2
_MODE_USE_GLOBAL = -1

def _mode_to_int(mode):
    """Convert mode string to int"""
    if mode is None:
        return _MODE_USE_GLOBAL
    if isinstance(mode, str):
        mode = mode.lower()
        if mode == 'cpu': return MODE_CPU
        if mode == 'gpu': return MODE_GPU
        if mode == 'auto': return MODE_AUTO
        raise ValueError(f"Invalid mode: {mode}. Use 'cpu', 'gpu', or 'auto'")
    return mode

_lib = None
_pkg_dir = os.path.dirname(os.path.abspath(__file__))

def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    
    # Cerca .so precompilata
    so_path = os.path.join(_pkg_dir, 'libadamah.so')
    
    if not os.path.exists(so_path):
        # Compila al primo uso
        c_path = os.path.join(_pkg_dir, 'adamah.c')
        if not os.path.exists(c_path):
            raise RuntimeError("adamah.c not found in package")
        
        print("ADAMAH: Compiling native library...")
        try:
            subprocess.check_call([
                'gcc', '-shared', '-fPIC', '-O3', '-fopenmp',
                c_path, '-o', so_path,
                '-lvulkan', '-lm'
            ], stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            raise RuntimeError("Failed to compile. Install: libvulkan-dev")
    
    _lib = ctypes.CDLL(so_path)
    
    # Setup function signatures
    _lib.adamah_init.restype = ctypes.c_int
    _lib.adamah_shutdown.restype = None
    _lib.adamah_set_mode.argtypes = [ctypes.c_int]
    _lib.adamah_set_mode.restype = None
    _lib.adamah_get_mode.restype = ctypes.c_int
    _lib.inject.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
    _lib.inject.restype = ctypes.c_int
    _lib.extract.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
    _lib.extract.restype = ctypes.c_int
    _lib.bufsize.argtypes = [ctypes.c_char_p]
    _lib.bufsize.restype = ctypes.c_uint32
    # _ex functions with mode parameter
    _lib.vop1_ex.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_int]
    _lib.vop1_ex.restype = ctypes.c_int
    _lib.vop2_ex.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_int]
    _lib.vop2_ex.restype = ctypes.c_int
    _lib.vops_ex.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_uint32, ctypes.c_int]
    _lib.vops_ex.restype = ctypes.c_int

    return _lib

class Adamah:
    """ADAMAH context manager"""
    
    def __init__(self):
        self._lib = _get_lib()
        self._lib.adamah_init()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
    
    def shutdown(self):
        if self._lib:
            self._lib.adamah_shutdown()

    def set_mode(self, mode):
        """Set execution mode: 'cpu', 'gpu', or 'auto'"""
        self._lib.adamah_set_mode(_mode_to_int(mode))

    def get_mode(self):
        """Get current execution mode"""
        mode = self._lib.adamah_get_mode()
        return ['cpu', 'gpu', 'auto'][mode]

    def put(self, name: str, data) -> int:
        """Inject data: CPU → GPU"""
        arr = np.ascontiguousarray(data, dtype=np.float32)
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return self._lib.inject(name.encode(), ptr, len(arr))
    
    def get(self, name: str, count: int = None) -> np.ndarray:
        """Extract data: GPU → CPU"""
        if count is None:
            count = self._lib.bufsize(name.encode())
        out = np.zeros(count, dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._lib.extract(name.encode(), ptr, count)
        return out
    
    def op1(self, op: int, dst: str, a: str, n: int, mode=None):
        """Unary op: dst = op(a)"""
        if mode is None:
            return self._lib.vop1(op, dst.encode(), a.encode(), n)
        return self._lib.vop1_ex(op, dst.encode(), a.encode(), n, _mode_to_int(mode))

    def op2(self, op: int, dst: str, a: str, b: str, n: int, mode=None):
        """Binary op: dst = a op b"""
        if mode is None:
            return self._lib.vop2(op, dst.encode(), a.encode(), b.encode(), n)
        return self._lib.vop2_ex(op, dst.encode(), a.encode(), b.encode(), n, _mode_to_int(mode))

    def ops(self, op: int, dst: str, a: str, scalar: float, n: int, mode=None):
        """Scalar op: dst = a op scalar"""
        if mode is None:
            return self._lib.vops(op, dst.encode(), a.encode(), ctypes.c_float(scalar), n)
        return self._lib.vops_ex(op, dst.encode(), a.encode(), ctypes.c_float(scalar), n, _mode_to_int(mode))
    
    def reduce(self, op: int, dst: str, a: str, n: int):
        """Reduce: dst[0] = reduce(a)"""
        return self._lib.vreduce(op, dst.encode(), a.encode(), n)
    
    def dot(self, dst: str, a: str, b: str, n: int):
        """Dot product"""
        return self._lib.vdot(dst.encode(), a.encode(), b.encode(), n)
    
    def softmax(self, buf: str, n: int):
        """In-place softmax"""
        return self._lib.vsoftmax(buf.encode(), n)
    
    def matvec(self, dst: str, mat: str, vec: str, rows: int, cols: int):
        """Matrix-vector multiply"""
        return self._lib.vmatvec(dst.encode(), mat.encode(), vec.encode(), rows, cols)
    
    def cumsum(self, dst: str, a: str, n: int):
        return self._lib.vcumsum(dst.encode(), a.encode(), n)
    
    def diff(self, dst: str, a: str, n: int):
        return self._lib.vdiff(dst.encode(), a.encode(), n)
    
    def integrate(self, dst: str, a: str, dx: float, n: int):
        return self._lib.vintegrate(dst.encode(), a.encode(), ctypes.c_float(dx), n)
    
    def derivative(self, dst: str, a: str, dx: float, n: int):
        return self._lib.vderivative(dst.encode(), a.encode(), ctypes.c_float(dx), n)
    
    def linspace(self, dst: str, start: float, stop: float, n: int):
        return self._lib.vlinspace(dst.encode(), ctypes.c_float(start), ctypes.c_float(stop), n)
    
    def arange(self, dst: str, start: float, step: float, n: int):
        return self._lib.varange(dst.encode(), ctypes.c_float(start), ctypes.c_float(step), n)
    
    # Convenience shortcuts
    def sin(self, dst, a, n, mode=None): return self.op1(SIN, dst, a, n, mode)
    def cos(self, dst, a, n, mode=None): return self.op1(COS, dst, a, n, mode)
    def exp(self, dst, a, n, mode=None): return self.op1(EXP, dst, a, n, mode)
    def log(self, dst, a, n, mode=None): return self.op1(LOG, dst, a, n, mode)
    def tanh(self, dst, a, n, mode=None): return self.op1(TANH, dst, a, n, mode)
    def relu(self, dst, a, n, mode=None): return self.op1(RELU, dst, a, n, mode)
    def sqrt(self, dst, a, n, mode=None): return self.op1(SQRT, dst, a, n, mode)
    def add(self, dst, a, b, n, mode=None): return self.op2(ADD, dst, a, b, n, mode)
    def sub(self, dst, a, b, n, mode=None): return self.op2(SUB, dst, a, b, n, mode)
    def mul(self, dst, a, b, n, mode=None): return self.op2(MUL, dst, a, b, n, mode)
    def div(self, dst, a, b, n, mode=None): return self.op2(DIV, dst, a, b, n, mode)
    
    # Maps
    def map_init(self, id: int, word_size: int, pack_size: int, n_packs: int):
        return self._lib.map_init(id, word_size, pack_size, n_packs)
    
    def map_destroy(self, id: int):
        return self._lib.map_destroy(id)
    
    def scatter(self, id: int, locs: str, vals: str, n: int):
        return self._lib.mscatter(id, locs.encode(), vals.encode(), n)
    
    def gather(self, id: int, locs: str, dst: str, n: int):
        return self._lib.mgather(id, locs.encode(), dst.encode(), n)


# Simple functional API
_ctx = None

def init():
    """Initialize ADAMAH"""
    global _ctx
    _ctx = Adamah()
    return _ctx

def shutdown():
    """Shutdown ADAMAH"""
    global _ctx
    if _ctx:
        _ctx.shutdown()
        _ctx = None

def set_mode(mode):
    """Set execution mode: 'cpu', 'gpu', or 'auto'"""
    return _ctx.set_mode(mode)

def get_mode():
    """Get current execution mode"""
    return _ctx.get_mode()

def put(name, data):
    """Inject array"""
    return _ctx.put(name, data)

def get(name, count=None):
    """Extract array"""
    return _ctx.get(name, count)

# Re-export common ops
def sin(dst, a, n): return _ctx.sin(dst, a, n)
def cos(dst, a, n): return _ctx.cos(dst, a, n)
def exp(dst, a, n): return _ctx.exp(dst, a, n)
def log(dst, a, n): return _ctx.log(dst, a, n)
def tanh(dst, a, n): return _ctx.tanh(dst, a, n)
def add(dst, a, b, n): return _ctx.add(dst, a, b, n)
def mul(dst, a, b, n): return _ctx.mul(dst, a, b, n)
def dot(dst, a, b, n): return _ctx.dot(dst, a, b, n)
def softmax(buf, n): return _ctx.softmax(buf, n)
def linspace(dst, start, stop, n): return _ctx.linspace(dst, start, stop, n)
