"""
ADAMAH v4.0 - Map-Centric GPU Compute

Pure GPU operations on Memory Maps.
scatter/gather for CPU I/O.

Usage:
    import adamah
    
    gpu = adamah.init()
    
    # Create map: 1M packs of 128 floats
    gpu.map_init(0, word_size=4, pack_size=128, n_packs=1_000_000)
    
    # Write data to map
    gpu.scatter(0, locs, data)
    
    # GPU operations (coming soon)
    # gpu.map_op("sin", 0, locs_in, locs_out)
    
    # Read data from map
    result = gpu.gather(0, locs, n_packs)
    
    gpu.shutdown()

CC BY-NC 4.0 - Samuele Scuglia - 2026-01-18
"""

import ctypes
import numpy as np
import os
import subprocess
from enum import IntEnum
from typing import Dict, Tuple, Any, Optional

__version__ = "4.0.0"
__author__ = "Samuele Scuglia"

# Enums for better type safety and readability
class UnaryOp(IntEnum):
    NEG = 0
    ABS = 1
    SQRT = 2
    EXP = 3
    LOG = 4
    TANH = 5
    RELU = 6
    GELU = 7
    SIN = 8
    COS = 9
    RECIP = 10
    SQR = 11

class BinaryOp(IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    POW = 4
    MIN = 5
    MAX = 6

class ReduceOp(IntEnum):
    SUM = 0
    MAX = 1
    MIN = 2

class BroadcastOp(IntEnum):
    MUL = 0
    DIV = 1
    ADD = 2
    SUB = 3

# Backwards compatibility aliases
OP_NEG, OP_ABS, OP_SQRT, OP_EXP, OP_LOG = UnaryOp.NEG, UnaryOp.ABS, UnaryOp.SQRT, UnaryOp.EXP, UnaryOp.LOG
OP_TANH, OP_RELU, OP_GELU, OP_SIN, OP_COS = UnaryOp.TANH, UnaryOp.RELU, UnaryOp.GELU, UnaryOp.SIN, UnaryOp.COS
OP_RECIP, OP_SQR = UnaryOp.RECIP, UnaryOp.SQR

OP_ADD, OP_SUB, OP_MUL, OP_DIV = BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV
OP_POW, OP_MIN, OP_MAX = BinaryOp.POW, BinaryOp.MIN, BinaryOp.MAX

REDUCE_SUM, REDUCE_MAX, REDUCE_MIN = ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN
BROADCAST_MUL, BROADCAST_DIV, BROADCAST_ADD, BROADCAST_SUB = BroadcastOp.MUL, BroadcastOp.DIV, BroadcastOp.ADD, BroadcastOp.SUB

_lib = None

def _compile_native_lib(c_path: str, so_path: str, pkg_dir: str) -> None:
    """Compiles the C library using GCC."""
    print("ADAMAH: Compiling native library...")
    shader_dir = os.path.join(pkg_dir, "shaders")
    cmd = [
        'gcc', '-shared', '-fPIC', '-O3', '-march=native',
        '-DSHADER_PATH="%s"' % shader_dir,
        c_path, '-o', so_path, '-lvulkan', '-lm'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            raise RuntimeError(f"Compile error: {result.stderr}\nMake sure you have GCC and Vulkan SDK installed.")
    except FileNotFoundError:
        raise RuntimeError("GCC not found. Please install build-essential.")
    print(f"ADAMAH: Compiled to {so_path}")

def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    
    pkg_dir = os.path.dirname(__file__)
    c_path = os.path.join(pkg_dir, "adamah.c")
    
    # Try multiple locations for .so (pip install may be read-only)
    possible_so_paths = [
        os.path.join(pkg_dir, "libadamah.so"),  # Package dir
        os.path.join(os.path.expanduser("~"), ".cache", "adamah", "libadamah.so"),  # User cache
        os.path.join("/tmp", "adamah", "libadamah.so"),  # Temp fallback
    ]
    
    so_path = None
    for path in possible_so_paths:
        if os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(c_path):
            so_path = path
            break
    
    if so_path is None:
        for path in possible_so_paths:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path + ".test", "w") as f:
                    f.write("test")
                os.remove(path + ".test")
                so_path = path
                break
            except (OSError, IOError):
                continue
        
        if so_path is None:
            raise RuntimeError("Cannot find writable location for compiled library")
        
        _compile_native_lib(c_path, so_path, pkg_dir)
    
    _lib = ctypes.CDLL(so_path)
    
    # Function signatures
    _lib.adamah_init.restype = ctypes.c_int
    _lib.adamah_shutdown.restype = None
    
    _lib.map_init.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    _lib.map_init.restype = ctypes.c_int
    _lib.map_destroy.argtypes = [ctypes.c_uint32]
    _lib.map_destroy.restype = ctypes.c_int
    _lib.map_size.argtypes = [ctypes.c_uint32]
    _lib.map_size.restype = ctypes.c_uint64
    
    _lib.mscatter.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p, ctypes.c_uint32]
    _lib.mscatter.restype = ctypes.c_int
    _lib.mgather.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p, ctypes.c_uint32]
    _lib.mgather.restype = ctypes.c_int
    
    _lib.map_save.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
    _lib.map_save.restype = ctypes.c_int
    _lib.map_load.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
    _lib.map_load.restype = ctypes.c_int
    
    _lib.map_op1.argtypes = [ctypes.c_uint32, ctypes.c_uint32, 
                             ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    _lib.map_op1.restype = ctypes.c_int
    _lib.map_op2.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                             ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                             ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    _lib.map_op2.restype = ctypes.c_int
    
    # Matmul: map_matmul(map_id, locs_a, locs_b, locs_c, M, K, N, n_ops)
    _lib.map_matmul.argtypes = [ctypes.c_uint32,
                                ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                ctypes.POINTER(ctypes.c_uint32),
                                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    _lib.map_matmul.restype = ctypes.c_int
    
    # Reduce: map_reduce(map_id, op, locs_src, locs_dst, n)
    _lib.map_reduce.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                ctypes.c_uint32]
    _lib.map_reduce.restype = ctypes.c_int
    
    # Broadcast: map_broadcast(map_id, op, locs_src, locs_scalar, locs_dst, n)
    _lib.map_broadcast.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                   ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                   ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    _lib.map_broadcast.restype = ctypes.c_int
    
    # Batching: batch_begin() and batch_end()
    _lib.batch_begin.argtypes = []
    _lib.batch_begin.restype = None
    _lib.batch_end.argtypes = []
    _lib.batch_end.restype = None
    
    # Softmax: map_softmax(map_id, locs_src, locs_dst, n_rows, row_size)
    _lib.map_softmax.argtypes = [ctypes.c_uint32,
                                 ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                 ctypes.c_uint32, ctypes.c_uint32]
    _lib.map_softmax.restype = ctypes.c_int
    
    # LayerNorm: map_layernorm(map_id, locs_src, locs_dst, locs_gamma, locs_beta, n_rows, dim, eps)
    _lib.map_layernorm.argtypes = [ctypes.c_uint32,
                                   ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                   ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                   ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float]
    _lib.map_layernorm.restype = ctypes.c_int

    # Sync
    _lib.adamah_sync.argtypes = []
    _lib.adamah_sync.restype = None

    # Unified FFN: adamah_fused_ffn(map_id, out, x, w1, b1, w2, b2, BT, D, apply_residual)
    _lib.adamah_fused_ffn.argtypes = [ctypes.c_uint32,
                                      ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    _lib.adamah_fused_ffn.restype = ctypes.c_int
    
    return _lib


class Adamah:
    """ADAMAH GPU Context"""
    
    def __init__(self):
        self._lib = _get_lib()
        ret = self._lib.adamah_init()
        if ret != 0:
            raise RuntimeError(f"Failed to initialize ADAMAH: {ret}")
        self._maps: Dict[int, Tuple[int, int, int, Any]] = {}  # id -> (word_size, pack_size, n_packs, dtype)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
    
    def shutdown(self):
        if self._lib:
            self._lib.adamah_shutdown()
            self._lib = None
    
    # ============================================
    # Batching - accumulate GPU commands
    # ============================================
    
    def batch_begin(self):
        """Start batching mode. GPU commands are accumulated without sync."""
        self._lib.batch_begin()
    
    def batch_end(self):
        """End batching mode. Submit all accumulated commands and sync."""
        self._lib.batch_end()

    def sync(self):
        """Block until all queued GPU work completes."""
        self._lib.adamah_sync()
    
    # ============================================
    # Memory Maps
    # ============================================
    
    def map_init(self, map_id: int, word_size: int = 4, pack_size: int = 1, n_packs: int = 1000000):
        """Create a memory map.
        
        Args:
            map_id: Map identifier (0-15)
            word_size: Bytes per element (4 = float32)
            pack_size: Elements per pack
            n_packs: Number of packs
            
        Total size = word_size * pack_size * n_packs bytes
        """
        ret = self._lib.map_init(map_id, word_size, pack_size, n_packs)
        if ret == 0:
            # Pre-calculate dtype to avoid checks in hot path
            if word_size == 4: dtype = np.float32
            elif word_size == 8: dtype = np.float64
            else: dtype = np.uint8
            self._maps[map_id] = (word_size, pack_size, n_packs, dtype)
        return ret
    
    def map_destroy(self, map_id: int):
        """Destroy a memory map."""
        ret = self._lib.map_destroy(map_id)
        if ret == 0 and map_id in self._maps:
            del self._maps[map_id]
        return ret
    
    def map_size(self, map_id: int) -> int:
        """Get number of packs in map."""
        return self._lib.map_size(map_id)
    
    # ============================================
    # Scatter / Gather
    # ============================================
    
    def scatter(self, map_id: int, locs, data):
        """Write data to map at specified locations.
        
        Args:
            map_id: Target map
            locs: Array of pack indices (uint32)
            data: Data to write (flattened, n_locs * pack_size elements)
        """
        if map_id not in self._maps:
            raise ValueError(f"Map {map_id} not initialized")
        
        word_size, pack_size, _, dtype = self._maps[map_id]
        
        locs = np.ascontiguousarray(locs, dtype=np.uint32)
        data = np.ascontiguousarray(data, dtype=dtype)
        
        locs_ptr = locs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        data_ptr = data.ctypes.data_as(ctypes.c_void_p)
        
        return self._lib.mscatter(map_id, locs_ptr, data_ptr, len(locs))
    
    def gather(self, map_id: int, locs, n_locs: int = None):
        """Read data from map at specified locations.
        
        Args:
            map_id: Source map
            locs: Array of pack indices (uint32)
            n_locs: Number of locations (optional, inferred from locs)
            
        Returns:
            numpy array of shape (n_locs * pack_size,)
        """
        if map_id not in self._maps:
            raise ValueError(f"Map {map_id} not initialized")
        
        word_size, pack_size, _, dtype = self._maps[map_id]
        
        locs = np.ascontiguousarray(locs, dtype=np.uint32)
        if n_locs is None:
            n_locs = len(locs)
        
        out = np.zeros(n_locs * pack_size, dtype=dtype)
        
        locs_ptr = locs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        out_ptr = out.ctypes.data_as(ctypes.c_void_p)
        
        self._lib.mgather(map_id, locs_ptr, out_ptr, n_locs)
        return out
    
    # ============================================
    # Persistence
    # ============================================
    
    def map_save(self, map_id: int, path: str):
        """Save map to file."""
        return self._lib.map_save(map_id, path.encode())
    
    def map_load(self, map_id: int, path: str):
        """Load map from file."""
        # Read header to get map info
        try:
            with open(path, 'rb') as f:
                import struct
                ws, ps, np = struct.unpack('III', f.read(12))
        except:
            return -1
        
        ret = self._lib.map_load(map_id, path.encode())
        if ret == 0:
            if ws == 4: dtype = np.float32
            elif ws == 8: dtype = np.float64
            else: dtype = np.uint8
            self._maps[map_id] = (ws, ps, np, dtype)
        return ret
    
    # ============================================
    # Map Operations (Pure GPU)
    # ============================================
    
    def map_op1(self, map_id: int, op: int, locs_src, locs_dst):
        """Unary operation on map: map[locs_dst] = op(map[locs_src])
        
        Args:
            map_id: Target map
            op: Operation code (OP_SIN, OP_COS, OP_EXP, etc.)
            locs_src: Source pack indices
            locs_dst: Destination pack indices
        """
        locs_src = np.ascontiguousarray(locs_src, dtype=np.uint32)
        locs_dst = np.ascontiguousarray(locs_dst, dtype=np.uint32)
        
        src_ptr = locs_src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        dst_ptr = locs_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        
        return self._lib.map_op1(map_id, op, src_ptr, dst_ptr, len(locs_src))
    
    def map_op2(self, map_id: int, op: int, locs_a, locs_b, locs_dst):
        """Binary operation on map: map[locs_dst] = map[locs_a] op map[locs_b]
        
        Args:
            map_id: Target map
            op: Operation code (OP_ADD, OP_MUL, etc.)
            locs_a: First operand pack indices
            locs_b: Second operand pack indices
            locs_dst: Destination pack indices
        """
        locs_a = np.ascontiguousarray(locs_a, dtype=np.uint32)
        locs_b = np.ascontiguousarray(locs_b, dtype=np.uint32)
        locs_dst = np.ascontiguousarray(locs_dst, dtype=np.uint32)
        
        a_ptr = locs_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        b_ptr = locs_b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        dst_ptr = locs_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        
        return self._lib.map_op2(map_id, op, a_ptr, b_ptr, dst_ptr, len(locs_a))
    
    # Convenience shortcuts
    def map_sin(self, map_id, locs_src, locs_dst):
        return self.map_op1(map_id, OP_SIN, locs_src, locs_dst)
    
    def map_cos(self, map_id, locs_src, locs_dst):
        return self.map_op1(map_id, OP_COS, locs_src, locs_dst)
    
    def map_exp(self, map_id, locs_src, locs_dst):
        return self.map_op1(map_id, OP_EXP, locs_src, locs_dst)
    
    def map_add(self, map_id, locs_a, locs_b, locs_dst):
        return self.map_op2(map_id, OP_ADD, locs_a, locs_b, locs_dst)
    
    def map_mul(self, map_id, locs_a, locs_b, locs_dst):
        return self.map_op2(map_id, OP_MUL, locs_a, locs_b, locs_dst)
    
    # ============================================
    # Matrix Multiplication
    # ============================================
    
    def map_matmul(self, map_id: int, locs_a, locs_b, locs_c, M: int, K: int, N: int):
        """Matrix multiplication: C = A @ B
        
        Each location points to a flattened matrix in the map:
        - A: M x K matrix (M*K floats starting at locs_a[i] * pack_size)
        - B: K x N matrix
        - C: M x N matrix (output)
        
        Args:
            map_id: Map containing all matrices
            locs_a: Pack indices for A matrices
            locs_b: Pack indices for B matrices  
            locs_c: Pack indices for C matrices (output)
            M, K, N: Matrix dimensions
        """
        locs_a = np.ascontiguousarray(locs_a, dtype=np.uint32)
        locs_b = np.ascontiguousarray(locs_b, dtype=np.uint32)
        locs_c = np.ascontiguousarray(locs_c, dtype=np.uint32)
        
        a_ptr = locs_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        b_ptr = locs_b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        c_ptr = locs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        
        return self._lib.map_matmul(map_id, a_ptr, b_ptr, c_ptr, M, K, N, len(locs_a))
    
    # ============================================
    # Reduce Operations
    # ============================================
    
    def map_reduce(self, map_id: int, op: int, locs_src, locs_dst):
        """Reduce operation along pack dimension.
        
        Reduces each pack to a single scalar value.
        Output is written to locs_dst (as single float, not pack).
        
        Args:
            map_id: Source map
            op: REDUCE_SUM, REDUCE_MAX, or REDUCE_MIN
            locs_src: Source pack indices
            locs_dst: Destination indices (for scalars)
        """
        locs_src = np.ascontiguousarray(locs_src, dtype=np.uint32)
        locs_dst = np.ascontiguousarray(locs_dst, dtype=np.uint32)
        
        src_ptr = locs_src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        dst_ptr = locs_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        
        return self._lib.map_reduce(map_id, op, src_ptr, dst_ptr, len(locs_src))
    
    def map_reduce_sum(self, map_id, locs_src, locs_dst):
        return self.map_reduce(map_id, REDUCE_SUM, locs_src, locs_dst)
    
    def map_reduce_max(self, map_id, locs_src, locs_dst):
        return self.map_reduce(map_id, REDUCE_MAX, locs_src, locs_dst)
    
    def map_reduce_min(self, map_id, locs_src, locs_dst):
        return self.map_reduce(map_id, REDUCE_MIN, locs_src, locs_dst)
    
    # ============================================
    # Broadcast Operations
    # ============================================
    
    def map_broadcast(self, map_id: int, op: int, locs_src, locs_scalar, locs_dst):
        """Broadcast scalar to all elements of pack.
        
        dst[i] = src[i] op scalar for each element in pack.
        
        Args:
            map_id: Map
            op: BROADCAST_MUL, BROADCAST_DIV, BROADCAST_ADD, BROADCAST_SUB
            locs_src: Source pack indices
            locs_scalar: Indices where scalars are stored (single float each)
            locs_dst: Destination pack indices
        """
        locs_src = np.ascontiguousarray(locs_src, dtype=np.uint32)
        locs_scalar = np.ascontiguousarray(locs_scalar, dtype=np.uint32)
        locs_dst = np.ascontiguousarray(locs_dst, dtype=np.uint32)
        
        src_ptr = locs_src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        scalar_ptr = locs_scalar.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        dst_ptr = locs_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        
        return self._lib.map_broadcast(map_id, op, src_ptr, scalar_ptr, dst_ptr, len(locs_src))
    
    def map_scale(self, map_id, locs_src, locs_scalar, locs_dst):
        """Multiply each pack by a scalar: dst = src * scalar"""
        return self.map_broadcast(map_id, BROADCAST_MUL, locs_src, locs_scalar, locs_dst)
    
    def map_div_scalar(self, map_id, locs_src, locs_scalar, locs_dst):
        """Divide each pack by a scalar: dst = src / scalar"""
        return self.map_broadcast(map_id, BROADCAST_DIV, locs_src, locs_scalar, locs_dst)
    
    # ============================================
    # Softmax (fused)
    # ============================================
    
    def map_softmax(self, map_id: int, locs_src, locs_dst, row_size: int):
        """Fused softmax: max, subtract, exp, sum, normalize.
        
        Each location points to the start of a row of row_size floats.
        Output: softmax(input) for each row.
        
        Args:
            map_id: Map containing data
            locs_src: Start index of each input row
            locs_dst: Start index of each output row
            row_size: Number of elements per row
        """
        locs_src = np.ascontiguousarray(locs_src, dtype=np.uint32)
        locs_dst = np.ascontiguousarray(locs_dst, dtype=np.uint32)
        n_rows = len(locs_src)
        
        src_ptr = locs_src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        dst_ptr = locs_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        
        return self._lib.map_softmax(map_id, src_ptr, dst_ptr, n_rows, row_size)
    
    # ============================================
    # LayerNorm (fused)
    # ============================================
    
    def map_layernorm(self, map_id: int, locs_src, locs_dst, locs_gamma, locs_beta, 
                      dim: int, eps: float = 1e-5):
        """Fused LayerNorm: mean, variance, normalize, scale, shift.
        
        Each location points to a vector of 'dim' floats.
        gamma and beta are learnable parameters (one per row or shared).
        
        Args:
            map_id: Map containing data
            locs_src: Start index of each input vector
            locs_dst: Start index of each output vector
            locs_gamma: Start index of gamma weights for each row
            locs_beta: Start index of beta weights for each row
            dim: Dimension of each vector
            eps: Epsilon for numerical stability
        """
        locs_src = np.ascontiguousarray(locs_src, dtype=np.uint32)
        locs_dst = np.ascontiguousarray(locs_dst, dtype=np.uint32)
        locs_gamma = np.ascontiguousarray(locs_gamma, dtype=np.uint32)
        locs_beta = np.ascontiguousarray(locs_beta, dtype=np.uint32)
        n_rows = len(locs_src)
        
        src_ptr = locs_src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        dst_ptr = locs_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        gamma_ptr = locs_gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        beta_ptr = locs_beta.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        
        return self._lib.map_layernorm(map_id, src_ptr, dst_ptr, gamma_ptr, beta_ptr,
                                       n_rows, dim, eps)

    # ============================================
    # Unified FFN (MLP)
    # ============================================
    def fused_ffn(self, map_id: int, out: int, x: int, w1: int, b1: int, w2: int, b2: int,
                  BT: int, D: int, apply_residual: int = 0):
        """Fused FFN: (X*W1 + b1) -> GELU -> (H*W2 + b2) (+ residual)."""
        return self._lib.adamah_fused_ffn(map_id, out, x, w1, b1, w2, b2, BT, D, apply_residual)


# Convenience function
def init():
    """Initialize ADAMAH and return context."""
    return Adamah()
