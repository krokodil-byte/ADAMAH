"""
ADAMAH v5.0 - GPU Compute Library

Clean API with auto-batching and lazy execution.

Usage:
    gpu = adamah.Adamah()

    # Arrays (handle-based) - WIP
    # x = gpu.array([1, 2, 3, 4])
    # y = gpu.array([5, 6, 7, 8])
    # z = gpu.add(x, y)  # Lazy - auto-batched
    # result = z.numpy()  # Sync + download

    # Maps (for existing code)
    gpu.map_create(0, 4, 128, 1000)
    gpu.scatter(0, locs, data)
    gpu.map_op2(0, OP_ADD, locs_a, locs_b, locs_out, n)
    result = gpu.gather(0, locs)

CC BY-NC 4.0 - Samuele Scuglia - 2026-01-24
"""

import ctypes
import numpy as np
import os
import struct
from contextlib import contextmanager
from typing import Optional, Tuple

__version__ = "5.0.0"

# Import UUCIS wrapper for benchmark compatibility
try:
    from .uucis import UUCISView
except ImportError:
    UUCISView = None

# Operation codes
OP_NEG = 0
OP_ABS = 1
OP_SQRT = 2
OP_EXP = 3
OP_LOG = 4
OP_TANH = 5
OP_RELU = 6
OP_GELU = 7
OP_SIN = 8
OP_COS = 9

OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_DIV = 3

# ============================================
# Helpers
# ============================================

def _unpack_ticket_handle(v: int) -> Tuple[int, int]:
    handle = v & 0xFFFFFFFF
    ticket = (v >> 32) & 0xFFFFFFFFFFFFFFFF
    return handle, ticket

# ============================================
# ArrayHandle - GPU array wrapper (WIP)
# ============================================

class ArrayHandle:
    """GPU array handle (for lazy execution API)."""

    def __init__(self, gpu, handle_id: int, shape, dtype='float32'):
        self.gpu = gpu
        self.id = handle_id
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dtype = np.dtype(dtype)
        self._freed = False

    def numpy(self) -> np.ndarray:
        """Download to numpy (triggers sync)."""
        if self._freed:
            raise RuntimeError("Cannot download freed array")
        self.gpu.sync()
        # WIP: implement array download
        raise NotImplementedError("ArrayHandle.numpy() is WIP")

    def free(self):
        """Explicitly free GPU memory."""
        if not self._freed:
            # WIP: implement array free
            self._freed = True

    def __del__(self):
        if not self._freed:
            try:
                self.free()
            except:
                pass

# ============================================
# Main Adamah Class
# ============================================

class Adamah:
    """ADAMAH GPU compute library."""

    def __init__(self, lib_path: Optional[str] = None, cache_mb: Optional[int] = None, cold_cache_mb: Optional[int] = None):
        """Initialize ADAMAH library."""
        if lib_path is None:
            # Auto-detect library path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(current_dir, 'adamah.so')

            if not os.path.exists(lib_path):
                raise FileNotFoundError(
                    f"adamah.so not found at {lib_path}\n"
                    f"Please build adamah.so and place it in this directory"
                )

        self._lib = ctypes.CDLL(lib_path)
        self._setup_ctypes()

        # Initialize Vulkan / cache pools
        if cache_mb is not None or cold_cache_mb is not None:
            init_ex = getattr(self._lib, "adamah_init_ex", None)
            if init_ex is None:
                ret = self._lib.adamah_init()
            else:
                hot_mb = int(cache_mb if cache_mb is not None else cold_cache_mb)
                cold_mb = int(cold_cache_mb if cold_cache_mb is not None else hot_mb)
                ret = init_ex(ctypes.c_uint64(hot_mb * 1024 * 1024), ctypes.c_uint64(cold_mb * 1024 * 1024))
        else:
            ret = self._lib.adamah_init()
        if ret != 0:
            raise RuntimeError(f"adamah_init failed with code {ret}")

        # Metrics
        self._metrics = {
            'gather_calls': 0,
            'scatter_calls': 0,
            'op_calls': 0,
            'total_bytes_cpu_to_gpu': 0,
            'total_bytes_gpu_to_cpu': 0,
        }

        # Map metadata (for UUCIS compatibility)
        self._maps = {}

    def _setup_ctypes(self):
        """Configure ctypes function signatures."""
        # Init/shutdown
        self._lib.adamah_init.argtypes = []
        self._lib.adamah_init.restype = ctypes.c_int
        init_ex = getattr(self._lib, "adamah_init_ex", None)
        if init_ex is not None:
            init_ex.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
            init_ex.restype = ctypes.c_int

        self._lib.adamah_shutdown.argtypes = []
        self._lib.adamah_shutdown.restype = None

        self._lib.adamah_sync.argtypes = []
        self._lib.adamah_sync.restype = None

        # Map operations
        self._lib.map_init.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_init.restype = ctypes.c_int

        self._lib.map_destroy.argtypes = [ctypes.c_uint32]
        self._lib.map_destroy.restype = ctypes.c_int

        self._lib.map_size.argtypes = [ctypes.c_uint32]
        self._lib.map_size.restype = ctypes.c_uint64

        self._lib.map_save.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
        self._lib.map_save.restype = ctypes.c_int

        self._lib.map_load.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
        self._lib.map_load.restype = ctypes.c_int

        # Scatter/gather
        self._lib.map_scatter.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32
        ]
        self._lib.map_scatter.restype = ctypes.c_uint64

        self._lib.map_gather.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32
        ]
        self._lib.map_gather.restype = ctypes.c_uint64

        # Operations
        self._lib.map_op1.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32
        ]
        self._lib.map_op1.restype = ctypes.c_int

        self._lib.map_op2.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32
        ]
        self._lib.map_op2.restype = ctypes.c_int

        self._lib.map_matmul.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32
        ]
        self._lib.map_matmul.restype = ctypes.c_int

        self._lib.map_softmax.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32, ctypes.c_uint32
        ]
        self._lib.map_softmax.restype = ctypes.c_int

        self._lib.map_layernorm.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float
        ]
        self._lib.map_layernorm.restype = ctypes.c_int

        self._lib.map_broadcast.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32
        ]
        self._lib.map_broadcast.restype = ctypes.c_int

        self._lib.map_reduce.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32
        ]
        self._lib.map_reduce.restype = ctypes.c_int

        # Device-only async helpers
        self._lib.map_upload_dev.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        self._lib.map_upload_dev.restype = ctypes.c_uint64
        self._lib.map_download_dev.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        self._lib.map_download_dev.restype = ctypes.c_int
        self._lib.map_gather_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_gather_dev.restype = ctypes.c_uint64
        self._lib.map_scatter_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_scatter_dev.restype = ctypes.c_uint64

        # Device-locs ops
        self._lib.map_op1_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                          ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_op1_dev.restype = ctypes.c_int
        self._lib.map_op2_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                          ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_op2_dev.restype = ctypes.c_int
        self._lib.map_matmul_dev.argtypes = [ctypes.c_uint32,
                                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_matmul_dev.restype = ctypes.c_int
        self._lib.map_softmax_dev.argtypes = [ctypes.c_uint32,
                                              ctypes.c_uint32, ctypes.c_uint32,
                                              ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_softmax_dev.restype = ctypes.c_int
        self._lib.map_layernorm_dev.argtypes = [ctypes.c_uint32,
                                                ctypes.c_uint32, ctypes.c_uint32,
                                                ctypes.c_uint32, ctypes.c_uint32,
                                                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float]
        self._lib.map_layernorm_dev.restype = ctypes.c_int
        self._lib.map_broadcast_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                                ctypes.c_uint32, ctypes.c_uint32,
                                                ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_broadcast_dev.restype = ctypes.c_int
        self._lib.map_reduce_dev.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                             ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self._lib.map_reduce_dev.restype = ctypes.c_int

        # Sync for async tickets
        self._lib.adamah_synchronize.argtypes = [ctypes.c_uint64]
        self._lib.adamah_synchronize.restype = None
        self._lib.adamah_synchronize_all.argtypes = []
        self._lib.adamah_synchronize_all.restype = None

        # Batching
        self._lib.batch_begin.argtypes = []
        self._lib.batch_begin.restype = None

        self._lib.batch_end.argtypes = []
        self._lib.batch_end.restype = None

    # ============================================
    # Map Operations (for compatibility)
    # ============================================

    def map_create(self, map_id: int, word_size: int, pack_size: int, n_packs: int):
        """Create memory map."""
        ret = self._lib.map_init(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(word_size),
            ctypes.c_uint32(pack_size),
            ctypes.c_uint32(n_packs)
        )
        if ret != 0:
            raise RuntimeError(f"map_init failed with code {ret}")

        # Store metadata for UUCIS (as tuple)
        dtype = np.float32 if word_size == 4 else np.float64
        self._maps[map_id] = (word_size, pack_size, n_packs, dtype)

    # Alias for compatibility
    def map_init(self, map_id: int, word_size: int, pack_size: int, n_packs: int):
        """Alias for map_create (compatibility)."""
        return self.map_create(map_id, word_size, pack_size, n_packs)

    def map_destroy(self, map_id: int):
        """Destroy memory map."""
        ret = self._lib.map_destroy(ctypes.c_uint32(map_id))
        if ret != 0:
            raise RuntimeError(f"map_destroy failed with code {ret}")

    def map_size(self, map_id: int) -> int:
        """Get map size in packs."""
        return int(self._lib.map_size(ctypes.c_uint32(map_id)))

    def map_save(self, map_id: int, path: str):
        """Save a map to disk."""
        path_b = os.fspath(path).encode()
        ret = self._lib.map_save(ctypes.c_uint32(map_id), ctypes.c_char_p(path_b))
        if ret != 0:
            raise RuntimeError(f"map_save failed with code {ret}")

    def map_load(self, map_id: int, path: str):
        """Load a map from disk (updates metadata)."""
        path_s = os.fspath(path)
        with open(path_s, "rb") as f:
            header = f.read(12)
        if len(header) != 12:
            raise RuntimeError("map_load failed: invalid header")
        word_size, pack_size, n_packs = struct.unpack("<III", header)
        ret = self._lib.map_load(ctypes.c_uint32(map_id), ctypes.c_char_p(path_s.encode()))
        if ret != 0:
            raise RuntimeError(f"map_load failed with code {ret}")
        if word_size == 4:
            dtype = np.float32
        elif word_size == 8:
            dtype = np.float64
        else:
            dtype = np.uint8
        self._maps[map_id] = (word_size, pack_size, n_packs, dtype)

    def scatter(self, map_id: int, locs: np.ndarray, data: np.ndarray) -> int:
        """Write data to map at specified locations."""
        self._metrics['scatter_calls'] += 1
        self._metrics['total_bytes_cpu_to_gpu'] += data.nbytes

        # Ensure contiguous arrays
        locs = np.ascontiguousarray(locs, dtype=np.uint32)
        data = np.ascontiguousarray(data, dtype=np.float32)

        n_locs = len(locs)

        ticket = self._lib.map_scatter(
            ctypes.c_uint32(map_id),
            locs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(n_locs)
        )

        return int(ticket)

    # Alias for compatibility
    map_scatter = scatter

    def gather(self, map_id: int, locs: np.ndarray, n_packs: Optional[int] = None, n_locs: Optional[int] = None) -> np.ndarray:
        """Read data from map at specified locations."""
        self._metrics['gather_calls'] += 1

        locs = np.ascontiguousarray(locs, dtype=np.uint32)
        # Accept both n_packs and n_locs (UUCIS compatibility)
        size = n_locs if n_locs is not None else (n_packs if n_packs is not None else len(locs))
        n_locs = size

        # Allocate output
        out = np.empty(n_locs, dtype=np.float32)

        self._lib.map_gather(
            ctypes.c_uint32(map_id),
            locs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(n_locs)
        )

        self._metrics['total_bytes_gpu_to_cpu'] += out.nbytes
        return out

    # Alias for compatibility
    map_gather = gather

    # ============================================
    # Device-only async helpers
    # ============================================

    def upload_dev(self, data, handle: int = 0):
        """Upload CPU data into a device-local buffer. Returns (handle, ticket)."""
        arr = np.ascontiguousarray(data)
        ptr = arr.ctypes.data_as(ctypes.c_void_p)
        packed = self._lib.map_upload_dev(ctypes.c_uint32(handle), ptr, ctypes.c_uint32(arr.nbytes))
        return _unpack_ticket_handle(int(packed))

    def download_dev(self, handle: int, n_elems: int, dtype=np.float32):
        """Download device-local buffer into a numpy array."""
        out = np.empty(n_elems, dtype=dtype)
        ptr = out.ctypes.data_as(ctypes.c_void_p)
        ret = self._lib.map_download_dev(ctypes.c_uint32(handle), ptr, ctypes.c_uint32(out.nbytes))
        if ret != 0:
            raise RuntimeError(f"map_download_dev failed with code {ret}")
        return out

    def map_gather_dev(self, map_id: int, locs_handle: int, n_locs: int):
        """Device-only gather. Returns (dst_handle, ticket, n_elems, dtype)."""
        if map_id not in self._maps:
            raise ValueError(f"Map {map_id} not initialized")
        word_size, pack_size, _, dtype = self._maps[map_id]
        packed = self._lib.map_gather_dev(ctypes.c_uint32(map_id), ctypes.c_uint32(locs_handle), ctypes.c_uint32(n_locs))
        dst_handle, ticket = _unpack_ticket_handle(int(packed))
        return dst_handle, ticket, n_locs * pack_size, dtype

    def map_scatter_dev(self, map_id: int, locs_handle: int, n_locs: int, src_handle: int):
        """Device-only scatter. Returns ticket."""
        if map_id not in self._maps:
            raise ValueError(f"Map {map_id} not initialized")
        packed = self._lib.map_scatter_dev(ctypes.c_uint32(map_id), ctypes.c_uint32(locs_handle),
                                           ctypes.c_uint32(n_locs), ctypes.c_uint32(src_handle))
        return int(packed)

    def synchronize(self, ticket: int):
        """Wait for a specific async ticket."""
        self._lib.adamah_synchronize(ctypes.c_uint64(ticket))

    def synchronize_all(self):
        """Wait for all async submissions."""
        self._lib.adamah_synchronize_all()

    # ============================================
    # GPU Operations
    # ============================================

    def map_op1(self, map_id: int, op: int, locs_in: np.ndarray, locs_out: np.ndarray, n: Optional[int] = None):
        """Unary operation on map."""
        self._metrics['op_calls'] += 1

        locs_in = np.ascontiguousarray(locs_in, dtype=np.uint32)
        locs_out = np.ascontiguousarray(locs_out, dtype=np.uint32)

        # Auto-compute n if not provided (UUCIS compatibility)
        if n is None:
            n = len(locs_in)

        ret = self._lib.map_op1(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            locs_in.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n)
        )

        if ret != 0:
            raise RuntimeError(f"map_op1 failed with code {ret}")

    def map_op1_dev(self, map_id: int, op: int, locs_in_handle: int, locs_out_handle: int, n: int):
        """Unary op using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_op1_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            ctypes.c_uint32(locs_in_handle),
            ctypes.c_uint32(locs_out_handle),
            ctypes.c_uint32(n)
        )
        if ret != 0:
            raise RuntimeError(f"map_op1_dev failed with code {ret}")

    def map_op2(self, map_id: int, op: int, locs_a: np.ndarray, locs_b: np.ndarray,
                locs_out: np.ndarray, n: Optional[int] = None):
        """Binary operation on map."""
        self._metrics['op_calls'] += 1

        locs_a = np.ascontiguousarray(locs_a, dtype=np.uint32)
        locs_b = np.ascontiguousarray(locs_b, dtype=np.uint32)
        locs_out = np.ascontiguousarray(locs_out, dtype=np.uint32)

        # Auto-compute n if not provided (UUCIS compatibility)
        if n is None:
            n = len(locs_a)

        ret = self._lib.map_op2(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            locs_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n)
        )

        if ret != 0:
            raise RuntimeError(f"map_op2 failed with code {ret}")

    def map_op2_dev(self, map_id: int, op: int, locs_a_handle: int, locs_b_handle: int,
                    locs_out_handle: int, n: int):
        """Binary op using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_op2_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            ctypes.c_uint32(locs_a_handle),
            ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(locs_out_handle),
            ctypes.c_uint32(n)
        )
        if ret != 0:
            raise RuntimeError(f"map_op2_dev failed with code {ret}")

    def map_matmul(self, map_id: int, locs_a: np.ndarray, locs_b: np.ndarray,
                   locs_c: np.ndarray, M: int, K: int, N: int, n_ops: int = 1):
        """Matrix multiplication: C = A @ B."""
        self._metrics['op_calls'] += 1

        locs_a = np.ascontiguousarray(locs_a, dtype=np.uint32)
        locs_b = np.ascontiguousarray(locs_b, dtype=np.uint32)
        locs_c = np.ascontiguousarray(locs_c, dtype=np.uint32)

        ret = self._lib.map_matmul(
            ctypes.c_uint32(map_id),
            locs_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(M),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(n_ops)
        )

        if ret != 0:
            raise RuntimeError(f"map_matmul failed with code {ret}")

    def map_matmul_dev(self, map_id: int, locs_a_handle: int, locs_b_handle: int,
                       locs_c_handle: int, M: int, K: int, N: int, n_ops: int = 1):
        """Matmul using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_matmul_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_a_handle),
            ctypes.c_uint32(locs_b_handle),
            ctypes.c_uint32(locs_c_handle),
            ctypes.c_uint32(M),
            ctypes.c_uint32(K),
            ctypes.c_uint32(N),
            ctypes.c_uint32(n_ops)
        )
        if ret != 0:
            raise RuntimeError(f"map_matmul_dev failed with code {ret}")

    def map_softmax(self, map_id: int, locs_in: np.ndarray, locs_out: np.ndarray,
                    row_size: int, n_rows: Optional[int] = None):
        """Softmax operation."""
        self._metrics['op_calls'] += 1

        locs_in = np.ascontiguousarray(locs_in, dtype=np.uint32)
        locs_out = np.ascontiguousarray(locs_out, dtype=np.uint32)

        # Auto-compute n_rows if not provided (UUCIS compatibility)
        if n_rows is None:
            n_rows = len(locs_in)

        ret = self._lib.map_softmax(
            ctypes.c_uint32(map_id),
            locs_in.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n_rows),
            ctypes.c_uint32(row_size)
        )

        if ret != 0:
            raise RuntimeError(f"map_softmax failed with code {ret}")

    def map_softmax_dev(self, map_id: int, locs_in_handle: int, locs_out_handle: int,
                        row_size: int, n_rows: int):
        """Softmax using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_softmax_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_in_handle),
            ctypes.c_uint32(locs_out_handle),
            ctypes.c_uint32(n_rows),
            ctypes.c_uint32(row_size)
        )
        if ret != 0:
            raise RuntimeError(f"map_softmax_dev failed with code {ret}")

    def map_layernorm(self, map_id: int, locs_src: np.ndarray, locs_dst: np.ndarray,
                      locs_gamma: np.ndarray, locs_beta: np.ndarray,
                      dim: int, eps: float = 1e-5, n_rows: Optional[int] = None):
        """Layer normalization."""
        self._metrics['op_calls'] += 1

        locs_src = np.ascontiguousarray(locs_src, dtype=np.uint32)
        locs_dst = np.ascontiguousarray(locs_dst, dtype=np.uint32)
        locs_gamma = np.ascontiguousarray(locs_gamma, dtype=np.uint32)
        locs_beta = np.ascontiguousarray(locs_beta, dtype=np.uint32)

        # Auto-compute n_rows if not provided (UUCIS compatibility)
        if n_rows is None:
            n_rows = len(locs_src)

        ret = self._lib.map_layernorm(
            ctypes.c_uint32(map_id),
            locs_src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_beta.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n_rows),
            ctypes.c_uint32(dim),
            ctypes.c_float(eps)
        )

        if ret != 0:
            raise RuntimeError(f"map_layernorm failed with code {ret}")

    def map_layernorm_dev(self, map_id: int, locs_src_handle: int, locs_dst_handle: int,
                          locs_gamma_handle: int, locs_beta_handle: int,
                          dim: int, eps: float = 1e-5, n_rows: int = 0):
        """LayerNorm using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_layernorm_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(locs_src_handle),
            ctypes.c_uint32(locs_dst_handle),
            ctypes.c_uint32(locs_gamma_handle),
            ctypes.c_uint32(locs_beta_handle),
            ctypes.c_uint32(n_rows),
            ctypes.c_uint32(dim),
            ctypes.c_float(eps)
        )
        if ret != 0:
            raise RuntimeError(f"map_layernorm_dev failed with code {ret}")

    def map_broadcast(self, map_id: int, op: int, locs_src: np.ndarray, locs_scalar: np.ndarray,
                      locs_dst: np.ndarray, n: Optional[int] = None):
        """Broadcast scalar operation (element-wise op with scalar)."""
        self._metrics['op_calls'] += 1

        locs_src = np.ascontiguousarray(locs_src, dtype=np.uint32)
        locs_scalar = np.ascontiguousarray(locs_scalar, dtype=np.uint32)
        locs_dst = np.ascontiguousarray(locs_dst, dtype=np.uint32)

        # Auto-compute n if not provided
        if n is None:
            n = len(locs_src)

        ret = self._lib.map_broadcast(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            locs_src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_scalar.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n)
        )

        if ret != 0:
            raise RuntimeError(f"map_broadcast failed with code {ret}")

    def map_broadcast_dev(self, map_id: int, op: int, locs_src_handle: int, locs_scalar_handle: int,
                          locs_dst_handle: int, n: int):
        """Broadcast using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_broadcast_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            ctypes.c_uint32(locs_src_handle),
            ctypes.c_uint32(locs_scalar_handle),
            ctypes.c_uint32(locs_dst_handle),
            ctypes.c_uint32(n)
        )
        if ret != 0:
            raise RuntimeError(f"map_broadcast_dev failed with code {ret}")

    def map_reduce(self, map_id: int, op: int, locs_src: np.ndarray, locs_dst: np.ndarray,
                   n: Optional[int] = None):
        """Reduce operation (sum/max/min along pack dimension)."""
        self._metrics['op_calls'] += 1

        locs_src = np.ascontiguousarray(locs_src, dtype=np.uint32)
        locs_dst = np.ascontiguousarray(locs_dst, dtype=np.uint32)

        # Auto-compute n if not provided
        if n is None:
            n = len(locs_src)

        ret = self._lib.map_reduce(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            locs_src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            locs_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_uint32(n)
        )

        if ret != 0:
            raise RuntimeError(f"map_reduce failed with code {ret}")

    def map_reduce_dev(self, map_id: int, op: int, locs_src_handle: int, locs_dst_handle: int, n: int):
        """Reduce using device-local locs buffers."""
        self._metrics['op_calls'] += 1
        ret = self._lib.map_reduce_dev(
            ctypes.c_uint32(map_id),
            ctypes.c_uint32(op),
            ctypes.c_uint32(locs_src_handle),
            ctypes.c_uint32(locs_dst_handle),
            ctypes.c_uint32(n)
        )
        if ret != 0:
            raise RuntimeError(f"map_reduce_dev failed with code {ret}")

    # ============================================
    # Batching
    # ============================================

    def batch_begin(self):
        """Begin batching mode (for UUCIS compatibility)."""
        self._lib.batch_begin()

    def batch_end(self):
        """End batching mode (for UUCIS compatibility)."""
        self._lib.batch_end()

    @contextmanager
    def batch(self):
        """Context manager for manual batching."""
        self.batch_begin()
        try:
            yield
        finally:
            self.batch_end()

    def sync(self):
        """Synchronize GPU (wait for all operations)."""
        self._lib.adamah_sync()

    # ============================================
    # Array API (WIP)
    # ============================================

    def array(self, data, target='vram'):
        """Upload numpy array to GPU (returns ArrayHandle)."""
        # WIP: implement handle-based array API
        raise NotImplementedError("array() API is WIP")

    def add(self, a: ArrayHandle, b: ArrayHandle, target='vram'):
        """Element-wise addition (lazy)."""
        raise NotImplementedError("add() API is WIP")

    def mul(self, a: ArrayHandle, b: ArrayHandle, target='vram'):
        """Element-wise multiplication (lazy)."""
        raise NotImplementedError("mul() API is WIP")

    # ============================================
    # Metrics
    # ============================================

    def reset_metrics(self):
        """Reset transfer metrics."""
        for key in self._metrics:
            self._metrics[key] = 0

    def print_metrics(self):
        """Print transfer metrics."""
        print("=== ADAMAH Metrics ===")
        print(f"gather() calls: {self._metrics['gather_calls']}")
        print(f"scatter() calls: {self._metrics['scatter_calls']}")
        print(f"Operation calls: {self._metrics['op_calls']}")
        print(f"CPU→GPU: {self._metrics['total_bytes_cpu_to_gpu'] / 1e6:.2f} MB")
        print(f"GPU→CPU: {self._metrics['total_bytes_gpu_to_cpu'] / 1e6:.2f} MB")

    # ============================================
    # Cleanup
    # ============================================

    def shutdown(self):
        """Shutdown ADAMAH and cleanup resources."""
        self._lib.adamah_shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except:
            pass

    @property
    def uucis(self):
        """UUCIS wrapper API (for benchmark compatibility)."""
        if not hasattr(self, '_uucis'):
            if UUCISView is None:
                raise ImportError("uucis.py not found - UUCIS API not available")
            self._uucis = UUCISView(self)
        return self._uucis

    def __repr__(self):
        return f"Adamah(version={__version__})"


# ============================================
# Convenience API
# ============================================

def init(cache_mb: Optional[int] = None, cold_cache_mb: Optional[int] = None) -> Adamah:
    """Initialize ADAMAH (convenience function)."""
    return Adamah(cache_mb=cache_mb, cold_cache_mb=cold_cache_mb)
