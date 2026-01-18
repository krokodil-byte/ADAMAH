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

__version__ = "4.0.0"
__author__ = "Samuele Scuglia"

# Op codes
OP_NEG, OP_ABS, OP_SQRT, OP_EXP, OP_LOG = 0, 1, 2, 3, 4
OP_TANH, OP_RELU, OP_GELU, OP_SIN, OP_COS = 5, 6, 7, 8, 9
OP_RECIP, OP_SQR = 10, 11

OP_ADD, OP_SUB, OP_MUL, OP_DIV = 0, 1, 2, 3
OP_POW, OP_MIN, OP_MAX = 4, 5, 6

_lib = None

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
    
    # Compile if needed
    if so_path is None:
        # Find writable location
        for path in possible_so_paths:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # Test write
                with open(path + ".test", "w") as f:
                    f.write("test")
                os.remove(path + ".test")
                so_path = path
                break
            except (OSError, IOError):
                continue
        
        if so_path is None:
            raise RuntimeError("Cannot find writable location for compiled library")
        
        print("ADAMAH: Compiling native library...")
        shader_dir = os.path.join(pkg_dir, "shaders")
        cmd = [
            'gcc', '-shared', '-fPIC', '-O3', '-march=native',
            '-DSHADER_PATH="%s"' % shader_dir,
            c_path, '-o', so_path, '-lvulkan', '-lm'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compile error: {result.stderr}")
            print("\nMake sure you have:")
            print("  - GCC installed: sudo apt install build-essential")
            print("  - Vulkan SDK: sudo apt install libvulkan-dev")
            raise RuntimeError("Compilation failed")
        print(f"ADAMAH: Compiled to {so_path}")
    
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
    
    return _lib

# Reduce op codes
REDUCE_SUM = 0
REDUCE_MAX = 1
REDUCE_MIN = 2

# Broadcast op codes
BROADCAST_MUL = 0
BROADCAST_DIV = 1
BROADCAST_ADD = 2
BROADCAST_SUB = 3


class Adamah:
    """ADAMAH GPU Context"""
    
    def __init__(self):
        self._lib = _get_lib()
        ret = self._lib.adamah_init()
        if ret != 0:
            raise RuntimeError(f"Failed to initialize ADAMAH: {ret}")
        self._maps = {}  # Track map info: id -> (word_size, pack_size, n_packs)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
    
    def shutdown(self):
        if self._lib:
            self._lib.adamah_shutdown()
            self._lib = None
    
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
            self._maps[map_id] = (word_size, pack_size, n_packs)
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
        
        word_size, pack_size, _ = self._maps[map_id]
        
        locs = np.ascontiguousarray(locs, dtype=np.uint32)
        
        # Determine dtype from word_size
        if word_size == 4:
            data = np.ascontiguousarray(data, dtype=np.float32)
        elif word_size == 8:
            data = np.ascontiguousarray(data, dtype=np.float64)
        else:
            data = np.ascontiguousarray(data, dtype=np.uint8)
        
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
        
        word_size, pack_size, _ = self._maps[map_id]
        
        locs = np.ascontiguousarray(locs, dtype=np.uint32)
        if n_locs is None:
            n_locs = len(locs)
        
        # Allocate output
        if word_size == 4:
            out = np.zeros(n_locs * pack_size, dtype=np.float32)
        elif word_size == 8:
            out = np.zeros(n_locs * pack_size, dtype=np.float64)
        else:
            out = np.zeros(n_locs * pack_size * word_size, dtype=np.uint8)
        
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
            self._maps[map_id] = (ws, ps, np)
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


# Convenience function
def init():
    """Initialize ADAMAH and return context."""
    return Adamah()
