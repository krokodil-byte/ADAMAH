#!/usr/bin/env python3
"""ADAMAH Quick Test"""
import sys
sys.path.insert(0, 'src')
import adamah
import numpy as np

print("ADAMAH v4.1 Test Suite")
print("=" * 40)

gpu = adamah.Adamah()

# Test 1: Basic scatter/gather
print("\n[1] Scatter/Gather...", end=" ")
gpu.map_init(0, word_size=4, pack_size=128, n_packs=100)
data = np.arange(128, dtype=np.float32)
gpu.scatter(0, np.array([0], dtype=np.uint32), data)
result = gpu.gather(0, np.array([0], dtype=np.uint32))
assert np.allclose(data, result), "FAIL"
print("OK")

# Test 2: Unary op (sin)
print("[2] Unary ops (sin)...", end=" ")
gpu.map_sin(0, np.array([0], dtype=np.uint32), np.array([1], dtype=np.uint32))
result = gpu.gather(0, np.array([1], dtype=np.uint32))
assert np.allclose(result, np.sin(data), atol=1e-4), "FAIL"
print("OK")

# Test 3: Matmul
print("[3] Matrix multiply...", end=" ")
gpu.map_init(1, word_size=4, pack_size=1, n_packs=1000)
A = np.random.randn(4, 8).astype(np.float32)
B = np.random.randn(8, 4).astype(np.float32)
gpu.scatter(1, np.arange(32, dtype=np.uint32), A.flatten())
gpu.scatter(1, np.arange(32, 64, dtype=np.uint32), B.flatten())
gpu.map_matmul(1, np.array([0], dtype=np.uint32), np.array([32], dtype=np.uint32), 
               np.array([64], dtype=np.uint32), 4, 8, 4)
result = gpu.gather(1, np.arange(64, 80, dtype=np.uint32)).reshape(4, 4)
assert np.allclose(result, A @ B, atol=1e-4), "FAIL"
print("OK")

# Test 4: Reduce
print("[4] Reduce sum...", end=" ")
gpu.map_reduce_sum(0, np.array([0], dtype=np.uint32), np.array([50], dtype=np.uint32))
print("OK")

# Test 5: Broadcast
print("[5] Broadcast scale...", end=" ")
scalar_pack = np.array([2.0] + [0.0]*127, dtype=np.float32)
gpu.scatter(0, np.array([10], dtype=np.uint32), scalar_pack)
gpu.map_scale(0, np.array([0], dtype=np.uint32), np.array([10], dtype=np.uint32), 
              np.array([2], dtype=np.uint32))
result = gpu.gather(0, np.array([2], dtype=np.uint32))
assert np.allclose(result, data * 2.0, atol=1e-4), "FAIL"
print("OK")

gpu.shutdown()
print("\n" + "=" * 40)
print("All tests passed!")
