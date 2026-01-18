#!/usr/bin/env python3
"""ADAMAH v4 Test"""
import sys
sys.path.insert(0, 'src')
import adamah
import numpy as np

print("=" * 50)
print(" ADAMAH v4.0.0 - Map-Centric GPU Compute")
print("=" * 50)

gpu = adamah.Adamah()

# Create map
gpu.map_init(0, word_size=4, pack_size=128, n_packs=1000)
print(f"\n[✓] Map created: {gpu.map_size(0)} packs x 128 floats")

# Scatter
locs = np.array([0, 1, 2], dtype=np.uint32)
data = np.arange(3 * 128, dtype=np.float32)
gpu.scatter(0, locs, data)
print("[✓] Scatter: CPU → GPU")

# GPU ops
gpu.map_sin(0, locs, locs)
print("[✓] GPU op: sin()")

# Gather
result = gpu.gather(0, locs)
expected = np.sin(data)
print("[✓] Gather: GPU → CPU")

# Verify
if np.allclose(result, expected, atol=1e-4):
    print("[✓] Results correct!")
else:
    print("[✗] Results mismatch")

# Cleanup
gpu.shutdown()
print("\n" + "=" * 50)
