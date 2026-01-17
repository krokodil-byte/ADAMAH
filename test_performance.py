#!/usr/bin/env python3
"""
Test CPU parallelism performance with OpenMP
"""
import sys
sys.path.insert(0, 'src')

import adamah
import numpy as np
import time

def benchmark(name, func, iterations=5):
    """Benchmark a function"""
    times = []
    for _ in range(iterations):
        start = time.time()
        func()
        times.append(time.time() - start)

    avg = np.mean(times)
    std = np.std(times)
    return avg, std

def main():
    print("=" * 60)
    print("ADAMAH CPU Parallelism Benchmark (OpenMP)")
    print("=" * 60)

    # Initialize
    adamah.init()
    adamah.set_mode('cpu')

    # Test sizes
    sizes = [1000, 10000, 100000, 1000000]

    for n in sizes:
        print(f"\n[Array size: {n:,} elements]")

        # Create test data
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)

        adamah.put("a", a)
        adamah.put("b", b)

        # Test binary operation (add)
        def test_add():
            adamah.add("c", "a", "b", n)

        avg, std = benchmark("add", test_add)
        throughput = n / avg / 1e6  # Million elements/sec
        print(f"  add:  {avg*1000:6.2f} ± {std*1000:5.2f} ms  ({throughput:.1f} M elem/s)")

        # Test unary operation (sin)
        def test_sin():
            adamah.sin("y", "a", n)

        avg, std = benchmark("sin", test_sin)
        throughput = n / avg / 1e6
        print(f"  sin:  {avg*1000:6.2f} ± {std*1000:5.2f} ms  ({throughput:.1f} M elem/s)")

        # Test scalar operation
        def test_mul_scalar():
            adamah._ctx.ops(adamah.MUL, "d", "a", 2.0, n)

        avg, std = benchmark("mul*2", test_mul_scalar)
        throughput = n / avg / 1e6
        print(f"  mul*2:{avg*1000:6.2f} ± {std*1000:5.2f} ms  ({throughput:.1f} M elem/s)")

    # Verify correctness
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    n = 1000
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    b = np.array([4, 3, 2, 1], dtype=np.float32)

    adamah.put("a", a)
    adamah.put("b", b)
    adamah.add("c", "a", "b", 4)
    adamah.sin("y", "a", 4)

    c = adamah.get("c")
    y = adamah.get("y")

    print(f"  add([1,2,3,4], [4,3,2,1]) = {c}")
    print(f"  sin([1,2,3,4])            = {y}")
    print(f"  Expected: [5, 5, 5, 5] and [0.841, 0.909, 0.141, -0.757]")

    # Check correctness
    expected_add = np.array([5, 5, 5, 5], dtype=np.float32)
    expected_sin = np.sin(a)

    if np.allclose(c, expected_add) and np.allclose(y, expected_sin, atol=0.001):
        print("\n✓ All results correct!")
    else:
        print("\n✗ Results incorrect!")
        return 1

    adamah.shutdown()

    print("\n" + "=" * 60)
    print("OpenMP parallelism: ENABLED (threshold: 1000 elements)")
    print("Arrays >= 1000 elements use multi-threading")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())
