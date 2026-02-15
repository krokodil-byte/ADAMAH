#!/usr/bin/env python3
"""
ADAMAH Benchmark: Simple Batches (multi-dtype)

Usage:
    python benchmark_simple_batches.py                # default: f32
    python benchmark_simple_batches.py --dtype bf16   # bfloat16 shaders
    python benchmark_simple_batches.py --dtype q8     # int8 quantized (matmul + scatter/gather only)

ADAMAH shader directories:
    f32/  -- full set: op1, op2, broadcast, reduce, softmax, layernorm, matmul, scatter, gather
    bf16/ -- full set (identical ops, bf16 storage on GPU, f32 host interface)
    q8/   -- partial: matmul, scatter, gather only
"""
import os
import sys
import time
import argparse
import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
    _CUPY_IMPORT_ERROR = ""
except Exception as exc:
    cp = None
    _CUPY_AVAILABLE = False
    _CUPY_IMPORT_ERROR = str(exc)

try:
    import torch
    _TORCH_AVAILABLE = torch.cuda.is_available()
    _TORCH_IMPORT_ERROR = "" if _TORCH_AVAILABLE else "CUDA not available"
except Exception as exc:
    torch = None
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = str(exc)


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

# Ensure CWD is the project root so adamah.so finds shaders/{f32,bf16,q8}/
os.chdir(ROOT)

import adamah  # noqa: E402

# ============================================
# Dtype registry
# ============================================
_FULL_OPS = {"op1", "op2", "broadcast", "reduce", "softmax", "layernorm", "matmul", "scatter", "gather"}
_Q8_OPS = {"matmul", "scatter", "gather"}

DTYPE_REGISTRY = {
    "f32": {
        "adamah_const": 0, "wordlength": 4, "label": "float32",
        "q8_group_size": 128, "supported_ops": _FULL_OPS,
        "torch_dtype": torch.float32 if torch else None,
    },
    "bf16": {
        "adamah_const": 1, "wordlength": 2, "label": "bfloat16",
        "q8_group_size": 128, "supported_ops": _FULL_OPS,
        "torch_dtype": torch.bfloat16 if torch else None,
    },
    "q8": {
        "adamah_const": 2, "wordlength": 1, "label": "int8 quantized",
        "q8_group_size": 128, "supported_ops": _Q8_OPS,
        "torch_dtype": torch.float32 if torch else None,
    },
}

# ============================================
# Constants
# ============================================
SEED = 1234
VEC_SIZE = 262_144
ROW_SIZE = 256
N_ROWS = VEC_SIZE // ROW_SIZE
CACHE_MB = 512

BATCH_SIZES = [10, 100, 1_000, 10_000, 100_000]
HEAVY_BATCH_SIZES = [10, 100, 1_000]
MAX_DESC_SETS = 8192
DESC_SET_HEADROOM = 64
TRANSFORMER_SEQ = 128
TRANSFORMER_DMODEL = 256
TRANSFORMER_ITERS = [10, 50, 100]

# ============================================
# Helpers
# ============================================
def now_ms():
    return time.perf_counter() * 1000.0

def bench_batch(label, fn, n_ops, use_batch=True, ops_per_iter=1,
                *, backend="adamah", section="", results=None, sync_fn=None):
    chunk_iters = max(1, (MAX_DESC_SETS - DESC_SET_HEADROOM) // max(1, ops_per_iter))
    t0 = now_ms()
    remaining = n_ops
    while remaining > 0:
        cur = min(remaining, chunk_iters)
        if use_batch and cur > 1:
            gpu.batch_begin()
        for _ in range(cur):
            fn()
        if use_batch and cur > 1:
            gpu.batch_end()
        remaining -= cur
    if sync_fn is not None:
        sync_fn()
    t1 = now_ms()
    total = t1 - t0
    per_op = total / (n_ops * max(1, ops_per_iter))
    print(f"{label:28s} {n_ops:7d}  {total:10.3f} ms  {per_op:10.6f} ms/op")
    if results is not None:
        results.append({"section": section, "label": label, "n_ops": n_ops,
                        "backend": backend, "total_ms": total, "per_op_ms": per_op})
    return total, per_op

def warmup(fn, n=3):
    for _ in range(n):
        fn()

def make_locs(base, size):
    return np.arange(size, dtype=np.uint32) + np.uint32(base)

def has_op(dcfg, op_name):
    return op_name in dcfg["supported_ops"]

# ============================================
# Typed map creation
# ============================================
def typed_carray_init(gpu, u, map_id, n_cells, dcfg):
    """Create map + register UUCIS metadata, dtype-aware."""
    adamah_dtype = dcfg["adamah_const"]
    n_cells = int(n_cells)
    if adamah_dtype == 0:
        return u.carray_init(map_id, n_cells, 4)
    gpu.map_create_typed(map_id, dtype=adamah_dtype, pack_size=1,
                         n_packs=n_cells, group_size=dcfg["q8_group_size"])
    wl = dcfg["wordlength"]
    u._loc.set_map(map_id, dim=1, n_cells=n_cells, wordlength=wl,
                   shape=(n_cells,), pack_size=1)
    return u.cache_locs(map_id, np.arange(n_cells, dtype=np.uint32))

# ============================================
# Summary
# ============================================
def summarize_results(results):
    if not results:
        return
    groups = {}
    for row in results:
        key = (row["section"], row["label"], row["n_ops"])
        groups.setdefault(key, {})[row["backend"]] = row

    print("\n" + "=" * 105)
    print("SUMMARY (ms/op)  speedup = other/adamah ( >1 means Adamah faster )")
    print("=" * 105)
    for section in sorted({r["section"] for r in results}):
        print(f"\n{section}")
        print(f"{'op':<28} {'n_ops':>7}  {'adamah':>10}  {'pytorch':>10}  {'cupy':>10}  {'vs torch':>10}  {'vs cupy':>10}")
        print("-" * 105)
        for (sec, label, n_ops), backends in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
            if sec != section:
                continue
            a = backends.get("adamah"); t = backends.get("pytorch"); c = backends.get("cupy")
            a_ms = f"{a['per_op_ms']:.6f}" if a else "n/a"
            t_ms = f"{t['per_op_ms']:.6f}" if t else "n/a"
            c_ms = f"{c['per_op_ms']:.6f}" if c else "n/a"
            speed_t_s = f"{t['per_op_ms']/a['per_op_ms']:7.2f}x" if a and t and a["per_op_ms"]>0 else "   n/a   "
            speed_c_s = f"{c['per_op_ms']/a['per_op_ms']:7.2f}x" if a and c and a["per_op_ms"]>0 else "   n/a   "
            print(f"{label:<28} {n_ops:>7}  {a_ms:>10}  {t_ms:>10}  {c_ms:>10}  {speed_t_s:>10}  {speed_c_s:>10}")

# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser(description="ADAMAH simple-batches benchmark (multi-dtype)")
    parser.add_argument("--dtype", choices=["f32", "bf16", "q8"], default="f32",
                        help="Numeric format: f32 (default), bf16, or q8")
    args = parser.parse_args()

    dcfg = DTYPE_REGISTRY[args.dtype]
    dtype_label = dcfg["label"]
    torch_dtype = dcfg["torch_dtype"]

    print("=" * 80)
    print(f"ADAMAH Simple Batches Benchmark  --  dtype: {args.dtype} ({dtype_label})")
    print("=" * 80)

    rng = np.random.default_rng(SEED)
    results = []
    global gpu
    gpu = adamah.init(CACHE_MB)
    u = gpu.uucis
    u.set_cached_batching(True)

    # Set ADAMAH dtype (loads corresponding shader set)
    if dcfg["adamah_const"] != 0:
        gpu.set_dtype(dcfg["adamah_const"])
        print(f"[ADAMAH] set_dtype -> {args.dtype} (shader set loaded)")
    else:
        print("[ADAMAH] using default f32 shaders")

    # ----------------------------------------------------------------
    # Map 0: elementwise / binary / broadcast
    # Layout: A | B | OUT | SCALAR | TMP
    # ----------------------------------------------------------------
    offset_a = 0; offset_b = VEC_SIZE; offset_out = VEC_SIZE*2
    offset_scalar = VEC_SIZE*3; offset_tmp = VEC_SIZE*4
    map0_cells = offset_tmp + VEC_SIZE

    _map0_full = typed_carray_init(gpu, u, 0, map0_cells, dcfg)

    data_a = rng.standard_normal(VEC_SIZE, dtype=np.float32)
    data_b = rng.standard_normal(VEC_SIZE, dtype=np.float32)
    data_tmp = np.zeros(VEC_SIZE, dtype=np.float32)

    locs_a = make_locs(offset_a, VEC_SIZE); locs_b = make_locs(offset_b, VEC_SIZE)
    locs_out = make_locs(offset_out, VEC_SIZE); locs_tmp = make_locs(offset_tmp, VEC_SIZE)
    locs_scalar = np.full(VEC_SIZE, np.uint32(offset_scalar), dtype=np.uint32)
    locs_scalar_single = np.array([offset_scalar], dtype=np.uint32)

    locs_a_c = u.cache_locs(0, locs_a); locs_b_c = u.cache_locs(0, locs_b)
    locs_out_c = u.cache_locs(0, locs_out); locs_tmp_c = u.cache_locs(0, locs_tmp)
    locs_scalar_c = u.cache_locs(0, locs_scalar)
    locs_scalar_single_c = u.cache_locs(0, locs_scalar_single)

    u.scatter(0, locs_a_c, data_a); u.scatter(0, locs_b_c, data_b)
    u.scatter(0, locs_out_c, data_tmp); u.scatter(0, locs_tmp_c, data_tmp)
    u.scatter(0, locs_scalar_single_c, np.array([1.2345], dtype=np.float32))

    perm = rng.permutation(VEC_SIZE).astype(np.uint32)
    locs_a_perm = perm + np.uint32(offset_a)
    locs_out_perm = perm + np.uint32(offset_out)
    locs_a_perm_c = u.cache_locs(0, locs_a_perm)
    locs_out_perm_c = u.cache_locs(0, locs_out_perm)

    # ----------------------------------------------------------------
    # Map 1: softmax — Layout: SRC | DST
    # ----------------------------------------------------------------
    _map1_full = typed_carray_init(gpu, u, 1, VEC_SIZE*2, dcfg)
    soft_src = rng.standard_normal(VEC_SIZE, dtype=np.float32)
    soft_dst = np.zeros(VEC_SIZE, dtype=np.float32)
    soft_locs_full_src_c = u.cache_locs(1, make_locs(0, VEC_SIZE))
    soft_locs_full_dst_c = u.cache_locs(1, make_locs(VEC_SIZE, VEC_SIZE))
    u.scatter(1, soft_locs_full_src_c, soft_src)
    u.scatter(1, soft_locs_full_dst_c, soft_dst)
    soft_locs_src_c = u.cache_locs(1, (np.arange(N_ROWS, dtype=np.uint32)*ROW_SIZE).astype(np.uint32))
    soft_locs_dst_c = u.cache_locs(1, (np.arange(N_ROWS, dtype=np.uint32)*ROW_SIZE+VEC_SIZE).astype(np.uint32))

    # ----------------------------------------------------------------
    # Map 2: layernorm — Layout: SRC | DST | GAMMA | BETA
    # ----------------------------------------------------------------
    _map2_full = typed_carray_init(gpu, u, 2, VEC_SIZE*4, dcfg)
    ln_src = rng.standard_normal(VEC_SIZE, dtype=np.float32)
    ln_gamma = rng.standard_normal(VEC_SIZE, dtype=np.float32)
    ln_beta = rng.standard_normal(VEC_SIZE, dtype=np.float32)
    u.scatter(2, u.cache_locs(2, make_locs(0, VEC_SIZE)), ln_src)
    u.scatter(2, u.cache_locs(2, make_locs(VEC_SIZE*2, VEC_SIZE)), ln_gamma)
    u.scatter(2, u.cache_locs(2, make_locs(VEC_SIZE*3, VEC_SIZE)), ln_beta)
    ln_locs_src_c = u.cache_locs(2, (np.arange(N_ROWS, dtype=np.uint32)*ROW_SIZE).astype(np.uint32))
    ln_locs_dst_c = u.cache_locs(2, (np.arange(N_ROWS, dtype=np.uint32)*ROW_SIZE+VEC_SIZE).astype(np.uint32))
    ln_locs_gamma_c = u.cache_locs(2, (np.arange(N_ROWS, dtype=np.uint32)*ROW_SIZE+VEC_SIZE*2).astype(np.uint32))
    ln_locs_beta_c = u.cache_locs(2, (np.arange(N_ROWS, dtype=np.uint32)*ROW_SIZE+VEC_SIZE*3).astype(np.uint32))

    # ----------------------------------------------------------------
    # Map 3: matmul — Layout: A | B | C
    # ----------------------------------------------------------------
    M = 64; K = 64; N = 64
    mm_size = M*K; mm_cells = mm_size + K*N + M*N
    _map3_full = typed_carray_init(gpu, u, 3, mm_cells, dcfg)
    mm_a = rng.standard_normal(mm_size, dtype=np.float32)
    mm_b = rng.standard_normal(K*N, dtype=np.float32)
    u.scatter(3, u.cache_locs(3, make_locs(0, mm_size)), mm_a)
    u.scatter(3, u.cache_locs(3, make_locs(mm_size, K*N)), mm_b)
    mm_a_base = 0; mm_b_base = mm_size; mm_c_base = mm_size + K*N

    # ----------------------------------------------------------------
    # Map 4: toy transformer block
    # ----------------------------------------------------------------
    t_seq = TRANSFORMER_SEQ; t_d = TRANSFORMER_DMODEL
    t_x = t_seq*t_d; t_scores = t_seq*t_seq
    t_wqk = t_d*t_seq; t_wv = t_d*t_d; t_wo = t_d*t_d

    off_x=0; off_wqk=off_x+t_x; off_wv=off_wqk+t_wqk; off_wo=off_wv+t_wv
    off_scores=off_wo+t_wo; off_scores_soft=off_scores+t_scores
    off_v=off_scores_soft+t_scores; off_attn=off_v+t_x; off_out=off_attn+t_x
    off_resid=off_out+t_x; off_ln=off_resid+t_x; off_gamma=off_ln+t_x
    off_beta=off_gamma+t_d; off_scale=off_beta+t_d; t_total=off_scale+1

    _map4_full = typed_carray_init(gpu, u, 4, t_total, dcfg)

    t_x_data = rng.standard_normal(t_x, dtype=np.float32)
    t_wqk_data = rng.standard_normal(t_wqk, dtype=np.float32)
    t_wv_data = rng.standard_normal(t_wv, dtype=np.float32)
    t_wo_data = rng.standard_normal(t_wo, dtype=np.float32)
    t_gamma = rng.standard_normal(t_d, dtype=np.float32)
    t_beta = rng.standard_normal(t_d, dtype=np.float32)
    t_scale = np.array([1.0/np.sqrt(float(t_d))], dtype=np.float32)

    locs_x_all_c = u.cache_locs(4, make_locs(off_x, t_x))
    locs_scores_all_c = u.cache_locs(4, make_locs(off_scores, t_scores))
    locs_scores_soft_all_c = u.cache_locs(4, make_locs(off_scores_soft, t_scores))
    locs_out_all_c = u.cache_locs(4, make_locs(off_out, t_x))
    locs_resid_all_c = u.cache_locs(4, make_locs(off_resid, t_x))
    locs_scale_all_c = u.cache_locs(4, np.full(t_scores, np.uint32(off_scale), dtype=np.uint32))

    u.scatter(4, locs_x_all_c, t_x_data)
    u.scatter(4, u.cache_locs(4, make_locs(off_wqk, t_wqk)), t_wqk_data)
    u.scatter(4, u.cache_locs(4, make_locs(off_wv, t_wv)), t_wv_data)
    u.scatter(4, u.cache_locs(4, make_locs(off_wo, t_wo)), t_wo_data)
    u.scatter(4, u.cache_locs(4, make_locs(off_gamma, t_d)), t_gamma)
    u.scatter(4, u.cache_locs(4, make_locs(off_beta, t_d)), t_beta)
    u.scatter(4, u.cache_locs(4, [off_scale]), t_scale)

    locs_x_base_c = u.cache_locs(4, np.array([off_x], dtype=np.uint32))
    locs_wqk_base_c = u.cache_locs(4, np.array([off_wqk], dtype=np.uint32))
    locs_wv_base_c = u.cache_locs(4, np.array([off_wv], dtype=np.uint32))
    locs_wo_base_c = u.cache_locs(4, np.array([off_wo], dtype=np.uint32))
    locs_scores_base_c = u.cache_locs(4, np.array([off_scores], dtype=np.uint32))
    locs_scores_soft_base_c = u.cache_locs(4, np.array([off_scores_soft], dtype=np.uint32))
    locs_v_base_c = u.cache_locs(4, np.array([off_v], dtype=np.uint32))
    locs_attn_base_c = u.cache_locs(4, np.array([off_attn], dtype=np.uint32))
    locs_out_base_c = u.cache_locs(4, np.array([off_out], dtype=np.uint32))

    locs_scores_rows_c = u.cache_locs(4, (np.arange(t_seq, dtype=np.uint32)*t_seq+np.uint32(off_scores)).astype(np.uint32))
    locs_scores_soft_rows_c = u.cache_locs(4, (np.arange(t_seq, dtype=np.uint32)*t_seq+np.uint32(off_scores_soft)).astype(np.uint32))
    locs_resid_rows_c = u.cache_locs(4, (np.arange(t_seq, dtype=np.uint32)*t_d+np.uint32(off_resid)).astype(np.uint32))
    locs_ln_rows_c = u.cache_locs(4, (np.arange(t_seq, dtype=np.uint32)*t_d+np.uint32(off_ln)).astype(np.uint32))
    locs_gamma_rows_c = u.cache_locs(4, np.full(t_seq, np.uint32(off_gamma), dtype=np.uint32))
    locs_beta_rows_c = u.cache_locs(4, np.full(t_seq, np.uint32(off_beta), dtype=np.uint32))

    # ----------------------------------------------------------------
    # Warmups (only for supported ops)
    # ----------------------------------------------------------------
    if has_op(dcfg, "op1"):
        warmup(lambda: u.mop1("EXP", 0, 0, locs_src=locs_a_c, locs_dst=locs_out_c))
    if has_op(dcfg, "op2"):
        warmup(lambda: u.mop2("ADD", 0, 0, 0, locs_a=locs_a_c, locs_b=locs_b_c, locs_dst=locs_out_c))
    if has_op(dcfg, "broadcast"):
        warmup(lambda: u.mop2("BROADCAST:ADD", 0, 0, 0,
                              locs_a=locs_a_c, locs_b=locs_scalar_c, locs_dst=locs_out_c))
    if has_op(dcfg, "reduce"):
        warmup(lambda: u.mop1("REDUCE:SUM", 0, 0, locs_src=locs_a_c, locs_dst=locs_out_c))
    if has_op(dcfg, "softmax"):
        warmup(lambda: u.mop1("SOFTMAX", 1, 1, locs_src=soft_locs_src_c, locs_dst=soft_locs_dst_c,
                              extra={"row_size": ROW_SIZE}))
    if has_op(dcfg, "layernorm"):
        warmup(lambda: u.mop1("LAYERNORM", 2, 2, locs_src=ln_locs_src_c, locs_dst=ln_locs_dst_c,
                              extra={"dim": ROW_SIZE, "eps": 1e-5,
                                     "locs_gamma": ln_locs_gamma_c, "locs_beta": ln_locs_beta_c}))
    if has_op(dcfg, "matmul"):
        mm1a = u.cache_locs(3, np.array([mm_a_base], dtype=np.uint32))
        mm1b = u.cache_locs(3, np.array([mm_b_base], dtype=np.uint32))
        mm1c = u.cache_locs(3, np.array([mm_c_base], dtype=np.uint32))
        warmup(lambda: u.mop2("MATMUL", 3, 3, 3, extra={
            "locs_a": mm1a, "locs_b": mm1b, "locs_c": mm1c, "M": M, "K": K, "N": N}))

    # ================================================================
    # ADAMAH benchmarks
    # ================================================================
    if has_op(dcfg, "op1"):
        print(f"\nElementwise / Binary / Broadcast [Adamah -- {args.dtype}]")
        print("op                           n_ops     total_ms     ms/op")
        for n_ops in BATCH_SIZES:
            bench_batch("Unary EXP (contig)", lambda: u.mop1("EXP", 0, 0, locs_src=locs_a_c, locs_dst=locs_out_c),
                        n_ops, backend="adamah", section="Elementwise / Binary / Broadcast", results=results)
            if n_ops <= 1_000:
                bench_batch("Unary EXP (perm)", lambda: u.mop1("EXP", 0, 0, locs_src=locs_a_perm_c, locs_dst=locs_out_perm_c),
                            n_ops, backend="adamah", section="Elementwise / Binary / Broadcast", results=results)
            bench_batch("Binary ADD (contig)", lambda: u.mop2("ADD", 0, 0, 0, locs_a=locs_a_c, locs_b=locs_b_c, locs_dst=locs_out_c),
                        n_ops, backend="adamah", section="Elementwise / Binary / Broadcast", results=results)
            bench_batch("Broadcast ADD (contig)", lambda: u.mop2("BROADCAST:ADD", 0, 0, 0, locs_a=locs_a_c, locs_b=locs_scalar_c, locs_dst=locs_out_c),
                        n_ops, backend="adamah", section="Elementwise / Binary / Broadcast", results=results)
            bench_batch("Reduce SUM (contig)", lambda: u.mop1("REDUCE:SUM", 0, 0, locs_src=locs_a_c, locs_dst=locs_out_c),
                        n_ops, backend="adamah", section="Elementwise / Binary / Broadcast", results=results)
            print("-" * 72)
    else:
        print(f"\n[SKIP] Elementwise / Binary / Broadcast -- not supported for {args.dtype}")

    if has_op(dcfg, "op1") and has_op(dcfg, "op2"):
        print(f"\nChained / Mixed batches [Adamah -- {args.dtype}]")
        print("op                           n_ops     total_ms     ms/op")
        for n_ops in BATCH_SIZES:
            def chain_same():
                u.mop1("EXP", 0, 0, locs_src=locs_a_c, locs_dst=locs_out_c)
                u.mop1("EXP", 0, 0, locs_src=locs_out_c, locs_dst=locs_tmp_c)
                u.mop1("EXP", 0, 0, locs_src=locs_tmp_c, locs_dst=locs_out_c)
            bench_batch("Chain 3x EXP", chain_same, n_ops, ops_per_iter=3,
                        backend="adamah", section="Chained / Mixed batches", results=results)
            def mixed():
                u.mop1("EXP", 0, 0, locs_src=locs_a_c, locs_dst=locs_out_c)
                u.mop2("ADD", 0, 0, 0, locs_a=locs_out_c, locs_b=locs_b_c, locs_dst=locs_tmp_c)
                u.mop2("BROADCAST:ADD", 0, 0, 0, locs_a=locs_tmp_c, locs_b=locs_scalar_c, locs_dst=locs_out_c)
            bench_batch("Mixed 3-op", mixed, n_ops, ops_per_iter=3,
                        backend="adamah", section="Chained / Mixed batches", results=results)
            print("-" * 72)
    else:
        print(f"\n[SKIP] Chained / Mixed batches -- not supported for {args.dtype}")

    _has_any_heavy = has_op(dcfg, "softmax") or has_op(dcfg, "layernorm") or has_op(dcfg, "matmul")
    if _has_any_heavy:
        print(f"\nSoftmax / LayerNorm / MatMul [Adamah -- {args.dtype}]")
        print("op                           n_ops     total_ms     ms/op")
        for n_ops in HEAVY_BATCH_SIZES:
            if has_op(dcfg, "softmax"):
                bench_batch("Softmax", lambda: u.mop1("SOFTMAX", 1, 1, locs_src=soft_locs_src_c, locs_dst=soft_locs_dst_c,
                            extra={"row_size": ROW_SIZE}), n_ops, backend="adamah",
                            section="Softmax / LayerNorm / MatMul", results=results)
            if has_op(dcfg, "layernorm"):
                bench_batch("LayerNorm", lambda: u.mop1("LAYERNORM", 2, 2, locs_src=ln_locs_src_c, locs_dst=ln_locs_dst_c,
                            extra={"dim": ROW_SIZE, "eps": 1e-5,
                                   "locs_gamma": ln_locs_gamma_c, "locs_beta": ln_locs_beta_c}),
                            n_ops, backend="adamah", section="Softmax / LayerNorm / MatMul", results=results)
            if has_op(dcfg, "matmul"):
                la = u.cache_locs(3, np.full(n_ops, np.uint32(mm_a_base), dtype=np.uint32))
                lb = u.cache_locs(3, np.full(n_ops, np.uint32(mm_b_base), dtype=np.uint32))
                lc = u.cache_locs(3, np.full(n_ops, np.uint32(mm_c_base), dtype=np.uint32))
                t0 = now_ms()
                u.mop2("MATMUL", 3, 3, 3, extra={"locs_a": la, "locs_b": lb, "locs_c": lc, "M": M, "K": K, "N": N})
                t1 = now_ms()
                total = t1 - t0; per_op = total / n_ops
                print(f"MatMul                    {n_ops:7d}  {total:10.3f} ms  {per_op:10.6f} ms/op")
                results.append({"section": "Softmax / LayerNorm / MatMul", "label": "MatMul",
                                "n_ops": n_ops, "backend": "adamah", "total_ms": total, "per_op_ms": per_op})
            print("-" * 72)

    if has_op(dcfg, "scatter"):
        print(f"\nScatter / Gather [Adamah -- {args.dtype}]")
        print("op                           n_ops     total_ms     ms/op")
        data_dev = u.to_cached(rng.standard_normal(VEC_SIZE, dtype=np.float32))
        for n_ops in HEAVY_BATCH_SIZES:
            t0 = now_ms()
            for _ in range(n_ops): u.scatter(0, locs_a_c, data_dev)
            gpu.synchronize_all(); t1 = now_ms()
            total = t1-t0; per_op = total/n_ops
            print(f"Scatter (dev)             {n_ops:7d}  {total:10.3f} ms  {per_op:10.6f} ms/op")
            results.append({"section": "Scatter / Gather", "label": "Scatter (dev)",
                            "n_ops": n_ops, "backend": "adamah", "total_ms": total, "per_op_ms": per_op})
            t0 = now_ms()
            for _ in range(n_ops): _ = u.gather(0, locs_a_c, target=u.cached(0, 0, np.float32))
            gpu.synchronize_all(); t1 = now_ms()
            total = t1-t0; per_op = total/n_ops
            print(f"Gather (dev)              {n_ops:7d}  {total:10.3f} ms  {per_op:10.6f} ms/op")
            results.append({"section": "Scatter / Gather", "label": "Gather (dev)",
                            "n_ops": n_ops, "backend": "adamah", "total_ms": total, "per_op_ms": per_op})
            print("-" * 72)

    _transformer_ops = {"op1", "op2", "broadcast", "softmax", "layernorm", "matmul"}
    if _transformer_ops.issubset(dcfg["supported_ops"]):
        print(f"\nTransformer block [Adamah -- {args.dtype}]")
        print("op                           n_ops     total_ms     ms/op")
        def transformer_block_adamah():
            u.mop2("MATMUL", 4, 4, 4, extra={"locs_a": locs_x_base_c, "locs_b": locs_wqk_base_c,
                    "locs_c": locs_scores_base_c, "M": t_seq, "K": t_d, "N": t_seq})
            u.mop2("BROADCAST:MUL", 4, 4, 4, locs_a=locs_scores_all_c, locs_b=locs_scale_all_c, locs_dst=locs_scores_all_c)
            u.mop1("SOFTMAX", 4, 4, locs_src=locs_scores_rows_c, locs_dst=locs_scores_soft_rows_c, extra={"row_size": t_seq})
            u.mop2("MATMUL", 4, 4, 4, extra={"locs_a": locs_x_base_c, "locs_b": locs_wv_base_c,
                    "locs_c": locs_v_base_c, "M": t_seq, "K": t_d, "N": t_d})
            u.mop2("MATMUL", 4, 4, 4, extra={"locs_a": locs_scores_soft_base_c, "locs_b": locs_v_base_c,
                    "locs_c": locs_attn_base_c, "M": t_seq, "K": t_seq, "N": t_d})
            u.mop2("MATMUL", 4, 4, 4, extra={"locs_a": locs_attn_base_c, "locs_b": locs_wo_base_c,
                    "locs_c": locs_out_base_c, "M": t_seq, "K": t_d, "N": t_d})
            u.mop2("ADD", 4, 4, 4, locs_a=locs_out_all_c, locs_b=locs_x_all_c, locs_dst=locs_resid_all_c)
            u.mop1("LAYERNORM", 4, 4, locs_src=locs_resid_rows_c, locs_dst=locs_ln_rows_c,
                   extra={"dim": t_d, "eps": 1e-5, "locs_gamma": locs_gamma_rows_c, "locs_beta": locs_beta_rows_c})

        for n_ops in TRANSFORMER_ITERS:
            bench_batch("Transformer block", transformer_block_adamah, n_ops,
                        backend="adamah", section="Transformer block", results=results)
            print("-" * 72)
    else:
        print(f"\n[SKIP] Transformer block -- requires full op set, not available for {args.dtype}")

    # ================================================================
    # CuPy benchmarks (always f32 baseline)
    # ================================================================
    if not _CUPY_AVAILABLE:
        print(f"\nCuPy not available: {_CUPY_IMPORT_ERROR}")
    else:
        cp.cuda.get_current_stream().synchronize()
        cp_a = cp.asarray(data_a); cp_b = cp.asarray(data_b)
        cp_out = cp.asarray(data_tmp); cp_tmp = cp.asarray(data_tmp)
        cp_scalar = cp.asarray(np.array([1.2345], dtype=np.float32))
        cp_perm = cp.asarray(perm)
        cp_a_perm = cp.take(cp_a, cp_perm); cp_out_perm = cp.empty_like(cp_a_perm)
        cp_soft_src = cp.asarray(soft_src).reshape(N_ROWS, ROW_SIZE)
        cp_soft_dst = cp.asarray(soft_dst).reshape(N_ROWS, ROW_SIZE)
        cp_ln_src = cp.asarray(ln_src).reshape(N_ROWS, ROW_SIZE)
        cp_ln_gamma = cp.asarray(ln_gamma).reshape(N_ROWS, ROW_SIZE)
        cp_ln_beta = cp.asarray(ln_beta).reshape(N_ROWS, ROW_SIZE)
        cp_ln_dst = cp.empty_like(cp_ln_src)
        cp_mm_a = cp.asarray(mm_a).reshape(M, K); cp_mm_b = cp.asarray(mm_b).reshape(K, N)
        cp_mm_c = cp.empty((M, N), dtype=cp_mm_a.dtype)
        cp_indices = cp.asarray(make_locs(offset_a, VEC_SIZE))
        cp_scatter_data = cp.asarray(rng.standard_normal(VEC_SIZE, dtype=np.float32))
        cp_t_x = cp.asarray(t_x_data).reshape(t_seq, t_d)
        cp_t_wqk = cp.asarray(t_wqk_data).reshape(t_d, t_seq)
        cp_t_wv = cp.asarray(t_wv_data).reshape(t_d, t_d)
        cp_t_wo = cp.asarray(t_wo_data).reshape(t_d, t_d)
        cp_t_scores = cp.empty((t_seq, t_seq), dtype=cp_t_x.dtype)
        cp_t_scores_soft = cp.empty_like(cp_t_scores)
        cp_t_v = cp.empty_like(cp_t_x); cp_t_attn = cp.empty_like(cp_t_x)
        cp_t_out = cp.empty_like(cp_t_x); cp_t_resid = cp.empty_like(cp_t_x)
        cp_t_ln = cp.empty_like(cp_t_x)
        cp_t_gamma = cp.asarray(t_gamma).reshape(1, t_d); cp_t_beta = cp.asarray(t_beta).reshape(1, t_d)
        cp_scale = cp.asarray(t_scale)
        def cupy_sync(): cp.cuda.get_current_stream().synchronize()

        if has_op(dcfg, "op1"):
            print(f"\nElementwise / Binary / Broadcast [CuPy -- f32 baseline]")
            print("op                           n_ops     total_ms     ms/op")
            for n_ops in BATCH_SIZES:
                bench_batch("Unary EXP (contig)", lambda: cp.exp(cp_a, out=cp_out), n_ops, use_batch=False,
                            backend="cupy", section="Elementwise / Binary / Broadcast", results=results, sync_fn=cupy_sync)
                if n_ops <= 1_000:
                    bench_batch("Unary EXP (perm)", lambda: cp.exp(cp_a_perm, out=cp_out_perm), n_ops, use_batch=False,
                                backend="cupy", section="Elementwise / Binary / Broadcast", results=results, sync_fn=cupy_sync)
                bench_batch("Binary ADD (contig)", lambda: cp.add(cp_a, cp_b, out=cp_out), n_ops, use_batch=False,
                            backend="cupy", section="Elementwise / Binary / Broadcast", results=results, sync_fn=cupy_sync)
                bench_batch("Broadcast ADD (contig)", lambda: cp.add(cp_a, cp_scalar, out=cp_out), n_ops, use_batch=False,
                            backend="cupy", section="Elementwise / Binary / Broadcast", results=results, sync_fn=cupy_sync)
                bench_batch("Reduce SUM (contig)", lambda: cp.copyto(cp_out, cp_a), n_ops, use_batch=False,
                            backend="cupy", section="Elementwise / Binary / Broadcast", results=results, sync_fn=cupy_sync)
                print("-" * 72)
        if has_op(dcfg, "op1") and has_op(dcfg, "op2"):
            print(f"\nChained / Mixed batches [CuPy -- f32 baseline]")
            print("op                           n_ops     total_ms     ms/op")
            for n_ops in BATCH_SIZES:
                def chain_same_c():
                    cp.exp(cp_a, out=cp_out); cp.exp(cp_out, out=cp_tmp); cp.exp(cp_tmp, out=cp_out)
                bench_batch("Chain 3x EXP", chain_same_c, n_ops, use_batch=False, ops_per_iter=3,
                            backend="cupy", section="Chained / Mixed batches", results=results, sync_fn=cupy_sync)
                def mixed_c():
                    cp.exp(cp_a, out=cp_out); cp.add(cp_out, cp_b, out=cp_tmp); cp.add(cp_tmp, cp_scalar, out=cp_out)
                bench_batch("Mixed 3-op", mixed_c, n_ops, use_batch=False, ops_per_iter=3,
                            backend="cupy", section="Chained / Mixed batches", results=results, sync_fn=cupy_sync)
                print("-" * 72)
        if _has_any_heavy:
            print(f"\nSoftmax / LayerNorm / MatMul [CuPy -- f32 baseline]")
            print("op                           n_ops     total_ms     ms/op")
            for n_ops in HEAVY_BATCH_SIZES:
                if has_op(dcfg, "softmax"):
                    def softmax_c():
                        maxv = cp.max(cp_soft_src, axis=1, keepdims=True)
                        expv = cp.exp(cp_soft_src - maxv)
                        cp_soft_dst[:] = expv / cp.sum(expv, axis=1, keepdims=True)
                    bench_batch("Softmax", softmax_c, n_ops, use_batch=False,
                                backend="cupy", section="Softmax / LayerNorm / MatMul", results=results, sync_fn=cupy_sync)
                if has_op(dcfg, "layernorm"):
                    def layernorm_c():
                        mean = cp.mean(cp_ln_src, axis=1, keepdims=True); var = cp.var(cp_ln_src, axis=1, keepdims=True)
                        cp_ln_dst[:] = (cp_ln_src - mean) / cp.sqrt(var + 1e-5)
                        cp_ln_dst[:] = cp_ln_dst * cp_ln_gamma + cp_ln_beta
                    bench_batch("LayerNorm", layernorm_c, n_ops, use_batch=False,
                                backend="cupy", section="Softmax / LayerNorm / MatMul", results=results, sync_fn=cupy_sync)
                if has_op(dcfg, "matmul"):
                    t0 = now_ms()
                    for _ in range(n_ops): cp.matmul(cp_mm_a, cp_mm_b, out=cp_mm_c)
                    cupy_sync(); t1 = now_ms(); total = t1-t0; per_op = total/n_ops
                    print(f"MatMul                    {n_ops:7d}  {total:10.3f} ms  {per_op:10.6f} ms/op")
                    results.append({"section": "Softmax / LayerNorm / MatMul", "label": "MatMul",
                                    "n_ops": n_ops, "backend": "cupy", "total_ms": total, "per_op_ms": per_op})
                print("-" * 72)
        if has_op(dcfg, "scatter"):
            print(f"\nScatter / Gather [CuPy -- f32 baseline]")
            print("op                           n_ops     total_ms     ms/op")
            for n_ops in HEAVY_BATCH_SIZES:
                t0 = now_ms()
                for _ in range(n_ops): cp_out[cp_indices] = cp_scatter_data
                cupy_sync(); t1 = now_ms(); total=t1-t0; per_op=total/n_ops
                print(f"Scatter (dev)             {n_ops:7d}  {total:10.3f} ms  {per_op:10.6f} ms/op")
                results.append({"section": "Scatter / Gather", "label": "Scatter (dev)", "n_ops": n_ops, "backend": "cupy", "total_ms": total, "per_op_ms": per_op})
                t0 = now_ms()
                for _ in range(n_ops): _ = cp_out[cp_indices]
                cupy_sync(); t1 = now_ms(); total=t1-t0; per_op=total/n_ops
                print(f"Gather (dev)              {n_ops:7d}  {total:10.3f} ms  {per_op:10.6f} ms/op")
                results.append({"section": "Scatter / Gather", "label": "Gather (dev)", "n_ops": n_ops, "backend": "cupy", "total_ms": total, "per_op_ms": per_op})
                print("-" * 72)
        if _transformer_ops.issubset(dcfg["supported_ops"]):
            print(f"\nTransformer block [CuPy -- f32 baseline]")
            print("op                           n_ops     total_ms     ms/op")
            def transformer_block_cupy():
                cp.matmul(cp_t_x, cp_t_wqk, out=cp_t_scores); cp.multiply(cp_t_scores, cp_scale, out=cp_t_scores)
                maxv = cp.max(cp_t_scores, axis=1, keepdims=True)
                cp.subtract(cp_t_scores, maxv, out=cp_t_scores_soft); cp.exp(cp_t_scores_soft, out=cp_t_scores_soft)
                denom = cp.sum(cp_t_scores_soft, axis=1, keepdims=True)
                cp.divide(cp_t_scores_soft, denom, out=cp_t_scores_soft)
                cp.matmul(cp_t_x, cp_t_wv, out=cp_t_v); cp.matmul(cp_t_scores_soft, cp_t_v, out=cp_t_attn)
                cp.matmul(cp_t_attn, cp_t_wo, out=cp_t_out); cp.add(cp_t_out, cp_t_x, out=cp_t_resid)
                mean = cp.mean(cp_t_resid, axis=1, keepdims=True); var = cp.var(cp_t_resid, axis=1, keepdims=True)
                cp_t_ln[:] = (cp_t_resid - mean) / cp.sqrt(var + 1e-5)
                cp_t_ln[:] = cp_t_ln * cp_t_gamma + cp_t_beta
            for n_ops in TRANSFORMER_ITERS:
                bench_batch("Transformer block", transformer_block_cupy, n_ops, use_batch=False,
                            backend="cupy", section="Transformer block", results=results, sync_fn=cupy_sync)
                print("-" * 72)

    # ================================================================
    # PyTorch benchmarks
    # ================================================================
    if not _TORCH_AVAILABLE:
        print(f"\nPyTorch CUDA not available: {_TORCH_IMPORT_ERROR}")
    else:
        print("\n" + "=" * 72)
        print(f"PyTorch Benchmarks -- dtype: {torch_dtype}")
        print("=" * 72)
        device = torch.device("cuda"); torch.cuda.synchronize()

        pt_a = torch.from_numpy(data_a).to(device=device, dtype=torch_dtype)
        pt_b = torch.from_numpy(data_b).to(device=device, dtype=torch_dtype)
        pt_out = torch.zeros_like(pt_a); pt_tmp = torch.zeros_like(pt_a)
        pt_scalar = torch.tensor([1.2345], dtype=torch_dtype, device=device)
        pt_perm = torch.from_numpy(perm.astype(np.int64)).to(device)
        pt_a_perm = pt_a[pt_perm]; pt_out_perm = torch.empty_like(pt_a_perm)
        pt_soft_src = torch.from_numpy(soft_src).to(device=device, dtype=torch_dtype).reshape(N_ROWS, ROW_SIZE)
        pt_soft_dst = torch.zeros_like(pt_soft_src)
        pt_ln_src = torch.from_numpy(ln_src).to(device=device, dtype=torch_dtype).reshape(N_ROWS, ROW_SIZE)
        pt_ln_gamma = torch.from_numpy(ln_gamma).to(device=device, dtype=torch_dtype).reshape(N_ROWS, ROW_SIZE)
        pt_ln_beta = torch.from_numpy(ln_beta).to(device=device, dtype=torch_dtype).reshape(N_ROWS, ROW_SIZE)
        pt_ln_dst = torch.empty_like(pt_ln_src)
        pt_mm_a = torch.from_numpy(mm_a).to(device=device, dtype=torch_dtype).reshape(M, K)
        pt_mm_b = torch.from_numpy(mm_b).to(device=device, dtype=torch_dtype).reshape(K, N)
        pt_mm_c = torch.empty((M, N), dtype=torch_dtype, device=device)
        pt_indices = torch.from_numpy(make_locs(offset_a, VEC_SIZE).astype(np.int64)).to(device)
        pt_scatter_data = torch.from_numpy(rng.standard_normal(VEC_SIZE).astype(np.float32)).to(device=device, dtype=torch_dtype)
        pt_t_x = torch.from_numpy(t_x_data).to(device=device, dtype=torch_dtype).reshape(t_seq, t_d)
        pt_t_wqk = torch.from_numpy(t_wqk_data).to(device=device, dtype=torch_dtype).reshape(t_d, t_seq)
        pt_t_wv = torch.from_numpy(t_wv_data).to(device=device, dtype=torch_dtype).reshape(t_d, t_d)
        pt_t_wo = torch.from_numpy(t_wo_data).to(device=device, dtype=torch_dtype).reshape(t_d, t_d)
        pt_t_scores = torch.empty((t_seq, t_seq), dtype=torch_dtype, device=device)
        pt_t_scores_soft = torch.empty_like(pt_t_scores)
        pt_t_v = torch.empty_like(pt_t_x); pt_t_attn = torch.empty_like(pt_t_x)
        pt_t_out = torch.empty_like(pt_t_x); pt_t_resid = torch.empty_like(pt_t_x)
        pt_t_ln = torch.empty_like(pt_t_x)
        pt_t_gamma = torch.from_numpy(t_gamma).to(device=device, dtype=torch_dtype).reshape(1, t_d)
        pt_t_beta = torch.from_numpy(t_beta).to(device=device, dtype=torch_dtype).reshape(1, t_d)
        pt_scale = torch.tensor(t_scale, device=device, dtype=torch_dtype)
        def torch_sync(): torch.cuda.synchronize()

        if has_op(dcfg, "op1"):
            print(f"\nElementwise / Binary / Broadcast [PyTorch -- {torch_dtype}]")
            print("op                           n_ops     total_ms     ms/op")
            for n_ops in BATCH_SIZES:
                bench_batch("Unary EXP (contig)", lambda: torch.exp(pt_a, out=pt_out), n_ops, use_batch=False,
                            backend="pytorch", section="Elementwise / Binary / Broadcast", results=results, sync_fn=torch_sync)
                if n_ops <= 1_000:
                    bench_batch("Unary EXP (perm)", lambda: torch.exp(pt_a_perm, out=pt_out_perm), n_ops, use_batch=False,
                                backend="pytorch", section="Elementwise / Binary / Broadcast", results=results, sync_fn=torch_sync)
                bench_batch("Binary ADD (contig)", lambda: torch.add(pt_a, pt_b, out=pt_out), n_ops, use_batch=False,
                            backend="pytorch", section="Elementwise / Binary / Broadcast", results=results, sync_fn=torch_sync)
                bench_batch("Broadcast ADD (contig)", lambda: torch.add(pt_a, pt_scalar, out=pt_out), n_ops, use_batch=False,
                            backend="pytorch", section="Elementwise / Binary / Broadcast", results=results, sync_fn=torch_sync)
                bench_batch("Reduce SUM (contig)", lambda: pt_out.copy_(pt_a), n_ops, use_batch=False,
                            backend="pytorch", section="Elementwise / Binary / Broadcast", results=results, sync_fn=torch_sync)
                print("-" * 72)
        if has_op(dcfg, "op1") and has_op(dcfg, "op2"):
            print(f"\nChained / Mixed batches [PyTorch -- {torch_dtype}]")
            print("op                           n_ops     total_ms     ms/op")
            for n_ops in BATCH_SIZES:
                def chain_same_pt():
                    torch.exp(pt_a, out=pt_out); torch.exp(pt_out, out=pt_tmp); torch.exp(pt_tmp, out=pt_out)
                bench_batch("Chain 3x EXP", chain_same_pt, n_ops, use_batch=False, ops_per_iter=3,
                            backend="pytorch", section="Chained / Mixed batches", results=results, sync_fn=torch_sync)
                def mixed_pt():
                    torch.exp(pt_a, out=pt_out); torch.add(pt_out, pt_b, out=pt_tmp); torch.add(pt_tmp, pt_scalar, out=pt_out)
                bench_batch("Mixed 3-op", mixed_pt, n_ops, use_batch=False, ops_per_iter=3,
                            backend="pytorch", section="Chained / Mixed batches", results=results, sync_fn=torch_sync)
                print("-" * 72)
        if _has_any_heavy:
            print(f"\nSoftmax / LayerNorm / MatMul [PyTorch -- {torch_dtype}]")
            print("op                           n_ops     total_ms     ms/op")
            for n_ops in HEAVY_BATCH_SIZES:
                if has_op(dcfg, "softmax"):
                    bench_batch("Softmax", lambda: torch.softmax(pt_soft_src, dim=1, out=pt_soft_dst), n_ops, use_batch=False,
                                backend="pytorch", section="Softmax / LayerNorm / MatMul", results=results, sync_fn=torch_sync)
                if has_op(dcfg, "layernorm"):
                    def layernorm_pt():
                        mean = pt_ln_src.mean(dim=1, keepdim=True); var = pt_ln_src.var(dim=1, keepdim=True, unbiased=False)
                        pt_ln_dst.copy_((pt_ln_src - mean) / torch.sqrt(var + 1e-5))
                        pt_ln_dst.mul_(pt_ln_gamma).add_(pt_ln_beta)
                    bench_batch("LayerNorm", layernorm_pt, n_ops, use_batch=False,
                                backend="pytorch", section="Softmax / LayerNorm / MatMul", results=results, sync_fn=torch_sync)
                if has_op(dcfg, "matmul"):
                    t0 = now_ms()
                    for _ in range(n_ops): torch.matmul(pt_mm_a, pt_mm_b, out=pt_mm_c)
                    torch_sync(); t1 = now_ms(); total=t1-t0; per_op=total/n_ops
                    print(f"MatMul                    {n_ops:7d}  {total:10.3f} ms  {per_op:10.6f} ms/op")
                    results.append({"section": "Softmax / LayerNorm / MatMul", "label": "MatMul",
                                    "n_ops": n_ops, "backend": "pytorch", "total_ms": total, "per_op_ms": per_op})
                print("-" * 72)
        if has_op(dcfg, "scatter"):
            print(f"\nScatter / Gather [PyTorch -- {torch_dtype}]")
            print("op                           n_ops     total_ms     ms/op")
            for n_ops in HEAVY_BATCH_SIZES:
                t0 = now_ms()
                for _ in range(n_ops): pt_out.scatter_(0, pt_indices, pt_scatter_data)
                torch_sync(); t1 = now_ms(); total=t1-t0; per_op=total/n_ops
                print(f"Scatter (dev)             {n_ops:7d}  {total:10.3f} ms  {per_op:10.6f} ms/op")
                results.append({"section": "Scatter / Gather", "label": "Scatter (dev)", "n_ops": n_ops, "backend": "pytorch", "total_ms": total, "per_op_ms": per_op})
                t0 = now_ms()
                for _ in range(n_ops): _ = pt_out[pt_indices]
                torch_sync(); t1 = now_ms(); total=t1-t0; per_op=total/n_ops
                print(f"Gather (dev)              {n_ops:7d}  {total:10.3f} ms  {per_op:10.6f} ms/op")
                results.append({"section": "Scatter / Gather", "label": "Gather (dev)", "n_ops": n_ops, "backend": "pytorch", "total_ms": total, "per_op_ms": per_op})
                print("-" * 72)
        if _transformer_ops.issubset(dcfg["supported_ops"]):
            print(f"\nTransformer block [PyTorch -- {torch_dtype}]")
            print("op                           n_ops     total_ms     ms/op")
            def transformer_block_pytorch():
                torch.matmul(pt_t_x, pt_t_wqk, out=pt_t_scores); pt_t_scores.mul_(pt_scale)
                maxv = pt_t_scores.max(dim=1, keepdim=True).values
                pt_t_scores_soft.copy_(pt_t_scores - maxv); pt_t_scores_soft.exp_()
                denom = pt_t_scores_soft.sum(dim=1, keepdim=True); pt_t_scores_soft.div_(denom)
                torch.matmul(pt_t_x, pt_t_wv, out=pt_t_v)
                torch.matmul(pt_t_scores_soft, pt_t_v, out=pt_t_attn)
                torch.matmul(pt_t_attn, pt_t_wo, out=pt_t_out)
                torch.add(pt_t_out, pt_t_x, out=pt_t_resid)
                mean = pt_t_resid.mean(dim=1, keepdim=True); var = pt_t_resid.var(dim=1, keepdim=True, unbiased=False)
                pt_t_ln.copy_((pt_t_resid - mean) / torch.sqrt(var + 1e-5))
                pt_t_ln.mul_(pt_t_gamma).add_(pt_t_beta)
            for n_ops in TRANSFORMER_ITERS:
                bench_batch("Transformer block", transformer_block_pytorch, n_ops, use_batch=False,
                            backend="pytorch", section="Transformer block", results=results, sync_fn=torch_sync)
                print("-" * 72)

    summarize_results(results)

if __name__ == "__main__":
    main()
