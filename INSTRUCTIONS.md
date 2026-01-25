# ADAMAH – Python Guide (API + Instructions)

This guide focuses on **Python usage** (the `adamah` API and the `UUCISView` wrapper) and is based on:

* `benchmarks/benchmark_simple_batches.py`
* `src/adamah/uucis.py`
* `src/adamah/__init__.py`
* `src/adamah/adamah.c` and shaders in `src/adamah/shaders/*.comp`

> Note: the **Array API** (`gpu.array`, `gpu.add`, `gpu.mul`) **exists but is not implemented**.
> At the moment, the recommended usage is **maps + UUCIS**.

---

## 1) Initialization and GPU object

```python
import adamah

gpu = adamah.init(cache_mb=512, cold_cache_mb=512)  # optional cache
u = gpu.uucis  # UUCIS wrapper for maps
```

* `adamah.init(...)` returns an `Adamah` object.
* `gpu.uucis` returns the UUCIS view (used by the benchmarks as well).

---

## 2) Maps, arrays, and vars (Python)

### Generic maps

```python
u.map_init(map_id=0, dim=1, n_cells=1000, wordlength=4, pack_size=1)
# or
u.cmap_init(...)  # same as above, but returns cached locs on GPU
```

### Arrays and variables

* **Array**: `array_init` creates a 1D map (`pack_size=1`)
* **Var**: `var_init` creates a 1D map with a single element

```python
u.array_init(map_id=1, n_cells=1024, wordlength=4)
# or with cached locs
locs_all = u.carray_init(map_id=1, n_cells=1024, wordlength=4)

u.var_init(map_id=2, wordlength=4)
# or
locs_var = u.cvar_init(map_id=2, wordlength=4)
```

> In practice, `array_init` and `var_init` are shortcuts of `map_init` for 1D maps.

---

## 3) Cached locs & cached data (fundamental)

UUCIS works **much better** if you use **cached locations and cached data** on the GPU.

```python
locs = u.cache_locs(map_id=0, location_list=[0, 1, 2, 3])
vec = u.to_cached(np.array([1,2,3,4], dtype=np.float32))
```

* `cache_locs(...)` → uploads locs to GPU → returns a `CachedVar`
* `to_cached(...)` → uploads data to GPU → returns a `CachedVar`
* `cached_wait(...)` → synchronizes a ticket (if cached batching is disabled)
* `cached_download(...)` → downloads data from GPU to CPU

### Strict cached ops (default = True)

By default, **many ops require cached locs**, otherwise they raise `RuntimeError`.

You can disable this behavior:

```python
u.set_strict_cached_ops(False)
```

---

## 4) Scatter / Gather

### CPU ↔ GPU

```python
u.scatter(map_id, locs_cached, data_np)
arr = u.gather(map_id, locs_cached)
```

### Device-only (GPU ↔ GPU)

```python
locs_c = u.cache_locs(map_id, locs)
src_c  = u.to_cached(data_np)

u.scatter(map_id, locs_c, src_c)          # device scatter
out_c = u.gather(map_id, locs_c, target=u.cached(0, 0, np.float32))
```

* If `target` is a `CachedVar`, `gather` stays on the GPU and returns a `CachedVar`.
* If `target` is `None`, `gather` downloads to CPU.

---

## 5) Instructions (ops) in Python

UUCIS exposes two macro calls:

* **`mop1`** (unary / reduce / softmax / layernorm)
* **`mop2`** (binary / broadcast / matmul)

The `op_type` string is case-insensitive and may include a prefix:

* `"EXP"` = `"UNARY:EXP"`
* `"ADD"` = `"BINARY:ADD"`
* `"REDUCE:SUM"`, `"BROADCAST:ADD"`, etc.

---

### 5.1 mop1 (unary / reduce / softmax / layernorm)

**Unary** (element-wise)

```python
u.mop1("EXP", map_id=0, target=0,
       locs_src=locs_a_c, locs_dst=locs_out_c)
```

Available operations:

* `NEG, ABS, SQRT, EXP, LOG, TANH, RELU, GELU, SIN, COS, RECIP, SQR`

**Reduce** (per-pack → scalar)

```python
u.mop1("REDUCE:SUM", map_id=0, target=0,
       locs_src=locs_a_c, locs_dst=locs_out_c)
```

Operations:

* `SUM, MAX, MIN`

**Softmax** (per row)

```python
u.mop1("SOFTMAX", map_id=1, target=1,
       locs_src=locs_rows_c, locs_dst=locs_rows_out_c,
       extra={"row_size": ROW_SIZE})
```

**LayerNorm**

```python
u.mop1("LAYERNORM", map_id=2, target=2,
       locs_src=locs_src_c, locs_dst=locs_dst_c,
       extra={"dim": ROW_SIZE, "eps": 1e-5,
              "locs_gamma": locs_gamma_c, "locs_beta": locs_beta_c})
```

---

### 5.2 mop2 (binary / broadcast / matmul)

**Binary** (element-wise)

```python
u.mop2("ADD", map_id_a=0, map_id_b=0, target=0,
       locs_a=locs_a_c, locs_b=locs_b_c, locs_dst=locs_out_c)
```

Available operations:

* `ADD, SUB, MUL, DIV, POW, MIN, MAX`

**Broadcast** (scalar per pack)

```python
u.mop2("BROADCAST:ADD", map_id_a=0, map_id_b=0, target=0,
       locs_a=locs_a_c, locs_b=locs_scalar_c, locs_dst=locs_out_c)
```

Available operations:

* `ADD, SUB, MUL, DIV`

**MatMul** (batched)

```python
u.mop2("MATMUL", 3, 3, 3, extra={
    "locs_a": locs_a_c,
    "locs_b": locs_b_c,
    "locs_c": locs_c_c,
    "M": M, "K": K, "N": N,
})
```

> `mop2` requires `map_id_a == map_id_b`.

---

## 6) Batching in Python

### Auto-batching (default = True)

UUCIS automatically batches ops to reduce overhead.

```python
u.set_auto_batching(True, limit=4096)
```

### Disable auto-batching (alternative)

If you want immediate execution (one submission per op):

```python
u.set_auto_batching(False)
```

In this case **no internal batch is opened**; each op is submitted immediately.

### Manual batching (explicit)

```python
with gpu.batch():
    u.mop1(...)
    u.mop2(...)
```

### Cached batching

```python
u.set_cached_batching(True)   # automatic sync of cached vars
u.set_cached_batching(False)  # you must call cached_wait manually
```

---

## 7) Flush and synchronization (Python)

In ADAMAH, “flush” can mean two different things:

### 7.1 Batch flush (GPU submission)

* With auto-batching enabled, ops are grouped and submitted in blocks.
* The flush happens **when the batch ends** (automatically or manually).

To force it:

```python
gpu.batch_end()          # if a manual batch is open
gpu.synchronize_all()    # wait for completion of all ops
```

If auto-batching is disabled (`u.set_auto_batching(False)`), ops are submitted immediately and there is no batch flush.

---

### 7.2 Staging memory flush/invalidate (CPU ↔ GPU)

When using CPU `scatter` / `gather`, ADAMAH internally handles:

* `vkFlushMappedMemoryRanges` after CPU writes
* `vkInvalidateMappedMemoryRanges` before CPU reads

This is automatic: **there is no Python method to call it manually**.

---

### 7.3 Cached uploads (tickets)

If cached batching is disabled, you must explicitly synchronize:

```python
u.set_cached_batching(False)
u.cached_wait(cached_var)      # or gpu.synchronize(ticket)
```

---

## 8) Minimal example (benchmark-style)

```python
import numpy as np
import adamah

gpu = adamah.init(512)
u = gpu.uucis

# map 0: float32 array
u.array_init(0, n_cells=1024, wordlength=4)
locs = np.arange(1024, dtype=np.uint32)
locs_c = u.cache_locs(0, locs)

# data
x = np.random.randn(1024).astype(np.float32)
y = np.random.randn(1024).astype(np.float32)

u.scatter(0, locs_c, x)

# y -> cached
y_c = u.to_cached(y)

# output
out_c = u.cache_locs(0, locs)  # same locs for output

# op: out = exp(x) + y (2 steps)
u.mop1("EXP", 0, 0, locs_src=locs_c, locs_dst=out_c)
u.mop2("ADD", 0, 0, 0, locs_a=out_c, locs_b=locs_c, locs_dst=out_c)

# download
res = u.gather(0, out_c)
```

---

## 9) What the benchmark uses

`benchmarks/benchmark_simple_batches.py` uses:

* `carray_init`, `cache_locs`, `scatter`, `gather`
* `mop1`: `EXP`, `REDUCE:SUM`, `SOFTMAX`, `LAYERNORM`
* `mop2`: `ADD`, `BROADCAST:ADD`, `MATMUL`
* batching (`gpu.batch_begin/end`) and cached locs for performance

---

## 10) Array API (current status)

Methods present but **not implemented**:

* `gpu.array(data)`
* `gpu.add(a, b)`
* `gpu.mul(a, b)`

Actual message:

```
NotImplementedError: array() API is WIP
```

So for now, **maps + UUCIS** must be used.

---

## 11) Quick notes

* Locations are **pack indices** (not byte offsets).
* `target` in `mop1/mop2` can be `None` or the same `map_id`.
* For performance, prefer **cached locs + batching**.
