# ADAMAH v4.0.0 - Guida Completa

**Map-Centric GPU Compute Library**

Autore: Samuele Scuglia  
Data: 18 Gennaio 2026  
Licenza: CC BY-NC 4.0

---

## Indice

1. [Introduzione](#introduzione)
2. [Installazione](#installazione)
3. [Concetti Fondamentali](#concetti-fondamentali)
4. [API Reference](#api-reference)
5. [Esempi Pratici](#esempi-pratici)
6. [Performance Tips](#performance-tips)
7. [Troubleshooting](#troubleshooting)

---

## Introduzione

ADAMAH è una libreria GPU compute basata su Vulkan con un'architettura **Map-Centric**. 

### Filosofia

A differenza delle librerie tradizionali (NumPy, CuPy) che operano su array lineari, ADAMAH usa **Memory Maps**: strutture dati GPU-resident accessibili tramite indici (scatter/gather).

```
Approccio Tradizionale:          Approccio ADAMAH:
─────────────────────            ─────────────────
array[0:1000] = sin(x)           map[locs] = sin(map[locs])
↓                                ↓
Opera su range contigui          Opera su posizioni sparse
```

### Quando usare ADAMAH

✅ **Usa ADAMAH per:**
- Embedding tables (NLP, RecSys)
- Sparse computations
- Lookup tables GPU-resident
- Operazioni su subset non contigui

❌ **Non usare ADAMAH per:**
- Operazioni dense su array contigui (usa NumPy/CuPy)
- Algebra lineare classica (usa cuBLAS)

---

## Installazione

### Requisiti

- Python 3.8+
- Vulkan SDK e driver GPU
- GCC con supporto C11
- numpy

### Ubuntu/Debian

```bash
# Installa Vulkan
sudo apt install vulkan-tools libvulkan-dev

# Verifica GPU
vulkaninfo | grep "deviceName"

# Installa ADAMAH
pip install adamah
# oppure da source:
cd ADAMAH-main
pip install -e .
```

### Verifica Installazione

```python
import adamah
gpu = adamah.Adamah()  # Stampa: "ADAMAH v4: NVIDIA GeForce RTX 3070"
gpu.shutdown()
```

---

## Concetti Fondamentali

### Memory Map

Una **Map** è un blocco di memoria GPU organizzato in **packs**:

```
Map Structure:
┌────────────────────────────────────────────────────────────┐
│ Pack 0: [elem0, elem1, elem2, ..., elem_{pack_size-1}]    │
│ Pack 1: [elem0, elem1, elem2, ..., elem_{pack_size-1}]    │
│ Pack 2: [elem0, elem1, elem2, ..., elem_{pack_size-1}]    │
│ ...                                                        │
│ Pack N: [elem0, elem1, elem2, ..., elem_{pack_size-1}]    │
└────────────────────────────────────────────────────────────┘
```

**Parametri Map:**
- `word_size`: byte per elemento (4 = float32, 8 = float64)
- `pack_size`: elementi per pack
- `n_packs`: numero totale di pack

**Dimensione totale** = word_size × pack_size × n_packs bytes

### Scatter / Gather

**Scatter** (CPU → GPU): scrive dati in posizioni specifiche della Map
**Gather** (GPU → CPU): legge dati da posizioni specifiche della Map

```python
# Scatter: scrivi 3 pack alle posizioni 0, 100, 999
locs = np.array([0, 100, 999], dtype=np.uint32)
data = np.array([...], dtype=np.float32)  # 3 * pack_size elementi
gpu.scatter(map_id, locs, data)

# Gather: leggi gli stessi pack
result = gpu.gather(map_id, locs)  # shape: (3 * pack_size,)
```

### GPU Operations

Le operazioni GPU lavorano **direttamente sulla Map** senza trasferire dati alla CPU:

```python
# map[dst_locs] = sin(map[src_locs])
gpu.map_sin(map_id, src_locs, dst_locs)

# map[out_locs] = map[a_locs] + map[b_locs]
gpu.map_add(map_id, a_locs, b_locs, out_locs)
```

**Importante:** `src_locs`, `dst_locs`, etc. sono array di **indici di pack**, non di elementi!

---

## API Reference

### Inizializzazione

```python
import adamah

# Crea contesto GPU
gpu = adamah.Adamah()

# Context manager (auto-shutdown)
with adamah.Adamah() as gpu:
    # ... operazioni ...
# shutdown automatico

# Shutdown manuale
gpu.shutdown()
```

### Memory Maps

#### `map_init(map_id, word_size, pack_size, n_packs)`

Crea una nuova Memory Map.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `map_id` | int | Identificatore (0-15) |
| `word_size` | int | Byte per elemento (default: 4) |
| `pack_size` | int | Elementi per pack (default: 1) |
| `n_packs` | int | Numero di pack (default: 1000000) |

```python
# Map di 1M float singoli
gpu.map_init(0, word_size=4, pack_size=1, n_packs=1_000_000)

# Map di 100k vettori da 768 float (embeddings)
gpu.map_init(1, word_size=4, pack_size=768, n_packs=100_000)

# Map di 50k matrici 4x4 (transforms)
gpu.map_init(2, word_size=4, pack_size=16, n_packs=50_000)
```

#### `map_destroy(map_id)`

Distrugge una Map e libera la memoria GPU.

```python
gpu.map_destroy(0)
```

#### `map_size(map_id)`

Ritorna il numero di pack nella Map.

```python
n = gpu.map_size(0)  # 1000000
```

### Data Transfer

#### `scatter(map_id, locs, data)`

Scrive dati nella Map alle posizioni specificate.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `map_id` | int | Map target |
| `locs` | np.array(uint32) | Indici dei pack |
| `data` | np.array | Dati (len = len(locs) × pack_size) |

```python
locs = np.array([0, 5, 10], dtype=np.uint32)
data = np.random.randn(3 * 128).astype(np.float32)  # 3 pack da 128
gpu.scatter(0, locs, data)
```

#### `gather(map_id, locs)`

Legge dati dalla Map alle posizioni specificate.

| Parametro | Tipo | Descrizione |
|-----------|------|-------------|
| `map_id` | int | Map sorgente |
| `locs` | np.array(uint32) | Indici dei pack |

**Ritorna:** np.array di shape (len(locs) × pack_size,)

```python
locs = np.array([0, 5, 10], dtype=np.uint32)
result = gpu.gather(0, locs)  # shape: (384,) se pack_size=128
```

### GPU Operations

#### `map_op1(map_id, op, locs_src, locs_dst)`

Operazione unaria: `map[locs_dst] = op(map[locs_src])`

```python
gpu.map_op1(0, adamah.OP_SIN, src_locs, dst_locs)
```

**Operazioni disponibili:**

| Costante | Operazione |
|----------|------------|
| `OP_NEG` | -x |
| `OP_ABS` | \|x\| |
| `OP_SQRT` | √x |
| `OP_EXP` | eˣ |
| `OP_LOG` | ln(x) |
| `OP_TANH` | tanh(x) |
| `OP_RELU` | max(0, x) |
| `OP_GELU` | GELU(x) |
| `OP_SIN` | sin(x) |
| `OP_COS` | cos(x) |
| `OP_RECIP` | 1/x |
| `OP_SQR` | x² |

#### `map_op2(map_id, op, locs_a, locs_b, locs_dst)`

Operazione binaria: `map[locs_dst] = map[locs_a] op map[locs_b]`

```python
gpu.map_op2(0, adamah.OP_ADD, a_locs, b_locs, out_locs)
```

**Operazioni disponibili:**

| Costante | Operazione |
|----------|------------|
| `OP_ADD` | a + b |
| `OP_SUB` | a - b |
| `OP_MUL` | a × b |
| `OP_DIV` | a / b |
| `OP_POW` | aᵇ |
| `OP_MIN` | min(a, b) |
| `OP_MAX` | max(a, b) |

#### Shortcuts

```python
# Unary
gpu.map_sin(map_id, src, dst)
gpu.map_cos(map_id, src, dst)
gpu.map_exp(map_id, src, dst)

# Binary
gpu.map_add(map_id, a, b, dst)
gpu.map_mul(map_id, a, b, dst)
```

### Persistence

#### `map_save(map_id, path)`

Salva una Map su disco.

```python
gpu.map_save(0, "embeddings.bin")
```

#### `map_load(map_id, path)`

Carica una Map da disco.

```python
gpu.map_load(0, "embeddings.bin")
```

**Formato file:**
```
[4 bytes] word_size (uint32)
[4 bytes] pack_size (uint32)
[4 bytes] n_packs (uint32)
[N bytes] data (word_size × pack_size × n_packs)
```

---

## Esempi Pratici

### Esempio 1: Lookup Table

```python
import adamah
import numpy as np

gpu = adamah.Adamah()

# Crea tabella: 10000 valori float
gpu.map_init(0, word_size=4, pack_size=1, n_packs=10000)

# Popola con valori precomputati (es. sin per angoli 0-360°)
angles = np.linspace(0, 2*np.pi, 10000).astype(np.float32)
sin_values = np.sin(angles)
all_locs = np.arange(10000, dtype=np.uint32)
gpu.scatter(0, all_locs, sin_values)

# Lookup veloce
query_locs = np.array([0, 2500, 5000, 7500], dtype=np.uint32)
results = gpu.gather(0, query_locs)
print(results)  # [0.0, 1.0, 0.0, -1.0] circa

gpu.shutdown()
```

### Esempio 2: Embedding Table (NLP)

```python
import adamah
import numpy as np

gpu = adamah.Adamah()

# Vocabulary: 50000 parole, embedding dim: 768
VOCAB_SIZE = 50000
EMBED_DIM = 768

gpu.map_init(0, word_size=4, pack_size=EMBED_DIM, n_packs=VOCAB_SIZE)

# Carica embeddings pretrained
embeddings = np.random.randn(VOCAB_SIZE, EMBED_DIM).astype(np.float32)
all_indices = np.arange(VOCAB_SIZE, dtype=np.uint32)
gpu.scatter(0, all_indices, embeddings.flatten())

# Sentence: "hello world" → token ids [1234, 5678]
token_ids = np.array([1234, 5678], dtype=np.uint32)
vectors = gpu.gather(0, token_ids)
vectors = vectors.reshape(2, EMBED_DIM)  # (2, 768)

print(f"Shape: {vectors.shape}")  # (2, 768)

gpu.shutdown()
```

### Esempio 3: Batch Processing con GPU Ops

```python
import adamah
import numpy as np

gpu = adamah.Adamah()

# Map con spazio per input (0-999) e output (1000-1999)
gpu.map_init(0, word_size=4, pack_size=256, n_packs=2000)

# Input data
input_locs = np.arange(100, dtype=np.uint32)  # pack 0-99
output_locs = input_locs + 1000               # pack 1000-1099

data = np.random.randn(100 * 256).astype(np.float32)
gpu.scatter(0, input_locs, data)

# Pipeline GPU: sin → relu → output
# Step 1: sin(input) → output
gpu.map_sin(0, input_locs, output_locs)

# Step 2: relu(output) → output (in-place)
gpu.map_op1(0, adamah.OP_RELU, output_locs, output_locs)

# Leggi risultato
result = gpu.gather(0, output_locs)
print(f"Result shape: {result.shape}")  # (25600,)

gpu.shutdown()
```

### Esempio 4: Multi-Map Operations

```python
import adamah
import numpy as np

gpu = adamah.Adamah()

# Map 0: Weights
# Map 1: Activations
# Map 2: Gradients

gpu.map_init(0, word_size=4, pack_size=512, n_packs=1000)  # Weights
gpu.map_init(1, word_size=4, pack_size=512, n_packs=1000)  # Activations
gpu.map_init(2, word_size=4, pack_size=512, n_packs=1000)  # Gradients

# Inizializza weights
weights = np.random.randn(1000 * 512).astype(np.float32) * 0.01
gpu.scatter(0, np.arange(1000, dtype=np.uint32), weights)

# Forward pass: activation = tanh(weights)
locs = np.arange(1000, dtype=np.uint32)
gpu.map_op1(0, adamah.OP_TANH, locs, locs)  # In map 0

# Copia a map 1
activations = gpu.gather(0, locs)
gpu.scatter(1, locs, activations)

# ... training loop ...

# Salva checkpoint
gpu.map_save(0, "weights.bin")

gpu.shutdown()
```

### Esempio 5: In-Place Operations

```python
import adamah
import numpy as np

gpu = adamah.Adamah()
gpu.map_init(0, word_size=4, pack_size=128, n_packs=1000)

# Carica dati
locs = np.array([0, 1, 2], dtype=np.uint32)
data = np.random.randn(3 * 128).astype(np.float32)
gpu.scatter(0, locs, data)

# Operazioni in-place (src == dst)
gpu.map_sin(0, locs, locs)    # sin in-place
gpu.map_exp(0, locs, locs)    # exp in-place (ora è exp(sin(x)))

# Risultato
result = gpu.gather(0, locs)

gpu.shutdown()
```

---

## Performance Tips

### 1. Minimizza i trasferimenti CPU↔GPU

```python
# ❌ Male: trasferimento per ogni operazione
for i in range(100):
    gpu.scatter(0, locs, data)
    gpu.map_sin(0, locs, locs)
    result = gpu.gather(0, locs)

# ✅ Bene: batch operations
gpu.scatter(0, all_locs, all_data)
for i in range(100):
    gpu.map_sin(0, locs, locs)  # Solo GPU
result = gpu.gather(0, locs)    # Un solo gather
```

### 2. Usa pack_size appropriato

```python
# ❌ Male: pack troppo piccoli (overhead)
gpu.map_init(0, pack_size=1, n_packs=1_000_000)

# ✅ Bene: pack allineati alla cache GPU (64-256)
gpu.map_init(0, pack_size=128, n_packs=10_000)
```

### 3. Riusa le Map

```python
# ❌ Male: crea/distrugge continuamente
for batch in batches:
    gpu.map_init(0, ...)
    # ... ops ...
    gpu.map_destroy(0)

# ✅ Bene: riusa la stessa map
gpu.map_init(0, pack_size=256, n_packs=max_batch_size)
for batch in batches:
    gpu.scatter(0, batch_locs, batch_data)
    # ... ops ...
gpu.map_destroy(0)
```

### 4. Usa locations contigue quando possibile

```python
# ❌ Male: accessi random
locs = np.array([999, 0, 500, 123], dtype=np.uint32)

# ✅ Bene: accessi contigui (cache-friendly)
locs = np.arange(0, 100, dtype=np.uint32)
```

---

## Troubleshooting

### "Vulkan error" all'inizializzazione

```bash
# Verifica driver
vulkaninfo | grep "deviceName"

# Se non funziona, installa driver
sudo apt install nvidia-driver-535  # NVIDIA
sudo apt install mesa-vulkan-drivers  # AMD/Intel
```

### "Shader not found"

Gli shader `.spv` devono essere in:
- `./shaders/`
- `./src/adamah/shaders/`

```bash
ls src/adamah/shaders/*.spv
# Deve mostrare map_op1.spv, map_op2.spv
```

### Risultati incorretti

1. Verifica dtype:
```python
locs = np.array([...], dtype=np.uint32)  # DEVE essere uint32
data = np.array([...], dtype=np.float32)  # DEVE essere float32
```

2. Verifica dimensioni:
```python
# len(data) DEVE essere = len(locs) * pack_size
assert len(data) == len(locs) * pack_size
```

### Memory error

```python
# Controlla dimensione Map
total_bytes = word_size * pack_size * n_packs
print(f"Map size: {total_bytes / 1e9:.2f} GB")

# RTX 3070 ha 8GB VRAM
# Non superare ~6GB per Map
```

---

## Architettura Interna

```
Python API (__init__.py)
         │
         ▼
    ctypes FFI
         │
         ▼
   adamah.c (Vulkan)
         │
    ┌────┴────┐
    ▼         ▼
 Maps      Pipelines
(VRAM)    (Compute)
    │         │
    └────┬────┘
         ▼
   GPU Execution
   (map_op1.spv)
   (map_op2.spv)
```

**Vulkan Resources:**
- `VkBuffer` per ogni Map (DEVICE_LOCAL)
- `VkBuffer` staging per trasferimenti
- `VkPipeline` per unary/binary ops
- `VkCommandBuffer` singolo per semplicità

---

## Changelog

### v4.0.0 (2026-01-18)
- Architettura Map-Centric
- API semplificata: scatter/gather + map_op
- Rimosso: CPU ops, mode selector, named buffers
- Shader dedicati per operazioni su Map

### v3.0.0
- Prima versione GPU funzionante
- Fix: CPU→GPU routing bug (130x speedup)

---

**© 2026 Samuele Scuglia - CC BY-NC 4.0**
