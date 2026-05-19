# HPA Scaling Benchmark Findings

**Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU  
**Commit:** `8e039fa`  
**Resolution tested:** 1×1, 84×84, 128×128 px  

---

## Throughput vs number of environments

All three resolutions behave nearly identically, indicating the GPU readback is
**not** the bottleneck at these env counts — something serial on the CPU is.

| envs | 1×1 (env·steps/s) | 84×84 | 128×128 |
|-----:|------------------:|------:|--------:|
|    1 |             2,981 | 2,384 |   2,690 |
|    2 |             4,566 | 4,221 |   3,962 |
|    4 |             5,936 | 4,421 |   5,007 |
|    8 |             7,741 | 7,059 |   6,559 |
|   16 |             8,581 | 6,744 |   6,450 |
|   32 |             6,734 | 5,860 |   5,922 |

Peak throughput is around **8–16 envs**, after which gains plateau and reverse.
Scaling efficiency drops to ~7% at 32 envs.

---

## Step latency breakdown at 32 envs, 128×128

### HPA_PROFILE (CPU wall time)

A profiled run (`-DHPA_PROFILE`, 2500 steps, no warmup) decomposed each step.
Window 1 (steps 1–1000) is inflated by GPU JIT and cache cold-starts; window 2
(steps 1001–2000) is the steady-state picture and aligns with the stored benchmark.

| Component | Window 1 (cold) | Window 2 (steady) |
|---|---|---|
| `force_draw` (CPU wall time) | 2,853 µs | 1,935 µs |
| `buffer_get_data` (CPU wall time) | 3,581 µs | 2,747 µs |
| Everything else (IPC, GDScript, semaphores) | — | ~300 µs |
| **Total step latency** | **~6.4 ms** | **~5.0 ms** |

### Nsight Systems (GPU timeline)

A Vulkan trace (`nsys profile --trace=vulkan,osrt`) reveals what actually happens
on the GPU. Two `vkQueueSubmit` + `vkWaitForFences` pairs occur per step:

1. **`force_draw`** — submits and waits for all 32 SubViewport renders
2. **compute dispatch** — submits and waits for the copy_viewports shader

The steady-state nsys numbers (after JIT warmup) show ~1.4× overhead vs baseline,
which is confirmed by the force_draw fence aligning with HPA_PROFILE:

| Component | nsys-observed (steady) | nsys-corrected |
|---|---|---|
| `force_draw` fence wait | 2.4 ms | ~1.85 ms ✓ |
| compute shader fence wait | 250 µs | ~190 µs |

### Corrected steady-state budget

| Component | Time | Notes |
|---|---|---|
| `force_draw` GPU work | ~1.85 ms | Real GPU cost — 32 SubViewport renders |
| CPU overhead in `buffer_get_data` | ~2.50 ms | PackedByteArray alloc + 2× memcpy of 2 MB |
| compute shader GPU work | ~0.19 ms | Tiny — GPU finishes almost instantly |
| GDScript agent collection + semaphores | ~0.50 ms | 32 virtual dispatches + IPC |
| **Total** | **~5.0 ms** | |

---

## Root cause of poor scaling

The original hypothesis — that per-dispatch GPU kernel launch overhead inside
`fetch_frame()` causes the poor scaling — is **not supported** by the Vulkan trace.
The compute shader fence wait is only ~190 µs for 32 envs, meaning the GPU finishes
almost instantly regardless of dispatch count.

The actual bottleneck is **CPU-side**: `buffer_get_data` allocates a `PackedByteArray`
and performs two full 2 MB memcpys per step (GPU-mapped memory → PackedByteArray →
shared memory). This is a Godot API limitation — there is no way to obtain a direct
pointer to GPU-mapped memory via the public RenderingDevice API.

**Revised optimization priorities:**

1. **Eliminate the double memcpy** — the ideal fix is a zero-copy path from the GPU
   staging buffer directly into shared memory. This likely requires either a Godot
   engine patch or a custom Vulkan memory import (mapping the shared memory region
   as a Vulkan host-visible buffer).
2. **Profile GDScript agent collection** — 32 virtual dispatches per step is measurable
   overhead; moving observation collection into C++ would remove it entirely.
3. **Collapsing compute dispatches** — low priority given the GPU is already fast,
   but still cheap to do and removes one source of per-env overhead.
