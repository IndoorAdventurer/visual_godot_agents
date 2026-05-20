# GPU Readback Profiling Findings

**Date:** 2026-05-20  
**Setup:** 32 environments, 128×128 observations, 4 channels (RGBA)

## Timing breakdown (per frame, steady state)

| Phase | Time | What it represents |
|---|---|---|
| `force_draw` | ~2.4 ms | Render thread building + submitting Vulkan commands |
| `render_gpu_wait` | ~3.4 ms | GPU executing the 32 SubViewport renders |
| `compute_dispatch` | ~17 µs | CPU recording compute commands (no Vulkan calls) |
| `compute_gpu_wait` | ~140 µs | GPU executing the copy-to-staging-buffer shader |
| `buffer_get_data` | ~500 µs | DMA transfer of ~2 MB from VRAM to CPU |
| **Total** | **~6.5 ms** | |

## Key findings

### force_draw is not GPU-synchronous

`RenderingServer::force_draw(false)` waits for the render thread to build and
submit Vulkan commands, then returns — before the GPU has finished executing
them. The 2.4 ms is the render-thread CPU cost. The actual GPU render time
(3.4 ms) is paid later.

### Compute commands are deferred until buffer_get_data

`compute_list_begin/dispatch/end` do not call any Vulkan APIs — they only
record into Godot's internal command list. Submission to the GPU is deferred
until `buffer_get_data` is called. Confirmed by the absence of any `vk*` calls
during the `compute_dispatch` window in the Vulkan trace (nsys).

The internal sequence inside `buffer_get_data`:
1. Allocate a CPU-visible readback buffer (~8 µs)
2. `vkWaitForFences` for the render work (~3 ms)
3. Submit the deferred compute dispatches + copy command (~60 µs)
4. `vkWaitForFences` for compute + copy (~270 µs)

Before this investigation, `buffer_get_data` appeared to take ~3.56 ms in
total; almost all of that was waiting for the GPU renders, not the readback.

### RenderingDevice::submit() / sync() only work on local devices

Calling these on the global rendering device (from `get_rendering_device()`)
produces: *"Only local devices can submit and sync."*

Workaround used for measurement: calling `buffer_get_data` on a tiny 4-byte
probe buffer creates an explicit GPU sync point without this restriction.

### RenderingServer::force_sync() is useless here

`force_sync()` syncs the CPU render-thread queue, not the GPU. After
`force_draw` returns (the render thread has already submitted its commands),
`force_sync` finds nothing to wait for and returns in nanoseconds.

## Optimization opportunity

The 3.4 ms GPU render wait is the dominant cost and is currently fully
sequential with compute + transfer. **Pipelining** — submitting frame N's
renders, immediately running physics for frame N+1, and only waiting for the
frame N fence at observation-send time — could hide most of this latency, at
the cost of one step of visual lag in observations (acceptable for RL).

The compute shader (140 µs) and DMA transfer (500 µs) are cheap enough that
further optimization there is not warranted.
