#pragma once

// NVTX profiling helpers. Compiled away entirely when VGA_PROFILE is not defined.
// Build with `vga_profile=1` to enable (see SConstruct).
//
// VGA_PROFILE_RANGE(label)  — RAII: covers from this line to end of enclosing scope.
// VGA_PROFILE_PUSH(label)   — explicit range start (must be paired with VGA_PROFILE_POP).
// VGA_PROFILE_POP()         — explicit range end.
#ifdef VGA_PROFILE
#include <nvtx3/nvToolsExt.h>
struct VgaRange {
    VgaRange(const char *label) { nvtxRangePushA(label); }
    ~VgaRange()                 { nvtxRangePop(); }
};
#define VGA_PROFILE_RANGE(label) VgaRange _vga_range_##__LINE__{label}
#define VGA_PROFILE_PUSH(label)  nvtxRangePushA(label)
#define VGA_PROFILE_POP()        nvtxRangePop()
#else
#define VGA_PROFILE_RANGE(label) ((void)0)
#define VGA_PROFILE_PUSH(label)  ((void)0)
#define VGA_PROFILE_POP()        ((void)0)
#endif
