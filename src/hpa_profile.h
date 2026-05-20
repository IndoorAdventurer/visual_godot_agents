#pragma once

// NVTX profiling helpers. Compiled away entirely when HPA_PROFILE is not defined.
// Build with `hpa_profile=1` to enable (see SConstruct).
//
// HPA_RANGE(label)  — RAII: covers from this line to end of enclosing scope.
// HPA_PUSH(label)   — explicit range start (must be paired with HPA_POP).
// HPA_POP()         — explicit range end.
#ifdef HPA_PROFILE
#include <nvtx3/nvToolsExt.h>
struct HpaRange {
    HpaRange(const char *label) { nvtxRangePushA(label); }
    ~HpaRange()                 { nvtxRangePop(); }
};
#define HPA_RANGE(label) HpaRange _hpa_range_##__LINE__{label}
#define HPA_PUSH(label)  nvtxRangePushA(label)
#define HPA_POP()        nvtxRangePop()
#else
#define HPA_RANGE(label) ((void)0)
#define HPA_PUSH(label)  ((void)0)
#define HPA_POP()        ((void)0)
#endif
