#pragma once
#include <cstdint>
#include <cstdio>

// Profiling instrumentation, compiled in only when -DHPA_PROFILE is passed to scons.
// Usage: call record_print() each step; it prints a summary line every k_interval steps.

struct ProfileStats {
    // Total steps since construction — never reset, used for the step label and interval check.
    uint64_t count  = 0;
    // Window accumulators — reset after each interval print.
    uint64_t sum_us = 0;
    uint64_t min_us = UINT64_MAX;
    uint64_t max_us = 0;

    // Records `us`. When count reaches a multiple of k_interval, prints a
    // [HPA_PROFILE] line to stdout and resets the window accumulators.
    void record_print(const char *label, uint64_t us, int num_envs) {
        ++count;
        sum_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
        if (count % k_interval == 0) {
            printf("[HPA_PROFILE] step=%llu envs=%d | %s:"
                   " mean=%.1f min=%llu max=%llu\n",
                   (unsigned long long)count, num_envs, label,
                   double(sum_us) / k_interval,
                   (unsigned long long)min_us,
                   (unsigned long long)max_us);
            fflush(stdout);
            sum_us = 0; min_us = UINT64_MAX; max_us = 0;
        }
    }

    static constexpr uint64_t k_interval = 1000;
};
