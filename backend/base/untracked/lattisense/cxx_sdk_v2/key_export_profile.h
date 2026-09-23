#ifndef CRYPTOFACE_KEY_EXPORT_PROFILE_H
#define CRYPTOFACE_KEY_EXPORT_PROFILE_H

// Diagnostic only: records metadata and timing, never coefficients or images.
// Disabled unless explicitly enabled before process startup. Summed duration is
// CPU export work across threads, NOT critical-path or end-to-end latency.
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace lattisense {
class KeyExportProfile {
public:
    using Clock = std::chrono::steady_clock;

    KeyExportProfile(const char* kind, uint64_t parameter, uint64_t key,
                     int level, int sp_level, int mf_bits, uint64_t element)
        : enabled_(enabled()), kind_(kind), parameter_(parameter), key_(key),
          level_(level), sp_level_(sp_level), mf_bits_(mf_bits), element_(element) {
        if (enabled_) start_ = Clock::now();
    }

    void finish(uint64_t bytes) {
        if (!enabled_ || finished_) return;
        const auto end = Clock::now();
        const auto start_us = std::chrono::duration_cast<std::chrono::microseconds>(
            start_.time_since_epoch()).count();
        const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start_).count();
        // One stdio call keeps complete records together across CPU workers.
        std::fprintf(stderr,
            "[Host Key Export] kind=%s parameter=%" PRIu64 " key=%" PRIu64
            " level=%d sp_level=%d mf_bits=%d element=%" PRIu64
            " start_us=%lld elapsed_us=%lld bytes=%" PRIu64 "\n",
            kind_, parameter_, key_, level_, sp_level_, mf_bits_, element_,
            static_cast<long long>(start_us), static_cast<long long>(elapsed_us), bytes);
        finished_ = true;
    }

private:
    static bool enabled() {
        static const bool value = [] {
            const char* flag = std::getenv("LATTISENSE_GPU_KEY_EXPORT_PROFILE");
            return flag != nullptr && std::strcmp(flag, "1") == 0;
        }();
        return value;
    }
    bool enabled_;
    bool finished_ = false;
    const char* kind_;
    uint64_t parameter_, key_;
    int level_, sp_level_, mf_bits_;
    uint64_t element_;
    Clock::time_point start_{};
};
}  // namespace lattisense
#endif
