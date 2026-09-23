#ifndef CRYPTOFACE_SCOPED_KEY_EXPORT_CACHE_H
#define CRYPTOFACE_SCOPED_KEY_EXPORT_CACHE_H

// Experimental, key-only host storage. The caller must guarantee immutable
// evaluation keys during a context revision, and start a new revision on key
// replacement. Per-request Go wrapper handles are intentionally NOT identities.
// The isolated integration is opt-in; production defaults remain unchanged.
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeindex>

namespace cryptoface {
enum class KeyKind { Relin, Galois, Switch };

struct KeyExportMetadata {
    KeyKind kind;
    int level;
    int special_level;
    int montgomery_bits;
    uint64_t galois_element;
    // Graph input identity distinguishes e.g. switch-dts from switch-std even
    // when their levels/formats happen to match. Never use a fresh Go handle.
    uint64_t source_key_slot;
    auto tuple() const {
        return std::make_tuple(kind, level, special_level, montgomery_bits, galois_element, source_key_slot);
    }
    bool operator<(const KeyExportMetadata& other) const { return tuple() < other.tuple(); }
    void validate() const {
        if (kind != KeyKind::Relin && kind != KeyKind::Galois && kind != KeyKind::Switch)
            throw std::invalid_argument("Only immutable evaluation keys may be cached");
        if (kind != KeyKind::Galois && galois_element != 0)
            throw std::invalid_argument("Galois element on non-Galois key");
    }
};

struct KeyContextIdentity {
    uintptr_t address;
    uint64_t handle;
    uint64_t key_revision;
    std::string parameter_signature;
    bool operator==(const KeyContextIdentity& other) const {
        return address == other.address && handle == other.handle &&
            key_revision == other.key_revision && parameter_signature == other.parameter_signature;
    }
};

class ScopedKeyExportCache {
    struct Entry {
        std::shared_ptr<void> value;
        std::type_index type;
        uint64_t bytes;
    };
    struct State {
        std::mutex mutex;
        const uint64_t cap;
        uint64_t used = 0, hits = 0, misses = 0, rejected = 0, generation = 0;
        bool active = false, has_identity = false;
        KeyContextIdentity identity{};
        std::map<KeyExportMetadata, Entry> entries;
        explicit State(uint64_t bytes) : cap(bytes) {}
        void clear() { entries.clear(); used = 0; }
    };
public:
    struct Stats {
        uint64_t cap, used, entries, hits, misses, rejected, generation;
    };
    class Request {
        std::shared_ptr<State> state_;
        bool completed_ = false;
        friend class ScopedKeyExportCache;
        explicit Request(std::shared_ptr<State> state) : state_(std::move(state)) {}
    public:
        Request(const Request&) = delete;
        Request& operator=(const Request&) = delete;
        ~Request() {
            std::lock_guard<std::mutex> lock(state_->mutex);
            // Abort/exception cannot leave partially admitted entries for reuse.
            // Outstanding consumers retain their own shared references.
            if (!completed_) state_->clear();
            state_->active = false;
        }
        void complete() { completed_ = true; }
    };

    explicit ScopedKeyExportCache(uint64_t cap) : state_(std::make_shared<State>(cap)) {
        if (cap > (uint64_t{32} << 30))
            throw std::invalid_argument("Host key export cap must not exceed 32 GiB");
    }
    ScopedKeyExportCache(const ScopedKeyExportCache&) = delete;
    ScopedKeyExportCache& operator=(const ScopedKeyExportCache&) = delete;

    bool enabled() const { return state_->cap != 0; }
    void invalidate() {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->active) throw std::logic_error("Cannot replace evaluation keys during a request");
        state_->clear();
        state_->has_identity = false;
        // The next begin() advances generation, also invalidating device keys
        // when the FheTaskGpu integration forwards generation as its identity.
    }

    Request begin(const KeyContextIdentity& identity, bool immutable_keys_contract) {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->active) throw std::logic_error("Concurrent runs on one FHE task are unsupported");
        if (state_->cap != 0) {
            if (!immutable_keys_contract || identity.address == 0 || identity.handle == 0 ||
                identity.parameter_signature.empty())
                throw std::invalid_argument("Host key reuse requires explicit immutable context identity");
            if (!state_->has_identity || !(identity == state_->identity)) {
                state_->clear();
                state_->identity = identity;
                state_->has_identity = true;
                ++state_->generation;
            }
        }
        state_->active = true;
        return Request(state_);
    }

    template<class T> std::shared_ptr<T> find(const KeyExportMetadata& key) {
        key.validate();
        std::lock_guard<std::mutex> lock(state_->mutex);
        require_active();
        if (state_->cap == 0) return {};
        const auto found = state_->entries.find(key);
        if (found == state_->entries.end()) { ++state_->misses; return {}; }
        if (found->second.type != std::type_index(typeid(T)))
            throw std::logic_error("Host key export type mismatch");
        ++state_->hits;
        return std::static_pointer_cast<T>(found->second.value);
    }

    template<class T> std::shared_ptr<T> admit(const KeyExportMetadata& key,
                                              std::shared_ptr<T> value, uint64_t bytes) {
        key.validate();
        if (!value || bytes == 0) throw std::invalid_argument("Empty exported key");
        std::lock_guard<std::mutex> lock(state_->mutex);
        require_active();
        const auto found = state_->entries.find(key);
        if (found != state_->entries.end()) {
            if (found->second.type != std::type_index(typeid(T)) || found->second.bytes != bytes)
                throw std::logic_error("Inconsistent export for immutable key metadata");
            return std::static_pointer_cast<T>(found->second.value);
        }
        if (bytes > state_->cap - state_->used) { ++state_->rejected; return value; }
        state_->entries.emplace(key, Entry{value, std::type_index(typeid(T)), bytes});
        state_->used += bytes;
        return value;
    }

    Stats stats() const {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return {state_->cap, state_->used, state_->entries.size(), state_->hits,
                state_->misses, state_->rejected, state_->generation};
    }
private:
    void require_active() const {
        if (!state_->active) throw std::logic_error("Host key export outside a request");
    }
    std::shared_ptr<State> state_;
};
}  // namespace cryptoface
#endif
