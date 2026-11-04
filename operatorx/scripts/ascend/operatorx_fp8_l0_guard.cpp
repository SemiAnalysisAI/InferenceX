#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace {
constexpr int kBlocked = 100000;
constexpr uint64_t kCapacity = 65536;
constexpr size_t kTilingBytes = 152;
constexpr const char *kKernel =
    "QuantBatchMatmulV3_ND_ND_fp8e4m3_fp8e4m3_fp32_bf16_high_performance";
struct Placeholder { uint32_t addrOffset; uint32_t dataOffset; };
static_assert(sizeof(Placeholder) == 8, "Installed placeholder ABI");
struct Config { uint32_t m = 0, n = 0, k = 0; int mode = 0; };
struct Decision {
    int code = 2;
    std::string reason = "not_inspected", kernel, hex;
    uint64_t key = 0, l0a = 0;
    uint32_t m = 0, n = 0, k = 0, bm = 0, bn = 0, bk = 0;
    size_t supplied = 0, args_bytes = 0;
    int64_t tiling_offset = -1, address_offset = -1;
};
std::mutex g_mutex;
Config g_config;
std::unordered_map<void *, uint64_t> g_keys;
std::unordered_set<std::string> g_logged;
std::string g_last = "{\"decision\":\"inactive\",\"observed\":false}";

uint32_t u32(const unsigned char *p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
std::string quote(const std::string &value) {
    std::string out = "\"";
    constexpr char digits[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') { out += '\\'; out += char(c); }
        else if (c < 32) { out += "\\u00"; out += digits[c >> 4]; out += digits[c & 15]; }
        else out += char(c);
    }
    return out + '"';
}
bool known_name(const std::string &name) {
    return name == kKernel || name == std::string(kKernel) + "_5124" ||
           name == std::string(kKernel) + "_5124_mix_aic" ||
           name == std::string(kKernel) + "_5124_mix_aiv";
}
Decision inspect(const char *kernel, uint64_t key, const void *data, size_t bytes,
                 uint32_t m, uint32_t n, uint32_t k) {
    Decision d;
    d.kernel = kernel ? kernel : ""; d.key = key; d.supplied = bytes;
    if (!known_name(d.kernel)) { d.reason = "unknown_kernel_identity"; return d; }
    if (key != 5124) { d.reason = "unknown_dispatch_key"; return d; }
    if (!data || bytes < kTilingBytes) { d.reason = "short_or_null_tiling"; return d; }
    const auto *p = static_cast<const unsigned char *>(data);
    constexpr char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < kTilingBytes; ++i) {
        d.hex += digits[p[i] >> 4]; d.hex += digits[p[i] & 15];
    }
    d.m = u32(p + 88); d.n = u32(p + 92); d.k = u32(p + 96);
    d.bm = u32(p + 100); d.bn = u32(p + 104); d.bk = u32(p + 108);
    for (size_t offset = 0; offset < 60; offset += 4) {
        if (u32(p + offset) != 1) { d.reason = "batch_fields_mismatch"; return d; }
    }
    if (u32(p + 60) != 32 || u32(p + 64) != 16) {
        d.reason = "quant_modes_mismatch"; return d;
    }
    if (u32(p + 68) != 0 || p[125] != 0) { d.reason = "unexpected_bias"; return d; }
    if (u32(p + 76) != 1 || u32(p + 80) != 128 || u32(p + 84) != 128) {
        d.reason = "scale_groups_mismatch"; return d;
    }
    if (!m || !n || !k || d.m != m || d.n != n || d.k != k) {
        d.reason = "matrix_shape_mismatch"; return d;
    }
    if (!d.bm || !d.bn || !d.bk || d.bm % 16 || d.bn % 16 || d.bk % 32) {
        d.reason = "invalid_base_tile_alignment_or_zero"; return d;
    }
    if (uint64_t(d.bm) > std::numeric_limits<uint64_t>::max() / 2 / d.bk) {
        d.reason = "l0a_extent_integer_overflow"; return d;
    }
    d.l0a = 2ULL * d.bm * d.bk;
    d.code = d.l0a > kCapacity ? 1 : 0;
    d.reason = d.code ? "fp8_ping_pong_a_exceeds_l0a" : "fp8_ping_pong_a_within_l0a";
    return d;
}
std::string json(const Decision &d, const Config &cfg) {
    std::ostringstream out;
    out << "{\"schema_version\":1,\"observed\":true,\"decision\":"
        << quote(d.code == 0 ? "within_capacity" : d.code == 1 ? "over_capacity" : "unverified")
        << ",\"reason\":" << quote(d.reason) << ",\"kernel_name\":" << quote(d.kernel)
        << ",\"dispatch_key\":" << d.key << ",\"mode\":" << cfg.mode
        << ",\"forward_allowed\":" << (cfg.mode == 2 && d.code == 0 ? "true" : "false")
        << ",\"expected_m\":" << cfg.m << ",\"expected_n\":" << cfg.n << ",\"expected_k\":" << cfg.k
        << ",\"m\":" << d.m << ",\"n\":" << d.n << ",\"k\":" << d.k
        << ",\"base_m\":" << d.bm << ",\"base_n\":" << d.bn << ",\"base_k\":" << d.bk
        << ",\"l0a_bytes\":" << d.l0a << ",\"l0a_capacity_bytes\":" << kCapacity
        << ",\"declared_tiling_bytes\":" << kTilingBytes << ",\"available_tiling_bytes\":" << d.supplied
        << ",\"args_bytes\":" << d.args_bytes << ",\"tiling_offset\":" << d.tiling_offset
        << ",\"placeholder_address_offset\":" << d.address_offset
        << ",\"tiling_hex\":" << quote(d.hex) << '}';
    return out.str();
}
void publish(const Decision &d, const Config &cfg, bool log) {
    const std::string result = json(d, cfg);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last = result;
    if (log && g_logged.insert(result).second) {
        std::fprintf(stderr, "OPERATORX_FP8_L0_GUARD %s\n", result.c_str());
        std::fflush(stderr);
    }
}
template <typename T> T next(const char *name) {
    return reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
}
} // namespace

// Modes: 0 inactive, 1 preflight (block every target), 2 enforce (forward fits).
extern "C" int operatorx_fp8_guard_configure(uint32_t m, uint32_t n, uint32_t k, int mode) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (mode < 0 || mode > 2 || (mode && (!m || !n || !k))) {
        g_config = Config{0, 0, 0, 1};
        g_last = "{\"decision\":\"unverified\",\"reason\":\"invalid_configuration\",\"observed\":false}";
        return kBlocked;
    }
    g_config = Config{m, n, k, mode};
    g_last = mode ? "{\"decision\":\"pending\",\"observed\":false}" :
                    "{\"decision\":\"inactive\",\"observed\":false}";
    return 0;
}
extern "C" const char *operatorx_fp8_guard_last_json() {
    thread_local std::string snapshot;
    std::lock_guard<std::mutex> lock(g_mutex);
    snapshot = g_last;
    return snapshot.c_str();
}
// Pure host decoder for regression tests. Does not initialize or call the NPU.
extern "C" int operatorx_fp8_guard_validate(const char *kernel, uint64_t key, const void *data,
                                          size_t bytes, uint32_t m, uint32_t n, uint32_t k) {
    Decision d = inspect(kernel, key, data, bytes, m, n, k);
    publish(d, Config{m, n, k, 1}, false);
    return d.code;
}

extern "C" int aclrtBinaryGetFunctionByEntry(void *binary, uint64_t entry, void **function) {
    using Fn = int (*)(void *, uint64_t, void **);
    static Fn real = next<Fn>("aclrtBinaryGetFunctionByEntry");
    if (!real) return kBlocked;
    const int result = real(binary, entry, function);
    if (!result && function && *function) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_keys[*function] = entry;
    }
    return result;
}

extern "C" int aclrtLaunchKernelWithHostArgs(void *function, uint32_t blocks, void *stream,
                                           void *cfg, void *args, size_t bytes,
                                           Placeholder *placeholders, size_t count) {
    using Launch = int (*)(void *, uint32_t, void *, void *, void *, size_t, Placeholder *, size_t);
    using Name = int (*)(void *, uint32_t, char *);
    static Launch real = next<Launch>("aclrtLaunchKernelWithHostArgs");
    static Name get_name = next<Name>("aclrtGetFunctionName");
    Config config;
    uint64_t key = 0;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        config = g_config;
        const auto it = g_keys.find(function);
        if (it != g_keys.end()) key = it->second;
    }
    if (!config.mode) return real ? real(function, blocks, stream, cfg, args, bytes, placeholders, count) : kBlocked;
    char buffer[2048] = {};
    Decision decision;
    if (!get_name || get_name(function, sizeof(buffer) - 1, buffer) != 0 || !buffer[0]) {
        decision.reason = "kernel_name_unavailable";
        publish(decision, config, true);
        return kBlocked;
    }
    const std::string name(buffer);
    // Preparation kernels are allowed. Any unknown matmul identity is blocked.
    if (name.find("Matmul") == std::string::npos && name.find("MatMul") == std::string::npos &&
        name.find("matmul") == std::string::npos) {
        return real ? real(function, blocks, stream, cfg, args, bytes, placeholders, count) : kBlocked;
    }
    // A name explicitly containing the compiled key can corroborate it when the
    // runtime obtained the handle before our entry hook. Never infer key 5124
    // from the unqualified base name.
    if (!key && known_name(name) && name != kKernel) key = 5124;
    decision.kernel = name; decision.key = key; decision.args_bytes = bytes;
    if (!known_name(name) || key != 5124) {
        decision.reason = "unknown_kernel_identity_or_dispatch_key";
    } else if (!args || !placeholders || !count || count > 64 || bytes > (1U << 20)) {
        decision.reason = "invalid_launch_argument_envelope";
    } else {
        size_t valid_count = 0;
        Decision last_invalid = decision;
        last_invalid.reason = "no_tiling_placeholder";
        for (size_t i = 0; i < count; ++i) {
            const auto ph = placeholders[i];
            if (ph.addrOffset > bytes || bytes - ph.addrOffset < sizeof(void *) ||
                ph.dataOffset > bytes || bytes - ph.dataOffset < kTilingBytes ||
                ph.dataOffset < ph.addrOffset + sizeof(void *)) continue;
            Decision candidate = inspect(name.c_str(), key, static_cast<unsigned char *>(args) + ph.dataOffset,
                                         bytes - ph.dataOffset, config.m, config.n, config.k);
            candidate.args_bytes = bytes; candidate.tiling_offset = ph.dataOffset;
            candidate.address_offset = ph.addrOffset;
            if (candidate.code != 2) { decision = candidate; ++valid_count; }
            else last_invalid = candidate;
        }
        if (!valid_count) decision = last_invalid;
        if (valid_count > 1) { decision.code = 2; decision.reason = "ambiguous_tiling_placeholders"; }
    }
    publish(decision, config, true);
    if (config.mode == 1 || decision.code != 0) return kBlocked;
    return real ? real(function, blocks, stream, cfg, args, bytes, placeholders, count) : kBlocked;
}
