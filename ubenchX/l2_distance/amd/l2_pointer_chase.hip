/**
 * CU-to-CU L2 / Infinity Cache latency distance measurement via pointer chasing
 * (AMD CDNA3 / CDNA4: MI300X gfx942, MI355X gfx950).
 *
 * Launches one single-wavefront workgroup per CU, pinned one-per-CU by requesting
 * the whole LDS. Lane 0 of each workgroup chases a random linked list of 128-byte
 * lines and records per-hop latency with s_memtime (shader clock). The chain is
 * sized to the per-XCD L2 (4 MB) by default; --chain-mb picks a larger chain that
 * misses L2 and sits in the Infinity Cache (the memory-side last-level cache), so the latency profile then
 * reflects XCD -> IOD -> HBM-stack distance instead of CU -> L2-channel distance.
 *
 * The mean absolute difference between each pair of CUs' per-address latency
 * profiles reveals the on-package topology: CUs sharing an XCD (and shader
 * engine) see similar profiles; XCDs on different IODs diverge.
 *
 * Outputs:
 *   stdout               cu_a,cu_b,mean_abs_diff,xcd_a,xcd_b   (N*N pairwise matrix)
 *   results/cu_info.csv  cu,xcd,se,cu_id,mean_latency           (per-CU summary)
 *   results/latency_profile.csv  sorted latency profile of CU 0
 *
 * cu is the packed hardware id (xcd << 6) | (se << 4) | cu_id, read from the
 * HW_ID and XCC_ID hardware registers.
 *
 * Ported from ubenchX/l2_distance/nvidia (NVIDIA Blackwell) to HIP.
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <map>
#include <vector>
#include <thread>
#include <algorithm>
#include <random>
#include <sys/stat.h>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHK(x) do { \
    hipError_t e = (x); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error: %s at %s:%d\n", \
                hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#ifndef NUM_CUS
#define NUM_CUS 256
#endif

#define LINE_SIZE 128
#define WAVE_SIZE 64
// Per-hop latencies are staged in LDS so the timed loop issues no global stores
// (stores share vmcnt with loads on gfx9 and would pollute the L2 under test).
// The whole LDS (160 KB on gfx950, 64 KB on gfx942) is requested per workgroup,
// which also pins exactly one workgroup per CU; lds_slots = LDS bytes / 4.

// s_getreg immediate: hwRegId[5:0] | offset[10:6] | (size-1)[15:11]
#define GETREG_IMM(id, off, sz) ((id) | ((off) << 6) | (((sz) - 1) << 11))
#define HWREG_HW_ID  4    // gfx9 HW_ID: CU_ID[11:8] SH_ID[12] SE_ID[15:13]
#define HWREG_XCC_ID 20   // gfx940+ XCC_ID: XCC_ID[3:0]

static __device__ __forceinline__ uint32_t read_hw_id() {
    return __builtin_amdgcn_s_getreg(GETREG_IMM(HWREG_HW_ID, 0, 32));
}
static __device__ __forceinline__ uint32_t read_xcc_id() {
    return __builtin_amdgcn_s_getreg(GETREG_IMM(HWREG_XCC_ID, 0, 32));
}
// Packed id: xcd<<8 | se<<5 | sh<<4 | cu  (raw hardware fields, decoded on the host)
static __device__ __forceinline__ uint32_t cu_tag() {
    uint32_t hw = read_hw_id();
    uint32_t xcc = read_xcc_id() & 0xf;
    uint32_t cu = (hw >> 8) & 0xf, sh = (hw >> 12) & 1, se = (hw >> 13) & 7;
    return (xcc << 8) | (se << 5) | (sh << 4) | cu;
}

// Untimed dependent load. sc1 alone selects device scope (SC1=1, SC0=0), which the
// CDNA3/CDNA4 ISA load-control table defines as: CU (vector L1) cache "Miss Evict",
// L2 "Coherent Cache Bypass" on parts with several L2s, last-level cache "Hit LRU".
// Locally homed memory stays coherently cached, so the load is served from the
// XCD's L2 (measured: ~212 cycles for a 4 MB chain vs ~500 for a 64 MB one), the
// CDNA analogue of CUDA's ld.global.cg.
static __device__ __forceinline__ const char* hop(const char* p) {
    asm volatile(
        "global_load_dwordx2 %[p], %[p], off sc1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : [p] "+v"(p) : : "memory");
    return p;
}

// Timed dependent load. Both s_memtime reads and the s_waitcnt live in one asm
// block so the compiler cannot move the second timestamp above the load's wait.
static __device__ __forceinline__ const char* timed_hop(const char* p, uint32_t& lat) {
    uint64_t t0, t1;
    asm volatile(
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        "s_memtime %[t0]\n\t"
        "global_load_dwordx2 %[p], %[p], off sc1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "s_memtime %[t1]\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : [p] "+v"(p), [t0] "=s"(t0), [t1] "=s"(t1)
        : : "memory");
    lat = (uint32_t)(t1 - t0);
    return p;
}

extern __shared__ uint32_t lds_lats[];

__global__ void __launch_bounds__(WAVE_SIZE)
chase_kernel(const char* const* __restrict__ starts,
             uint32_t*          __restrict__ hop_latencies,
             uint32_t*          __restrict__ cu_tags,
             int num_hops, int warm_hops, int lds_slots, int rewarm)
{
    cg::grid_group grid = cg::this_grid();
    const int bid  = blockIdx.x;
    const int lane = threadIdx.x;
    if (lane == 0) cu_tags[bid] = cu_tag();

    const char* p = starts[bid];
    // Untimed traversal: fills this XCD's L2 (or the Infinity Cache) with the
    // chain and lets clocks ramp before anyone starts timing.
    if (lane == 0)
        for (int i = 0; i < warm_hops; i++) p = hop(p);
    grid.sync();

    p = starts[bid];
    uint32_t* my_lats = hop_latencies + (size_t)bid * num_hops;
    for (int base = 0; base < num_hops; base += lds_slots) {
        const int n = min(lds_slots, num_hops - base);
        if (lane == 0) {
            for (int i = 0; i < n; i++) {
                uint32_t lat;
                p = timed_hop(p, lat);
                lds_lats[i] = lat;
            }
        }
        __syncthreads();
        // Streaming (nt) stores so the dump does not displace the chain from L2.
        for (int i = lane; i < n; i += WAVE_SIZE)
            __builtin_nontemporal_store(lds_lats[i], my_lats + base + i);
        __syncthreads();
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");   // drain the dump before timing again
        if (rewarm && lane == 0 && base + n < num_hops) {
            // L2-resident chain split over several chunks (small LDS, e.g. 64 KB on
            // gfx942): re-touch the next chunk's lines so the dump traffic cannot leave
            // cold misses in it. Skipped for chains that exceed L2, where it would only
            // hand the timed loop a few spurious L2 hits.
            const char* q = p;
            const int ahead = min(lds_slots, num_hops - base - n);
            for (int i = 0; i < ahead; i++) q = hop(q);
            if (q == nullptr) cu_tags[bid] = 0xfffffffeu;    // keep the re-warm chain live
        }
    }
    if (p == nullptr) cu_tags[bid] = 0xffffffffu;           // keep the chain live
}

struct CuInfo { uint32_t xcd, se, sh, cu; };
static CuInfo decode_tag(uint32_t t) {
    return CuInfo{ (t >> 8) & 0xf, (t >> 5) & 7, (t >> 4) & 1, t & 0xf };
}
// (xcd << 6) | (se << 4) | cu_id, the value HIP's __smid() returns (checked on MI355X).
// SH_ID is left out because the CDNA3/CDNA4 ISA define it as always zero; main() refuses
// to emit rows if the labels ever collide.
static uint32_t cu_label(const CuInfo& c) { return (c.xcd << 6) | (c.se << 4) | c.cu; }

static FILE* open_result(const char* path) {
    mkdir("results", 0755);   // ignore EEXIST; fopen reports anything else
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s for writing\n", path); exit(1); }
    return f;
}

static void flush_caches(size_t bytes) {
    uint8_t* d;
    CHK(hipMalloc(&d, bytes));
    CHK(hipMemset(d, 0xFF, bytes));
    CHK(hipDeviceSynchronize());
    CHK(hipFree(d));
}

static void build_chain(char* h_buf, char* d_buf, size_t size,
                        std::mt19937& rng, int& start_link) {
    int num_links = size / LINE_SIZE;
    std::vector<int> perm(num_links);
    for (int i = 0; i < num_links; i++) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int i = 0; i < num_links; i++) {
        int cur  = perm[i];
        int next = perm[(i + 1) % num_links];
        *(char**)(h_buf + (size_t)cur * LINE_SIZE) = d_buf + (size_t)next * LINE_SIZE;
    }
    start_link = perm[0];
}

static void usage(const char* argv0) {
    fprintf(stderr,
        "usage: %s [--chain-mb MB] [--hops N] [--passes N] [--warm-hops N] [--device D]\n"
        "  --chain-mb   chain size in MiB (default: per-XCD L2 size reported by HIP)\n"
        "  --hops       timed hops per CU (default 0 = full chain)\n"
        "  --passes     averaging passes with shuffled start offsets (default 5)\n"
        "  --warm-hops  untimed hops per CU before timing (default -1 = full chain, 0 = none)\n",
        argv0);
}

int main(int argc, char** argv) {
    double chain_mb = 0;
    int num_hops_override = 0, num_passes = 5, warm_override = -1, dev = 0;
    for (int i = 1; i < argc; i++) {
        auto need = [&](const char* flag) {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", flag); usage(argv[0]); exit(2); }
            return argv[++i];
        };
        if (!strcmp(argv[i], "--chain-mb"))      chain_mb = atof(need("--chain-mb"));
        else if (!strcmp(argv[i], "--hops"))     num_hops_override = atoi(need("--hops"));
        else if (!strcmp(argv[i], "--passes"))   num_passes = atoi(need("--passes"));
        else if (!strcmp(argv[i], "--warm-hops")) warm_override = atoi(need("--warm-hops"));
        else if (!strcmp(argv[i], "--device"))   dev = atoi(need("--device"));
        else { usage(argv[0]); return 2; }
    }

    CHK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    CHK(hipGetDeviceProperties(&prop, dev));
    int num_cus = prop.multiProcessorCount;
    if (num_cus != NUM_CUS) {
        fprintf(stderr, "ERROR: compiled for NUM_CUS=%d but device has %d CUs; rebuild with -DNUM_CUS=%d\n",
                NUM_CUS, num_cus, num_cus);
        return 1;
    }

    int l2_size = 0;
    CHK(hipDeviceGetAttribute(&l2_size, hipDeviceAttributeL2CacheSize, dev));
    int lds_bytes = 0;
    CHK(hipDeviceGetAttribute(&lds_bytes, hipDeviceAttributeMaxSharedMemoryPerBlock, dev));
    int supports_coop = 0;
    CHK(hipDeviceGetAttribute(&supports_coop, hipDeviceAttributeCooperativeLaunch, dev));
    if (!supports_coop) { fprintf(stderr, "ERROR: no cooperative launch\n"); return 1; }
    int lds_slots = lds_bytes / (int)sizeof(uint32_t);
    if (lds_slots < 1024) {
        fprintf(stderr, "ERROR: LDS per block is only %d B\n", lds_bytes);
        return 1;
    }

    size_t chain_size = chain_mb > 0 ? (size_t)(chain_mb * 1024 * 1024) : (size_t)l2_size;
    chain_size = (chain_size / LINE_SIZE) * LINE_SIZE;
    int num_links = chain_size / LINE_SIZE;
    int num_hops  = (num_hops_override > 0) ? num_hops_override : num_links;
    int warm_hops = (warm_override < 0) ? num_links : warm_override;
    int rewarm = (chain_size <= (size_t)l2_size) ? 1 : 0;
    // Flush both the per-XCD L2s and the memory-side Infinity Cache between passes.
    size_t flush_bytes = std::max<size_t>((size_t)1 << 30, 4 * chain_size);

    fprintf(stderr, "CU L2 Pointer Chase Benchmark\n");
    fprintf(stderr, "  GPU        = %s (%s)\n", prop.name, prop.gcnArchName);
    fprintf(stderr, "  CUs        = %d\n", num_cus);
    fprintf(stderr, "  L2 size    = %.1f MB (per XCD)\n", l2_size / (1024.0 * 1024.0));
    fprintf(stderr, "  LDS/block  = %d KB (%d latency slots per timed chunk%s)\n", lds_bytes / 1024, lds_slots,
            (rewarm && num_hops > lds_slots) ? ", re-warm between chunks" : "");
    fprintf(stderr, "  chain      = %.1f MB (%d links)%s\n", chain_size / (1024.0 * 1024.0), num_links,
            chain_size <= (size_t)l2_size ? "  [L2-resident]" : "  [exceeds L2: Infinity Cache / HBM]");
    fprintf(stderr, "  num_hops   = %d (%s)\n", num_hops, num_hops == num_links ? "full chain" : "partial");
    fprintf(stderr, "  warm_hops  = %d\n", warm_hops);
    fprintf(stderr, "  num_passes = %d\n", num_passes);
    fprintf(stderr, "  clock      = %d MHz (shader clock; latencies are s_memtime cycles)\n", prop.clockRate / 1000);

    fprintf(stderr, "\n--- Building chain ---\n");
    char* h_buf = new char[chain_size];
    char* d_data;
    CHK(hipMalloc(&d_data, chain_size));

    std::mt19937 rng(42);
    int start_link;
    build_chain(h_buf, d_data, chain_size, rng, start_link);
    CHK(hipMemcpy(d_data, h_buf, chain_size, hipMemcpyHostToDevice));

    std::vector<int> chain_order(num_links);
    {
        int cur = start_link;
        for (int i = 0; i < num_links; i++) {
            chain_order[i] = cur;
            char* next_dev_ptr = *(char**)(h_buf + (size_t)cur * LINE_SIZE);
            cur = (next_dev_ptr - d_data) / LINE_SIZE;
        }
    }

    int cu_spacing = num_links / NUM_CUS;
    fprintf(stderr, "  CU spacing = %d links (%d KB)\n", cu_spacing, cu_spacing * LINE_SIZE / 1024);

    const char** d_starts;
    CHK(hipMalloc(&d_starts, NUM_CUS * sizeof(const char*)));

    size_t results_size = (size_t)NUM_CUS * num_hops * sizeof(uint32_t);
    fprintf(stderr, "  results buf = %.1f MB\n", results_size / (1024.0 * 1024.0));
    uint32_t* d_hop_lats;
    CHK(hipMalloc(&d_hop_lats, results_size));
    uint32_t* d_cu_tags;
    CHK(hipMalloc(&d_cu_tags, NUM_CUS * sizeof(uint32_t)));

    CHK(hipFuncSetAttribute((const void*)chase_kernel,
                            hipFuncAttributeMaxDynamicSharedMemorySize, lds_bytes));

    float*    accum  = new float[(size_t)NUM_CUS * num_links]();
    uint16_t* visits = new uint16_t[(size_t)NUM_CUS * num_links]();
    uint32_t h_cu_tags[NUM_CUS];
    uint32_t* h_hop_lats = new uint32_t[(size_t)NUM_CUS * num_hops];
    std::vector<const char*> h_starts(NUM_CUS);
    std::vector<std::vector<int>> hop_links(NUM_CUS);

    std::vector<int> start_perm(NUM_CUS);
    for (int i = 0; i < NUM_CUS; i++) start_perm[i] = i;

    for (int pass = 0; pass < num_passes; pass++) {
        fprintf(stderr, "\n--- Pass %d/%d ---\n", pass + 1, num_passes);
        if (pass > 0) std::shuffle(start_perm.begin(), start_perm.end(), rng);

        for (int i = 0; i < NUM_CUS; i++) {
            int start_pos = (start_perm[i] * cu_spacing) % num_links;
            hop_links[i].resize(num_hops);
            for (int hop = 0; hop < num_hops; hop++)
                hop_links[i][hop] = chain_order[(start_pos + hop) % num_links];
            h_starts[i] = d_data + (size_t)hop_links[i][0] * LINE_SIZE;
        }
        CHK(hipMemcpy(d_starts, h_starts.data(), NUM_CUS * sizeof(const char*), hipMemcpyHostToDevice));

        flush_caches(flush_bytes);

        void* args[] = { &d_starts, &d_hop_lats, &d_cu_tags, &num_hops, &warm_hops, &lds_slots, &rewarm };
        CHK(hipLaunchCooperativeKernel((const void*)chase_kernel, dim3(NUM_CUS), dim3(WAVE_SIZE),
                                       args, lds_bytes, 0));
        CHK(hipDeviceSynchronize());

        CHK(hipMemcpy(h_cu_tags, d_cu_tags, NUM_CUS * sizeof(uint32_t), hipMemcpyDeviceToHost));
        CHK(hipMemcpy(h_hop_lats, d_hop_lats, results_size, hipMemcpyDeviceToHost));

        std::set<uint32_t> unique(h_cu_tags, h_cu_tags + NUM_CUS);
        fprintf(stderr, "  Unique CUs: %zu / %d\n", unique.size(), NUM_CUS);
        if (unique.size() != (size_t)NUM_CUS) {
            fprintf(stderr, "ERROR: workgroups were not pinned one-per-CU; aborting\n");
            return 1;
        }

        double pass_sum = 0;
        for (int b = 0; b < NUM_CUS; b++) {
            float*    row = accum  + (size_t)b * num_links;
            uint16_t* cnt = visits + (size_t)b * num_links;
            const uint32_t* lats = h_hop_lats + (size_t)b * num_hops;
            for (int hop = 0; hop < num_hops; hop++) {
                row[hop_links[b][hop]] += lats[hop];
                cnt[hop_links[b][hop]] += 1;
                pass_sum += lats[hop];
            }
        }
        {
            // Latency percentiles over a strided sample of this pass's hops.
            std::vector<uint32_t> sample;
            size_t total = (size_t)NUM_CUS * num_hops, stride = std::max<size_t>(1, total / 200000);
            for (size_t i = 0; i < total; i += stride) sample.push_back(h_hop_lats[i]);
            std::sort(sample.begin(), sample.end());
            auto pct = [&](double q) { return sample[std::min(sample.size() - 1, (size_t)(q * sample.size()))]; };
            fprintf(stderr, "  hop latency this pass: mean %.1f  p1 %u  p50 %u  p90 %u  p99 %u  max %u cycles\n",
                    pass_sum / (double)total, pct(0.01), pct(0.50), pct(0.90), pct(0.99), sample.back());
        }
    }
    delete[] h_hop_lats;
    delete[] h_buf;

    // Average each (CU, link) over the passes that visited it; links a CU never
    // visited (partial --hops) are excluded from every statistic below.
    std::vector<char> visited((size_t)NUM_CUS * num_links, 0);
    for (size_t i = 0; i < (size_t)NUM_CUS * num_links; i++) {
        if (visits[i]) { accum[i] /= visits[i]; visited[i] = 1; }
    }
    delete[] visits;

    fprintf(stderr, "\n--- Computing results ---\n");
    std::vector<CuInfo> info(NUM_CUS);
    std::vector<uint32_t> label(NUM_CUS);
    for (int b = 0; b < NUM_CUS; b++) { info[b] = decode_tag(h_cu_tags[b]); label[b] = cu_label(info[b]); }
    {
        // The packed label drops SH_ID, which the CDNA3/CDNA4 ISA documents as always zero;
        // refuse to emit ambiguous rows if some part ever populates it.
        std::set<uint32_t> unique_labels(label.begin(), label.end());
        if (unique_labels.size() != (size_t)NUM_CUS) {
            fprintf(stderr, "ERROR: packed CU labels collide (%zu unique of %d); SH_ID is non-zero on this part\n",
                    unique_labels.size(), NUM_CUS);
            return 1;
        }
    }

    std::vector<double> cu_mean(NUM_CUS);
    for (int i = 0; i < NUM_CUS; i++) {
        double sum = 0; long n = 0;
        for (int j = 0; j < num_links; j++)
            if (visited[(size_t)i * num_links + j]) { sum += accum[(size_t)i * num_links + j]; n++; }
        cu_mean[i] = n ? sum / n : 0;
    }

    std::map<uint32_t, std::vector<double>> per_xcd;
    for (int b = 0; b < NUM_CUS; b++) per_xcd[info[b].xcd].push_back(cu_mean[b]);
    for (auto& [x, v] : per_xcd) {
        double s = 0; for (double m : v) s += m;
        fprintf(stderr, "  XCD %u: %zu CUs, mean latency %.1f cycles\n", x, v.size(), s / v.size());
    }

    FILE* f_cu = open_result("results/cu_info.csv");
    fprintf(f_cu, "cu,xcd,se,cu_id,mean_latency\n");
    for (int b = 0; b < NUM_CUS; b++)
        fprintf(f_cu, "%u,%u,%u,%u,%.1f\n", label[b], info[b].xcd, info[b].se, info[b].cu, cu_mean[b]);
    fclose(f_cu);
    fprintf(stderr, "  Wrote results/cu_info.csv\n");

    {
        const float* row = accum;
        std::vector<double> sorted_lats;
        for (int j = 0; j < num_links; j++) if (visited[j]) sorted_lats.push_back(row[j]);
        std::sort(sorted_lats.begin(), sorted_lats.end());
        FILE* f_prof = open_result("results/latency_profile.csv");
        fprintf(f_prof, "cu,xcd,rank,latency\n");
        for (size_t j = 0; j < sorted_lats.size(); j++)
            fprintf(f_prof, "%u,%u,%zu,%.1f\n", label[0], info[0].xcd, j, sorted_lats[j]);
        fclose(f_prof);
        fprintf(stderr, "  Wrote results/latency_profile.csv (%zu rows)\n", sorted_lats.size());
    }

    // Pairwise mean |diff| over addresses both CUs visited; rows split across host threads.
    std::vector<double> mad((size_t)NUM_CUS * NUM_CUS, 0.0);
    unsigned nthreads = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < nthreads; t++) {
        workers.emplace_back([&, t]() {
            for (int a = t; a < NUM_CUS; a += nthreads) {
                const float* row_a = accum + (size_t)a * num_links;
                const char*  vis_a = visited.data() + (size_t)a * num_links;
                for (int b = 0; b < NUM_CUS; b++) {
                    const float* row_b = accum + (size_t)b * num_links;
                    const char*  vis_b = visited.data() + (size_t)b * num_links;
                    double abs_sum = 0; long n = 0;
                    for (int j = 0; j < num_links; j++) {
                        if (!(vis_a[j] && vis_b[j])) continue;
                        double diff = (double)row_a[j] - (double)row_b[j];
                        abs_sum += diff >= 0 ? diff : -diff;
                        n++;
                    }
                    mad[(size_t)a * NUM_CUS + b] = n ? abs_sum / n : 0;
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    printf("cu_a,cu_b,mean_abs_diff,xcd_a,xcd_b\n");
    for (int a = 0; a < NUM_CUS; a++)
        for (int b = 0; b < NUM_CUS; b++)
            printf("%u,%u,%.1f,%u,%u\n", label[a], label[b], mad[(size_t)a * NUM_CUS + b],
                   info[a].xcd, info[b].xcd);

    delete[] accum;
    CHK(hipFree(d_data));
    CHK(hipFree(d_hop_lats));
    CHK(hipFree(d_cu_tags));
    CHK(hipFree(d_starts));
    return 0;
}
