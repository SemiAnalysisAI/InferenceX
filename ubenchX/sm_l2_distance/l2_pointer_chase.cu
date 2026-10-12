/**
 * SM-to-SM L2 cache latency distance measurement via pointer chasing.
 *
 * Launches one warp per SM; each warp chases a random linked list through the
 * L2 cache and records per-hop latencies. The mean absolute difference between
 * each pair of SMs' latency profiles reveals how much their L2 access patterns
 * diverge — SMs sharing a GPC (and die) see similar profiles, SMs on different
 * dies diverge further.
 *
 * Outputs:
 *   stdout  — sm_a,sm_b,mean_abs_diff,gpc_a,gpc_b  (N*N pairwise matrix)
 *   results/sm_info.csv — sm,gpc,mean_latency (per-SM summary)
 *
 * Ported from SemiAnalysisAI/microbench-blackwell (sm_l2_distance/).
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", \
                cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#ifndef NUM_SMS
#define NUM_SMS 148
#endif

#define LINE_SIZE 128

static __device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ void __launch_bounds__(1)
chase_kernel(const char** __restrict__ starts,
             uint32_t*    __restrict__ hop_latencies,
             uint32_t*    __restrict__ sm_ids,
             int num_hops)
{
    cg::grid_group grid = cg::this_grid();
    int bid = blockIdx.x;
    sm_ids[bid] = get_smid();

    const char* p = starts[bid];

    for (int i = 0; i < 64; i++) {
        asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p) :: "memory");
    }
    grid.sync();

    p = starts[bid];
    uint32_t* my_lats = hop_latencies + (size_t)bid * num_hops;

    for (int hop = 0; hop < num_hops; hop++) {
        uint64_t t0 = clock64();
        asm volatile("ld.global.cg.u64 %0, [%0];" : "+l"(p) :: "memory");
        asm volatile("mov.u64 %0, %1;" : "=l"(p) : "l"(p));
        uint64_t t1 = clock64();
        my_lats[hop] = (uint32_t)(t1 - t0);
    }

    if (p == nullptr)
        asm volatile("st.global.u64 [%0], %0;" :: "l"(starts[bid]) : "memory");
}

extern __shared__ char gpc_smem[];

__global__ void gpc_query_kernel(uint32_t* out) {
    if (threadIdx.x == 0) gpc_smem[0] = 1;
    cg::cluster_group cluster = cg::this_cluster();
    if (threadIdx.x == 0) {
        out[blockIdx.x * 2 + 0] = get_smid();
        out[blockIdx.x * 2 + 1] = blockIdx.x / cluster.num_blocks();
    }
    cluster.sync();
}

struct UF {
    std::map<uint32_t, uint32_t> p;
    uint32_t find(uint32_t x) {
        if (p.find(x) == p.end()) p[x] = x;
        return p[x] == x ? x : p[x] = find(p[x]);
    }
    void unite(uint32_t a, uint32_t b) { p[find(a)] = find(b); }
};

static std::map<uint32_t, int> discover_gpcs(int num_sms) {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    int max_smem = prop.sharedMemPerBlockOptin;
    CHK(cudaFuncSetAttribute(gpc_query_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));
    CHK(cudaFuncSetAttribute(gpc_query_kernel,
            cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    int max_cluster = 16;
    {
        cudaLaunchConfig_t cfg = {};
        cfg.blockDim = dim3(32);
        cfg.gridDim  = dim3(128);
        cfg.dynamicSmemBytes = max_smem;
        if (cudaOccupancyMaxPotentialClusterSize(
                &max_cluster, (void*)gpc_query_kernel, &cfg) != cudaSuccess)
            max_cluster = 16;
    }
    if (max_cluster < 2) max_cluster = 16;

    UF uf;
    std::set<uint32_t> all_sms;
    for (int csz = 2; csz <= max_cluster; csz++) {
        int nblocks = ((num_sms + csz - 1) / csz) * csz;
        size_t out_sz = nblocks * 2 * sizeof(uint32_t);
        uint32_t* d_out;
        CHK(cudaMalloc(&d_out, out_sz));
        CHK(cudaMemset(d_out, 0xff, out_sz));
        cudaLaunchConfig_t config = {};
        config.gridDim = dim3(nblocks);
        config.blockDim = dim3(32);
        config.dynamicSmemBytes = max_smem;
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {(unsigned)csz, 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;
        CHK(cudaLaunchKernelEx(&config, gpc_query_kernel, d_out));
        CHK(cudaDeviceSynchronize());
        std::vector<uint32_t> h_out(nblocks * 2);
        CHK(cudaMemcpy(h_out.data(), d_out, out_sz, cudaMemcpyDeviceToHost));
        CHK(cudaFree(d_out));
        std::map<uint32_t, std::vector<uint32_t>> clusters;
        for (int b = 0; b < nblocks; b++) {
            uint32_t smid = h_out[b*2], cid = h_out[b*2+1];
            if (smid != 0xffffffff) {
                clusters[cid].push_back(smid);
                all_sms.insert(smid);
            }
        }
        for (auto& [cid, sms] : clusters)
            for (size_t i = 1; i < sms.size(); i++)
                uf.unite(sms[0], sms[i]);
    }
    if ((int)all_sms.size() != num_sms)
        fprintf(stderr, "WARNING: found %zu SMs, expected %d\n", all_sms.size(), num_sms);

    std::map<uint32_t, std::vector<uint32_t>> gpc_groups;
    for (uint32_t s : all_sms) gpc_groups[uf.find(s)].push_back(s);
    std::map<uint32_t, int> sm_to_gpc;
    int gpc_id = 0;
    for (auto& [rep, sms] : gpc_groups) {
        std::sort(sms.begin(), sms.end());
        fprintf(stderr, "  GPC %2d: %2zu SMs [", gpc_id, sms.size());
        for (size_t i = 0; i < sms.size(); i++) {
            fprintf(stderr, "%s%u", i?",":"", sms[i]);
            sm_to_gpc[sms[i]] = gpc_id;
        }
        fprintf(stderr, "]\n");
        gpc_id++;
    }
    return sm_to_gpc;
}

static void flush_l2(int dev) {
    int l2_size = 0;
    cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, dev);
    uint8_t* d;
    CHK(cudaMalloc(&d, l2_size));
    CHK(cudaMemset(d, 0xFF, l2_size));
    CHK(cudaDeviceSynchronize());
    CHK(cudaFree(d));
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
        *(char**)(h_buf + cur * LINE_SIZE) = d_buf + next * LINE_SIZE;
    }
    start_link = perm[0];
}

int main(int argc, char** argv) {
    int num_hops_override = (argc > 1) ? atoi(argv[1]) : 0;
    int num_passes = (argc > 2) ? atoi(argv[2]) : 5;

    int dev = 0;
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, dev));
    int num_sms = prop.multiProcessorCount;

    int l2_size = 0;
    cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, dev);
    size_t chain_size = ((size_t)l2_size / LINE_SIZE) * LINE_SIZE;
    int num_links = chain_size / LINE_SIZE;

    int supports_coop = 0;
    cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch, dev);
    if (!supports_coop) { fprintf(stderr, "ERROR: no cooperative launch\n"); return 1; }

    int num_hops = (num_hops_override > 0) ? num_hops_override : num_links;

    fprintf(stderr, "L2 Pointer Chase Benchmark\n");
    fprintf(stderr, "  GPU        = %s\n", prop.name);
    fprintf(stderr, "  SMs        = %d\n", num_sms);
    fprintf(stderr, "  L2 size    = %.1f MB\n", l2_size / (1024.0 * 1024.0));
    fprintf(stderr, "  chain      = %.1f MB (%d links)\n",
            chain_size / (1024.0 * 1024.0), num_links);
    fprintf(stderr, "  num_hops   = %d (%s)\n", num_hops,
            num_hops == num_links ? "full chain" : "partial");
    fprintf(stderr, "  num_passes = %d\n", num_passes);

    fprintf(stderr, "\n--- GPC Discovery ---\n");
    auto sm_to_gpc = discover_gpcs(num_sms);

    fprintf(stderr, "\n--- Building chain ---\n");
    char* h_buf = new char[chain_size];
    char* d_data;
    CHK(cudaMalloc(&d_data, chain_size));

    std::mt19937 rng(42);
    int start_link;
    build_chain(h_buf, d_data, chain_size, rng, start_link);
    CHK(cudaMemcpy(d_data, h_buf, chain_size, cudaMemcpyHostToDevice));

    std::vector<int> chain_order(num_links);
    {
        int cur = start_link;
        for (int i = 0; i < num_links; i++) {
            chain_order[i] = cur;
            char* next_dev_ptr = *(char**)(h_buf + cur * LINE_SIZE);
            cur = (next_dev_ptr - d_data) / LINE_SIZE;
        }
    }

    int sm_spacing = num_links / NUM_SMS;
    fprintf(stderr, "  SM spacing = %d links (%d KB)\n",
            sm_spacing, sm_spacing * LINE_SIZE / 1024);

    const char** d_starts;
    CHK(cudaMalloc(&d_starts, NUM_SMS * sizeof(const char*)));

    size_t results_size = (size_t)NUM_SMS * num_hops * sizeof(uint32_t);
    fprintf(stderr, "  results buf = %.1f MB\n", results_size / (1024.0 * 1024.0));

    uint32_t* d_hop_lats;
    CHK(cudaMalloc(&d_hop_lats, results_size));

    uint32_t* d_sm_ids;
    CHK(cudaMalloc(&d_sm_ids, NUM_SMS * sizeof(uint32_t)));

    double* accum = new double[(size_t)NUM_SMS * num_links]();

    uint32_t h_sm_ids[NUM_SMS];
    uint32_t* h_hop_lats = new uint32_t[(size_t)NUM_SMS * num_hops];
    std::vector<const char*> h_starts(NUM_SMS);
    std::vector<std::vector<size_t>> hop_addrs(NUM_SMS);

    std::vector<int> start_perm(NUM_SMS);
    for (int i = 0; i < NUM_SMS; i++) start_perm[i] = i;

    for (int pass = 0; pass < num_passes; pass++) {
        fprintf(stderr, "\n--- Pass %d/%d ---\n", pass + 1, num_passes);

        if (pass > 0)
            std::shuffle(start_perm.begin(), start_perm.end(), rng);

        for (int i = 0; i < NUM_SMS; i++) {
            int start_pos = (start_perm[i] * sm_spacing) % num_links;
            int link = chain_order[start_pos];
            h_starts[i] = d_data + link * LINE_SIZE;

            hop_addrs[i].resize(num_hops);
            int cur = link;
            for (int hop = 0; hop < num_hops; hop++) {
                hop_addrs[i][hop] = (size_t)cur * LINE_SIZE;
                char* next_dev_ptr = *(char**)(h_buf + cur * LINE_SIZE);
                cur = (next_dev_ptr - d_data) / LINE_SIZE;
            }
        }

        CHK(cudaMemcpy(d_starts, h_starts.data(),
                       NUM_SMS * sizeof(const char*), cudaMemcpyHostToDevice));

        flush_l2(dev);

        void* args[] = { &d_starts, &d_hop_lats, &d_sm_ids, &num_hops };
        CHK(cudaLaunchCooperativeKernel(
            (void*)chase_kernel, dim3(NUM_SMS), dim3(1), args));
        CHK(cudaDeviceSynchronize());

        CHK(cudaMemcpy(h_sm_ids, d_sm_ids, NUM_SMS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(h_hop_lats, d_hop_lats, results_size,
                       cudaMemcpyDeviceToHost));

        if (pass == 0) {
            std::set<uint32_t> unique(h_sm_ids, h_sm_ids + NUM_SMS);
            fprintf(stderr, "  Unique SMs: %zu / %d\n", unique.size(), NUM_SMS);
        }

        for (int b = 0; b < NUM_SMS; b++) {
            for (int hop = 0; hop < num_hops; hop++) {
                int link_idx = hop_addrs[b][hop] / LINE_SIZE;
                accum[(size_t)b * num_links + link_idx] += h_hop_lats[b * num_hops + hop];
            }
        }
    }
    delete[] h_hop_lats;
    delete[] h_buf;

    double inv_passes = 1.0 / num_passes;
    for (size_t i = 0; i < (size_t)NUM_SMS * num_links; i++)
        accum[i] *= inv_passes;

    fprintf(stderr, "\n--- Computing results ---\n");
    std::vector<double> sm_mean(NUM_SMS);
    for (int i = 0; i < NUM_SMS; i++) {
        double sum = 0;
        for (int j = 0; j < num_links; j++)
            sum += accum[(size_t)i * num_links + j];
        sm_mean[i] = sum / num_links;
    }

    FILE* f_sm = fopen("results/sm_info.csv", "w");
    fprintf(f_sm, "sm,gpc,mean_latency\n");
    for (int b = 0; b < NUM_SMS; b++) {
        uint32_t smid = h_sm_ids[b];
        int gpc = sm_to_gpc.count(smid) ? sm_to_gpc[smid] : -1;
        fprintf(f_sm, "%u,%d,%.1f\n", smid, gpc, sm_mean[b]);
    }
    fclose(f_sm);
    fprintf(stderr, "  Wrote results/sm_info.csv\n");

    {
        uint32_t smid = h_sm_ids[0];
        int gpc = sm_to_gpc.count(smid) ? sm_to_gpc[smid] : -1;
        fprintf(stderr, "  Writing latency profile for SM %u (GPC %d)...\n", smid, gpc);

        const double* row = accum;
        std::vector<double> sorted_lats(num_links);
        for (int j = 0; j < num_links; j++) sorted_lats[j] = row[j];
        std::sort(sorted_lats.begin(), sorted_lats.end());

        FILE* f_prof = fopen("results/latency_profile.csv", "w");
        fprintf(f_prof, "sm,gpc,rank,latency\n");
        for (int j = 0; j < num_links; j++) {
            fprintf(f_prof, "%u,%d,%d,%.1f\n", smid, gpc, j, sorted_lats[j]);
        }
        fclose(f_prof);
        fprintf(stderr, "  Wrote results/latency_profile.csv (%d rows)\n", num_links);
    }

    printf("sm_a,sm_b,mean_abs_diff,gpc_a,gpc_b\n");
    for (int a = 0; a < NUM_SMS; a++) {
        uint32_t sm_a = h_sm_ids[a];
        int gpc_a = sm_to_gpc.count(sm_a) ? sm_to_gpc[sm_a] : -1;
        for (int b = 0; b < NUM_SMS; b++) {
            uint32_t sm_b = h_sm_ids[b];
            int gpc_b = sm_to_gpc.count(sm_b) ? sm_to_gpc[sm_b] : -1;

            double abs_sum = 0;
            const double* row_a = accum + (size_t)a * num_links;
            const double* row_b = accum + (size_t)b * num_links;
            for (int j = 0; j < num_links; j++) {
                double diff = row_a[j] - row_b[j];
                abs_sum += (diff >= 0) ? diff : -diff;
            }
            double mad = abs_sum / num_links;

            printf("%u,%u,%.1f,%d,%d\n", sm_a, sm_b, mad, gpc_a, gpc_b);
        }
    }

    delete[] accum;
    cudaFree(d_data);
    cudaFree(d_hop_lats);
    cudaFree(d_sm_ids);
    cudaFree(d_starts);
    return 0;
}
