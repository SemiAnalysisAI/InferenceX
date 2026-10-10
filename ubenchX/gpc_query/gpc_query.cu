/*
 * [100% WRITTEN BY CLAUDE CODE]
 * gpc_query.cu - Query GPC/TPC configuration via cluster scheduling
 * Output: [tpc0, tpc1, ...] sorted descending, or error on stderr
 *
 * Ported from SemiAnalysisAI/microbench-blackwell (tools/gpc_query.cu).
 */

#include <cstdio>
#include <cstdlib>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__device__ __forceinline__ uint32_t get_smid() {
    uint32_t r; asm("mov.u32 %0, %%smid;" : "=r"(r)); return r;
}

extern __shared__ char smem[];

__global__ void gpc_query_kernel(uint32_t *out) {
    if (threadIdx.x == 0) smem[0] = 1;
    
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

int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    
    if (prop.major < 9) {
        fprintf(stderr, "error: requires sm_90+\n");
        return 1;
    }

    int num_sms = prop.multiProcessorCount;
    int max_smem = prop.sharedMemPerBlockOptin;
    
    CHK(cudaFuncSetAttribute(gpc_query_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));
    CHK(cudaFuncSetAttribute(gpc_query_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

    int max_cluster = 16;
    {
        cudaLaunchConfig_t cfg = {};
        cfg.blockDim = dim3(32, 1, 1);
        cfg.gridDim = dim3(128, 1, 1);
        cfg.dynamicSmemBytes = max_smem;
        if (cudaOccupancyMaxPotentialClusterSize(&max_cluster, (void*)gpc_query_kernel, &cfg) != cudaSuccess)
            max_cluster = 16;
    }
    if (max_cluster < 2) max_cluster = 16;

    UF uf;
    std::set<uint32_t> all_sms;

    for (int csz = 2; csz <= max_cluster; csz++) {
        int nblocks = ((num_sms + csz - 1) / csz) * csz;
        size_t out_sz = nblocks * 2 * sizeof(uint32_t);
        
        uint32_t *d_out;
        CHK(cudaMalloc(&d_out, out_sz));
        CHK(cudaMemset(d_out, 0xff, out_sz));
        
        cudaLaunchConfig_t config = {};
        config.gridDim = dim3(nblocks, 1, 1);
        config.blockDim = dim3(32, 1, 1);
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
            uint32_t smid = h_out[b * 2 + 0];
            uint32_t cid = h_out[b * 2 + 1];
            if (smid != 0xffffffff) {
                clusters[cid].push_back(smid);
                all_sms.insert(smid);
            }
        }
        
        for (auto &[cid, sms] : clusters)
            for (size_t i = 1; i < sms.size(); i++)
                uf.unite(sms[0], sms[i]);
    }

    if (all_sms.size() != (size_t)num_sms) {
        fprintf(stderr, "error: detected %zu SMs, expected %d\n", all_sms.size(), num_sms);
        return 1;
    }

    std::map<uint32_t, std::vector<uint32_t>> gpcs;
    for (uint32_t s : all_sms)
        gpcs[uf.find(s)].push_back(s);

    std::vector<int> tpcs;
    for (auto &[_, sms] : gpcs)
        tpcs.push_back(sms.size() / 2);
    std::sort(tpcs.rbegin(), tpcs.rend());

    printf("[");
    for (size_t i = 0; i < tpcs.size(); i++)
        printf("%s%d", i ? ", " : "", tpcs[i]);
    printf("]\n");

    return 0;
}
