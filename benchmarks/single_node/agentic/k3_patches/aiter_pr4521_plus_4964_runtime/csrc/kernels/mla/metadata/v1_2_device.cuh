// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "mla_metadata.h"
#include "v1_comm.cuh"

template <int32_t kPackedQoLenPerWg_,
          bool kQoSplits_,
          int32_t kUniSeqlenQo_,
          bool kLdsBatchInfo_,
          bool kIsSparse_ = false>
struct MlaMetadataV12Traits
{
    static constexpr int32_t kPackedQoLenPerWg      = kPackedQoLenPerWg_;
    static constexpr int32_t kPackedQoLenPerWg_log2 = __builtin_ctz(kPackedQoLenPerWg);
    static constexpr bool kQoSplits                 = kQoSplits_;
    // <= -1: read from seqlens_qo_indptr
    // ==  0: read from MlaMetadataV1KernelParameter::uni_seqlen_qo
    // >=  1: read from MlaMetadataV12Traits::kUniSeqlenQo
    static constexpr int32_t kUniSeqlenQo  = kUniSeqlenQo_;
    static constexpr int32_t kIsSparse     = kIsSparse_;
    static constexpr int32_t kLdsBatchInfo = kLdsBatchInfo_;
};

static constexpr int32_t MLA_V12_FILL_WARPS = 8;

template <typename Traits>
__device__ __forceinline__ int32_t mla_v12_num_qo_tiles(const MlaMetadataV1KernelParameter& params,
                                                        QoState<Traits>& qo_state,
                                                        const int32_t batch_idx)
{
    if constexpr(Traits::kQoSplits)
    {
        const int32_t seqlen_qo = qo_state.get_seqlen(batch_idx);
        if(params.num_heads * 2 > Traits::kPackedQoLenPerWg)
        {
            return seqlen_qo;
        }
        const int32_t packed_qo_len = seqlen_qo * params.num_heads;
        return integer_divide_ceil_power2(
            packed_qo_len, Traits::kPackedQoLenPerWg, Traits::kPackedQoLenPerWg_log2);
    }
    else
    {
        return 1;
    }
}

template <typename Traits>
__device__ __forceinline__ int32_t
mla_v12_compute_sum_blocks(const MlaMetadataV1KernelParameter& params,
                           QoState<Traits>& qo_state,
                           int32_t* p_lds_seqlens_qo,
                           int32_t* p_lds_seqlens_kv,
                           const int32_t ori_seqlen_qo,
                           const int32_t num_batches,
                           const int32_t lane_idx)
{
    int32_t sum_blocks = 0;
    for(int32_t bid = lane_idx; bid < num_batches; bid += opus::get_warp_size())
    {
        const int32_t bid_ori = Traits::kIsSparse ? (bid / ori_seqlen_qo / params.qk_batch_ratio)
                                                  : (bid / params.qk_batch_ratio);
        const int32_t kv_end  = params.p_seqlens_kv_indptr[bid_ori + 1];
        const int32_t seqlen_kv =
            Traits::kIsSparse ? min(kv_end - params.p_seqlens_kv_indptr[bid_ori], params.topk)
                              : (kv_end - params.p_seqlens_kv_indptr[bid_ori]);

        if constexpr(Traits::kLdsBatchInfo)
        {
            p_lds_seqlens_kv[bid] = seqlen_kv;
        }

        const int32_t num_blocks = integer_divide_ceil_power2(
            seqlen_kv, params.kv_granularity, params.kv_granularity_log2);
        const int32_t num_qo_tiles = mla_v12_num_qo_tiles<Traits>(params, qo_state, bid);
        sum_blocks += (num_blocks + params.fixed_over_head_num_blocks) * num_qo_tiles;

        if constexpr(QoState<Traits>::is_unique() == false)
        {
            p_lds_seqlens_qo[bid] =
                params.p_seqlens_qo_indptr[bid_ori + 1] - params.p_seqlens_qo_indptr[bid_ori];
        }
    }

    return aiter::warpReduce<aiter::AddFunctor, decltype(sum_blocks), opus::get_warp_size()>(
        sum_blocks);
}

// Parallel planner: Phase 1 (warp 0) runs an O(num_batches) scan
// recording each batch's start CU / remainder / prefix counts;
// phase 2 fills every batch's fragments in parallel (one warp per batch).
template <typename Traits>
__launch_bounds__(opus::get_warp_size() * MLA_V12_FILL_WARPS, 1) __global__
    void kn_get_mla_metadata_v1_2_parallel(MlaMetadataV1KernelParameter params)
{
    using QoState = QoState<Traits>;

    const int32_t num_batches   = params.num_batches;
    const int32_t ori_seqlen_qo = params.ori_seqlen_qo;

    extern __shared__ uint8_t p_smem[];
    int32_t* p_lds_seqlens_qo = reinterpret_cast<int32_t*>(p_smem);
    int32_t* p_lds_seqlens_kv = p_lds_seqlens_qo + (QoState::is_unique() ? 0 : num_batches);
    int32_t* p_lds_after      = p_lds_seqlens_kv + (Traits::kLdsBatchInfo ? num_batches : 0);

    // Scalars [payload, num_works, last_reduce_indptr] followed by five per-batch
    // prefix arrays produced by the phase-1 scan.
    int32_t* p_lds_scalars        = p_lds_after;
    int32_t* p_lds_start_cu       = p_lds_scalars + 3;
    int32_t* p_lds_remain_payload = p_lds_start_cu + num_batches;
    int32_t* p_lds_works_before   = p_lds_remain_payload + num_batches;
    int32_t* p_lds_reduce_before  = p_lds_works_before + num_batches;
    int32_t* p_lds_partial_before = p_lds_reduce_before + num_batches;

    QoState qo_state(
        params.uni_seqlen_qo, ori_seqlen_qo, p_lds_seqlens_qo, params.p_seqlens_qo_indptr);

    MlaWorkInfo* p_work_info_set = reinterpret_cast<MlaWorkInfo*>(params.p_work_info_set_raw);

    const int32_t tid       = threadIdx.x;
    const int32_t lane_idx  = opus::lane_id();
    const int32_t warp_id   = tid / opus::get_warp_size();
    const int32_t num_warps = blockDim.x / opus::get_warp_size();
    const int32_t overhead  = params.fixed_over_head_num_blocks;
    const int32_t kv_gran   = params.kv_granularity;

    // Phase 1 (warp 0): closed-form scan, no stores
    if(warp_id == 0)
    {
        const int32_t sum_blocks = mla_v12_compute_sum_blocks<Traits>(params,
                                                                      qo_state,
                                                                      p_lds_seqlens_qo,
                                                                      p_lds_seqlens_kv,
                                                                      ori_seqlen_qo,
                                                                      num_batches,
                                                                      lane_idx);

        const int32_t payload       = integer_divide_ceil(sum_blocks, params.num_splits) + overhead;
        const int32_t blocks_per_cu = payload - overhead;

        if(lane_idx == 0)
        {
            params.p_reduce_indptr[0] = 0;
            params.p_work_indptr[0]   = 0;
            params.p_work_metadata_ptrs[0] =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.p_work_indptr));
            params.p_work_metadata_ptrs[1] =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p_work_info_set));
            p_lds_scalars[0] = payload;

            int32_t curr_cu            = 0;
            int32_t remain_payload     = payload;
            int32_t num_works          = 0;
            int32_t last_reduce_indptr = 0;
            int32_t partial_idx        = 0;
            for(int32_t bid = 0; bid < num_batches; ++bid)
            {
                p_lds_start_cu[bid]       = curr_cu;
                p_lds_remain_payload[bid] = remain_payload;
                p_lds_works_before[bid]   = num_works;
                p_lds_reduce_before[bid]  = last_reduce_indptr;
                p_lds_partial_before[bid] = partial_idx;

                const int32_t seqlen_kv =
                    Traits::kLdsBatchInfo
                        ? p_lds_seqlens_kv[bid]
                        : (params.p_seqlens_kv_indptr[bid + 1] - params.p_seqlens_kv_indptr[bid]);
                const int32_t num_kv_blocks =
                    integer_divide_ceil_power2(seqlen_kv, kv_gran, params.kv_granularity_log2);
                const int32_t qo_tile_size = qo_state.get_seqlen(bid);

                if(num_kv_blocks + overhead <= remain_payload)
                {
                    num_works += 1;
                    remain_payload -= (num_kv_blocks + overhead);
                }
                else
                {
                    int32_t num_fresh_frags, num_frags;
                    if(remain_payload > overhead)
                    {
                        const int32_t remain_blocks = num_kv_blocks - (remain_payload - overhead);
                        num_fresh_frags = integer_divide_ceil(remain_blocks, blocks_per_cu);
                        num_frags       = num_fresh_frags + 1;
                        curr_cu += num_fresh_frags;
                        remain_payload =
                            payload -
                            ((remain_blocks - (num_fresh_frags - 1) * blocks_per_cu) + overhead);
                    }
                    else
                    {
                        num_fresh_frags = integer_divide_ceil(num_kv_blocks, blocks_per_cu);
                        num_frags       = num_fresh_frags;
                        curr_cu += num_fresh_frags;
                        remain_payload =
                            payload -
                            ((num_kv_blocks - (num_fresh_frags - 1) * blocks_per_cu) + overhead);
                    }
                    num_works += num_frags;
                    if(num_frags > 1)
                    {
                        last_reduce_indptr += num_frags;
                        partial_idx += num_frags * qo_tile_size;
                    }
                }
            }
            p_lds_scalars[1] = num_works;          // total works
            p_lds_scalars[2] = last_reduce_indptr; // total reduce tiles
        }
    }

    __syncthreads();

    const int32_t payload       = p_lds_scalars[0];
    const int32_t blocks_per_cu = payload - overhead;
    const int32_t total_works   = p_lds_scalars[1];
    const int32_t num_cu        = params.num_cu;

    // Phase 2a: init work_indptr to total_works
    for(int32_t cid = tid + 1; cid <= num_cu; cid += blockDim.x)
    {
        params.p_work_indptr[cid] = total_works;
    }

    __syncthreads();

    // Phase 2b: fill works / reduce / work_indptr
    for(int32_t bid = warp_id; bid < num_batches; bid += num_warps)
    {
        const int32_t start_cu           = p_lds_start_cu[bid];
        const int32_t remain_payload     = p_lds_remain_payload[bid];
        const int32_t num_works_before   = p_lds_works_before[bid];
        const int32_t reduce_before      = p_lds_reduce_before[bid];
        const int32_t partial_idx_before = p_lds_partial_before[bid];

        const int32_t kv_indptr0 = params.p_seqlens_kv_indptr[0];
        const int32_t kv_begin   = params.p_seqlens_kv_indptr[bid] - kv_indptr0;
        const int32_t kv_end     = params.p_seqlens_kv_indptr[bid + 1] - kv_indptr0;
        const int32_t seqlen_kv =
            Traits::kLdsBatchInfo ? p_lds_seqlens_kv[bid] : (kv_end - kv_begin);
        const int32_t num_kv_blocks =
            integer_divide_ceil_power2(seqlen_kv, kv_gran, params.kv_granularity_log2);
        const int32_t qo_tile_size = qo_state.get_seqlen(bid);
        const int32_t qo_start     = qo_state.get_begin(bid);
        const int32_t qo_end       = qo_state.get_end(bid);

        const bool fits_current_cu = (num_kv_blocks + overhead <= remain_payload);
        int32_t num_fresh_frags, num_frags;
        bool waste_start_cu;
        if(fits_current_cu)
        {
            num_fresh_frags = 1;
            num_frags       = 1;
            waste_start_cu  = false;
        }
        else if(remain_payload > overhead)
        {
            const int32_t remain_blocks = num_kv_blocks - (remain_payload - overhead);
            num_fresh_frags             = integer_divide_ceil(remain_blocks, blocks_per_cu);
            num_frags                   = num_fresh_frags + 1;
            waste_start_cu              = false;
        }
        else
        {
            num_fresh_frags = integer_divide_ceil(num_kv_blocks, blocks_per_cu);
            num_frags       = num_fresh_frags;
            waste_start_cu  = true;
        }
        const bool is_split             = (num_frags > 1);
        const int32_t first_frag_blocks = remain_payload - overhead;

        // Per-batch reduce bookkeeping + the wasted-CU close (lane 0).
        if(lane_idx == 0)
        {
            if(is_split)
            {
                params.p_reduce_indptr[bid + 1]        = reduce_before + num_frags;
                params.p_reduce_final_map[bid * 2]     = qo_start;
                params.p_reduce_final_map[bid * 2 + 1] = qo_end;
            }
            else
            {
                params.p_reduce_indptr[bid + 1] = reduce_before;
            }
            if(waste_start_cu && (start_cu + 1 <= num_cu))
            {
                params.p_work_indptr[start_cu + 1] = num_works_before;
            }
        }

        // Each fragment -> one work covering kv blocks [block_begin, block_end)
        // of this batch, landing in CU frag_cu.
        for(int32_t frag_idx = lane_idx; frag_idx < num_frags; frag_idx += opus::get_warp_size())
        {
            int32_t block_begin, block_end, frag_cu;
            if(fits_current_cu)
            {
                block_begin = 0;
                block_end   = num_kv_blocks;
                frag_cu     = start_cu;
            }
            else if(!waste_start_cu)
            {
                block_begin =
                    (frag_idx == 0) ? 0 : (first_frag_blocks + (frag_idx - 1) * blocks_per_cu);
                block_end = (frag_idx < num_fresh_frags)
                                ? (first_frag_blocks + frag_idx * blocks_per_cu)
                                : num_kv_blocks;
                frag_cu   = start_cu + frag_idx;
            }
            else
            {
                block_begin = frag_idx * blocks_per_cu;
                block_end   = (frag_idx < num_fresh_frags - 1) ? ((frag_idx + 1) * blocks_per_cu)
                                                               : num_kv_blocks;
                frag_cu     = start_cu + 1 + frag_idx;
            }

            const int32_t frag_kv_start = kv_begin + block_begin * kv_gran;
            const int32_t frag_kv_end   = opus::min(kv_begin + block_end * kv_gran, kv_end);
            const int32_t work_idx      = num_works_before + frag_idx;
            const int32_t partial_qo_loc =
                is_split ? (partial_idx_before + frag_idx * qo_tile_size) : -1;

            MlaWorkInfo work_info{};
            work_info.batch_idx       = bid;
            work_info.qo_start        = qo_start;
            work_info.qo_end          = qo_end;
            work_info.kv_start        = frag_kv_start;
            work_info.kv_end          = frag_kv_end;
            work_info.kv_offset       = kv_end - frag_kv_end;
            work_info.partial_qo_loc  = partial_qo_loc;
            p_work_info_set[work_idx] = work_info;

            if(is_split)
            {
                params.p_reduce_partial_map[reduce_before + frag_idx] =
                    partial_idx_before + frag_idx * qo_tile_size;
            }

            // Non-final fragments fully fill (close) their CU with one work.
            const bool is_last_frag = (frag_idx == num_frags - 1);
            if(!is_last_frag && (frag_cu + 1 <= num_cu))
            {
                params.p_work_indptr[frag_cu + 1] = work_idx + 1;
            }
        }
    }

    // Phase 3: fill the reduce_indptr tail
    const int32_t total_reduce = p_lds_scalars[2];
    for(int32_t i = num_batches + tid; i < params.reduce_indptr_size; i += blockDim.x)
    {
        params.p_reduce_indptr[i] = total_reduce;
    }
}

template <typename Traits>
__launch_bounds__(opus::get_warp_size(), 1) __global__
    void kn_get_mla_metadata_v1_2(MlaMetadataV1KernelParameter params)
{
    using QoState = QoState<Traits>;

    const int32_t ori_seqlen_qo = [&]() {
        if constexpr(Traits::kIsSparse)
        {
            return params.p_seqlens_qo_indptr[1] - params.p_seqlens_qo_indptr[0];
        }
        else
        {
            return params.ori_seqlen_qo;
        }
    }();

    const int32_t num_batches = [&]() {
        if constexpr(Traits::kIsSparse)
        {
            return params.num_batches * ori_seqlen_qo;
        }
        else
        {
            return params.num_batches;
        }
    }();

    extern __shared__ uint8_t p_smem[];
    int32_t* p_lds_seqlens_qo = reinterpret_cast<int32_t*>(p_smem);
    int32_t* p_lds_seqlens_kv = p_lds_seqlens_qo + (QoState::is_unique() ? 0 : num_batches);

    QoState qo_state(
        params.uni_seqlen_qo, ori_seqlen_qo, p_lds_seqlens_qo, params.p_seqlens_qo_indptr);

    const int32_t lane_idx = opus::lane_id();

    MlaWorkInfo* p_work_info_set = reinterpret_cast<MlaWorkInfo*>(params.p_work_info_set_raw);

    const int32_t sum_blocks = mla_v12_compute_sum_blocks<Traits>(
        params, qo_state, p_lds_seqlens_qo, p_lds_seqlens_kv, ori_seqlen_qo, num_batches, lane_idx);

    if(lane_idx == 0)
    {
        params.p_reduce_indptr[0] = 0;
        params.p_work_indptr[0]   = 0;
        params.p_work_metadata_ptrs[0] =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.p_work_indptr));
        params.p_work_metadata_ptrs[1] =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p_work_info_set));
    }

    // expected payload handled by each cu part.
    const int32_t payload =
        integer_divide_ceil(sum_blocks, params.num_splits) + params.fixed_over_head_num_blocks;
    const int32_t page_size   = params.page_size;
    int32_t curr_batch        = 0; // batch ID of the batch which is under review
    int32_t curr_kv_block     = 0; // #blocks handled by previous cu part(s)
    int32_t curr_n_split_idx  = 0; // #cu parts used to handle current batch
    int32_t curr_qo_tile_idx  = 0;
    int32_t curr_sub_head_idx = 0;

    int32_t curr_kv_begin = 0;
    // The size of 1st element equals to the end loc of the 1st element.
    int32_t curr_kv_end    = Traits::kLdsBatchInfo ? p_lds_seqlens_kv[0]
                             : Traits::kIsSparse   ? min(params.p_seqlens_kv_indptr[1], params.topk)
                                                   : params.p_seqlens_kv_indptr[1];
    int32_t curr_kv_seqlen = curr_kv_end - curr_kv_begin;

    int32_t num_works          = 0;
    int32_t partial_idx        = 0;
    int32_t tot_qo_tiles       = 0;
    int32_t last_reduce_indptr = 0;
    bool cur_tail_done         = false;

    for(int32_t cid = 0; cid < params.num_cu; ++cid)
    {
        int32_t remain_payload = payload;
        while(curr_batch < num_batches)
        {
            const int32_t num_qo_tiles = mla_v12_num_qo_tiles<Traits>(params, qo_state, curr_batch);
            const int32_t qo_tile_size =
                integer_divide_ceil(qo_state.get_seqlen(curr_batch), num_qo_tiles);
            const int32_t num_kv_blocks = integer_divide_ceil_power2(
                curr_kv_seqlen, params.kv_granularity, params.kv_granularity_log2);
            const int32_t remain_kv_blocks = num_kv_blocks - curr_kv_block;

            // If current cu part is able to handle this batch of seqences
            if(remain_payload >= (remain_kv_blocks + params.fixed_over_head_num_blocks) ||
               cur_tail_done)
            {
                const int32_t num_splits = curr_n_split_idx + 1;

                auto fill_work_info = [&](const int32_t split_idx) {
                    const int32_t global_qo_tile_idx = tot_qo_tiles;
                    const int32_t curr_batch_kv =
                        Traits::kIsSparse ? (curr_batch / ori_seqlen_qo / params.qk_batch_ratio)
                                          : curr_batch;

                    MlaWorkInfo work_info{};
                    work_info.batch_idx = curr_batch_kv;
                    work_info.qo_start =
                        qo_state.get_begin(curr_batch) + curr_qo_tile_idx * qo_tile_size;
                    work_info.qo_end =
                        opus::min(work_info.qo_start + qo_tile_size, qo_state.get_end(curr_batch));
                    work_info.kv_start = curr_kv_begin + (curr_kv_block * params.kv_granularity);
                    if(page_size == 1)
                    {
                        // round-robin CP: no local-causal trim (full local kv per
                        // work); kernel masks on global positions.
                        int32_t batch_tail =
                            params.is_cp_round_robin ? 0 : (num_qo_tiles - 1 - curr_qo_tile_idx);
                        if constexpr(!Traits::kIsSparse)
                        {
                            if(params.qk_batch_ratio != 1)
                            {
                                batch_tail =
                                    num_qo_tiles -
                                    (work_info.qo_start / params.qk_batch_ratio) % ori_seqlen_qo -
                                    1;
                            }
                        }
                        batch_tail       = opus::max(batch_tail, 0);
                        work_info.kv_end = opus::min(work_info.kv_start +
                                                         (remain_kv_blocks * params.kv_granularity),
                                                     curr_kv_end - batch_tail);
                        if((curr_kv_end - work_info.kv_end < params.tail_done_threshold &&
                            curr_kv_end - work_info.kv_end > 0) ||
                           cur_tail_done)
                        {
                            work_info.kv_end = opus::min(curr_kv_end - batch_tail, curr_kv_end);
                        }
                        work_info.kv_offset = curr_kv_end - work_info.kv_end;
                        if(Traits::kIsSparse && params.qk_batch_ratio == 1)
                        {
                            work_info.batch_idx = curr_batch / ori_seqlen_qo;
                            work_info.kv_offset += ori_seqlen_qo - 1 - (curr_batch % ori_seqlen_qo);
                        }
                    }
                    else
                    {
                        work_info.kv_end = opus::min(work_info.kv_start +
                                                         (remain_kv_blocks * params.kv_granularity),
                                                     curr_kv_end);
                        work_info.kv_offset =
                            (curr_kv_end - work_info.kv_end == 0)
                                ? 0
                                : ((curr_kv_end - work_info.kv_end - 1) * page_size +
                                   params.p_kv_last_page_lens[curr_batch_kv]);
                    }
                    // split related info
                    if(curr_n_split_idx > 0)
                    {
                        // set work info
                        work_info.partial_qo_loc = partial_idx;

                        // set reduce info
                        params.p_reduce_indptr[global_qo_tile_idx + 1] =
                            last_reduce_indptr + num_splits;
                        params.p_reduce_final_map[global_qo_tile_idx * 2]     = work_info.qo_start;
                        params.p_reduce_final_map[global_qo_tile_idx * 2 + 1] = work_info.qo_end;
                        params.p_reduce_partial_map[last_reduce_indptr + split_idx] =
                            partial_idx - (curr_n_split_idx - split_idx) * qo_tile_size;
                    }
                    else
                    {
                        work_info.partial_qo_loc                       = -1;
                        params.p_reduce_indptr[global_qo_tile_idx + 1] = last_reduce_indptr;
                    }

                    p_work_info_set[num_works] = work_info;
                };

                // record a work in work_info_set
                if(curr_n_split_idx > 0)
                {
                    for(int32_t idx = lane_idx; idx < num_splits; idx += opus::get_warp_size())
                    {
                        fill_work_info(idx);
                    }

                    partial_idx += qo_tile_size;
                    last_reduce_indptr += num_splits;
                }
                else
                {
                    fill_work_info(0);
                }

                tot_qo_tiles += 1;
                num_works += 1;

                remain_payload -= (remain_kv_blocks + params.fixed_over_head_num_blocks);

                // update state
                curr_qo_tile_idx =
                    (curr_qo_tile_idx == (num_qo_tiles - 1)) ? 0 : (curr_qo_tile_idx + 1);
                if((Traits::kQoSplits == false) || (curr_qo_tile_idx == 0))
                {
                    ++curr_batch;
                    // same as curr_sub_head_idx = (curr_sub_head_idx + 1) % params.qk_batch_ratio;
                    curr_sub_head_idx = (curr_sub_head_idx == (params.qk_batch_ratio - 1))
                                            ? 0
                                            : (curr_sub_head_idx + 1);
                    if(curr_batch < num_batches)
                    {
                        if(curr_sub_head_idx == 0)
                        {
                            if constexpr(Traits::kLdsBatchInfo)
                            {
                                curr_kv_seqlen = p_lds_seqlens_kv[curr_batch];
                            }
                            else
                            {
                                const int32_t bid_ori =
                                    Traits::kIsSparse
                                        ? (curr_batch / ori_seqlen_qo / params.qk_batch_ratio)
                                        : (curr_batch / params.qk_batch_ratio);
                                curr_kv_seqlen = params.p_seqlens_kv_indptr[bid_ori + 1] -
                                                 params.p_seqlens_kv_indptr[bid_ori];
                                curr_kv_seqlen = Traits::kIsSparse
                                                     ? min(curr_kv_seqlen, params.topk)
                                                     : curr_kv_seqlen;
                            }
                            curr_kv_begin =
                                Traits::kIsSparse ? (curr_kv_begin + params.topk) : curr_kv_end;
                            curr_kv_end = curr_kv_begin + curr_kv_seqlen;
                        }
                        curr_kv_block    = 0;
                        curr_n_split_idx = 0;
                        cur_tail_done    = false;
                    }
                }
                else
                {
                    curr_kv_block    = 0;
                    curr_n_split_idx = 0;
                    cur_tail_done    = false;
                }
            }
            else
            {
                if(remain_payload > params.fixed_over_head_num_blocks)
                {
                    const int32_t consuming_blks =
                        remain_payload - params.fixed_over_head_num_blocks;

                    auto fill_work_info = [&]() {
                        const int32_t curr_batch_kv =
                            Traits::kIsSparse ? (curr_batch / ori_seqlen_qo / params.qk_batch_ratio)
                                              : curr_batch;
                        MlaWorkInfo work_info{};
                        work_info.batch_idx = curr_batch_kv;
                        work_info.qo_start =
                            qo_state.get_begin(curr_batch) + curr_qo_tile_idx * qo_tile_size;
                        work_info.qo_end = opus::min(work_info.qo_start + qo_tile_size,
                                                     qo_state.get_end(curr_batch));
                        work_info.kv_start =
                            curr_kv_begin + (curr_kv_block * params.kv_granularity);
                        if(page_size == 1)
                        {
                            // round-robin CP: no local-causal trim (see note above).
                            int32_t batch_tail = params.is_cp_round_robin
                                                     ? 0
                                                     : (num_qo_tiles - 1 - curr_qo_tile_idx);
                            if constexpr(!Traits::kIsSparse)
                            {
                                if(params.qk_batch_ratio != 1)
                                {
                                    batch_tail = num_qo_tiles -
                                                 (work_info.qo_start / params.qk_batch_ratio) %
                                                     ori_seqlen_qo -
                                                 1;
                                }
                            }
                            batch_tail       = opus::max(batch_tail, 0);
                            work_info.kv_end = opus::min(
                                work_info.kv_start + (consuming_blks * params.kv_granularity),
                                curr_kv_end - batch_tail);
                            if(curr_kv_end - work_info.kv_end < params.tail_done_threshold)
                            {
                                cur_tail_done    = true;
                                work_info.kv_end = opus::min(curr_kv_end, curr_kv_end - batch_tail);
                            }
                            work_info.kv_offset = curr_kv_end - work_info.kv_end;
                            if(Traits::kIsSparse && params.qk_batch_ratio == 1)
                            {
                                work_info.batch_idx = curr_batch / ori_seqlen_qo;
                                work_info.kv_offset +=
                                    ori_seqlen_qo - 1 - (curr_batch % ori_seqlen_qo);
                            }
                        }
                        else
                        {
                            work_info.kv_end = opus::min(
                                work_info.kv_start + (consuming_blks * params.kv_granularity),
                                curr_kv_end);
                            work_info.kv_offset =
                                (curr_kv_end - work_info.kv_end == 0)
                                    ? 0
                                    : ((curr_kv_end - work_info.kv_end - 1) * page_size +
                                       params.p_kv_last_page_lens[curr_batch_kv]);
                        }
                        work_info.partial_qo_loc = partial_idx;
                        if(!cur_tail_done)
                        {
                            p_work_info_set[num_works] = work_info;
                        }
                    };

                    // record a work in work_info_set
                    fill_work_info();
                    if(!cur_tail_done)
                    {
                        partial_idx += qo_tile_size;
                        num_works += 1;

                        // update state
                        curr_kv_block += consuming_blks;
                        ++curr_n_split_idx;
                    }
                }
                if(!cur_tail_done)
                {
                    break;
                }
            }
        }

        params.p_work_indptr[cid + 1] = num_works;
    }

    for(int32_t i = tot_qo_tiles + lane_idx; i < params.reduce_indptr_size;
        i += opus::get_warp_size())
    {
        params.p_reduce_indptr[i] = last_reduce_indptr;
    }
}

template <int32_t kPackedQoLenPerWg, bool kQoSplits, int32_t kUniSeqlenQo, bool kIsSparse>
void dispatch_mla_metadata_v1_2_device(const MlaMetadataV1KernelParameter& params,
                                       const hipStream_t stream,
                                       const int32_t max_seqlen_qo,
                                       const int32_t warp_size,
                                       const int32_t lds_size)
{
    const dim3 grid = dim3(1, 1, 1);

    using DummyTraits =
        MlaMetadataV12Traits<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, true, kIsSparse>;
    const bool is_unique              = QoState<DummyTraits>::is_unique();
    const int32_t lds_bytes_per_batch = sizeof(int32_t) * (is_unique ? 1 : 2);
    const int32_t max_qo_tiles =
        kQoSplits ? (integer_divide_ceil(max_seqlen_qo, kPackedQoLenPerWg)) : 1;
    const int32_t max_lds_batch_size = lds_size / lds_bytes_per_batch;

    const char* parallel_env   = std::getenv("AITER_MLA_META_USE_PARALLEL");
    const bool parallel_wanted = (parallel_env == nullptr) || (std::atoi(parallel_env) != 0);
    const bool use_parallel = parallel_wanted && (max_seqlen_qo == 1) && !kQoSplits && !kIsSparse &&
                              (params.page_size == 1) && (params.qk_batch_ratio == 1);
    const int32_t scratch_bytes =
        static_cast<int32_t>(sizeof(int32_t)) * (3 + 5 * params.num_batches);
    const int32_t qo_bytes =
        is_unique ? 0 : static_cast<int32_t>(sizeof(int32_t)) * params.num_batches;
    const int32_t kv_bytes   = static_cast<int32_t>(sizeof(int32_t)) * params.num_batches;
    const int32_t fill_block = warp_size * MLA_V12_FILL_WARPS;

    if(use_parallel && (scratch_bytes + qo_bytes + kv_bytes <= lds_size))
    {
        using Traits =
            MlaMetadataV12Traits<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, true, kIsSparse>;
        kn_get_mla_metadata_v1_2_parallel<Traits><<<grid, fill_block, lds_size, stream>>>(params);
    }
    else if(use_parallel && (scratch_bytes + qo_bytes <= lds_size))
    {
        using Traits =
            MlaMetadataV12Traits<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, false, kIsSparse>;
        kn_get_mla_metadata_v1_2_parallel<Traits><<<grid, fill_block, lds_size, stream>>>(params);
    }
    else if(params.num_batches <= max_lds_batch_size)
    {
        using Traits =
            MlaMetadataV12Traits<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, true, kIsSparse>;
        kn_get_mla_metadata_v1_2<Traits><<<grid, warp_size, lds_size, stream>>>(params);
    }
    else
    {
        using Traits =
            MlaMetadataV12Traits<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, false, kIsSparse>;
        kn_get_mla_metadata_v1_2<Traits><<<grid, warp_size, lds_size, stream>>>(params);
    }
}

// HK MLA m16x4 kernel runs at occupancy=2 (gfx950 + 64 q-tokens per tile, gated on
// AITER_ENABLE_EXPERIMENTAL same as the dispatch in aiter/mla.py:use_hk). When it
// applies, the m16x4 launch site spawns 2*num_cu workgroups; the work distribution
// here must produce work_indptr sized to match so the second occupancy slot actually
// receives work. Detection mirrors hk_decode_fwd dispatch (num_heads * max_seqlen_qo
// == 64) and uses ORIGINAL num_heads/max_seqlen_qo (pre-fold). V32 uses fp8 across
// nope+rope; V40 uses fp8 nope + bf16 rope.
static inline int32_t mla_metadata_cluster_multiplier(const std::string& arch_id,
                                                      const bool enable_experimental,
                                                      const int32_t num_heads,
                                                      const int32_t max_seqlen_qo,
                                                      const MlaVersion mla_version,
                                                      const at::ScalarType q_nope_dtype,
                                                      const at::ScalarType q_rope_dtype,
                                                      const at::ScalarType kv_nope_dtype,
                                                      const at::ScalarType kv_rope_dtype)
{
    auto is_fp8 = [](const at::ScalarType dtype) {
        return dtype == at::ScalarType::Float8_e4m3fnuz || dtype == at::ScalarType::Float8_e4m3fn;
    };
    auto is_bf16 = [](const at::ScalarType dtype) { return dtype == at::ScalarType::BFloat16; };

    const bool dtype_ok =
        ((mla_version == MlaVersion::V32) && is_fp8(q_nope_dtype) && is_fp8(q_rope_dtype) &&
         is_fp8(kv_nope_dtype) && is_fp8(kv_rope_dtype)) ||
        ((mla_version == MlaVersion::V40) && is_fp8(q_nope_dtype) && is_bf16(q_rope_dtype) &&
         is_fp8(kv_nope_dtype) && is_bf16(kv_rope_dtype));

    const bool is_hk_m16x4 = enable_experimental && (arch_id == "gfx950") &&
                             (num_heads * max_seqlen_qo == 64) && dtype_ok;

    return is_hk_m16x4 ? 2 : 1;
}

void get_mla_metadata_v1_2_device(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
                                  const torch::Tensor& seqlens_kv_indptr, // [batch size + 1]
                                  const torch::Tensor& kv_last_page_lens, // [batch size]
                                  const int32_t num_heads_per_head_k,
                                  const int32_t num_heads_k,
                                  const bool is_causal,
                                  const int32_t page_size,
                                  const int32_t kv_granularity,
                                  const int32_t max_seqlen_qo,
                                  const int32_t ori_uni_seqlen_qo,
                                  const int32_t topk,
                                  const int32_t max_split_per_batch,
                                  const at::ScalarType q_dtype,
                                  const at::ScalarType kv_dtype,
                                  const at::ScalarType q_rope_dtype,
                                  const at::ScalarType kv_rope_dtype,
                                  const bool is_cp_round_robin,
                                  const MlaVersion mla_version,
                                  torch::Tensor& work_metadata_ptrs,
                                  torch::Tensor& work_info_set,
                                  torch::Tensor& work_indptr,
                                  torch::Tensor& reduce_indptr,
                                  torch::Tensor& reduce_final_map,
                                  torch::Tensor& reduce_partial_map)
{
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    hipGetDevice(&dev);
    hipGetDeviceProperties(&dev_prop, dev);

    const bool is_sparse = (topk >= 0);

    int32_t num_batches     = seqlens_kv_indptr.size(0) - 1;
    int32_t num_heads       = num_heads_k * num_heads_per_head_k;
    int32_t qk_batch_ratio  = 1;
    int32_t qk_seqlen_ratio = 1;
    int32_t uni_seqlen_qo   = ori_uni_seqlen_qo;

    auto arch_id = get_gpu_arch();

    // In the following cases, we use #head=16 to simulate cases which is not natively supported by
    // mla main kernel.
    const bool q_is_fp8 =
        (q_dtype == at::ScalarType::Float8_e4m3fnuz || q_dtype == at::ScalarType::Float8_e4m3fn);
    const bool kv_is_fp8 =
        (kv_dtype == at::ScalarType::Float8_e4m3fnuz || kv_dtype == at::ScalarType::Float8_e4m3fn);

    const bool enable_experimental = std::getenv("AITER_ENABLE_EXPERIMENTAL") != nullptr &&
                                     std::atoi(std::getenv("AITER_ENABLE_EXPERIMENTAL")) != 0;

    const int32_t cluster_multiplier = mla_metadata_cluster_multiplier(arch_id,
                                                                       enable_experimental,
                                                                       num_heads,
                                                                       max_seqlen_qo,
                                                                       mla_version,
                                                                       q_dtype,
                                                                       q_rope_dtype,
                                                                       kv_dtype,
                                                                       kv_rope_dtype);
    const int32_t num_clusters = (dev_prop.multiProcessorCount * cluster_multiplier) / num_heads_k;

    // Gate on arch_id consistent with hk_mla_v32_decode_fwd dispatch (gfx942/gfx950).
    // Otherwise this would mark shapes as natively supported on archs where the
    // HK kernels are unavailable, producing metadata that downstream kernels
    // cannot consume.
    const bool hk_mtp_experimental =
        (arch_id == "gfx942" || arch_id == "gfx950") && (q_is_fp8 && kv_is_fp8) &&
        (num_heads * max_seqlen_qo == 128) &&
        ((num_heads == 16) || (num_heads == 32) || (num_heads == 64) || (num_heads == 128)) &&
        enable_experimental;

    const bool natively_supported =
        (num_heads == 16) ||
        ((arch_id == "gfx942" || arch_id == "gfx950") && (num_heads == 64) && q_is_fp8 &&
         kv_is_fp8 && (max_seqlen_qo == 1)) ||
        ((arch_id == "gfx950") && !q_is_fp8 && !kv_is_fp8) ||
        ((arch_id == "gfx942") && (num_heads == 128) && q_is_fp8 && kv_is_fp8) ||
        ((arch_id == "gfx950") && q_is_fp8 && kv_is_fp8 &&
         ((num_heads == 32) || (num_heads == 64) || (num_heads == 128))) ||
        ((arch_id == "gfx950") && q_is_fp8 && kv_is_fp8 && (num_heads == 96) &&
         (max_seqlen_qo <= 6)) ||
        hk_mtp_experimental;

    if(!natively_supported && (num_heads % 16 == 0))
    {
        qk_batch_ratio = num_heads / 16;
        num_heads      = 16;
        num_batches *= qk_batch_ratio;
    }

    TORCH_CHECK(
        natively_supported || (num_heads == 16) || (num_heads == 128) ||
            ((num_heads == 32) && q_is_fp8 && kv_is_fp8) ||
            ((num_heads == 64) && q_is_fp8 && kv_is_fp8 && (max_seqlen_qo == 1)) ||
            ((arch_id == "gfx950") && (num_heads == 8) && (max_seqlen_qo == 4) && q_is_fp8 &&
             kv_is_fp8) ||
            ((arch_id == "gfx942") && (num_heads == 8) && (max_seqlen_qo == 2) && !q_is_fp8 &&
             !kv_is_fp8) ||
            ((arch_id == "gfx950") && !q_is_fp8 && !kv_is_fp8) ||
            ((arch_id == "gfx950") && q_is_fp8 && kv_is_fp8 &&
             (((num_heads == 32) && (max_seqlen_qo == 4)) || (num_heads == 64) ||
              (num_heads == 128))) ||
            hk_mtp_experimental,
        __func__,
        ": only supports #heads in [16, 64, 128], or (#head, uni_seqlen_qo) = (16*N, 1) where "
        "N is in [2, 8), or (#head, max_seqlen_qo) = (8, 4) where q and kv are fp8, "
        "or q and kv are bf16 on gfx950")

    int32_t num_splits = max_split_per_batch < 0
                             ? num_clusters
                             : min(num_clusters, max_split_per_batch * num_batches);

    MlaMetadataV1KernelParameter params = {};
    params.p_work_metadata_ptrs         = work_metadata_ptrs.data_ptr<uint64_t>();
    params.p_work_indptr                = work_indptr.data_ptr<int32_t>();
    params.p_work_info_set_raw          = work_info_set.data_ptr<int32_t>();
    params.p_reduce_indptr              = reduce_indptr.data_ptr<int32_t>();
    params.p_reduce_final_map           = reduce_final_map.data_ptr<int32_t>();
    params.p_reduce_partial_map         = reduce_partial_map.data_ptr<int32_t>();
    params.p_seqlens_qo_indptr          = seqlens_qo_indptr.data_ptr<int32_t>();
    params.p_seqlens_kv_indptr          = seqlens_kv_indptr.data_ptr<int32_t>();
    params.p_kv_last_page_lens          = kv_last_page_lens.data_ptr<int32_t>();
    params.num_batches                  = num_batches;
    params.num_heads                    = num_heads;
    params.num_cu                       = num_clusters;
    params.num_splits                   = num_splits;
    params.reduce_indptr_size           = reduce_indptr.size(0);
    params.page_size                    = page_size;
    params.kv_granularity               = kv_granularity;
    params.kv_granularity_log2          = __builtin_ctz(kv_granularity);
    params.uni_seqlen_qo                = uni_seqlen_qo;
    params.ori_seqlen_qo                = ori_uni_seqlen_qo;
    params.is_causal                    = is_causal;
    params.is_cp_round_robin            = is_cp_round_robin;
    params.topk                         = (topk < 0) ? topk : (topk + page_size - 1) / page_size;
    params.qk_batch_ratio               = qk_batch_ratio;
    params.fixed_over_head_num_blocks   = max(1, (16 + page_size - 1) / page_size);
    params.tail_done_threshold          = max_seqlen_qo;

    int32_t kPackedQoLenPerWg = 128;
    if((arch_id == "gfx950") && !q_is_fp8 && !kv_is_fp8 && (num_heads * max_seqlen_qo >= 64) &&
       (num_heads <= 64) && (((num_heads * max_seqlen_qo) < 128) || (num_heads == 48)))
    {
        kPackedQoLenPerWg = 64;
    }
    else if((arch_id == "gfx950") && q_is_fp8 && kv_is_fp8 && (num_heads == 32) &&
            (max_seqlen_qo == 3))
    {
        kPackedQoLenPerWg = 64;
    }

    // launch kernel
    MLA_METADATA_DISPATCHER(
        max_seqlen_qo * num_heads_per_head_k,
        kPackedQoLenPerWg,
        params.uni_seqlen_qo,
        topk,
        dispatch_mla_metadata_v1_2_device<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, kIsSparse>(
            params,
            stream,
            max_seqlen_qo,
            dev_prop.warpSize,
            dev_prop.maxSharedMemoryPerMultiProcessor));
}
