F="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
s=open(F).read(); orig=s

# --- A) relax the gate (2 occurrences) ---
g_old="_fp8_mla_prefill_supported() and self.num_heads % 16 == 0"
g_new="_fp8_mla_prefill_supported() and (self.num_heads % 16 == 0 or 0 < self.num_heads < 16)"
n=s.count(g_old); assert n>=1, "gate anchor not found"; s=s.replace(g_old,g_new)

# --- B) pad q/k/v to 16 in _mla_fp8_prefill_attn ---
b_old="""        fp8_dtype = current_platform.fp8_dtype()
        total_q = q.shape[0]
        nhead = self.num_heads
        v_head_dim = self.v_head_dim
        tile_q = _FP8_PREFILL_TILE_Q"""
b_new="""        fp8_dtype = current_platform.fp8_dtype()
        total_q = q.shape[0]
        # PATCH(fp8-prefill-pad): PS asm prefill + mla_reduce_v1 need 16-aligned
        # heads and the metadata is built for _num_attention_heads=16. K3 has 12
        # heads/rank -> replicate-pad q/k/v to 16 (heads independent over shared
        # KV, exact like decode); output sliced back to the real head count.
        _real_nhead = self.num_heads
        _pad16 = _real_nhead < 16
        if _pad16:
            q = AiterMLAHelper.get_mla_padded_q(_real_nhead, q)
            k = AiterMLAHelper.get_mla_padded_q(_real_nhead, k)
            v = AiterMLAHelper.get_mla_padded_q(_real_nhead, v)
        nhead = 16 if _pad16 else self.num_heads
        v_head_dim = self.v_head_dim
        tile_q = _FP8_PREFILL_TILE_Q"""
assert b_old in s, "nhead anchor not found"; s=s.replace(b_old,b_new,1)

# --- C) out_3d: padded case needs its own buffer (can't alias real-head out) ---
c_old="        out_3d = out.view(total_q, nhead, v_head_dim)"
c_new="""        if _pad16:
            out_3d = torch.empty(
                total_q, nhead, v_head_dim, dtype=out.dtype, device=out.device
            )
        else:
            out_3d = out.view(total_q, nhead, v_head_dim)"""
assert c_old in s, "out_3d anchor not found"; s=s.replace(c_old,c_new,1)

# --- D) copy padded output back into the caller's real-head buffer ---
d_old="""            0,
            out_3d,
            final_lse,
        )"""
d_new="""            0,
            out_3d,
            final_lse,
        )

        if _pad16:
            out.view(total_q, _real_nhead, v_head_dim).copy_(
                out_3d[:, :_real_nhead, :]
            )"""
assert d_old in s, "reduce anchor not found"; s=s.replace(d_old,d_new,1)

assert s!=orig
open(F,"w").write(s)
print("fp8 prefill pad patch applied; gate replaced:",n,"; markers:",s.count("PATCH(fp8-prefill-pad)"))
