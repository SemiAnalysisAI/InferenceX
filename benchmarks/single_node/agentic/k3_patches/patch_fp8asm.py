F="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
s=open(F).read(); orig=s
a='''    @staticmethod
    def get_mla_padded_q(num_heads: int, q: torch.Tensor) -> torch.Tensor:
        return (
            q
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else q.repeat_interleave(
                AiterMLAHelper._AITER_MIN_MLA_HEADS // num_heads, dim=1
            )
        )'''
b='''    @staticmethod
    def get_mla_padded_q(num_heads: int, q: torch.Tensor) -> torch.Tensor:
        MIN = AiterMLAHelper._AITER_MIN_MLA_HEADS
        if num_heads >= MIN:
            return q
        if MIN % num_heads == 0:
            return q.repeat_interleave(MIN // num_heads, dim=1)
        # PATCH(fp8-asm): non-divisor head counts (K3=12) -> replicate-append
        # real heads to 16 (zero-pad produced garbage output).
        return torch.cat([q, q[:, : MIN - num_heads, :]], dim=1)'''
assert a in s, "padded_q anchor"; s=s.replace(a,b)
c='''    @staticmethod
    def get_mla_unpadded_o(num_heads: int, o: torch.Tensor) -> torch.Tensor:
        return (
            o
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else o[:, :: AiterMLAHelper._AITER_MIN_MLA_HEADS // num_heads, :]
        )'''
d='''    @staticmethod
    def get_mla_unpadded_o(num_heads: int, o: torch.Tensor) -> torch.Tensor:
        MIN = AiterMLAHelper._AITER_MIN_MLA_HEADS
        if num_heads >= MIN:
            return o
        if MIN % num_heads == 0:
            return o[:, :: MIN // num_heads, :]
        # PATCH(fp8-asm): drop appended heads; first num_heads are the real ones.
        return o[:, :num_heads, :]'''
assert c in s, "unpadded_o anchor"; s=s.replace(c,d)
e='''    @staticmethod
    def use_gluon_decode(num_heads: int, max_qo_len: int) -> bool:
        return num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS and max_qo_len == 1'''
f='''    @staticmethod
    def use_gluon_decode(num_heads: int, max_qo_len: int) -> bool:
        # PATCH(fp8-asm): route single-token decode to ASM 576/512 (append-pad
        # to 16). Keep gluon only for multi-token verify (no asm kernel there).
        if max_qo_len == 1:
            return False
        return num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS and max_qo_len == 1'''
assert e in s, "use_gluon_decode anchor"; s=s.replace(e,f)
assert s!=orig
open(F,"w").write(s)
print("patched OK; markers:", s.count("PATCH(fp8-asm)"))
