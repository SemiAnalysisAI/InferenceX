F="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
s=open(F).read(); orig=s
old="        num_head_k = self.num_heads"
new="        num_head_k = max(16, self.num_heads)  # PATCH(fp8-prefill-pad): 16-head PS metadata to match padded q/k/v (gcd(16,256)=16 -> ~960 tiles, not 4032)"
n=s.count(old)
assert n==2, f"expected 2 occurrences, found {n}"
s=s.replace(old,new)
open(F,"w").write(s)
print("PS metadata num_head_k -> max(16,num_heads) at both sites; replaced:",n)
