## Experimental: DeepSeek Sparse Attention

Approximate **decode-time** compute + memory model for **DeepSeek Sparse Attention (DSA)**.

---

### Symbols

- $H$: number of query heads
- $D$: head dim for the *indexer* dot product
- $L$: full context length (a.k.a. $n$ or $n_\text{ctx}$)
- $k$: sparse selection size (Top-k)
- $d_{qk}$: per-head QK dimension
- $d_v$: per-head V dimension

---

## FLOPs Model

### 1) Indexer FLOPs

From the [vllm blog](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html), the relevant operand shapes are:

- Query for 1 token: $(H, D)$  
- Context keys: $(L, D)$  
- Logits: $(L, H)$  
- Head weights: $(H)$  
- Weighted logits: $(L)$  

#### Components

1. **Logits (Q × K)**  
   Query × context dot products:
   - FLOPs: $2MNK = 2 \cdot L \cdot D \cdot H$

2. **Weighting logits by head weights**  
   Multiply-and-reduce across heads:
   - Approx FLOPs: $2HL$
   - More accurate: $L(2H-1)$

#### Result

Indexer FLOPs:
- $$\text{FLOPs}_\text{indexer} = 2LHD + L(2H-1)$$

---

### 2) MLA Decode FLOPs (dense)

In decode, **1 query vector per head** attends over $N$ keys (here $N = n_\text{ctx}$):

For **one head**:

- **QK FLOPs** for 1 token: $2 \cdot 1 \cdot N \cdot d_{qk}$
- **PV FLOPs** for 1 token: $2 \cdot 1 \cdot N \cdot d_v$

So per head:
- $\text{FLOPs}\_{\text{1 head}} = 2N(d_{qk} + d_v) $

Multiply by number of query heads $H$:
- $\text{FLOPs}\_\text{MLA}(N) = 2H N (d_{qk} + d_v)$

In the baseline (dense) case, $N = L$:
- $\text{FLOPs}\_\text{MLA,dense} = 2H L (d_{qk} + d_v)$

---

### 3) DSA Decode FLOPs (sparse)

For **Sparse DSA MLA**, the attention context is restricted to Top-k:
- $N = k$

So:
- $\text{FLOPs}\_\text{MLA,sparse} = 2H k (d_{qk} + d_v)$

Total DSA decode FLOPs:
- $\text{FLOPs}\_\text{DSA decode} = \text{FLOPs}\_\text{indexer} + \text{FLOPs}_\text{MLA}(k)$

Baseline dense MLA decode FLOPs:
- $\text{FLOPs}\_\text{baseline decode} = \text{FLOPs}_\text{MLA}(L)$

---

## Bytes Model

### KV cache sizes

Two KV formats:

1. **Indexer KV (from [vLLM code](https://github.com/vllm-project/vllm/blob/6fa6e7ef0c9b485e8a684211e96691731aad6faa/vllm/utils/deep_gemm.py#L305))**
   - `kv_cache_fp8` is `uint8` with shape `[num_blocks, block_size, 1, D+4]`
   - Interpreted as **$(D+4)$ bytes per token** for indexer KV reads

2. **FlashMLA KV (from [vLLM docs](https://docs.vllm.ai/en/stable/api/vllm/v1/attention/backends/mla/flashmla_sparse/))**
   - Each token’s KV cache: **656 bytes**
   - Breakdown: 512 bytes fp8 “NoPE” + 16 bytes (4 float32 scales) + 128 bytes (64 bf16 RoPE)

So:
- Indexer KV bytes per token: $b_\text{idxKV} = D + 4$
- FlashMLA KV bytes per token: $b_\text{MLAKV} = 656$

---

### 1) Indexer bytes per token

Operations:

1. Read $Q$ once: $(H, D)$ in fp8  
2. Read head weights once: $(H)$ in fp32  
3. Read KV for full context: $L(D+4)$  
4. Compute logits + run Top-k: write + read logits of length $L$  
5. Write Top-k indices as int32: $4k$

Let:

- $b_Q$: bytes for reading $Q$
- $b_W$: bytes for reading head weights
- $b_\ell$: bytes per logit element (e.g., fp16 = 2, fp32 = 4)

Then:
- $\text{Bytes}\_\text{indexer} = b_Q + b_W + L(D+4) + 2(L \cdot b_\ell) + 4k$

Where concretely:
- $b_Q = H \cdot D \cdot 1 \quad (\text{fp8} \Rightarrow 1\text{ byte/elem})$
- $b_W = H \cdot 4 \quad (\text{fp32} \Rightarrow 4\text{ bytes/elem})$

---

### 2) Sparse MLA decode bytes per token

Operations:

1. Read query $Q$: $H \cdot d_{qk}$
2. Read indices: $4k$
3. Read KV cache for selected tokens: $k \cdot 656$
4. Write output: $H \cdot d_v$

Thus:
- $\text{Bytes}\_\text{MLA,sparse} = (2H d_{qk}) + 4k + 656k + (H d_v)$

---

## Benchmark command

```bash
python3 bench.py --batch-size 32 --min-len 2048 --max-len 131072 --step 2048 --topk 2048
```


---

<img width="1462" height="1004" alt="image" src="https://github.com/user-attachments/assets/95ab200b-328e-4298-ab50-27243b5fd1a3" />

 

<img width="2084" height="727" alt="image" src="https://github.com/user-attachments/assets/120dc407-ea49-4af6-b5cb-24a831a34224" />


