# Trainium3 rev2 exp / activation throughput (trn3.48xlarge, NeuronCore 0, LNC=1)

Source: `trn3_48xlarge_rev2_2026-11-06_act.csv`.

### exp / activation, one NeuronCore (LNC=1, pdim=128, SBUF -> SBUF)

| kernel | engine / opcode | fp32 -> | fdim | per op (fp32 in) | II ns | II cycles | elements / cycle / partition | GB/s in (median, min-max) | extra instrs | traces |
|---|---|---|---|---|---|---|---|---|---|---|
| exp_vector_softmax | Vector EXPONENTIAL | bf16 | 8 | 4 KiB | 46.7 | 56.1 | 0.143 | 88 (88-88) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | bf16 | 16 | 8 KiB | 46.7 | 56.0 | 0.285 | 175 (175-175) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | bf16 | 32 | 16 KiB | 46.7 | 56.0 | 0.572 | 351 (351-351) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | bf16 | 64 | 32 KiB | 46.5 | 55.9 | 1.146 | 704 (704-704) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | bf16 | 128 | 64 KiB | 66.8 | 80.1 | 1.598 | 982 (982-982) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | bf16 | 256 | 128 KiB | 127.6 | 153.1 | 1.672 | 1027 (1027-1027) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | bf16 | 512 | 256 KiB | 234.3 | 281.1 | 1.821 | 1119 (1119-1119) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | bf16 | 1024 | 512 KiB | 447.6 | 537.1 | 1.907 | 1171 (1171-1171) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | bf16 | 2048 | 1 MiB | 874.3 | 1049.1 | 1.952 | 1199 (1199-1199) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | bf16 | 4096 | 2 MiB | 1727.6 | 2073.1 | 1.976 | 1214 (1214-1214) | 0 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | bf16 | 8 | 4 KiB | 25.9 | 31.1 | 0.257 | 158 (158-158) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | bf16 | 16 | 8 KiB | 32.6 | 39.1 | 0.409 | 251 (251-251) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | bf16 | 32 | 16 KiB | 45.9 | 55.1 | 0.581 | 357 (357-357) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | bf16 | 64 | 32 KiB | 72.6 | 87.1 | 0.735 | 451 (451-451) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | bf16 | 128 | 64 KiB | 125.9 | 151.1 | 0.847 | 520 (520-520) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | bf16 | 256 | 128 KiB | 232.6 | 279.1 | 0.917 | 564 (564-564) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | bf16 | 512 | 256 KiB | 445.9 | 535.1 | 0.957 | 588 (588-588) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | bf16 | 1024 | 512 KiB | 872.6 | 1047.1 | 0.978 | 601 (601-601) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | bf16 | 2048 | 1 MiB | 1725.9 | 2071.1 | 0.989 | 608 (608-608) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | bf16 | 4096 | 2 MiB | 3432.6 | 4119.1 | 0.994 | 611 (611-611) | 1 | 3 |
| act_silu | Scalar ACTIVATE | bf16 | 8 | 4 KiB | 25.1 | 30.1 | 0.266 | 163 (163-163) | 1 | 3 |
| act_silu | Scalar ACTIVATE | bf16 | 16 | 8 KiB | 31.8 | 38.1 | 0.420 | 258 (258-258) | 1 | 3 |
| act_silu | Scalar ACTIVATE | bf16 | 32 | 16 KiB | 45.1 | 54.1 | 0.591 | 363 (363-363) | 1 | 3 |
| act_silu | Scalar ACTIVATE | bf16 | 64 | 32 KiB | 71.8 | 86.1 | 0.743 | 457 (457-457) | 1 | 3 |
| act_silu | Scalar ACTIVATE | bf16 | 128 | 64 KiB | 125.1 | 150.1 | 0.853 | 524 (524-524) | 1 | 3 |
| act_silu | Scalar ACTIVATE | bf16 | 256 | 128 KiB | 231.8 | 278.1 | 0.921 | 566 (566-566) | 1 | 3 |
| act_silu | Scalar ACTIVATE | bf16 | 512 | 256 KiB | 445.1 | 534.1 | 0.959 | 589 (589-589) | 1 | 3 |
| act_silu | Scalar ACTIVATE | bf16 | 1024 | 512 KiB | 871.8 | 1046.1 | 0.979 | 601 (601-601) | 1 | 3 |
| act_silu | Scalar ACTIVATE | bf16 | 2048 | 1 MiB | 1725.1 | 2070.1 | 0.989 | 608 (608-608) | 1 | 3 |
| act_silu | Scalar ACTIVATE | bf16 | 4096 | 2 MiB | 3431.8 | 4118.1 | 0.995 | 611 (611-611) | 1 | 3 |
| act_sigmoid | Scalar ACTIVATE | fp32 | 8 | 4 KiB | 25.1 | 30.1 | 0.266 | 163 (163-163) | 1 | 3 |
| act_sigmoid | Scalar ACTIVATE | fp32 | 16 | 8 KiB | 31.8 | 38.1 | 0.420 | 258 (258-258) | 1 | 3 |
| act_sigmoid | Scalar ACTIVATE | fp32 | 32 | 16 KiB | 45.1 | 54.1 | 0.591 | 363 (363-363) | 1 | 3 |
| act_sigmoid | Scalar ACTIVATE | fp32 | 64 | 32 KiB | 71.8 | 86.1 | 0.743 | 457 (457-457) | 1 | 3 |
| act_sigmoid | Scalar ACTIVATE | fp32 | 128 | 64 KiB | 125.1 | 150.1 | 0.853 | 524 (524-524) | 1 | 3 |
| act_sigmoid | Scalar ACTIVATE | fp32 | 256 | 128 KiB | 231.8 | 278.1 | 0.921 | 566 (566-566) | 1 | 3 |
| act_sigmoid | Scalar ACTIVATE | fp32 | 512 | 256 KiB | 445.1 | 534.1 | 0.959 | 589 (589-589) | 1 | 3 |
| act_sigmoid | Scalar ACTIVATE | fp32 | 1024 | 512 KiB | 871.8 | 1046.1 | 0.979 | 601 (601-601) | 1 | 3 |
| act_sigmoid | Scalar ACTIVATE | fp32 | 2048 | 1 MiB | 1725.1 | 2070.1 | 0.989 | 608 (608-608) | 1 | 3 |
| act_sigmoid | Scalar ACTIVATE | fp32 | 4096 | 2 MiB | 3431.8 | 4118.1 | 0.995 | 611 (611-611) | 1 | 3 |
| act_rsqrt | Scalar ACTIVATE | fp32 | 8 | 4 KiB | 25.1 | 30.1 | 0.266 | 163 (163-163) | 1 | 3 |
| act_rsqrt | Scalar ACTIVATE | fp32 | 16 | 8 KiB | 31.8 | 38.1 | 0.420 | 258 (258-258) | 1 | 3 |
| act_rsqrt | Scalar ACTIVATE | fp32 | 32 | 16 KiB | 45.1 | 54.1 | 0.591 | 363 (363-363) | 1 | 3 |
| act_rsqrt | Scalar ACTIVATE | fp32 | 64 | 32 KiB | 71.8 | 86.1 | 0.743 | 457 (457-457) | 1 | 3 |
| act_rsqrt | Scalar ACTIVATE | fp32 | 128 | 64 KiB | 125.1 | 150.1 | 0.853 | 524 (524-524) | 1 | 3 |
| act_rsqrt | Scalar ACTIVATE | fp32 | 256 | 128 KiB | 231.8 | 278.1 | 0.921 | 566 (566-566) | 1 | 3 |
| act_rsqrt | Scalar ACTIVATE | fp32 | 512 | 256 KiB | 445.1 | 534.1 | 0.959 | 589 (589-589) | 1 | 3 |
| act_rsqrt | Scalar ACTIVATE | fp32 | 1024 | 512 KiB | 871.8 | 1046.1 | 0.979 | 601 (601-601) | 1 | 3 |
| act_rsqrt | Scalar ACTIVATE | fp32 | 2048 | 1 MiB | 1725.1 | 2070.1 | 0.989 | 608 (608-608) | 1 | 3 |
| act_rsqrt | Scalar ACTIVATE | fp32 | 4096 | 2 MiB | 3431.8 | 4118.1 | 0.995 | 611 (611-611) | 1 | 3 |
| act_softplus | Scalar ACTIVATE | fp32 | 8 | 4 KiB | 25.1 | 30.1 | 0.266 | 163 (163-163) | 1 | 3 |
| act_softplus | Scalar ACTIVATE | fp32 | 16 | 8 KiB | 31.8 | 38.1 | 0.420 | 258 (258-258) | 1 | 3 |
| act_softplus | Scalar ACTIVATE | fp32 | 32 | 16 KiB | 45.1 | 54.1 | 0.591 | 363 (363-363) | 1 | 3 |
| act_softplus | Scalar ACTIVATE | fp32 | 64 | 32 KiB | 71.8 | 86.1 | 0.743 | 457 (457-457) | 1 | 3 |
| act_softplus | Scalar ACTIVATE | fp32 | 128 | 64 KiB | 125.1 | 150.1 | 0.853 | 524 (524-524) | 1 | 3 |
| act_softplus | Scalar ACTIVATE | fp32 | 256 | 128 KiB | 231.8 | 278.1 | 0.921 | 566 (566-566) | 1 | 3 |
| act_softplus | Scalar ACTIVATE | fp32 | 512 | 256 KiB | 445.1 | 534.1 | 0.959 | 589 (589-589) | 1 | 3 |
| act_softplus | Scalar ACTIVATE | fp32 | 1024 | 512 KiB | 871.8 | 1046.1 | 0.979 | 601 (601-601) | 1 | 3 |
| act_softplus | Scalar ACTIVATE | fp32 | 2048 | 1 MiB | 1725.1 | 2070.1 | 0.989 | 608 (608-608) | 1 | 3 |
| act_softplus | Scalar ACTIVATE | fp32 | 4096 | 2 MiB | 3431.8 | 4118.1 | 0.995 | 611 (611-611) | 1 | 3 |
| act_sqrt | Scalar ACTIVATE | fp32 | 8 | 4 KiB | 25.1 | 30.1 | 0.266 | 163 (163-163) | 1 | 3 |
| act_sqrt | Scalar ACTIVATE | fp32 | 16 | 8 KiB | 31.8 | 38.1 | 0.420 | 258 (258-258) | 1 | 3 |
| act_sqrt | Scalar ACTIVATE | fp32 | 32 | 16 KiB | 45.1 | 54.1 | 0.591 | 363 (363-363) | 1 | 3 |
| act_sqrt | Scalar ACTIVATE | fp32 | 64 | 32 KiB | 71.8 | 86.1 | 0.743 | 457 (457-457) | 1 | 3 |
| act_sqrt | Scalar ACTIVATE | fp32 | 128 | 64 KiB | 125.1 | 150.1 | 0.853 | 524 (524-524) | 1 | 3 |
| act_sqrt | Scalar ACTIVATE | fp32 | 256 | 128 KiB | 231.8 | 278.1 | 0.921 | 566 (566-566) | 1 | 3 |
| act_sqrt | Scalar ACTIVATE | fp32 | 512 | 256 KiB | 445.1 | 534.1 | 0.959 | 589 (589-589) | 1 | 3 |
| act_sqrt | Scalar ACTIVATE | fp32 | 1024 | 512 KiB | 871.8 | 1046.1 | 0.979 | 601 (601-601) | 1 | 3 |
| act_sqrt | Scalar ACTIVATE | fp32 | 2048 | 1 MiB | 1725.1 | 2070.1 | 0.989 | 608 (608-608) | 1 | 3 |
| act_sqrt | Scalar ACTIVATE | fp32 | 4096 | 2 MiB | 3431.8 | 4118.1 | 0.995 | 611 (611-611) | 1 | 3 |
| act_tanh | Scalar ACTIVATE | fp32 | 8 | 4 KiB | 25.1 | 30.1 | 0.266 | 163 (163-163) | 1 | 3 |
| act_tanh | Scalar ACTIVATE | fp32 | 16 | 8 KiB | 31.8 | 38.1 | 0.420 | 258 (258-258) | 1 | 3 |
| act_tanh | Scalar ACTIVATE | fp32 | 32 | 16 KiB | 45.1 | 54.1 | 0.591 | 363 (363-363) | 1 | 3 |
| act_tanh | Scalar ACTIVATE | fp32 | 64 | 32 KiB | 71.8 | 86.1 | 0.743 | 457 (457-457) | 1 | 3 |
| act_tanh | Scalar ACTIVATE | fp32 | 128 | 64 KiB | 125.1 | 150.1 | 0.853 | 524 (524-524) | 1 | 3 |
| act_tanh | Scalar ACTIVATE | fp32 | 256 | 128 KiB | 231.8 | 278.1 | 0.921 | 566 (566-566) | 1 | 3 |
| act_tanh | Scalar ACTIVATE | fp32 | 512 | 256 KiB | 445.1 | 534.1 | 0.959 | 589 (589-589) | 1 | 3 |
| act_tanh | Scalar ACTIVATE | fp32 | 1024 | 512 KiB | 871.8 | 1046.1 | 0.979 | 601 (601-601) | 1 | 3 |
| act_tanh | Scalar ACTIVATE | fp32 | 2048 | 1 MiB | 1725.1 | 2070.1 | 0.989 | 608 (608-608) | 1 | 3 |
| act_tanh | Scalar ACTIVATE | fp32 | 4096 | 2 MiB | 3431.8 | 4118.1 | 0.995 | 611 (611-611) | 1 | 3 |

### Fit: II cycles = issue overhead + fdim x cycles per element (fdim >= 256)

| kernel | issue overhead (cycles) | cycles per element per partition | elements per cycle | streaming GB/s | max abs error (cycles) |
|---|---|---|---|---|---|
| exp_vector_softmax | 25.1 | 0.500 | 2.000 | 1229 | 0.0 |
| exp_scalar_softmax | 23.1 | 1.000 | 1.000 | 614 | 0.0 |
| act_silu | 22.1 | 1.000 | 1.000 | 614 | 0.0 |
| act_sigmoid | 22.1 | 1.000 | 1.000 | 614 | 0.0 |
| act_rsqrt | 22.1 | 1.000 | 1.000 | 614 | 0.0 |
| act_softplus | 22.1 | 1.000 | 1.000 | 614 | 0.0 |
| act_sqrt | 22.1 | 1.000 | 1.000 | 614 | 0.0 |
| act_tanh | 22.1 | 1.000 | 1.000 | 614 | 0.0 |

