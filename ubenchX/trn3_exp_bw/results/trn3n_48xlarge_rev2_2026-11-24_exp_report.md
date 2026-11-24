# Trainium3 rev2 softmax exp throughput (trn3n.48xlarge, NeuronCore 0, LNC=1)

Source: `trn3n_48xlarge_rev2_2026-11-24_exp.csv`. Fit over fdim >= 512.

### exp / activation, one NeuronCore (LNC=1, pdim=128, SBUF -> SBUF)

| kernel | engine / opcode | dtype | fdim | per op (in) | II ns | II cycles | elements / cycle / partition | GB/s in (median, min-max) | extra instrs | traces |
|---|---|---|---|---|---|---|---|---|---|---|
| exp_vector_softmax | Vector EXPONENTIAL | fp32 -> bf16 | 8 | 4 KiB | 46.7 | 56.1 | 0.143 | 88 (88-88) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | fp32 -> bf16 | 16 | 8 KiB | 46.7 | 56.0 | 0.285 | 175 (175-175) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | fp32 -> bf16 | 32 | 16 KiB | 46.7 | 56.0 | 0.572 | 351 (351-351) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | fp32 -> bf16 | 64 | 32 KiB | 46.6 | 55.9 | 1.145 | 704 (704-704) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | fp32 -> bf16 | 128 | 64 KiB | 66.8 | 80.1 | 1.598 | 982 (982-982) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | fp32 -> bf16 | 256 | 128 KiB | 126.8 | 152.1 | 1.683 | 1034 (1034-1034) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | fp32 -> bf16 | 512 | 256 KiB | 233.4 | 280.1 | 1.828 | 1123 (1123-1123) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | fp32 -> bf16 | 1024 | 512 KiB | 446.7 | 536.1 | 1.910 | 1174 (1174-1174) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | fp32 -> bf16 | 2048 | 1 MiB | 873.4 | 1048.1 | 1.954 | 1201 (1201-1201) | 0 | 3 |
| exp_vector_softmax | Vector EXPONENTIAL | fp32 -> bf16 | 4096 | 2 MiB | 1726.7 | 2072.1 | 1.977 | 1215 (1215-1215) | 0 | 3 |
| exp_vector_softmax_bf16 | Vector EXPONENTIAL | bf16 -> bf16 | 8 | 2 KiB | 46.7 | 56.1 | 0.143 | 44 (44-44) | 0 | 3 |
| exp_vector_softmax_bf16 | Vector EXPONENTIAL | bf16 -> bf16 | 16 | 4 KiB | 46.7 | 56.1 | 0.285 | 88 (88-88) | 0 | 3 |
| exp_vector_softmax_bf16 | Vector EXPONENTIAL | bf16 -> bf16 | 32 | 8 KiB | 46.7 | 56.1 | 0.571 | 175 (175-175) | 0 | 3 |
| exp_vector_softmax_bf16 | Vector EXPONENTIAL | bf16 -> bf16 | 64 | 16 KiB | 46.7 | 56.0 | 1.143 | 351 (351-351) | 0 | 3 |
| exp_vector_softmax_bf16 | Vector EXPONENTIAL | bf16 -> bf16 | 128 | 32 KiB | 46.6 | 55.9 | 2.290 | 703 (703-703) | 0 | 3 |
| exp_vector_softmax_bf16 | Vector EXPONENTIAL | bf16 -> bf16 | 256 | 64 KiB | 68.4 | 82.1 | 3.118 | 958 (958-958) | 0 | 3 |
| exp_vector_softmax_bf16 | Vector EXPONENTIAL | bf16 -> bf16 | 512 | 128 KiB | 128.4 | 154.1 | 3.322 | 1021 (1021-1021) | 0 | 3 |
| exp_vector_softmax_bf16 | Vector EXPONENTIAL | bf16 -> bf16 | 1024 | 256 KiB | 235.1 | 282.1 | 3.630 | 1115 (1115-1115) | 0 | 3 |
| exp_vector_softmax_bf16 | Vector EXPONENTIAL | bf16 -> bf16 | 2048 | 512 KiB | 448.4 | 538.1 | 3.806 | 1169 (1169-1169) | 0 | 3 |
| exp_vector_softmax_bf16 | Vector EXPONENTIAL | bf16 -> bf16 | 4096 | 1 MiB | 875.1 | 1050.1 | 3.901 | 1198 (1198-1198) | 0 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | fp32 -> bf16 | 8 | 4 KiB | 25.9 | 31.1 | 0.257 | 158 (158-158) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | fp32 -> bf16 | 16 | 8 KiB | 32.6 | 39.1 | 0.409 | 251 (251-251) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | fp32 -> bf16 | 32 | 16 KiB | 45.9 | 55.1 | 0.581 | 357 (357-357) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | fp32 -> bf16 | 64 | 32 KiB | 72.6 | 87.1 | 0.735 | 451 (451-451) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | fp32 -> bf16 | 128 | 64 KiB | 125.9 | 151.1 | 0.847 | 520 (520-520) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | fp32 -> bf16 | 256 | 128 KiB | 232.6 | 279.1 | 0.917 | 564 (564-564) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | fp32 -> bf16 | 512 | 256 KiB | 445.9 | 535.1 | 0.957 | 588 (588-588) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | fp32 -> bf16 | 1024 | 512 KiB | 872.6 | 1047.1 | 0.978 | 601 (601-601) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | fp32 -> bf16 | 2048 | 1 MiB | 1725.9 | 2071.1 | 0.989 | 608 (608-608) | 1 | 3 |
| exp_scalar_softmax | Scalar ACTIVATE | fp32 -> bf16 | 4096 | 2 MiB | 3432.6 | 4119.1 | 0.994 | 611 (611-611) | 1 | 3 |

### Fit: II cycles = issue overhead + fdim x cycles per element (fdim >= 512)

| kernel | issue overhead (cycles) | cycles per element per partition | elements per cycle | streaming GB/s | max abs error (cycles) |
|---|---|---|---|---|---|
| exp_vector_softmax | 24.1 | 0.500 | 2.000 | 1229 | 0.0 |
| exp_vector_softmax_bf16 | 26.1 | 0.250 | 4.000 | 1229 | 0.0 |
| exp_scalar_softmax | 23.1 | 1.000 | 1.000 | 614 | 0.0 |

