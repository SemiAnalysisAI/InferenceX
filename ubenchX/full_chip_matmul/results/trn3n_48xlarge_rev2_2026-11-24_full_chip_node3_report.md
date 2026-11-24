M = 128, N = 512, 8192 matmuls per loop iteration, 128 iterations per execution.

### TFLOP/s, utilization and scaling

Utilization is TFLOP/s over the TensorE peak at the profile's clock: 2 K M FLOPs per clock cycle per core, with K = 128 for bf16, 256 for fp8 double_row and 512 for MXFP8, times the number of cores. The clock is the one the TensorE counters show in the solo run.

| profile | mode (kernel) | TensorE clock GHz | peak TFLOP/s per core | solo TFLOP/s | solo utilization | chip TFLOP/s | chip utilization | cores | chip / (cores x solo) | per-core TFLOP/s in chip run | chip TFLOP/s, first / last 10 s | window s | sysfs throttle mean, solo core | sysfs throttle mean, chip cores | profiler throttle level, last 1 s of chip probe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | 2.40 | 78.6 | 76.85 | 97.7% | 563.9 | 89.6% | 8 | 0.9171 | 70.48-70.49 | 580.8 / 565.4 | 90 | 99.2% (min 50%) | 90.9% (min 50%) | 91.3% |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | 2.40 | 157.3 | 153.71 | 97.7% | 1032.7 | 82.1% | 8 | 0.8398 | 129.07-129.10 | 1077.2 / 1030.6 | 90 | 99.2% (min 50%) | 83.1% (min 50%) | 83.3% |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | 2.40 | 314.6 | 267.68 | 85.1% | 1592.0 | 63.3% | 8 | 0.7434 | 197.86-200.14 | 1670.0 / 1584.4 | 90 | 92.2% (min 50%) | 67.0-67.7% (min 50%) | 66.9% |

### Per-core TFLOP/s with all cores running

| profile | mode (kernel) | solo | core 0 | core 1 | core 2 | core 3 | core 4 | core 5 | core 6 | core 7 | slowest single execution |
|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | 76.85 | 70.49 | 70.49 | 70.49 | 70.49 | 70.48 | 70.49 | 70.49 | 70.49 | 69.28 |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | 153.71 | 129.08 | 129.08 | 129.08 | 129.07 | 129.09 | 129.10 | 129.10 | 129.09 | 126.92 |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | 267.68 | 197.86 | 197.88 | 197.86 | 197.88 | 200.12 | 200.14 | 200.12 | 200.12 | 183.55 |

### Chip TFLOP/s over time

Sum of the cores per 15-second bin, by the bin's start in seconds; `*` marks bins in which the first core pauses for its probe.

| profile | mode (kernel) | 0 | 15 | 30 | 45 | 60 | 75 | 90 |
|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | 575 | 556 | 560 | 563 | 564 | 565 | 569* |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | 1063 | 1021 | 1024 | 1028 | 1030 | 1031 | 1040* |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | 1640 | 1582 | 1583 | 1582 | 1580 | 1583 | 1600* |

### sysfs throttle statistic and TensorE counters by phase

| profile | mode (kernel) | phase | cores | sysfs cores | mean % (per-core average) | lowest window mean % | min % | max % | entries | distinct windows per core | TensorE column-streaming duty % | TensorE clock GHz (counters) | counter TFLOP/s, sum | attribution text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-72.5 | 50.2 | 50 | 84 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 17 | 97.7 | 2.400 | 76.9 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 17-18 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-69.4 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | chip | active | 0,1,2,3,4,5,6,7 | 90.9 | 89.4 | 50 | 100 | 1 | 97-98 | 89.4 | 2.400 | 562.7 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-73.3 | 50.2 | 50 | 92 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-58.8 | 50.2 | 50 | 67 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 17 | 97.7 | 2.400 | 153.7 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 17 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-86.7 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | chip | active | 0,1,2,3,4,5,6,7 | 83.1 | 82.0 | 50 | 91 | 1 | 97-98 | 81.8 | 2.400 | 1029.2 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-72.5 | 50.2 | 50 | 84 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 8-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | solo | active | 4 | 92.2 | 92.2 | 50 | 93 | 1 | 17 | 84.5 | 2.400 | 265.8 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 17-18 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-77.3 | 50.2 | 50 | 92 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | chip | active | 0,1,2,3,4,5,6,7 | 67.0-67.7 | 62.0 | 50 | 71 | 1 | 97-98 | 62.7-63.3 | 2.400 | 1585.2 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-58.8 | 50.2 | 50 | 67 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |

### Profiled probes

| profile | mode (kernel) | phase | runtime core | profiler neuroncore_idx | at s | opcode | source dtypes | tensor_instruction_type | FLOPs per MATMUL: profiler raw_flops / 2KMN | MATMULs captured / issued | probe ms | median II ns | probe TFLOP/s | level %, first 0.2 s | level %, last 1 s | level % every 0.5 s | throttle source: k range / n, share of probe x events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | solo | 0 | 4 | 18 | MATMUL | bf16 | REGULAR | 16777216 / 16777216 | 4151462/8388608 | 1831 | 216.0 | 76.9 | 99.5 | 99.5 | 100 99 99 100 | activity_0 8-8/8 99.0% x1024; activity_1 4-4/8 1.0% x1024 |
| as_found | bf16 nc_matmul (`mm_bf16`) | chip | 0 | 4 | 90 | MATMUL | bf16 | REGULAR | 16777216 / 16777216 | 4151462/8388608 | 1962 | 227.0 | 71.7 | 99.5 | 91.3 | 100 92 91 91 | power_3 231-241/254 80.2% x824; activity_0 8-8/8 11.1% x124; temp_0 245-253/254 7.7% x92; activity_1 4-4/8 1.0% x1024 |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | 0 | 4 | 18 | MATMUL | fp8e4 | REGULAR | 33554432 / 33554432 | 4147920/8388608 | 1831 | 216.0 | 153.8 | 99.5 | 99.5 | 100 99 99 100 | activity_0 8-8/8 99.0% x1024; activity_1 4-4/8 1.0% x1024 |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | chip | 0 | 4 | 91 | MATMUL | fp8e4 | REGULAR | 33554432 / 33554432 | 4147920/8388608 | 2148 | 243.0 | 131.0 | 92.9 | 83.3 | 94 87 82 84 84 | power_3 208-247/254 81.5% x845; temp_0 222-247/254 17.6% x229; activity_1 4-4/8 0.8% x1024 |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | solo | 0 | 4 | 19 | MATMUL_MX | fp8e4 | REGULAR | 67108864 / 67108864 | 4150449/8388608 | 2032 | 221.0 | 277.0 | 99.4 | 91.5 | 99 99 91 92 92 | temp_0 229-253/254 65.7% x754; activity_0 8-8/8 33.2% x370; activity_1 4-4/8 1.1% x1024 |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | chip | 0 | 4 | 91 | MATMUL_MX | fp8e4 | REGULAR | 67108864 / 67108864 | 4150449/8388608 | 2813 | 343.0 | 200.1 | 74.6 | 66.9 | 78 67 67 67 67 67 | power_3 169-199/254 91.5% x953; temp_0 180-199/254 7.7% x108; activity_1 4-4/8 0.8% x1024 |
