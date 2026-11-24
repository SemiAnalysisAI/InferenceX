M = 128, N = 512, 8192 matmuls per loop iteration, 128 iterations per execution.

### TFLOP/s, utilization and scaling

Utilization is TFLOP/s over the TensorE peak at the profile's clock: 2 K M FLOPs per clock cycle per core, with K = 128 for bf16, 256 for fp8 double_row and 512 for MXFP8 and MXFP4, times the number of cores. The clock is the one the TensorE counters show in the solo run.

| profile | mode (kernel) | TensorE clock GHz | peak TFLOP/s per core | solo TFLOP/s | solo utilization | chip TFLOP/s | chip utilization | cores | chip / (cores x solo) | per-core TFLOP/s in chip run | chip TFLOP/s, first / last 10 s | window s | sysfs throttle mean, solo core | sysfs throttle mean, chip cores | profiler throttle level, last 1 s of chip probe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | 2.00 | 262.1 | 256.08 | 97.7% | 2048.6 | 97.7% | 8 | 1.0000 | 256.07-256.08 | 2048.5 / 2048.7 | 168 | 99.2% (min 50%) | 99.2% (min 50%) | 99.5% |
| 2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 2.40 | 314.6 | 296.97 | 94.4% | 1729.1 | 68.7% | 8 | 0.7278 | 216.05-216.22 | 1797.7 / 1722.3 | 60 | 99.2% (min 50%) | 73.1-73.2% (min 50%) | 73.3% |
| 2.4ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | 2.40 | 314.6 | 307.29 | 97.7% | 2159.1 | 85.8% | 8 | 0.8783 | 269.56-270.22 | 2202.9 / 2168.9 | 168 | 99.2% (min 50%) | 86.9-87.2% (min 50%) | 87.6% |

### Per-core TFLOP/s with all cores running

| profile | mode (kernel) | solo | core 0 | core 1 | core 2 | core 3 | core 4 | core 5 | core 6 | core 7 | slowest single execution |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | 256.08 | 256.08 | 256.07 | 256.08 | 256.07 | 256.08 | 256.08 | 256.08 | 256.08 | 255.25 |
| 2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 296.97 | 216.08 | 216.05 | 216.07 | 216.07 | 216.20 | 216.20 | 216.22 | 216.20 | 205.59 |
| 2.4ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | 307.29 | 269.56 | 269.58 | 269.57 | 269.56 | 270.21 | 270.22 | 270.21 | 270.20 | 259.78 |

### Chip TFLOP/s over time

Sum of the cores per 15-second bin, by the bin's start in seconds; `*` marks bins in which the first core pauses for its probe.

| profile | mode (kernel) | 0 | 15 | 30 | 45 | 60 | 75 | 90 | 105 | 120 | 135 | 150 | 165 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | 2049 | 2049 | 2049 | 2049 | 2049 | 2049 | 2049 | 2049 | 2049 | 2049 | 2049 | 2049* |
| 2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 1769 | 1708 | 1716 | 1722 | 1736* |  |  |  |  |  |  |  |
| 2.4ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | 2177 | 2116 | 2138 | 2152 | 2159 | 2164 | 2166 | 2167 | 2168 | 2170 | 2169 | 2183* |

### sysfs throttle statistic and TensorE counters by phase

| profile | mode (kernel) | phase | cores | sysfs cores | mean % (per-core average) | lowest window mean % | min % | max % | entries | distinct windows per core | TensorE column-streaming duty % | TensorE clock GHz (counters) | counter TFLOP/s, sum | attribution text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-74.5 | 50.2 | 50 | 89 | 1 | 1 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 42 | 97.7 | 2.000 | 256.1 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 1 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-88.6 | 50.2 | 50 | 100 | 1 | 1 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | chip | active | 0,1,2,3,4,5,6,7 | 99.2 | 99.2 | 50 | 100 | 1 | 154-155 | 97.7 | 2.000 | 2048.6 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1-8 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 8-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| 2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 17 | 94.4 | 2.400 | 297.0 | Source attribution unavailable |
| 2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 17-18 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| 2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-78.8 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| 2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | chip | active | 0,1,2,3,4,5,6,7 | 73.1-73.2 | 72.2 | 50 | 77 | 1 | 64-65 | 68.3 | 2.400 | 1719.4 | Source attribution unavailable |
| 2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-68.6 | 50.2 | 50 | 74 | 1 | 1-8 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| 2.4ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 1 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| 2.4ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 50 | 97.7 | 2.400 | 307.3 | Source attribution unavailable |
| 2.4ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 1 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| 2.4ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-96.1 | 50.2 | 50 | 100 | 1 | 1 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| 2.4ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | chip | active | 0,1,2,3,4,5,6,7 | 86.9-87.2 | 83.9 | 50 | 92 | 1 | 184-185 | 85.6-85.8 | 2.400 | 2156.4 | Source attribution unavailable |
| 2.4ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-74.5 | 50.2 | 50 | 89 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |

### Profiled probes

| profile | mode (kernel) | phase | runtime core | profiler neuroncore_idx | at s | opcode | source dtypes | tensor_instruction_type | FLOPs per MATMUL: profiler raw_flops / 2KMN | MATMULs captured / issued | probe ms | median II ns | probe TFLOP/s | level %, first 0.2 s | level %, last 1 s | level % every 0.5 s | throttle source: k range / n, share of probe x events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | solo | 0 | 4 | 48 | MATMUL_MX | mxfp4 | REGULAR | 67108864 / 67108864 | 4149942/8388608 | 2198 | 259.0 | 256.1 | 99.5 | 99.5 | 99 99 99 99 99 | activity_0 8-8/8 98.9% x1024; activity_1 4-4/8 1.0% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | chip | 0 | 4 | 169 | MATMUL_MX | mxfp4 | REGULAR | 67108864 / 67108864 | 4149942/8388608 | 2198 | 259.0 | 256.1 | 99.5 | 99.5 | 99 99 99 99 99 | activity_0 8-8/8 98.9% x1024; activity_1 4-4/8 1.0% x1024 |
| 2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | 0 | 4 | 18 | MATMUL_MX | fp8e4 | REGULAR | 67108864 / 67108864 | 4150449/8388608 | 1893 | 221.0 | 297.4 | 99.4 | 99.4 | 99 99 99 99 | activity_0 8-8/8 98.8% x1024; activity_1 4-4/8 1.2% x1024 |
| 2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | chip | 0 | 4 | 61 | MATMUL_MX | fp8e4 | REGULAR | 67108864 / 67108864 | 4150449/8388608 | 2594 | 291.0 | 217.0 | 84.1 | 73.3 | 85 70 73 73 73 73 | power_3 175-220/254 98.7% x1036; activity_1 4-4/8 0.9% x1024; temp_0 205-207/254 0.4% x8 |
| 2.4ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | solo | 0 | 4 | 48 | MATMUL_MX | mxfp4 | REGULAR | 67108864 / 67108864 | 4149942/8388608 | 1832 | 216.0 | 307.4 | 99.5 | 99.5 | 99 99 99 99 | activity_0 8-8/8 98.9% x1024; activity_1 4-4/8 1.0% x1024 |
| 2.4ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | chip | 0 | 4 | 168 | MATMUL_MX | mxfp4 | REGULAR | 67108864 / 67108864 | 4149942/8388608 | 2046 | 241.0 | 275.2 | 98.8 | 87.6 | 99 87 87 88 87 | power_3 219-252/254 91.4% x949; activity_0 8-8/8 4.9% x58; temp_0 239-247/254 2.8% x39; activity_1 4-4/8 1.0% x1024 |
