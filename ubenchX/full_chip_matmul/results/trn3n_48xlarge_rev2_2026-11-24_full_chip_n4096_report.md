M = 128, N = 4096, 1024 matmuls per loop iteration, 128 iterations per execution.

### TFLOP/s, utilization and scaling

Utilization is TFLOP/s over the TensorE peak at the profile's clock: 2 K M FLOPs per clock cycle per core, with K = 128 for bf16, 256 for fp8 double_row and 512 for MXFP8 and MXFP4, times the number of cores. The clock is the one the TensorE counters show in the solo run.

| profile | mode (kernel) | TensorE clock GHz | peak TFLOP/s per core | solo TFLOP/s | solo utilization | chip TFLOP/s | chip utilization | cores | chip / (cores x solo) | per-core TFLOP/s in chip run | chip TFLOP/s, first / last 10 s | window s | sysfs throttle mean, solo core | sysfs throttle mean, chip cores | profiler throttle level, last 1 s of chip probe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | 2.40 | 78.6 | 77.44 | 98.5% | 619.4 | 98.4% | 8 | 0.9998 | 77.40-77.44 | 619.4 / 619.5 | 168 | 99.2% (min 50%) | 99.2% (min 50%) | 99.3% |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | 2.40 | 157.3 | 154.38 | 98.2% | 1145.4 | 91.0% | 8 | 0.9274 | 143.09-143.25 | 1163.2 / 1151.1 | 168 | 98.8% (min 50%) | 91.6-91.8% (min 50%) | 92.1% |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | 2.40 | 314.6 | 290.31 | 92.3% | 1744.5 | 69.3% | 8 | 0.7511 | 218.02-218.09 | 1808.6 / 1747.1 | 168 | 98.5% (min 50%) | 74.3% (min 50%) | 74.5% |
| as_found | MXFP4 nc_matmul_mx (`mm_mxfp4`) | 2.40 | 314.6 | 309.94 | 98.5% | 2187.7 | 86.9% | 8 | 0.8823 | 272.88-274.04 | 2235.3 / 2198.0 | 168 | 99.2% (min 50%) | 87.2-87.6% (min 50%) | 87.8% |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | 2.00 | 65.5 | 64.51 | 98.4% | 516.1 | 98.4% | 8 | 1.0000 | 64.51 | 516.1 / 516.1 | 168 | 99.2% (min 50%) | 99.2% (min 50%) | 99.3% |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | 2.00 | 131.1 | 128.66 | 98.2% | 1029.2 | 98.2% | 8 | 1.0000 | 128.65-128.66 | 1029.1 / 1029.2 | 168 | 98.8% (min 50%) | 98.8% (min 50%) | 99.2% |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 2.00 | 262.1 | 241.63 | 92.2% | 1790.9 | 85.4% | 8 | 0.9265 | 223.49-224.25 | 1818.2 / 1801.3 | 168 | 98.5% (min 50%) | 90.9-91.1% (min 50%) | 91.4% |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | 2.00 | 262.1 | 258.21 | 98.5% | 2065.6 | 98.5% | 8 | 1.0000 | 258.19-258.21 | 2065.5 / 2065.6 | 168 | 99.2% (min 50%) | 99.2% (min 50%) | 99.4% |

### Per-core TFLOP/s with all cores running

| profile | mode (kernel) | solo | core 0 | core 1 | core 2 | core 3 | core 4 | core 5 | core 6 | core 7 | slowest single execution |
|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | 77.44 | 77.40 | 77.40 | 77.40 | 77.40 | 77.44 | 77.43 | 77.43 | 77.44 | 76.53 |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | 154.38 | 143.09 | 143.09 | 143.09 | 143.10 | 143.24 | 143.25 | 143.25 | 143.25 | 138.68 |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | 290.31 | 218.04 | 218.02 | 218.06 | 218.03 | 218.09 | 218.06 | 218.09 | 218.06 | 208.63 |
| as_found | MXFP4 nc_matmul_mx (`mm_mxfp4`) | 309.94 | 272.89 | 272.88 | 272.89 | 272.89 | 274.04 | 274.04 | 274.04 | 274.04 | 261.62 |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | 64.51 | 64.51 | 64.51 | 64.51 | 64.51 | 64.51 | 64.51 | 64.51 | 64.51 | 64.47 |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | 128.66 | 128.65 | 128.65 | 128.66 | 128.66 | 128.66 | 128.65 | 128.66 | 128.66 | 128.56 |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 241.63 | 223.49 | 223.50 | 223.50 | 223.50 | 224.22 | 224.22 | 224.25 | 224.23 | 213.95 |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | 258.21 | 258.19 | 258.21 | 258.21 | 258.19 | 258.20 | 258.19 | 258.21 | 258.21 | 258.04 |

### Chip TFLOP/s over time

Sum of the cores per 15-second bin, by the bin's start in seconds; `*` marks bins in which the first core pauses for its probe.

| profile | mode (kernel) | 0 | 15 | 30 | 45 | 60 | 75 | 90 | 105 | 120 | 135 | 150 | 165 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | 619 | 618 | 619 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 620 | 620* |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | 1152 | 1123 | 1134 | 1142 | 1146 | 1148 | 1150 | 1151 | 1150 | 1151 | 1151 | 1157* |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | 1784 | 1717 | 1728 | 1736 | 1741 | 1744 | 1746 | 1748 | 1747 | 1748 | 1747 | 1758* |
| as_found | MXFP4 nc_matmul_mx (`mm_mxfp4`) | 2209 | 2144 | 2165 | 2179 | 2187 | 2193 | 2195 | 2198 | 2197 | 2198 | 2198 | 2210* |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | 516 | 516 | 516 | 516 | 516 | 516 | 516 | 516 | 516 | 516 | 516 | 516* |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | 1029 | 1029 | 1029 | 1029 | 1029 | 1029 | 1029 | 1029 | 1029 | 1029 | 1029 | 1029* |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 1799 | 1754 | 1771 | 1784 | 1791 | 1797 | 1798 | 1801 | 1800 | 1801 | 1802 | 1811* |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | 2066 | 2066 | 2066 | 2066 | 2066 | 2066 | 2066 | 2066 | 2066 | 2066 | 2066 | 2066* |

### sysfs throttle statistic and TensorE counters by phase

| profile | mode (kernel) | phase | cores | sysfs cores | mean % (per-core average) | lowest window mean % | min % | max % | entries | distinct windows per core | TensorE column-streaming duty % | TensorE clock GHz (counters) | counter TFLOP/s, sum | attribution text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 8-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 51 | 98.5 | 2.400 | 77.4 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 50-51 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-77.6 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | chip | active | 0,1,2,3,4,5,6,7 | 99.2 | 98.0 | 50 | 100 | 1 | 184-185 | 98.4-98.5 | 2.400 | 619.4 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-75.7 | 50.2 | 50 | 100 | 1 | 1-8 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-75.7 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | active | 4 | 98.8 | 98.8 | 50 | 100 | 1 | 51 | 98.2 | 2.400 | 154.4 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 50-51 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-98.8 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | chip | active | 0,1,2,3,4,5,6,7 | 91.6-91.8 | 89.0 | 50 | 98 | 1 | 184-185 | 90.9-91.0 | 2.400 | 1144.3 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-76.1 | 50.2 | 50 | 93 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-76.1 | 50.2 | 50 | 93 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | solo | active | 4 | 98.5 | 98.4 | 50 | 100 | 1 | 51 | 92.3 | 2.400 | 290.3 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 50-51 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-98.8 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | chip | active | 0,1,2,3,4,5,6,7 | 74.3 | 73.3 | 50 | 78 | 1 | 184-185 | 69.2 | 2.400 | 1741.1 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-74.5 | 50.2 | 50 | 75 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP4 nc_matmul_mx (`mm_mxfp4`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-74.5 | 50.2 | 50 | 75 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP4 nc_matmul_mx (`mm_mxfp4`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 51 | 98.5 | 2.400 | 309.9 | Source attribution unavailable |
| as_found | MXFP4 nc_matmul_mx (`mm_mxfp4`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 50-51 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP4 nc_matmul_mx (`mm_mxfp4`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-96.5 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP4 nc_matmul_mx (`mm_mxfp4`) | chip | active | 0,1,2,3,4,5,6,7 | 87.2-87.6 | 83.9 | 50 | 89 | 1 | 185 | 86.6-87.0 | 2.400 | 2185.0 | Source attribution unavailable |
| as_found | MXFP4 nc_matmul_mx (`mm_mxfp4`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-87.5 | 50.2 | 50 | 88 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-87.5 | 50.2 | 50 | 88 | 1 | 1 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 43 | 98.4 | 2.000 | 64.5 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 1 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | chip | active | 0,1,2,3,4,5,6,7 | 99.2 | 99.2 | 50 | 100 | 1 | 154-155 | 98.4 | 2.000 | 516.1 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1-8 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1-8 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | active | 4 | 98.8 | 98.8 | 50 | 100 | 1 | 42 | 98.2 | 2.000 | 128.7 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 42-43 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-98.8 | 50.2 | 50 | 100 | 1 | 1-7 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | chip | active | 0,1,2,3,4,5,6,7 | 98.8 | 98.8 | 50 | 100 | 1 | 154-155 | 98.2 | 2.000 | 1029.2 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-98.8 | 50.2 | 50 | 100 | 1 | 1-7 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-98.8 | 50.2 | 50 | 100 | 1 | 1-8 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | active | 4 | 98.5 | 98.4 | 50 | 100 | 1 | 43 | 92.2 | 2.000 | 241.6 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 42-43 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-98.8 | 50.2 | 50 | 100 | 1 | 1-7 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | chip | active | 0,1,2,3,4,5,6,7 | 90.9-91.1 | 88.6 | 50 | 100 | 1 | 154-155 | 85.2-85.5 | 2.000 | 1789.2 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-91.4 | 50.2 | 50 | 93 | 1 | 1-7 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-91.4 | 50.2 | 50 | 93 | 1 | 1-8 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 43 | 98.5 | 2.000 | 258.2 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 42-43 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1-7 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | chip | active | 0,1,2,3,4,5,6,7 | 99.2 | 99.2 | 50 | 100 | 1 | 154 | 98.5 | 2.000 | 2065.6 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1-8 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |

### Profiled probes

| profile | mode (kernel) | phase | runtime core | profiler neuroncore_idx | at s | opcode | source dtypes | tensor_instruction_type | FLOPs per MATMUL: profiler raw_flops / 2KMN | MATMULs captured / issued | probe ms | median II ns | probe TFLOP/s | level %, first 0.2 s | level %, last 1 s | level % every 0.5 s | throttle source: k range / n, share of probe x events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | solo | 0 | 4 | 49 | MATMUL | bf16 | REGULAR | 134217728 / 134217728 | 1048576/1048576 | 1817 | 1709.0 | 77.5 | 99.3 | 99.3 | 99 99 99 99 | activity_0 8-8/8 98.7% x1024; activity_1 4-4/8 1.3% x1024 |
| as_found | bf16 nc_matmul (`mm_bf16`) | chip | 0 | 4 | 169 | MATMUL | bf16 | REGULAR | 134217728 / 134217728 | 1048576/1048576 | 1817 | 1709.0 | 77.5 | 99.3 | 99.3 | 99 99 99 99 | activity_0 8-8/8 98.7% x1024; activity_1 4-4/8 1.3% x1024 |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | 0 | 4 | 48 | MATMUL | fp8e4 | REGULAR | 268435456 / 268435456 | 1048576/1048576 | 1823 | 1709.0 | 154.4 | 99.2 | 99.2 | 99 99 99 99 | activity_0 8-8/8 98.4% x1024; activity_1 4-4/8 1.6% x1024 |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | chip | 0 | 4 | 169 | MATMUL | fp8e4 | REGULAR | 268435456 / 268435456 | 1048576/1048576 | 1930 | 1836.0 | 145.8 | 99.2 | 92.1 | 99 93 92 92 | power_3 235-253/254 87.8% x914; activity_0 8-8/8 10.7% x118; activity_1 4-4/8 1.5% x1024 |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | solo | 0 | 4 | 49 | MATMUL_MX | fp8e4 | REGULAR | 536870912 / 536870912 | 1048576/1048576 | 1954 | 1768.0 | 288.1 | 98.8 | 98.8 | 99 99 99 99 | activity_0 8-8/8 97.7% x1024; activity_1 4-4/8 2.3% x1024 |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | chip | 0 | 4 | 169 | MATMUL_MX | fp8e4 | REGULAR | 536870912 / 536870912 | 1048576/1048576 | 2553 | 2405.0 | 220.5 | 84.2 | 74.5 | 86 75 74 74 75 75 | power_3 190-220/254 98.2% x1037; activity_1 4-4/8 1.8% x1024 |
| as_found | MXFP4 nc_matmul_mx (`mm_mxfp4`) | solo | 0 | 4 | 49 | MATMUL_MX | mxfp4 | REGULAR | 536870912 / 536870912 | 1048576/1048576 | 1816 | 1709.0 | 310.0 | 99.4 | 99.4 | 99 99 99 99 | activity_0 8-8/8 98.7% x1024; activity_1 4-4/8 1.3% x1024 |
| as_found | MXFP4 nc_matmul_mx (`mm_mxfp4`) | chip | 0 | 4 | 169 | MATMUL_MX | mxfp4 | REGULAR | 536870912 / 536870912 | 1048576/1048576 | 2024 | 1934.0 | 278.1 | 98.1 | 87.8 | 99 88 87 88 88 | power_3 221-250/254 85.7% x885; temp_0 237-248/254 7.9% x100; activity_0 8-8/8 5.3% x62; activity_1 4-4/8 1.1% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | solo | 0 | 4 | 49 | MATMUL | bf16 | REGULAR | 134217728 / 134217728 | 1048576/1048576 | 2181 | 2051.0 | 64.5 | 99.3 | 99.3 | 99 99 99 99 99 | activity_0 8-8/8 98.7% x1024; activity_1 4-4/8 1.3% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | chip | 0 | 4 | 169 | MATMUL | bf16 | REGULAR | 134217728 / 134217728 | 1048576/1048576 | 2181 | 2051.0 | 64.5 | 99.3 | 99.3 | 99 99 99 99 99 | activity_0 8-8/8 98.7% x1024; activity_1 4-4/8 1.3% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | 0 | 4 | 49 | MATMUL | fp8e4 | REGULAR | 268435456 / 268435456 | 1048576/1048576 | 2187 | 2051.0 | 128.7 | 99.2 | 99.2 | 99 99 99 99 99 | activity_0 8-8/8 98.4% x1024; activity_1 4-4/8 1.6% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | chip | 0 | 4 | 169 | MATMUL | fp8e4 | REGULAR | 268435456 / 268435456 | 1048576/1048576 | 2187 | 2051.0 | 128.7 | 99.2 | 99.2 | 99 99 99 99 99 | activity_0 8-8/8 98.4% x1024; activity_1 4-4/8 1.6% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | 0 | 4 | 48 | MATMUL_MX | fp8e4 | REGULAR | 536870912 / 536870912 | 1048576/1048576 | 2329 | 2122.0 | 241.7 | 98.8 | 98.8 | 99 99 99 99 99 | activity_0 8-8/8 97.6% x1024; activity_1 4-4/8 2.4% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | chip | 0 | 4 | 169 | MATMUL_MX | fp8e4 | REGULAR | 536870912 / 536870912 | 1048576/1048576 | 2477 | 2309.0 | 227.3 | 98.8 | 91.4 | 99 93 91 92 91 | power_3 233-250/254 89.0% x940; activity_0 8-8/8 8.7% x98; activity_1 4-4/8 2.2% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | solo | 0 | 4 | 49 | MATMUL_MX | mxfp4 | REGULAR | 536870912 / 536870912 | 1048576/1048576 | 2180 | 2051.0 | 258.3 | 99.4 | 99.4 | 99 99 99 99 99 | activity_0 8-8/8 98.7% x1024; activity_1 4-4/8 1.3% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | MXFP4 nc_matmul_mx (`mm_mxfp4`) | chip | 0 | 4 | 169 | MATMUL_MX | mxfp4 | REGULAR | 536870912 / 536870912 | 1048576/1048576 | 2180 | 2051.0 | 258.3 | 99.4 | 99.4 | 99 99 99 99 99 | activity_0 8-8/8 98.7% x1024; activity_1 4-4/8 1.3% x1024 |
