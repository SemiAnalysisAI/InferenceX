M = 128, N = 512, 8192 matmuls per loop iteration, 128 iterations per execution.

### TFLOP/s, utilization and scaling

Utilization is TFLOP/s over the TensorE peak at the profile's clock: 2 K M FLOPs per clock cycle per core, with K = 128 for bf16, 256 for fp8 double_row and 512 for MXFP8, times the number of cores. The clock is the one the TensorE counters show in the solo run.

| profile | mode (kernel) | TensorE clock GHz | peak TFLOP/s per core | solo TFLOP/s | solo utilization | chip TFLOP/s | chip utilization | cores | chip / (cores x solo) | per-core TFLOP/s in chip run | chip TFLOP/s, first / last 10 s | window s | sysfs throttle mean, solo core | sysfs throttle mean, chip cores | profiler throttle level, last 1 s of chip probe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | 2.40 | 78.6 | 76.86 | 97.7% | 614.0 | 97.6% | 8 | 0.9986 | 76.74-76.75 | 614.8 / 614.8 | 168 | 99.2% (min 50%) | 99.1% (min 50%) | 99.5% |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | 2.40 | 157.3 | 153.72 | 97.7% | 1127.5 | 89.6% | 8 | 0.9169 | 140.92-140.97 | 1145.1 / 1132.9 | 168 | 99.2% (min 50%) | 90.9% (min 50%) | 91.7% |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | 2.40 | 314.6 | 297.08 | 94.4% | 1727.8 | 68.7% | 8 | 0.7270 | 215.94-216.00 | 1805.9 / 1727.5 | 168 | 99.2% (min 50%) | 73.6-73.7% (min 50%) | 73.9% |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | 2.00 | 65.5 | 64.06 | 97.7% | 512.4 | 97.7% | 8 | 0.9999 | 64.05-64.06 | 512.4 / 512.5 | 168 | 99.2% (min 50%) | 99.2% (min 50%) | 99.5% |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | 2.00 | 131.1 | 128.15 | 97.8% | 1025.1 | 97.8% | 8 | 0.9999 | 128.12-128.14 | 1025.0 / 1025.1 | 168 | 99.2% (min 50%) | 99.2% (min 50%) | 99.5% |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 2.00 | 262.1 | 248.35 | 94.7% | 1766.9 | 84.3% | 8 | 0.8893 | 220.71-221.02 | 1802.4 / 1772.6 | 168 | 99.2% (min 50%) | 90.6-90.9% (min 50%) | 91.2% |
| restored_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 2.40 | 314.6 | 297.06 | 94.4% | 1728.1 | 68.7% | 8 | 0.7272 | 215.93-216.09 | 1800.9 / 1721.5 | 60 | 99.2% (min 50%) | 73.0-73.1% (min 50%) | 73.3% |
| set_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 2.40 | 314.6 | 296.90 | 94.4% | 1727.0 | 68.6% | 8 | 0.7271 | 215.81-215.95 | 1799.0 / 1720.2 | 60 | 99.2% (min 50%) | 72.9-73.0% (min 50%) | 73.9% |

### Per-core TFLOP/s with all cores running

| profile | mode (kernel) | solo | core 0 | core 1 | core 2 | core 3 | core 4 | core 5 | core 6 | core 7 | slowest single execution |
|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | 76.86 | 76.75 | 76.75 | 76.74 | 76.75 | 76.74 | 76.75 | 76.75 | 76.75 | 75.22 |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | 153.72 | 140.93 | 140.92 | 140.92 | 140.92 | 140.96 | 140.96 | 140.96 | 140.97 | 137.25 |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | 297.08 | 215.94 | 215.94 | 215.94 | 215.95 | 216.00 | 215.99 | 216.00 | 216.00 | 210.09 |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | 64.06 | 64.06 | 64.06 | 64.05 | 64.06 | 64.05 | 64.05 | 64.06 | 64.06 | 64.01 |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | 128.15 | 128.14 | 128.14 | 128.12 | 128.12 | 128.12 | 128.13 | 128.14 | 128.14 | 127.69 |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 248.35 | 220.73 | 220.73 | 220.71 | 220.71 | 221.02 | 221.01 | 221.01 | 221.00 | 213.40 |
| restored_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 297.06 | 215.94 | 215.95 | 215.93 | 215.94 | 216.05 | 216.07 | 216.08 | 216.09 | 210.17 |
| set_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 296.90 | 215.81 | 215.81 | 215.81 | 215.83 | 215.94 | 215.95 | 215.95 | 215.94 | 209.92 |

### Chip TFLOP/s over time

Sum of the cores per 15-second bin, by the bin's start in seconds; `*` marks bins in which the first core pauses for its probe.

| profile | mode (kernel) | 0 | 15 | 30 | 45 | 60 | 75 | 90 | 105 | 120 | 135 | 150 | 165 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | 614 | 608 | 613 | 615 | 615 | 615 | 615 | 615 | 615 | 615 | 615 | 615* |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | 1134 | 1105 | 1116 | 1124 | 1128 | 1130 | 1132 | 1133 | 1134 | 1133 | 1133 | 1140* |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | 1779 | 1707 | 1714 | 1718 | 1724 | 1725 | 1727 | 1727 | 1727 | 1727 | 1728 | 1741* |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | 512 | 512 | 512 | 512 | 512 | 512 | 512 | 512 | 512 | 512 | 512 | 512* |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | 1025 | 1025 | 1025 | 1025 | 1025 | 1025 | 1025 | 1025 | 1025 | 1025 | 1025 | 1025* |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 1781 | 1736 | 1753 | 1761 | 1767 | 1770 | 1771 | 1773 | 1773 | 1775 | 1773 | 1787* |
| restored_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 1774 | 1704 | 1712 | 1721 | 1739* |  |  |  |  |  |  |  |
| set_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | 1771 | 1704 | 1712 | 1719 | 1736* |  |  |  |  |  |  |  |

### sysfs throttle statistic and TensorE counters by phase

| profile | mode (kernel) | phase | cores | sysfs cores | mean % (per-core average) | lowest window mean % | min % | max % | entries | distinct windows per core | TensorE column-streaming duty % | TensorE clock GHz (counters) | counter TFLOP/s, sum | attribution text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-65.1 | 50.2 | 50 | 86 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 50 | 97.7 | 2.400 | 76.9 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 50-51 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-87.5 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | chip | active | 0,1,2,3,4,5,6,7 | 99.1 | 97.6 | 50 | 100 | 1 | 184-185 | 97.6 | 2.400 | 614.0 | Source attribution unavailable |
| as_found | bf16 nc_matmul (`mm_bf16`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-92.9 | 50.2 | 50 | 100 | 1 | 1-8 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-92.9 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 51 | 97.7 | 2.400 | 153.7 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 50-51 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-94.9 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | chip | active | 0,1,2,3,4,5,6,7 | 90.9 | 88.6 | 50 | 100 | 1 | 184-185 | 89.5 | 2.400 | 1126.4 | Source attribution unavailable |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-66.7 | 50.2 | 50 | 93 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-66.7 | 50.2 | 50 | 93 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 50 | 94.4 | 2.400 | 297.1 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 50-51 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-74.9 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | chip | active | 0,1,2,3,4,5,6,7 | 73.6-73.7 | 71.8 | 50 | 76 | 1 | 184-185 | 68.5 | 2.400 | 1724.1 | Source attribution unavailable |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-73.7 | 50.2 | 50 | 75 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-72.5 | 50.2 | 50 | 75 | 1 | 1 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 43 | 97.7 | 2.000 | 64.1 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 1 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | chip | active | 0,1,2,3,4,5,6,7 | 99.2 | 99.2 | 50 | 100 | 1 | 154 | 97.7 | 2.000 | 512.4 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1-8 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1-8 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 42 | 97.8 | 2.000 | 128.1 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 42-43 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1-8 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | chip | active | 0,1,2,3,4,5,6,7 | 99.2 | 99.2 | 50 | 100 | 1 | 154-155 | 97.7-97.8 | 2.000 | 1025.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1-7 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1-8 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 42 | 94.7 | 2.000 | 248.4 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 42-43 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1-8 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | chip | active | 0,1,2,3,4,5,6,7 | 90.6-90.9 | 86.7 | 50 | 95 | 1 | 154-155 | 84.1-84.2 | 2.000 | 1764.9 | Source attribution unavailable |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-91.0 | 50.2 | 50 | 92 | 1 | 1-7 | 0.0 | 2.000 | 0.0 | Source attribution unavailable |
| restored_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 8-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| restored_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 17 | 94.4 | 2.400 | 297.1 | Source attribution unavailable |
| restored_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 17-18 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| restored_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-99.2 | 50.2 | 50 | 100 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| restored_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | chip | active | 0,1,2,3,4,5,6,7 | 73.0-73.1 | 71.8 | 50 | 76 | 1 | 64 | 68.3 | 2.400 | 1718.0 | Source attribution unavailable |
| restored_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-66.3 | 50.2 | 50 | 74 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| set_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_before | idle | 0,1,2,3,4,5,6,7 | 50.2-73.7 | 50.2 | 50 | 75 | 1 | 1 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| set_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | active | 4 | 99.2 | 99.2 | 50 | 100 | 1 | 17 | 94.4 | 2.400 | 297.0 | Source attribution unavailable |
| set_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | idle | 0,1,2,3,5,6,7 | 50.2 | 50.2 | 50 | 50 | 1 | 1 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| set_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_mid | idle | 0,1,2,3,4,5,6,7 | 50.2-74.1 | 50.2 | 50 | 100 | 1 | 1 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |
| set_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | chip | active | 0,1,2,3,4,5,6,7 | 72.9-73.0 | 72.2 | 50 | 76 | 1 | 64-65 | 68.2 | 2.400 | 1716.9 | Source attribution unavailable |
| set_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | idle_after | idle | 0,1,2,3,4,5,6,7 | 50.2-72.5 | 50.2 | 50 | 75 | 1 | 1-9 | 0.0 | 2.400 | 0.0 | Source attribution unavailable |

### Profiled probes

| profile | mode (kernel) | phase | runtime core | profiler neuroncore_idx | at s | opcode | source dtypes | tensor_instruction_type | FLOPs per MATMUL: profiler raw_flops / 2KMN | MATMULs captured / issued | probe ms | median II ns | probe TFLOP/s | level %, first 0.2 s | level %, last 1 s | level % every 0.5 s | throttle source: k range / n, share of probe x events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as_found | bf16 nc_matmul (`mm_bf16`) | solo | 0 | 4 | 48 | MATMUL | bf16 | REGULAR | 16777216 / 16777216 | 4151462/8388608 | 1831 | 216.0 | 76.9 | 99.5 | 99.5 | 100 99 99 100 | activity_0 8-8/8 99.0% x1024; activity_1 4-4/8 1.0% x1024 |
| as_found | bf16 nc_matmul (`mm_bf16`) | chip | 0 | 4 | 168 | MATMUL | bf16 | REGULAR | 16777216 / 16777216 | 4151462/8388608 | 1831 | 216.0 | 76.9 | 99.5 | 99.5 | 100 99 99 100 | activity_0 8-8/8 99.0% x1024; activity_1 4-4/8 1.0% x1024 |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | 0 | 4 | 48 | MATMUL | fp8e4 | REGULAR | 33554432 / 33554432 | 4147920/8388608 | 1831 | 216.0 | 153.8 | 99.5 | 99.5 | 100 99 99 100 | activity_0 8-8/8 99.0% x1024; activity_1 4-4/8 1.0% x1024 |
| as_found | fp8 double_row nc_matmul (`mm_fp8dr`) | chip | 0 | 4 | 168 | MATMUL | fp8e4 | REGULAR | 33554432 / 33554432 | 4147920/8388608 | 1956 | 229.0 | 143.9 | 99.5 | 91.7 | 100 92 91 92 | power_3 232-249/254 87.1% x902; activity_0 8-8/8 12.0% x133; activity_1 4-4/8 1.0% x1024 |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | solo | 0 | 4 | 48 | MATMUL_MX | fp8e4 | REGULAR | 67108864 / 67108864 | 4150449/8388608 | 1892 | 221.0 | 297.6 | 99.4 | 99.4 | 99 99 99 99 | activity_0 8-8/8 98.8% x1024; activity_1 4-4/8 1.2% x1024 |
| as_found | MXFP8 nc_matmul_mx (`mm_mx`) | chip | 0 | 4 | 169 | MATMUL_MX | fp8e4 | REGULAR | 67108864 / 67108864 | 4150449/8388608 | 2584 | 290.0 | 217.9 | 83.2 | 73.9 | 84 73 73 73 74 74 | power_3 184-214/254 99.1% x1039; activity_1 4-4/8 0.9% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | solo | 0 | 4 | 48 | MATMUL | bf16 | REGULAR | 16777216 / 16777216 | 4151462/8388608 | 2196 | 259.0 | 64.1 | 99.5 | 99.5 | 100 100 100 100 100 | activity_0 8-8/8 99.0% x1024; activity_1 4-4/8 1.0% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | bf16 nc_matmul (`mm_bf16`) | chip | 0 | 4 | 168 | MATMUL | bf16 | REGULAR | 16777216 / 16777216 | 4151462/8388608 | 2196 | 259.0 | 64.1 | 99.5 | 99.5 | 100 100 100 100 100 | activity_0 8-8/8 99.0% x1024; activity_1 4-4/8 1.0% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | solo | 0 | 4 | 48 | MATMUL | fp8e4 | REGULAR | 33554432 / 33554432 | 4147920/8388608 | 2196 | 259.0 | 128.2 | 99.5 | 99.5 | 100 100 100 100 100 | activity_0 8-8/8 99.0% x1024; activity_1 4-4/8 1.0% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | fp8 double_row nc_matmul (`mm_fp8dr`) | chip | 0 | 4 | 168 | MATMUL | fp8e4 | REGULAR | 33554432 / 33554432 | 4147920/8388608 | 2196 | 259.0 | 128.2 | 99.5 | 99.5 | 100 100 100 100 100 | activity_0 8-8/8 99.0% x1024; activity_1 4-4/8 1.0% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | 0 | 4 | 49 | MATMUL_MX | fp8e4 | REGULAR | 67108864 / 67108864 | 4150449/8388608 | 2266 | 265.0 | 248.4 | 99.4 | 99.4 | 99 99 99 99 99 | activity_0 8-8/8 98.9% x1024; activity_1 4-4/8 1.1% x1024 |
| 2.0ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | chip | 0 | 4 | 168 | MATMUL_MX | fp8e4 | REGULAR | 67108864 / 67108864 | 4150449/8388608 | 2511 | 292.0 | 224.2 | 99.2 | 91.2 | 99 92 91 91 91 91 | power_3 231-245/254 91.6% x952; activity_0 8-8/8 6.9% x79; activity_1 4-4/8 1.0% x1024; temp_0 251-253/254 0.4% x8 |
| restored_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | 0 | 4 | 18 | MATMUL_MX | fp8e4 | REGULAR | 67108864 / 67108864 | 4150449/8388608 | 1895 | 221.0 | 297.0 | 99.4 | 99.4 | 99 99 99 99 | activity_0 8-8/8 98.8% x1024; activity_1 4-4/8 1.2% x1024 |
| restored_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | chip | 0 | 4 | 61 | MATMUL_MX | fp8e4 | REGULAR | 67108864 / 67108864 | 4150449/8388608 | 2580 | 290.0 | 218.2 | 85.1 | 73.3 | 86 72 73 73 73 73 | power_3 184-220/254 99.1% x1039; activity_1 4-4/8 0.9% x1024 |
| set_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | solo | 0 | 4 | 18 | MATMUL_MX | fp8e4 | REGULAR | 67108864 / 67108864 | 4150449/8388608 | 1897 | 221.0 | 296.8 | 99.4 | 99.4 | 99 99 99 99 | activity_0 8-8/8 98.8% x1024; activity_1 4-4/8 1.2% x1024 |
| set_2.4ghz_tpb_1.2ghz_soc | MXFP8 nc_matmul_mx (`mm_mx`) | chip | 0 | 4 | 61 | MATMUL_MX | fp8e4 | REGULAR | 67108864 / 67108864 | 4150449/8388608 | 2585 | 292.0 | 217.8 | 83.6 | 73.9 | 85 72 73 74 74 73 | power_3 184-216/254 99.1% x1039; activity_1 4-4/8 0.9% x1024 |
