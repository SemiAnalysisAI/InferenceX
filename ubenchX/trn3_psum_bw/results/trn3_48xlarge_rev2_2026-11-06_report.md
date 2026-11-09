# Trainium3 rev2 PSUM -> SBUF tensor_copy (trn3.48xlarge, NeuronCore 0, LNC=1)

## Sweep, each engine alone (`trn3_48xlarge_rev2_2026-11-05_psum.csv`, repeats r0 and r1)

### PSUM -> SBUF tensor_copy, one NeuronCore (LNC=1)

| kernel | pdim | fdim (fp32) | PSUM row | per copy | PSUM read GB/s (median, min-max) | SBUF write GB/s | ns per copy | instr duration ns | wait ns | traces | whole-kernel GB/s (lower bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy_vector | 128 | 64 | 256 B | 32 KiB | 497 (497-497) | 497 | 66 | 159 | 1 | 6 | 112 |
| copy_vector | 128 | 128 | 512 B | 64 KiB | 550 (550-550) | 550 | 119 | 213 | 1 | 6 | 253 |
| copy_vector | 128 | 256 | 1 KiB | 128 KiB | 580 (580-580) | 580 | 226 | 319 | 1 | 6 | 406 |
| copy_vector | 128 | 512 | 2 KiB | 256 KiB | 597 (597-597) | 597 | 439 | 532 | 1 | 6 | 507 |
| copy_vector | 128 | 1024 | 4 KiB | 512 KiB | 605 (605-605) | 605 | 866 | 959 | 1 | 6 | 564 |
| copy_vector | 128 | 2048 | 8 KiB | 1 MiB | 610 (610-610) | 610 | 1719 | 1812 | 1 | 6 | 589 |
| copy_vector | 128 | 4096 | 16 KiB | 2 MiB | 612 (612-612) | 612 | 3426 | 3519 | 3426 | 6 | 601 |
| copy_vector | 64 | 512 | 2 KiB | 128 KiB | 298 (298-298) | 298 | 439 | 532 | 1 | 6 | 254 |
| copy_vector | 32 | 512 | 2 KiB | 64 KiB | 149 (149-149) | 149 | 439 | 532 | 1 | 6 | 127 |
| copy_scalar | 128 | 64 | 256 B | 32 KiB | 545 (545-545) | 545 | 60 | 168 | 1 | 6 | 114 |
| copy_scalar | 128 | 128 | 512 B | 64 KiB | 577 (577-577) | 577 | 113 | 222 | 1 | 6 | 272 |
| copy_scalar | 128 | 256 | 1 KiB | 128 KiB | 595 (595-595) | 595 | 220 | 328 | 1 | 6 | 432 |
| copy_scalar | 128 | 512 | 2 KiB | 256 KiB | 605 (605-605) | 605 | 433 | 542 | 1 | 6 | 515 |
| copy_scalar | 128 | 1024 | 4 KiB | 512 KiB | 610 (610-610) | 610 | 860 | 968 | 1 | 6 | 572 |
| copy_scalar | 128 | 2048 | 8 KiB | 1 MiB | 612 (612-612) | 612 | 1713 | 1822 | 1 | 6 | 597 |
| copy_scalar | 128 | 4096 | 16 KiB | 2 MiB | 613 (613-613) | 613 | 3420 | 3528 | 3420 | 6 | 603 |

### Fit: ns per copy = fixed + bytes / streaming rate (pdim=128)

| kernel | fixed ns per copy | streaming PSUM read GB/s | max abs error ns |
|---|---|---|---|
| copy_vector | 13 | 614 | 0 |
| copy_scalar | 7 | 614 | 0 |

### Run-to-run (median GB/s per sweep repeat, pdim=128)

| kernel | fdim | r0 GB/s | r1 GB/s | spread |
|---|---|---|---|---|
| copy_vector | 64 | 497 | 497 | 0.0% |
| copy_vector | 128 | 550 | 550 | 0.0% |
| copy_vector | 256 | 580 | 580 | 0.0% |
| copy_vector | 512 | 597 | 597 | 0.0% |
| copy_vector | 1024 | 605 | 605 | 0.0% |
| copy_vector | 2048 | 610 | 610 | 0.0% |
| copy_vector | 4096 | 612 | 612 | 0.0% |
| copy_scalar | 64 | 545 | 545 | 0.0% |
| copy_scalar | 128 | 577 | 577 | 0.0% |
| copy_scalar | 256 | 595 | 595 | 0.0% |
| copy_scalar | 512 | 605 | 605 | 0.0% |
| copy_scalar | 1024 | 610 | 610 | 0.0% |
| copy_scalar | 2048 | 612 | 612 | 0.0% |
| copy_scalar | 4096 | 613 | 613 | 0.0% |

## Fine sweep, fdim 256 to 4096 in steps of 256 (`trn3_48xlarge_rev2_2026-11-05_psum_fine.csv`)

### PSUM -> SBUF tensor_copy, one NeuronCore (LNC=1)

| kernel | pdim | fdim (fp32) | PSUM row | per copy | PSUM read GB/s (median, min-max) | SBUF write GB/s | ns per copy | instr duration ns | wait ns | traces | whole-kernel GB/s (lower bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| copy_vector | 128 | 256 | 1 KiB | 128 KiB | 580 (580-580) | 580 | 226 | 319 | 1 | 6 | 405 |
| copy_vector | 128 | 512 | 2 KiB | 256 KiB | 597 (597-597) | 597 | 439 | 532 | 1 | 6 | 507 |
| copy_vector | 128 | 768 | 3 KiB | 384 KiB | 603 (603-603) | 603 | 653 | 746 | 1 | 6 | 546 |
| copy_vector | 128 | 1024 | 4 KiB | 512 KiB | 605 (605-605) | 605 | 866 | 959 | 1 | 6 | 564 |
| copy_vector | 128 | 1280 | 5 KiB | 640 KiB | 607 (607-607) | 607 | 1079 | 1172 | 1 | 6 | 574 |
| copy_vector | 128 | 1536 | 6 KiB | 768 KiB | 608 (608-608) | 608 | 1293 | 1386 | 1 | 6 | 580 |
| copy_vector | 128 | 1792 | 7 KiB | 896 KiB | 609 (609-609) | 609 | 1506 | 1599 | 1 | 6 | 586 |
| copy_vector | 128 | 2048 | 8 KiB | 1 MiB | 610 (610-610) | 610 | 1719 | 1812 | 1 | 6 | 589 |
| copy_vector | 128 | 2304 | 9 KiB | 1.125 MiB | 610 (610-610) | 610 | 1933 | 2026 | 1 | 6 | 592 |
| copy_vector | 128 | 2560 | 10 KiB | 1.25 MiB | 611 (611-611) | 611 | 2146 | 2239 | 1 | 6 | 594 |
| copy_vector | 128 | 2816 | 11 KiB | 1.375 MiB | 611 (611-611) | 611 | 2359 | 2452 | 1 | 6 | 596 |
| copy_vector | 128 | 3072 | 12 KiB | 1.5 MiB | 611 (611-611) | 611 | 2573 | 2666 | 2573 | 6 | 597 |
| copy_vector | 128 | 3328 | 13 KiB | 1.625 MiB | 612 (612-612) | 612 | 2786 | 2879 | 2786 | 6 | 598 |
| copy_vector | 128 | 3584 | 14 KiB | 1.75 MiB | 612 (612-612) | 612 | 2999 | 3092 | 2999 | 6 | 600 |
| copy_vector | 128 | 3840 | 15 KiB | 1.875 MiB | 612 (612-612) | 612 | 3213 | 3306 | 3212 | 6 | 601 |
| copy_vector | 128 | 4096 | 16 KiB | 2 MiB | 612 (612-612) | 612 | 3426 | 3519 | 3426 | 6 | 601 |
| copy_vector | 64 | 512 | 2 KiB | 128 KiB | 298 (298-298) | 298 | 439 | 532 | 1 | 6 | 254 |
| copy_vector | 32 | 512 | 2 KiB | 64 KiB | 149 (149-149) | 149 | 439 | 532 | 1 | 6 | 127 |
| copy_scalar | 128 | 256 | 1 KiB | 128 KiB | 595 (595-595) | 595 | 220 | 328 | 1 | 6 | 431 |
| copy_scalar | 128 | 512 | 2 KiB | 256 KiB | 605 (605-605) | 605 | 433 | 542 | 1 | 6 | 515 |
| copy_scalar | 128 | 768 | 3 KiB | 384 KiB | 608 (608-608) | 608 | 647 | 755 | 1 | 6 | 564 |
| copy_scalar | 128 | 1024 | 4 KiB | 512 KiB | 610 (610-610) | 610 | 860 | 968 | 1 | 6 | 572 |
| copy_scalar | 128 | 1280 | 5 KiB | 640 KiB | 610 (610-610) | 610 | 1073 | 1182 | 1 | 6 | 589 |
| copy_scalar | 128 | 1536 | 6 KiB | 768 KiB | 611 (611-611) | 611 | 1287 | 1395 | 1 | 6 | 583 |
| copy_scalar | 128 | 1792 | 7 KiB | 896 KiB | 612 (612-612) | 612 | 1500 | 1608 | 1 | 6 | 593 |
| copy_scalar | 128 | 2048 | 8 KiB | 1 MiB | 612 (612-612) | 612 | 1713 | 1822 | 1 | 6 | 597 |
| copy_scalar | 128 | 2304 | 9 KiB | 1.125 MiB | 612 (612-612) | 612 | 1927 | 2035 | 1 | 6 | 596 |
| copy_scalar | 128 | 2560 | 10 KiB | 1.25 MiB | 612 (612-612) | 612 | 2140 | 2248 | 1 | 6 | 604 |
| copy_scalar | 128 | 2816 | 11 KiB | 1.375 MiB | 613 (613-613) | 613 | 2353 | 2462 | 1 | 6 | 599 |
| copy_scalar | 128 | 3072 | 12 KiB | 1.5 MiB | 613 (613-613) | 613 | 2567 | 2675 | 2567 | 6 | 601 |
| copy_scalar | 128 | 3328 | 13 KiB | 1.625 MiB | 613 (613-613) | 613 | 2780 | 2888 | 2780 | 6 | 600 |
| copy_scalar | 128 | 3584 | 14 KiB | 1.75 MiB | 613 (613-613) | 613 | 2993 | 3102 | 2993 | 6 | 606 |
| copy_scalar | 128 | 3840 | 15 KiB | 1.875 MiB | 613 (613-613) | 613 | 3207 | 3315 | 3207 | 6 | 604 |
| copy_scalar | 128 | 4096 | 16 KiB | 2 MiB | 613 (613-613) | 613 | 3420 | 3528 | 3420 | 6 | 603 |

### Fit: ns per copy = fixed + bytes / streaming rate (pdim=128)

| kernel | fixed ns per copy | streaming PSUM read GB/s | max abs error ns |
|---|---|---|---|
| copy_vector | 13 | 614 | 0 |
| copy_scalar | 7 | 614 | 0 |

### Run-to-run (median GB/s per sweep repeat, pdim=128)

| kernel | fdim | fine_r0 GB/s | fine_r1 GB/s | spread |
|---|---|---|---|---|
| copy_vector | 256 | 580 | 580 | 0.0% |
| copy_vector | 512 | 597 | 597 | 0.0% |
| copy_vector | 768 | 603 | 603 | 0.0% |
| copy_vector | 1024 | 605 | 605 | 0.0% |
| copy_vector | 1280 | 607 | 607 | 0.0% |
| copy_vector | 1536 | 608 | 608 | 0.0% |
| copy_vector | 1792 | 609 | 609 | 0.0% |
| copy_vector | 2048 | 610 | 610 | 0.0% |
| copy_vector | 2304 | 610 | 610 | 0.0% |
| copy_vector | 2560 | 611 | 611 | 0.0% |
| copy_vector | 2816 | 611 | 611 | 0.0% |
| copy_vector | 3072 | 611 | 611 | 0.0% |
| copy_vector | 3328 | 612 | 612 | 0.0% |
| copy_vector | 3584 | 612 | 612 | 0.0% |
| copy_vector | 3840 | 612 | 612 | 0.0% |
| copy_vector | 4096 | 612 | 612 | 0.0% |
| copy_scalar | 256 | 595 | 595 | 0.0% |
| copy_scalar | 512 | 605 | 605 | 0.0% |
| copy_scalar | 768 | 608 | 608 | 0.0% |
| copy_scalar | 1024 | 610 | 610 | 0.0% |
| copy_scalar | 1280 | 610 | 610 | 0.0% |
| copy_scalar | 1536 | 611 | 611 | 0.0% |
| copy_scalar | 1792 | 612 | 612 | 0.0% |
| copy_scalar | 2048 | 612 | 612 | 0.0% |
| copy_scalar | 2304 | 612 | 612 | 0.0% |
| copy_scalar | 2560 | 612 | 612 | 0.0% |
| copy_scalar | 2816 | 613 | 613 | 0.0% |
| copy_scalar | 3072 | 613 | 613 | 0.0% |
| copy_scalar | 3328 | 613 | 613 | 0.0% |
| copy_scalar | 3584 | 613 | 613 | 0.0% |
| copy_scalar | 3840 | 613 | 613 | 0.0% |
| copy_scalar | 4096 | 613 | 613 | 0.0% |

## VectorE and ScalarE concurrently, dual_alternate (`trn3_48xlarge_rev2_2026-11-06_dual.csv`)

GB/s here is the aggregate of both engines over the combined window; fdim and per-copy size are per engine.

### PSUM -> SBUF tensor_copy, one NeuronCore (LNC=1)

| kernel | pdim | fdim (fp32) | PSUM row | per copy | PSUM read GB/s (median, min-max) | SBUF write GB/s | ns per copy | instr duration ns | wait ns | traces | whole-kernel GB/s (lower bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dual_alternate | 128 | 64 | 256 B | 32 KiB | 993 (993-993) | 993 | 33 | 168 | 1 | 3 | 212 |
| dual_alternate | 128 | 128 | 512 B | 64 KiB | 1099 (1099-1099) | 1099 | 60 | 219 | 1 | 3 | 496 |
| dual_alternate | 128 | 256 | 1 KiB | 128 KiB | 1157 (1157-1157) | 1157 | 113 | 324 | 1 | 3 | 812 |
| dual_alternate | 128 | 512 | 2 KiB | 256 KiB | 1187 (1187-1187) | 1187 | 221 | 539 | 1 | 3 | 1015 |
| dual_alternate | 128 | 1024 | 4 KiB | 512 KiB | 1208 (1208-1208) | 1208 | 434 | 965 | 1 | 3 | 1123 |
| dual_alternate | 128 | 2048 | 8 KiB | 1 MiB | 1219 (1219-1219) | 1219 | 860 | 1819 | 1714 | 3 | 1178 |

### Fit: ns per copy = fixed + bytes / streaming rate (pdim=128)

| kernel | fixed ns per copy | streaming PSUM read GB/s | max abs error ns |
|---|---|---|---|
| dual_alternate | 7 | 1228 | 1 |

### Run-to-run (median GB/s per sweep repeat, pdim=128)

| kernel | fdim | dual_r0 GB/s | spread |
|---|---|---|---|
| dual_alternate | 64 | 993 | 0.0% |
| dual_alternate | 128 | 1099 | 0.0% |
| dual_alternate | 256 | 1157 | 0.0% |
| dual_alternate | 512 | 1187 | 0.0% |
| dual_alternate | 1024 | 1208 | 0.0% |
| dual_alternate | 2048 | 1219 | 0.0% |

