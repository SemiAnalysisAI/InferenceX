<div align="center">

**English** | [中文](./TRAINIUM3_HARDWARE_ACCOUNTING_zh.md)

</div>

# Trainium3 hardware accounting for OperatorX calibration

## Scope

This assessment covers the Trainium3 partition visible on the current
`trn3pd98.3xlarge` host. The compute calibration unit is one LNC=2 logical
NeuronCore (`NC_v4d`): two physical NeuronCore-v4 cores sharing one HBM bank.
The other logical devices in the partition remain idle. There are no collectives,
NeuronLink assumptions, or UltraServer extrapolations in the GEMM assessment.

## Observed inventory

The environment manifest is embedded in the compact OperatorX result. The 2026-07-31
host reported:

| Evidence | Observation |
|---|---|
| EC2 instance type | `trn3pd98.3xlarge` |
| PCI identity | one `NeuronDevice (Trainium3)` |
| `neuron-ls -j` physical count | `nc_count: 8` |
| `neuron-ls -j` exposed IDs | `0, 1, 2, 3` |
| Device memory | 154,618,822,656 bytes = 144 GiB |
| Torch/XLA cross-check | four devices, kind `NC_v4d` |
| NKI compile target | `trn3`, LNC=2 |
| NKI | `0.3.0+23928721754.g18aa1271` |
| Neuron compiler | `2.24.5133.0+58f8de22` |
| Neuron runtime | `2.31.24` |
| Neuron profiler/tools | `2.29.18.0` |

The apparent eight-versus-four core discrepancy is expected under LNC=2: the chip
has eight physical NeuronCore-v4 cores, grouped in pairs into four logical
NeuronCores. The NKI compiler artifact records `lnc: 2`, and the benchmark launches
the kernel with grid degree two.

## Public specification cross-check

Primary references:

- [AWS Trainium3 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/about-neuron/arch/neuron-hardware/trainium3.html)
- [AWS logical NeuronCore configuration](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html)
- [AWS Trainium3 NKI architecture guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium3_arch.html)
- [AWS NKI LNC guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/nki/get-started/about/lnc.html)

AWS publishes eight physical cores, 144 GiB HBM, 671 BF16/FP16/TF32 TFLOP/s,
2,517 MXFP8/MXFP4 TFLOP/s, 183 FP32 TFLOP/s, and 4.9 TB/s HBM bandwidth per
Trainium3 chip. Dividing the chip into its four LNC=2/HBM-bank units gives the
following audit:

| Quantity | Published chip | Derived LNC=2 | Current SimulatorX `trn3` | Assessment policy |
|---|---:|---:|---:|---|
| HBM capacity | 144 GiB | 36 GiB | 36 GiB | agreed |
| HBM bandwidth | 4.9 TB/s | 1.225 TB/s | 1.225 TB/s | use current value, flag public-doc conflict |
| BF16 peak | 671 TFLOP/s | 167.75 TFLOP/s | 158 TFLOP/s | assess current value as-is |
| MXFP8/MXFP4 peak | 2,517 TFLOP/s | 629.25 TFLOP/s | 630 TFLOP/s | effectively agreed |
| FP32 peak | 183 TFLOP/s | 45.75 TFLOP/s | 40 TFLOP/s | out of BF16 pilot scope |
| static memory latency | not published | unknown | 200 ns | provisional |
| launch latency | not published | unknown | 300 ns | provisional |

The NKI architecture guide currently says 4.7 TB/s while the Trainium3 architecture
page says 4.9 TB/s. SimulatorX uses 4.9 TB/s divided evenly across the four HBM-bank
units. This assessment records the discrepancy and does not alter the device config.
Likewise, it reports the baseline produced by the current 158 TFLOP/s BF16 value
rather than silently replacing it with the 167.75 TFLOP/s headline-derived value.

## Accounting contract

- OperatorX requested shapes are logical model GEMMs.
- The NKI kernel's executed/padded shape is recorded separately.
- NKI profile `hardware_flops` must match `2*M*N*K` for the executed shape within
  0.01% in every retained compact sample. This remains a tight identity diagnostic
  while allowing the profiler's derived counter to omit a boundary-tile contribution.
- SimulatorX projects the requested logical shape on one `trn3` LNC=2 device.
- NKI padding is real implementation overhead, so it remains visible in the model
  error rather than being hidden by projecting the padded shape.
- Profile HBM read/write counters are retained as measured implementation traffic;
  they are not required to equal logical A+B+C bytes.
- Results apply only to one LNC=2 on this partition and make no collective or
  full-system correlation claim.
