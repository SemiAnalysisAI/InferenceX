"""Names of the three kernels for tables and plots, and the TensorE peak they are compared with."""

LABEL = {"mm_bf16": "bf16 nc_matmul", "mm_fp8dr": "fp8 double_row nc_matmul", "mm_mx": "MXFP8 nc_matmul_mx"}


def label(kernel):
    return f"{LABEL[kernel]} (`{kernel}`)"


def peak_tflops(k, m, clock_ghz):
    """One NeuronCore's TensorE peak: 2 K M FLOPs in every clock cycle, one moving column per cycle."""
    return 2 * k * m * clock_ghz / 1e3
