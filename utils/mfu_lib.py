"""
Shared library for MFU (Model FLOPS Utilization) trace analysis.

This module encapsulates the core functionality required to analyse PyTorch
profiler traces for GEMM operations, grouped GEMM (MoE) operations,
communication overlap and network roofline.  The goal of this library is
to centralise all common logic so that the command‑line interface simply
parses arguments and dispatches work to the routines defined here.

The original implementation of the MFU trace analyser contained a large
amount of duplicated code spread across multiple functions.  This module
breaks that monolith into reusable components whilst preserving the
behaviour and output of the original script.  In particular it exposes
dataclasses for configuration and result types, helper functions for
extracting dimensions from CPU operations and GPU kernels, routines for
computing FLOPs, bytes and roofline metrics, and high level analysis
functions for GEMM kernels, grouped GEMM kernels, layer breakdown,
communication overlap and network rooflines.  A summary printer is also
provided.

Clients should instantiate a :class:`Config` to describe the model
architecture (hidden size, number of experts, etc.) and select a
:class:`GPUSpecs` entry from :data:`GPU_SPECS`.  These objects are then
passed to the analysis routines.
"""

from __future__ import annotations

import json
import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable
from collections import defaultdict

###############################################################################
# Data classes
###############################################################################

@dataclass
class GPUSpecs:
    """GPU specifications for MFU/MBU calculation.

    The fields mirror those used in the original script.  Peak TFLOPS are
    provided per data type along with memory bandwidth, number of SMs and
    cache sizes.  NVLink bandwidth is given in GB/s and assumed bidirectional.
    """

    name: str
    fp16_tflops: float
    fp8_tflops: float
    fp4_tflops: float
    memory_bw_tb_s: float
    num_sms: int
    l2_cache_mb: float = 50.0
    nvlink_bw_gb_s: float = 900.0


@dataclass
class Config:
    """Configuration for model dimensions and parallelism.

    This dataclass captures the parameters that vary between models and runs.
    It replaces the global ``MODEL_CONFIG`` dictionary used in the original
    implementation.  All routines that need knowledge of the model (e.g. for
    inferring CUDA graph dimensions or computing expert sizes) accept a
    :class:`Config` instance.

    In addition to tensor parallelism (TP), this configuration introduces
    ``ep_degree`` to describe expert parallelism (EP).  When EP is enabled,
    experts are partitioned across ``ep_degree`` groups and each group holds
    ``num_experts/ep_degree`` experts.  Many memory and FLOP calculations
    depend on the number of local experts, which can be obtained via
    :pyattr:`local_experts`.
    """

    hidden_size: int = 7168
    num_experts: int = 256
    expert_intermediate_size: int = 2048  # Total intermediate size before TP division
    decode_batch_size: int = 64
    tp_degree: int = 8
    ep_degree: int = 1  # Expert parallelism degree (EP). Use 1 for TP-only.
    expert_saturation_scale: float = 70.0  # Controls saturation speed

    # Default model precision for GEMM outputs.
    #
    # Many models accumulate results in BF16 by default even when the inputs
    # are lower precision (e.g. FP8).  This field allows callers to
    # override the fallback output dtype used when the profiler does not
    # provide an explicit C dtype.  For example, specifying
    # ``model_dtype='fp16'`` will cause fallback kernels to be treated as
    # producing FP16 outputs rather than the default BF16.  This value is
    # normalised via :func:`normalize_dtype` when used.
    model_dtype: str = 'bf16'

    @property
    def expert_intermediate_per_gpu(self) -> int:
        """Intermediate size per GPU after tensor parallelism division."""
        return self.expert_intermediate_size // max(self.tp_degree, 1)

    @property
    def local_experts(self) -> int:
        """Number of experts resident on a single EP rank.

        With expert parallelism, the total number of experts is partitioned
        across ``ep_degree`` groups.  Each group holds ``num_experts / ep_degree``
        experts.  When ``ep_degree=1``, this property simply returns
        ``num_experts``, meaning that all experts are present on each rank (pure TP).
        """
        return max(self.num_experts // max(self.ep_degree, 1), 1)

    def estimate_activated_experts(self, batch_size: int, top_k: int) -> int:
        """Estimate activated experts for a given batch size."""
        import math
        max_experts = self.local_experts
        if batch_size <= 0:
            return top_k
        saturation = 1.0 - math.exp(-batch_size / self.expert_saturation_scale)
        activated = top_k + (max_experts - top_k) * saturation
        return min(int(activated), max_experts)


@dataclass
class KernelClassification:
    """Classification result for a GPU kernel."""
    category: str
    subcategory: str
    is_gemm: bool
    dtype: str
    source: str


@dataclass
class GemmInfo:
    """Information about an analysed GEMM kernel."""

    m: int
    n: int
    k: int
    dtype: str
    input_dtype: str = ""
    output_dtype: str = ""
    a_dtype: str = ""
    b_dtype: str = ""
    c_dtype: str = ""
    duration_us: float = 0.0
    flops: int = 0
    tflops: float = 0.0
    mfu: float = 0.0
    bytes_accessed: int = 0
    achieved_bw_tb_s: float = 0.0
    mbu: float = 0.0
    arithmetic_intensity: float = 0.0
    roofline_tflops: float = 0.0
    roofline_bound: str = ""
    kernel_name: str = ""
    external_id: int = 0
    layer_type: str = ""
    activation_bytes: int = 0
    weight_bytes: int = 0
    effective_mbu: float = 0.0
    l2_cache_benefit: float = 0.0
    timestamp_us: float = 0.0
    correlation_id: int = 0
    tp_rank: str = ""
    stream_id: int = 0


@dataclass
class GroupedGemmInfo:
    """Information about a grouped GEMM operation (e.g. fused MoE)."""

    num_tokens: int
    top_k: int
    num_experts: int
    hidden_size: int
    w1_intermediate: int
    w2_intermediate: int
    input_dtype: str = "bf16"
    weight_dtype: str = "fp8"
    output_dtype: str = "bf16"
    total_token_expert_pairs: int = 0
    w1_flops: int = 0
    w2_flops: int = 0
    total_flops: int = 0
    input_bytes: int = 0
    w1_weight_bytes: int = 0
    w2_weight_bytes: int = 0
    output_bytes: int = 0
    total_bytes: int = 0
    duration_us: float = 0.0
    tflops: float = 0.0
    mfu: float = 0.0
    achieved_bw_tb_s: float = 0.0
    mbu: float = 0.0
    arithmetic_intensity: float = 0.0
    roofline_bound: str = ""
    kernel_name: str = ""
    external_id: int = 0
    num_kernels: int = 0
    timestamp_us: float = 0.0
    correlation_id: int = 0
    tp_rank: str = ""


###############################################################################
# GPU specification database
###############################################################################

GPU_SPECS: Dict[str, GPUSpecs] = {
    "B200": GPUSpecs(
        name="NVIDIA B200 SXM",
        fp16_tflops=2250.0,
        fp8_tflops=4500.0,
        fp4_tflops=9000.0,
        memory_bw_tb_s=8.0,
        num_sms=160,
        l2_cache_mb=128.0,
        nvlink_bw_gb_s=1800.0,
    ),
    "H200": GPUSpecs(
        name="NVIDIA H200 SXM",
        fp16_tflops=989.4,
        fp8_tflops=1978.9,
        fp4_tflops=0.0,
        memory_bw_tb_s=4.8,
        num_sms=132,
        l2_cache_mb=80.0,
        nvlink_bw_gb_s=900.0,
    ),
    "H100": GPUSpecs(
        name="NVIDIA H100 SXM",
        fp16_tflops=989.4,
        fp8_tflops=1978.9,
        fp4_tflops=0.0,
        memory_bw_tb_s=3.35,
        num_sms=132,
        l2_cache_mb=50.0,
        nvlink_bw_gb_s=900.0,
    ),
    "A100": GPUSpecs(
        name="NVIDIA A100 SXM",
        fp16_tflops=312.0,
        fp8_tflops=0.0,
        fp4_tflops=0.0,
        memory_bw_tb_s=2.0,
        num_sms=108,
        l2_cache_mb=40.0,
        nvlink_bw_gb_s=600.0,
    ),
}


###############################################################################
# Pattern definitions for kernel classification
###############################################################################

GEMM_KERNEL_PATTERNS = {
    'deep_gemm_fp8': {
        'match': lambda name: 'deep_gemm' in name.lower(),
        'is_gemm': True,
        'dtype': 'fp8',
        'source': 'deep_gemm',
        'subcategory': 'fp8_gemm',
    },
    'nvjet_cublas': {
        'match': lambda name: 'nvjet' in name.lower(),
        'is_gemm': True,
        'dtype': 'bf16',
        'source': 'cublas',
        'subcategory': 'cublas_gemm',
    },
    'cublas_gemm': {
        'match': lambda name: 'cublas' in name.lower() and 'gemm' in name.lower(),
        'is_gemm': True,
        'dtype': 'bf16',
        'source': 'cublas',
        'subcategory': 'cublas_gemm',
    },
    'cutlass_gemm': {
        'match': lambda name: 'cutlass' in name.lower() and ('gemm' in name.lower() or 'matmul' in name.lower()),
        'is_gemm': True,
        'dtype': 'bf16',
        'source': 'cutlass',
        'subcategory': 'cutlass_gemm',
    },
    'generic_gemm': {
        'match': lambda name: ('gemm' in name.lower() or 'matmul' in name.lower()) and 'deep_gemm' not in name.lower() and 'nvjet' not in name.lower(),
        'is_gemm': True,
        'dtype': 'bf16',
        'source': 'generic',
        'subcategory': 'other_gemm',
    },
}


COMM_KERNEL_PATTERNS = {
    'nccl_allreduce': {
        'match': lambda name: 'allreduce' in name.lower(),
        'subcategory': 'allreduce',
    },
    'nccl_allgather': {
        'match': lambda name: 'allgather' in name.lower() or 'all_gather' in name.lower(),
        'subcategory': 'all_gather',
    },
    'nccl_reducescatter': {
        'match': lambda name: 'reducescatter' in name.lower() or 'reduce_scatter' in name.lower(),
        'subcategory': 'reduce_scatter',
    },
    'cross_device_reduce': {
        'match': lambda name: 'cross_device_reduce' in name.lower(),
        'subcategory': 'cross_device_reduce',
    },
    'nccl_other': {
        'match': lambda name: 'nccl' in name.lower(),
        'subcategory': 'nccl_other',
    },
}


ATTENTION_KERNEL_PATTERNS = {
    'flash_attention': {
        'match': lambda name: 'flashinfer' in name.lower() or 'flash_attn' in name.lower() or 'fmha' in name.lower(),
        'subcategory': 'flash_attention',
    },
    'mla_attention': {
        'match': lambda name: 'batchmlapageattention' in name.lower() or 'prefillwithkvcache' in name.lower(),
        'subcategory': 'mla_attention',
    },
}


NORM_KERNEL_PATTERNS = {
    'rmsnorm': {
        'match': lambda name: 'rmsnorm' in name.lower(),
        'subcategory': 'rmsnorm',
    },
    'layernorm': {
        'match': lambda name: 'layernorm' in name.lower(),
        'subcategory': 'layernorm',
    },
}


###############################################################################
# Helper functions for dtype handling and metrics
###############################################################################

def normalize_dtype(dt: Optional[str]) -> str:
    """Normalize various dtype strings to canonical short names.

    The profiler sometimes emits data type strings with different naming
    conventions.  This helper function maps those to canonical forms such as
    ``fp8``, ``bf16``, ``fp16`` and ``fp32``.  Unknown dtypes are returned as
    lowercased strings.
    """
    if not dt:
        return ""
    s = str(dt).lower()
    if any(x in s for x in ["float8", "fp8", "e4m3", "e5m2"]):
        return "fp8"
    if any(x in s for x in ["bfloat16", "bf16"]):
        return "bf16"
    if any(x in s for x in ["float16", "fp16", "half"]):
        return "fp16"
    if any(x in s for x in ["float32", "fp32"]):
        return "fp32"
    if "int8" in s:
        return "int8"
    return s


def get_bytes_per_element(dtype: str) -> float:
    """Return bytes per element for a given dtype.

    Sub‑byte types such as FP4 are represented as fractional bytes.  The
    returned value may be a float to support these fractions.
    """
    dtype_lower = str(dtype).lower() if dtype else ""
    if any(x in dtype_lower for x in ["float4", "fp4", "e2m1"]):
        return 0.5
    if any(x in dtype_lower for x in ["float8", "fp8", "e4m3", "e5m2"]):
        return 1
    if any(x in dtype_lower for x in ["float16", "fp16", "bfloat16", "bf16", "half"]):
        return 2
    if any(x in dtype_lower for x in ["float32", "fp32"]):
        return 4
    return 2


def compute_dtype_from_inputs(a_dtype: str, b_dtype: str) -> str:
    """Heuristically determine the compute dtype from input dtypes.

    The logic prefers fp8 when present, then fp16, then bf16.  If none of
    these special cases match the two dtypes, one of them is returned.
    """
    dts = {normalize_dtype(a_dtype), normalize_dtype(b_dtype)}
    if "fp8" in dts:
        return "fp8"
    if "fp16" in dts:
        return "fp16"
    if "bf16" in dts:
        return "bf16"
    # Fallback to the first non‑empty dtype
    return next(iter(dts - {""}), "bf16")


def calculate_gemm_flops(m: int, n: int, k: int) -> int:
    """Calculate FLOPs for a GEMM operation ``C = A @ B``.

    Each element of the output matrix involves a multiply and an add.  Thus
    FLOPs are computed as ``2 * m * n * k``.
    """
    return 2 * m * n * k


def calculate_gemm_bytes(m: int, n: int, k: int,
                         a_dtype: str = 'bf16', b_dtype: str = 'bf16',
                         c_dtype: str = 'bf16') -> int:
    """Compute total bytes accessed for GEMM ``C = A @ B``.

    Reads ``A`` (``m × k``), reads ``B`` (``k × n``) and writes ``C``
    (``m × n``).  Supports sub‑byte types by rounding up to the nearest whole
    element count when necessary.
    """
    a_bytes = get_bytes_per_element(a_dtype)
    b_bytes = get_bytes_per_element(b_dtype)
    c_bytes = get_bytes_per_element(c_dtype)
    # Use integer rounding for sub‑byte types (e.g. FP4)
    bytes_a = int(m * k * a_bytes) if a_bytes >= 1 else ((m * k + 1) // 2)
    bytes_b = int(k * n * b_bytes) if b_bytes >= 1 else ((k * n + 1) // 2)
    bytes_c = int(m * n * c_bytes) if c_bytes >= 1 else ((m * n + 1) // 2)
    return bytes_a + bytes_b + bytes_c


def calculate_gemm_bytes_breakdown(m: int, n: int, k: int,
                                   a_dtype: str = 'bf16', b_dtype: str = 'bf16',
                                   c_dtype: str = 'bf16') -> Tuple[int, int, int]:
    """Return breakdown of bytes for GEMM.

    Returns a tuple ``(activation_bytes, weight_bytes, total_bytes)`` where
    ``activation_bytes`` is the sum of ``A`` and ``C`` (assuming weights ``B``
    can be served from cache), ``weight_bytes`` is the bytes for ``B`` and
    ``total_bytes`` is the sum.
    """
    a_bytes = get_bytes_per_element(a_dtype)
    b_bytes = get_bytes_per_element(b_dtype)
    c_bytes = get_bytes_per_element(c_dtype)
    bytes_a = int(m * k * a_bytes) if a_bytes >= 1 else ((m * k + 1) // 2)
    bytes_b = int(k * n * b_bytes) if b_bytes >= 1 else ((k * n + 1) // 2)
    bytes_c = int(m * n * c_bytes) if c_bytes >= 1 else ((m * n + 1) // 2)
    activation_bytes = bytes_a + bytes_c
    weight_bytes = bytes_b
    total_bytes = bytes_a + bytes_b + bytes_c
    return activation_bytes, weight_bytes, total_bytes


def calculate_arithmetic_intensity(flops: int, bytes_accessed: int) -> float:
    """Compute arithmetic intensity (FLOPs per byte)."""
    return flops / bytes_accessed if bytes_accessed > 0 else 0.0


def calculate_mfu(flops: int, duration_us: float, peak_tflops: float) -> float:
    """Compute Model FLOPS Utilization (MFU)."""
    if duration_us <= 0:
        return 0.0
    duration_s = duration_us / 1e6
    achieved_tflops = (flops / 1e12) / duration_s
    return (achieved_tflops / peak_tflops) * 100.0 if peak_tflops > 0 else 0.0


def calculate_mbu(bytes_accessed: int, duration_us: float, peak_bw_tb_s: float) -> float:
    """Compute Memory Bandwidth Utilization (MBU)."""
    if duration_us <= 0:
        return 0.0
    duration_s = duration_us / 1e6
    achieved_bw_tb_s = (bytes_accessed / 1e12) / duration_s
    return (achieved_bw_tb_s / peak_bw_tb_s) * 100.0 if peak_bw_tb_s > 0 else 0.0


def calculate_roofline_tflops(arithmetic_intensity: float, gpu_specs: GPUSpecs,
                              peak_tflops: float) -> Tuple[float, str]:
    """Return roofline‐based theoretical TFLOPS and bound type.

    The roofline model states that attainable performance is the minimum of
    compute peak and the product of memory bandwidth and arithmetic
    intensity.  ``peak_tflops`` corresponds to the compute peak for the
    operation’s dtype (returned by :func:`get_dtype_peak_tflops`).
    """
    if arithmetic_intensity <= 0:
        return 0.0, "unknown"
    memory_bound_tflops = gpu_specs.memory_bw_tb_s * arithmetic_intensity
    compute_bound_tflops = peak_tflops
    if memory_bound_tflops < compute_bound_tflops:
        return memory_bound_tflops, "memory"
    else:
        return compute_bound_tflops, "compute"


def get_dtype_peak_tflops(dtype: str, gpu_specs: GPUSpecs) -> float:
    """Return peak TFLOPS for a given dtype from the GPU specs.

    FP4 operations fall back to FP8 if unavailable.  FP8 falls back to
    FP16/BF16 if unavailable.  Unknown dtypes default to FP16/BF16 peak.
    """
    dtype_lower = str(dtype).lower()
    if any(x in dtype_lower for x in ["float4", "fp4", "e2m1"]):
        return gpu_specs.fp4_tflops if gpu_specs.fp4_tflops > 0 else (
            gpu_specs.fp8_tflops if gpu_specs.fp8_tflops > 0 else gpu_specs.fp16_tflops
        )
    if any(x in dtype_lower for x in ["float8", "fp8", "e4m3", "e5m2"]):
        return gpu_specs.fp8_tflops if gpu_specs.fp8_tflops > 0 else gpu_specs.fp16_tflops
    return gpu_specs.fp16_tflops

###############################################################################
# Dimension extraction from CPU operations and kernels
###############################################################################

CPU_OP_GEMM_PATTERNS = {
    'deep_gemm_fp8': {
        'match': lambda name: 'deep_gemm' in name.lower() or 'fp8_gemm' in name.lower(),
        'dtype': 'fp8',
    },
    'aten_mm': {
        'match': lambda name: name in ['aten::mm', 'aten::matmul'],
        'dtype': 'bf16',
    },
    'aten_linear': {
        'match': lambda name: name == 'aten::linear',
        'dtype': 'bf16',
    },
}


def extract_dimensions_from_cpu_op(event: Dict[str, Any]) -> Optional[Tuple[int, int, int, str, str, str]]:
    """Extract matrix dimensions and dtypes from a CPU op event.

    Returns a tuple ``(M, N, K, A_dtype, B_dtype, C_dtype)`` where the dtypes
    correspond to the input tensors and output for the GEMM.  If the op
    cannot be interpreted as a GEMM (e.g. insufficient information), returns
    ``None``.
    """
    args = event.get('args', {})
    input_dims = args.get('Input Dims', [])
    input_types = args.get('Input type', [])
    name = event.get('name', '')

    if not input_dims:
        return None

    # Attempt to identify deep_gemm FP8 operations
    if 'deep_gemm' in name.lower() and len(input_dims) >= 5:
        # Format: [M, K], scale, [N, K], scale, [M, N]
        a_dims = input_dims[0]
        b_dims = input_dims[2]
        if not (isinstance(a_dims, list) and len(a_dims) >= 2 and isinstance(b_dims, list) and len(b_dims) >= 1):
            return None
        m, k = a_dims[0], a_dims[1]
        n = b_dims[0]
        # Types: A_type, A_scale, B_type, B_scale, C_type
        types = [normalize_dtype(t) for t in input_types] if input_types else []
        a_dtype = types[0] if len(types) >= 1 else 'bf16'
        b_dtype = types[2] if len(types) >= 3 else a_dtype
        c_dtype = types[4] if len(types) >= 5 else (a_dtype if a_dtype == b_dtype else 'bf16')
        return m, n, k, a_dtype, b_dtype, c_dtype

    # aten::mm / aten::matmul: Input dims [[M,K],[K,N]]
    if name in ['aten::mm', 'aten::matmul'] and len(input_dims) >= 2:
        a_dims = input_dims[0]
        b_dims = input_dims[1]
        if not (isinstance(a_dims, list) and len(a_dims) >= 2 and isinstance(b_dims, list) and len(b_dims) >= 2):
            return None
        m, k = a_dims[0], a_dims[1]
        n = b_dims[1]
        types = [normalize_dtype(t) for t in input_types] if input_types else []
        a_dtype = types[0] if len(types) >= 1 else 'bf16'
        b_dtype = types[1] if len(types) >= 2 else a_dtype
        c_dtype = a_dtype if a_dtype == b_dtype else 'bf16'
        return m, n, k, a_dtype, b_dtype, c_dtype

    # aten::linear: Input dims [[M,K],[N,K], bias]
    if name == 'aten::linear' and len(input_dims) >= 2:
        a_dims = input_dims[0]
        w_dims = input_dims[1]
        if not (isinstance(a_dims, list) and len(a_dims) >= 2 and isinstance(w_dims, list) and len(w_dims) >= 2):
            return None
        # Handle batched input: [B, M, K] -> effective M = B*M
        if len(a_dims) == 2:
            m, k = a_dims
        elif len(a_dims) == 3:
            m = a_dims[0] * a_dims[1]
            k = a_dims[2]
        else:
            return None
        n = w_dims[0]  # Weight dims [N,K]
        types = [normalize_dtype(t) for t in input_types] if input_types else []
        a_dtype = types[0] if len(types) >= 1 else 'bf16'
        b_dtype = types[1] if len(types) >= 2 else a_dtype
        c_dtype = a_dtype if a_dtype == b_dtype else 'bf16'
        return m, n, k, a_dtype, b_dtype, c_dtype

    return None


def extract_tp_rank(pid: Any) -> Optional[str]:
    """Extract tensor parallel rank from a PID string or number."""
    if pid is None:
        return None
    match = re.search(r'\[TP(\d+)\]', str(pid))
    if match:
        return match.group(1)
    return str(pid)


def parse_deep_gemm_kernel_dims(kernel_name: str, grid: List[int],
                                cpu_op_dims: Optional[Tuple[int, int, int]] = None) -> Optional[Tuple[int, int, int, str]]:
    """Parse deep_gemm kernel template parameters to infer dimensions.

    The deep_gemm implementation names kernels with a template signature like
    ``deep_gemm::sm90_fp8_gemm_1d2d_impl<..., N, K, ..., M_tile, N_tile, K_tile, ...>``.
    If ``cpu_op_dims`` is provided, the M dimension is taken from it; otherwise
    it is inferred from the grid dimensions under the assumption that the grid
    x dimension is ``ceil(M/m_tile) * ceil(N/n_tile)``.
    Returns a tuple ``(M, N, K, dtype)`` where ``dtype`` is the compute dtype
    (fp8 or bf16).
    """
    match = re.search(r'deep_gemm::[^<]*<[^,]*,\s*(\d+)u,\s*(\d+)u,[^,]*,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u', kernel_name)
    if not match:
        return None
    n = int(match.group(1))
    k = int(match.group(2))
    m_tile = int(match.group(3))
    n_tile = int(match.group(4))
    # Determine M dimension
    if cpu_op_dims:
        m = cpu_op_dims[0]  # Use M from CPU op
    else:
        grid_x = grid[0] if grid else 1
        # number of tiles along N dimension
        num_n_tiles = (n + n_tile - 1) // n_tile
        if grid_x <= num_n_tiles:
            # Single M tile
            m = m_tile
        else:
            num_m_tiles = max(grid_x // num_n_tiles, 1)
            m = num_m_tiles * m_tile
    dtype = 'fp8' if 'fp8' in kernel_name.lower() else 'bf16'
    return (m, n, k, dtype)


def infer_cuda_graph_kernel_dims(kernel_name: str, grid: List[int],
                                 config: Optional[Config] = None,
                                 sibling_dims: Optional[Dict[str, Tuple[int, int]]] = None) -> Optional[Tuple[int, int, int, str, str]]:
    """Infer dimensions for CUDA graph replayed kernels.

    See the original implementation for detailed heuristics.  The inference
    relies on a combination of sibling kernel dimensions (obtained from
    prefill kernels with External ID) and known model architecture.  The
    ``config`` argument provides hidden size, expert intermediate size and
    decode batch size.  Returns a tuple ``(M, N, K, dtype, layer_type)``.
    """
    # Allow config to be None for fallback heuristics
    hidden = config.hidden_size if config else 7168
    tp_degree = config.tp_degree if config else 8
    expert_intermediate_per_gpu = (config.expert_intermediate_size // max(tp_degree, 1)) if config else (2048 // 8)
    decode_batch = config.decode_batch_size if config else 64
    name_lower = kernel_name.lower()
    grid_tuple = tuple(grid) if grid else ()
    # Strategy 1: use sibling dimensions for nvjet_tst_128x8 (shared expert)
    if sibling_dims and 'nvjet_tst_128x8' in name_lower and 'nvjet_tst_128x8' in sibling_dims:
        n, k = sibling_dims['nvjet_tst_128x8']
        return (decode_batch, n, k, 'bf16', 'FFN')
    # Strategy 2: use sibling dims for nvjet_tst_64x8 based on grid
    if sibling_dims and 'nvjet_tst_64x8' in name_lower and 'nvjet_tst_64x64' in sibling_dims:
        intermediate_per_gpu, hidden_from_sibling = sibling_dims['nvjet_tst_64x64']
        if grid_tuple == (2, 64, 1):
            # Down projection: [M, intermediate] @ [intermediate, hidden]
            return (decode_batch, hidden_from_sibling, intermediate_per_gpu, 'bf16', 'FFN')
        elif grid_tuple == (2, 16, 1):
            # Up projection: [M, hidden] @ [hidden, intermediate]
            return (decode_batch, intermediate_per_gpu, hidden_from_sibling, 'bf16', 'FFN')
    # Strategy 3: use model knowledge for nvjet_tst_64x8
    if 'nvjet_tst_64x8' in name_lower:
        if grid_tuple == (2, 64, 1):
            return (decode_batch, hidden, expert_intermediate_per_gpu, 'bf16', 'FFN')
        elif grid_tuple == (2, 16, 1):
            return (decode_batch, expert_intermediate_per_gpu, hidden, 'bf16', 'FFN')
    # Shared expert nvjet_tst_128x8 fallback
    if 'nvjet_tst_128x8' in name_lower:
        return (decode_batch, 16160, hidden, 'bf16', 'FFN')
    # nvjet_tst_64x64 kernels are handled via CPU op dims
    if 'nvjet_tst_64x64' in name_lower:
        return None
    # router_gemm kernels
    if 'router_gemm' in name_lower:
        num_experts = config.num_experts
        return (decode_batch, num_experts, hidden, 'bf16', 'FFN')
    return None


def classify_kernel(kernel_name: str) -> KernelClassification:
    """Classify a GPU kernel by examining its name against known patterns."""
    for pattern in GEMM_KERNEL_PATTERNS.values():
        if pattern['match'](kernel_name):
            return KernelClassification(
                category='gemm',
                subcategory=pattern['subcategory'],
                is_gemm=True,
                dtype=pattern['dtype'],
                source=pattern['source'],
            )
    for pattern in COMM_KERNEL_PATTERNS.values():
        if pattern['match'](kernel_name):
            return KernelClassification(
                category='communication',
                subcategory=pattern['subcategory'],
                is_gemm=False,
                dtype='',
                source='nccl',
            )
    for pattern in ATTENTION_KERNEL_PATTERNS.values():
        if pattern['match'](kernel_name):
            return KernelClassification(
                category='attention',
                subcategory=pattern['subcategory'],
                is_gemm=False,
                dtype='',
                source='flashinfer',
            )
    for pattern in NORM_KERNEL_PATTERNS.values():
        if pattern['match'](kernel_name):
            return KernelClassification(
                category='normalization',
                subcategory=pattern['subcategory'],
                is_gemm=False,
                dtype='',
                source='custom',
            )
    return KernelClassification(
        category='other',
        subcategory='unknown',
        is_gemm=False,
        dtype='',
        source='unknown',
    )


def classify_layer_type(m: int, n: int, k: int, kernel_name: str = "") -> str:
    """Heuristically classify a GEMM as belonging to QKVO, FFN or other layers."""
    hidden_size = 7168
    num_experts = 256
    # MoE router
    if k == hidden_size and n == num_experts:
        return 'FFN'
    # MoE gate/router variants
    if k == 512 and n in [4096, 2048, 1024]:
        return 'FFN'
    # MoE FFN projections: up or down
    if (n == 4608 and k == hidden_size) or (n == hidden_size and k == 4608):
        return 'FFN'
    if n > 10000 or k > 10000:
        return 'FFN'
    # Attention projections
    if k == hidden_size and n in [2112, 2048, 2304, 2560]:
        return 'QKVO'
    if n == hidden_size and k in [2048, 2112, 2304, 2560]:
        return 'QKVO'
    if (n == 3072 and k == 1536) or (n == 1536 and k == 3072):
        return 'QKVO'
    if k == hidden_size:
        return 'QKVO'
    if n == hidden_size:
        return 'QKVO'
    return 'Other'


###############################################################################
# CPU op and sibling dimension maps
###############################################################################

def build_cpu_op_dims_map(events: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int], Tuple[int, int, int, str, str, str]]:
    """Build a map from (TP rank, External ID) to GEMM dimensions and dtypes.

    This helper consolidates the repeated logic used in several analysis
    routines.  It iterates over CPU op events, extracts dimensions via
    :func:`extract_dimensions_from_cpu_op` and stores them keyed by TP rank and
    External ID.  The map also includes entries for adjacent external IDs
    (``ext_id ± 1``) since GPU kernels may use offset IDs.
    """
    cpu_op_dims: Dict[Tuple[str, int], Tuple[int, int, int, str, str, str]] = {}
    for event in events:
        if event.get('cat') != 'cpu_op':
            continue
        ext_id = event.get('args', {}).get('External id')
        if ext_id is None:
            continue
        tp_rank = extract_tp_rank(event.get('pid'))
        dims = extract_dimensions_from_cpu_op(event)
        if dims:
            cpu_op_dims[(tp_rank, ext_id)] = dims
            # Also map neighbouring IDs to the same dims to catch child kernels
            cpu_op_dims[(tp_rank, ext_id + 1)] = dims
            cpu_op_dims[(tp_rank, ext_id - 1)] = dims
    return cpu_op_dims


def build_sibling_dims_map(events: Iterable[Dict[str, Any]],
                           cpu_op_dims: Dict[Tuple[str, int], Tuple[int, int, int, str, str, str]]) -> Dict[str, Tuple[int, int]]:
    """Build a map of kernel signatures to (N, K) dimensions using CPU op dims.

    For kernels like ``nvjet_tst_64x64`` the External ID identifies a CPU op
    with dimensions (M,N,K).  The sibling map stores ``(N, K)`` keyed by
    signature so that decode kernels (without External ID) can later infer
    their shapes.
    """
    sibling_dims: Dict[str, Tuple[int, int]] = {}
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        name = event.get('name', '')
        ext_id = event.get('args', {}).get('External id')
        if ext_id is None:
            continue
        match = re.search(r'nvjet_tst_(\d+x\d+)', name.lower())
        if not match:
            continue
        signature = f"nvjet_tst_{match.group(1)}"
        tp_rank = extract_tp_rank(event.get('pid'))
        # Try several neighbouring ext_ids
        for key_ext in [ext_id, ext_id - 1, ext_id + 1]:
            dims = cpu_op_dims.get((tp_rank, key_ext))
            if dims and len(dims) >= 3:
                # dims are (m,n,k,...) so take (n,k)
                sibling_dims.setdefault(signature, (dims[1], dims[2]))
                break
    return sibling_dims


###############################################################################
# GEMM kernel analysis
###############################################################################

def analyze_gemm_kernels(events: List[Dict[str, Any]], gpu_specs: GPUSpecs, config: Config) -> List[GemmInfo]:
    """Analyse all GEMM/MatMul kernels in a trace and compute performance metrics.

    The returned list contains a :class:`GemmInfo` entry for each kernel
    identified as GEMM with known dimensions.  Dimension extraction proceeds in
    order of priority: CPU op correlation, deep_gemm template parsing, CUDA
    graph inference.  Metrics such as MFU, MBU, arithmetic intensity and
    roofline bound are computed for each kernel.
    """
    gemm_infos: List[GemmInfo] = []
    # Build CPU op dimension map once
    cpu_op_dims = build_cpu_op_dims_map(events)
    # Build sibling dimension map for nvjet kernels
    sibling_dims = build_sibling_dims_map(events, cpu_op_dims)
    seen_kernels = set()
    unmatched = defaultdict(lambda: {'count': 0, 'time_us': 0})

    for event in events:
        if event.get('cat') != 'kernel':
            continue
        name = event.get('name', '')
        classification = classify_kernel(name)
        if not classification.is_gemm:
            continue
        duration_us = event.get('dur', 0)
        if duration_us <= 0:
            continue
        ext_id = event.get('args', {}).get('External id')
        tp_rank = extract_tp_rank(event.get('pid'))
        grid = event.get('args', {}).get('grid', [1, 1, 1])
        ts = event.get('ts', 0)
        kernel_key = (tp_rank, ts, name[:50])
        if kernel_key in seen_kernels:
            continue
        seen_kernels.add(kernel_key)
        # Extract dims via CPU op map
        dims = None
        if ext_id is not None:
            dims = cpu_op_dims.get((tp_rank, ext_id)) or cpu_op_dims.get((tp_rank, ext_id - 1)) or cpu_op_dims.get((tp_rank, ext_id + 1))
        # Parse deep_gemm template if needed
        if dims is None and classification.source == 'deep_gemm':
            parsed = parse_deep_gemm_kernel_dims(name, grid, None)
            if parsed:
                m_, n_, k_, dtype_ = parsed
                # Deep GEMM kernels use FP8 inputs.  The accumulator/output is
                # typically BF16 by default.  To allow users to override this
                # behaviour, use the configured model dtype for the output.
                dims = (m_, n_, k_, 'fp8', 'fp8', normalize_dtype(config.model_dtype))
        # Infer CUDA graph kernels
        inferred_layer_type = None
        if dims is None and ext_id is None:
            inferred = infer_cuda_graph_kernel_dims(name, grid, config, sibling_dims=sibling_dims)
            if inferred:
                m_, n_, k_, dtype_, inferred_layer_type = inferred
                # Use the model's default dtype for the output when inferring
                # CUDA graph kernel dimensions.  When the compute dtype is
                # FP8 the output is often BF16 in many models, but the model
                # dtype parameter allows this behaviour to be customised.
                c_dtype = normalize_dtype(config.model_dtype)
                dims = (m_, n_, k_, dtype_, dtype_, c_dtype)
        if dims is None:
            # Record unmatched for debugging
            unmatched[classification.subcategory]['count'] += 1
            unmatched[classification.subcategory]['time_us'] += duration_us
            continue
        # Unpack dims to (m,n,k,a,b,c)
        if len(dims) >= 6:
            m, n, k, a_dtype, b_dtype, c_dtype = dims[:6]
        elif len(dims) == 5:
            m, n, k, input_dtype, output_dtype = dims
            a_dtype = b_dtype = input_dtype
            c_dtype = output_dtype
        else:
            # Only (M,N,K,input_dtype) provided.  Both A and B use the same
            # input dtype.  Use the configured model dtype as the fallback
            # output dtype instead of assuming BF16.  This allows users to
            # specify the precision of GEMM outputs when the profiler does not
            # record a C dtype (e.g. FP16 models).
            m, n, k, input_dtype = dims
            a_dtype = b_dtype = input_dtype
            # Normalise the configured model dtype so that abbreviations like
            # 'float16' map to canonical short names.  If the input dtype is
            # FP8 we still respect the model dtype for the output.
            c_dtype = normalize_dtype(config.model_dtype)
        if m <= 0 or n <= 0 or k <= 0:
            continue
        # Override dtype from classification if not specified
        if not a_dtype and classification.dtype:
            a_dtype = classification.dtype
        if not b_dtype and classification.dtype:
            b_dtype = classification.dtype
        if not c_dtype and classification.dtype:
            c_dtype = 'bf16' if classification.dtype == 'fp8' else classification.dtype
        compute_dtype = compute_dtype_from_inputs(a_dtype, b_dtype)
        # Compute metrics
        flops = calculate_gemm_flops(m, n, k)
        bytes_accessed = calculate_gemm_bytes(m, n, k, a_dtype, b_dtype, c_dtype)
        activation_bytes, weight_bytes, _ = calculate_gemm_bytes_breakdown(m, n, k, a_dtype, b_dtype, c_dtype)
        peak_tflops = get_dtype_peak_tflops(compute_dtype, gpu_specs)
        duration_s = duration_us / 1e6
        achieved_tflops = (flops / 1e12) / duration_s
        achieved_bw_tb_s = (bytes_accessed / 1e12) / duration_s
        mfu = calculate_mfu(flops, duration_us, peak_tflops)
        mbu = calculate_mbu(bytes_accessed, duration_us, gpu_specs.memory_bw_tb_s)
        effective_mbu = calculate_mbu(activation_bytes, duration_us, gpu_specs.memory_bw_tb_s)
        l2_cache_benefit = (bytes_accessed / activation_bytes) if activation_bytes > 0 else 1.0
        ai = calculate_arithmetic_intensity(flops, bytes_accessed)
        roofline_tflops, roofline_bound = calculate_roofline_tflops(ai, gpu_specs, peak_tflops)
        layer_type = inferred_layer_type if inferred_layer_type else classify_layer_type(m, n, k, name)
        correlation_id = event.get('args', {}).get('correlation', 0)
        stream_id = event.get('args', {}).get('stream', 0)
        gemm_infos.append(GemmInfo(
            m=m, n=n, k=k,
            dtype=compute_dtype,
            input_dtype=(a_dtype if a_dtype == b_dtype else 'mixed'),
            output_dtype=c_dtype,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            duration_us=duration_us,
            flops=flops,
            tflops=achieved_tflops,
            mfu=mfu,
            bytes_accessed=bytes_accessed,
            achieved_bw_tb_s=achieved_bw_tb_s,
            mbu=mbu,
            arithmetic_intensity=ai,
            roofline_tflops=roofline_tflops,
            roofline_bound=roofline_bound,
            kernel_name=name,
            external_id=ext_id if ext_id is not None else 0,
            layer_type=layer_type,
            activation_bytes=activation_bytes,
            weight_bytes=weight_bytes,
            effective_mbu=effective_mbu,
            l2_cache_benefit=l2_cache_benefit,
            timestamp_us=event.get('ts', 0),
            correlation_id=correlation_id,
            tp_rank=tp_rank if tp_rank else "",
            stream_id=stream_id if stream_id else 0,
        ))
    return gemm_infos


###############################################################################
# Grouped GEMM (fused MoE) analysis
###############################################################################

def analyze_grouped_gemm_kernels(events: List[Dict[str, Any]], gpu_specs: GPUSpecs, config: Config) -> List[GroupedGemmInfo]:
    """Analyse fused MoE kernels and compute grouped GEMM metrics.

    This function supports both prefill and decode phases.  For prefill
    kernels (those with External ID) the CPU op event provides the full
    dimensions.  For decode kernels (those without External ID) the analysis
    infers dimensions based on typical decode batch sizes and heuristics.  The
    returned list contains one entry per grouped GEMM operation aggregated
    across all kernel calls belonging to that operation.
    """
    grouped_infos: List[GroupedGemmInfo] = []
    # Identify TP ranks (to infer number of GPUs)
    tp_ranks = set()
    for event in events:
        pid = event.get('pid')
        match = re.search(r'\[TP(\d+)\]', str(pid))
        if match:
            tp_ranks.add(match.group(1))
    num_gpus = max(len(tp_ranks), 1)
    # Build map from (tp_rank, ext_id) to fused expert dims
    fused_expert_ops: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for event in events:
        if event.get('cat') != 'cpu_op':
            continue
        name = event.get('name', '')
        if 'inplace_fused_experts' not in name and 'fused_experts' not in name:
            continue
        ext_id = event.get('args', {}).get('External id')
        if ext_id is None:
            continue
        tp_rank = extract_tp_rank(event.get('pid'))
        args = event.get('args', {})
        input_dims = args.get('Input Dims', [])
        input_types = args.get('Input type', [])
        if len(input_dims) < 5:
            continue
        input_shape = input_dims[0] if input_dims[0] else []
        w1_shape = input_dims[1] if len(input_dims) > 1 else []
        w2_shape = input_dims[2] if len(input_dims) > 2 else []
        topk_shape = input_dims[3] if len(input_dims) > 3 else []
        if not (len(input_shape) >= 2 and len(w1_shape) >= 3 and len(w2_shape) >= 3 and len(topk_shape) >= 2):
            continue
        num_tokens = input_shape[0]
        hidden_size = input_shape[1]
        num_experts_local = w1_shape[0]
        w1_intermediate = w1_shape[1]
        w2_intermediate = w2_shape[2]
        top_k = topk_shape[1]
        input_dtype = normalize_dtype(input_types[0]) if input_types else 'bf16'
        weight_dtype = normalize_dtype(input_types[1]) if len(input_types) > 1 else 'fp8'
        output_dtype = input_dtype
        fused_expert_ops[(tp_rank, ext_id)] = {
            'num_tokens': num_tokens,
            'hidden_size': hidden_size,
            'num_experts': num_experts_local,
            'w1_intermediate': w1_intermediate,
            'w2_intermediate': w2_intermediate,
            'top_k': top_k,
            'input_dtype': input_dtype,
            'weight_dtype': weight_dtype,
            'output_dtype': output_dtype,
            'ts': event.get('ts', 0),
        }
    # Collect fused_moe kernels
    moe_kernels_by_ext = defaultdict(list)
    moe_kernels_no_ext = []
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        name = event.get('name', '')
        if not name.startswith('fused_moe_kernel'):
            continue
        ext_id = event.get('args', {}).get('External id')
        tp_rank = extract_tp_rank(event.get('pid'))
        kernel_info = {
            'name': name,
            'dur': event.get('dur', 0),
            'ts': event.get('ts', 0),
            'correlation': event.get('args', {}).get('correlation', 0),
            'grid': event.get('args', {}).get('grid', []),
            'ext_id': ext_id,
            'tp_rank': tp_rank,
        }
        if ext_id is not None:
            moe_kernels_by_ext[(tp_rank, ext_id)].append(kernel_info)
        else:
            moe_kernels_no_ext.append(kernel_info)
    processed_ext_ids = set()
    # Prefill kernels (with External ID)
    for (tp_rank, ext_id), kernels in moe_kernels_by_ext.items():
        if ext_id in processed_ext_ids:
            continue
        processed_ext_ids.add(ext_id)
        dims = fused_expert_ops.get((tp_rank, ext_id))
        if dims is None:
            continue
        num_tokens = dims['num_tokens']
        hidden_size = dims['hidden_size']
        # Determine number of experts and local experts
        # dims['num_experts'] corresponds to the local experts for TP-only runs.
        # When expert parallelism is enabled, the total experts is provided by
        # config.num_experts and local experts per EP rank is config.local_experts.
        if config.ep_degree > 1:
            num_total_experts = config.num_experts
            num_local_experts = config.local_experts
        else:
            num_total_experts = dims['num_experts']
            num_local_experts = dims['num_experts']
        w1_intermediate = dims['w1_intermediate']
        w2_intermediate = dims['w2_intermediate']
        top_k = dims['top_k']
        input_dtype = dims['input_dtype']
        weight_dtype = dims['weight_dtype']
        output_dtype = dims['output_dtype']
        total_duration_us = sum(k['dur'] for k in kernels)
        total_pairs = num_tokens * top_k
        # For EP, each EP rank handles a fraction of the total token‑expert pairs
        pairs_per_rank = total_pairs / max(config.ep_degree, 1)
        # FLOPs per EP rank
        w1_flops = 2 * pairs_per_rank * hidden_size * w1_intermediate
        w2_flops = 2 * pairs_per_rank * w2_intermediate * hidden_size
        total_flops = w1_flops + w2_flops
        # Memory bytes per EP rank
        input_bytes = int(num_tokens * hidden_size * get_bytes_per_element(input_dtype))
        weight_bytes_elem = get_bytes_per_element(weight_dtype)
        # Each rank stores only its local experts when ep_degree>1
        w1_weight_bytes = int(num_local_experts * w1_intermediate * hidden_size * weight_bytes_elem)
        w2_weight_bytes = int(num_local_experts * hidden_size * w2_intermediate * weight_bytes_elem)
        output_bytes = int(num_tokens * hidden_size * get_bytes_per_element(output_dtype))
        total_bytes = input_bytes + w1_weight_bytes + w2_weight_bytes + output_bytes
        duration_s = total_duration_us / 1e6
        achieved_tflops = (total_flops / 1e12) / duration_s if duration_s > 0 else 0
        achieved_bw_tb_s = (total_bytes / 1e12) / duration_s if duration_s > 0 else 0
        peak_tflops = get_dtype_peak_tflops(weight_dtype, gpu_specs)
        mfu = (achieved_tflops / peak_tflops) * 100.0 if peak_tflops > 0 else 0
        mbu = (achieved_bw_tb_s / gpu_specs.memory_bw_tb_s) * 100.0 if gpu_specs.memory_bw_tb_s > 0 else 0
        ai = total_flops / total_bytes if total_bytes > 0 else 0
        memory_bound_tflops = gpu_specs.memory_bw_tb_s * ai
        roofline_bound = 'memory' if memory_bound_tflops < peak_tflops else 'compute'
        grouped_infos.append(GroupedGemmInfo(
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts_local,
            hidden_size=hidden_size,
            w1_intermediate=w1_intermediate,
            w2_intermediate=w2_intermediate,
            input_dtype=input_dtype,
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            total_token_expert_pairs=total_pairs,
            w1_flops=w1_flops,
            w2_flops=w2_flops,
            total_flops=total_flops,
            input_bytes=input_bytes,
            w1_weight_bytes=w1_weight_bytes,
            w2_weight_bytes=w2_weight_bytes,
            output_bytes=output_bytes,
            total_bytes=total_bytes,
            duration_us=total_duration_us,
            tflops=achieved_tflops,
            mfu=mfu,
            achieved_bw_tb_s=achieved_bw_tb_s,
            mbu=mbu,
            arithmetic_intensity=ai,
            roofline_bound=roofline_bound,
            kernel_name='fused_moe_kernel',
            external_id=ext_id,
            num_kernels=len(kernels),
            timestamp_us=kernels[0]['ts'] if kernels else 0,
            correlation_id=kernels[0]['correlation'] if kernels else 0,
            tp_rank=tp_rank if tp_rank else "",
        ))
    # Decode kernels (without External ID)
    if moe_kernels_no_ext and fused_expert_ops:
        sample_dims = next(iter(fused_expert_ops.values()))
        decode_batch_size = config.decode_batch_size
        hidden_size = sample_dims['hidden_size']
        # Determine experts for EP: sample_dims['num_experts'] holds local experts for TP-only
        if config.ep_degree > 1:
            num_total_experts = config.num_experts
            num_local_experts = config.local_experts
        else:
            num_total_experts = sample_dims['num_experts']
            num_local_experts = sample_dims['num_experts']
        w1_intermediate = sample_dims['w1_intermediate']
        w2_intermediate = sample_dims['w2_intermediate']
        top_k = sample_dims['top_k']
        input_dtype = sample_dims['input_dtype']
        weight_dtype = sample_dims['weight_dtype']
        output_dtype = sample_dims['output_dtype']
        total_pairs = decode_batch_size * top_k
        # Group by grid pattern
        grid_patterns = defaultdict(list)
        for kinfo in moe_kernels_no_ext:
            grid_key = tuple(kinfo['grid']) if kinfo['grid'] else ()
            grid_patterns[grid_key].append(kinfo)
        for grid_key, kernels in grid_patterns.items():
            if not kernels:
                continue
            total_dur_all_gpus_us = sum(k['dur'] for k in kernels)
            total_dur_per_gpu_us = total_dur_all_gpus_us / num_gpus
            num_kernel_calls_per_gpu = len(kernels) // num_gpus if num_gpus > 0 else len(kernels)
            # Heuristic: small grid means W1 (gate+up), large grid means W2 (down)
            is_w1 = grid_key and grid_key[0] < 5000
            # FLOPs per kernel call: adjust for EP.  Each EP rank handles
            # ``pairs_per_rank = total_pairs / ep_degree`` token-expert pairs.
            pairs_per_rank = total_pairs / max(config.ep_degree, 1)
            if is_w1:
                flops_per_kernel = 2 * pairs_per_rank * hidden_size * w1_intermediate
            else:
                flops_per_kernel = 2 * pairs_per_rank * w2_intermediate * hidden_size
            total_flops_per_gpu = flops_per_kernel * num_kernel_calls_per_gpu
            # Estimate memory usage: we assume 60% utilisation of local experts
            est_experts_used = config.estimate_activated_experts(decode_batch_size, top_k)
            input_bytes = int(decode_batch_size * hidden_size * get_bytes_per_element(input_dtype))
            if is_w1:
                weight_bytes = int(est_experts_used * w1_intermediate * hidden_size * get_bytes_per_element(weight_dtype))
                output_bytes_per_call = int(pairs_per_rank * w1_intermediate * get_bytes_per_element(output_dtype))
            else:
                weight_bytes = int(est_experts_used * hidden_size * w2_intermediate * get_bytes_per_element(weight_dtype))
                # Output of w2 is [M, hidden_size] per rank
                output_bytes_per_call = int(decode_batch_size * hidden_size * get_bytes_per_element(output_dtype))
            bytes_per_call = input_bytes + weight_bytes + output_bytes_per_call
            total_bytes_per_gpu = bytes_per_call * num_kernel_calls_per_gpu
            duration_s = total_dur_per_gpu_us / 1e6
            achieved_tflops = (total_flops_per_gpu / 1e12) / duration_s if duration_s > 0 else 0
            achieved_bw_tb_s = (total_bytes_per_gpu / 1e12) / duration_s if duration_s > 0 else 0
            peak_tflops = get_dtype_peak_tflops(weight_dtype, gpu_specs)
            mfu = (achieved_tflops / peak_tflops) * 100.0 if peak_tflops > 0 else 0
            mbu = (achieved_bw_tb_s / gpu_specs.memory_bw_tb_s) * 100.0 if gpu_specs.memory_bw_tb_s > 0 else 0
            # Adjust unrealistic MBU
            if mbu > 100:
                estimated_actual_bw = gpu_specs.memory_bw_tb_s * 0.9
                total_bytes_per_gpu = int(estimated_actual_bw * 1e12 * duration_s)
                achieved_bw_tb_s = estimated_actual_bw
                mbu = 90.0
            ai = total_flops_per_gpu / total_bytes_per_gpu if total_bytes_per_gpu > 0 else 0
            memory_bound_tflops = gpu_specs.memory_bw_tb_s * ai
            bound = 'memory' if memory_bound_tflops < peak_tflops else 'compute'
            grouped_infos.append(GroupedGemmInfo(
                num_tokens=decode_batch_size,
                top_k=top_k,
                # Report the total number of experts in the model.  Local experts
                # depend on ep_degree but the full model has num_total_experts.
                num_experts=num_total_experts,
                hidden_size=hidden_size,
                w1_intermediate=w1_intermediate if is_w1 else 0,
                w2_intermediate=w2_intermediate if not is_w1 else 0,
                input_dtype=input_dtype,
                weight_dtype=weight_dtype,
                output_dtype=output_dtype,
                total_token_expert_pairs=total_pairs,
                w1_flops=total_flops_per_gpu if is_w1 else 0,
                w2_flops=total_flops_per_gpu if not is_w1 else 0,
                total_flops=total_flops_per_gpu,
                input_bytes=input_bytes * num_kernel_calls_per_gpu,
                w1_weight_bytes=weight_bytes * num_kernel_calls_per_gpu if is_w1 else 0,
                w2_weight_bytes=weight_bytes * num_kernel_calls_per_gpu if not is_w1 else 0,
                output_bytes=output_bytes_per_call * num_kernel_calls_per_gpu,
                total_bytes=total_bytes_per_gpu,
                duration_us=total_dur_per_gpu_us,
                tflops=achieved_tflops,
                mfu=mfu,
                achieved_bw_tb_s=achieved_bw_tb_s,
                mbu=mbu,
                arithmetic_intensity=ai,
                roofline_bound=bound,
                kernel_name=f"fused_moe_kernel (decode, {'w1' if is_w1 else 'w2'})",
                external_id=0,
                num_kernels=len(kernels),
                timestamp_us=kernels[0]['ts'] if kernels else 0,
                correlation_id=kernels[0]['correlation'] if kernels else 0,
                tp_rank="*",
            ))
    return grouped_infos


###############################################################################
# Layer time breakdown analysis
###############################################################################

def analyze_layer_time_breakdown(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Analyse total time spent per layer type (QKVO, SDPA, FFN, etc.)."""
    tp_ranks = set()
    for event in events:
        pid = event.get('pid', '')
        match = re.search(r'\[TP(\d+)\]', str(pid))
        if match:
            tp_ranks.add(match.group(1))
    num_gpus = max(len(tp_ranks), 1)
    layer_times = {
        'QKVO': {'time_us': 0.0, 'count': 0},
        'SDPA': {'time_us': 0.0, 'count': 0},
        'FFN': {'time_us': 0.0, 'count': 0},
        'Normalization': {'time_us': 0.0, 'count': 0},
        'Communication': {'time_us': 0.0, 'count': 0},
        'Other': {'time_us': 0.0, 'count': 0},
    }
    # Build CPU op dims map for GEMM classification
    cpu_op_dims = build_cpu_op_dims_map(events)
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        name = event.get('name', '')
        dur = event.get('dur', 0)
        if dur <= 0:
            continue
        layer_type = None
        name_lower = name.lower()
        # Communication (exclude long warmup)
        if any(x in name_lower for x in ['nccl', 'cross_device_reduce', 'allreduce', 'allgather', 'all_gather', 'reducescatter', 'reduce_scatter']):
            if dur < 1e6:
                layer_type = 'Communication'
            else:
                continue
        if layer_type is None and ('rmsnorm' in name_lower or 'layernorm' in name_lower):
            layer_type = 'Normalization'
        if layer_type is None and any(x in name_lower for x in ['flashinfer', 'attention', 'mla', 'fmha']):
            if 'BatchMLAPageAttention' in name or 'PrefillWithKVCache' in name:
                layer_type = 'SDPA'
            elif 'Rotary' in name:
                layer_type = 'QKVO'
        # GEMM kernels
        if layer_type is None and any(x in name_lower for x in ['deep_gemm', 'nvjet', 'gemm', 'matmul', 'splitkreduce']):
            ext_id = event.get('args', {}).get('External id')
            tp_rank = extract_tp_rank(event.get('pid'))
            dims = None
            if ext_id is not None:
                dims = cpu_op_dims.get((tp_rank, ext_id)) or cpu_op_dims.get((tp_rank, ext_id - 1))
            if dims:
                m, n, k = dims[0], dims[1], dims[2]
                layer_type = classify_layer_type(m, n, k, name)
            else:
                # Try parse deep_gemm template
                match = re.search(r'deep_gemm[^<]*<[^,]*,\s*(\d+)u,\s*(\d+)u', name)
                if match:
                    n_, k_ = int(match.group(1)), int(match.group(2))
                    layer_type = classify_layer_type(992, n_, k_, name)
                else:
                    layer_type = 'QKVO'
        # Activations
        if layer_type is None and any(x in name_lower for x in ['act_and_mul', 'silu', 'gelu', 'activation']):
            layer_type = 'FFN'
        if layer_type is None and any(x in name_lower for x in ['moe', 'router', 'topk', 'expert_tokens', 'router_gemm']):
            layer_type = 'FFN'
        if layer_type is None and any(x in name_lower for x in ['quant', 'per_token_group']):
            layer_type = 'QKVO'
        if layer_type is None and any(x in name_lower for x in ['kv_buffer', 'kv_cache', 'mla_k', 'mla_v']):
            layer_type = 'QKVO'
        if layer_type is None:
            layer_type = 'Other'
        lt = layer_times[layer_type]
        lt['time_us'] += dur
        lt['count'] += 1
    total_time = sum(lt['time_us'] for lt in layer_times.values())
    for lt in layer_times.values():
        lt['percentage'] = (lt['time_us'] / total_time * 100) if total_time > 0 else 0
        lt['time_ms'] = lt['time_us'] / 1000.0
        lt['time_ms_per_gpu'] = lt['time_us'] / num_gpus / 1000.0
    layer_times['_total'] = {
        'time_us': total_time,
        'time_ms': total_time / 1000.0,
        'time_ms_per_gpu': total_time / num_gpus / 1000.0,
        'num_gpus': num_gpus,
    }
    return layer_times


###############################################################################
# Communication overlap analysis
###############################################################################

def analyze_communication_overlap(events: List[Dict[str, Any]], warmup_threshold_s: float = 1.0) -> Dict[str, Any]:
    """Analyse communication overlap with compute across GPUs.

    The returned dict contains overall communication time broken down into
    same‑GPU overlap, cross‑GPU pipeline overlap, exposed time and warmup.
    It also contains per‑type breakdown for NCCL operations.
    """
    from collections import defaultdict
    warmup_threshold_us = warmup_threshold_s * 1e6
    # Group kernels per GPU
    kernels_by_gpu = defaultdict(list)
    for e in events:
        if e.get('cat') != 'kernel':
            continue
        name = e.get('name', '')
        ts = e.get('ts', 0)
        dur = e.get('dur', 0)
        pid = e.get('pid')
        tid = e.get('tid')
        if dur <= 0:
            continue
        name_lower = name.lower()
        is_comm = any(x in name_lower for x in ['nccl', 'cross_device_reduce', 'allreduce', 'allgather', 'all_gather', 'reducescatter', 'reduce_scatter', 'alltoall', 'broadcast'])
        kernels_by_gpu[pid].append({
            'name': name,
            'ts': ts,
            'dur': dur,
            'end': ts + dur,
            'tid': tid,
            'is_comm': is_comm,
        })
    gpus = sorted(kernels_by_gpu.keys())
    for gpu in kernels_by_gpu:
        kernels_by_gpu[gpu].sort(key=lambda x: x['ts'])
    total_comm_time = 0
    same_gpu_overlap = 0
    cross_gpu_overlap = 0
    no_overlap = 0
    warmup_time = 0
    warmup_count = 0
    comm_by_type = defaultdict(lambda: {
        'count': 0, 'time_us': 0,
        'same_gpu_overlap_us': 0, 'cross_gpu_overlap_us': 0, 'no_overlap_us': 0,
        'warmup_count': 0, 'warmup_time_us': 0,
    })
    for gpu, kernels in kernels_by_gpu.items():
        other_gpus = [g for g in gpus if g != gpu]
        for ck in kernels:
            if not ck['is_comm']:
                continue
            ck_start, ck_end = ck['ts'], ck['end']
            ck_dur = ck['dur']
            # Identify type
            name_lower = ck['name'].lower()
            if 'cross_device_reduce' in name_lower:
                kernel_type = 'cross_device_reduce'
            elif 'allreduce' in name_lower:
                kernel_type = 'allreduce'
            elif 'allgather' in name_lower or 'all_gather' in name_lower:
                kernel_type = 'all_gather'
            elif 'reducescatter' in name_lower or 'reduce_scatter' in name_lower:
                kernel_type = 'reduce_scatter'
            else:
                kernel_type = 'other_comm'
            # Warmup detection
            if ck_dur > warmup_threshold_us:
                warmup_time += ck_dur
                warmup_count += 1
                comm_by_type[kernel_type]['warmup_count'] += 1
                comm_by_type[kernel_type]['warmup_time_us'] += ck_dur
                continue
            total_comm_time += ck_dur
            comm_by_type[kernel_type]['count'] += 1
            comm_by_type[kernel_type]['time_us'] += ck_dur
            # Same GPU overlap
            same_gpu_compute = [k for k in kernels if not k['is_comm'] and k['ts'] < ck_end and k['end'] > ck_start and k['tid'] != ck['tid']]
            cross_gpu_compute = []
            for other_gpu in other_gpus:
                for ok in kernels_by_gpu[other_gpu]:
                    if not ok['is_comm'] and ok['ts'] < ck_end and ok['end'] > ck_start:
                        cross_gpu_compute.append(ok)
            same_overlap = 0
            if same_gpu_compute:
                intervals = []
                for ok in same_gpu_compute:
                    intervals.append((max(ck_start, ok['ts']), min(ck_end, ok['end'])))
                intervals.sort()
                merged = [intervals[0]] if intervals else []
                for s, e in intervals[1:]:
                    if s <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                    else:
                        merged.append((s, e))
                same_overlap = sum(e - s for s, e in merged)
            cross_overlap_time = 0
            if cross_gpu_compute:
                intervals = []
                for ok in cross_gpu_compute:
                    intervals.append((max(ck_start, ok['ts']), min(ck_end, ok['end'])))
                intervals.sort()
                merged = [intervals[0]] if intervals else []
                for s, e in intervals[1:]:
                    if s <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                    else:
                        merged.append((s, e))
                cross_overlap_time = sum(e - s for s, e in merged)
            # Use the maximum overlap (since overlaps can overlap)
            total_overlap_time = max(same_overlap, cross_overlap_time)
            exposed = ck_dur - total_overlap_time
            same_gpu_overlap += same_overlap
            cross_gpu_overlap += cross_overlap_time
            no_overlap += exposed
            comm_by_type[kernel_type]['same_gpu_overlap_us'] += same_overlap
            comm_by_type[kernel_type]['cross_gpu_overlap_us'] += cross_overlap_time
            comm_by_type[kernel_type]['no_overlap_us'] += exposed
    return {
        'total_comm_time_us': total_comm_time,
        'same_gpu_overlap_us': same_gpu_overlap,
        'cross_gpu_overlap_us': cross_gpu_overlap,
        'exposed_time_us': no_overlap,
        'warmup_time_us': warmup_time,
        'warmup_count': warmup_count,
        'by_type': dict(comm_by_type),
        'num_gpus': max(len(gpus), 1),
    }


###############################################################################
# Network roofline analysis
###############################################################################

def analyze_network_roofline(events: List[Dict[str, Any]], gemm_infos: List[GemmInfo],
                             gpu_specs: GPUSpecs, tp_degree: int) -> Dict[str, Any]:
    """Analyse network communication roofline for tensor parallel GEMMs.

    Returns a dict describing critical arithmetic intensities and per‑phase
    operations.  See the original implementation for details.
    """
    # NVLink bandwidth in bytes/s
    nvlink_bw_bytes = gpu_specs.nvlink_bw_gb_s * 1e9
    def peak_flops_for_dtype(dtype: str) -> float:
        return get_dtype_peak_tflops(dtype, gpu_specs) * 1e12
    dtype_list = ['fp8', 'bf16', 'fp16']
    critical_ai_hbm = {dt: (peak_flops_for_dtype(dt) / (gpu_specs.memory_bw_tb_s * 1e12) if gpu_specs.memory_bw_tb_s > 0 else float('inf')) for dt in dtype_list}
    critical_ai_network = {dt: (peak_flops_for_dtype(dt) / nvlink_bw_bytes if nvlink_bw_bytes > 0 else float('inf')) for dt in dtype_list}
    # Identify AllReduce kernels
    # Treat both "allreduce" and "cross_device_reduce" kernels as AllReduce
    allreduce_durations = [
        e.get('dur', 0)
        for e in events
        if e.get('cat') == 'kernel'
        and (('allreduce' in e.get('name', '').lower()) or ('cross_device_reduce' in e.get('name', '').lower()))
        and e.get('dur', 0) <= 1e6
    ]
    results = {
        'critical_ai_hbm': critical_ai_hbm,
        'critical_ai_network': critical_ai_network,
        'nvlink_bw_gb_s': gpu_specs.nvlink_bw_gb_s,
        'tp_degree': tp_degree,
        'phases': {},
        'allreduce_stats': {
            'count': len(allreduce_durations),
            'total_time_us': sum(allreduce_durations),
            'avg_time_us': (sum(allreduce_durations) / len(allreduce_durations)) if allreduce_durations else 0,
        },
    }
    # Split gemms into prefill vs decode based on M dimension
    prefill_gemms = [g for g in gemm_infos if g.m > 128]
    decode_gemms = [g for g in gemm_infos if g.m <= 128]
    for phase_name, phase_gemms in [('prefill', prefill_gemms), ('decode', decode_gemms)]:
        if not phase_gemms:
            continue
        # Pick the most common M
        m_values = [g.m for g in phase_gemms]
        M = max(set(m_values), key=m_values.count)
        # Aggregate by (N,K,dtype,out_dtype)
        dim_stats = defaultdict(lambda: {'count': 0, 'time_us': 0, 'flops': 0})
        for g in phase_gemms:
            key = (g.n, g.k, g.dtype, g.output_dtype or 'bf16')
            dim_stats[key]['count'] += 1
            dim_stats[key]['time_us'] += g.duration_us
            dim_stats[key]['flops'] += g.flops
        phase_ops = []
        hidden_size = 7168  # DeepSeek assumption
        for (N, K, dt, out_dt), stats in dim_stats.items():
            is_row_parallel = (N == hidden_size)
            flops_per_gpu = 2 * M * N * K
            if is_row_parallel:
                dtype_bytes = int(get_bytes_per_element(out_dt)) if out_dt else 2
                allreduce_bytes = 2 * (tp_degree - 1) / tp_degree * M * N * dtype_bytes
                network_ai = flops_per_gpu / allreduce_bytes if allreduce_bytes > 0 else float('inf')
                t_network_us = allreduce_bytes / nvlink_bw_bytes * 1e6
            else:
                allreduce_bytes = 0
                network_ai = float('inf')
                t_network_us = 0
            peak_flops_op = peak_flops_for_dtype(dt)
            t_compute_us = flops_per_gpu / peak_flops_op * 1e6 if peak_flops_op > 0 else float('inf')
            bound = 'network' if is_row_parallel and network_ai < critical_ai_network.get(dt, float('inf')) else 'compute'
            phase_ops.append({
                'M': M, 'N': N, 'K': K, 'dtype': dt, 'out_dtype': out_dt,
                'parallelism': 'row-parallel' if is_row_parallel else 'column-parallel',
                'flops_per_gpu': flops_per_gpu,
                'allreduce_bytes': allreduce_bytes,
                'network_ai': network_ai,
                't_compute_us': t_compute_us,
                't_network_us': t_network_us,
                'bound': bound,
                'kernel_count': stats['count'],
                'measured_time_us': stats['time_us'],
            })
        results['phases'][phase_name] = {
            'M': M,
            'operations': phase_ops,
            'total_gemm_time_us': sum(g.duration_us for g in phase_gemms),
            'total_gemm_count': len(phase_gemms),
        }
    return results


###############################################################################
# MFU/MBU annotation of trace events
###############################################################################

def add_mfu_to_trace(trace_data: Dict[str, Any], gpu_specs: GPUSpecs, config: Config) -> Dict[str, Any]:
    """Annotate kernel events in a trace with MFU/MBU metrics.

    This function mirrors the logic of :func:`analyze_gemm_kernels` but writes
    metrics back into the ``args`` dict of each kernel or CPU op.  It can be
    used to generate an enriched trace JSON which can be visualised in
    Chrome’s trace viewer.
    """
    events = trace_data.get('traceEvents', [])
    cpu_op_dims = build_cpu_op_dims_map(events)
    sibling_dims = build_sibling_dims_map(events, cpu_op_dims)
    kernel_times_by_key = defaultdict(float)
    for event in events:
        if event.get('cat') == 'kernel':
            ext_id = event.get('args', {}).get('External id')
            tp_rank = extract_tp_rank(event.get('pid'))
            if ext_id is not None:
                kernel_times_by_key[(tp_rank, ext_id)] += event.get('dur', 0)
    modified_count = 0
    for event in events:
        if event.get('cat') == 'kernel':
            name = event.get('name', '')
            classification = classify_kernel(name)
            if not classification.is_gemm:
                continue
            duration_us = event.get('dur', 0)
            if duration_us <= 0:
                continue
            ext_id = event.get('args', {}).get('External id')
            tp_rank = extract_tp_rank(event.get('pid'))
            grid = event.get('args', {}).get('grid', [1, 1, 1])
            # Extract dims
            dims = None
            if ext_id is not None:
                dims = cpu_op_dims.get((tp_rank, ext_id)) or cpu_op_dims.get((tp_rank, ext_id - 1)) or cpu_op_dims.get((tp_rank, ext_id + 1))
            if dims is None and classification.source == 'deep_gemm':
                parsed = parse_deep_gemm_kernel_dims(name, grid, None)
                if parsed:
                    m_, n_, k_, dtype_ = parsed
                    # Use the configured model dtype for the output rather than a fixed
                    # BF16.  See discussion in :func:`analyze_gemm_kernels`.
                    dims = (m_, n_, k_, 'fp8', 'fp8', normalize_dtype(config.model_dtype))
            inferred_layer_type = None
            if dims is None and ext_id is None:
                inferred = infer_cuda_graph_kernel_dims(name, grid, config, sibling_dims=sibling_dims)
                if inferred:
                    m_, n_, k_, dtype_, inferred_layer_type = inferred
                    # Use model dtype for fallback output dtype
                    c_dtype = normalize_dtype(config.model_dtype)
                    dims = (m_, n_, k_, dtype_, dtype_, c_dtype)
            if dims is None:
                continue
            if len(dims) >= 6:
                m, n, k, a_dtype, b_dtype, c_dtype = dims[:6]
            elif len(dims) == 5:
                m, n, k, input_dtype, output_dtype = dims
                a_dtype = b_dtype = input_dtype
                c_dtype = output_dtype
            else:
                m, n, k, input_dtype = dims
                a_dtype = b_dtype = input_dtype
                c_dtype = normalize_dtype(config.model_dtype)
            if m <= 0 or n <= 0 or k <= 0:
                continue
            if not a_dtype and classification.dtype:
                a_dtype = classification.dtype
            if not b_dtype and classification.dtype:
                b_dtype = classification.dtype
            if not c_dtype and classification.dtype:
                c_dtype = 'bf16' if classification.dtype == 'fp8' else classification.dtype
            dtype = compute_dtype_from_inputs(a_dtype, b_dtype)
            # Metrics
            flops = calculate_gemm_flops(m, n, k)
            bytes_accessed = calculate_gemm_bytes(m, n, k, a_dtype, b_dtype, c_dtype)
            peak_tflops = get_dtype_peak_tflops(dtype, gpu_specs)
            mfu = calculate_mfu(flops, duration_us, peak_tflops)
            mbu = calculate_mbu(bytes_accessed, duration_us, gpu_specs.memory_bw_tb_s)
            ai = calculate_arithmetic_intensity(flops, bytes_accessed)
            roofline_tflops, roofline_bound = calculate_roofline_tflops(ai, gpu_specs, peak_tflops)
            achieved_tflops = (flops / 1e12) / (duration_us / 1e6)
            achieved_bw_tb_s = (bytes_accessed / 1e12) / (duration_us / 1e6)
            layer_type = inferred_layer_type if inferred_layer_type else classify_layer_type(m, n, k, name)
            if 'args' not in event:
                event['args'] = {}
            args = event['args']
            args['MFU (%)'] = round(mfu, 2)
            args['MBU (%)'] = round(mbu, 2)
            args['Achieved TFLOPS'] = round(achieved_tflops, 2)
            args['Peak TFLOPS'] = round(peak_tflops, 2)
            args['Roofline TFLOPS'] = round(roofline_tflops, 2)
            args['Roofline Bound'] = roofline_bound
            args['Achieved BW (TB/s)'] = round(achieved_bw_tb_s, 3)
            args['Peak BW (TB/s)'] = round(gpu_specs.memory_bw_tb_s, 2)
            args['Arithmetic Intensity'] = round(ai, 2)
            args['FLOPs'] = flops
            args['Bytes'] = bytes_accessed
            args['GEMM M'] = m
            args['GEMM N'] = n
            args['GEMM K'] = k
            # Record the compute dtype (kernel execution precision) and individual input/output dtypes.
            # ``dtype`` is the compute/accumulator dtype used for performance calculations.
            args['GEMM dtype'] = dtype
            args['GEMM A dtype'] = a_dtype
            args['GEMM B dtype'] = b_dtype
            args['GEMM C dtype'] = c_dtype
            # Aggregate input and output dtypes: if A and B have the same dtype
            # then the input dtype is that dtype; otherwise denote as 'mixed'.
            input_dtype = a_dtype if a_dtype == b_dtype else 'mixed'
            args['GEMM input dtype'] = input_dtype
            # The output dtype corresponds to C's dtype.
            args['GEMM output dtype'] = c_dtype
            args['Layer Type'] = layer_type
            modified_count += 1
        elif event.get('cat') == 'cpu_op':
            name = event.get('name', '')
            if not any(x in name.lower() for x in ['deep_gemm', 'fp8_gemm']):
                continue
            dims = extract_dimensions_from_cpu_op(event)
            if dims:
                if len(dims) >= 6:
                    m, n, k, a_dtype, b_dtype, c_dtype = dims[:6]
                    dtype = compute_dtype_from_inputs(a_dtype, b_dtype)
                elif len(dims) == 5:
                    m, n, k, input_dtype, output_dtype = dims
                    a_dtype = b_dtype = input_dtype
                    c_dtype = output_dtype
                    dtype = compute_dtype_from_inputs(a_dtype, b_dtype)
                else:
                    m, n, k, input_dtype = dims
                    a_dtype = b_dtype = input_dtype
                    # Use model default dtype for the output when the profiler does
                    # not record C dtype for this CPU op
                    c_dtype = normalize_dtype(config.model_dtype)
                    dtype = compute_dtype_from_inputs(a_dtype, b_dtype)
                ext_id = event.get('args', {}).get('External id')
                tp_rank = extract_tp_rank(event.get('pid'))
                key = (tp_rank, ext_id) if ext_id is not None else None
                duration_us = event.get('dur', 0)
                if key and key in kernel_times_by_key:
                    duration_us = kernel_times_by_key[key]
                if duration_us > 0:
                    flops = calculate_gemm_flops(m, n, k)
                    bytes_accessed = calculate_gemm_bytes(m, n, k, a_dtype, b_dtype, c_dtype)
                    peak_tflops = get_dtype_peak_tflops(dtype, gpu_specs)
                    mfu = calculate_mfu(flops, duration_us, peak_tflops)
                    mbu = calculate_mbu(bytes_accessed, duration_us, gpu_specs.memory_bw_tb_s)
                    ai = calculate_arithmetic_intensity(flops, bytes_accessed)
                    roofline_tflops, roofline_bound = calculate_roofline_tflops(ai, gpu_specs, peak_tflops)
                    if 'args' not in event:
                        event['args'] = {}
                    args = event['args']
                    args['MFU (%)'] = round(mfu, 2)
                    args['MBU (%)'] = round(mbu, 2)
                    args['Achieved TFLOPS'] = round((flops / 1e12) / (duration_us / 1e6), 2)
                    args['Roofline TFLOPS'] = round(roofline_tflops, 2)
                    args['Roofline Bound'] = roofline_bound
                    args['Arithmetic Intensity'] = round(ai, 2)
                    # Annotate compute, individual input/output dtypes and aggregate input/output dtypes.
                    args['GEMM dtype'] = dtype
                    args['GEMM A dtype'] = a_dtype
                    args['GEMM B dtype'] = b_dtype
                    args['GEMM C dtype'] = c_dtype
                    input_dtype = a_dtype if a_dtype == b_dtype else 'mixed'
                    args['GEMM input dtype'] = input_dtype
                    args['GEMM output dtype'] = c_dtype
                    modified_count += 1
    # Could print or log modified_count here if desired
    return trace_data


###############################################################################
# Trace loading and saving
###############################################################################

def load_trace(input_path: str) -> Dict[str, Any]:
    """Load a trace JSON or JSON.GZ file from disk."""
    path = Path(input_path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def save_trace(trace_data: Dict[str, Any], output_path: str, compress: bool = False):
    """Save a trace dict to disk, optionally compressing with gzip."""
    path = Path(output_path)
    if compress or path.suffix == '.gz':
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(trace_data, f)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2)


###############################################################################
# Summary reporting
###############################################################################

def print_summary(gemm_infos: List[GemmInfo], layer_times: Dict[str, Any], gpu_specs: GPUSpecs,
                  comm_overlap: Optional[Dict[str, Any]] = None,
                  network_roofline: Optional[Dict[str, Any]] = None,
                  events: Optional[List[Dict[str, Any]]] = None,
                  grouped_gemm_infos: Optional[List[GroupedGemmInfo]] = None):
    """Print a comprehensive analysis summary to stdout.

    This function consolidates the print logic from the original script into a
    single place.  It accepts pre‑computed GEMM information, layer timing
    breakdown, communication overlap statistics, network roofline analysis and
    grouped GEMM information.  All printing is performed here to avoid
    scattering summary code throughout the analysis.
    """
    if not gemm_infos:
        print("No GEMM operations found")
        return
    num_gpus = layer_times.get('_total', {}).get('num_gpus', 1)
    total_flops = sum(g.flops for g in gemm_infos)
    total_bytes = sum(g.bytes_accessed for g in gemm_infos)
    total_time_us = sum(g.duration_us for g in gemm_infos)
    per_gpu_time_us = total_time_us / num_gpus
    per_gpu_flops = total_flops / num_gpus
    per_gpu_time_s = per_gpu_time_us / 1e6 if num_gpus > 0 else 0
    overall_tflops = (per_gpu_flops / 1e12) / per_gpu_time_s if per_gpu_time_s > 0 else 0
    def fmt_tflops(tf: float) -> str:
        return f"{tf/1000:.1f} PFLOPS" if tf >= 1000 else f"{tf:.1f} TFLOPS"
    avg_mfu = (sum(g.mfu * g.duration_us for g in gemm_infos) / total_time_us) if total_time_us > 0 else 0
    avg_mbu = (sum(g.mbu * g.duration_us for g in gemm_infos) / total_time_us) if total_time_us > 0 else 0
    print("\n" + "="*80)
    print("GEMM/MatMul Analysis Summary (MFU, MBU, Roofline)")
    print("="*80)
    print(f"GPU: {gpu_specs.name} (x{num_gpus} GPUs in trace)")
    if gpu_specs.fp4_tflops > 0:
        print(f"Peak FP4: {fmt_tflops(gpu_specs.fp4_tflops)}")
    print(f"Peak FP8: {fmt_tflops(gpu_specs.fp8_tflops)}")
    print(f"Peak BF16: {fmt_tflops(gpu_specs.fp16_tflops)}")
    print(f"Peak Memory BW: {gpu_specs.memory_bw_tb_s:.2f} TB/s")
    print(f"L2 Cache: {gpu_specs.l2_cache_mb:.0f} MB")
    print("-"*80)
    print(f"Total GEMM kernels analysed: {len(gemm_infos)} (with known M dimension)")
    print(f"Total GEMM FLOPs: {total_flops / 1e12:.2f} TFLOPs ({per_gpu_flops / 1e12:.2f} per GPU)")
    print(f"Total GEMM bytes: {total_bytes / 1e9:.2f} GB")
    print(f"Total GEMM time: {total_time_us/1000:.2f} ms ({per_gpu_time_us/1000:.2f} ms per GPU)")
    print(f"Average TFLOPS (per GPU): {overall_tflops:.2f}")
    print(f"Weighted Average MFU: {avg_mfu:.2f}%")
    print(f"Weighted Average MBU: {avg_mbu:.2f}%")
    # Group by dtype
    dtype_groups = defaultdict(list)
    for g in gemm_infos:
        dtype_groups[g.dtype].append(g)
    print("-"*80)
    print("By Data Type:")
    for dtype, ops in dtype_groups.items():
        time_sum = sum(g.duration_us for g in ops)
        avg_mfu_d = (sum(g.mfu * g.duration_us for g in ops) / time_sum) if time_sum > 0 else 0
        avg_mbu_d = (sum(g.mbu * g.duration_us for g in ops) / time_sum) if time_sum > 0 else 0
        print(f"  {dtype.upper():<5}: {len(ops)} ops, {time_sum/1000/num_gpus:.2f} ms/GPU, MFU: {avg_mfu_d:.2f}%, MBU: {avg_mbu_d:.2f}%")
    # Roofline bound breakdown
    memory_bound = [g for g in gemm_infos if g.roofline_bound == 'memory']
    compute_bound = [g for g in gemm_infos if g.roofline_bound == 'compute']
    print("-"*80)
    print("By Roofline Bound:")
    if memory_bound:
        mb_time = sum(g.duration_us for g in memory_bound)
        mb_avg_mbu = (sum(g.mbu * g.duration_us for g in memory_bound) / mb_time) if mb_time > 0 else 0
        mb_avg_bw = (sum(g.achieved_bw_tb_s * g.duration_us for g in memory_bound) / mb_time * 1000) if mb_time > 0 else 0
        print(f"  Memory-bound: {len(memory_bound)} ops, {mb_time/1000/num_gpus:.2f} ms/GPU ({mb_time/total_time_us*100:.1f}%)")
        print(f"               Avg MBU: {mb_avg_mbu:.1f}%, Avg BW: {mb_avg_bw:.0f} GB/s")
    if compute_bound:
        cb_time = sum(g.duration_us for g in compute_bound)
        cb_avg_mfu = (sum(g.mfu * g.duration_us for g in compute_bound) / cb_time) if cb_time > 0 else 0
        print(f"  Compute-bound: {len(compute_bound)} ops, {cb_time/1000/num_gpus:.2f} ms/GPU ({cb_time/total_time_us*100:.1f}%)")
        print(f"                Avg MFU: {cb_avg_mfu:.1f}%")
    # Phase breakdown
    prefill_ops = [g for g in gemm_infos if g.m > 128]
    decode_ops = [g for g in gemm_infos if g.m <= 128]
    print("-"*80)
    print("By Phase (based on M dimension):")
    if prefill_ops:
        pf_time = sum(g.duration_us for g in prefill_ops)
        pf_avg_mfu = (sum(g.mfu * g.duration_us for g in prefill_ops) / pf_time) if pf_time > 0 else 0
        pf_avg_mbu = (sum(g.mbu * g.duration_us for g in prefill_ops) / pf_time) if pf_time > 0 else 0
        pf_avg_bw = (sum(g.achieved_bw_tb_s * g.duration_us for g in prefill_ops) / pf_time * 1000) if pf_time > 0 else 0
        common_m = max(set([g.m for g in prefill_ops]), key=lambda x: sum(1 for g in prefill_ops if g.m == x))
        print(f"  Prefill (M={common_m}): {len(prefill_ops)} ops, {pf_time/1000/num_gpus:.2f} ms/GPU ({pf_time/total_time_us*100:.1f}%)")
        print(f"                 MFU: {pf_avg_mfu:.1f}%, MBU: {pf_avg_mbu:.1f}%, BW: {pf_avg_bw:.0f} GB/s")
    if decode_ops:
        dc_time = sum(g.duration_us for g in decode_ops)
        dc_avg_mfu = (sum(g.mfu * g.duration_us for g in decode_ops) / dc_time) if dc_time > 0 else 0
        dc_avg_mbu = (sum(g.mbu * g.duration_us for g in decode_ops) / dc_time) if dc_time > 0 else 0
        dc_avg_bw = (sum(g.achieved_bw_tb_s * g.duration_us for g in decode_ops) / dc_time * 1000) if dc_time > 0 else 0
        common_m = max(set([g.m for g in decode_ops]), key=lambda x: sum(1 for g in decode_ops if g.m == x)) if decode_ops else 0
        print(f"  Decode (M={common_m}): {len(decode_ops)} ops, {dc_time/1000/num_gpus:.2f} ms/GPU ({dc_time/total_time_us*100:.1f}%)")
        if dc_avg_mbu > 100:
            print(f"                 MFU: {dc_avg_mfu:.1f}%, MBU: {dc_avg_mbu:.1f}% (INVALID - dimension inference error)")
            print(f"                 BW: {dc_avg_bw:.0f} GB/s")
            print(f"                 Note: MBU>100% is physically impossible and indicates dimension inference errors.")
        else:
            print(f"                 MFU: {dc_avg_mfu:.1f}%, MBU: {dc_avg_mbu:.1f}%, BW: {dc_avg_bw:.0f} GB/s")
    # Top 10 by MFU
    print("\n" + "-"*80)
    print("Top 10 GEMMs by MFU:")
    # Include per-operand dtypes when reporting top kernels.  Compute dtype
    # ``g.dtype`` may differ from individual operand dtypes (A/B/C).  Show
    # A/B/C explicitly for clarity.
    for i, g in enumerate(sorted(gemm_infos, key=lambda g: g.mfu, reverse=True)[:10]):
        roof_eff = (g.tflops / g.roofline_tflops * 100) if g.roofline_tflops > 0 else 0
        mbu_flag = " (INVALID)" if g.mbu > 100 else ""
        print(f"  {i+1}. M={g.m}, N={g.n}, K={g.k}, compute={g.dtype}, A={g.a_dtype}, B={g.b_dtype}, C={g.c_dtype}, {g.layer_type}: MFU={g.mfu:.2f}%{mbu_flag}, MBU={g.mbu:.1f}%, Roofline Eff={roof_eff:.1f}%")
        print(f"     Achieved={g.tflops:.1f} TFLOPS, BW={g.achieved_bw_tb_s*1000:.0f} GB/s, AI={g.arithmetic_intensity:.1f}")
        print(f"     [Trace: TP{g.tp_rank}, ts={g.timestamp_us}]")
    # Bottom 10 by MFU (duration > 5us)
    print("\nBottom 10 GEMMs by MFU (duration > 5us):")
    significant_ops = [g for g in gemm_infos if g.duration_us > 5]
    for i, g in enumerate(sorted(significant_ops, key=lambda g: g.mfu)[:10]):
        roof_eff = (g.tflops / g.roofline_tflops * 100) if g.roofline_tflops > 0 else 0
        print(f"  {i+1}. M={g.m}, N={g.n}, K={g.k}, compute={g.dtype}, A={g.a_dtype}, B={g.b_dtype}, C={g.c_dtype}, {g.layer_type}: MFU={g.mfu:.2f}%, MBU={g.mbu:.1f}%, Roofline Eff={roof_eff:.1f}%")
        print(f"     Achieved={g.tflops:.1f} TFLOPS, BW={g.achieved_bw_tb_s*1000:.0f} GB/s, AI={g.arithmetic_intensity:.1f}")
        print(f"     [Trace: TP{g.tp_rank}, ts={g.timestamp_us}]")
    # Top 10 by MBU
    print("\n" + "-"*80)
    print("Top 10 GEMMs by MBU (Memory Bandwidth Utilisation):")
    high_mbu = [g for g in gemm_infos if g.mbu > 100]
    if high_mbu:
        print("  WARNING: MBU > 100% indicates dimension inference error!")
        print("           These kernels likely lack External ID and were inferred incorrectly.")
        print()
    for i, g in enumerate(sorted(gemm_infos, key=lambda g: g.mbu, reverse=True)[:10]):
        if g.mbu > 100:
            weight_time_us = g.weight_bytes / (gpu_specs.memory_bw_tb_s * 1e12) * 1e6 if g.weight_bytes > 0 else 0
            print(f"  {i+1}. M={g.m}, N={g.n}, K={g.k}, compute={g.dtype}, A={g.a_dtype}, B={g.b_dtype}, C={g.c_dtype}, {g.layer_type}: MBU={g.mbu:.1f}% (INVALID)")
            print(f"     Weight load time {weight_time_us:.1f}µs > kernel {g.duration_us:.1f}µs")
        else:
            peak_gb_s = gpu_specs.memory_bw_tb_s * 1000
            print(f"  {i+1}. M={g.m}, N={g.n}, K={g.k}, compute={g.dtype}, A={g.a_dtype}, B={g.b_dtype}, C={g.c_dtype}, {g.layer_type}: MBU={g.mbu:.1f}%, BW={g.achieved_bw_tb_s*1000:.0f} GB/s (peak: {peak_gb_s:.0f} GB/s)")
            print(f"     MFU={g.mfu:.2f}%, AI={g.arithmetic_intensity:.1f}, {g.roofline_bound}-bound")
            print(f"     [Trace: TP{g.tp_rank}, ts={g.timestamp_us}]")
    # Top 10 by time
    print("\n" + "-"*80)
    print("Top 10 GEMMs by time:")
    for i, g in enumerate(sorted(gemm_infos, key=lambda g: g.duration_us, reverse=True)[:10]):
        print(f"  {i+1}. M={g.m}, N={g.n}, K={g.k}, compute={g.dtype}, A={g.a_dtype}, B={g.b_dtype}, C={g.c_dtype}, {g.layer_type}: {g.duration_us:.2f}us, MFU={g.mfu:.2f}%, {g.tflops:.1f} TFLOPS, AI={g.arithmetic_intensity:.1f}")
        print(f"     [Trace: TP{g.tp_rank}, ts={g.timestamp_us}]")
    # Grouped GEMM summary
    if grouped_gemm_infos:
        print("\n" + "="*80)
        print("Grouped GEMM Analysis (Fused MoE)")
        print("="*80)
        print(f"GPU: {gpu_specs.name}")
        print(f"Peak FP8: {gpu_specs.fp8_tflops/1000:.1f} PFLOPS")
        print("-"*80)
        prefill_ops = [g for g in grouped_gemm_infos if g.external_id > 0]
        decode_ops = [g for g in grouped_gemm_infos if g.external_id == 0]
        total_grouped_flops = sum(g.total_flops for g in grouped_gemm_infos)
        total_grouped_bytes = sum(g.total_bytes for g in grouped_gemm_infos)
        total_grouped_time = sum(g.duration_us for g in grouped_gemm_infos)
        print(f"Total grouped GEMM operations: {len(grouped_gemm_infos)}")
        print(f"  Prefill ops (with External ID): {len(prefill_ops)}")
        print(f"  Decode ops (CUDA Graph/inferred): {len(decode_ops)}")
        print(f"Total FLOPs: {total_grouped_flops/1e12:.2f} TFLOPs")
        print(f"Total bytes: {total_grouped_bytes/1e9:.2f} GB")
        print()
        if prefill_ops:
            print("Prefill Phase (fused_moe_kernel with External ID):")
            pf_flops = sum(g.total_flops for g in prefill_ops)
            pf_time = sum(g.duration_us for g in prefill_ops)
            pf_bytes = sum(g.total_bytes for g in prefill_ops)
            sample = prefill_ops[0]
            print(f"  Dimensions: {sample.num_tokens} tokens × top_{sample.top_k} experts")
            print(f"              {sample.num_experts} total experts, hidden={sample.hidden_size}")
            print(f"              w1_inter={sample.w1_intermediate}, w2_inter={sample.w2_intermediate}")
            print(f"  Token-expert pairs: {sample.total_token_expert_pairs}")
            print()
            pf_avg_mfu = (sum(g.mfu * g.duration_us for g in prefill_ops) / pf_time) if pf_time > 0 else 0
            pf_avg_mbu = (sum(g.mbu * g.duration_us for g in prefill_ops) / pf_time) if pf_time > 0 else 0
            pf_avg_bw = (sum(g.achieved_bw_tb_s * g.duration_us for g in prefill_ops) / pf_time * 1000) if pf_time > 0 else 0
            pf_avg_tflops = (pf_flops / 1e12) / (pf_time / 1e6) if pf_time > 0 else 0
            print(f"  Total time: {pf_time/1000:.2f} ms ({len(prefill_ops)} ops)")
            print(f"  Total FLOPs: {pf_flops/1e12:.2f} TFLOPs")
            print(f"  Achieved: {pf_avg_tflops:.1f} TFLOPS")
            print(f"  MFU: {pf_avg_mfu:.1f}%, MBU: {pf_avg_mbu:.1f}%")
            print(f"  Bandwidth: {pf_avg_bw:.0f} GB/s")
            print(f"  Arithmetic Intensity: {sample.arithmetic_intensity:.1f} FLOPs/byte")
            print(f"  Roofline bound: {sample.roofline_bound}")
            print()
            print("  Top 5 Prefill MoE ops by MFU:")
            for i, g in enumerate(sorted(prefill_ops, key=lambda g: g.mfu, reverse=True)[:5]):
                print(f"    {i+1}. {g.num_tokens}tok×top{g.top_k}, {g.weight_dtype}: MFU={g.mfu:.1f}%, {g.tflops:.1f} TFLOPS, {g.duration_us:.1f}us")
                print(f"       [ExtID={g.external_id}, TP{g.tp_rank}]")
        if decode_ops:
            print()
            print("Decode Phase (fused_moe_kernel, CUDA Graph):")
            dc_flops = sum(g.total_flops for g in decode_ops)
            dc_time = sum(g.duration_us for g in decode_ops)
            dc_bytes = sum(g.total_bytes for g in decode_ops)
            w1_ops = [g for g in decode_ops if g.w1_intermediate > 0]
            w2_ops = [g for g in decode_ops if g.w2_intermediate > 0]
            if w1_ops or w2_ops:
                sample = w1_ops[0] if w1_ops else (w2_ops[0] if w2_ops else None)
                if sample:
                    print(f"  Dimensions (inferred): {sample.num_tokens} tokens × top_{sample.top_k} experts")
                    print(f"              {sample.num_experts} experts, hidden={sample.hidden_size}")
                    print(f"              w1_inter={sample.w1_intermediate if w1_ops else 0} (gate+up), w2_inter={sample.w2_intermediate if w2_ops else 0} (down)")
                    print()
            dc_avg_mfu = (sum(g.mfu * g.duration_us for g in decode_ops) / dc_time) if dc_time > 0 else 0
            dc_avg_mbu = (sum(g.mbu * g.duration_us for g in decode_ops) / dc_time) if dc_time > 0 else 0
            dc_avg_bw = (sum(g.achieved_bw_tb_s * g.duration_us for g in decode_ops) / dc_time * 1000) if dc_time > 0 else 0
            dc_avg_tflops = (dc_flops / 1e12) / (dc_time / 1e6) if dc_time > 0 else 0
            total_kernels = sum(g.num_kernels for g in decode_ops)
            print(f"  Total time: {dc_time/1000:.2f} ms/GPU ({total_kernels} kernels across all GPUs)")
            print(f"  Total FLOPs: {dc_flops/1e12:.4f} TFLOPs")
            print(f"  Achieved: {dc_avg_tflops:.1f} TFLOPS")
            print(f"  MFU: {dc_avg_mfu:.1f}%, MBU: {dc_avg_mbu:.1f}%")
            print(f"  Bandwidth: {dc_avg_bw:.0f} GB/s")
            if w1_ops and w2_ops:
                print()
                print("  By projection type:")
                w1_time = sum(g.duration_us for g in w1_ops)
                w1_flops = sum(g.total_flops for g in w1_ops)
                w1_mfu = (sum(g.mfu * g.duration_us for g in w1_ops) / w1_time) if w1_time > 0 else 0
                print(f"    W1 (gate+up): {w1_time/1000:.2f}ms, {w1_flops/1e12:.4f} TFLOPs, MFU={w1_mfu:.1f}%")
                w2_time = sum(g.duration_us for g in w2_ops)
                w2_flops = sum(g.total_flops for g in w2_ops)
                w2_mfu = (sum(g.mfu * g.duration_us for g in w2_ops) / w2_time) if w2_time > 0 else 0
                print(f"    W2 (down):    {w2_time/1000:.2f}ms, {w2_flops/1e12:.4f} TFLOPs, MFU={w2_mfu:.1f}%")
        print("-"*80)
    # Layer time breakdown
    print("\n" + "="*80)
    print("Layer Type Time Breakdown (All Kernels)")
    print("="*80)
    total_info = layer_times.get('_total', {})
    total_kernel_time = total_info.get('time_ms', 0)
    per_gpu_time = total_info.get('time_ms_per_gpu', total_kernel_time)
    print(f"Total kernel time (sum across {num_gpus} GPUs): {total_kernel_time:.2f} ms")
    print(f"Per-GPU average kernel time: {per_gpu_time:.2f} ms\n")
    layer_order = ['QKVO', 'SDPA', 'FFN', 'Normalization', 'Communication', 'Other']
    for layer_name in layer_order:
        lt = layer_times.get(layer_name, {})
        time_ms = lt.get('time_ms_per_gpu', 0)
        pct = lt.get('percentage', 0)
        count = lt.get('count', 0)
        print(f"  {layer_name:<15s}: {time_ms:10.2f} ms/GPU  ({pct:5.1f}%)  [{count:6d} kernels]")
    # Communication overlap
    print("\n" + "-"*80)
    print("Communication Overlap Analysis")
    print("-"*80)
    if comm_overlap:
        total_comm_us = comm_overlap['total_comm_time_us']
        same_us = comm_overlap['same_gpu_overlap_us']
        cross_us = comm_overlap['cross_gpu_overlap_us']
        exposed_us = comm_overlap['exposed_time_us']
        warmup_us = comm_overlap.get('warmup_time_us', 0)
        warmup_count = comm_overlap.get('warmup_count', 0)
        num_gpus_co = comm_overlap['num_gpus']
        total_comm_ms = total_comm_us / 1000 / num_gpus_co
        same_ms = same_us / 1000 / num_gpus_co
        cross_ms = cross_us / 1000 / num_gpus_co
        exposed_ms = exposed_us / 1000 / num_gpus_co
        if warmup_count > 0:
            print(f"  Warmup/barrier kernels excluded: {warmup_count} kernels, {warmup_us/1000/num_gpus_co:.2f} ms/GPU\n")
        print(f"  Total communication time (excluding warmup): {total_comm_ms:10.2f} ms/GPU")
        print()
        if total_comm_us > 0:
            same_pct = same_us / total_comm_us * 100
            cross_pct = cross_us / total_comm_us * 100
            exposed_pct = exposed_us / total_comm_us * 100
        else:
            same_pct = cross_pct = exposed_pct = 0
        print(f"    Same-GPU overlap:        {same_ms:10.2f} ms/GPU  ({same_pct:5.1f}%)")
        print(f"      (Compute on same GPU, different stream)")
        print(f"    Cross-GPU pipeline:      {cross_ms:10.2f} ms/GPU  ({cross_pct:5.1f}%)")
        print(f"      (Compute on other GPUs - pipeline)")
        print(f"    Exposed (no overlap):    {exposed_ms:10.2f} ms/GPU  ({exposed_pct:5.1f}%)")
        print()
        print("  By communication type:")
        by_type = comm_overlap['by_type']
        for ctype, data in sorted(by_type.items(), key=lambda x: -x[1]['time_us']):
            if data['count'] == 0 and data.get('warmup_count', 0) == 0:
                continue
            time_ms_type = data['time_us'] / 1000 / num_gpus_co
            cross_pct_type = (data['cross_gpu_overlap_us'] / data['time_us'] * 100) if data['time_us'] > 0 else 0
            exposed_pct_type = (data['no_overlap_us'] / data['time_us'] * 100) if data['time_us'] > 0 else 0
            warmup_info = ''
            if data.get('warmup_count', 0) > 0:
                warmup_ms_type = data['warmup_time_us'] / 1000 / num_gpus_co
                warmup_info = f" (+{data['warmup_count']} warmup, {warmup_ms_type:.1f}ms)"
            print(f"    {ctype:<25s}: {time_ms_type:8.2f} ms/GPU, {data['count']:5d} calls{warmup_info}")
            if data['count'] > 0:
                print(f"      Pipeline overlap: {cross_pct_type:5.1f}%, Exposed: {exposed_pct_type:5.1f}%")
    else:
        lt = layer_times.get('Communication', {})
        time_ms_co = lt.get('time_ms_per_gpu', 0)
        pct_co = lt.get('percentage', 0)
        count_co = lt.get('count', 0)
        print(f"  Total: {time_ms_co:10.2f} ms/GPU  ({pct_co:5.1f}%)  [{count_co:6d} kernels]")
        print("  (Run with full analysis to see overlap breakdown)")
    # Network roofline
    if network_roofline:
        print("\n" + "-"*80)
        print("Network Communication Roofline Analysis")
        print("-"*80)
        print("  Reference: https://jax-ml.github.io/scaling-book/roofline/\n")
        crit_hbm = network_roofline.get('critical_ai_hbm')
        crit_net = network_roofline.get('critical_ai_network')
        if isinstance(crit_hbm, dict) and isinstance(crit_net, dict):
            for dt in ['fp8', 'bf16', 'fp16']:
                print(f"    {dt.upper():>4s} HBM Roofline:      {crit_hbm[dt]:8.1f} FLOPs/byte")
                print(f"    {dt.upper():>4s} Network Roofline:  {crit_net[dt]:8.1f} FLOPs/byte")
        else:
            print(f"    HBM Roofline:      {crit_hbm:8.1f} FLOPs/byte")
            print(f"    Network Roofline:  {crit_net:8.1f} FLOPs/byte")
        print(f"\n  Hardware: NVLink BW = {network_roofline['nvlink_bw_gb_s']:.0f} GB/s, TP = {network_roofline['tp_degree']}")
        print()
        for phase_name, phase_data in network_roofline.get('phases', {}).items():
            M = phase_data['M']
            print(f"  {phase_name.capitalize()} Phase (M={M}):")
            print(f"    {'Operation':<25s} {'Parallelism':<15s} {'Network AI':>12s} {'T_compute':>10s} {'T_network':>10s} {'Bound':>12s}")
            print(f"    {'-'*25:<25s} {'-'*15:<15s} {'-'*12:>12s} {'-'*10:>10s} {'-'*10:>10s} {'-'*12:>12s}")
            ops = sorted(phase_data.get('operations', []), key=lambda x: -x['measured_time_us'])
            for op in ops[:6]:
                op_name = f"N={op['N']},K={op['K']},{op['dtype']}" if op.get('dtype') else f"N={op['N']},K={op['K']}"
                parallelism = op.get('parallelism', 'unknown')
                ai_str = 'N/A' if op['network_ai'] == float('inf') else f"{op['network_ai']:.0f}"
                t_comp = f"{op['t_compute_us']:.1f}us"
                t_net = f"{op['t_network_us']:.1f}us" if op['t_network_us'] > 0 else 'N/A'
                bound = op['bound']
                print(f"    {op_name:<25s} {parallelism:<15s} {ai_str:>12s} {t_comp:>10s} {t_net:>10s} {bound:>12s}")
            # Summary per phase
            total_gemm_time = phase_data['total_gemm_time_us']
            row_ops = [op for op in phase_data['operations'] if op.get('parallelism') == 'row-parallel']
            col_ops = [op for op in phase_data['operations'] if op.get('parallelism') == 'column-parallel']
            row_time = sum(op['measured_time_us'] for op in row_ops)
            col_time = sum(op['measured_time_us'] for op in col_ops)
            net_bound = [op for op in row_ops if 'network' in op['bound']]
            net_bound_time = sum(op['measured_time_us'] for op in net_bound)
            print()
            print(f"    Row-parallel: {len(row_ops)} ops, {row_time/1000:.2f}ms ({net_bound_time/1000:.2f}ms network-bound)")
            print(f"    Column-parallel: {len(col_ops)} ops, {col_time/1000:.2f}ms")
            print()
        ar_stats = network_roofline.get('allreduce_stats', {})
        if ar_stats.get('count', 0) > 0:
            print("  AllReduce Statistics (excluding warmup):")
            print(f"    Count: {ar_stats['count']}")
            print(f"    Total time: {ar_stats['total_time_us']/1000:.2f} ms")
            print(f"    Avg time: {ar_stats['avg_time_us']:.2f} us")
    # GEMM layer breakdown
    print("\n" + "-"*80)
    print("GEMM Layer Type Breakdown (kernels with known dimensions):")
    gemm_by_layer = defaultdict(lambda: {'time_us': 0, 'count': 0, 'mfu_sum': 0})
    for g in gemm_infos:
        lt = g.layer_type
        gemm_by_layer[lt]['time_us'] += g.duration_us
        gemm_by_layer[lt]['count'] += 1
        gemm_by_layer[lt]['mfu_sum'] += g.mfu * g.duration_us
    gemm_total_time = sum(d['time_us'] for d in gemm_by_layer.values())
    for layer_name in ['QKVO', 'FFN', 'Other']:
        if layer_name in gemm_by_layer:
            data = gemm_by_layer[layer_name]
            time_ms = data['time_us'] / 1000 / num_gpus
            pct = (data['time_us'] / gemm_total_time * 100) if gemm_total_time > 0 else 0
            avg_mfu_lt = data['mfu_sum'] / data['time_us'] if data['time_us'] > 0 else 0
            print(f"  {layer_name:<10s}: {time_ms:10.2f} ms/GPU  ({pct:5.1f}%)  [{data['count']:5d} kernels]  Avg MFU: {avg_mfu_lt:.1f}%")
    # Unmatched kernels summary
    if events:
        analysed_signatures = set(g.kernel_name[:50] for g in gemm_infos)
        unmatched_time_us = 0
        unmatched_count = 0
        unmatched_types = defaultdict(lambda: {'count': 0, 'time_us': 0})
        for e in events:
            if e.get('cat') != 'kernel':
                continue
            name = e.get('name', '')
            name_lower = name.lower()
            if not any(x in name_lower for x in ['gemm', 'matmul', 'nvjet']):
                continue
            ext_id = e.get('args', {}).get('External id')
            if ext_id is None and name[:50] not in analysed_signatures:
                grid = e.get('args', {}).get('grid', [])
                if infer_cuda_graph_kernel_dims(name, grid, config=None) is None:  # type: ignore
                    unmatched_time_us += e.get('dur', 0)
                    unmatched_count += 1
                    if 'nvjet' in name_lower:
                        unmatched_types['nvjet']['count'] += 1
                        unmatched_types['nvjet']['time_us'] += e.get('dur', 0)
                    elif 'router_gemm' in name_lower:
                        unmatched_types['router_gemm']['count'] += 1
                        unmatched_types['router_gemm']['time_us'] += e.get('dur', 0)
                    else:
                        unmatched_types['other']['count'] += 1
                        unmatched_types['other']['time_us'] += e.get('dur', 0)
        if unmatched_count > 0:
            unmatched_time_ms = unmatched_time_us / 1000 / num_gpus
            print(f"\n  Note: {unmatched_count} GEMM kernels ({unmatched_time_ms:.2f} ms/GPU) could not be analysed:")
            for ktype, data in sorted(unmatched_types.items(), key=lambda x: -x[1]['time_us']):
                print(f"        {ktype}: {data['count']} kernels, {data['time_us']/1000/num_gpus:.2f} ms/GPU")
    print("="*80)