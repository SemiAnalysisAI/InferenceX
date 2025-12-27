#!/usr/bin/env python3
"""
MFU (Model FLOPS Utilization) Trace Analyzer for PyTorch Profiler Traces

This script analyzes PyTorch profiler traces and adds MFU metrics to matmul operations.
Supports multiple GPU types (B200, H200, H100, A100) and configurable model parameters.

Features:
- MFU (Model FLOPS Utilization) calculation
- MBU (Memory Bandwidth Utilization) calculation  
- Arithmetic Intensity analysis
- Roofline model theoretical TFLOPS
- Layer-wise time breakdown (QKVO projection, SDPA, FFN)
- Support for FP4, FP8, BF16, FP16 data types
- Configurable tensor parallelism and batch sizes

Supported GPUs:
- B200: FP4 (9000 TFLOPS), FP8 (4500), BF16 (2250), 8 TB/s HBM, 128MB L2
- H200: FP8 (1979 TFLOPS), BF16 (989), 4.8 TB/s HBM, 80MB L2
- H100: FP8 (1979 TFLOPS), BF16 (989), 3.35 TB/s HBM, 50MB L2
- A100: BF16 (312 TFLOPS), 2 TB/s HBM, 40MB L2

Usage:
    # Basic usage with H200
    python mfu_trace_analyzer.py input_trace.json --summary-only
    
    # With B200 and custom config
    python mfu_trace_analyzer.py input_trace.json --gpu B200 --tp 8 --decode-batch-size 64
    
    # Full analysis with output
    python mfu_trace_analyzer.py input_trace.json output_trace.json --gpu H200
"""

import json
import re
import argparse
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class GPUSpecs:
    """GPU specifications for MFU/MBU calculation"""
    name: str
    fp16_tflops: float      # BF16/FP16 peak TFLOPS
    fp8_tflops: float       # FP8 peak TFLOPS (if supported)
    fp4_tflops: float       # FP4 peak TFLOPS (if supported, 0 if not)
    memory_bw_tb_s: float   # Memory bandwidth in TB/s
    num_sms: int
    l2_cache_mb: float = 50.0  # L2 cache size in MB
    nvlink_bw_gb_s: float = 900.0  # NVLink bandwidth per GPU (GB/s, bidirectional)
    
# GPU specifications database
GPU_SPECS = {
    "B200": GPUSpecs(
        name="NVIDIA B200 SXM",
        fp16_tflops=2250.0,     # BF16 Tensor Core peak (dense)
        fp8_tflops=4500.0,      # FP8 Tensor Core peak (2x BF16)
        fp4_tflops=9000.0,      # FP4 Tensor Core peak (4x BF16)
        memory_bw_tb_s=8.0,     # HBM3e bandwidth (8 TB/s)
        num_sms=160,            # Blackwell SM count
        l2_cache_mb=128.0,      # 128 MB L2 cache
        nvlink_bw_gb_s=1800.0   # NVLink 5.0: 1.8 TB/s bidirectional
    ),
    "H200": GPUSpecs(
        name="NVIDIA H200 SXM",
        fp16_tflops=989.4,      # BF16 Tensor Core peak
        fp8_tflops=1978.9,      # FP8 Tensor Core peak (2x BF16)
        fp4_tflops=0.0,         # H200 doesn't have FP4
        memory_bw_tb_s=4.8,     # HBM3e bandwidth
        num_sms=132,
        l2_cache_mb=80.0,       # 80 MB L2 cache
        nvlink_bw_gb_s=900.0    # NVLink 4.0: 900 GB/s bidirectional
    ),
    "H100": GPUSpecs(
        name="NVIDIA H100 SXM", 
        fp16_tflops=989.4,
        fp8_tflops=1978.9,
        fp4_tflops=0.0,         # H100 doesn't have FP4
        memory_bw_tb_s=3.35,    # HBM3 bandwidth
        num_sms=132,
        l2_cache_mb=50.0,       # 50 MB L2 cache
        nvlink_bw_gb_s=900.0    # NVLink 4.0
    )
}


# =============================================================================
# KERNEL CLASSIFICATION
# =============================================================================
# This section defines how different kernel types are identified and grouped.
# Modify these patterns to support new kernel types or frameworks.

@dataclass
class KernelClassification:
    """Classification result for a GPU kernel"""
    category: str           # 'gemm', 'communication', 'attention', 'normalization', 'other'
    subcategory: str        # More specific type within category
    is_gemm: bool          # Whether this kernel performs matrix multiplication
    dtype: str             # Data type (fp8, bf16, fp16, etc.)
    source: str            # Source framework (deep_gemm, cublas, nvjet, nccl, etc.)

# -----------------------------------------------------------------------------
# GEMM Kernel Patterns
# These patterns identify matrix multiplication kernels from different sources
# -----------------------------------------------------------------------------
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
        'dtype': 'bf16',  # nvjet is typically BF16 cuBLASLt
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
        'match': lambda name: ('gemm' in name.lower() or 'matmul' in name.lower()) and \
                              'deep_gemm' not in name.lower() and 'nvjet' not in name.lower(),
        'is_gemm': True,
        'dtype': 'bf16',
        'source': 'generic',
        'subcategory': 'other_gemm',
    },
}

# -----------------------------------------------------------------------------
# Communication Kernel Patterns
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Attention Kernel Patterns
# -----------------------------------------------------------------------------
ATTENTION_KERNEL_PATTERNS = {
    'flash_attention': {
        'match': lambda name: 'flashinfer' in name.lower() or 'flash_attn' in name.lower() or \
                              'fmha' in name.lower(),
        'subcategory': 'flash_attention',
    },
    'mla_attention': {
        'match': lambda name: 'batchmlapageattention' in name.lower() or \
                              'prefillwithkvcache' in name.lower(),
        'subcategory': 'mla_attention',
    },
}

# -----------------------------------------------------------------------------
# Normalization Kernel Patterns  
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# CUDA Graph Kernel Dimension Inference
# -----------------------------------------------------------------------------
# CUDA Graph replayed kernels don't have External ID linkage to CPU ops.
# We infer dimensions from kernel name + grid pattern + model architecture.
#
# Format: (kernel_pattern, grid_pattern) -> (M, N, K, dtype, layer_type)
# grid_pattern: tuple of grid dimensions, or None to match any grid

# Default model configuration for dimension inference
# These can be overridden via command line arguments
DEFAULT_MODEL_CONFIG = {
    'hidden_size': 7168,
    'num_experts': 256,
    'expert_intermediate_size': 2048,  # Per expert (total, before TP division)
    'decode_batch_size': 64,  # Common decode batch size
    'tp_degree': 8,  # Tensor parallelism degree - divides expert_intermediate
}

# Global model config that will be updated from command line args
MODEL_CONFIG = DEFAULT_MODEL_CONFIG.copy()

def infer_cuda_graph_kernel_dims(kernel_name: str, grid: List[int], 
                                  model_config: Dict = None,
                                  sibling_dims: Dict = None) -> Optional[Tuple[int, int, int, str, str]]:
    """
    Infer dimensions for CUDA Graph replayed kernels based on kernel name and grid.
    
    CUDA Graph replayed kernels (cudaGraphLaunch) don't have External ID linkage 
    because they bypass PyTorch's operator dispatch during replay.
    
    Inference strategies (in order of priority):
    1. Match to "sibling" kernels - same kernel type that HAS External ID (prefill versions)
    2. Use model architecture knowledge (DeepSeek-R1 dimensions)
    
    Args:
        sibling_dims: Dict mapping kernel_signature -> (N, K) from kernels with External ID
                      e.g., {"nvjet_tst_64x64": (256, 7168)} 
    
    Returns (M, N, K, dtype, layer_type) or None if cannot infer.
    """
    if model_config is None:
        model_config = MODEL_CONFIG
    
    hidden = model_config.get('hidden_size', 7168)
    tp_degree = model_config.get('tp_degree', 8)
    # With TP=8, expert_intermediate is divided: 2048/8 = 256
    expert_intermediate_per_gpu = model_config.get('expert_intermediate_size', 2048) // tp_degree
    decode_batch = model_config.get('decode_batch_size', 64)
    
    name_lower = kernel_name.lower()
    grid_tuple = tuple(grid) if grid else ()
    
    # Strategy 1: Use sibling dims for nvjet_tst_128x8 (shared expert)
    # The sibling with External ID has M=1 (single token routing in prefill)
    # but CUDA Graph decode kernels should use decode_batch
    if sibling_dims and 'nvjet_tst_128x8' in name_lower and 'nvjet_tst_128x8' in sibling_dims:
        n, k = sibling_dims['nvjet_tst_128x8']
        # For CUDA Graph kernels (no External ID), use decode batch size
        return (decode_batch, n, k, 'bf16', 'FFN')
    
    # Strategy 2: Use sibling dims to get architecture parameters, but use grid for direction
    # nvjet_tst_64x64 with External ID gives us N=256 (intermediate), K=7168 (hidden)
    # nvjet_tst_64x8 has same architecture but different grid patterns for UP vs DOWN
    if sibling_dims and 'nvjet_tst_64x8' in name_lower and 'nvjet_tst_64x64' in sibling_dims:
        # nvjet_tst_64x64 gives [hidden, intermediate_per_gpu] -> N=256, K=7168
        # This is the UP projection: hidden -> intermediate
        intermediate_per_gpu, hidden_from_sibling = sibling_dims['nvjet_tst_64x64']
        
        # Use grid to determine if this is UP or DOWN projection
        if grid_tuple == (2, 64, 1):
            # More tiles → DOWN projection: intermediate -> hidden
            # [M, intermediate] @ [intermediate, hidden] -> [M, hidden]
            return (decode_batch, hidden_from_sibling, intermediate_per_gpu, 'bf16', 'FFN')
        elif grid_tuple == (2, 16, 1):
            # Fewer tiles → UP projection: hidden -> intermediate
            # [M, hidden] @ [hidden, intermediate] -> [M, intermediate]
            return (decode_batch, intermediate_per_gpu, hidden_from_sibling, 'bf16', 'FFN')
    
    # Strategy 2: Use model architecture knowledge
    # nvjet_tst_64x8 - MoE expert FFN during decode
    # Reference: nvjet_tst_64x64 with M=992, N=256, K=7168 (N=256 = 2048/8)
    if 'nvjet_tst_64x8' in name_lower:
        # MoE FFN with TP=8: N = expert_intermediate / TP = 256
        if grid_tuple == (2, 64, 1):
            # More tiles → likely DOWN projection (output to hidden)
            # [M, 256] @ [256, 7168] → [M, 7168]
            return (decode_batch, hidden, expert_intermediate_per_gpu, 'bf16', 'FFN')
        elif grid_tuple == (2, 16, 1):
            # Fewer tiles → likely UP projection (output to intermediate_per_gpu)
            # [M, 7168] @ [7168, 256] → [M, 256]
            return (decode_batch, expert_intermediate_per_gpu, hidden, 'bf16', 'FFN')
    
    # nvjet_tst_128x8 - Shared expert (handled by sibling matching above)
    # If sibling matching didn't work, fall back to decode_batch
    if 'nvjet_tst_128x8' in name_lower:
        # Shared expert: intermediate=16160, hidden=7168
        return (decode_batch, 16160, hidden, 'bf16', 'FFN')
    
    # nvjet_tst_64x64 - Has External ID linkage in traces
    if 'nvjet_tst_64x64' in name_lower:
        # Let External ID linkage handle this (known: M=992, N=256, K=7168)
        return None
    
    # router_gemm - MoE router kernel
    if 'router_gemm' in name_lower:
        num_experts = model_config.get('num_experts', 256)
        return (decode_batch, num_experts, hidden, 'bf16', 'FFN')
    
    return None


def build_sibling_dims_map(events: List[Dict], cpu_op_dims: Dict) -> Dict[str, Tuple[int, int]]:
    """
    Build a map of kernel signatures to their known (N, K) dimensions.
    
    This finds kernels that DO have External ID linkage (typically from prefill phase)
    and extracts their N, K dimensions. These can be used to infer dimensions for
    "sibling" kernels in CUDA Graphs that don't have External ID.
    
    Returns: Dict mapping kernel_signature -> (N, K)
             e.g., {"nvjet_tst_64x64": (256, 7168)}
    """
    sibling_dims = {}
    
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        
        name = event.get('name', '')
        ext_id = event.get('args', {}).get('External id')
        
        if ext_id is None:
            continue
        
        # Extract kernel signature (e.g., "nvjet_tst_64x64" from full name)
        if 'nvjet_tst_' in name.lower():
            import re
            match = re.search(r'nvjet_tst_(\d+x\d+)', name.lower())
            if match:
                sig = f"nvjet_tst_{match.group(1)}"
                
                # Find dimensions from CPU op
                tp_rank = extract_tp_rank(event.get('pid'))
                for key_ext_id in [ext_id, ext_id - 1, ext_id + 1]:
                    key = (tp_rank, key_ext_id)
                    if key in cpu_op_dims:
                        dims = cpu_op_dims[key]
                        if len(dims) >= 3:
                            m, n, k = dims[0], dims[1], dims[2]
                            if sig not in sibling_dims:
                                sibling_dims[sig] = (n, k)
                        break
    
    return sibling_dims

# -----------------------------------------------------------------------------
# CPU Op Patterns for dimension extraction
# Maps CPU op names to extraction method
# -----------------------------------------------------------------------------
CPU_OP_GEMM_PATTERNS = {
    # deep_gemm FP8 ops
    'deep_gemm_fp8': {
        'match': lambda name: 'deep_gemm' in name.lower() or 'fp8_gemm' in name.lower(),
        'dtype': 'fp8',
    },
    # aten::mm, aten::matmul (standard PyTorch ops)
    'aten_mm': {
        'match': lambda name: name in ['aten::mm', 'aten::matmul'],
        'dtype': 'bf16',
    },
    # aten::linear (includes bias)
    'aten_linear': {
        'match': lambda name: name == 'aten::linear',
        'dtype': 'bf16',
    },
}


def classify_kernel(kernel_name: str) -> KernelClassification:
    """
    Classify a GPU kernel based on its name.
    Returns classification with category, type, and metadata.
    """
    # Check GEMM patterns first (most important for MFU)
    for pattern_name, pattern in GEMM_KERNEL_PATTERNS.items():
        if pattern['match'](kernel_name):
            return KernelClassification(
                category='gemm',
                subcategory=pattern['subcategory'],
                is_gemm=True,
                dtype=pattern['dtype'],
                source=pattern['source']
            )
    
    # Check communication patterns
    for pattern_name, pattern in COMM_KERNEL_PATTERNS.items():
        if pattern['match'](kernel_name):
            return KernelClassification(
                category='communication',
                subcategory=pattern['subcategory'],
                is_gemm=False,
                dtype='',
                source='nccl'
            )
    
    # Check attention patterns
    for pattern_name, pattern in ATTENTION_KERNEL_PATTERNS.items():
        if pattern['match'](kernel_name):
            return KernelClassification(
                category='attention',
                subcategory=pattern['subcategory'],
                is_gemm=False,
                dtype='',
                source='flashinfer'
            )
    
    # Check normalization patterns
    for pattern_name, pattern in NORM_KERNEL_PATTERNS.items():
        if pattern['match'](kernel_name):
            return KernelClassification(
                category='normalization',
                subcategory=pattern['subcategory'],
                is_gemm=False,
                dtype='',
                source='custom'
            )
    
    # Default: other
    return KernelClassification(
        category='other',
        subcategory='unknown',
        is_gemm=False,
        dtype='',
        source='unknown'
    )


def extract_dims_from_aten_mm(event: Dict) -> Optional[Tuple[int, int, int, str]]:
    """
    Extract M, N, K dimensions from aten::mm or aten::matmul CPU op.
    
    aten::mm: Input Dims = [[M, K], [K, N]]
    aten::matmul: Same as mm for 2D inputs
    
    Returns (M, N, K, dtype) or None
    """
    args = event.get('args', {})
    input_dims = args.get('Input Dims', [])
    input_types = args.get('Input type', [])
    
    if len(input_dims) < 2:
        return None
    
    # For mm: [M, K] @ [K, N] -> [M, N]
    dim_a = input_dims[0]
    dim_b = input_dims[1]
    
    if not dim_a or not dim_b or len(dim_a) < 2 or len(dim_b) < 2:
        return None
    
    m = dim_a[0]
    k = dim_a[1]
    n = dim_b[1]  # K should equal dim_b[0]
    
    # Determine dtype
    dtype = 'bf16'
    if input_types:
        type_str = str(input_types[0]).lower()
        if 'float8' in type_str or 'fp8' in type_str:
            dtype = 'fp8'
        elif 'float16' in type_str or 'half' in type_str:
            dtype = 'fp16'
        elif 'bfloat16' in type_str:
            dtype = 'bf16'
    
    return (m, n, k, dtype)


def extract_dims_from_aten_linear(event: Dict) -> Optional[Tuple[int, int, int, str]]:
    """
    Extract M, N, K dimensions from aten::linear CPU op.
    
    aten::linear: Input Dims = [[M, K], [N, K], [N] or []]
    (weight is stored as [N, K], bias is [N])
    
    Returns (M, N, K, dtype) or None
    """
    args = event.get('args', {})
    input_dims = args.get('Input Dims', [])
    input_types = args.get('Input type', [])
    
    if len(input_dims) < 2:
        return None
    
    dim_input = input_dims[0]  # [M, K] or [batch, M, K]
    dim_weight = input_dims[1]  # [N, K]
    
    if not dim_input or not dim_weight:
        return None
    
    # Handle batched input
    if len(dim_input) == 2:
        m, k = dim_input
    elif len(dim_input) == 3:
        m = dim_input[0] * dim_input[1]  # batch * seq_len
        k = dim_input[2]
    else:
        return None
    
    if len(dim_weight) < 2:
        return None
    n = dim_weight[0]  # Weight is [N, K]
    
    # Determine dtype
    dtype = 'bf16'
    if input_types:
        type_str = str(input_types[0]).lower()
        if 'float8' in type_str or 'fp8' in type_str:
            dtype = 'fp8'
        elif 'float16' in type_str or 'half' in type_str:
            dtype = 'fp16'
        elif 'bfloat16' in type_str:
            dtype = 'bf16'
    
    return (m, n, k, dtype)


# =============================================================================
# END KERNEL CLASSIFICATION
# =============================================================================


@dataclass
class GemmInfo:
    """Information about a GEMM operation"""
    m: int
    n: int
    k: int
    dtype: str  # Primary/compute dtype for display (e.g., 'fp8' or 'bf16')
    input_dtype: str = ""  # Back-compat: summary of inputs (or 'mixed')
    output_dtype: str = ""  # Back-compat: equals c_dtype
    a_dtype: str = ""       # Operand A dtype
    b_dtype: str = ""       # Operand B dtype
    c_dtype: str = ""       # Output (C) dtype
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
    layer_type: str = ""  # QKVO, SDPA, FFN, Other
    # L2 cache-aware metrics
    activation_bytes: int = 0      # A + C bytes (through HBM when weight cached)
    weight_bytes: int = 0          # B bytes (may be served from L2 cache)
    effective_mbu: float = 0.0     # MBU assuming weight is in L2 cache
    l2_cache_benefit: float = 0.0  # Ratio of theoretical to effective bytes
    # Trace identifiers for locating kernel in profiler
    timestamp_us: float = 0.0      # Kernel start timestamp (ts field)
    correlation_id: int = 0        # CUDA correlation ID for trace lookup
    tp_rank: str = ""              # TP rank (e.g., "00" from "[TP00]")
    stream_id: int = 0             # CUDA stream ID


@dataclass
class GroupedGemmInfo:
    """Information about a Grouped GEMM operation (e.g., fused MoE)"""
    # Problem dimensions
    num_tokens: int           # M: number of input tokens
    top_k: int               # Number of experts per token
    num_experts: int         # E: total number of experts
    hidden_size: int         # K: hidden dimension
    w1_intermediate: int     # N1: gate+up intermediate dimension (per GPU)
    w2_intermediate: int     # N2: down projection intermediate (per GPU)
    
    # Data types
    input_dtype: str = "bf16"   # Input activation dtype
    weight_dtype: str = "fp8"   # Weight dtype (w1 and w2)
    output_dtype: str = "bf16"  # Output activation dtype
    
    # Computed metrics
    total_token_expert_pairs: int = 0  # M * top_k
    w1_flops: int = 0         # FLOPs for gate+up projection
    w2_flops: int = 0         # FLOPs for down projection  
    total_flops: int = 0
    
    # Memory access (bytes)
    input_bytes: int = 0      # Input activations
    w1_weight_bytes: int = 0  # w1 weights
    w2_weight_bytes: int = 0  # w2 weights
    output_bytes: int = 0     # Output activations
    total_bytes: int = 0
    
    # Performance metrics
    duration_us: float = 0.0
    tflops: float = 0.0
    mfu: float = 0.0
    achieved_bw_tb_s: float = 0.0
    mbu: float = 0.0
    arithmetic_intensity: float = 0.0
    roofline_bound: str = ""
    
    # Kernel info
    kernel_name: str = ""
    external_id: int = 0
    num_kernels: int = 0      # Number of GPU kernels for this op
    
    # Trace identifiers
    timestamp_us: float = 0.0
    correlation_id: int = 0
    tp_rank: str = ""


def get_bytes_per_element(dtype: str) -> float:
    """Get bytes per element for a dtype.
    
    Returns float to handle sub-byte types like FP4 (0.5 bytes).
    """
    dtype_lower = dtype.lower()
    if 'float4' in dtype_lower or 'fp4' in dtype_lower or 'e2m1' in dtype_lower:
        return 0.5  # FP4 is 4 bits = 0.5 bytes
    elif 'float8' in dtype_lower or 'fp8' in dtype_lower or 'e4m3' in dtype_lower or 'e5m2' in dtype_lower:
        return 1  # FP8 is 1 byte
    elif 'float16' in dtype_lower or 'fp16' in dtype_lower or 'bfloat16' in dtype_lower or 'bf16' in dtype_lower:
        return 2  # FP16/BF16 is 2 bytes
    elif 'float32' in dtype_lower or 'fp32' in dtype_lower:
        return 4
    else:
        return 2  # Default to 2 bytes


def normalize_dtype(dt: Optional[str]) -> str:
    """Normalize various dtype strings to canonical short names."""
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


def compute_dtype_from_inputs(a_dtype: str, b_dtype: str) -> str:
    """Heuristic: compute dtype based on A/B inputs.

    - If any input is fp8 => fp8 Tensor Core path
    - Else if any is fp16 => fp16
    - Else if any is bf16 => bf16
    - Else fallback to a_dtype or b_dtype
    """
    dts = {normalize_dtype(a_dtype), normalize_dtype(b_dtype)}
    if "fp8" in dts:
        return "fp8"
    if "fp16" in dts:
        return "fp16"
    if "bf16" in dts:
        return "bf16"
    # fallback
    return next(iter(dts - {""}), "bf16")


def calculate_gemm_bytes(m: int, n: int, k: int,
                         a_dtype: str = 'bf16', b_dtype: str = 'bf16', c_dtype: str = 'bf16') -> int:
    """
    Calculate bytes accessed for GEMM: C = A @ B
    A: [M, K], B: [K, N], C: [M, N]

    Uses separate dtypes for A, B, and C to support mixed GEMMs.
    """
    a_bytes = get_bytes_per_element(a_dtype)
    b_bytes = get_bytes_per_element(b_dtype)
    c_bytes = get_bytes_per_element(c_dtype)

    # Read A, B; Write C
    bytes_a = int(m * k * a_bytes) if a_bytes >= 1 else ( (m * k + 1) // 2 )
    bytes_b = int(k * n * b_bytes) if b_bytes >= 1 else ( (k * n + 1) // 2 )
    bytes_c = int(m * n * c_bytes) if c_bytes >= 1 else ( (m * n + 1) // 2 )

    return bytes_a + bytes_b + bytes_c


def calculate_gemm_bytes_breakdown(m: int, n: int, k: int,
                                   a_dtype: str = 'bf16', b_dtype: str = 'bf16', c_dtype: str = 'bf16') -> Tuple[int, int, int]:
    """
    Calculate bytes breakdown for GEMM: C = A @ B
    A: [M, K], B: [K, N], C: [M, N]
    
    Returns (activation_bytes, weight_bytes, total_bytes)
    where activation_bytes = A + C (through HBM when weight cached)
          weight_bytes = B (may be served from L2 cache)
    
    For inference workloads, weight matrix B is often reused and may be cached in L2.
    """
    a_bytes = get_bytes_per_element(a_dtype)
    b_bytes = get_bytes_per_element(b_dtype)
    c_bytes = get_bytes_per_element(c_dtype)

    # A = input activations, B = weight, C = output activations
    bytes_a = int(m * k * a_bytes) if a_bytes >= 1 else ((m * k + 1) // 2)
    bytes_b = int(k * n * b_bytes) if b_bytes >= 1 else ((k * n + 1) // 2)
    bytes_c = int(m * n * c_bytes) if c_bytes >= 1 else ((m * n + 1) // 2)
    
    activation_bytes = bytes_a + bytes_c
    weight_bytes = bytes_b
    total_bytes = bytes_a + bytes_b + bytes_c
    
    return (activation_bytes, weight_bytes, total_bytes)


def calculate_gemm_flops(m: int, n: int, k: int) -> int:
    """Calculate FLOPs for a GEMM operation: C = A @ B where A is MxK and B is KxN"""
    return 2 * m * n * k


def calculate_arithmetic_intensity(flops: int, bytes_accessed: int) -> float:
    """Calculate arithmetic intensity (FLOPs per byte)"""
    if bytes_accessed <= 0:
        return 0.0
    return flops / bytes_accessed


def calculate_roofline_tflops(arithmetic_intensity: float, gpu_specs: GPUSpecs, peak_tflops: float) -> Tuple[float, str]:
    """
    Calculate theoretical TFLOPS using simple roofline model.
    Returns (theoretical_tflops, bound_type)
    
    Roofline: min(peak_compute, memory_bw * arithmetic_intensity)
    """
    if arithmetic_intensity <= 0:
        return 0.0, "unknown"
    
    # Memory-bound TFLOPS = bandwidth * arithmetic_intensity
    # bandwidth is in TB/s, we want TFLOPS, so: TB/s * FLOP/byte = TFLOP/s
    memory_bound_tflops = gpu_specs.memory_bw_tb_s * 1e12 * arithmetic_intensity / 1e12
    
    # Compute-bound TFLOPS = peak TFLOPS
    compute_bound_tflops = peak_tflops
    
    if memory_bound_tflops < compute_bound_tflops:
        return memory_bound_tflops, "memory"
    else:
        return compute_bound_tflops, "compute"


def calculate_mfu(flops: int, duration_us: float, peak_tflops: float) -> float:
    """Calculate MFU (Model FLOPS Utilization)"""
    if duration_us <= 0:
        return 0.0
    
    duration_s = duration_us / 1e6
    achieved_tflops = (flops / 1e12) / duration_s
    mfu = (achieved_tflops / peak_tflops) * 100.0
    return mfu


def calculate_mbu(bytes_accessed: int, duration_us: float, peak_bw_tb_s: float) -> float:
    """Calculate MBU (Memory Bandwidth Utilization)"""
    if duration_us <= 0:
        return 0.0
    
    duration_s = duration_us / 1e6
    achieved_bw_tb_s = (bytes_accessed / 1e12) / duration_s
    mbu = (achieved_bw_tb_s / peak_bw_tb_s) * 100.0
    return mbu


def get_dtype_peak_tflops(dtype: str, gpu_specs: GPUSpecs) -> float:
    """Get peak TFLOPS based on data type.
    
    Falls back to lower precision peak if higher precision not supported.
    """
    dtype_lower = dtype.lower()
    if 'float4' in dtype_lower or 'fp4' in dtype_lower or 'e2m1' in dtype_lower:
        # FP4 support - fall back to FP8 if not available
        if gpu_specs.fp4_tflops > 0:
            return gpu_specs.fp4_tflops
        elif gpu_specs.fp8_tflops > 0:
            return gpu_specs.fp8_tflops
        else:
            return gpu_specs.fp16_tflops
    elif 'float8' in dtype_lower or 'fp8' in dtype_lower or 'e4m3' in dtype_lower or 'e5m2' in dtype_lower:
        return gpu_specs.fp8_tflops if gpu_specs.fp8_tflops > 0 else gpu_specs.fp16_tflops
    else:
        return gpu_specs.fp16_tflops


def classify_layer_type(m: int, n: int, k: int, kernel_name: str = "") -> str:
    """
    Classify GEMM operation into layer type based on dimensions.
    
    DeepSeek-R1 / DeepSeek-V3 architecture:
    - Hidden Size: 7168
    - Num Layers: 61
    - Attn Heads: 128, KV Heads: 128, Head Dim: 56
    - Intermediate Size: 18432
    - MLA (Multi-head Latent Attention) with compressed KV cache
    - MoE: 256 experts, top-8 routing
    
    With TP8 (tensor parallel = 8), dimensions may be divided.
    
    GEMM patterns observed:
    - N=2112, K=7168: Q/K/V projection (MLA compressed dim = 2112)
    - N=7168, K=2048 or K=2304: Output projection back to hidden
    - N=3072, K=1536: MLA latent compression/decompression
    - N=256, K=7168: MoE router (hidden -> num_experts logits)
    - N=4096, K=512: MoE gate/router variant
    - N=4608, K=7168: MoE FFN up projection
    - N=7168, K=4608: MoE FFN down projection
    """
    hidden_size = 7168
    num_experts = 256
    
    # =========================================================================
    # FFN / MoE patterns (check first - more specific)
    # =========================================================================
    
    # MoE router: [M, hidden] @ [hidden, num_experts] -> routing logits
    # This is FFN, not attention!
    if k == hidden_size and n == num_experts:
        return "FFN"  # MoE router
    
    # MoE gate/router variants
    if k == 512 and n in [4096, 2048, 1024]:
        return "FFN"  # MoE routing
    
    # MoE FFN projections
    # N=4608, K=7168: up projection (part of 18432 intermediate split across experts/TP)
    # N=7168, K=4608: down projection
    if (n == 4608 and k == hidden_size) or (n == hidden_size and k == 4608):
        return "FFN"
    
    # Large intermediate dimensions are typically FFN
    if n > 10000 or k > 10000:
        return "FFN"
    
    # =========================================================================
    # QKVO / Attention patterns
    # =========================================================================
    
    # MLA attention projections
    # N=2112: compressed attention head output (Q/K/V for MLA)
    # 2112 = q_lora_rank(1536) + kv_lora_rank(512) + rope(64)
    if k == hidden_size and n in [2112, 2048, 2304, 2560]:
        return "QKVO"
    
    # Output projection: N=7168 (back to hidden)
    if n == hidden_size and k in [2048, 2112, 2304, 2560]:
        return "QKVO"
    
    # MLA latent compression/decompression: N=3072, K=1536 or vice versa
    if (n == 3072 and k == 1536) or (n == 1536 and k == 3072):
        return "QKVO"
    
    # =========================================================================
    # Default classification
    # =========================================================================
    
    # Default based on hidden dimension involvement
    if k == hidden_size:
        return "QKVO"  # Input from hidden states (likely attention)
    if n == hidden_size:
        return "QKVO"  # Output to hidden states (likely attention)
    
    return "Other"


def parse_deep_gemm_kernel_dims(kernel_name: str, grid: List[int], 
                                 cpu_op_dims: Optional[Tuple[int, int, int]] = None,
                                 reference_m: int = 992) -> Optional[Tuple[int, int, int, str]]:
    """
    Parse deep_gemm kernel name to extract M, N, K dimensions.
    
    Template: void deep_gemm::sm90_fp8_gemm_1d2d_impl<0u, N, K, 1u, M_tile, N_tile, K_tile, ...>
    
    For kernels with CPU op correlation, use CPU op dims for M.
    For kernels without External ID (decode kernels), infer M from grid pattern:
      - grid[0] = num_n_tiles (typically)
      - If m_tile=64 and pattern suggests 1 M tile, M=64 (decode batch)
    
    Returns (M, N, K, dtype) tuple or None if cannot be determined
    """
    match = re.search(
        r'deep_gemm::sm90_fp8_gemm_1d2d_impl<(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u,\s*(\d+)u',
        kernel_name
    )
    if not match:
        return None
    
    n = int(match.group(2))
    k = int(match.group(3))
    m_tile = int(match.group(5))
    n_tile = int(match.group(6))
    
    # Determine M
    if cpu_op_dims:
        m = cpu_op_dims[0]  # Use M from CPU op (prefill)
    else:
        # For kernels without External ID (decode kernels):
        # Infer M from grid pattern
        # grid[0] typically = ceil(N/n_tile) for decode or ceil(M/m_tile) * ceil(N/n_tile) for prefill
        grid_x = grid[0] if grid else 1
        num_n_tiles = (n + n_tile - 1) // n_tile
        
        if grid_x > 0 and num_n_tiles > 0:
            # If grid_x == num_n_tiles, then num_m_tiles = 1, so M = m_tile
            # If grid_x > num_n_tiles, then num_m_tiles = grid_x // num_n_tiles
            if grid_x <= num_n_tiles:
                # Single M tile - decode with small batch
                m = m_tile  # Typically 64 for decode
            else:
                num_m_tiles = grid_x // num_n_tiles
                m = num_m_tiles * m_tile
        else:
            # Fallback: if m_tile=64, likely decode with M=64
            m = m_tile if m_tile <= 128 else 64
    
    dtype = 'fp8' if 'fp8' in kernel_name.lower() else 'bf16'
    return (m, n, k, dtype)


def extract_dimensions_from_cpu_op(event: Dict) -> Optional[Tuple[int, int, int, str, str, str]]:
    """
    Extract M, N, K dimensions and dtypes from CPU op.
    
    For sglang::deep_gemm_fp8_fp8_bf16_nt:
      Input types: ['c10::Float8_e4m3fn', 'float', 'c10::Float8_e4m3fn', 'float', 'c10::BFloat16']
      Input Dims: [[M, K], [M, ?], [N, K], [?, ?], [M, N]]
      A = [M, K] (fp8), B = [N, K].T (fp8), C = [M, N] (bf16)
    
    Returns (M, N, K, input_dtype, output_dtype) or None
    """
    args = event.get('args', {})
    input_dims = args.get('Input Dims', [])
    input_types = args.get('Input type', [])
    name = event.get('name', '')
    
    if not input_dims:
        return None
    
    # Parse dtypes for A, B, C
    a_dtype = ''
    b_dtype = ''
    c_dtype = ''
    
    if input_types:
        # For deep_gemm: types are [A_type, A_scale, B_type, B_scale, C_type]
        # For aten::mm/linear: types are [A_type, B_type, (bias?)]
        types = [normalize_dtype(t) for t in input_types]
        if 'deep_gemm' in name and len(types) >= 5:
            a_dtype = types[0]
            b_dtype = types[2]
            c_dtype = types[4]
        else:
            if len(types) >= 1:
                a_dtype = types[0]
            if len(types) >= 2:
                b_dtype = types[1]
            # Many aten ops don’t list output dtype; infer when inputs match
            if a_dtype and b_dtype and a_dtype == b_dtype:
                c_dtype = a_dtype
    
    # sglang::deep_gemm_fp8_fp8_bf16_nt format:
    if 'deep_gemm' in name and len(input_dims) >= 5:
        a_dims = input_dims[0]  # [M, K]
        b_dims = input_dims[2]  # [N, K]
        
        if isinstance(a_dims, list) and len(a_dims) >= 2:
            m = a_dims[0]
            k = a_dims[1]
        else:
            return None
            
        if isinstance(b_dims, list) and len(b_dims) >= 1:
            n = b_dims[0]
        else:
            return None
            
        return (m, n, k, a_dtype or 'bf16', b_dtype or (a_dtype or 'bf16'), c_dtype or (a_dtype or b_dtype or 'bf16'))
    
    # aten::mm format: Input Dims: [[M, K], [K, N]]
    if 'aten::mm' in name and len(input_dims) >= 2:
        a_dims = input_dims[0]
        b_dims = input_dims[1]
        
        if isinstance(a_dims, list) and len(a_dims) >= 2:
            m = a_dims[0]
            k = a_dims[1]
        else:
            return None
            
        if isinstance(b_dims, list) and len(b_dims) >= 2:
            n = b_dims[1]
        else:
            return None
            
        return (m, n, k, a_dtype or 'bf16', b_dtype or (a_dtype or 'bf16'), c_dtype or (a_dtype or b_dtype or 'bf16'))
    
    # aten::linear format
    if 'aten::linear' in name and len(input_dims) >= 2:
        a_dims = input_dims[0]
        b_dims = input_dims[1]
        
        if isinstance(a_dims, list) and len(a_dims) >= 2:
            m = a_dims[0]
            k = a_dims[1]
        else:
            return None
            
        if isinstance(b_dims, list) and len(b_dims) >= 2:
            n = b_dims[0]
        else:
            return None
            
        return (m, n, k, a_dtype or 'bf16', b_dtype or (a_dtype or 'bf16'), c_dtype or (a_dtype or b_dtype or 'bf16'))
    
    return None


def extract_tp_rank(pid) -> Optional[str]:
    """Extract TP rank from PID like '[TP06] 6' or '[TP06] 729201'"""
    if pid is None:
        return None
    pid_str = str(pid)
    match = re.search(r'\[TP(\d+)\]', pid_str)
    if match:
        return match.group(1)
    return pid_str


def analyze_all_gemm_kernels(events: List[Dict], gpu_specs: GPUSpecs) -> List[GemmInfo]:
    """
    Analyze ALL GEMM kernels and calculate MFU/MBU metrics.
    
    Dimension extraction sources (in order of priority):
    1. CPU op correlation via External ID (deep_gemm, aten::mm, aten::linear)
    2. Kernel name parsing (deep_gemm kernels encode N, K in template params)
    
    Supported kernel types (see GEMM_KERNEL_PATTERNS):
    - deep_gemm: FP8 GEMM from DeepGEMM library
    - nvjet: BF16 GEMM from cuBLASLt 
    - cublas/cutlass: Other BLAS implementations
    - generic: Any kernel with 'gemm' or 'matmul' in name
    """
    gemm_infos = []
    
    # =========================================================================
    # STEP 1: Build CPU op dimensions map for External ID correlation
    # =========================================================================
    cpu_op_dims = {}  # (tp_rank, ext_id) -> (m, n, k, a_dtype, b_dtype, c_dtype)
    
    for event in events:
        if event.get('cat') != 'cpu_op':
            continue
        
        name = event.get('name', '')
        ext_id = event.get('args', {}).get('External id')
        if ext_id is None:
            continue
            
        tp_rank = extract_tp_rank(event.get('pid'))
        dims = None
        
        # Check each CPU op pattern and extract dimensions with dtypes
        for pattern_name, pattern in CPU_OP_GEMM_PATTERNS.items():
            if pattern['match'](name):
                # Use unified extraction that returns (m, n, k, a_dtype, b_dtype, c_dtype)
                dims = extract_dimensions_from_cpu_op(event)
                break
        
        if dims:
            cpu_op_dims[(tp_rank, ext_id)] = dims
            # Child kernels often have ext_id + 1
            cpu_op_dims[(tp_rank, ext_id + 1)] = dims
    
    # =========================================================================
    # STEP 1.5: Build sibling dimensions map for CUDA Graph kernels
    # =========================================================================
    # This finds kernels with External ID (prefill) that can inform dimensions
    # for similar kernels without External ID (decode in CUDA Graphs)
    sibling_dims = build_sibling_dims_map(events, cpu_op_dims)
    
    # =========================================================================
    # STEP 2: Process all GPU kernels
    # =========================================================================
    seen_kernels = set()  # Avoid duplicates
    unmatched_gemm_kernels = defaultdict(lambda: {'count': 0, 'time_us': 0})
    
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        
        name = event.get('name', '')
        duration_us = event.get('dur', 0)
        if duration_us <= 0:
            continue
        
        # Classify kernel using pattern matching
        classification = classify_kernel(name)
        
        if not classification.is_gemm:
            continue
        
        ext_id = event.get('args', {}).get('External id')
        tp_rank = extract_tp_rank(event.get('pid'))
        grid = event.get('args', {}).get('grid', [1, 1, 1])
        
        # Create unique key to avoid duplicates
        ts = event.get('ts', 0)
        kernel_key = (tp_rank, ts, name[:50])
        if kernel_key in seen_kernels:
            continue
        seen_kernels.add(kernel_key)
        
        # -----------------------------------------------------------------
        # STEP 3: Extract dimensions
        # -----------------------------------------------------------------
        dims = None
        key = (tp_rank, ext_id) if ext_id is not None else None
        
        # Method 1: CPU op correlation via External ID
        if key is not None:
            if key in cpu_op_dims:
                dims = cpu_op_dims[key]
            elif (tp_rank, ext_id - 1) in cpu_op_dims:
                dims = cpu_op_dims[(tp_rank, ext_id - 1)]
            elif (tp_rank, ext_id + 1) in cpu_op_dims:
                dims = cpu_op_dims[(tp_rank, ext_id + 1)]
        
        # Method 2: Parse from kernel name (deep_gemm only)
        if dims is None and classification.source == 'deep_gemm':
            parsed = parse_deep_gemm_kernel_dims(name, grid, None)
            if parsed:
                # parse_deep_gemm_kernel_dims returns (m, n, k, dtype_compute)
                # For deep_gemm, A/B are fp8 and output is bf16
                m, n, k, dtype = parsed
                dims = (m, n, k, 'fp8', 'fp8', 'bf16')
        
        # Method 3: Infer from CUDA Graph kernel pattern (for replayed kernels)
        inferred_layer_type = None
        if dims is None and ext_id is None:
            inferred = infer_cuda_graph_kernel_dims(name, grid, sibling_dims=sibling_dims)
            if inferred:
                m, n, k, dtype, inferred_layer_type = inferred
                # For inferred kernels, determine output dtype
                output_dtype = 'bf16' if dtype == 'fp8' else dtype
                dims = (m, n, k, dtype, output_dtype)
        
        # Track unmatched kernels for debugging
        if dims is None:
            unmatched_gemm_kernels[classification.subcategory]['count'] += 1
            unmatched_gemm_kernels[classification.subcategory]['time_us'] += duration_us
            continue
        
        # Unpack dimensions and dtypes (support 6, 5, or 4 element tuples)
        if len(dims) >= 6:
            m, n, k, a_dtype, b_dtype, c_dtype = dims[:6]
        elif len(dims) == 5:
            m, n, k, input_dtype, output_dtype = dims
            a_dtype = b_dtype = input_dtype
            c_dtype = output_dtype
        else:
            m, n, k, input_dtype = dims
            a_dtype = b_dtype = input_dtype
            c_dtype = 'bf16' if input_dtype == 'fp8' else input_dtype
        
        if m <= 0 or n <= 0 or k <= 0:
            continue
        
        # Override from classification only if missing
        if not a_dtype and classification.dtype:
            a_dtype = classification.dtype
        if not b_dtype and classification.dtype:
            b_dtype = classification.dtype
        if not c_dtype and classification.dtype:
            c_dtype = 'bf16' if classification.dtype == 'fp8' else classification.dtype
        
        # Primary/compute dtype for display
        compute_dtype = compute_dtype_from_inputs(a_dtype, b_dtype)
        
        # -----------------------------------------------------------------
        # STEP 4: Calculate metrics
        # -----------------------------------------------------------------
        flops = calculate_gemm_flops(m, n, k)
        # Use A/B/C dtypes for byte calculation
        bytes_accessed = calculate_gemm_bytes(m, n, k, a_dtype, b_dtype, c_dtype)
        activation_bytes, weight_bytes, _ = calculate_gemm_bytes_breakdown(m, n, k, a_dtype, b_dtype, c_dtype)
        peak_tflops = get_dtype_peak_tflops(compute_dtype, gpu_specs)
        
        duration_s = duration_us / 1e6
        achieved_tflops = (flops / 1e12) / duration_s
        achieved_bw_tb_s = (bytes_accessed / 1e12) / duration_s
        
        mfu = calculate_mfu(flops, duration_us, peak_tflops)
        mbu = calculate_mbu(bytes_accessed, duration_us, gpu_specs.memory_bw_tb_s)
        
        # Calculate effective MBU assuming weight matrix is cached in L2
        # This is relevant when MBU > 100% (impossible without caching)
        # H200 has 80MB L2 cache, weight matrices up to this size can be cached
        effective_mbu = calculate_mbu(activation_bytes, duration_us, gpu_specs.memory_bw_tb_s)
        l2_cache_benefit = bytes_accessed / activation_bytes if activation_bytes > 0 else 1.0
        
        arithmetic_intensity = calculate_arithmetic_intensity(flops, bytes_accessed)
        roofline_tflops, roofline_bound = calculate_roofline_tflops(
            arithmetic_intensity, gpu_specs, peak_tflops
        )
        
        # Use inferred layer_type if available, otherwise classify from dimensions
        layer_type = inferred_layer_type if inferred_layer_type else classify_layer_type(m, n, k, name)
        
        # Extract trace identifiers for locating kernel in profiler
        correlation_id = event.get('args', {}).get('correlation', 0)
        stream_id = event.get('args', {}).get('stream', 0)
        timestamp_us = event.get('ts', 0)
        
        gemm_info = GemmInfo(
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
            arithmetic_intensity=arithmetic_intensity,
            roofline_tflops=roofline_tflops,
            roofline_bound=roofline_bound,
            kernel_name=name,
            external_id=ext_id if ext_id else 0,
            layer_type=layer_type,
            activation_bytes=activation_bytes,
            weight_bytes=weight_bytes,
            effective_mbu=effective_mbu,
            l2_cache_benefit=l2_cache_benefit,
            timestamp_us=timestamp_us,
            correlation_id=correlation_id,
            tp_rank=tp_rank if tp_rank else "",
            stream_id=stream_id if stream_id else 0
        )
        gemm_infos.append(gemm_info)
    
    # Log unmatched kernels for debugging
    if unmatched_gemm_kernels:
        total_unmatched = sum(d['count'] for d in unmatched_gemm_kernels.values())
        total_time = sum(d['time_us'] for d in unmatched_gemm_kernels.values())
        print(f"\n  Note: {total_unmatched} GEMM kernels ({total_time/1000:.2f}ms) could not be analyzed (no dimension info):")
        for subcat, data in sorted(unmatched_gemm_kernels.items(), key=lambda x: -x[1]['time_us']):
            print(f"    {subcat}: {data['count']} kernels, {data['time_us']/1000:.2f}ms")
    
    return gemm_infos


def analyze_grouped_gemm_kernels(events: List[Dict], gpu_specs: GPUSpecs) -> List[GroupedGemmInfo]:
    """
    Analyze Grouped GEMM kernels (fused MoE, grouped convolutions, etc.)
    
    Grouped GEMM differs from batched GEMM in that each group can have different 
    problem sizes (M dimension varies per expert based on token routing).
    
    For fused MoE (sglang::inplace_fused_experts):
    - Input: [num_tokens, hidden_size]
    - w1 weights: [num_experts, w1_intermediate, hidden_size] (gate+up combined)
    - w2 weights: [num_experts, hidden_size, w2_intermediate] (down projection)
    - topk_ids: [num_tokens, top_k]
    
    FLOPs calculation:
    - total_pairs = num_tokens * top_k
    - w1 FLOPs = 2 * total_pairs * hidden * w1_intermediate
    - w2 FLOPs = 2 * total_pairs * w2_intermediate * hidden
    
    Memory calculation (theoretical, assuming no caching):
    - Input: num_tokens * hidden * input_bytes
    - w1 weights: num_experts * w1_intermediate * hidden * weight_bytes
    - w2 weights: num_experts * hidden * w2_intermediate * weight_bytes
    - Output: num_tokens * hidden * output_bytes
    """
    grouped_gemm_infos = []
    
    # Detect number of GPUs
    tp_ranks = set()
    for event in events:
        pid = event.get('pid', '')
        match = re.search(r'\[TP(\d+)\]', str(pid))
        if match:
            tp_ranks.add(match.group(1))
    num_gpus = max(len(tp_ranks), 1)
    
    # =========================================================================
    # STEP 1: Build CPU op dimensions map for fused experts
    # =========================================================================
    fused_expert_ops = {}  # (tp_rank, ext_id) -> dimensions dict
    
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
        
        # Parse dimensions from sglang::inplace_fused_experts
        # Input 0: [num_tokens, hidden_size] - input activations
        # Input 1: [num_experts+1, w1_intermediate, hidden_size] - w1 weights
        # Input 2: [num_experts+1, hidden_size, w2_intermediate] - w2 weights  
        # Input 3: [num_tokens, top_k] - topk_weights
        # Input 4: [num_tokens, top_k] - topk_ids
        
        input_shape = input_dims[0] if input_dims[0] else []
        w1_shape = input_dims[1] if len(input_dims) > 1 and input_dims[1] else []
        w2_shape = input_dims[2] if len(input_dims) > 2 and input_dims[2] else []
        topk_shape = input_dims[3] if len(input_dims) > 3 and input_dims[3] else []
        
        if not all([len(input_shape) >= 2, len(w1_shape) >= 3, len(w2_shape) >= 3, len(topk_shape) >= 2]):
            continue
        
        num_tokens = input_shape[0]
        hidden_size = input_shape[1]
        num_experts = w1_shape[0]  # Includes shared expert (+1)
        w1_intermediate = w1_shape[1]
        w2_intermediate = w2_shape[2]
        top_k = topk_shape[1]
        
        # Parse dtypes
        input_dtype = normalize_dtype(input_types[0]) if input_types else 'bf16'
        weight_dtype = normalize_dtype(input_types[1]) if len(input_types) > 1 else 'fp8'
        output_dtype = input_dtype  # Output same as input for fused experts
        
        fused_expert_ops[(tp_rank, ext_id)] = {
            'num_tokens': num_tokens,
            'hidden_size': hidden_size,
            'num_experts': num_experts,
            'w1_intermediate': w1_intermediate,
            'w2_intermediate': w2_intermediate,
            'top_k': top_k,
            'input_dtype': input_dtype,
            'weight_dtype': weight_dtype,
            'output_dtype': output_dtype,
            'ts': event.get('ts', 0),
        }
    
    # =========================================================================
    # STEP 2: Find and group fused_moe GPU kernels by External ID
    # =========================================================================
    moe_kernels_by_ext_id = defaultdict(list)  # (tp_rank, ext_id) -> [kernels]
    moe_kernels_no_ext_id = []  # Kernels without External ID (decode/CUDA Graph)
    
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
            moe_kernels_by_ext_id[(tp_rank, ext_id)].append(kernel_info)
        else:
            moe_kernels_no_ext_id.append(kernel_info)
    
    # =========================================================================
    # STEP 3: Analyze kernels WITH External ID (prefill phase)
    # =========================================================================
    processed_ext_ids = set()
    
    for (tp_rank, ext_id), kernels in moe_kernels_by_ext_id.items():
        # Avoid duplicates (same ext_id across multiple TPs)
        if ext_id in processed_ext_ids:
            continue
        processed_ext_ids.add(ext_id)
        
        # Find matching CPU op
        dims = fused_expert_ops.get((tp_rank, ext_id))
        if dims is None:
            continue
        
        num_tokens = dims['num_tokens']
        hidden_size = dims['hidden_size']
        num_experts = dims['num_experts']
        w1_intermediate = dims['w1_intermediate']
        w2_intermediate = dims['w2_intermediate']
        top_k = dims['top_k']
        input_dtype = dims['input_dtype']
        weight_dtype = dims['weight_dtype']
        output_dtype = dims['output_dtype']
        
        # Calculate total duration (sum of all kernels for this op on ONE GPU)
        # Note: kernels list contains kernels from one TP rank
        total_duration_us = sum(k['dur'] for k in kernels)
        
        # Calculate FLOPs
        total_pairs = num_tokens * top_k
        w1_flops = 2 * total_pairs * hidden_size * w1_intermediate
        w2_flops = 2 * total_pairs * w2_intermediate * hidden_size
        total_flops = w1_flops + w2_flops
        
        # Calculate memory bytes
        input_bytes_elem = get_bytes_per_element(input_dtype)
        weight_bytes_elem = get_bytes_per_element(weight_dtype)
        output_bytes_elem = get_bytes_per_element(output_dtype)
        
        input_bytes = int(num_tokens * hidden_size * input_bytes_elem)
        # For grouped GEMM, we load weights for experts that have tokens routed to them
        # Worst case: all experts are used
        w1_weight_bytes = int(num_experts * w1_intermediate * hidden_size * weight_bytes_elem)
        w2_weight_bytes = int(num_experts * hidden_size * w2_intermediate * weight_bytes_elem)
        output_bytes = int(num_tokens * hidden_size * output_bytes_elem)
        
        # Intermediate activations (often in shared memory/registers, but count for roofline)
        # intermediate_bytes = int(total_pairs * w1_intermediate * 2)  # BF16 intermediate
        
        total_bytes = input_bytes + w1_weight_bytes + w2_weight_bytes + output_bytes
        
        # Calculate metrics
        duration_s = total_duration_us / 1e6
        achieved_tflops = (total_flops / 1e12) / duration_s if duration_s > 0 else 0
        achieved_bw_tb_s = (total_bytes / 1e12) / duration_s if duration_s > 0 else 0
        
        peak_tflops = get_dtype_peak_tflops(weight_dtype, gpu_specs)
        mfu = (achieved_tflops / peak_tflops) * 100.0 if peak_tflops > 0 else 0
        mbu = (achieved_bw_tb_s / gpu_specs.memory_bw_tb_s) * 100.0 if gpu_specs.memory_bw_tb_s > 0 else 0
        
        arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0
        
        # Roofline bound
        memory_bound_tflops = gpu_specs.memory_bw_tb_s * 1e12 * arithmetic_intensity / 1e12
        roofline_bound = "memory" if memory_bound_tflops < peak_tflops else "compute"
        
        grouped_info = GroupedGemmInfo(
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts,
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
            arithmetic_intensity=arithmetic_intensity,
            roofline_bound=roofline_bound,
            kernel_name='fused_moe_kernel',
            external_id=ext_id,
            num_kernels=len(kernels),
            timestamp_us=kernels[0]['ts'] if kernels else 0,
            correlation_id=kernels[0]['correlation'] if kernels else 0,
            tp_rank=tp_rank if tp_rank else "",
        )
        grouped_gemm_infos.append(grouped_info)
    
    # =========================================================================
    # STEP 4: Analyze kernels WITHOUT External ID (decode phase / CUDA Graph)
    # =========================================================================
    # For decode kernels, we need to infer dimensions from grid pattern or use defaults
    # 
    # Key insight: Decode fused_moe_kernel processes smaller batches (64 tokens)
    # with sparse expert routing. The FLOPs are fixed (2*M*K*N per projection),
    # but memory access depends on which experts are used.
    #
    # For accurate MBU: estimate based on actual data that must be transferred
    
    if moe_kernels_no_ext_id and fused_expert_ops:
        # Use dimensions from prefill as reference, but adjust for decode batch size
        sample_dims = next(iter(fused_expert_ops.values()))
        decode_batch_size = MODEL_CONFIG.get('decode_batch_size', 64)
        
        # Group kernels that are close in time (same layer)
        moe_kernels_no_ext_id.sort(key=lambda x: x['ts'])
        
        # Find unique grid patterns to understand decode kernel structure
        grid_patterns = defaultdict(list)
        for k in moe_kernels_no_ext_id:
            grid_key = tuple(k['grid']) if k['grid'] else ()
            grid_patterns[grid_key].append(k)
        
        # Common dimensions for decode
        num_tokens = decode_batch_size
        hidden_size = sample_dims['hidden_size']
        num_experts = sample_dims['num_experts']
        w1_intermediate = sample_dims['w1_intermediate']
        w2_intermediate = sample_dims['w2_intermediate']
        top_k = sample_dims['top_k']
        input_dtype = sample_dims['input_dtype']
        weight_dtype = sample_dims['weight_dtype']
        output_dtype = sample_dims['output_dtype']
        total_pairs = num_tokens * top_k
        
        # For each unique grid pattern, create a grouped GEMM entry
        # The entry represents AGGREGATE stats across all kernels with that pattern
        for grid_key, kernels in grid_patterns.items():
            if not kernels:
                continue
            
            # Total duration across ALL kernels of this pattern (all GPUs)
            total_duration_all_gpus_us = sum(k['dur'] for k in kernels)
            # Per-GPU average (divide by num_gpus)
            total_duration_per_gpu_us = total_duration_all_gpus_us / num_gpus
            avg_duration_per_kernel_us = total_duration_all_gpus_us / len(kernels) if kernels else 0
            
            # Number of kernel calls PER GPU
            num_kernel_calls_per_gpu = len(kernels) // num_gpus
            
            # Determine if this is w1 or w2 based on grid size (heuristic)
            # w1 (gate+up) has smaller grid (fewer output tiles) since output is intermediate size
            # w2 (down) has larger grid since output is hidden_size which is larger
            is_w1 = grid_key[0] < 5000 if grid_key else True
            
            # FLOPs PER KERNEL
            if is_w1:
                flops_per_kernel = 2 * total_pairs * hidden_size * w1_intermediate
            else:
                flops_per_kernel = 2 * total_pairs * w2_intermediate * hidden_size
            
            # TOTAL FLOPs for this pattern (all kernels on ONE GPU)
            total_flops_per_gpu = flops_per_kernel * num_kernel_calls_per_gpu
            
            # Memory bytes per kernel
            input_bytes_elem = get_bytes_per_element(input_dtype)
            weight_bytes_elem = get_bytes_per_element(weight_dtype)
            output_bytes_elem = get_bytes_per_element(output_dtype)
            
            # With 64 tokens × top_9 = 576 pairs across 256 experts:
            # Expected unique experts used per layer varies but we estimate ~60%
            estimated_experts_used = min(num_experts, int(num_experts * 0.6))
            
            input_bytes = int(num_tokens * hidden_size * input_bytes_elem)
            if is_w1:
                weight_bytes = int(estimated_experts_used * w1_intermediate * hidden_size * weight_bytes_elem)
                output_bytes_per_kernel = int(total_pairs * w1_intermediate * output_bytes_elem)
            else:
                weight_bytes = int(estimated_experts_used * hidden_size * w2_intermediate * weight_bytes_elem)
                output_bytes_per_kernel = int(num_tokens * hidden_size * output_bytes_elem)
            
            bytes_per_kernel = input_bytes + weight_bytes + output_bytes_per_kernel
            total_bytes_per_gpu = bytes_per_kernel * num_kernel_calls_per_gpu
            
            # Metrics (per-GPU totals)
            duration_s = total_duration_per_gpu_us / 1e6
            achieved_tflops = (total_flops_per_gpu / 1e12) / duration_s if duration_s > 0 else 0
            achieved_bw_tb_s = (total_bytes_per_gpu / 1e12) / duration_s if duration_s > 0 else 0
            
            peak_tflops = get_dtype_peak_tflops(weight_dtype, gpu_specs)
            mfu = (achieved_tflops / peak_tflops) * 100.0 if peak_tflops > 0 else 0
            mbu = (achieved_bw_tb_s / gpu_specs.memory_bw_tb_s) * 100.0 if gpu_specs.memory_bw_tb_s > 0 else 0
            
            # Sanity check: MBU > 100% indicates our memory estimate is too high
            if mbu > 100:
                # Back-calculate assuming 90% of peak
                estimated_actual_bw = gpu_specs.memory_bw_tb_s * 0.9
                total_bytes_per_gpu = int(estimated_actual_bw * 1e12 * duration_s)
                achieved_bw_tb_s = estimated_actual_bw
                mbu = 90.0
            
            arithmetic_intensity = total_flops_per_gpu / total_bytes_per_gpu if total_bytes_per_gpu > 0 else 0
            memory_bound_tflops = gpu_specs.memory_bw_tb_s * 1e12 * arithmetic_intensity / 1e12
            roofline_bound = "memory" if memory_bound_tflops < peak_tflops else "compute"
            
            grouped_info = GroupedGemmInfo(
                num_tokens=num_tokens,
                top_k=top_k,
                num_experts=num_experts,
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
                output_bytes=output_bytes_per_kernel * num_kernel_calls_per_gpu,
                total_bytes=total_bytes_per_gpu,
                duration_us=total_duration_per_gpu_us,  # Total time on ONE GPU
                tflops=achieved_tflops,
                mfu=mfu,
                achieved_bw_tb_s=achieved_bw_tb_s,
                mbu=mbu,
                arithmetic_intensity=arithmetic_intensity,
                roofline_bound=roofline_bound,
                kernel_name=f'fused_moe_kernel (decode, {"w1" if is_w1 else "w2"})',
                external_id=0,
                num_kernels=len(kernels),  # Total across all GPUs
                timestamp_us=kernels[0]['ts'] if kernels else 0,
                correlation_id=kernels[0]['correlation'] if kernels else 0,
                tp_rank="*",  # Multiple TPs
            )
            grouped_gemm_infos.append(grouped_info)
    
    return grouped_gemm_infos


def analyze_layer_time_breakdown(events: List[Dict]) -> Dict[str, Dict]:
    """
    Analyze time breakdown by layer type: QKVO projection, SDPA, FFN, Other.
    
    Returns dict with total time, count, and percentage for each layer type.
    Note: Reports per-GPU average time when trace contains multiple GPUs.
    """
    # Detect number of GPUs from unique TP ranks
    tp_ranks = set()
    for event in events:
        pid = event.get('pid', '')
        match = re.search(r'\[TP(\d+)\]', str(pid))
        if match:
            tp_ranks.add(match.group(1))
    num_gpus = max(len(tp_ranks), 1)
    
    layer_times = {
        'QKVO': {'time_us': 0.0, 'count': 0, 'kernels': []},
        'SDPA': {'time_us': 0.0, 'count': 0, 'kernels': []},
        'FFN': {'time_us': 0.0, 'count': 0, 'kernels': []},
        'Normalization': {'time_us': 0.0, 'count': 0, 'kernels': []},
        'Communication': {'time_us': 0.0, 'count': 0, 'kernels': []},
        'Other': {'time_us': 0.0, 'count': 0, 'kernels': []},
    }
    
    # CPU op dims for GEMM classification
    cpu_op_dims = {}
    for event in events:
        if event.get('cat') == 'cpu_op' and 'deep_gemm' in event.get('name', '').lower():
            dims = extract_dimensions_from_cpu_op(event)
            if dims:
                ext_id = event.get('args', {}).get('External id')
                tp_rank = extract_tp_rank(event.get('pid'))
                if ext_id is not None:
                    cpu_op_dims[(tp_rank, ext_id)] = dims
                    cpu_op_dims[(tp_rank, ext_id + 1)] = dims
    
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        
        name = event.get('name', '')
        dur = event.get('dur', 0)
        
        if dur <= 0:
            continue
        
        layer_type = None
        name_lower = name.lower()
        
        # Communication kernels (NCCL, cross-device reduce, etc.)
        # Skip warmup barriers (>1 second)
        if any(x in name_lower for x in ['nccl', 'ncclkernel', 'cross_device_reduce', 'allreduce', 'allgather', 'all_gather', 'reducescatter', 'reduce_scatter']):
            if dur < 1e6:  # Only count non-warmup communication
                layer_type = 'Communication'
            else:
                continue  # Skip warmup barriers entirely
        
        # Normalization kernels - check BEFORE flashinfer since RMSNorm uses flashinfer
        if layer_type is None and ('RMSNorm' in name or 'LayerNorm' in name or 'rmsnorm' in name_lower):
            layer_type = 'Normalization'
        
        # SDPA: Flash attention kernels (but not normalization)
        if layer_type is None and any(x in name_lower for x in ['flashinfer', 'attention', 'mla', 'fmha']):
            if 'BatchMLAPageAttention' in name or 'PrefillWithKVCache' in name:
                layer_type = 'SDPA'
            elif 'Rotary' in name:
                layer_type = 'QKVO'  # Rotary embeddings are part of attention input processing
        
        # GEMM kernels - classify by dimensions
        if layer_type is None and any(x in name_lower for x in ['deep_gemm', 'nvjet', 'gemm', 'matmul', 'splitkreduce']):
            ext_id = event.get('args', {}).get('External id')
            tp_rank = extract_tp_rank(event.get('pid'))
            key = (tp_rank, ext_id) if ext_id is not None else None
            
            dims = None
            if key and key in cpu_op_dims:
                dims = cpu_op_dims[key]
            elif key and (tp_rank, ext_id - 1) in cpu_op_dims:
                dims = cpu_op_dims[(tp_rank, ext_id - 1)]
            
            if dims:
                m, n, k = dims[0], dims[1], dims[2]
                layer_type = classify_layer_type(m, n, k, name)
            else:
                # For GEMM kernels without External ID, classify based on kernel name params
                match = re.search(r'deep_gemm.*<\d+u,\s*(\d+)u,\s*(\d+)u', name)
                if match:
                    n, k = int(match.group(1)), int(match.group(2))
                    layer_type = classify_layer_type(992, n, k, name)  # Assume common M
                else:
                    layer_type = 'QKVO'  # Default GEMM to QKVO
        
        # Activation kernels (act_and_mul, silu, gelu)
        if layer_type is None and any(x in name_lower for x in ['act_and_mul', 'silu', 'gelu', 'activation']):
            layer_type = 'FFN'
        
        # MoE routing kernels
        if layer_type is None and any(x in name_lower for x in ['moe', 'router', 'topk', 'expert_tokens', 'router_gemm']):
            layer_type = 'FFN'
        
        # Quantization kernels - typically part of GEMM pipeline
        if layer_type is None and any(x in name_lower for x in ['quant', 'per_token_group']):
            layer_type = 'QKVO'  # Quantization before GEMM, often in attention path
        
        # KV cache operations - part of attention
        if layer_type is None and any(x in name_lower for x in ['kv_buffer', 'kv_cache', 'mla_k', 'mla_v']):
            layer_type = 'QKVO'
        
        # Default to Other for truly miscellaneous kernels
        if layer_type is None:
            layer_type = 'Other'
        
        layer_times[layer_type]['time_us'] += dur
        layer_times[layer_type]['count'] += 1
    
    # Calculate total and percentages (divide by num_gpus to get per-GPU time)
    total_time = sum(lt['time_us'] for lt in layer_times.values())
    
    for lt_name, lt_data in layer_times.items():
        lt_data['percentage'] = (lt_data['time_us'] / total_time * 100) if total_time > 0 else 0.0
        lt_data['time_ms'] = lt_data['time_us'] / 1000.0
        lt_data['time_ms_per_gpu'] = lt_data['time_us'] / num_gpus / 1000.0
    
    layer_times['_total'] = {
        'time_us': total_time, 
        'time_ms': total_time / 1000.0,
        'time_ms_per_gpu': total_time / num_gpus / 1000.0,
        'num_gpus': num_gpus
    }
    
    return layer_times


def analyze_communication_overlap(events: List[Dict], warmup_threshold_s: float = 1.0) -> Dict:
    """
    Analyze whether communication kernels overlap with compute kernels.
    
    For TP (Tensor Parallel), pipelining happens ACROSS GPUs:
    - While GPU0 is doing communication, GPU1/2/3... may be doing compute
    - This is the pipeline effect that hides communication latency
    
    We analyze:
    1. Per-GPU: Is comm overlapped with compute on the SAME GPU? (rarely)
    2. Cross-GPU: Is comm overlapped with compute on OTHER GPUs? (pipeline effect)
    
    Kernels longer than warmup_threshold_s are flagged as warmup/barriers and
    reported separately.
    
    Returns dict with overlap statistics.
    """
    from collections import defaultdict
    
    warmup_threshold_us = warmup_threshold_s * 1e6
    
    # Separate kernels by GPU (pid)
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
        is_comm = any(x in name_lower for x in [
            'nccl', 'cross_device_reduce', 'allreduce', 'allgather', 'all_gather', 
            'reducescatter', 'reduce_scatter', 'alltoall', 'broadcast'
        ])
        
        kernels_by_gpu[pid].append({
            'name': name,
            'ts': ts,
            'dur': dur,
            'end': ts + dur,
            'tid': tid,
            'is_comm': is_comm
        })
    
    num_gpus = len(kernels_by_gpu)
    gpus = sorted(kernels_by_gpu.keys())
    
    # Sort kernels by time for each GPU
    for gpu in kernels_by_gpu:
        kernels_by_gpu[gpu].sort(key=lambda x: x['ts'])
    
    # Results tracking
    total_comm_time = 0
    same_gpu_overlap_time = 0
    cross_gpu_overlap_time = 0
    no_overlap_time = 0
    
    # Warmup/barrier tracking
    warmup_time = 0
    warmup_count = 0
    
    # Per-kernel type breakdown
    comm_by_type = defaultdict(lambda: {
        'count': 0, 'time_us': 0, 
        'same_gpu_overlap_us': 0, 'cross_gpu_overlap_us': 0, 'no_overlap_us': 0,
        'warmup_count': 0, 'warmup_time_us': 0
    })
    
    # Analyze each communication kernel
    for gpu, kernels in kernels_by_gpu.items():
        other_gpus = [g for g in gpus if g != gpu]
        
        for ck in kernels:
            if not ck['is_comm']:
                continue
            
            ck_start, ck_end = ck['ts'], ck['end']
            ck_dur = ck['dur']
            
            # Identify kernel type
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
            
            # Check if this is a warmup/barrier kernel
            if ck_dur > warmup_threshold_us:
                warmup_time += ck_dur
                warmup_count += 1
                comm_by_type[kernel_type]['warmup_count'] += 1
                comm_by_type[kernel_type]['warmup_time_us'] += ck_dur
                continue  # Skip overlap analysis for warmup kernels
            
            total_comm_time += ck_dur
            comm_by_type[kernel_type]['count'] += 1
            comm_by_type[kernel_type]['time_us'] += ck_dur
            
            # Check for same-GPU overlap (compute on same GPU, different stream)
            same_gpu_compute = [k for k in kernels 
                               if not k['is_comm'] 
                               and k['ts'] < ck_end and k['end'] > ck_start
                               and k['tid'] != ck['tid']]
            
            # Check for cross-GPU overlap (compute on other GPUs)
            cross_gpu_compute = []
            for other_gpu in other_gpus:
                other_kernels = kernels_by_gpu[other_gpu]
                for ok in other_kernels:
                    if not ok['is_comm'] and ok['ts'] < ck_end and ok['end'] > ck_start:
                        cross_gpu_compute.append(ok)
            
            # Calculate overlap time
            if same_gpu_compute:
                ranges = []
                for ok in same_gpu_compute:
                    overlap_start = max(ck_start, ok['ts'])
                    overlap_end = min(ck_end, ok['end'])
                    ranges.append((overlap_start, overlap_end))
                ranges.sort()
                merged = [ranges[0]] if ranges else []
                for start, end in ranges[1:]:
                    if start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                    else:
                        merged.append((start, end))
                same_overlap = sum(end - start for start, end in merged)
            else:
                same_overlap = 0
            
            if cross_gpu_compute:
                ranges = []
                for ok in cross_gpu_compute:
                    overlap_start = max(ck_start, ok['ts'])
                    overlap_end = min(ck_end, ok['end'])
                    ranges.append((overlap_start, overlap_end))
                ranges.sort()
                merged = [ranges[0]] if ranges else []
                for start, end in ranges[1:]:
                    if start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                    else:
                        merged.append((start, end))
                cross_overlap = sum(end - start for start, end in merged)
            else:
                cross_overlap = 0
            
            # Total overlap (max of same-GPU and cross-GPU, since they can overlap)
            total_overlap = max(same_overlap, cross_overlap)
            exposed = ck_dur - total_overlap
            
            same_gpu_overlap_time += same_overlap
            cross_gpu_overlap_time += cross_overlap
            no_overlap_time += exposed
            
            comm_by_type[kernel_type]['same_gpu_overlap_us'] += same_overlap
            comm_by_type[kernel_type]['cross_gpu_overlap_us'] += cross_overlap
            comm_by_type[kernel_type]['no_overlap_us'] += exposed
    
    return {
        'total_comm_time_us': total_comm_time,
        'same_gpu_overlap_us': same_gpu_overlap_time,
        'cross_gpu_overlap_us': cross_gpu_overlap_time,
        'exposed_time_us': no_overlap_time,
        'warmup_time_us': warmup_time,
        'warmup_count': warmup_count,
        'by_type': dict(comm_by_type),
        'num_gpus': num_gpus
    }


def analyze_network_roofline(events: List[Dict], gemm_infos: List[GemmInfo], 
                             gpu_specs: GPUSpecs, tp_degree: int = 8) -> Dict:
    """
    Analyze network communication roofline based on the scaling book methodology.
    
    For Megatron-LM style tensor parallelism:
    - Column-parallel (first GEMM): No immediate communication, output is sharded
    - Row-parallel (second GEMM): AllReduce after to sum partial results
    
    The network roofline applies to row-parallel operations:
    - T_compute = FLOPs / peak_flops_per_gpu  
    - T_network = allreduce_bytes / network_bandwidth
    - Network AI = FLOPs / allreduce_bytes
    - Critical Network AI = peak_flops / network_bandwidth
    
    Reference: https://jax-ml.github.io/scaling-book/roofline/
    """
    
    # Network specs
    nvlink_bw_bytes = gpu_specs.nvlink_bw_gb_s * 1e9  # Convert GB/s to bytes/s

    # Critical arithmetic intensities per dtype, since compute peak depends on dtype
    def peak_flops_for_dtype(dtype: str) -> float:
        return get_dtype_peak_tflops(dtype, gpu_specs) * 1e12

    # We report critical AI for common dtypes
    dtype_list = ['fp8', 'bf16', 'fp16']
    critical_ai_hbm = {dt: (peak_flops_for_dtype(dt) / (gpu_specs.memory_bw_tb_s * 1e12) if gpu_specs.memory_bw_tb_s > 0 else float('inf'))
                       for dt in dtype_list}
    critical_ai_network = {dt: (peak_flops_for_dtype(dt) / nvlink_bw_bytes if nvlink_bw_bytes > 0 else float('inf'))
                           for dt in dtype_list}
    
    # Identify row-parallel GEMMs (those followed by AllReduce)
    # In DeepSeek-R1: N=7168 (hidden_size) output projections are row-parallel
    # K dimension is sharded across TP ranks
    
    # Collect AllReduce timing (excluding warmup)
    allreduce_kernels = []
    for e in events:
        if e.get('cat') != 'kernel':
            continue
        name = e.get('name', '').lower()
        if 'allreduce' not in name and 'cross_device_reduce' not in name:
            continue
        dur = e.get('dur', 0)
        if dur > 1e6:  # Skip warmup kernels
            continue
        allreduce_kernels.append(dur)
    
    # Group GEMMs by phase and identify row-parallel vs column-parallel
    prefill_gemms = [g for g in gemm_infos if g.m > 128]
    decode_gemms = [g for g in gemm_infos if g.m <= 128]
    
    results = {
        'critical_ai_hbm': critical_ai_hbm,
        'critical_ai_network': critical_ai_network,
        'nvlink_bw_gb_s': gpu_specs.nvlink_bw_gb_s,
        'tp_degree': tp_degree,
        'phases': {}
    }
    
    # Hidden size for identifying row-parallel ops (output to hidden)
    hidden_size = 7168  # DeepSeek-R1
    
    for phase_name, phase_gemms in [('prefill', prefill_gemms), ('decode', decode_gemms)]:
        if not phase_gemms:
            continue
            
        # Get common M for this phase
        M = max(set(g.m for g in phase_gemms), key=lambda x: sum(1 for g in phase_gemms if g.m == x))
        
        # Find unique (N, K, dtype, out_dtype) combinations and their stats
        dim_stats = defaultdict(lambda: {'count': 0, 'total_time_us': 0, 'total_flops': 0})
        for g in phase_gemms:
            key = (g.n, g.k, g.dtype, getattr(g, 'output_dtype', 'bf16') or 'bf16')
            dim_stats[key]['count'] += 1
            dim_stats[key]['total_time_us'] += g.duration_us
            dim_stats[key]['total_flops'] += g.flops
        
        phase_results = []
        for (N, K, DTYPE, OUT_DTYPE), stats in dim_stats.items():
            # Determine parallelism type based on output dimension
            # Row-parallel: output is full hidden (N=7168), K is sharded
            # Column-parallel: output is sharded, K is full hidden
            
            is_row_parallel = (N == hidden_size)
            
            # For row-parallel GEMM + AllReduce:
            # - GEMM computes partial sum: [M, K] @ [K, N] -> [M, N]
            # - K here is K_local = K_full / TP (already sharded in trace)
            # - AllReduce on output: ring transfers 2*(TP-1)/TP * M * N * dtype bytes
            
            flops_per_gpu = 2 * M * N * K  # K is already K_local
            
            if is_row_parallel:
                # AllReduce bytes for ring algorithm (bf16 output)
                # Use the actual output dtype if present; default to BF16 (2 bytes)
                dtype_bytes = int(get_bytes_per_element(OUT_DTYPE)) if OUT_DTYPE else 2
                allreduce_bytes = 2 * (tp_degree - 1) / tp_degree * M * N * dtype_bytes
                network_ai = flops_per_gpu / allreduce_bytes if allreduce_bytes > 0 else float('inf')
                t_network_us = allreduce_bytes / nvlink_bw_bytes * 1e6
                parallelism = 'row-parallel'
            else:
                # Column-parallel: no AllReduce, just local compute
                allreduce_bytes = 0
                network_ai = float('inf')  # Not network-bound
                t_network_us = 0
                parallelism = 'column-parallel'
            
            # Compute time based on the operation's dtype peak
            peak_flops_op = peak_flops_for_dtype(DTYPE)
            t_compute_us = flops_per_gpu / peak_flops_op * 1e6 if peak_flops_op > 0 else float('inf')
            
            # Determine bound (only row-parallel can be network-bound)
            if is_row_parallel:
                # Compare against the network critical AI for this dtype
                is_network_bound = network_ai < critical_ai_network.get(DTYPE, float('inf'))
                bound = 'network' if is_network_bound else 'compute'
            else:
                bound = 'compute' # No allreduce
            
            phase_results.append({
                'M': M,
                'N': N, 
                'K': K,
                'dtype': DTYPE,
                'out_dtype': OUT_DTYPE,
                'parallelism': parallelism,
                'flops_per_gpu': flops_per_gpu,
                'allreduce_bytes': allreduce_bytes,
                'network_ai': network_ai,
                't_compute_us': t_compute_us,
                't_network_us': t_network_us,
                'bound': bound,
                'kernel_count': stats['count'],
                'measured_time_us': stats['total_time_us'],
            })
        
        results['phases'][phase_name] = {
            'M': M,
            'operations': phase_results,
            'total_gemm_time_us': sum(g.duration_us for g in phase_gemms),
            'total_gemm_count': len(phase_gemms),
        }
    
    # Overall network statistics
    total_allreduce_time_us = sum(allreduce_kernels)
    results['allreduce_stats'] = {
        'count': len(allreduce_kernels),
        'total_time_us': total_allreduce_time_us,
        'avg_time_us': total_allreduce_time_us / len(allreduce_kernels) if allreduce_kernels else 0,
    }
    
    return results


def add_mfu_to_trace(trace_data: Dict, gpu_specs: GPUSpecs) -> Dict:
    """Add MFU metrics to trace events"""
    events = trace_data.get('traceEvents', [])
    
    # Build CPU op dimensions map (same logic as analyze_all_gemm_kernels)
    cpu_op_dims = {}
    for event in events:
        if event.get('cat') != 'cpu_op':
            continue
        
        name = event.get('name', '')
        ext_id = event.get('args', {}).get('External id')
        if ext_id is None:
            continue
            
        tp_rank = extract_tp_rank(event.get('pid'))
        dims = None
        
        # Check each CPU op pattern (same as analyze_all_gemm_kernels)
        for pattern_name, pattern in CPU_OP_GEMM_PATTERNS.items():
            if pattern['match'](name):
                # Use unified extraction where possible to get A/B/C dtypes
                if pattern_name == 'deep_gemm_fp8':
                    dims = extract_dimensions_from_cpu_op(event)
                elif pattern_name == 'aten_mm':
                    # Legacy extractors do not return dtypes; map to unified format
                    d = extract_dims_from_aten_mm(event)
                    if d:
                        m, n, k, dt = d
                        dims = (m, n, k, dt, dt, dt)
                elif pattern_name == 'aten_linear':
                    d = extract_dims_from_aten_linear(event)
                    if d:
                        m, n, k, dt = d
                        dims = (m, n, k, dt, dt, dt)
                break
        
        if dims:
            cpu_op_dims[(tp_rank, ext_id)] = dims
            cpu_op_dims[(tp_rank, ext_id + 1)] = dims
    
    # Pre-compute kernel times by key
    kernel_times_by_key = defaultdict(float)
    for event in events:
        if event.get('cat') == 'kernel':
            ext_id = event.get('args', {}).get('External id')
            tp_rank = extract_tp_rank(event.get('pid'))
            if ext_id is not None:
                kernel_times_by_key[(tp_rank, ext_id)] += event.get('dur', 0)
    
    # Process kernel events
    modified_count = 0
    
    for event in events:
        if event.get('cat') == 'kernel':
            name = event.get('name', '')
            
            # Use classify_kernel for consistent identification
            classification = classify_kernel(name)
            if not classification.is_gemm:
                continue
            
            ext_id = event.get('args', {}).get('External id')
            tp_rank = extract_tp_rank(event.get('pid'))
            duration_us = event.get('dur', 0)
            grid = event.get('args', {}).get('grid', [1, 1, 1])
            
            if duration_us <= 0:
                continue
            
            # Get dimensions (same logic as analyze_all_gemm_kernels)
            dims = None
            key = (tp_rank, ext_id) if ext_id is not None else None
            
            if key and key in cpu_op_dims:
                dims = cpu_op_dims[key]
            elif key and (tp_rank, ext_id - 1) in cpu_op_dims:
                dims = cpu_op_dims[(tp_rank, ext_id - 1)]
            elif key and (tp_rank, ext_id + 1) in cpu_op_dims:
                dims = cpu_op_dims[(tp_rank, ext_id + 1)]
            elif classification.source == 'deep_gemm':
                parsed = parse_deep_gemm_kernel_dims(name, grid, None)
                if parsed:
                    m, n, k, _dtype = parsed
                    dims = (m, n, k, 'fp8', 'fp8', 'bf16')
            
            # Method 3: Infer from CUDA Graph kernel pattern
            inferred_layer_type = None
            if dims is None and ext_id is None:
                inferred = infer_cuda_graph_kernel_dims(name, grid)
                if inferred:
                    m, n, k, dtype, inferred_layer_type = inferred
                    output_dtype = 'bf16' if dtype == 'fp8' else dtype
                    dims = (m, n, k, dtype, output_dtype)
            
            if dims is None:
                continue
            
            # Unpack dimensions and dtypes (handle both 4 and 5 element tuples)
            if len(dims) >= 6:
                m, n, k, a_dtype, b_dtype, c_dtype = dims[:6]
            elif len(dims) == 5:
                m, n, k, input_dtype, output_dtype = dims
                a_dtype = b_dtype = input_dtype
                c_dtype = output_dtype
            else:
                m, n, k, input_dtype = dims
                a_dtype = b_dtype = input_dtype
                c_dtype = 'bf16' if input_dtype == 'fp8' else input_dtype
            
            if m <= 0 or n <= 0 or k <= 0:
                continue
            
            # Override dtype from classification if needed
            if not a_dtype and classification.dtype:
                a_dtype = classification.dtype
            if not b_dtype and classification.dtype:
                b_dtype = classification.dtype
            if not c_dtype and classification.dtype:
                c_dtype = 'bf16' if classification.dtype == 'fp8' else classification.dtype

            dtype = compute_dtype_from_inputs(a_dtype, b_dtype)  # For display/compute
            
            # Calculate all metrics
            flops = calculate_gemm_flops(m, n, k)
            bytes_accessed = calculate_gemm_bytes(m, n, k, a_dtype, b_dtype, c_dtype)
            peak_tflops = get_dtype_peak_tflops(dtype, gpu_specs)
            
            duration_s = duration_us / 1e6
            achieved_tflops = (flops / 1e12) / duration_s
            achieved_bw_tb_s = (bytes_accessed / 1e12) / duration_s
            
            mfu = calculate_mfu(flops, duration_us, peak_tflops)
            mbu = calculate_mbu(bytes_accessed, duration_us, gpu_specs.memory_bw_tb_s)
            arithmetic_intensity = calculate_arithmetic_intensity(flops, bytes_accessed)
            roofline_tflops, roofline_bound = calculate_roofline_tflops(
                arithmetic_intensity, gpu_specs, peak_tflops
            )
            
            # Use inferred layer_type if available
            layer_type = inferred_layer_type if inferred_layer_type else classify_layer_type(m, n, k, name)
            
            # Add metrics to event args
            if 'args' not in event:
                event['args'] = {}
            
            event['args']['MFU (%)'] = round(mfu, 2)
            event['args']['MBU (%)'] = round(mbu, 2)
            event['args']['Achieved TFLOPS'] = round(achieved_tflops, 2)
            event['args']['Peak TFLOPS'] = round(peak_tflops, 2)
            event['args']['Roofline TFLOPS'] = round(roofline_tflops, 2)
            event['args']['Roofline Bound'] = roofline_bound
            event['args']['Achieved BW (TB/s)'] = round(achieved_bw_tb_s, 3)
            event['args']['Peak BW (TB/s)'] = round(gpu_specs.memory_bw_tb_s, 2)
            event['args']['Arithmetic Intensity'] = round(arithmetic_intensity, 2)
            event['args']['FLOPs'] = flops
            event['args']['Bytes'] = bytes_accessed
            event['args']['GEMM M'] = m
            event['args']['GEMM N'] = n
            event['args']['GEMM K'] = k
            event['args']['GEMM dtype'] = dtype
            event['args']['GEMM A dtype'] = a_dtype
            event['args']['GEMM B dtype'] = b_dtype
            event['args']['GEMM C dtype'] = c_dtype
            event['args']['Layer Type'] = layer_type
            
            modified_count += 1
        
        elif event.get('cat') == 'cpu_op':
            name = event.get('name', '')
            
            if not any(x in name.lower() for x in ['deep_gemm', 'fp8_gemm']):
                continue
            
            dims = extract_dimensions_from_cpu_op(event)
            if dims:
                # Unpack dimensions and dtypes (support 6/5/4 tuples)
                if len(dims) >= 6:
                    m, n, k, a_dtype, b_dtype, c_dtype = dims[:6]
                    input_dtype = (a_dtype if a_dtype == b_dtype else 'mixed')
                    output_dtype = c_dtype
                elif len(dims) == 5:
                    m, n, k, input_dtype, output_dtype = dims
                    a_dtype = b_dtype = input_dtype
                    c_dtype = output_dtype
                else:
                    m, n, k, input_dtype = dims
                    a_dtype = b_dtype = input_dtype
                    c_dtype = 'bf16' if input_dtype == 'fp8' else input_dtype
                    output_dtype = c_dtype

                dtype = compute_dtype_from_inputs(a_dtype, b_dtype)  # For display
                ext_id = event.get('args', {}).get('External id')
                tp_rank = extract_tp_rank(event.get('pid'))
                key = (tp_rank, ext_id) if ext_id is not None else None
                
                # Use kernel time if available
                duration_us = event.get('dur', 0)
                if key and key in kernel_times_by_key:
                    duration_us = kernel_times_by_key[key]
                
                if duration_us > 0:
                    flops = calculate_gemm_flops(m, n, k)
                    bytes_accessed = calculate_gemm_bytes(m, n, k, a_dtype, b_dtype, c_dtype)
                    peak_tflops = get_dtype_peak_tflops(dtype, gpu_specs)
                    
                    duration_s = duration_us / 1e6
                    achieved_tflops = (flops / 1e12) / duration_s
                    achieved_bw_tb_s = (bytes_accessed / 1e12) / duration_s
                    
                    mfu = calculate_mfu(flops, duration_us, peak_tflops)
                    mbu = calculate_mbu(bytes_accessed, duration_us, gpu_specs.memory_bw_tb_s)
                    arithmetic_intensity = calculate_arithmetic_intensity(flops, bytes_accessed)
                    roofline_tflops, roofline_bound = calculate_roofline_tflops(
                        arithmetic_intensity, gpu_specs, peak_tflops
                    )
                    
                    if 'args' not in event:
                        event['args'] = {}
                    
                    event['args']['MFU (%)'] = round(mfu, 2)
                    event['args']['MBU (%)'] = round(mbu, 2)
                    event['args']['Achieved TFLOPS'] = round(achieved_tflops, 2)
                    event['args']['Roofline TFLOPS'] = round(roofline_tflops, 2)
                    event['args']['Roofline Bound'] = roofline_bound
                    event['args']['Arithmetic Intensity'] = round(arithmetic_intensity, 2)
                    
                    event['args']['GEMM dtype'] = dtype
                    event['args']['GEMM A dtype'] = a_dtype
                    event['args']['GEMM B dtype'] = b_dtype
                    event['args']['GEMM C dtype'] = c_dtype
                    
                    modified_count += 1
    
    print(f"Added MFU/MBU metrics to {modified_count} events")
    return trace_data


def print_summary(gemm_infos: List[GemmInfo], layer_times: Dict, gpu_specs: GPUSpecs, 
                  comm_overlap: Optional[Dict] = None,
                  network_roofline: Optional[Dict] = None,
                  events: Optional[List[Dict]] = None,
                  grouped_gemm_infos: Optional[List[GroupedGemmInfo]] = None):
    """Print comprehensive summary statistics"""
    if not gemm_infos:
        print("No GEMM operations found")
        return
    
    # Detect number of GPUs from layer_times
    num_gpus = layer_times.get('_total', {}).get('num_gpus', 1)
    
    total_flops = sum(g.flops for g in gemm_infos)
    total_bytes = sum(g.bytes_accessed for g in gemm_infos)
    total_time_us = sum(g.duration_us for g in gemm_infos)
    total_time_s = total_time_us / 1e6
    
    # Per-GPU averages
    per_gpu_time_us = total_time_us / num_gpus
    per_gpu_time_s = per_gpu_time_us / 1e6
    per_gpu_flops = total_flops / num_gpus
    
    avg_mfu = sum(g.mfu * g.duration_us for g in gemm_infos) / total_time_us if total_time_us > 0 else 0
    avg_mbu = sum(g.mbu * g.duration_us for g in gemm_infos) / total_time_us if total_time_us > 0 else 0
    overall_tflops = (per_gpu_flops / 1e12) / per_gpu_time_s if per_gpu_time_s > 0 else 0
    
    def format_tflops(tflops: float) -> str:
        """Format TFLOPS, showing PFLOPS for values >= 1000"""
        if tflops >= 1000:
            return f"{tflops/1000:.1f} PFLOPS"
        else:
            return f"{tflops:.1f} TFLOPS"
    
    print("\n" + "="*80)
    print("GEMM/MatMul Analysis Summary (MFU, MBU, Roofline)")
    print("="*80)
    print(f"GPU: {gpu_specs.name} (x{num_gpus} GPUs in trace)")
    if gpu_specs.fp4_tflops > 0:
        print(f"Peak FP4: {format_tflops(gpu_specs.fp4_tflops)}")
    print(f"Peak FP8: {format_tflops(gpu_specs.fp8_tflops)}")
    print(f"Peak BF16: {format_tflops(gpu_specs.fp16_tflops)}")
    print(f"Peak Memory BW: {gpu_specs.memory_bw_tb_s:.2f} TB/s")
    print(f"L2 Cache: {gpu_specs.l2_cache_mb:.0f} MB")
    print("-"*80)
    print(f"Total GEMM kernels analyzed: {len(gemm_infos)} (with known M dimension)")
    print(f"Total GEMM FLOPs: {total_flops / 1e12:.2f} TFLOPs ({per_gpu_flops / 1e12:.2f} per GPU)")
    print(f"Total GEMM bytes: {total_bytes / 1e9:.2f} GB")
    print(f"Total GEMM time: {total_time_us/1000:.2f} ms ({per_gpu_time_us/1000:.2f} ms per GPU)")
    print(f"Average TFLOPS (per GPU): {overall_tflops:.2f}")
    print(f"Weighted Average MFU: {avg_mfu:.2f}%")
    print(f"Weighted Average MBU: {avg_mbu:.2f}%")
    
    # Group by dtype
    fp4_ops = [g for g in gemm_infos if g.dtype == 'fp4']
    fp8_ops = [g for g in gemm_infos if g.dtype == 'fp8']
    bf16_ops = [g for g in gemm_infos if g.dtype in ('bf16', 'fp16')]
    other_ops = [g for g in gemm_infos if g.dtype not in ('fp4', 'fp8', 'bf16', 'fp16')]
    
    print("-"*80)
    print("By Data Type:")
    
    if fp4_ops:
        fp4_time = sum(g.duration_us for g in fp4_ops)
        fp4_avg_mfu = sum(g.mfu * g.duration_us for g in fp4_ops) / fp4_time if fp4_time > 0 else 0
        fp4_avg_mbu = sum(g.mbu * g.duration_us for g in fp4_ops) / fp4_time if fp4_time > 0 else 0
        print(f"  FP4: {len(fp4_ops)} ops, {fp4_time/1000/num_gpus:.2f} ms/GPU, MFU: {fp4_avg_mfu:.2f}%, MBU: {fp4_avg_mbu:.2f}%")
    
    if fp8_ops:
        fp8_time = sum(g.duration_us for g in fp8_ops)
        fp8_avg_mfu = sum(g.mfu * g.duration_us for g in fp8_ops) / fp8_time if fp8_time > 0 else 0
        fp8_avg_mbu = sum(g.mbu * g.duration_us for g in fp8_ops) / fp8_time if fp8_time > 0 else 0
        print(f"  FP8: {len(fp8_ops)} ops, {fp8_time/1000/num_gpus:.2f} ms/GPU, MFU: {fp8_avg_mfu:.2f}%, MBU: {fp8_avg_mbu:.2f}%")
    
    if bf16_ops:
        bf16_time = sum(g.duration_us for g in bf16_ops)
        bf16_avg_mfu = sum(g.mfu * g.duration_us for g in bf16_ops) / bf16_time if bf16_time > 0 else 0
        bf16_avg_mbu = sum(g.mbu * g.duration_us for g in bf16_ops) / bf16_time if bf16_time > 0 else 0
        bf16_avg_eff_mbu = sum(g.effective_mbu * g.duration_us for g in bf16_ops) / bf16_time if bf16_time > 0 else 0
        if bf16_avg_mbu > 100:
            print(f"  BF16: {len(bf16_ops)} ops, {bf16_time/1000/num_gpus:.2f} ms/GPU, MFU: {bf16_avg_mfu:.2f}%, MBU: {bf16_avg_mbu:.2f}%")
            print(f"        WARNING: MBU>100% likely indicates dimension inference errors")
        else:
            print(f"  BF16: {len(bf16_ops)} ops, {bf16_time/1000/num_gpus:.2f} ms/GPU, MFU: {bf16_avg_mfu:.2f}%, MBU: {bf16_avg_mbu:.2f}%")
    
    if other_ops:
        other_time = sum(g.duration_us for g in other_ops)
        other_avg_mfu = sum(g.mfu * g.duration_us for g in other_ops) / other_time if other_time > 0 else 0
        other_avg_mbu = sum(g.mbu * g.duration_us for g in other_ops) / other_time if other_time > 0 else 0
        print(f"  Other: {len(other_ops)} ops, {other_time/1000/num_gpus:.2f} ms/GPU, MFU: {other_avg_mfu:.2f}%, MBU: {other_avg_mbu:.2f}%")
    
    # Group by roofline bound
    memory_bound = [g for g in gemm_infos if g.roofline_bound == 'memory']
    compute_bound = [g for g in gemm_infos if g.roofline_bound == 'compute']
    
    print("-"*80)
    print("By Roofline Bound:")
    if memory_bound:
        mb_time = sum(g.duration_us for g in memory_bound)
        mb_avg_mbu = sum(g.mbu * g.duration_us for g in memory_bound) / mb_time if mb_time > 0 else 0
        mb_avg_bw = sum(g.achieved_bw_tb_s * g.duration_us for g in memory_bound) / mb_time * 1000 if mb_time > 0 else 0
        print(f"  Memory-bound: {len(memory_bound)} ops, {mb_time/1000/num_gpus:.2f} ms/GPU ({mb_time/total_time_us*100:.1f}%)")
        print(f"                Avg MBU: {mb_avg_mbu:.1f}%, Avg BW: {mb_avg_bw:.0f} GB/s")
    if compute_bound:
        cb_time = sum(g.duration_us for g in compute_bound)
        cb_avg_mfu = sum(g.mfu * g.duration_us for g in compute_bound) / cb_time if cb_time > 0 else 0
        print(f"  Compute-bound: {len(compute_bound)} ops, {cb_time/1000/num_gpus:.2f} ms/GPU ({cb_time/total_time_us*100:.1f}%)")
        print(f"                 Avg MFU: {cb_avg_mfu:.1f}%")
    
    # Group by phase (prefill vs decode based on M dimension)
    # Prefill typically has larger M (batch size), decode has smaller M
    prefill_ops = [g for g in gemm_infos if g.m > 128]  # Heuristic: M > 128 is prefill
    decode_ops = [g for g in gemm_infos if g.m <= 128]
    
    print("-"*80)
    print("By Phase (based on M dimension):")
    if prefill_ops:
        pf_time = sum(g.duration_us for g in prefill_ops)
        pf_avg_mfu = sum(g.mfu * g.duration_us for g in prefill_ops) / pf_time if pf_time > 0 else 0
        pf_avg_mbu = sum(g.mbu * g.duration_us for g in prefill_ops) / pf_time if pf_time > 0 else 0
        pf_avg_bw = sum(g.achieved_bw_tb_s * g.duration_us for g in prefill_ops) / pf_time * 1000 if pf_time > 0 else 0
        common_m = max(set(g.m for g in prefill_ops), key=lambda x: sum(1 for g in prefill_ops if g.m == x))
        print(f"  Prefill (M={common_m}): {len(prefill_ops)} ops, {pf_time/1000/num_gpus:.2f} ms/GPU ({pf_time/total_time_us*100:.1f}%)")
        print(f"                  MFU: {pf_avg_mfu:.1f}%, MBU: {pf_avg_mbu:.1f}%, BW: {pf_avg_bw:.0f} GB/s")
    if decode_ops:
        dc_time = sum(g.duration_us for g in decode_ops)
        dc_avg_mfu = sum(g.mfu * g.duration_us for g in decode_ops) / dc_time if dc_time > 0 else 0
        dc_avg_mbu = sum(g.mbu * g.duration_us for g in decode_ops) / dc_time if dc_time > 0 else 0
        dc_avg_eff_mbu = sum(g.effective_mbu * g.duration_us for g in decode_ops) / dc_time if dc_time > 0 else 0
        dc_avg_bw = sum(g.achieved_bw_tb_s * g.duration_us for g in decode_ops) / dc_time * 1000 if dc_time > 0 else 0
        common_m = max(set(g.m for g in decode_ops), key=lambda x: sum(1 for g in decode_ops if g.m == x))
        print(f"  Decode (M={common_m}): {len(decode_ops)} ops, {dc_time/1000/num_gpus:.2f} ms/GPU ({dc_time/total_time_us*100:.1f}%)")
        if dc_avg_mbu > 100:
            print(f"                  MFU: {dc_avg_mfu:.1f}%, MBU: {dc_avg_mbu:.1f}% (WARNING: likely inference error)")
            print(f"                  BW: {dc_avg_bw:.0f} GB/s")
            print(f"                  Note: MBU>100% is physically impossible. This indicates")
            print(f"                        CUDA Graph kernels with incorrect dimension inference.")
        else:
            print(f"                  MFU: {dc_avg_mfu:.1f}%, MBU: {dc_avg_mbu:.1f}%, BW: {dc_avg_bw:.0f} GB/s")
            print(f"                  Note: Low MBU in decode is expected - small batch size limits")
            print(f"                        parallelism needed to saturate memory bandwidth.")
    
    # Top 10 by MFU
    print("\n" + "-"*80)
    print("Top 10 GEMMs by MFU:")
    sorted_by_mfu = sorted(gemm_infos, key=lambda g: g.mfu, reverse=True)[:10]
    for i, g in enumerate(sorted_by_mfu):
        roofline_eff = (g.tflops / g.roofline_tflops * 100) if g.roofline_tflops > 0 else 0
        print(f"  {i+1}. M={g.m}, N={g.n}, K={g.k}, {g.dtype}, {g.layer_type}:")
        if g.mbu > 100:
            print(f"      MFU={g.mfu:.2f}%, MBU={g.mbu:.1f}% (INVALID - dimension inference error)")
        else:
            print(f"      MFU={g.mfu:.2f}%, MBU={g.mbu:.1f}%, Roofline Eff={roofline_eff:.1f}%")
        print(f"      Achieved={g.tflops:.1f} TFLOPS, BW={g.achieved_bw_tb_s*1000:.0f} GB/s")
        print(f"      AI={g.arithmetic_intensity:.1f} FLOP/B, {g.duration_us:.2f}us, {g.roofline_bound}-bound")
        print(f"      [Trace: TP{g.tp_rank}, corr={g.correlation_id}, ts={g.timestamp_us:.0f}]")
    
    # Bottom 10 by MFU (excluding very short ops)
    print("\nBottom 10 GEMMs by MFU (duration > 5us):")
    significant_ops = [g for g in gemm_infos if g.duration_us > 5]
    sorted_by_mfu_asc = sorted(significant_ops, key=lambda g: g.mfu)[:10]
    for i, g in enumerate(sorted_by_mfu_asc):
        roofline_eff = (g.tflops / g.roofline_tflops * 100) if g.roofline_tflops > 0 else 0
        print(f"  {i+1}. M={g.m}, N={g.n}, K={g.k}, {g.dtype}, {g.layer_type}:")
        print(f"      MFU={g.mfu:.2f}%, MBU={g.mbu:.1f}%, Roofline Eff={roofline_eff:.1f}%")
        print(f"      Achieved={g.tflops:.1f} TFLOPS, BW={g.achieved_bw_tb_s*1000:.0f} GB/s")
        print(f"      AI={g.arithmetic_intensity:.1f} FLOP/B, {g.duration_us:.2f}us, {g.roofline_bound}-bound")
        print(f"      [Trace: TP{g.tp_rank}, corr={g.correlation_id}, ts={g.timestamp_us:.0f}]")
    
    # Top 10 by MBU (memory bandwidth utilization)
    print("\n" + "-"*80)
    print("Top 10 GEMMs by MBU (Memory Bandwidth Utilization):")
    
    # Check if there are ops with MBU > 100% (indicates dimension inference error)
    high_mbu_ops = [g for g in gemm_infos if g.mbu > 100]
    if high_mbu_ops:
        print("  WARNING: MBU > 100% indicates likely dimension inference error!")
        print("           Physical constraint: weight load time from HBM must be <= kernel time.")
        print("           These kernels lack External ID and dimensions were estimated.")
        print()
    
    sorted_by_mbu = sorted(gemm_infos, key=lambda g: g.mbu, reverse=True)[:10]
    for i, g in enumerate(sorted_by_mbu):
        print(f"  {i+1}. M={g.m}, N={g.n}, K={g.k}, {g.dtype}, {g.layer_type}:")
        if g.mbu > 100:
            # Flag as unreliable; compute weight load time using selected GPU's HBM BW
            weight_time_us = g.weight_bytes / (gpu_specs.memory_bw_tb_s * 1e12) * 1e6
            print(f"      MBU={g.mbu:.1f}% (INVALID - weight load takes {weight_time_us:.1f}µs > kernel {g.duration_us:.1f}µs)")
            print(f"      Likely cause: incorrect dimension inference for CUDA Graph kernel")
        else:
            peak_gb_s = gpu_specs.memory_bw_tb_s * 1000.0
            print(f"      MBU={g.mbu:.1f}%, BW={g.achieved_bw_tb_s*1000:.0f} GB/s (peak: {peak_gb_s:.0f} GB/s)")
        print(f"      MFU={g.mfu:.2f}%, AI={g.arithmetic_intensity:.1f} FLOP/B, {g.roofline_bound}-bound")
        print(f"      [Trace: TP{g.tp_rank}, corr={g.correlation_id}, ts={g.timestamp_us:.0f}]")
    
    # Top 10 by time
    print("\n" + "-"*80)
    print("Top 10 GEMMs by time:")
    sorted_by_time = sorted(gemm_infos, key=lambda g: g.duration_us, reverse=True)[:10]
    for i, g in enumerate(sorted_by_time):
        print(f"  {i+1}. M={g.m}, N={g.n}, K={g.k}, {g.dtype}, {g.layer_type}:")
        print(f"      {g.duration_us:.2f}us, MFU={g.mfu:.2f}%, {g.tflops:.1f} TFLOPS, "
              f"AI={g.arithmetic_intensity:.1f} FLOP/B")
        print(f"      [Trace: TP{g.tp_rank}, corr={g.correlation_id}, ts={g.timestamp_us:.0f}]")
    
    # =========================================================================
    # Grouped GEMM (Fused MoE) Analysis
    # =========================================================================
    if grouped_gemm_infos:
        print("\n" + "="*80)
        print("Grouped GEMM Analysis (Fused MoE)")
        print("="*80)
        print(f"GPU: {gpu_specs.name}")
        print(f"Peak FP8: {gpu_specs.fp8_tflops/1000:.1f} PFLOPS")
        print("-"*80)
        
        # Separate prefill and decode ops
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
        
        # Prefill Analysis
        if prefill_ops:
            print("Prefill Phase (fused_moe_kernel with External ID):")
            pf_flops = sum(g.total_flops for g in prefill_ops)
            pf_time = sum(g.duration_us for g in prefill_ops)
            pf_bytes = sum(g.total_bytes for g in prefill_ops)
            
            # Get representative dimensions
            sample = prefill_ops[0]
            print(f"  Dimensions: {sample.num_tokens} tokens × top_{sample.top_k} experts")
            print(f"              {sample.num_experts} total experts, hidden={sample.hidden_size}")
            print(f"              w1_inter={sample.w1_intermediate}, w2_inter={sample.w2_intermediate}")
            print(f"  Token-expert pairs: {sample.total_token_expert_pairs}")
            print()
            
            avg_mfu = sum(g.mfu * g.duration_us for g in prefill_ops) / pf_time if pf_time > 0 else 0
            avg_mbu = sum(g.mbu * g.duration_us for g in prefill_ops) / pf_time if pf_time > 0 else 0
            avg_bw = sum(g.achieved_bw_tb_s * g.duration_us for g in prefill_ops) / pf_time * 1000 if pf_time > 0 else 0
            avg_tflops = (pf_flops / 1e12) / (pf_time / 1e6) if pf_time > 0 else 0
            
            print(f"  Total time: {pf_time/1000:.2f} ms ({len(prefill_ops)} ops)")
            print(f"  Total FLOPs: {pf_flops/1e12:.2f} TFLOPs")
            print(f"  Achieved: {avg_tflops:.1f} TFLOPS")
            print(f"  MFU: {avg_mfu:.1f}%, MBU: {avg_mbu:.1f}%")
            print(f"  Bandwidth: {avg_bw:.0f} GB/s")
            print(f"  Arithmetic Intensity: {sample.arithmetic_intensity:.1f} FLOPs/byte")
            print(f"  Roofline bound: {sample.roofline_bound}")
            print()
            
            # Top prefill ops by MFU
            print("  Top 5 Prefill MoE ops by MFU:")
            sorted_pf = sorted(prefill_ops, key=lambda g: g.mfu, reverse=True)[:5]
            for i, g in enumerate(sorted_pf):
                print(f"    {i+1}. {g.num_tokens}tok×top{g.top_k}, {g.weight_dtype}:")
                print(f"       MFU={g.mfu:.1f}%, {g.tflops:.1f} TFLOPS, {g.duration_us:.1f}us")
                print(f"       [ExtID={g.external_id}, TP{g.tp_rank}]")
        
        # Decode Analysis
        if decode_ops:
            print()
            print("Decode Phase (fused_moe_kernel, CUDA Graph):")
            dc_flops = sum(g.total_flops for g in decode_ops)
            dc_time = sum(g.duration_us for g in decode_ops)
            dc_bytes = sum(g.total_bytes for g in decode_ops)
            
            # Get representative dimensions (combine w1 and w2)
            w1_ops = [g for g in decode_ops if g.w1_intermediate > 0]
            w2_ops = [g for g in decode_ops if g.w2_intermediate > 0]
            
            if w1_ops:
                sample = w1_ops[0]
                print(f"  Dimensions (inferred): {sample.num_tokens} tokens × top_{sample.top_k} experts")
                print(f"              {sample.num_experts} experts, hidden={sample.hidden_size}")
                # Show w1 and w2 dimensions
                w1_inter = w1_ops[0].w1_intermediate if w1_ops else 0
                w2_inter = w2_ops[0].w2_intermediate if w2_ops else 0
                print(f"              w1_inter={w1_inter} (gate+up), w2_inter={w2_inter} (down)")
                print()
            
            avg_mfu = sum(g.mfu * g.duration_us for g in decode_ops) / dc_time if dc_time > 0 else 0
            avg_mbu = sum(g.mbu * g.duration_us for g in decode_ops) / dc_time if dc_time > 0 else 0
            avg_bw = sum(g.achieved_bw_tb_s * g.duration_us for g in decode_ops) / dc_time * 1000 if dc_time > 0 else 0
            avg_tflops = (dc_flops / 1e12) / (dc_time / 1e6) if dc_time > 0 else 0
            
            total_kernels = sum(g.num_kernels for g in decode_ops)
            print(f"  Total time: {dc_time/1000:.2f} ms/GPU ({total_kernels} kernels across all GPUs)")
            print(f"  Total FLOPs: {dc_flops/1e12:.4f} TFLOPs")
            print(f"  Achieved: {avg_tflops:.1f} TFLOPS")
            print(f"  MFU: {avg_mfu:.1f}%, MBU: {avg_mbu:.1f}%")
            print(f"  Bandwidth: {avg_bw:.0f} GB/s")
            
            # Show breakdown by kernel type (w1 vs w2)
            if w1_ops and w2_ops:
                print()
                print("  By projection type:")
                w1_time = sum(g.duration_us for g in w1_ops)
                w1_flops = sum(g.total_flops for g in w1_ops)
                w1_mfu = sum(g.mfu * g.duration_us for g in w1_ops) / w1_time if w1_time > 0 else 0
                print(f"    W1 (gate+up): {w1_time/1000:.2f}ms, {w1_flops/1e12:.4f} TFLOPs, MFU={w1_mfu:.1f}%")
                
                w2_time = sum(g.duration_us for g in w2_ops)
                w2_flops = sum(g.total_flops for g in w2_ops)
                w2_mfu = sum(g.mfu * g.duration_us for g in w2_ops) / w2_time if w2_time > 0 else 0
                print(f"    W2 (down):    {w2_time/1000:.2f}ms, {w2_flops/1e12:.4f} TFLOPs, MFU={w2_mfu:.1f}%")
        
        print("-"*80)
    
    # Layer type breakdown
    print("\n" + "="*80)
    print("Layer Type Time Breakdown (All Kernels)")
    print("="*80)
    
    total_info = layer_times.get('_total', {})
    total_kernel_time = total_info.get('time_ms', 0)
    num_gpus = total_info.get('num_gpus', 1)
    per_gpu_time = total_info.get('time_ms_per_gpu', total_kernel_time)
    
    print(f"Total kernel time (sum across {num_gpus} GPUs): {total_kernel_time:.2f} ms")
    print(f"Per-GPU average kernel time: {per_gpu_time:.2f} ms\n")
    
    # Print in order of importance (excluding Communication - handled separately)
    layer_order = ['QKVO', 'SDPA', 'FFN', 'Normalization', 'Communication', 'Other']
    for layer_name in layer_order:
        lt = layer_times.get(layer_name, {})
        time_ms = lt.get('time_ms_per_gpu', 0)
        pct = lt.get('percentage', 0)
        count = lt.get('count', 0)
        print(f"  {layer_name:15s}: {time_ms:10.2f} ms/GPU  ({pct:5.1f}%)  [{count:6d} kernels]")
    
    # Communication overlap analysis
    print("\n" + "-"*80)
    print("Communication Overlap Analysis")
    print("-"*80)
    
    if comm_overlap:
        total_comm_us = comm_overlap['total_comm_time_us']
        same_gpu_us = comm_overlap['same_gpu_overlap_us']
        cross_gpu_us = comm_overlap['cross_gpu_overlap_us']
        exposed_us = comm_overlap['exposed_time_us']
        warmup_us = comm_overlap.get('warmup_time_us', 0)
        warmup_count = comm_overlap.get('warmup_count', 0)
        num_gpus = comm_overlap['num_gpus']
        
        # Per-GPU times (divide by num_gpus for per-GPU average)
        total_comm_ms = total_comm_us / 1000 / num_gpus
        same_gpu_ms = same_gpu_us / 1000 / num_gpus
        cross_gpu_ms = cross_gpu_us / 1000 / num_gpus
        exposed_ms = exposed_us / 1000 / num_gpus
        warmup_ms = warmup_us / 1000 / num_gpus
        
        same_gpu_pct = (same_gpu_us / total_comm_us * 100) if total_comm_us > 0 else 0
        cross_gpu_pct = (cross_gpu_us / total_comm_us * 100) if total_comm_us > 0 else 0
        exposed_pct = (exposed_us / total_comm_us * 100) if total_comm_us > 0 else 0
        
        if warmup_count > 0:
            print(f"  Warmup/barrier kernels excluded: {warmup_count} kernels, {warmup_ms:.2f} ms/GPU")
            print()
        
        print(f"  Total communication time (excluding warmup): {total_comm_ms:10.2f} ms/GPU")
        print()
        print(f"  Overlap analysis:")
        print(f"    Same-GPU overlap:           {same_gpu_ms:10.2f} ms/GPU  ({same_gpu_pct:5.1f}%)")
        print(f"      (Compute on same GPU, different stream)")
        print(f"    Cross-GPU pipeline overlap: {cross_gpu_ms:10.2f} ms/GPU  ({cross_gpu_pct:5.1f}%)")
        print(f"      (Compute on other GPUs - TP pipeline effect)")
        print(f"    Exposed (no overlap):       {exposed_ms:10.2f} ms/GPU  ({exposed_pct:5.1f}%)")
        print(f"      (All GPUs blocked on communication)")
        print()
        
        # Breakdown by communication type
        print("  By communication type:")
        by_type = comm_overlap['by_type']
        for comm_type, data in sorted(by_type.items(), key=lambda x: -x[1]['time_us']):
            if data['count'] == 0 and data.get('warmup_count', 0) == 0:
                continue
            
            time_ms = data['time_us'] / 1000 / num_gpus
            cross_overlap_us = data['cross_gpu_overlap_us']
            exposed_type_us = data['no_overlap_us']
            cross_pct = (cross_overlap_us / data['time_us'] * 100) if data['time_us'] > 0 else 0
            exposed_type_pct = (exposed_type_us / data['time_us'] * 100) if data['time_us'] > 0 else 0
            
            warmup_info = ""
            if data.get('warmup_count', 0) > 0:
                warmup_ms = data['warmup_time_us'] / 1000 / num_gpus
                warmup_info = f" (+{data['warmup_count']} warmup, {warmup_ms:.1f}ms)"
            
            print(f"    {comm_type:25s}: {time_ms:8.2f} ms/GPU, {data['count']:5d} calls{warmup_info}")
            if data['count'] > 0:
                print(f"      Pipeline overlap: {cross_pct:5.1f}%, Exposed: {exposed_type_pct:5.1f}%")
    else:
        # Fallback to old Communication stats
        lt = layer_times.get('Communication', {})
        time_ms = lt.get('time_ms_per_gpu', 0)
        pct = lt.get('percentage', 0)
        count = lt.get('count', 0)
        print(f"  Total: {time_ms:10.2f} ms/GPU  ({pct:5.1f}%)  [{count:6d} kernels]")
        print("  (Run with full analysis to see overlap breakdown)")
    
    # Network Roofline Analysis
    if network_roofline:
        print("\n" + "-"*80)
        print("Network Communication Roofline Analysis")
        print("-"*80)
        print(f"  Reference: https://jax-ml.github.io/scaling-book/roofline/")
        print()
        print(f"  Critical Arithmetic Intensity (AI):")
        # Handle both legacy scalar and new per-dtype dict formats
        crit_hbm = network_roofline.get('critical_ai_hbm')
        crit_net = network_roofline.get('critical_ai_network')
        if isinstance(crit_hbm, dict) and isinstance(crit_net, dict):
            for dt in ['fp8', 'bf16', 'fp16']:
                if dt in crit_hbm and dt in crit_net:
                    print(f"    {dt.upper():>4s} HBM Roofline:     {crit_hbm[dt]:8.1f} FLOPs/byte")
                    print(f"    {dt.upper():>4s} Network Roofline: {crit_net[dt]:8.1f} FLOPs/byte")
        else:
            # Legacy behavior
            print(f"    HBM Roofline:     {network_roofline['critical_ai_hbm']:8.1f} FLOPs/byte")
            print(f"    Network Roofline: {network_roofline['critical_ai_network']:8.1f} FLOPs/byte")
        print(f"    (Operations with AI < critical are bounded by that resource)")
        print()
        print(f"  Hardware: NVLink BW = {network_roofline['nvlink_bw_gb_s']:.0f} GB/s, TP = {network_roofline['tp_degree']}")
        print()
        
        for phase_name, phase_data in network_roofline.get('phases', {}).items():
            M = phase_data['M']
            print(f"  {phase_name.capitalize()} Phase (M={M}):")
            print(f"    {'Operation':<25s} {'Parallelism':<15s} {'Network AI':>12s} {'T_compute':>10s} {'T_network':>10s} {'Bound':>12s}")
            print(f"    {'-'*25:<25s} {'-'*15:<15s} {'-'*12:>12s} {'-'*10:>10s} {'-'*10:>10s} {'-'*12:>12s}")
            
            ops = phase_data.get('operations', [])
            # Sort by time
            ops_sorted = sorted(ops, key=lambda x: -x['measured_time_us'])
            
            for op in ops_sorted[:6]:  # Top 6 by time
                op_name = f"N={op['N']},K={op['K']}"
                parallelism = op.get('parallelism', 'unknown')[:12]
                if op['network_ai'] == float('inf'):
                    ai_str = "N/A"
                else:
                    ai_str = f"{op['network_ai']:.0f}"
                t_comp = f"{op['t_compute_us']:.1f}us"
                t_net = f"{op['t_network_us']:.1f}us" if op['t_network_us'] > 0 else "N/A"
                bound = op['bound'][:12]
                # Include dtype hint if available
                dtype_hint = op.get('dtype')
                if dtype_hint:
                    op_name = f"{op_name},{dtype_hint}"
                print(f"    {op_name:<25s} {parallelism:<15s} {ai_str:>12s} {t_comp:>10s} {t_net:>10s} {bound:>12s}")
            
            # Summary
            total_gemm_time = phase_data['total_gemm_time_us']
            row_parallel_ops = [op for op in ops if op.get('parallelism') == 'row-parallel']
            col_parallel_ops = [op for op in ops if op.get('parallelism') == 'column-parallel']
            
            row_time = sum(op['measured_time_us'] for op in row_parallel_ops)
            col_time = sum(op['measured_time_us'] for op in col_parallel_ops)
            
            network_bound = [op for op in row_parallel_ops if 'network' in op['bound']]
            net_bound_time = sum(op['measured_time_us'] for op in network_bound)
            
            print()
            print(f"    Row-parallel (with AllReduce): {len(row_parallel_ops)} ops, {row_time/1000:.2f}ms")
            if row_parallel_ops:
                print(f"      Network-bound: {len(network_bound)} ops ({net_bound_time/1000:.2f}ms)")
            print(f"    Column-parallel (no AllReduce): {len(col_parallel_ops)} ops, {col_time/1000:.2f}ms")
            print()
        
        # AllReduce stats
        ar_stats = network_roofline.get('allreduce_stats', {})
        if ar_stats.get('count', 0) > 0:
            print(f"  AllReduce Statistics (excluding warmup):")
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
    num_gpus = layer_times.get('_total', {}).get('num_gpus', 1)
    
    for layer_name in ['QKVO', 'FFN', 'Other']:
        if layer_name in gemm_by_layer:
            data = gemm_by_layer[layer_name]
            time_ms = data['time_us'] / 1000 / num_gpus  # Per GPU
            pct = (data['time_us'] / gemm_total_time * 100) if gemm_total_time > 0 else 0
            avg_mfu = data['mfu_sum'] / data['time_us'] if data['time_us'] > 0 else 0
            print(f"  {layer_name:10s}: {time_ms:10.2f} ms/GPU  ({pct:5.1f}%)  "
                  f"[{data['count']:5d} kernels]  Avg MFU: {avg_mfu:.1f}%")
    
    # Show truly unmatched kernels (those that couldn't be inferred)
    if events:
        # Build set of analyzed kernel signatures to check against
        analyzed_kernels = set()
        for g in gemm_infos:
            # Use kernel name prefix as signature
            analyzed_kernels.add(g.kernel_name[:50])
        
        unmatched_time_us = 0
        unmatched_count = 0
        unmatched_types = defaultdict(lambda: {'count': 0, 'time_us': 0})
        
        for e in events:
            if e.get('cat') != 'kernel':
                continue
            name = e.get('name', '')
            name_lower = name.lower()
            
            # Only check GEMM-like kernels
            if not any(x in name_lower for x in ['gemm', 'matmul', 'nvjet']):
                continue
            
            # Check if this kernel was NOT analyzed
            # (not in our analyzed set AND has no External ID)
            ext_id = e.get('args', {}).get('External id')
            if ext_id is None and name[:50] not in analyzed_kernels:
                # Double-check: could we have inferred dimensions?
                grid = e.get('args', {}).get('grid', [])
                inferred = infer_cuda_graph_kernel_dims(name, grid)
                if inferred is None:
                    unmatched_time_us += e.get('dur', 0)
                    unmatched_count += 1
                    # Categorize
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
            print(f"\n  Note: {unmatched_count} GEMM kernels ({unmatched_time_ms:.2f} ms/GPU) could not be analyzed:")
            for ktype, data in sorted(unmatched_types.items(), key=lambda x: -x[1]['time_us']):
                print(f"        {ktype}: {data['count']} kernels, {data['time_us']/1000/num_gpus:.2f} ms/GPU")
    
    print("="*80)


def load_trace(input_path: str) -> Dict:
    """Load trace file (supports both .json and .json.gz)"""
    path = Path(input_path)
    
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def save_trace(trace_data: Dict, output_path: str, compress: bool = False):
    """Save trace file"""
    path = Path(output_path)
    
    if compress or path.suffix == '.gz':
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(trace_data, f)
    else:
        with open(path, 'w') as f:
            json.dump(trace_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Add MFU/MBU metrics to PyTorch profiler traces'
    )
    parser.add_argument('input_trace', help='Input trace file (.json or .json.gz)')
    parser.add_argument('output_trace', nargs='?', default=None,
                        help='Output trace file (optional)')
    parser.add_argument('--gpu', default='H200', choices=list(GPU_SPECS.keys()),
                        help='GPU model for peak FLOPS calculation (default: H200)')
    parser.add_argument('--compress', action='store_true',
                        help='Compress output with gzip')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only print summary, do not modify trace')
    parser.add_argument('--batch-size', type=int, default=992,
                        help='Prefill batch size hint for kernels without External ID (default: 992)')
    
    # Model configuration options
    parser.add_argument('--decode-batch-size', type=int, default=64,
                        help='Decode batch size for CUDA Graph kernels (default: 64)')
    parser.add_argument('--tp', '--tp-degree', type=int, default=8, dest='tp_degree',
                        help='Tensor parallelism degree (default: 8)')
    parser.add_argument('--hidden-size', type=int, default=7168,
                        help='Model hidden size (default: 7168 for DeepSeek-R1)')
    parser.add_argument('--expert-intermediate-size', type=int, default=2048,
                        help='Expert intermediate size before TP division (default: 2048)')
    parser.add_argument('--num-experts', type=int, default=256,
                        help='Number of experts for MoE models (default: 256)')
    
    args = parser.parse_args()
    
    # Update global model config from command line args
    global MODEL_CONFIG
    MODEL_CONFIG = {
        'hidden_size': args.hidden_size,
        'num_experts': args.num_experts,
        'expert_intermediate_size': args.expert_intermediate_size,
        'decode_batch_size': args.decode_batch_size,
        'tp_degree': args.tp_degree,
    }
    
    def format_tflops(tflops: float) -> str:
        """Format TFLOPS, showing PFLOPS for values >= 1000"""
        if tflops >= 1000:
            return f"{tflops/1000:.1f} PFLOPS"
        else:
            return f"{tflops:.1f} TFLOPS"
    
    gpu_specs = GPU_SPECS[args.gpu]
    print(f"Using GPU specs: {gpu_specs.name}")
    if gpu_specs.fp4_tflops > 0:
        print(f"  FP4 Peak: {format_tflops(gpu_specs.fp4_tflops)}")
    print(f"  FP8 Peak: {format_tflops(gpu_specs.fp8_tflops)}")
    print(f"  BF16 Peak: {format_tflops(gpu_specs.fp16_tflops)}")
    print(f"  Memory BW: {gpu_specs.memory_bw_tb_s} TB/s")
    print(f"  L2 Cache: {gpu_specs.l2_cache_mb} MB")
    
    print(f"\nModel config:")
    print(f"  TP degree: {MODEL_CONFIG['tp_degree']}")
    print(f"  Decode batch size: {MODEL_CONFIG['decode_batch_size']}")
    print(f"  Hidden size: {MODEL_CONFIG['hidden_size']}")
    print(f"  Expert intermediate (per GPU): {MODEL_CONFIG['expert_intermediate_size'] // MODEL_CONFIG['tp_degree']}")
    
    # Load trace
    print(f"\nLoading trace from {args.input_trace}...")
    trace_data = load_trace(args.input_trace)
    
    events = trace_data.get('traceEvents', [])
    print(f"Loaded {len(events)} events")
    
    # Analyze GEMM operations
    print("\nAnalyzing GEMM operations...")
    gemm_infos = analyze_all_gemm_kernels(events, gpu_specs)
    
    # Analyze grouped GEMM operations (fused MoE)
    print("Analyzing grouped GEMM operations (fused MoE)...")
    grouped_gemm_infos = analyze_grouped_gemm_kernels(events, gpu_specs)
    
    # Analyze layer time breakdown
    print("Analyzing layer time breakdown...")
    layer_times = analyze_layer_time_breakdown(events)
    
    # Analyze communication overlap
    print("Analyzing communication overlap...")
    comm_overlap = analyze_communication_overlap(events)
    
    # Analyze network roofline
    print("Analyzing network communication roofline...")
    num_gpus = layer_times.get('_total', {}).get('num_gpus', 8)
    network_roofline = analyze_network_roofline(events, gemm_infos, gpu_specs, tp_degree=num_gpus)
    
    # Print summary
    print_summary(gemm_infos, layer_times, gpu_specs, comm_overlap, network_roofline, events, grouped_gemm_infos)
    
    if not args.summary_only and args.output_trace:
        # Add MFU to trace
        print("\nAdding MFU/MBU metrics to trace events...")
        trace_data = add_mfu_to_trace(trace_data, gpu_specs)
        
        # Save output
        print(f"\nSaving modified trace to {args.output_trace}...")
        save_trace(trace_data, args.output_trace, args.compress)
        print("Done!")
    elif not args.summary_only and not args.output_trace:
        print("\nNo output file specified. Use --summary-only or provide an output path.")


if __name__ == '__main__':
    main()