#!/usr/bin/env python3
"""
Command line interface for the MFU Trace Analyzer.

This script wraps the functionality provided by :mod:`mfu_lib` with an
argument parser.  It loads a PyTorch profiler trace, analyses GEMM and
grouped GEMM kernels, communication overlap and network rooflines, prints a
human‑readable summary and optionally writes an enriched trace back to disk.

Usage examples:

    # Basic usage with H200
    python mfu_trace_analyzer.py input_trace.json --summary-only

    # With B200 and custom config
    python mfu_trace_analyzer.py input_trace.json --gpu B200 --tp 8 --decode-batch-size 64

    # Full analysis with output
    python mfu_trace_analyzer.py input_trace.json output_trace.json --gpu H200
"""

import argparse
from typing import Optional

from mfu_lib import (
    GPU_SPECS, Config, load_trace, save_trace,
    analyze_gemm_kernels, analyze_grouped_gemm_kernels,
    analyze_layer_time_breakdown, analyze_communication_overlap,
    analyze_network_roofline, print_summary, add_mfu_to_trace
)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description='Add MFU/MBU metrics to PyTorch profiler traces'
    )
    parser.add_argument('input_trace', help='Input trace file (.json or .json.gz)')
    parser.add_argument('output_trace', nargs='?', default=None, help='Output trace file (optional)')
    parser.add_argument('--gpu', default='H200', choices=list(GPU_SPECS.keys()), help='GPU model for peak FLOPS calculation')
    parser.add_argument('--compress', action='store_true', help='Compress output with gzip')
    parser.add_argument('--summary-only', action='store_true', help='Only print summary, do not modify trace')
    parser.add_argument('--batch-size', type=int, default=992, help='Prefill batch size hint for kernels without External ID (unused)')
    parser.add_argument('--decode-batch-size', type=int, default=64, help='Decode batch size for CUDA Graph kernels (default: 64)')
    parser.add_argument('--tp', '--tp-degree', type=int, default=8, dest='tp_degree', help='Tensor parallelism degree (default: 8)')
    parser.add_argument('--hidden-size', type=int, default=7168, help='Model hidden size')
    parser.add_argument('--expert-intermediate-size', type=int, default=2048, help='Expert intermediate size before TP division')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts for MoE models')
    parser.add_argument('--ep', '--ep-degree', type=int, default=1, dest='ep_degree', help='Expert parallelism degree (default: 1)')
    parser.add_argument('--model-dtype', type=str, default='bf16', dest='model_dtype',
                        help='Default model precision for GEMM outputs when unspecified (e.g. bf16, fp16, fp8)')
    args = parser.parse_args(argv)
    # Build configuration
    config = Config(
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        expert_intermediate_size=args.expert_intermediate_size,
        decode_batch_size=args.decode_batch_size,
        tp_degree=args.tp_degree,
        ep_degree=args.ep_degree,
        model_dtype=args.model_dtype,
    )
    gpu_specs = GPU_SPECS[args.gpu]
    # Print config summary
    def fmt_tflops(tf: float) -> str:
        return f"{tf/1000:.1f} PFLOPS" if tf >= 1000 else f"{tf:.1f} TFLOPS"
    print(f"Using GPU specs: {gpu_specs.name}")
    if gpu_specs.fp4_tflops > 0:
        print(f"  FP4 Peak: {fmt_tflops(gpu_specs.fp4_tflops)}")
    print(f"  FP8 Peak: {fmt_tflops(gpu_specs.fp8_tflops)}")
    print(f"  BF16 Peak: {fmt_tflops(gpu_specs.fp16_tflops)}")
    print(f"  Memory BW: {gpu_specs.memory_bw_tb_s} TB/s")
    print(f"  L2 Cache: {gpu_specs.l2_cache_mb} MB\n")
    print("Model config:")
    print(f"  TP degree: {config.tp_degree}")
    print(f"  EP degree: {config.ep_degree}")
    print(f"  Decode batch size: {config.decode_batch_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Expert intermediate (per GPU): {config.expert_intermediate_per_gpu}\n")
    print(f"  Model dtype: {config.model_dtype}\n")
    # Load trace
    print(f"Loading trace from {args.input_trace}...")
    trace_data = load_trace(args.input_trace)
    events = trace_data.get('traceEvents', [])
    print(f"Loaded {len(events)} events\n")
    # Perform analyses
    print("Analyzing GEMM operations...")
    gemm_infos = analyze_gemm_kernels(events, gpu_specs, config)
    print("Analyzing grouped GEMM operations (fused MoE)...")
    grouped_infos = analyze_grouped_gemm_kernels(events, gpu_specs, config)
    print("Analyzing layer time breakdown...")
    layer_times = analyze_layer_time_breakdown(events)
    print("Analyzing communication overlap...")
    comm_overlap = analyze_communication_overlap(events)
    # Number of GPUs for network roofline (derived from layer_times)
    num_gpus = layer_times.get('_total', {}).get('num_gpus', config.tp_degree)
    print("Analyzing network communication roofline...")
    network_roof = analyze_network_roofline(events, gemm_infos, gpu_specs, tp_degree=num_gpus)
    # Print summary
    print_summary(gemm_infos, layer_times, gpu_specs, comm_overlap, network_roof, events, grouped_infos)
    # Optionally annotate and save trace
    if not args.summary_only:
        if args.output_trace:
            print("\nAdding MFU/MBU metrics to trace events...")
            trace_data = add_mfu_to_trace(trace_data, gpu_specs, config)
            print(f"\nSaving modified trace to {args.output_trace}...")
            save_trace(trace_data, args.output_trace, args.compress)
            print("Done!")
        else:
            print("\nNo output file specified. Use --summary-only or provide an output path.")


if __name__ == '__main__':
    main()