import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional
from tabulate import tabulate

# Header constants
MODEL = "Model"
SERVED_MODEL = "Served Model"
HARDWARE = "Hardware"
FRAMEWORK = "Framework"
PRECISION = "Precision"
ISL = "ISL"
OSL = "OSL"
TP = "TP"
EP = "EP"
DP_ATTENTION = "DP Attention"
CONC = "Conc"
TTFT = "TTFT (ms)"
TPOT = "TPOT (ms)"
INTERACTIVITY = "Interactivity (tok/s/user)"
E2EL = "E2EL (s)"
TPUT_PER_GPU = "TPUT per GPU"
OUTPUT_TPUT_PER_GPU = "Output TPUT per GPU"
INPUT_TPUT_PER_GPU = "Input TPUT per GPU"
PREFILL_TP = "Prefill TP"
PREFILL_EP = "Prefill EP"
PREFILL_DP_ATTN = "Prefill DP Attn"
PREFILL_WORKERS = "Prefill Workers"
PREFILL_GPUS = "Prefill GPUs"
DECODE_TP = "Decode TP"
DECODE_EP = "Decode EP"
DECODE_DP_ATTN = "Decode DP Attn"
DECODE_WORKERS = "Decode Workers"
DECODE_GPUS = "Decode GPUs"

# Eval constants
TASK = "Task"
EM_STRICT = "EM Strict"
EM_FLEXIBLE = "EM Flexible"
N_EFF = "N (eff)"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file and return dict, or None on error."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize.py <results_dir>")
        sys.exit(1)

    results = []
    results_dir = Path(sys.argv[1])
    for result_path in results_dir.rglob('*.json'):
        result = load_json(result_path)
        if result and 'is_multinode' in result:
            results.append(result)

    single_node_results = [r for r in results if not r['is_multinode']]
    multinode_results = [r for r in results if r['is_multinode']]

    # Single-node and multi-node results have different fields and therefore need to be printed separately
    if single_node_results:
        single_node_results.sort(key=lambda r: (
            r['infmax_model_prefix'], r['hw'], r['framework'], r['precision'], r['isl'], r['osl'], r['tp'], r['ep'], r['conc']))

        single_node_headers = [
            MODEL, SERVED_MODEL, HARDWARE, FRAMEWORK, PRECISION, ISL, OSL, TP, EP, DP_ATTENTION,
            CONC, TTFT, TPOT, INTERACTIVITY, E2EL, TPUT_PER_GPU, OUTPUT_TPUT_PER_GPU, INPUT_TPUT_PER_GPU
        ]

        single_node_rows = [
            [
                r['infmax_model_prefix'],
                r['model'],
                r['hw'].upper(),
                r['framework'].upper(),
                r['precision'].upper(),
                r['isl'],
                r['osl'],
                r['tp'],
                r['ep'],
                r['dp_attention'],
                r['conc'],
                f"{r['median_ttft'] * 1000:.4f}",
                f"{r['median_tpot'] * 1000:.4f}",
                f"{r['median_intvty']:.4f}",
                f"{r['median_e2el']:.4f}",
                f"{r['tput_per_gpu']:.4f}",
                f"{r['output_tput_per_gpu']:.4f}",
                f"{r['input_tput_per_gpu']:.4f}",
            ]
            for r in single_node_results
        ]

        print("## Single-Node Results\n")
        print(tabulate(single_node_rows, headers=single_node_headers, tablefmt="github"))
        print("\n")

    if multinode_results:
        multinode_results.sort(key=lambda r: (r['infmax_model_prefix'], r['hw'], r['framework'], r['precision'], r['isl'],
                            r['osl'], r['prefill_tp'], r['prefill_ep'], r['decode_tp'], r['decode_ep'], r['conc']))

        multinode_headers = [
            MODEL, SERVED_MODEL, HARDWARE, FRAMEWORK, PRECISION, ISL, OSL,
            PREFILL_TP, PREFILL_EP, PREFILL_DP_ATTN, PREFILL_WORKERS, PREFILL_GPUS,
            DECODE_TP, DECODE_EP, DECODE_DP_ATTN, DECODE_WORKERS, DECODE_GPUS,
            CONC, TTFT, TPOT, INTERACTIVITY, E2EL, TPUT_PER_GPU, OUTPUT_TPUT_PER_GPU, INPUT_TPUT_PER_GPU
        ]

        multinode_rows = [
            [
                r['infmax_model_prefix'],
                r['model'],
                r['hw'].upper(),
                r['framework'].upper(),
                r['precision'].upper(),
                r['isl'],
                r['osl'],
                r['prefill_tp'],
                r['prefill_ep'],
                r['prefill_dp_attention'],
                r['prefill_num_workers'],
                r['num_prefill_gpu'],
                r['decode_tp'],
                r['decode_ep'],
                r['decode_dp_attention'],
                r['decode_num_workers'],
                r['num_decode_gpu'],
                r['conc'],
                f"{r['median_ttft'] * 1000:.4f}",
                f"{r['median_tpot'] * 1000:.4f}",
                f"{r['median_intvty']:.4f}",
                f"{r['median_e2el']:.4f}",
                f"{r['tput_per_gpu']:.4f}",
                f"{r['output_tput_per_gpu']:.4f}",
                f"{r['input_tput_per_gpu']:.4f}",
            ]
            for r in multinode_results
        ]

        print("## Multi-Node Results\n")
        print(tabulate(multinode_rows, headers=multinode_headers, tablefmt="github"))


if __name__ == "__main__":
    main()