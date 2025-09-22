import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt


results_dir = Path(sys.argv[1])
exp_name = sys.argv[2]
hw_color = {
    'h100': 'lightgreen',
    'h200': 'green',           # H200 VLLM
    'h200-trt': 'darkgreen',   # H200 TRT-LLM
    'b200': 'black',            # B200 VLLM
    'b200-trt': 'gray',      # B200 TRT-LLM
    'mi300x': 'pink',
    'mi325x': 'red',
    'mi355x': 'purple',
    'gb200': 'orange',          # GB200 TRT-LLM and SGlang
}

results = []
for result_path in results_dir.rglob(f'*.json'):
    with open(result_path) as f:
        result = json.load(f)
    results.append(result)


def plot_tput_vs_e2el(precision_filter=None):
    fig, ax = plt.subplots()
    
    # Filter results by precision if specified
    filtered_results = results
    if precision_filter is not None:
        filtered_results = [r for r in results if r.get('precision', 'fp8') == precision_filter]

    for hw_label, color in hw_color.items():
        # Separate fp8 and fp4 results for this hardware
        fp8_results = [r for r in filtered_results if r['hw'] == hw_label and r.get('precision', 'fp8') == 'fp8']
        fp4_results = [r for r in filtered_results if r['hw'] == hw_label and r.get('precision', 'fp8') == 'fp4']
        
        # Plot fp8 results with circles
        if fp8_results:
            xs_fp8 = [r['median_e2el'] for r in fp8_results]
            ys_fp8 = [r['tput_per_gpu'] for r in fp8_results]
            ax.scatter(xs_fp8, ys_fp8, label=f"{hw_label.upper()} (fp8)", color=color, marker='o', s=60)
        
        # Plot fp4 results with squares
        if fp4_results:
            xs_fp4 = [r['median_e2el'] for r in fp4_results]
            ys_fp4 = [r['tput_per_gpu'] for r in fp4_results]
            ax.scatter(xs_fp4, ys_fp4, label=f"{hw_label.upper()} (fp4)", color=color, marker='s', s=60)

    for result in filtered_results:
        x, y = result['median_e2el'], result['tput_per_gpu']
        ax.annotate(str(result['tp']), (x, y), textcoords='offset points', xytext=(3, 3), ha='left', fontsize=8)

    ax.set_xlabel('End-to-end Latency (s)')
    ax.set_ylabel('Throughput per GPU (tok/s)')
    ax.legend(title='GPU Type')
    fig.tight_layout()

    precision_suffix = f"_{precision_filter}" if precision_filter else ""
    fig.savefig(f'tput_vs_e2el_{exp_name}{precision_suffix}.png', bbox_inches='tight')
    plt.close(fig)


def plot_tput_vs_intvty(precision_filter=None):
    fig, ax = plt.subplots()
    
    # Filter results by precision if specified
    filtered_results = results
    if precision_filter is not None:
        filtered_results = [r for r in results if r.get('precision', 'fp8') == precision_filter]

    for hw_label, color in hw_color.items():
        # Separate fp8 and fp4 results for this hardware
        fp8_results = [r for r in filtered_results if r['hw'] == hw_label and r.get('precision', 'fp8') == 'fp8']
        fp4_results = [r for r in filtered_results if r['hw'] == hw_label and r.get('precision', 'fp8') == 'fp4']
        
        # Plot fp8 results with circles
        if fp8_results:
            xs_fp8 = [r['median_intvty'] for r in fp8_results]
            ys_fp8 = [r['tput_per_gpu'] for r in fp8_results]
            ax.scatter(xs_fp8, ys_fp8, label=f"{hw_label.upper()} (fp8)", color=color, marker='o', s=60)
        
        # Plot fp4 results with squares
        if fp4_results:
            xs_fp4 = [r['median_intvty'] for r in fp4_results]
            ys_fp4 = [r['tput_per_gpu'] for r in fp4_results]
            ax.scatter(xs_fp4, ys_fp4, label=f"{hw_label.upper()} (fp4)", color=color, marker='s', s=60)

    for result in filtered_results:
        x, y = result['median_intvty'], result['tput_per_gpu']
        ax.annotate(str(result['tp']), (x, y), textcoords='offset points', xytext=(3, 3), ha='left', fontsize=8)

    ax.set_xlabel('Interactivity (tok/s/user)')
    ax.set_ylabel('Throughput per GPU (tok/s)')
    ax.legend(title='GPU Type')
    fig.tight_layout()

    precision_suffix = f"_{precision_filter}" if precision_filter else ""
    fig.savefig(f'tput_vs_intvty_{exp_name}{precision_suffix}.png', bbox_inches='tight')
    plt.close(fig)


def plot_tput_vs_e2el_for_model(model_results, model_name):
    fig, ax = plt.subplots()
    
    for hw_label, color in hw_color.items():
        # Separate fp8 and fp4 results for this hardware
        fp8_results = [r for r in model_results if r['hw'] == hw_label and r.get('precision', 'fp8') == 'fp8']
        fp4_results = [r for r in model_results if r['hw'] == hw_label and r.get('precision', 'fp8') == 'fp4']
        
        # Plot fp8 results with circles
        if fp8_results:
            xs_fp8 = [r['median_e2el'] for r in fp8_results]
            ys_fp8 = [r['tput_per_gpu'] for r in fp8_results]
            ax.scatter(xs_fp8, ys_fp8, label=f"{hw_label.upper()} (fp8)", color=color, marker='o', s=60)
        
        # Plot fp4 results with squares
        if fp4_results:
            xs_fp4 = [r['median_e2el'] for r in fp4_results]
            ys_fp4 = [r['tput_per_gpu'] for r in fp4_results]
            ax.scatter(xs_fp4, ys_fp4, label=f"{hw_label.upper()} (fp4)", color=color, marker='s', s=60)

    for result in model_results:
        x, y = result['median_e2el'], result['tput_per_gpu']
        ax.annotate(str(result['tp']), (x, y), textcoords='offset points', xytext=(3, 3), ha='left', fontsize=8)

    ax.set_xlabel('End-to-end Latency (s)')
    ax.set_ylabel('Throughput per GPU (tok/s)')
    ax.legend(title='Hardware + Framework')
    ax.set_title(f'{model_name} - All Frameworks')
    fig.tight_layout()

    # Extract model identifier from model name
    model_id = model_name.split('/')[-1].split('-')[0] if '/' in model_name else model_name
    fig.savefig(f'tput_vs_e2el_{model_id}_{exp_name}.png', bbox_inches='tight')
    plt.close(fig)


def plot_tput_vs_intvty_for_model(model_results, model_name):
    fig, ax = plt.subplots()
    
    for hw_label, color in hw_color.items():
        # Separate fp8 and fp4 results for this hardware
        fp8_results = [r for r in model_results if r['hw'] == hw_label and r.get('precision', 'fp8') == 'fp8']
        fp4_results = [r for r in model_results if r['hw'] == hw_label and r.get('precision', 'fp8') == 'fp4']
        
        # Plot fp8 results with circles
        if fp8_results:
            xs_fp8 = [r['median_intvty'] for r in fp8_results]
            ys_fp8 = [r['tput_per_gpu'] for r in fp8_results]
            ax.scatter(xs_fp8, ys_fp8, label=f"{hw_label.upper()} (fp8)", color=color, marker='o', s=60)
        
        # Plot fp4 results with squares
        if fp4_results:
            xs_fp4 = [r['median_intvty'] for r in fp4_results]
            ys_fp4 = [r['tput_per_gpu'] for r in fp4_results]
            ax.scatter(xs_fp4, ys_fp4, label=f"{hw_label.upper()} (fp4)", color=color, marker='s', s=60)

    for result in model_results:
        x, y = result['median_intvty'], result['tput_per_gpu']
        ax.annotate(str(result['tp']), (x, y), textcoords='offset points', xytext=(3, 3), ha='left', fontsize=8)

    ax.set_xlabel('Interactivity (tok/s/user)')
    ax.set_ylabel('Throughput per GPU (tok/s)')
    ax.legend(title='Hardware + Framework')
    ax.set_title(f'{model_name} - All Frameworks')
    fig.tight_layout()

    # Extract model identifier from model name
    model_id = model_name.split('/')[-1].split('-')[0] if '/' in model_name else model_name
    fig.savefig(f'tput_vs_intvty_{model_id}_{exp_name}.png', bbox_inches='tight')
    plt.close(fig)


# Create one plot per model showing all frameworks and hardware
# Group results by model family (70b, dsr1, etc.) instead of full model name
def get_model_family(model_name):
    if '70b' in model_name.lower() or 'llama-3.3-70b' in model_name.lower():
        return '70b'
    elif 'dsr1' in model_name.lower() or 'deepseek-r1' in model_name.lower():
        return 'dsr1'
    else:
        # Fallback to first part of model name
        return model_name.split('/')[-1].split('-')[0] if '/' in model_name else model_name

model_families = set(get_model_family(r.get('model', 'unknown')) for r in results)

for model_family in model_families:
    # Filter results for this model family
    model_results = [r for r in results if get_model_family(r.get('model', 'unknown')) == model_family]
    
    # Create plots for this model family
    plot_tput_vs_e2el_for_model(model_results, model_family)
    plot_tput_vs_intvty_for_model(model_results, model_family)
