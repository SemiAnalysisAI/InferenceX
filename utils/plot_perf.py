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
    'mi355x': 'purple'         
}

results = []
for result_path in results_dir.rglob(f'*.json'):
    with open(result_path) as f:
        result = json.load(f)
    # Create combined hw+framework identifier for plotting
    if result.get('framework') == 'trt':
        # Check if hardware name already contains framework suffix
        if result['hw'].endswith('-trt'):
            result['hw_label'] = result['hw']
        else:
            result['hw_label'] = f"{result['hw']}-trt"
    else:
        result['hw_label'] = result['hw']
    results.append(result)


def plot_tput_vs_e2el(precision_filter=None):
    fig, ax = plt.subplots()
    
    # Filter results by precision if specified
    filtered_results = results
    if precision_filter is not None:
        filtered_results = [r for r in results if r.get('precision', 'fp8') == precision_filter]

    for hw_label, color in hw_color.items():
        xs = [result['median_e2el'] for result in filtered_results if result['hw_label'] == hw_label]
        ys = [result['tput_per_gpu'] for result in filtered_results if result['hw_label'] == hw_label]
        if xs and ys:
            ax.scatter(xs, ys, label=hw_label.upper(), color=color)

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
        xs = [result['median_intvty'] for result in filtered_results if result['hw_label'] == hw_label]
        ys = [result['tput_per_gpu'] for result in filtered_results if result['hw_label'] == hw_label]
        if xs and ys:
            ax.scatter(xs, ys, label=hw_label.upper(), color=color)

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
        xs = [result['median_e2el'] for result in model_results if result['hw_label'] == hw_label]
        ys = [result['tput_per_gpu'] for result in model_results if result['hw_label'] == hw_label]
        if xs and ys:
            ax.scatter(xs, ys, label=hw_label.upper(), color=color)

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
        xs = [result['median_intvty'] for result in model_results if result['hw_label'] == hw_label]
        ys = [result['tput_per_gpu'] for result in model_results if result['hw_label'] == hw_label]
        if xs and ys:
            ax.scatter(xs, ys, label=hw_label.upper(), color=color)

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
# Group results by model (70b, dsr1, etc.)
models = set(r.get('model', 'unknown') for r in results)

for model in models:
    # Filter results for this model
    model_results = [r for r in results if r.get('model', 'unknown') == model]
    
    # Create plots for this model
    plot_tput_vs_e2el_for_model(model_results, model)
    plot_tput_vs_intvty_for_model(model_results, model)
