import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt


results_dir = Path(sys.argv[1])
exp_name = sys.argv[2]
hw_color = {
    'h100': 'lightgreen',
    'h200': 'darkgreen',
    'b200': 'black',
    'mi300x': 'pink',
    'mi325x': 'red',
    'mi355x': 'purple'
}

results = []
for result_path in results_dir.rglob(f'*.json'):
    with open(result_path) as f:
        result = json.load(f)
    results.append(result)


def plot_tput_vs_e2el():
    fig, ax = plt.subplots()

    # Group by hardware, framework, and precision
    for hw in set(result['hw'] for result in results):
        for framework in set(result.get('framework', 'vLLM') for result in results if result['hw'] == hw):
            for precision in set(result.get('precision', 'fp8') for result in results if result['hw'] == hw and result.get('framework', 'vLLM') == framework):
                xs = [result.get('e2el', result.get('median_e2el', 0)) for result in results if result['hw'] == hw and result.get('framework', 'vLLM') == framework and result.get('precision', 'fp8') == precision]
                ys = [result['tput_per_gpu'] for result in results if result['hw'] == hw and result.get('framework', 'vLLM') == framework and result.get('precision', 'fp8') == precision]
                if xs and ys:
                    # Only add framework label for TRT-LLM, keep vLLM simple
                    if framework == 'TRT-LLM':
                        label = f"{hw.upper()}-TRT-{precision.upper()}"
                    else:
                        label = f"{hw.upper()}-{precision.upper()}"
                    color = hw_color.get(hw.lower(), 'blue')
                    ax.scatter(xs, ys, label=label, color=color, alpha=0.7)

    for result in results:
        x = result.get('e2el', result.get('median_e2el', 0))
        y = result['tput_per_gpu']
        ax.annotate(f"{result['tp']}-{result.get('precision', 'fp8').upper()}", (x, y), textcoords='offset points', xytext=(3, 3), ha='left', fontsize=8)

    ax.set_xlabel('End-to-end Latency (s)')
    ax.set_ylabel('Throughput per GPU (tok/s)')
    ax.legend(title='GPU Type')
    fig.tight_layout()

    fig.savefig(f'tput_vs_e2el_{exp_name}.png', bbox_inches='tight')
    plt.close(fig)


def plot_tput_vs_intvty():
    fig, ax = plt.subplots()

    # Group by hardware, framework, and precision
    for hw in set(result['hw'] for result in results):
        for framework in set(result.get('framework', 'vLLM') for result in results if result['hw'] == hw):
            for precision in set(result.get('precision', 'fp8') for result in results if result['hw'] == hw and result.get('framework', 'vLLM') == framework):
                xs = [result.get('intvty', result.get('median_intvty', 0)) for result in results if result['hw'] == hw and result.get('framework', 'vLLM') == framework and result.get('precision', 'fp8') == precision]
                ys = [result['tput_per_gpu'] for result in results if result['hw'] == hw and result.get('framework', 'vLLM') == framework and result.get('precision', 'fp8') == precision]
                if xs and ys:
                    # Only add framework label for TRT-LLM, keep vLLM simple
                    if framework == 'TRT-LLM':
                        label = f"{hw.upper()}-TRT-{precision.upper()}"
                    else:
                        label = f"{hw.upper()}-{precision.upper()}"
                    color = hw_color.get(hw.lower(), 'blue')
                    ax.scatter(xs, ys, label=label, color=color, alpha=0.7)

    for result in results:
        x = result.get('intvty', result.get('median_intvty', 0))
        y = result['tput_per_gpu']
        ax.annotate(f"{result['tp']}-{result.get('precision', 'fp8').upper()}", (x, y), textcoords='offset points', xytext=(3, 3), ha='left', fontsize=8)

    ax.set_xlabel('Interactivity (tok/s/user)')
    ax.set_ylabel('Throughput per GPU (tok/s)')
    ax.legend(title='GPU Type')
    fig.tight_layout()

    fig.savefig(f'tput_vs_intvty_{exp_name}.png', bbox_inches='tight')
    plt.close(fig)


plot_tput_vs_e2el()
plot_tput_vs_intvty()
