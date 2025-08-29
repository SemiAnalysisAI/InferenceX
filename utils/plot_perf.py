import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt


results_dir = Path(sys.argv[1])
exp_name = sys.argv[2]
hw_color = {
    'h100': 'lightgreen',
    'h200': 'green',
    'h200-trt': 'darkgreen',
    'b200': 'black',
    'b200-trt': 'darkblue',
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

    for hw, color in hw_color.items():
        xs = [result['median_e2el'] for result in results if result['hw'] == hw]
        ys = [result['tput_per_gpu'] for result in results if result['hw'] == hw]
        if xs and ys:
            ax.scatter(xs, ys, label=hw.upper(), color=color)

    for result in results:
        x, y = result['median_e2el'], result['tput_per_gpu']
        ax.annotate(str(result['tp']), (x, y), textcoords='offset points', xytext=(3, 3), ha='left', fontsize=8)

    ax.set_xlabel('End-to-end Latency (s)')
    ax.set_ylabel('Throughput per GPU (tok/s)')
    ax.legend(title='GPU Type')
    fig.tight_layout()

    fig.savefig(f'tput_vs_e2el_{exp_name}.png', bbox_inches='tight')
    plt.close(fig)


def plot_tput_vs_intvty():
    fig, ax = plt.subplots()

    for hw, color in hw_color.items():
        xs = [result['median_intvty'] for result in results if result['hw'] == hw]
        ys = [result['tput_per_gpu'] for result in results if result['hw'] == hw]
        if xs and ys:
            ax.scatter(xs, ys, label=hw.upper(), color=color)

    for result in results:
        x, y = result['median_intvty'], result['tput_per_gpu']
        ax.annotate(str(result['tp']), (x, y), textcoords='offset points', xytext=(3, 3), ha='left', fontsize=8)

    ax.set_xlabel('Interactivity (tok/s/user)')
    ax.set_ylabel('Throughput per GPU (tok/s)')
    ax.legend(title='GPU Type')
    fig.tight_layout()

    fig.savefig(f'tput_vs_intvty_{exp_name}.png', bbox_inches='tight')
    plt.close(fig)


plot_tput_vs_e2el()
plot_tput_vs_intvty()
