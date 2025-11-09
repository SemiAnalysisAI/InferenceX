import sys
import base64
from pathlib import Path


def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, 'rb') as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded


def display_graphs(exp_name):
    """Display performance graphs in GitHub Actions summary."""
    print("\n## Performance Graphs\n")
    
    # Find all generated graphs
    current_dir = Path('.')
    
    # Look for tput_vs_intvty graphs
    intvty_graphs = sorted(current_dir.glob(f'tput_vs_intvty_*_{exp_name}.png'))
    e2el_graphs = sorted(current_dir.glob(f'tput_vs_e2el_*_{exp_name}.png'))
    
    # Display interactivity graphs
    if intvty_graphs:
        print("### Throughput vs Interactivity\n")
        for graph in intvty_graphs:
            # Extract model name from filename
            model_name = graph.name.replace(f'tput_vs_intvty_', '').replace(f'_{exp_name}.png', '')
            base64_image = encode_image_to_base64(graph)
            print(f"#### {model_name.upper()}\n")
            print(f"![Throughput vs Interactivity - {model_name}](data:image/png;base64,{base64_image})\n")
    
    # Display end-to-end latency graphs
    if e2el_graphs:
        print("### Throughput vs End-to-End Latency\n")
        for graph in e2el_graphs:
            # Extract model name from filename
            model_name = graph.name.replace(f'tput_vs_e2el_', '').replace(f'_{exp_name}.png', '')
            base64_image = encode_image_to_base64(graph)
            print(f"#### {model_name.upper()}\n")
            print(f"![Throughput vs E2E Latency - {model_name}](data:image/png;base64,{base64_image})\n")
    
    if not intvty_graphs and not e2el_graphs:
        print("*No performance graphs were generated.*\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 display_graphs.py <exp_name>")
        sys.exit(1)
    
    exp_name = sys.argv[1]
    display_graphs(exp_name)
