# InferenceMAX™ Contribution Roadmap with Implementation Examples

## 📌 Important Note

InferenceMAX™ **already has production-grade CI/CD** with:
- Nightly scheduled GitHub Actions workflows
- Matrix-based execution across hardware/frameworks
- Automated result collection and plotting
- Multi-node support for GB200 NVL72
- Success rate tracking

This roadmap focuses on **enhancing** the existing infrastructure and adding missing capabilities.

---

## 🎯 Immediate Priorities (Week 1-4)

### 1. Add Basic Testing Infrastructure

**Why it's critical:** Prevents regressions and ensures code quality as more contributors join.

**Example Implementation:**

```python
# tests/test_process_result.py
import pytest
import json
from pathlib import Path
from utils.process_result import process_result

@pytest.fixture
def sample_benchmark_result():
    return {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "total_token_throughput": 8000.5,
        "output_throughput": 4000.25,
        "max_concurrency": 128,
        "median_ttft_ms": 50.5,
        "median_tpot_ms": 10.2,
        "median_e2el_ms": 5000.0
    }

def test_process_result_basic(sample_benchmark_result, tmp_path):
    result_file = tmp_path / "test_result.json"
    with open(result_file, 'w') as f:
        json.dump(sample_benchmark_result, f)
    
    # Test processing
    processed = process_result("h100", 8, str(result_file.stem), "vllm", "fp8")
    
    assert processed['hw'] == 'h100'
    assert processed['tp'] == 8
    assert processed['tput_per_gpu'] == pytest.approx(1000.0625)
    assert 'median_ttft' in processed
    assert 'median_intvty' in processed
```

---

### 2. Document Existing CI/CD Architecture

**Why it's critical:** Complex workflow system needs documentation for maintainability and contributor onboarding.

**Example Structure:**

````markdown
# InferenceMAX™ Architecture Documentation

## GitHub Actions Workflow Architecture

### Overview

InferenceMAX uses a **hierarchical reusable workflow system** to run benchmarks across multiple hardware platforms nightly.

```
Scheduler (full-sweep-*-scheduler.yml)
  ├─> Model Template (70b-tmpl.yml, dsr1-tmpl.yml, gptoss-tmpl.yml)
  │    ├─> Benchmark Template (benchmark-tmpl.yml) [Matrix Execution]
  │    │    ├─> Run benchmarks (runners/launch_*.sh)
  │    │    ├─> Process results (utils/process_result.py)
  │    │    └─> Upload artifacts
  │    └─> Collect Results (collect-results.yml)
  │         ├─> Aggregate JSON (utils/collect_results.py)
  │         ├─> Generate plots (utils/plot_perf.py)
  │         └─> Create summary (utils/summarize.py)
  └─> Success Rate Tracking (utils/calc_success_rate.py)
```

### Workflow Types

**1. Schedulers** (`*-scheduler.yml`)
- Run on cron schedule (nightly at 23:00 UTC)
- Coordinate full benchmark sweeps
- Example: `full-sweep-1k1k-scheduler.yml`

**2. Model Templates** (`*-tmpl.yml`)
- Define benchmarks for specific models
- Configure hardware/framework combinations
- Examples: `70b-tmpl.yml` (Llama 70B), `dsr1-tmpl.yml` (DeepSeek-R1)

**3. Benchmark Template** (`benchmark-tmpl.yml`)
- Reusable core workflow
- Matrix execution: TP sizes × concurrency levels
- Handles Docker/Slurm resource cleanup
- Calls launch scripts for each runner

**4. Collection Template** (`collect-results.yml`)
- Aggregates artifacts from benchmark runs
- Generates performance plots
- Creates markdown summaries

### Matrix Execution

Benchmarks run in a matrix across:
- **TP (Tensor Parallel)**: [1, 2, 4, 8]
- **Concurrency**: [4, 8, 16, 32, 64, 128]
- **Hardware**: H100, H200, B200, GB200, MI300X, MI325X, MI355X
- **Frameworks**: vLLM, SGLang, TensorRT-LLM, Dynamo

Example: 4 TP sizes × 5 concurrency levels = 20 parallel jobs per hardware

### Adding New Hardware

1. Create runner configuration in `runners/launch_<hw>.sh`
2. Add hardware to model template (e.g., `70b-tmpl.yml`)
3. Add benchmark job calling `benchmark-tmpl.yml`
4. Configure TP list and concurrency for that hardware
5. Update `utils/calc_success_rate.py` GPU_SKUS list

### Environment Variables

Required for all benchmarks:
- `HF_TOKEN`: HuggingFace token for model access
- `HF_HUB_CACHE`: Cache directory path
- `MODEL`: Model identifier
- `TP`: Tensor parallel size
- `CONC`: Max concurrency
- `ISL`: Input sequence length
- `OSL`: Output sequence length
- `MAX_MODEL_LEN`: Maximum model length
- `RANDOM_RANGE_RATIO`: Input/output length variance
````

---

### 3. Add Input Validation and Error Handling

**Why it's critical:** Makes utilities more robust and user-friendly.

**Example Implementation:**

```python
# utils/process_result.py (enhanced version)
import sys
import json
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_inputs(
    hw: str, 
    tp_size: int, 
    result_filename: str, 
    framework: str, 
    precision: str
) -> None:
    """Validate input parameters."""
    valid_hw = ['h100', 'h200', 'b200', 'gb200', 'mi300x', 'mi325x', 'mi355x']
    valid_frameworks = ['vllm', 'sglang', 'trt-llm']
    valid_precisions = ['fp8', 'fp4', 'fp16', 'int8']
    
    if hw not in valid_hw:
        raise ValueError(f"Invalid hardware: {hw}. Must be one of {valid_hw}")
    
    if tp_size <= 0:
        raise ValueError(f"Invalid tensor parallel size: {tp_size}. Must be positive")
    
    if framework not in valid_frameworks:
        raise ValueError(f"Invalid framework: {framework}. Must be one of {valid_frameworks}")
    
    if precision not in valid_precisions:
        raise ValueError(f"Invalid precision: {precision}. Must be one of {valid_precisions}")
    
    result_path = Path(f'{result_filename}.json')
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")

def process_result(
    hw: str,
    tp_size: int, 
    result_filename: str,
    framework: str,
    precision: str,
    mtp: Optional[str] = None
) -> dict:
    """
    Process benchmark result and calculate per-GPU metrics.
    
    Args:
        hw: Hardware identifier (e.g., 'h100', 'mi300x')
        tp_size: Tensor parallelism size
        result_filename: Path to result JSON file (without extension)
        framework: Inference framework used ('vllm', 'sglang', 'trt-llm')
        precision: Model precision ('fp8', 'fp4', etc.)
        mtp: Optional multi-token prediction parameter
        
    Returns:
        Dictionary with processed metrics
        
    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If result file doesn't exist
        KeyError: If required fields missing from result
    """
    try:
        # Validate inputs
        validate_inputs(hw, tp_size, result_filename, framework, precision)
        
        # Load result file
        result_path = Path(f'{result_filename}.json')
        logger.info(f"Loading result from {result_path}")
        
        with open(result_path) as f:
            bmk_result = json.load(f)
        
        # Validate required fields
        required_fields = ['total_token_throughput', 'output_throughput', 
                         'max_concurrency', 'model_id']
        missing_fields = [f for f in required_fields if f not in bmk_result]
        if missing_fields:
            raise KeyError(f"Missing required fields: {missing_fields}")
        
        # Calculate metrics
        tput_per_gpu = float(bmk_result['total_token_throughput']) / tp_size
        output_tput_per_gpu = float(bmk_result['output_throughput']) / tp_size
        input_tput_per_gpu = tput_per_gpu - output_tput_per_gpu
        
        data = {
            'hw': hw,
            'tp': tp_size,
            'conc': int(bmk_result['max_concurrency']),
            'model': bmk_result['model_id'],
            'framework': framework,
            'precision': precision,
            'tput_per_gpu': round(tput_per_gpu, 4),
            'output_tput_per_gpu': round(output_tput_per_gpu, 4),
            'input_tput_per_gpu': round(input_tput_per_gpu, 4)
        }
        
        if mtp:
            data['mtp'] = mtp
        
        # Process timing metrics
        for key, value in bmk_result.items():
            if key.endswith('ms'):
                data[key.replace('_ms', '')] = round(float(value) / 1000.0, 6)
            if 'tpot' in key:
                # Calculate interactivity (tokens/sec/user)
                intvty_key = key.replace('_ms', '').replace('tpot', 'intvty')
                data[intvty_key] = round(1000.0 / float(value), 4)
        
        logger.info(f"Successfully processed result for {hw} with TP={tp_size}")
        
        # Save processed result
        output_path = Path(f'agg_{result_filename}.json')
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved aggregated result to {output_path}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error processing result: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python process_result.py <hw> <tp_size> <result_filename> <framework> <precision> [mtp]")
        print("\nExample: python process_result.py h100 8 benchmark_001 vllm fp8")
        sys.exit(1)
    
    try:
        hw = sys.argv[1]
        tp_size = int(sys.argv[2])
        result_filename = sys.argv[3]
        framework = sys.argv[4]
        precision = sys.argv[5]
        mtp = sys.argv[6] if len(sys.argv) == 7 else None
        
        data = process_result(hw, tp_size, result_filename, framework, precision, mtp)
        print(json.dumps(data, indent=2))
        
    except Exception as e:
        logger.error(f"Failed to process result: {e}")
        sys.exit(1)
```

---

## 🚀 Short-term Goals (Month 1-2)

### 4. Create CLI Tool

**Why it matters:** Simplifies benchmark execution and result analysis.

**Example Implementation:**

```python
# cli.py
import click
import json
from pathlib import Path
from typing import Optional
import subprocess

@click.group()
@click.version_option(version='0.1.0')
def cli():
    """InferenceMAX CLI - Benchmark LLM Inference Performance"""
    pass

@cli.command()
@click.option('--model', required=True, help='HuggingFace model ID')
@click.option('--hardware', required=True, type=click.Choice(['h100', 'h200', 'b200', 'mi300x', 'mi325x', 'mi355x']))
@click.option('--framework', default='vllm', type=click.Choice(['vllm', 'sglang', 'trt-llm']))
@click.option('--precision', default='fp8', type=click.Choice(['fp8', 'fp4', 'fp16']))
@click.option('--tp', default=1, type=int, help='Tensor parallelism size')
@click.option('--concurrency', default=32, type=int, help='Max concurrency')
@click.option('--input-len', default=128, type=int, help='Input sequence length')
@click.option('--output-len', default=128, type=int, help='Output sequence length')
@click.option('--dry-run', is_flag=True, help='Print command without executing')
def benchmark(model, hardware, framework, precision, tp, concurrency, input_len, output_len, dry_run):
    """Run a benchmark with specified configuration."""
    
    click.echo(f"🚀 Starting benchmark:")
    click.echo(f"  Model: {model}")
    click.echo(f"  Hardware: {hardware.upper()}")
    click.echo(f"  Framework: {framework.upper()}")
    click.echo(f"  Precision: {precision.upper()}")
    click.echo(f"  TP Size: {tp}")
    
    # Construct benchmark script path
    model_shortname = "70b" if "70b" in model.lower() else "dsr1"
    script = f"benchmarks/{model_shortname}_{precision}_{hardware}_docker.sh"
    
    if not Path(script).exists():
        click.echo(f"❌ Benchmark script not found: {script}", err=True)
        return 1
    
    # Set environment variables
    env = {
        'MODEL': model,
        'TP': str(tp),
        'CONC': str(concurrency),
        'ISL': str(input_len),
        'OSL': str(output_len),
        'MAX_MODEL_LEN': str(input_len + output_len),
        'PORT': '8888'
    }
    
    cmd = ['bash', script]
    
    if dry_run:
        click.echo(f"\n📝 Would execute: {' '.join(cmd)}")
        click.echo(f"Environment variables:")
        for k, v in env.items():
            click.echo(f"  {k}={v}")
        return 0
    
    click.echo(f"\n▶️  Executing benchmark...")
    result = subprocess.run(cmd, env={**os.environ, **env})
    
    if result.returncode == 0:
        click.echo(f"✅ Benchmark completed successfully!")
    else:
        click.echo(f"❌ Benchmark failed with code {result.returncode}", err=True)
    
    return result.returncode

@cli.command()
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['table', 'json', 'csv']), default='table')
@click.option('--output', type=click.Path(), help='Output file path')
def report(results_dir, format, output):
    """Generate a report from benchmark results."""
    
    results = []
    for result_file in Path(results_dir).rglob('*.json'):
        with open(result_file) as f:
            results.append(json.load(f))
    
    click.echo(f"📊 Found {len(results)} result files")
    
    if format == 'table':
        from tabulate import tabulate
        headers = ['Hardware', 'Framework', 'TP', 'Throughput/GPU', 'Latency (s)']
        rows = [
            [r['hw'].upper(), r.get('framework', 'vllm').upper(), 
             r['tp'], f"{r['tput_per_gpu']:.2f}", f"{r.get('median_e2el', 0):.4f}"]
            for r in results
        ]
        table = tabulate(rows, headers=headers, tablefmt='github')
        
        if output:
            Path(output).write_text(table)
            click.echo(f"✅ Report saved to {output}")
        else:
            click.echo(table)
    
    elif format == 'json':
        output_data = json.dumps(results, indent=2)
        if output:
            Path(output).write_text(output_data)
            click.echo(f"✅ Report saved to {output}")
        else:
            click.echo(output_data)
    
    elif format == 'csv':
        import csv
        if not output:
            output = 'report.csv'
        
        with open(output, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        click.echo(f"✅ CSV report saved to {output}")

@cli.command()
@click.argument('result_files', nargs=-1, type=click.Path(exists=True))
def compare(result_files):
    """Compare multiple benchmark results."""
    
    if len(result_files) < 2:
        click.echo("❌ Please provide at least 2 result files to compare", err=True)
        return 1
    
    results = []
    for path in result_files:
        with open(path) as f:
            results.append(json.load(f))
    
    click.echo(f"🔍 Comparing {len(results)} results:\n")
    
    baseline = results[0]
    click.echo(f"Baseline: {baseline['hw'].upper()} - {baseline.get('framework', 'vllm').upper()}")
    
    for i, result in enumerate(results[1:], 1):
        hw_name = f"{result['hw'].upper()} - {result.get('framework', 'vllm').upper()}"
        tput_diff = ((result['tput_per_gpu'] - baseline['tput_per_gpu']) / baseline['tput_per_gpu']) * 100
        
        emoji = "📈" if tput_diff > 0 else "📉"
        click.echo(f"{emoji} {hw_name}: {tput_diff:+.2f}% throughput vs baseline")

if __name__ == '__main__':
    cli()
```

**Usage:**
```bash
# Run a benchmark
python cli.py benchmark --model meta-llama/Llama-3.3-70B-Instruct --hardware h100 --tp 8

# Generate report
python cli.py report results/ --format table

# Compare results
python cli.py compare result1.json result2.json result3.json
```

---

### 5. Add GitHub Actions CI/CD

**Why it matters:** Automates testing and validation.

**Example Implementation:**

```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test-python-utils:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov pandas matplotlib
    
    - name: Run tests
      run: |
        pytest tests/ --cov=utils --cov-report=xml --cov-report=term
    
    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml

  lint-shell-scripts:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install shellcheck
      run: sudo apt-get install -y shellcheck
    
    - name: Lint benchmark scripts
      run: |
        find benchmarks/ -name "*.sh" -exec shellcheck {} +
    
    - name: Lint runner scripts
      run: |
        find runners/ -name "*.sh" -exec shellcheck {} +

  validate-json-schema:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install jsonschema
      run: pip install jsonschema
    
    - name: Validate sample results
      run: |
        python scripts/validate_result_schema.py

  check-python-formatting:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install formatting tools
      run: pip install black isort mypy
    
    - name: Check formatting with black
      run: black --check utils/
    
    - name: Check imports with isort
      run: isort --check-only utils/
    
    - name: Type check with mypy
      run: mypy utils/ --ignore-missing-imports
```

---

## 🎨 Medium-term Goals (Month 3-4)

### 6. Interactive Dashboard with Streamlit

**Why it matters:** Makes results accessible and interactive.

**Example Implementation:**

```python
# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

st.set_page_config(page_title="InferenceMAX Dashboard", layout="wide")

st.title("🚀 InferenceMAX™ Performance Dashboard")
st.markdown("Real-time LLM Inference Benchmarking Results")

# Sidebar filters
st.sidebar.header("Filters")

# Load all results
@st.cache_data
def load_results():
    results = []
    for result_file in Path('results/').rglob('*.json'):
        try:
            with open(result_file) as f:
                result = json.load(f)
                results.append(result)
        except:
            continue
    return pd.DataFrame(results)

df = load_results()

if df.empty:
    st.warning("No results found. Please run benchmarks first.")
    st.stop()

# Filters
hardware_options = df['hw'].unique().tolist()
selected_hw = st.sidebar.multiselect("Hardware", hardware_options, default=hardware_options)

framework_options = df['framework'].unique().tolist() if 'framework' in df.columns else ['vllm']
selected_framework = st.sidebar.multiselect("Framework", framework_options, default=framework_options)

precision_options = df['precision'].unique().tolist() if 'precision' in df.columns else ['fp8']
selected_precision = st.sidebar.multiselect("Precision", precision_options, default=precision_options)

# Filter data
filtered_df = df[
    (df['hw'].isin(selected_hw)) &
    (df['framework'].isin(selected_framework) if 'framework' in df.columns else True) &
    (df['precision'].isin(selected_precision) if 'precision' in df.columns else True)
]

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Benchmarks",
        len(filtered_df),
        delta=None
    )

with col2:
    avg_tput = filtered_df['tput_per_gpu'].mean()
    st.metric(
        "Avg Throughput/GPU",
        f"{avg_tput:.2f} tok/s"
    )

with col3:
    max_tput = filtered_df['tput_per_gpu'].max()
    best_hw = filtered_df.loc[filtered_df['tput_per_gpu'].idxmax(), 'hw']
    st.metric(
        "Best Throughput",
        f"{max_tput:.2f} tok/s",
        delta=f"{best_hw.upper()}"
    )

with col4:
    if 'median_e2el' in filtered_df.columns:
        avg_latency = filtered_df['median_e2el'].mean()
        st.metric(
            "Avg Latency",
            f"{avg_latency:.2f} s"
        )

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "📈 Trends", "🔍 Details", "⚖️ Compare"])

with tab1:
    st.subheader("Throughput vs Latency")
    
    if 'median_e2el' in filtered_df.columns and not filtered_df['median_e2el'].isna().all():
        fig = px.scatter(
            filtered_df,
            x='median_e2el',
            y='tput_per_gpu',
            color='hw',
            size='tp',
            hover_data=['model', 'framework', 'precision', 'conc'],
            title="Performance Characteristics",
            labels={
                'median_e2el': 'End-to-End Latency (s)',
                'tput_per_gpu': 'Throughput per GPU (tok/s)',
                'hw': 'Hardware'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Latency data not available in results")
    
    # Hardware comparison
    st.subheader("Hardware Comparison")
    
    hw_comparison = filtered_df.groupby('hw')['tput_per_gpu'].agg(['mean', 'max', 'min']).reset_index()
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=hw_comparison['hw'],
        y=hw_comparison['mean'],
        name='Average',
        text=hw_comparison['mean'].round(2),
        textposition='outside'
    ))
    fig2.update_layout(
        title="Average Throughput by Hardware",
        xaxis_title="Hardware",
        yaxis_title="Throughput per GPU (tok/s)",
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Performance Trends")
    st.info("Time-series analysis coming soon! Add timestamp field to results.")

with tab3:
    st.subheader("Detailed Results")
    
    # Allow sorting
    sort_by = st.selectbox("Sort by", ['tput_per_gpu', 'hw', 'tp', 'model'])
    ascending = st.checkbox("Ascending", value=False)
    
    display_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="benchmark_results.csv",
        mime="text/csv"
    )

with tab4:
    st.subheader("Hardware Comparison")
    
    if len(selected_hw) >= 2:
        comparison_metric = st.selectbox(
            "Metric to compare",
            ['tput_per_gpu', 'median_e2el', 'median_ttft', 'median_tpot']
        )
        
        comparison_df = filtered_df.groupby('hw')[comparison_metric].mean().reset_index()
        
        fig3 = px.bar(
            comparison_df,
            x='hw',
            y=comparison_metric,
            title=f"Average {comparison_metric} by Hardware",
            text=comparison_metric
        )
        fig3.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Relative performance
        st.subheader("Relative Performance")
        baseline_hw = st.selectbox("Select baseline", selected_hw)
        baseline_value = comparison_df[comparison_df['hw'] == baseline_hw][comparison_metric].values[0]
        
        comparison_df['relative'] = ((comparison_df[comparison_metric] - baseline_value) / baseline_value * 100)
        
        fig4 = px.bar(
            comparison_df,
            x='hw',
            y='relative',
            title=f"Performance vs {baseline_hw.upper()} (%)",
            text='relative',
            color='relative',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig4.update_traces(texttemplate='%{text:+.1f}%', textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Select at least 2 hardware types to compare")

# Run with: streamlit run dashboard.py
```

---

## 🔬 Advanced Contributions (Month 5+)

### 7. Add Cost Analysis Module

```python
# utils/cost_analysis.py
from dataclasses import dataclass
from typing import Dict
import json

@dataclass
class GPUPricing:
    """GPU pricing information per hour."""
    name: str
    price_per_hour: float  # USD
    cloud_provider: str
    tdp_watts: int  # Thermal Design Power
    
GPU_PRICING = {
    'h100': [
        GPUPricing('h100', 3.67, 'AWS', 700),
        GPUPricing('h100', 3.18, 'GCP', 700),
        GPUPricing('h100', 2.89, 'OCI', 700),
    ],
    'h200': [
        GPUPricing('h200', 4.5, 'AWS', 700),
        GPUPricing('h200', 3.8, 'OCI', 700),
    ],
    'mi300x': [
        GPUPricing('mi300x', 2.5, 'Azure', 750),
        GPUPricing('mi300x', 2.3, 'OCI', 750),
    ],
    # Add more...
}

def calculate_cost_per_million_tokens(
    throughput_per_gpu: float,
    num_gpus: int,
    gpu_type: str,
    cloud_provider: str = None
) -> Dict:
    """Calculate cost per million tokens."""
    
    # Get pricing
    pricing_options = GPU_PRICING.get(gpu_type, [])
    
    if cloud_provider:
        pricing = next((p for p in pricing_options if p.cloud_provider == cloud_provider), None)
    else:
        # Use cheapest option
        pricing = min(pricing_options, key=lambda x: x.price_per_hour)
    
    if not pricing:
        raise ValueError(f"No pricing found for {gpu_type}")
    
    # Calculate
    tokens_per_hour = throughput_per_gpu * num_gpus * 3600  # tokens/sec * seconds
    cost_per_hour = pricing.price_per_hour * num_gpus
    cost_per_million_tokens = (cost_per_hour / tokens_per_hour) * 1_000_000
    
    # Energy cost (assuming $0.10/kWh)
    energy_kwh = (pricing.tdp_watts * num_gpus) / 1000
    energy_cost_per_hour = energy_kwh * 0.10
    total_cost_per_hour = cost_per_hour + energy_cost_per_hour
    
    return {
        'cloud_provider': pricing.cloud_provider,
        'gpu_price_per_hour': cost_per_hour,
        'energy_cost_per_hour': energy_cost_per_hour,
        'total_cost_per_hour': total_cost_per_hour,
        'cost_per_million_tokens': cost_per_million_tokens,
        'tokens_per_dollar': 1_000_000 / cost_per_million_tokens,
        'carbon_intensity_gCO2_per_token': calculate_carbon(energy_kwh)
    }

def calculate_carbon(energy_kwh: float) -> float:
    """Estimate carbon footprint."""
    # Global average: ~475g CO2/kWh
    return energy_kwh * 475
```

---

## 📋 Implementation Checklist

### Phase 1: Foundation (Weeks 1-4)
- [ ] Add pytest testing framework
- [ ] Create CONTRIBUTING.md
- [ ] Add input validation to all utilities
- [ ] Create issue templates
- [ ] Add PR template
- [ ] Set up pre-commit hooks

### Phase 2: Automation (Weeks 5-8)
- [ ] Implement GitHub Actions CI
- [ ] Add shell script linting
- [ ] Create CLI tool
- [ ] Add result validation
- [ ] Implement Docker Compose for local dev
- [ ] Add automated documentation builds

### Phase 3: Enhancement (Weeks 9-12)
- [ ] Create interactive dashboard
- [ ] Add cost analysis module
- [ ] Implement database backend
- [ ] Add time-series tracking
- [ ] Create comparison tools
- [ ] Add export utilities

### Phase 4: Expansion (Weeks 13-16)
- [ ] Add new hardware support
- [ ] Expand framework coverage
- [ ] Create plugin architecture
- [ ] Add advanced analytics
- [ ] Implement ML-based predictions
- [ ] Create comprehensive docs site

---

## 🤝 Getting Started

1. Pick an item from Phase 1
2. Open an issue describing your plan
3. Fork and create a branch
4. Implement with tests
5. Submit PR with documentation
6. Iterate based on feedback

Remember: **Start small, iterate often!**

