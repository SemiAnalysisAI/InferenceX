# Good First Issues for InferenceMAX™

Welcome to InferenceMAX™! This document contains beginner-friendly contribution ideas that will help you get familiar with the codebase while making meaningful improvements.

---

## 🟢 Super Easy Issues (30 minutes - 2 hours)

### 1. Add Type Hints to Python Utilities

**Difficulty:** ⭐  
**Skills needed:** Python basics  
**Impact:** Code quality and IDE support

**Task:** Add type hints to functions in Python utility files.

**Example - Before:**
```python
def extract_gpu_from_name(job_name):
    job_lower = job_name.lower()
    for gpu in GPU_SKUS:
        if re.search(rf'\b{gpu}(?:-|\b)', job_lower):
            return gpu
```

**After:**
```python
from typing import Optional

def extract_gpu_from_name(job_name: str) -> Optional[str]:
    """Extract GPU name from job name string."""
    job_lower = job_name.lower()
    for gpu in GPU_SKUS:
        if re.search(rf'\b{gpu}(?:-|\b)', job_lower):
            return gpu
    return None
```

**Files to modify:**
- `utils/calc_success_rate.py`
- `utils/collect_results.py`
- `utils/process_result.py`
- `utils/summarize.py`

---

### 2. Add Docstrings to All Functions

**Difficulty:** ⭐  
**Skills needed:** Python documentation  
**Impact:** Better code understanding

**Task:** Add comprehensive docstrings to all Python functions.

**Example:**
```python
def calculate_gpu_success_rates():
    """
    Calculate success rates for each GPU type from GitHub Actions workflow run.
    
    Connects to GitHub API using GITHUB_TOKEN environment variable and analyzes
    job completion status for the current workflow run identified by GITHUB_RUN_ID.
    
    Returns:
        dict: Success rate statistics for each GPU, with keys:
            - n_success (int): Number of successful jobs
            - total (int): Total number of jobs (excluding skipped)
    
    Raises:
        Exception: If GitHub authentication fails or run cannot be found
        
    Environment Variables:
        GITHUB_TOKEN: GitHub personal access token
        GITHUB_RUN_ID: Current workflow run ID
        REPO_NAME: Repository name in format "owner/repo"
    """
    # implementation...
```

**Files to modify:** All Python files in `utils/`

---

### 3. Add Error Messages to Shell Scripts

**Difficulty:** ⭐  
**Skills needed:** Basic bash  
**Impact:** Better debugging experience

**Task:** Add error checking and helpful error messages to benchmark scripts.

**Example - Before:**
```bash
#!/usr/bin/env bash
vllm serve $MODEL --port $PORT
```

**After:**
```bash
#!/usr/bin/env bash

# Check required environment variables
if [ -z "$MODEL" ]; then
    echo "Error: MODEL environment variable not set"
    echo "Usage: MODEL=meta-llama/Llama-3.3-70B-Instruct ./script.sh"
    exit 1
fi

if [ -z "$PORT" ]; then
    echo "Error: PORT environment variable not set"
    exit 1
fi

echo "Starting vLLM server with model: $MODEL on port: $PORT"

vllm serve $MODEL --port $PORT || {
    echo "Error: Failed to start vLLM server"
    echo "Check logs above for details"
    exit 1
}
```

**Files to modify:** Scripts in `benchmarks/` directory

---

### 4. Create requirements.txt File

**Difficulty:** ⭐  
**Skills needed:** Python packaging basics  
**Impact:** Easier setup for contributors

**Task:** Create a comprehensive requirements.txt file listing all Python dependencies.

**Create: `requirements.txt`**
```txt
# Core dependencies
pandas>=2.0.0
matplotlib>=3.7.0
datasets>=2.14.0

# API clients
PyGithub>=2.0.0
requests>=2.31.0

# Testing (development)
pytest>=7.4.0
pytest-cov>=4.1.0
```

**Create: `requirements-dev.txt`**
```txt
-r requirements.txt

# Development tools
black>=23.0.0
isort>=5.12.0
mypy>=1.5.0
pylint>=2.17.0
pre-commit>=3.4.0

# Documentation
sphinx>=7.0.0
sphinx-rtd-theme>=1.3.0
```

---

### 5. Add .gitignore Improvements

**Difficulty:** ⭐  
**Skills needed:** Git basics  
**Impact:** Cleaner repository

**Task:** Expand .gitignore to cover common temporary files.

**Add to `.gitignore`:**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
venv/
env/
ENV/

# Testing
.pytest_cache/
.coverage
htmlcov/
*.cover
.hypothesis/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Results and logs
results/
*.log
*.json.backup
agg_*.json

# Temporary files
*.tmp
.DS_Store
config.yaml

# Docker
.dockerignore
```

---

## 🟡 Easy Issues (2-4 hours)

### 6. Add Input Validation with Better Error Messages

**Difficulty:** ⭐⭐  
**Skills needed:** Python, error handling  
**Impact:** Better user experience

**Task:** Add validation for command-line arguments in Python utilities.

**Example for `utils/process_result.py`:**
```python
import sys
from pathlib import Path

def validate_args():
    """Validate command line arguments with helpful error messages."""
    if len(sys.argv) < 6:
        print("❌ Error: Insufficient arguments")
        print("\nUsage:")
        print("  python process_result.py <hardware> <tp_size> <result_file> <framework> <precision> [mtp]")
        print("\nArguments:")
        print("  hardware    : GPU type (h100, h200, b200, mi300x, mi325x, mi355x)")
        print("  tp_size     : Tensor parallel size (positive integer)")
        print("  result_file : Result JSON filename (without .json extension)")
        print("  framework   : Inference framework (vllm, sglang, trt-llm)")
        print("  precision   : Model precision (fp8, fp4, fp16)")
        print("  mtp         : [Optional] Multi-token prediction parameter")
        print("\nExample:")
        print("  python process_result.py h100 8 benchmark_001 vllm fp8")
        sys.exit(1)
    
    hardware = sys.argv[1]
    valid_hardware = ['h100', 'h200', 'b200', 'gb200', 'mi300x', 'mi325x', 'mi355x']
    if hardware not in valid_hardware:
        print(f"❌ Error: Invalid hardware '{hardware}'")
        print(f"Valid options: {', '.join(valid_hardware)}")
        sys.exit(1)
    
    try:
        tp_size = int(sys.argv[2])
        if tp_size <= 0:
            raise ValueError()
    except ValueError:
        print(f"❌ Error: tp_size must be a positive integer, got '{sys.argv[2]}'")
        sys.exit(1)
    
    result_file = sys.argv[3]
    if not Path(f"{result_file}.json").exists():
        print(f"❌ Error: Result file '{result_file}.json' not found")
        print(f"Current directory: {Path.cwd()}")
        sys.exit(1)
    
    return hardware, tp_size, result_file, sys.argv[4], sys.argv[5], sys.argv[6] if len(sys.argv) > 6 else None

# Use at start of main
if __name__ == "__main__":
    hardware, tp_size, result_file, framework, precision, mtp = validate_args()
    # ... rest of code
```

---

### 7. Create Unit Tests for collect_results.py

**Difficulty:** ⭐⭐  
**Skills needed:** Python, pytest  
**Impact:** Code reliability

**Task:** Write tests for the collect_results utility.

**Create: `tests/test_collect_results.py`**
```python
import pytest
import json
from pathlib import Path
import sys
import subprocess

@pytest.fixture
def sample_results_dir(tmp_path):
    """Create sample result files for testing."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Create sample result files
    result1 = {
        "hw": "h100",
        "tp": 8,
        "tput_per_gpu": 1000.5,
        "model": "test-model"
    }
    
    result2 = {
        "hw": "mi300x",
        "tp": 4,
        "tput_per_gpu": 950.2,
        "model": "test-model"
    }
    
    (results_dir / "result1.json").write_text(json.dumps(result1))
    (results_dir / "result2.json").write_text(json.dumps(result2))
    
    return results_dir

def test_collect_results_basic(sample_results_dir, tmp_path):
    """Test basic collection of results."""
    output_file = tmp_path / "agg_test.json"
    
    # Run collect_results
    result = subprocess.run(
        [sys.executable, "utils/collect_results.py", str(sample_results_dir), "test"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_file.exists(), "Output file not created"
    
    # Verify output
    with open(output_file) as f:
        collected = json.load(f)
    
    assert isinstance(collected, list)
    assert len(collected) == 2
    assert any(r['hw'] == 'h100' for r in collected)
    assert any(r['hw'] == 'mi300x' for r in collected)

def test_collect_results_empty_dir(tmp_path):
    """Test collection from empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    result = subprocess.run(
        [sys.executable, "utils/collect_results.py", str(empty_dir), "test"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    
    output_file = Path("agg_test.json")
    with open(output_file) as f:
        collected = json.load(f)
    
    assert collected == []

def test_collect_results_nested_dirs(tmp_path):
    """Test collection from nested directories."""
    base_dir = tmp_path / "nested"
    sub_dir1 = base_dir / "sub1"
    sub_dir2 = base_dir / "sub2"
    
    sub_dir1.mkdir(parents=True)
    sub_dir2.mkdir(parents=True)
    
    result1 = {"hw": "h100", "tp": 8}
    result2 = {"hw": "h200", "tp": 4}
    
    (sub_dir1 / "result1.json").write_text(json.dumps(result1))
    (sub_dir2 / "result2.json").write_text(json.dumps(result2))
    
    result = subprocess.run(
        [sys.executable, "utils/collect_results.py", str(base_dir), "nested"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    
    with open("agg_nested.json") as f:
        collected = json.load(f)
    
    assert len(collected) == 2
```

---

### 8. Add Logging to Python Scripts

**Difficulty:** ⭐⭐  
**Skills needed:** Python logging  
**Impact:** Better debugging

**Task:** Replace print statements with proper logging.

**Example for any utility file:**
```python
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('inferencemax.log')
    ]
)

logger = logging.getLogger(__name__)

# Replace print statements
# Before: print(f"Processing {count} results")
# After:
logger.info(f"Processing {count} results")

# For errors
# Before: print(f"Error: {e}")
# After:
logger.error(f"Failed to process result: {e}", exc_info=True)

# For debugging
logger.debug(f"Intermediate value: {value}")

# For warnings
logger.warning(f"Using default value for {param}")
```

---

### 9. Create Example Notebook

**Difficulty:** ⭐⭐  
**Skills needed:** Jupyter, Python, data analysis  
**Impact:** Better documentation and examples

**Task:** Create a Jupyter notebook showing how to analyze benchmark results.

**Create: `examples/analyze_results.ipynb`**
```python
# Cell 1
"""
# InferenceMAX Results Analysis Tutorial

This notebook demonstrates how to analyze and visualize InferenceMAX benchmark results.
"""

# Cell 2
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Cell 3
# Load sample results
results = []
for result_file in Path('../results').rglob('*.json'):
    with open(result_file) as f:
        results.append(json.load(f))

df = pd.DataFrame(results)
print(f"Loaded {len(df)} benchmark results")
df.head()

# Cell 4
# Calculate summary statistics
summary = df.groupby('hw')['tput_per_gpu'].agg(['mean', 'std', 'min', 'max'])
print("Throughput Summary by Hardware:")
summary

# Cell 5
# Visualize throughput by hardware
plt.figure(figsize=(12, 6))
df.boxplot(column='tput_per_gpu', by='hw')
plt.title('Throughput Distribution by Hardware')
plt.ylabel('Throughput per GPU (tokens/sec)')
plt.xlabel('Hardware')
plt.suptitle('')  # Remove default title
plt.show()

# Cell 6
# Compare different tensor parallel sizes
tp_comparison = df.groupby(['hw', 'tp'])['tput_per_gpu'].mean().unstack()
tp_comparison.plot(kind='bar', figsize=(12, 6))
plt.title('Throughput by TP Size')
plt.ylabel('Throughput per GPU (tokens/sec)')
plt.xlabel('Hardware')
plt.legend(title='TP Size')
plt.show()

# Cell 7
# Performance vs concurrency
plt.figure(figsize=(12, 6))
for hw in df['hw'].unique():
    hw_data = df[df['hw'] == hw]
    plt.scatter(hw_data['conc'], hw_data['tput_per_gpu'], label=hw.upper(), s=100)

plt.xlabel('Concurrency')
plt.ylabel('Throughput per GPU (tokens/sec)')
plt.title('Throughput vs Concurrency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### 10. Add shellcheck Configuration

**Difficulty:** ⭐⭐  
**Skills needed:** Shell scripting, linting  
**Impact:** Better code quality

**Task:** Add shellcheck configuration and fix common issues.

**Create: `.shellcheckrc`**
```bash
# Disable specific warnings that are acceptable in our scripts

# SC2086: Double quote to prevent globbing and word splitting
# We intentionally use unquoted variables in some contexts
disable=SC2086

# SC1091: Not following sourced files
# We source files that might not exist in all environments
disable=SC1091

# SC2155: Declare and assign separately to avoid masking return values
# Our usage is safe
disable=SC2155
```

**Create: `scripts/lint_shell.sh`**
```bash
#!/bin/bash

echo "🔍 Running shellcheck on all shell scripts..."

find benchmarks/ runners/ -name "*.sh" -type f | while read -r script; do
    echo "Checking: $script"
    shellcheck "$script" || echo "❌ Failed: $script"
done

echo "✅ Shellcheck complete!"
```

---

## 🟠 Medium Issues (4-8 hours)

### 11. Create Configuration File Support

**Difficulty:** ⭐⭐⭐  
**Skills needed:** Python, YAML/JSON  
**Impact:** More flexible configuration

**Task:** Add support for configuration files instead of environment variables.

**Create: `config/default.yaml`**
```yaml
# InferenceMAX Default Configuration

# Hardware settings
hardware:
  default_gpu: h100
  supported:
    - h100
    - h200
    - b200
    - mi300x
    - mi325x
    - mi355x

# Framework settings
frameworks:
  vllm:
    default_port: 8888
    max_model_len: 10240
    gpu_memory_utilization: 0.9
  sglang:
    default_port: 8889
    mem_fraction_static: 0.8
  trt-llm:
    default_port: 8890

# Benchmark settings
benchmark:
  default_concurrency: 32
  default_input_len: 128
  default_output_len: 128
  num_prompts_multiplier: 10
  request_rate: inf

# Result storage
results:
  directory: ./results
  format: json
  aggregate_prefix: agg_

# Logging
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: inferencemax.log
```

**Create: `utils/config.py`**
```python
import yaml
from pathlib import Path
from typing import Any, Dict
import os

class Config:
    """Configuration manager for InferenceMAX."""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        """Load configuration from file."""
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        # Override with environment variable if exists
        env_key = f"INFERENCEMAX_{key.upper().replace('.', '_')}"
        return os.getenv(env_key, value)
    
    def get_hardware_config(self, hw: str) -> Dict:
        """Get hardware-specific configuration."""
        return self.get(f'hardware.{hw}', {})
    
    def get_framework_config(self, framework: str) -> Dict:
        """Get framework-specific configuration."""
        return self.get(f'frameworks.{framework}', {})

# Singleton instance
_config = None

def get_config(config_path: str = "config/default.yaml") -> Config:
    """Get or create config instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
```

---

### 12. Add Result Schema Validation

**Difficulty:** ⭐⭐⭐  
**Skills needed:** Python, JSON Schema  
**Impact:** Data quality assurance

**Task:** Define and validate JSON schema for benchmark results.

**Create: `schemas/result_schema.json`**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "InferenceMAX Benchmark Result",
  "type": "object",
  "required": [
    "model_id",
    "total_token_throughput",
    "output_throughput",
    "max_concurrency"
  ],
  "properties": {
    "model_id": {
      "type": "string",
      "description": "HuggingFace model identifier"
    },
    "total_token_throughput": {
      "type": "number",
      "minimum": 0,
      "description": "Total tokens per second"
    },
    "output_throughput": {
      "type": "number",
      "minimum": 0
    },
    "max_concurrency": {
      "type": "integer",
      "minimum": 1
    },
    "median_ttft_ms": {
      "type": "number",
      "minimum": 0,
      "description": "Time to first token in milliseconds"
    },
    "median_tpot_ms": {
      "type": "number",
      "minimum": 0,
      "description": "Time per output token in milliseconds"
    },
    "median_e2el_ms": {
      "type": "number",
      "minimum": 0,
      "description": "End-to-end latency in milliseconds"
    }
  }
}
```

**Create: `utils/validate_results.py`**
```python
import json
import jsonschema
from pathlib import Path
from typing import List, Tuple

def load_schema() -> dict:
    """Load the result JSON schema."""
    schema_path = Path("schemas/result_schema.json")
    with open(schema_path) as f:
        return json.load(f)

def validate_result(result: dict, schema: dict = None) -> Tuple[bool, List[str]]:
    """
    Validate a result against the schema.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if schema is None:
        schema = load_schema()
    
    try:
        jsonschema.validate(instance=result, schema=schema)
        return True, []
    except jsonschema.ValidationError as e:
        return False, [str(e)]
    except jsonschema.SchemaError as e:
        return False, [f"Schema error: {e}"]

def validate_result_file(result_path: Path) -> Tuple[bool, List[str]]:
    """Validate a result JSON file."""
    try:
        with open(result_path) as f:
            result = json.load(f)
        return validate_result(result)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return False, [f"File not found: {result_path}"]

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_results.py <result_file.json>")
        sys.exit(1)
    
    result_path = Path(sys.argv[1])
    is_valid, errors = validate_result_file(result_path)
    
    if is_valid:
        print(f"✅ {result_path} is valid")
    else:
        print(f"❌ {result_path} is invalid:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
```

---

## 🎓 How to Get Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your contribution
4. **Pick an issue** from above (start with 🟢 if you're new)
5. **Make your changes** with clear commits
6. **Test your changes** locally
7. **Submit a Pull Request** with description
8. **Respond to feedback** from reviewers

---

## 💡 Tips for Success

- **Start small**: Pick one issue to start, don't try to do everything
- **Ask questions**: Open an issue or discussion if you're unsure
- **Read existing code**: Understand the patterns used
- **Test locally**: Make sure your changes work
- **Document changes**: Update docs if needed
- **Be patient**: Reviews may take time

---

## 📚 Resources

- [Git Basics](https://git-scm.com/book/en/v2)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [pytest Documentation](https://docs.pytest.org/)
- [YAML Syntax](https://yaml.org/)
- [JSON Schema](https://json-schema.org/)

---

## ✅ After Your First Contribution

Once you've successfully contributed:
1. You can move to medium difficulty issues
2. Help review other contributors' PRs
3. Suggest new improvements
4. Mentor other first-time contributors

**Welcome to the InferenceMAX community! 🚀**

