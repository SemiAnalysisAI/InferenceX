#!/usr/bin/env bash
set -eo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../../benchmarks/benchmark_lib.sh" --validation-only
check_env_vars SRT_ROOT SRT_STATUS_ENDPOINT SRTCTL_STATUS_TOKEN SRT_TEST_OUTPUT SRT_TEST_SHA

[[ "$(git -C "$SRT_ROOT" rev-parse HEAD)" == "$SRT_TEST_SHA" ]]
cd "$SRT_ROOT"
sha256sum --check streaming-binaries.sha256
chmod +x bin/tachometer-scraper configs/process-exporter
export PATH="$SRT_ROOT/bin:$PATH"
# Keep uv and its compute environment under the run's isolated shared checkout.
cp "$(command -v uv)" bin/uv
export UV_PROJECT_ENVIRONMENT="$SRT_ROOT/.venv-compute"
uv sync --python 3.12 --no-dev
export SRTSLURM_CONFIG="$SRT_ROOT/srtslurm.yaml"
uv run --no-sync python - <<'PY'
import os
from pathlib import Path
import yaml
config = {
    'cluster': 'b300-dsxe',
    'default_account': 'benchmark',
    'default_partition': 'batch_1',
    'default_time_limit': '00:10:00',
    'gpus_per_node': 8,
    'network_interface': '',
    'srtctl_root': os.environ['SRT_ROOT'],
    'model_paths': {'services-only': os.environ['SRT_ROOT']},
    'use_gpus_per_node_directive': False,
    'use_exclusive_sbatch_directive': True,
    'default_sbatch_directives': {'exclude': 'dsxe-sa-b300-prd0-gpu-16', 'cpus-per-task': '8'},
    'reporting': {'status': {
        'endpoint': os.environ['SRT_STATUS_ENDPOINT'],
        'token_env': 'SRTCTL_STATUS_TOKEN',
        'logging-stream-interval': 5,
    }},
}
Path(os.environ['SRTSLURM_CONFIG']).write_text(yaml.safe_dump(config))
PY
mkdir -p "$SRT_TEST_OUTPUT"
uv run --no-sync srtctl apply -f "$script_dir/recipe.yaml" -o "$SRT_TEST_OUTPUT" --json > "$SRT_TEST_OUTPUT/submission.json"
job_id=$(uv run --no-sync python - "$SRT_TEST_OUTPUT/submission.json" <<'PY'
import json
import sys
from pathlib import Path
records = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.startswith('{')]
print(records[-1]['slurm_job_id'])
PY
)
[[ "$job_id" =~ ^[0-9]+$ ]]
printf '%s\n' "$job_id" > "$SRT_TEST_OUTPUT/job-id.txt"
echo "SRT streaming job: b300-dsxe:$job_id"
trap 'scancel "$job_id" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
while [[ -n "$(squeue --noheader --jobs "$job_id" --format '%i')" ]]; do
    sleep 5
done
state=$(sacct --noheader --jobs "$job_id" --format State --parsable2 | head -1)
[[ "$state" == COMPLETED ]]
uv run --no-sync python - "$SRT_TEST_OUTPUT/$job_id/logs" <<'PY'
import sys
from pathlib import Path
import pyarrow.parquet as pq
logs = Path(sys.argv[1])
texts = [p.read_text(errors='replace') for p in logs.rglob('*') if p.suffix in {'.log', '.out', '.err'}]
assert any('SRT_STREAM_STDOUT 00 ✓' in text and 'SRT_STREAM_STDERR 17 ✓' in text and 'SRT_STREAM_COMPLETE' in text for text in texts)
capture = logs / 'tachometer/local/final.parquet'
rows = pq.read_table(capture)
assert rows.num_rows > 0, 'Tachometer captured no real process metrics'
print(f'Local evidence: complete stdout/stderr; {rows.num_rows} Tachometer rows')
PY
