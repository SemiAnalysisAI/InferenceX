#!/usr/bin/env bash
set -euo pipefail

echo "Running temporary DSv4 AITER sparse/indexer kernel tests"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-}"

export PYTHONUNBUFFERED=1

AITER_KERNEL_TEST_REPO=${AITER_KERNEL_TEST_REPO:-https://github.com/Oseltamivir/aiter.git}
AITER_KERNEL_TEST_REF=${AITER_KERNEL_TEST_REF:-dsv4-sparse-indexer-pr}
AITER_KERNEL_TEST_SHA=${AITER_KERNEL_TEST_SHA:-0923d27163ae5b722be27ea980e447fe6c3c7308}
AITER_KERNEL_TEST_DIR=${AITER_KERNEL_TEST_DIR:-/tmp/aiter-dsv4-kernel-test}
export AITER_KERNEL_TEST_REPO AITER_KERNEL_TEST_REF AITER_KERNEL_TEST_SHA AITER_KERNEL_TEST_DIR

rm -rf "$AITER_KERNEL_TEST_DIR"
git clone --filter=blob:none "$AITER_KERNEL_TEST_REPO" "$AITER_KERNEL_TEST_DIR"
(
    cd "$AITER_KERNEL_TEST_DIR"
    git fetch --depth=1 origin "$AITER_KERNEL_TEST_REF"
    git checkout --force "$AITER_KERNEL_TEST_SHA"
    test "$(git rev-parse HEAD)" = "$AITER_KERNEL_TEST_SHA"

    python3 - <<'PYEOF'
import importlib.util
import torch

print(f"torch={torch.__version__}")
print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("FATAL: torch.cuda.is_available() is false")
if importlib.util.find_spec("triton") is None:
    raise SystemExit("FATAL: triton is not importable")
PYEOF

    PYTHONPATH="$AITER_KERNEL_TEST_DIR:${PYTHONPATH:-}" \
        python3 -m pytest -q \
            op_tests/test_sparse_mqa_sink.py \
            op_tests/test_dsv4_indexer.py \
            -s
)

# The benchmark workflow expects a result JSON for non-eval runs. Emit a clearly
# marked placeholder so this temporary kernel-only runner can complete the job.
if [ -n "${RESULT_FILENAME:-}" ]; then
    python3 - <<'PYEOF'
import json
import os

result_filename = os.environ["RESULT_FILENAME"]
payload = {
    "model_id": os.environ.get("MODEL", "aiter-dsv4-kernel-test"),
    "max_concurrency": int(os.environ.get("CONC", "1")),
    "total_token_throughput": 0.0,
    "output_throughput": 0.0,
    "temporary_kernel_test": True,
    "aiter_kernel_test_sha": os.environ.get(
        "AITER_KERNEL_TEST_SHA",
        "0923d27163ae5b722be27ea980e447fe6c3c7308",
    ),
}
with open(f"/workspace/{result_filename}.json", "w") as handle:
    json.dump(payload, handle, indent=2)
print(f"Wrote temporary benchmark placeholder: /workspace/{result_filename}.json")
PYEOF
fi

echo "Temporary DSv4 AITER kernel tests completed"
