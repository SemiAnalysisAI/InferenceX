#!/usr/bin/env bash
#SBATCH --job-name=dsv4-verify
#SBATCH --account=amd-aifw-aim
#SBATCH --qos=amd-aifw-aim-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --nodelist=crsuse2-m2m-006
#SBATCH --output=/home/jiacao/InferenceX/dsv4-verify-%j.out
#
# Ground-truth the committed image dsv4-pro-fp4-mi355x:f8d03e77-patched. The build
# job's own verify greps were mis-targeted (looked under fused_moe/ for the
# flydsl_mega_moe enum that actually lives in config/kernel.py). These probes call
# the real code paths -- no GPU needed, all pure-Python config/oracle/env reads.
set -uo pipefail
hostname; date -u

IMG="dsv4-pro-fp4-mi355x:3ee2df30-patched"
docker image inspect "$IMG" >/dev/null 2>&1 \
    || { echo "FATAL: image $IMG not on this node; run on the build node (006)"; exit 1; }
run(){ docker run --rm --entrypoint /bin/bash "$IMG" -c "$1"; }

echo "=== [DEP8] --moe-backend flydsl_mega_moe accepted by the enum? (#51918) ==="
run 'python3 - <<PY
import typing
try:
    from vllm.config.kernel import MoEBackend
    opts=list(typing.get_args(MoEBackend))
except Exception as e:
    print("enum import FAILED:",type(e).__name__,e); raise SystemExit(0)
print("accepted --moe-backend:",opts)
print("VERDICT flydsl_mega_moe accepted:", "flydsl_mega_moe" in opts)
PY'

echo
echo "=== [DEP8] the two new MegaMoE modules landed? (#51918 new files) ==="
run 'V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))");
     for f in models/deepseek_v4/amd/mega_moe_experts.py models/deepseek_v4/amd/mega_moe_runtime.py; do
       [ -f "$V/$f" ] && echo "present  $f" || echo "MISSING  $f"; done'

echo
echo "=== [DEP8] aiter MegaMoE kernels present ==="
run 'V=$(python3 -c "import aiter,os;print(os.path.dirname(aiter.__file__))");
     ls "$V/ops/flydsl/kernels/mega_moe/" 2>/dev/null | head'

echo
echo "=== [TP8] 384-wide MXFP4 shard oracle (#51473, expect 384) ==="
run 'python3 - <<PY
import traceback
try:
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend, mxfp4_round_up_hidden_size_and_intermediate_size as f)
    from vllm.model_executor.layers.fused_moe.config import MoEActivation
    h,i=f(Mxfp4MoeBackend.AITER_MXFP4_BF16,7168,384,activation=MoEActivation.SILU)
    print("intermediate 384 ->",i)
    print("VERDICT:", "384 shard PRESERVED" if i==384 else "ROUNDED to %d"%i)
except Exception:
    import sys
    traceback.print_exc(file=sys.stdout)
    print("VERDICT: PROBE ERROR")
PY'

echo
echo "=== [TP8] FSE env + wiring (base #4269 present; #48728 DROPPED) ==="
run 'python3 -c "import vllm.envs as e; print(\"VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS =\", e.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS)"'
run 'V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))");
     M="$V/models/deepseek_v4/amd/model.py";
     echo "base OLD FSE (_fuse_shared_experts_enabled), expect >0:";
     grep -c "_fuse_shared_experts_enabled" "$M" 2>/dev/null;
     echo "#48728 hetero FSE (_should_fuse_shared_expert), expect 0 (dropped):";
     grep -c "_should_fuse_shared_expert" "$M" 2>/dev/null'

echo
echo "=== [gluon] env + vllm wiring (#51714) ==="
run 'python3 -c "import vllm.envs as e; print(\"VLLM_ROCM_DSV4_SPARSE_GLUON =\", e.VLLM_ROCM_DSV4_SPARSE_GLUON)"'
run 'V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))");
     grep -c "_DSV4_SPARSE_GLUON" "$V/v1/attention/ops/rocm_aiter_mla_sparse.py"'

echo
echo "=== [#4673] aiter gluon buffer_load int64 span fix present? ==="
run 'V=$(python3 -c "import aiter,os;print(os.path.dirname(aiter.__file__))");
     echo "max_addressable_bytes in common_utils.py:";
     grep -c "max_addressable_bytes" "$V/ops/triton/utils/common_utils.py" 2>/dev/null || echo 0;
     echo "max_addressable_bytes in routing pa_decode_sparse.py:";
     grep -c "max_addressable_bytes" "$V/ops/triton/attention/pa_decode_sparse.py" 2>/dev/null || echo 0'

date -u
