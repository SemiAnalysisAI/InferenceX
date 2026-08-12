#!/usr/bin/env bash
# =============================================================================
# apply_k3_fp4_fp8asm_dspark_patches.sh
#
# Turn the pinned base ROCm vLLM nightly image into the EXACT container the
# Kimi-K3 FP4 MI355X DSpark agentic benchmark runs in. Run this INSIDE a
# container started from:
#
#   vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263
#
# (method borrowed from InferenceX #2508: fetch/build the deltas the base image
# lacks, apply them into the installed dist-packages + a node-local aiter, then
# verify by anchor grep). Self-contained and idempotent — no bind mounts, no
# host paths. Everything it needs ships in ./k3_patches/.
#
#   docker run -d --name k3-dspark-benchmark \
#       --ipc=host --network=host --shm-size=137438953472 \
#       --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
#       --security-opt seccomp=unconfined --security-opt label=disable \
#       --cap-add=SYS_PTRACE -e GPU_ARCHS=gfx950 \
#       --entrypoint sleep \
#       vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263 infinity
#   docker cp benchmarks/single_node/agentic k3-dspark-benchmark:/opt/k3-recipe
#   docker exec k3-dspark-benchmark bash /opt/k3-recipe/apply_k3_fp4_fp8asm_dspark_patches.sh
#
# Result matches `setup_benchmark.sh setup-dspark` from the source tree exactly:
#   - aiter rebuilt at pin 55dbc4f47 (#4579 d3ddaabf9 + #4575 22beb1caa)
#   - bundled tuned K3 GEMM CSV installed + merged -> merged_bf16_tuned_gemm.csv
#   - triton 3.7.0 + tabulate (nightly ships 3.6.0)
#   - 5 vLLM ASM base patches (decode #50578, fp8 prefill PR-A, PS metadata16,
#     skip-k3-fp8-ps, wvSplitK #50618)
#   - DSpark fp8-asm enablement layer (apply_dspark_fp8asm.sh)
#   - FlyDSL->torch decode-GEMM reroute (patch_flydsl_decode_to_torch.sh)
#
# Overridable knobs (env):
#   AITER_PIN     aiter commit to build (default 55dbc4f47...)
#   AITER_SRC     pre-cloned aiter checkout to stage instead of git clone
#   LOCAL_AITER   install location (default /opt/aiter-local; the serve scripts
#                 reference this path for the merged GEMM CSV)
#   SKIP_TRITON=1 skip the triton 3.7.0 upgrade (if the image already has it)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES="$SCRIPT_DIR/k3_patches"

AITER_PIN="${AITER_PIN:-55dbc4f475da26c23cdaf73ce6ed38342a2d7f83}"
AITER_4579="${AITER_4579:-d3ddaabf9}"   # int-32 K offset fix
AITER_4575="${AITER_4575:-22beb1caa}"   # int-32 V offset fix
LOCAL_AITER="${LOCAL_AITER:-/opt/aiter-local}"
AITER_REPO="${AITER_REPO:-https://github.com/ROCm/aiter.git}"
DIST="${DIST:-/usr/local/lib/python3.12/dist-packages}"
MLA="$DIST/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
UTILS="$DIST/vllm/model_executor/layers/utils.py"
TUNED_CSV="$PATCHES/kimik3_bf16_tuned_gemm.csv"

say() { echo; echo "############### $* ###############"; }

[ -d "$PATCHES" ] || { echo "!! missing $PATCHES bundle next to this script" >&2; exit 1; }
[ -f "$TUNED_CSV" ] || { echo "!! missing tuned GEMM CSV $TUNED_CSV" >&2; exit 1; }
[ -f "$MLA" ] || { echo "!! $MLA not found — is this the pinned base image?" >&2; exit 1; }

# ---------------------------------------------------------------------------
say "1/6 build node-local aiter @ $AITER_PIN (#4579 + #4575 K/V int-32 offsets)"
# Stage from AITER_SRC if provided, else clone. JIT-compiles on demand against
# the container torch + system triton (PREBUILD_KERNELS=0, AITER_USE_SYSTEM_TRITON=1).
export PREBUILD_KERNELS=0 AITER_USE_SYSTEM_TRITON=1
if [ -d "$LOCAL_AITER/.git" ]; then
  echo "  reusing existing $LOCAL_AITER checkout"
elif [ -n "${AITER_SRC:-}" ] && [ -d "$AITER_SRC/.git" ]; then
  echo "  staging aiter from $AITER_SRC"
  rm -rf "$LOCAL_AITER"; cp -a "$AITER_SRC" "$LOCAL_AITER"
else
  echo "  cloning $AITER_REPO -> $LOCAL_AITER"
  rm -rf "$LOCAL_AITER"; git clone "$AITER_REPO" "$LOCAL_AITER"
fi
git config --global --add safe.directory "$LOCAL_AITER"
git -C "$LOCAL_AITER" fetch --tags --depth=1 origin "$AITER_PIN" 2>/dev/null \
  || git -C "$LOCAL_AITER" fetch --tags origin 2>/dev/null || true
git -C "$LOCAL_AITER" reset --hard "$AITER_PIN"
git -C "$LOCAL_AITER" submodule update --init 3rdparty/composable_kernel
[ -d "$LOCAL_AITER/3rdparty/composable_kernel/include" ] \
  || { echo "!! composable_kernel submodule not populated" >&2; exit 1; }
git -C "$LOCAL_AITER" merge-base --is-ancestor "$AITER_4579" HEAD \
  || { echo "!! aiter missing #4579 ($AITER_4579) after checkout $AITER_PIN" >&2; exit 1; }
git -C "$LOCAL_AITER" merge-base --is-ancestor "$AITER_4575" HEAD \
  || { echo "!! aiter missing #4575 ($AITER_4575) after checkout $AITER_PIN" >&2; exit 1; }
echo "  aiter HEAD: $(git -C "$LOCAL_AITER" log --oneline -1)"
# Never inherit stale JIT batons/build from a prior tree (blocks rank 0 in RCCL).
rm -rf "$LOCAL_AITER/aiter/jit/build"
find "$LOCAL_AITER/aiter/jit" -maxdepth 1 -name "module_*.so" -delete 2>/dev/null || true
pip uninstall -y aiter amd-aiter >/dev/null 2>&1 || true
( cd "$LOCAL_AITER" && pip install -e . --no-build-isolation --no-deps )
rm -rf /root/aiter; ln -s "$LOCAL_AITER" /root/aiter
python3 -c "import aiter; assert '/opt/aiter-local' in aiter.__file__ or '/root/aiter' in aiter.__file__, aiter.__file__; print('  aiter:', aiter.__file__)"

# ---------------------------------------------------------------------------
say "2/6 install + merge tuned K3 BF16 GEMM CSV"
CONFIGS="$LOCAL_AITER/aiter/configs"
mkdir -p "$CONFIGS/model_configs"
cp "$TUNED_CSV" "$CONFIGS/model_configs/kimik3_bf16_tuned_gemm.csv"
cmp -s "$TUNED_CSV" "$CONFIGS/model_configs/kimik3_bf16_tuned_gemm.csv" \
  || { echo "!! tuned GEMM CSV copy verification failed" >&2; exit 1; }
python3 - "$CONFIGS" <<'PY'
import os, shutil, sys
from pathlib import Path
from aiter.jit.core import AITER_CONFIGS
configs = Path(sys.argv[1])
sources = [configs / "bf16_tuned_gemm.csv"]
sources.extend(
    p for p in sorted((configs / "model_configs").glob("*bf16_tuned_gemm*.csv"))
    if "untuned" not in p.name
)
source_list = os.pathsep.join(str(p) for p in sources if p.is_file())
if not source_list:
    raise SystemExit("ERROR: no BF16 tuned GEMM CSVs found")
try:
    merged = AITER_CONFIGS.update_config_files(source_list, "bf16_tuned_gemm")
except RuntimeError as exc:
    # aiter raises once after resolving cross-file dupes in place; second pass is clean.
    if "Auto-resolved by keeping best performing" not in str(exc):
        raise
    merged = AITER_CONFIGS.update_config_files(source_list, "bf16_tuned_gemm")
dest = configs / "merged_bf16_tuned_gemm.csv"
shutil.copyfile(merged, dest)
print(f"  merged BF16 GEMM CSV -> {dest}")
PY

# ---------------------------------------------------------------------------
if [ "${SKIP_TRITON:-0}" = "1" ]; then
  say "3/6 triton upgrade SKIPPED (SKIP_TRITON=1)"
else
  say "3/6 triton 3.7.0 + tabulate (nightly ships 3.6.0)"
  pip install -q --extra-index-url https://pypi.amd.com/triton/release/rocm-7.2.0/simple/ \
    triton==3.7.0 tabulate
fi
python3 -c "import triton; print('  triton', triton.__version__)"

# ---------------------------------------------------------------------------
say "4/6 vLLM ASM base patches (decode #50578, fp8 prefill PR-A, PS16, skip-k3-fp8-ps, wvSplitK #50618)"
if grep -q "PATCH(fp8-asm)" "$MLA" && grep -q "PATCH(fp8-prefill-pad)" "$MLA" \
   && grep -q "num_head_k = max(16, self.num_heads)" "$MLA" \
   && grep -q "PATCH(skip-k3-fp8-ps)" "$MLA" \
   && grep -q "PATCH(vLLM #50618)" "$UTILS"; then
  echo "  all 5 ASM patches already present"
else
  for p in patch_fp8asm.py patch_fp8_prefill.py patch_ps_metadata16.py patch_skip_k3_fp8_ps.py patch_wvsplitk.py; do
    echo "  applying $p ..."
    python3 "$PATCHES/$p"
  done
fi

# ---------------------------------------------------------------------------
say "5/6 DSpark fp8-asm enablement layer"
bash "$PATCHES/apply_dspark_fp8asm.sh"

# ---------------------------------------------------------------------------
say "6/6 FlyDSL -> torch decode-GEMM reroute (cudagraph-capturable dense GEMMs)"
CSV="$CONFIGS/merged_bf16_tuned_gemm.csv" bash "$PATCHES/patch_flydsl_decode_to_torch.sh"

# ---------------------------------------------------------------------------
say "VERIFY (matches setup_benchmark.sh verify-dspark-patches)"
AITER_MLA="$LOCAL_AITER/aiter/mla.py"
KDA="$DIST/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py"
ok=1
chk() { local n; n=$(grep -c "$2" "$1" 2>/dev/null || echo 0); \
        if [ "$n" -ge "$3" ]; then echo "  OK   $4 ($n)"; else echo "  FAIL $4 ($n < $3)"; ok=0; fi; }
chk "$MLA"   "PATCH(fp8-asm)"                 1 "decode pad-to-16 (#50578)"
chk "$MLA"   "PATCH(fp8-prefill-pad)"         1 "fp8 prefill pad (PR-A)"
chk "$MLA"   "num_head_k = max(16, self.num_heads)" 1 "PS metadata16 (PR-A)"
chk "$MLA"   "PATCH(skip-k3-fp8-ps)"          1 "skip K3 fp8 PS"
chk "$UTILS" "PATCH(vLLM #50618)"             1 "wvSplitK (#50618)"
chk "$MLA"   "_mtp_decode_qlen"               1 "DSpark _mtp_decode_qlen"
chk "$MLA"   'method == "dspark"'             1 "dspark verify qlen branch"
chk "$MLA"   "uses_asm_decode"                2 "persistent-metadata gate"
chk "$AITER_MLA" "80: 64"                     1 "aiter get_block_n_fp8 key 80"
chk "$AITER_MLA" "get_block_n_fp8.get("       1 "aiter get_block_n_fp8.get()"
chk "$KDA"   "stride_indices_seq"             5 "KDA PR#27 stride fix"
python3 -c "import vllm.v1.attention.backends.mla.rocm_aiter_mla; print('  IMPORT_OK')"
[ "$ok" = 1 ] || { echo; echo "!! one or more anchors missing — see FAIL lines above" >&2; exit 1; }

echo
echo "DONE — container matches k3-dspark-benchmark. Serve with:"
echo "  export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm"
echo "  NUM_SPEC=2 PORT=8890 GPU_MEM=0.95 MAX_NUM_SEQS=64 MNBT=16384 \\"
echo "    SYNTHETIC_ACCEPT_LEN=2.51 bash <serve script>"
