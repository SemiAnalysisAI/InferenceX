#!/usr/bin/env bash
# =============================================================================
# apply_k3_container_patches.sh
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
#   docker exec k3-dspark-benchmark bash /opt/k3-recipe/apply_k3_container_patches.sh
#
# Result matches the measured unified-v2 runtime (k3-unified-v2-from-cb810):
#   - aiter rebuilt at pin 55dbc4f47 (#4579 d3ddaabf9 + #4575 22beb1caa)
#   - aiter a16w16 split-K disabled under graph replay (patch_aiter_splitk_cudagraph.py)
#   - bundled tuned K3 GEMM CSV installed + merged -> merged_bf16_tuned_gemm.csv
#   - triton 3.7.0 + triton_kernels + flydsl 0.3.0 (nightly ships triton 3.6.0)
#   - 5 vLLM ASM base patches (decode #50578, fp8 prefill PR-A, PS metadata16,
#     skip-k3-fp8-ps, wvSplitK #50618)
#   - DSpark fp8-asm enablement layer (apply_dspark_fp8asm.sh)
#   - full MLA small-head helper w/ gfx950 Gluon gate (patch_mla_small_head_helper.py)
#   - FlyDSL->torch decode-GEMM reroute (patch_flydsl_decode_to_torch.sh)
#   - vLLM #50183 rejection-sampler NaN argmax guard, #50649 KDA stage gate,
#     #52047 hybrid EAGLE KV-offload group annotation
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
say "1/9 build node-local aiter @ $AITER_PIN (#4579 + #4575 K/V int-32 offsets)"
# Stage from AITER_SRC if provided, else clone. JIT-compiles on demand against
# the container torch + system triton (AITER_USE_SYSTEM_TRITON=1).
export PREBUILD_KERNELS="${PREBUILD_KERNELS:-0}" AITER_USE_SYSTEM_TRITON=1
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
git -C "$LOCAL_AITER" fetch --tags origin "$AITER_PIN" 2>/dev/null \
  || git -C "$LOCAL_AITER" fetch --tags origin 2>/dev/null || true
if [ "$(git -C "$LOCAL_AITER" rev-parse --is-shallow-repository)" = "true" ]; then
  git -C "$LOCAL_AITER" fetch --unshallow --tags origin 2>/dev/null || true
fi
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
# Must run after the checkout above (reset --hard would revert it) and before
# anything JIT-compiles the a16w16 module.
say "2/9 aiter a16w16 split-K graph-safety guard (AITER_ALLOW_SPLITK=1 to re-enable)"
python3 "$PATCHES/patch_aiter_splitk_cudagraph.py"

# ---------------------------------------------------------------------------
say "3/9 install tuned K3 BF16 GEMM CSV"
CONFIGS="$LOCAL_AITER/aiter/configs"
mkdir -p "$CONFIGS/model_configs"
cp "$TUNED_CSV" "$CONFIGS/model_configs/kimik3_bf16_tuned_gemm.csv"
cmp -s "$TUNED_CSV" "$CONFIGS/model_configs/kimik3_bf16_tuned_gemm.csv" \
  || { echo "!! tuned GEMM CSV copy verification failed" >&2; exit 1; }
# LIVE MERGE by default. The measured unified-v2 table (merged_bf16_tuned_gemm
# .v1.csv, 3032 lines) is NOT usable against a JIT-built aiter, despite coming
# from the same aiter pin — installing it kills the server during graph capture:
#
#   [AITER] opus_gemm_arch_gfx950.cuh:156 Kernel id 1212 not found in a16w16
#   fp32 tune lookup table
#   [aiter] opus split-K workspace prewarm on the graph capture stream failed
#
# Two independent reasons:
#   1. solidx/kernel ids in a tuned CSV index the kernel catalog of the build
#      that produced it. v1 came from a container-commit image whose aiter had
#      a different (prebuilt) catalog; here PREBUILD_KERNELS=0 JIT-compiles a
#      subset, so id 1212 does not exist and the lookup throws.
#   2. v1 is missing shapes the bundled kimik3 table provides — e.g. gfx950
#      (N=6288,K=7168) and (N=1536,K=128), both of which the K3 graph capture
#      requests and both of which logged "not found tuned config".
#
# The live merge is generated by aiter's own tooling against THIS build, so its
# indices are valid by construction. K3_MEASURED_GEMM_CSV=1 opts into the
# measured table (only meaningful with a matching prebuilt aiter).
MERGED_DEST="$CONFIGS/merged_bf16_tuned_gemm.csv"
PREMERGED_CSV="$PATCHES/merged_bf16_tuned_gemm.v1.csv"
PREMERGED_SHA=72d56b89d3aa57c29ae01f594929543ab9dae98be2df358fc20ce977a0a82a3e
if [ "${K3_MEASURED_GEMM_CSV:-0}" = "1" ] && [ -f "$PREMERGED_CSV" ]; then
  got=$(sha256sum "$PREMERGED_CSV" | awk '{print $1}')
  [ "$got" = "$PREMERGED_SHA" ] \
    || { echo "!! measured GEMM CSV sha mismatch: $got != $PREMERGED_SHA" >&2; exit 1; }
  cp "$PREMERGED_CSV" "$MERGED_DEST"
  echo "  installed measured unified-v2 merged BF16 GEMM CSV ($(wc -l < "$MERGED_DEST") lines, sha ${PREMERGED_SHA:0:12})"
  echo "  !! K3_MEASURED_GEMM_CSV=1: kernel ids must match this aiter build or graph capture dies" >&2
else
  echo "  live-merging BF16 GEMM CSVs against this aiter build"
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
fi

# ---------------------------------------------------------------------------
if [ "${SKIP_TRITON:-0}" = "1" ]; then
  say "4/9 triton upgrade SKIPPED (SKIP_TRITON=1)"
else
  say "4/9 triton 3.7.0 + triton_kernels + unified-v2 runtime wheels"
  pip install -q --extra-index-url https://pypi.amd.com/triton/release/rocm-7.2.0/simple/ \
    triton==3.7.0 tabulate triton_kernels==1.0.0+amd.rocm7.2.0.git89002410
  # flydsl/hf_transfer/py-spy: the unified-v2 pins. fastsafetensors cuts weight
  # load 745s -> 140s and is what --load-format fastsafetensors needs.
  pip install -q flydsl==0.3.0 hf_transfer==0.1.9 py-spy==0.4.2 fastsafetensors
fi
python3 -c "import triton; print('  triton', triton.__version__)"

# ---------------------------------------------------------------------------
say "5/9 vLLM ASM base patches (decode #50578, fp8 prefill PR-A, PS16, skip-k3-fp8-ps, wvSplitK #50618)"
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
say "6/9 DSpark fp8-asm enablement layer"
bash "$PATCHES/apply_dspark_fp8asm.sh"

# ---------------------------------------------------------------------------
# Upgrades the stub helper the layer above installs into the full unified-v2
# helper, so "gluon" only wins where a gfx950 Gluon decode build exists.
say "7/9 full MLA small-head helper (gfx950 Gluon gate)"
python3 "$PATCHES/patch_mla_small_head_helper.py"

# ---------------------------------------------------------------------------
say "8/9 FlyDSL -> torch decode-GEMM reroute (cudagraph-capturable dense GEMMs)"
CSV="$CONFIGS/merged_bf16_tuned_gemm.csv" bash "$PATCHES/patch_flydsl_decode_to_torch.sh"

# ---------------------------------------------------------------------------
say "9/9 vLLM #50183 NaN argmax guard, #50649 KDA stage gate, #52047 hybrid EAGLE offload"
python3 "$PATCHES/patch_rejection_nan_argmax.py"
python3 "$PATCHES/patch_kda_autotune_stages.py"
python3 "$PATCHES/patch_offload_eagle_hybrid.py"

# ---------------------------------------------------------------------------
say "VERIFY (matches setup_benchmark.sh verify-dspark-patches)"
AITER_MLA="$LOCAL_AITER/aiter/mla.py"
AITER_SPLITK="$LOCAL_AITER/csrc/py_itfs_cu/asm_gemm_a16w16.cu"
KDA="$DIST/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py"
KDA_CHUNK="$DIST/vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py"
REJECT="$DIST/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
OFFLOAD_SCHED="$DIST/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
KV_UTILS="$DIST/vllm/v1/core/kv_cache_utils.py"
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
chk "$MLA"   "def _gluon_mla_decode_supported" 1 "full small-head helper (gfx950 gluon gate)"
chk "$AITER_SPLITK" "PATCH(splitk-cudagraph)" 1 "aiter split-K graph guard"
chk "$AITER_SPLITK" "PATCH(splitk-grid-guard)" 1 "aiter split-K grid guard"
chk "$REJECT" "NaN breaks tl.argmax index bounds" 1 "rejection NaN guard (#50183)"
chk "$KDA_CHUNK" "_RECOMPUTE_W_U_NUM_STAGES"  2 "KDA stage gate (#50649)"
chk "$OFFLOAD_SCHED" "OFFLOAD_EAGLE_PREFIX_VETO" 1 "full-attn eagle prefix veto"
chk "$KV_UTILS" "_annotate_eagle_groups_from_draft_spec" 2 "hybrid EAGLE group annotation (#52047)"
python3 -c "import vllm.v1.attention.backends.mla.rocm_aiter_mla; print('  IMPORT_OK')"

# Package pins + real import checks. verify_unified_image.sh asserts versions
# only, which cannot catch a wheel that installs at the right version but does
# not provide the module vLLM imports (triton_kernels.matmul_ogs is exactly
# that case). Report both. Warn by default — the MoE path here is
# --moe-backend aiter, so a missing triton_kernels is a fallback, not a
# failure. REQUIRE_UNIFIED_PINS=1 makes any mismatch fatal.
python3 - <<'PY'
import os
from importlib.metadata import version, PackageNotFoundError

expected = {
    "amd-aiter": "0.1.19.post3.dev40+g55dbc4f47",
    "flydsl": "0.3.0",
    "hf-transfer": "0.1.9",
    "py-spy": "0.4.2",
    "triton": "3.7.0+amd.rocm7.2.0.git89002410",
    "triton_kernels": "1.0.0+amd.rocm7.2.0.git89002410",
}
bad = []
for pkg, want in expected.items():
    try:
        got = version(pkg)
    except PackageNotFoundError:
        got = "MISSING"
    mark = "OK  " if got == want else "DIFF"
    if got != want:
        bad.append(f"{pkg}: expected {want}, got {got}")
    print(f"  {mark} {pkg} {got}")

for mod in ("triton_kernels.matmul_ogs", "fastsafetensors"):
    try:
        __import__(mod)
        print(f"  OK   import {mod}")
    except Exception as exc:
        bad.append(f"import {mod} failed: {exc}")
        print(f"  DIFF import {mod} FAILED: {exc}")

if bad and os.environ.get("REQUIRE_UNIFIED_PINS") == "1":
    raise SystemExit("!! REQUIRE_UNIFIED_PINS=1 and pins/imports differ:\n  " + "\n  ".join(bad))
if bad:
    print("!! package pins/imports differ from unified-v2 (non-fatal):")
    for b in bad:
        print(f"     {b}")
PY
[ "$ok" = 1 ] || { echo; echo "!! one or more anchors missing — see FAIL lines above" >&2; exit 1; }

echo
echo "DONE — container matches k3-dspark-benchmark. Serve with:"
echo "  export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm"
echo "  NUM_SPEC=2 PORT=8890 GPU_MEM=0.95 MAX_NUM_SEQS=64 MNBT=16384 \\"
echo "    SYNTHETIC_ACCEPT_LEN=2.51 bash <serve script>"
