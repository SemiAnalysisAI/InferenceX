#!/usr/bin/env bash
# Apply vLLM #50514 (open) — speculative decoding under pipeline parallel.
# Drafter loads on the last PP stage only (draft PP=1); aux hidden states
# forward across PP stages. Vendored for Kimi-K3 TP×PP2 + DSpark on the
# pinned ROCm nightly (cb8104839), which predates the PR.
#
# Usage (inside container, ideally AFTER apply_k3_container_patches.sh):
#   bash /workspace/experimental/kimik3-v4/apply_vllm_50514_pp_spec.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${PATCH_DIR:-${SCRIPT_DIR}/vllm_pr50514_commits}"
MARKER="Drafter runs on the last PP stage only"

ROOT="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"
VLLM_DIR="$ROOT/vllm"
TARGET="$VLLM_DIR/config/speculative.py"
[[ -d "$VLLM_DIR" ]] || { echo "ERROR: no vllm under ROOT=$ROOT" >&2; exit 1; }

if grep -qF "$MARKER" "$TARGET" 2>/dev/null; then
  echo "[50514] already applied (marker in speculative.py); skip"
  # Still ensure AMD target opts into aux-over-PP.
else
  NEED_APPLY=1
fi

filter_tests() {
  local src="$1" dst="$2"
  python3 - "$src" "$dst" <<'PY'
import sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text(errors="replace")
chunks = text.split("diff --git ")
out = []
for i, chunk in enumerate(chunks):
    if i == 0:
        if chunk.strip():
            out.append(chunk if chunk.endswith("\n") else chunk + "\n")
        continue
    first = chunk.split("\n", 1)[0]
    path = first.split(" ")[0].removeprefix("a/")
    if path.startswith("tests/"):
        continue
    out.append("diff --git " + (chunk if chunk.endswith("\n") else chunk + "\n"))
dst.write_text("".join(out))
PY
}

surgical_draft_pp1() {
  python3 - "$TARGET" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
t = p.read_text()
old = "pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,"
new = (
    "# Drafter runs on the last PP stage only; do not inherit target PP.\n"
    "            pipeline_parallel_size=1,"
)
if "Drafter runs on the last PP stage only" in t:
    print("surgical: already marked")
    return
if old not in t:
    raise SystemExit(f"surgical: expected line missing in {p}")
# Only replace inside create_draft_parallel_config
idx = t.find("def create_draft_parallel_config")
if idx < 0:
    raise SystemExit("surgical: create_draft_parallel_config not found")
# find the pipeline_parallel_size= line after that def
sub = t[idx:]
pos = sub.find(old)
if pos < 0:
    raise SystemExit("surgical: PP inherit line not found in create_draft_parallel_config")
# Prefer comment+PP=1 form from #50514
sub2 = sub[:pos] + new + sub[pos + len(old):]
t2 = t[:idx] + sub2
p.write_text(t2)
print(f"surgical: set draft pipeline_parallel_size=1 in {p}")
PY
}

opt_in_amd_aux() {
  local AMD_MODEL="$VLLM_DIR/models/kimi_k3/amd/model.py"
  [[ -f "$AMD_MODEL" ]] || return 0
  if grep -qF "supports_aux_hidden_states_over_pp" "$AMD_MODEL"; then
    echo "[50514] amd/model.py already has supports_aux_hidden_states_over_pp"
  else
    python3 - "$AMD_MODEL" <<'PY'
from pathlib import Path
import re, sys
p = Path(sys.argv[1])
t = p.read_text()
m = re.search(r"class KimiK3ForConditionalGeneration\([^)]*\):", t)
if not m:
    print("WARN: KimiK3ForConditionalGeneration not found in amd/model.py", file=sys.stderr)
    sys.exit(0)
flag = "\n    supports_aux_hidden_states_over_pp = True\n"
t = t[: m.end()] + flag + t[m.end() :]
p.write_text(t)
print(f"patched {p}: supports_aux_hidden_states_over_pp = True")
PY
  fi

  # ROCm loads KimiLinearModel from amd/linear.py (not nvidia/model.py).
  # supports_aux_hidden_states_over_pp() inspects language_model.model — that
  # inner decoder — so the flag + pack_local_aux handoff must live here.
  # Also provision target embed_tokens on the last PP rank so DSpark/EAGLE
  # can maybe_share_target_embed() (PR #50514 only patched nvidia/model.py).
  local AMD_LINEAR="$VLLM_DIR/models/kimi_k3/amd/linear.py"
  [[ -f "$AMD_LINEAR" ]] || return 0
  python3 - "$AMD_LINEAR" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
t = p.read_text()
changed = False
aux_marker = "# [50514] amd linear aux-over-PP"
embed_marker = "# [50514] amd linear target embed on last PP"

# --- aux-over-PP (flag + pack + recv) ---
if aux_marker in t:
    print(f"{p}: aux-over-PP already patched")
else:
    # 1) Class-level opt-in on the inner decoder.
    old_cls = "class KimiLinearModel(nn.Module, EagleModelMixin):\n"
    new_cls = (
        "class KimiLinearModel(nn.Module, EagleModelMixin):\n"
        "    # Local aux taps are sent to the last PP rank for DSpark/EAGLE3 drafting.\n"
        "    supports_aux_hidden_states_over_pp = True\n"
        f"    {aux_marker}\n"
    )
    if old_cls not in t:
        raise SystemExit(f"{p}: KimiLinearModel class header not found")
    t = t.replace(old_cls, new_cls, 1)

    # 2) Recv upstream aux on the last rank before local capture.
    old_aux_init = """        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors[\"hidden_states\"]
            residual = intermediate_tensors[\"residual\"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
"""
    new_aux_init = """        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors[\"hidden_states\"]
            residual = intermediate_tensors[\"residual\"]

        # Earlier stages' taps arrive only on the last rank (#50514).
        remote_aux: list[torch.Tensor] = []
        if get_pp_group().is_last_rank and self.aux_hidden_state_layers:
            remote_aux = self.recv_remote_aux_from_producers(intermediate_tensors)

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
"""
    if old_aux_init not in t:
        raise SystemExit(f"{p}: forward aux-init block not found")
    t = t.replace(old_aux_init, new_aux_init, 1)

    # 3) Pack local aux into IntermediateTensors on non-last ranks (both branches).
    old_ret = """            if not get_pp_group().is_last_rank:
                return IntermediateTensors(
                    {\"hidden_states\": hidden_states, \"residual\": residual}
                )
"""
    new_ret = """            if not get_pp_group().is_last_rank:
                tensors = {
                    \"hidden_states\": hidden_states,
                    \"residual\": residual,
                    **self.pack_local_aux_for_last(aux_hidden_states),
                }
                return IntermediateTensors(tensors)
"""
    # attn-res branch uses 8-space indent without the extra 4 spaces from `if attn_res is None`
    old_ret2 = """        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {\"hidden_states\": hidden_states, \"residual\": residual}
            )
"""
    new_ret2 = """        if not get_pp_group().is_last_rank:
            tensors = {
                \"hidden_states\": hidden_states,
                \"residual\": residual,
                **self.pack_local_aux_for_last(aux_hidden_states),
            }
            return IntermediateTensors(tensors)
"""
    n1 = t.count(old_ret)
    n2 = t.count(old_ret2)
    if n1:
        t = t.replace(old_ret, new_ret)
    if n2:
        t = t.replace(old_ret2, new_ret2)
    if n1 + n2 == 0:
        raise SystemExit(f"{p}: IntermediateTensors return sites not found")

    # 4) Prepend remote aux before returning draft taps on the last rank.
    old_fin1 = """            if residual is not None:
                hidden_states = hidden_states + residual
            if aux_hidden_states:
                return hidden_states, aux_hidden_states
            return hidden_states
"""
    new_fin1 = """            if residual is not None:
                hidden_states = hidden_states + residual
            aux_hidden_states = remote_aux + aux_hidden_states
            if aux_hidden_states:
                return hidden_states, aux_hidden_states
            return hidden_states
"""
    old_fin2 = """        # NOTE: the final norm is applied in compute_logits instead of here, so
        # the MTP draft model receives the pre-norm hidden states.
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states
"""
    new_fin2 = """        # NOTE: the final norm is applied in compute_logits instead of here, so
        # the MTP draft model receives the pre-norm hidden states.
        aux_hidden_states = remote_aux + aux_hidden_states
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states
"""
    if old_fin1 not in t or old_fin2 not in t:
        raise SystemExit(f"{p}: last-rank aux return sites not found")
    t = t.replace(old_fin1, new_fin1, 1)
    t = t.replace(old_fin2, new_fin2, 1)
    changed = True
    print(f"patched {p}: KimiLinearModel aux-over-PP (flag+pack+recv)")

# --- target embed on last PP rank (idempotent even if aux was applied earlier) ---
# Call site is multiline: spec_decode_needs_target_embed(\n vllm_config\n)
if embed_marker in t or "is_first_rank or spec_decode_needs_target_embed" in t:
    print(f"{p}: target-embed-on-last-PP already patched")
else:
    # Import the helper next to the other utils imports.
    old_imp = """from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    get_spec_layer_idx_from_weight_name,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
"""
    new_imp = """from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    get_spec_layer_idx_from_weight_name,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    spec_decode_needs_target_embed,
)
"""
    if old_imp not in t:
        raise SystemExit(f"{p}: utils import block not found for embed patch")
    t = t.replace(old_imp, new_imp, 1)

    old_embed = """        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
"""
    new_embed = f"""        # DSpark/EAGLE drafts on the last PP stage alias the target's
        # embed_tokens; provision it here so maybe_share_target_embed works.
        {embed_marker}
        if get_pp_group().is_first_rank or spec_decode_needs_target_embed(
            vllm_config
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{{prefix}}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
"""
    if old_embed not in t:
        raise SystemExit(f"{p}: embed_tokens first-rank block not found")
    t = t.replace(old_embed, new_embed, 1)
    changed = True
    print(f"patched {p}: KimiLinearModel target embed on last PP rank")

if changed:
    p.write_text(t)
PY
}

amd_last_stage_target_embed() {
  # #50514 patches the NVIDIA KimiLinearModel to build embed_tokens on the last
  # PP rank too (`get_pp_group().is_first_rank or
  # spec_decode_needs_target_embed(vllm_config)`), because the DSpark drafter
  # runs there and aliases the target's input embedding. ROCm instead loads
  # amd/linear.py, which the PR never touches, so the last stage keeps a
  # PPMissingLayer and draft load dies with "ships no input embedding of its
  # own". Mirror the NVIDIA change here.
  local AMD_LINEAR="$VLLM_DIR/models/kimi_k3/amd/linear.py"
  [[ -f "$AMD_LINEAR" ]] || return 0
  python3 - "$AMD_LINEAR" <<'PY'
from pathlib import Path
import re
import sys

p = Path(sys.argv[1])
t = p.read_text()
marker = "# [50514] amd linear target-embed on last PP stage"
if marker in t:
    print(f"{p}: already has last-stage target embed")
    sys.exit(0)

# opt_in_amd_aux widens the same guard via the upstream helper; this function is
# only the fallback for builds where that hunk did not apply.
if "is_first_rank or spec_decode_needs_target_embed" in t:
    print(f"{p}: embed guard already widened by opt_in_amd_aux")
    sys.exit(0)

helper = f'''

def _spec_needs_target_embed(vllm_config) -> bool:
    {marker}
    # Prefer the upstream helper; fall back to its semantics when the
    # models/utils.py hunk of #50514 did not apply to this build.
    try:
        from vllm.model_executor.models.utils import spec_decode_needs_target_embed
    except ImportError:
        pass
    else:
        return spec_decode_needs_target_embed(vllm_config)

    from vllm.distributed.parallel_state import get_pp_group

    if getattr(vllm_config, "speculative_config", None) is None:
        return False
    pp_group = get_pp_group()
    return pp_group.world_size > 1 and pp_group.is_last_rank

'''

# Insert the helper just above the KimiLinearModel class definition.
cls_re = re.compile(r"^class KimiLinearModel\(", re.MULTILINE)
m = cls_re.search(t)
if not m:
    raise SystemExit(f"{p}: KimiLinearModel class not found")
t = t[: m.start()] + helper.lstrip("\n") + "\n" + t[m.start() :]

# Widen the embed_tokens guard: the is_first_rank check that builds embed_tokens.
guard_re = re.compile(
    r"(?P<indent>[ \t]*)if get_pp_group\(\)\.is_first_rank:\n"
    r"(?P<body>(?:[ \t]*\n)*[ \t]*self\.embed_tokens\s*=)"
)
matches = [mm for mm in guard_re.finditer(t)]
if len(matches) != 1:
    raise SystemExit(
        f"{p}: expected exactly 1 embed_tokens is_first_rank guard, found {len(matches)}"
    )
mm = matches[0]
new_guard = (
    f"{mm.group('indent')}if get_pp_group().is_first_rank or "
    f"_spec_needs_target_embed(vllm_config):\n{mm.group('body')}"
)
t = t[: mm.start()] + new_guard + t[mm.end() :]

p.write_text(t)
print(f"patched {p}: last PP stage builds target embed_tokens for DSpark")
PY
}

lift_eagle3_pp_guard() {
  # #50514 deletes the whole "EAGLE3 with pipeline parallelism" unsupported
  # block. Commenting only the append line leaves an empty `if` body and
  # IndentationError — remove the full 4-line if instead.
  local VLLM_CFG="$VLLM_DIR/config/vllm.py"
  python3 - "$VLLM_CFG" <<'PY'
from pathlib import Path
import re, sys
p = Path(sys.argv[1])
t = p.read_text()
# Undo a prior broken comment-only edit if present.
t = t.replace(
    "# [50514] lifted: EAGLE3/DSpark under PP via last-stage drafter\n"
    "            # unsupported.append(\"EAGLE3 with pipeline parallelism\")",
    'unsupported.append("EAGLE3 with pipeline parallelism")',
)
pat = re.compile(
    r"\n[ \t]*if \(\n"
    r"[ \t]*speculative_config\.method == \"eagle3\"\n"
    r"[ \t]*and self\.parallel_config\.pipeline_parallel_size > 1\n"
    r"[ \t]*\):\n"
    r"[ \t]*unsupported\.append\(\"EAGLE3 with pipeline parallelism\"\)\n",
)
new, n = pat.subn(
    "\n            # [50514] EAGLE3/DSpark under PP via last-stage drafter "
    "(guard removed)\n",
    t,
    count=1,
)
if n:
    p.write_text(new)
    print("vllm.py: removed EAGLE3+PP unsupported if-block")
elif 'unsupported.append("EAGLE3 with pipeline parallelism")' not in new:
    print("vllm.py: EAGLE3+PP guard already absent")
else:
    # Fallback: line-oriented delete around the append
    lines = t.splitlines(keepends=True)
    out = []
    i = 0
    removed = False
    while i < len(lines):
        if 'unsupported.append("EAGLE3 with pipeline parallelism")' in lines[i]:
            # drop preceding if (...) : block (up to ~5 lines back)
            while out and out[-1].strip() in {")", "):"} or (
                out and ("speculative_config.method" in out[-1]
                         or "pipeline_parallel_size" in out[-1]
                         or out[-1].strip().startswith("if ")
                         or out[-1].strip() == "if ("
                         or out[-1].strip() == "and")
            ):
                out.pop()
            # also pop bare 'if (' if left
            while out and out[-1].strip() in {"if (", "if ("}:
                out.pop()
            if out and out[-1].lstrip().startswith("if ("):
                out.pop()
            out.append(
                "            # [50514] EAGLE3/DSpark under PP via last-stage "
                "drafter (guard removed)\n"
            )
            removed = True
            i += 1
            continue
        out.append(lines[i])
        i += 1
    if not removed:
        raise SystemExit("vllm.py: failed to locate EAGLE3+PP guard")
    p.write_text("".join(out))
    print("vllm.py: removed EAGLE3+PP guard (fallback)")
PY
}

separate_draft_capture_phase() {
  # capture_model() already captures the target before speculator.capture(), but
  # only the last PP stage owns a speculator. Without phase barriers, earlier PP
  # stages can enter warmup_kernels (and block in PP isend) while the last stage
  # is still flushing the draft graph's custom-all-reduce buffers.
  local MODEL_RUNNER="$VLLM_DIR/v1/worker/gpu/model_runner.py"
  [[ -f "$MODEL_RUNNER" ]] || return 0
  python3 - "$MODEL_RUNNER" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
t = p.read_text()
marker = "# [50514] isolate draft cudagraph capture across PP"
if marker in t:
    print(f"{p}: draft capture phase barriers already patched")
    raise SystemExit(0)

old_import = """from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    prepare_communication_buffer_for_model,
)
"""
new_import = """from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    get_world_group,
    prepare_communication_buffer_for_model,
)
"""
if old_import not in t:
    raise SystemExit(f"{p}: parallel_state import block not found")
t = t.replace(old_import, new_import, 1)

old_capture = """            if self.speculator is not None:
                self.speculator.capture()
"""
new_capture = f"""            # The target graph is captured by every PP stage above. The draft
            # graph exists only on the last stage, so make it a separate phase:
            # non-last stages wait instead of entering PP warmup early.
            {marker}
            draft_over_pp = (
                self.speculative_config is not None
                and self.parallel_config.pipeline_parallel_size > 1
            )
            if draft_over_pp:
                get_world_group().barrier()
            if self.speculator is not None:
                self.speculator.capture()
            if draft_over_pp:
                get_world_group().barrier()
"""
if old_capture not in t:
    raise SystemExit(f"{p}: speculator capture block not found")
t = t.replace(old_capture, new_capture, 1)
p.write_text(t)
print(f"patched {p}: isolated last-stage draft cudagraph capture phase")
PY
}

WS="${WS:-/tmp/vllm_50514_apply}"
mkdir -p "$WS"
echo "[50514] ROOT=$ROOT"

if [[ "${NEED_APPLY:-0}" == "1" ]]; then
  if [[ -d "$PATCH_DIR" ]] && compgen -G "$PATCH_DIR/*.patch" >/dev/null; then
    cd "$ROOT"
    for patchf in $(ls "$PATCH_DIR"/*.patch | sort); do
      filt="$WS/$(basename "$patchf").notests"
      filter_tests "$patchf" "$filt"
      echo "[50514] applying $(basename "$patchf") (tests stripped)..."
      set +e
      patch -p1 --forward --batch -r "$WS/$(basename "$patchf").rej" < "$filt" \
        >"$WS/$(basename "$patchf").log" 2>&1
      rc=$?
      set -e
      # Summarize
      if grep -q "FAILED\|Skipping patch\|ignored" "$WS/$(basename "$patchf").log"; then
        echo "  WARN: partial apply rc=$rc — see $WS/$(basename "$patchf").log"
        grep -E "FAILED|Skipping|succeeded|ignored" "$WS/$(basename "$patchf").log" | tail -30 || true
      else
        echo "  OK rc=$rc"
      fi
    done
  elif [[ -f "${SCRIPT_DIR}/vllm_pr50514.patch" ]]; then
    echo "[50514] WARN: per-commit dir missing; falling back to monolithic patch"
    filter_tests "${SCRIPT_DIR}/vllm_pr50514.patch" "$WS/mono.notests.patch"
    cd "$ROOT"
    set +e
    patch -p1 --forward --batch < "$WS/mono.notests.patch" >"$WS/mono.log" 2>&1
    set -e
    tail -40 "$WS/mono.log" || true
  else
    echo "[50514] WARN: no patch files; surgical draft PP=1 only"
  fi

  # Always ensure the draft-PP=1 marker — this is the SupportsPP boot fix.
  if ! grep -qF "$MARKER" "$TARGET" 2>/dev/null; then
    echo "[50514] marker missing after patch; applying surgical draft PP=1"
    surgical_draft_pp1
  fi
fi

lift_eagle3_pp_guard
opt_in_amd_aux
amd_last_stage_target_embed
separate_draft_capture_phase

# Overlay fallback for worker files that often reject against cb810+apply_k3.
OVERLAY="${SCRIPT_DIR}/vllm_pr50514_overlay"
if [[ -d "$OVERLAY/vllm" ]]; then
  for rel in \
    v1/worker/gpu/pp_utils.py \
    v1/worker/gpu/spec_decode/dspark/utils.py \
    v1/worker/gpu/spec_decode/eagle/eagle3_utils.py \
    v1/worker/gpu/spec_decode/eagle/utils.py \
    v1/worker/gpu/spec_decode/dflash/utils.py
  do
    src="$OVERLAY/vllm/$rel"
    dst="$VLLM_DIR/$rel"
    [[ -f "$src" ]] || continue
    # Only overlay if the #50514 symbol is still missing.
    if ! grep -qE "supports_aux_hidden_states_over_pp|pack_local_aux|last PP|pipeline_parallel_size=1" "$dst" 2>/dev/null \
       && ! grep -qE "forward_aux|aux_hidden_states_over_pp|share.*embed|PPMissingLayer" "$dst" 2>/dev/null; then
      echo "[50514] overlay $rel (post-patch symbols missing)"
      cp -f "$src" "$dst"
    fi
  done
fi

# Draft loads only on the last PP stage; the global --load-format is
# fastsafetensors, whose loader broadcasts shards over the *world* process
# group. Stages that skip draft loading (PP<last) never join that collective, so
# the run deadlocks until the distributed timeout. Force-refresh dspark/utils.py
# to the overlay that loads the draft via a per-rank safetensors read. The
# generic overlay loop above skips this file once any #50514 symbol is present,
# so guard on the load-fix marker explicitly.
DSPARK_UTILS="$VLLM_DIR/v1/worker/gpu/spec_decode/dspark/utils.py"
DSPARK_SRC="$OVERLAY/vllm/v1/worker/gpu/spec_decode/dspark/utils.py"
DRAFT_LOAD_MARKER="[50514] draft loads only on the last PP stage"
if [[ -f "$DSPARK_SRC" ]]; then
  if ! grep -qF "$DRAFT_LOAD_MARKER" "$DSPARK_UTILS" 2>/dev/null; then
    echo "[50514] refreshing dspark/utils.py (draft per-rank safetensors load)"
    cp -f "$DSPARK_SRC" "$DSPARK_UTILS"
  else
    echo "[50514] dspark/utils.py already has draft safetensors load fix"
  fi
  if ! grep -qF "$DRAFT_LOAD_MARKER" "$DSPARK_UTILS" 2>/dev/null; then
    echo "[50514] ERROR: draft safetensors load fix missing after refresh" >&2
    exit 1
  fi
fi

if ! grep -qF "$MARKER" "$TARGET" 2>/dev/null; then
  echo "[50514] ERROR: draft PP=1 marker still missing" >&2
  exit 1
fi

# Sanity: draft config must hard-code PP=1
if ! grep -A20 "def create_draft_parallel_config" "$TARGET" | grep -q "pipeline_parallel_size=1"; then
  echo "[50514] ERROR: create_draft_parallel_config does not set pipeline_parallel_size=1" >&2
  exit 1
fi

echo "[50514] done (draft last-PP-stage path armed)"
