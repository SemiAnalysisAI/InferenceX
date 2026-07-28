#!/bin/bash
# Setup script for the Kimi-K3 vLLM bring-up image (vllm/vllm-openai:kimi-k3).
# srt-slurm runs this in every worker container before dynamo install and
# worker startup (recipe field: setup_script).

set -euo pipefail

# The image's first decode step crashes in the KDA hybrid-state postprocess:
#   vllm/v1/worker/gpu/model_states/mamba_hybrid.py, postprocess_state:
#   IndexError: index_fill_(): Expected dtype int64 for index.
# torch's index_fill_ requires an int64 index tensor, but the runner passes
# the int32 idx_mapping (hit by moonshotai/Kimi-K3 agentic bring-up, first
# decode step, engine v0.1.dev19262+gb6bbf29dd). Coerce the index to int64.
# Idempotent: exits 0 if the patch is already applied.
python3 - <<'PY'
import pathlib
import re

import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh

path = pathlib.Path(mh.__file__)
src = path.read_text()
if "idx_mapping.long()" in src:
    print(f"mamba_hybrid index_fill_ patch already applied: {path}")
    raise SystemExit(0)

new, n = re.subn(
    r"index_fill_\(\s*0,\s*idx_mapping,",
    "index_fill_(0, idx_mapping.long(),",
    src,
)
if n != 1:
    raise SystemExit(
        f"expected exactly one index_fill_(0, idx_mapping, ...) call in "
        f"{path}, found {n} — image layout changed, refusing to patch"
    )
path.write_text(new)
print(f"Patched mamba_hybrid index_fill_ index dtype: {path}")
PY

# The DSpark draft head (Inferact/Kimi-K3-DSpark) does not implement
# SupportsPP, and SpeculativeConfig verifies the draft model against a
# parallel config that unconditionally inherits the target's
# pipeline_parallel_size — so TP8xPP2 dies at engine init with
# "NotImplementedError: Pipeline parallelism is not supported for this
# model". At runtime, however, V1 drafters are loaded ONLY on the final
# pipeline stage (vllm-project/vllm#16568: "drafters are only loaded in the
# last pp stage, which essentially means draft_pipeline_parallel_size=1"),
# so the correct check is against a pp=1 view of the parallel config.
# Idempotent; refuses to patch if the call-site shape changed.
python3 - <<'PY'
import pathlib
import re

import vllm.config.speculative as sp

path = pathlib.Path(sp.__file__)
src = path.read_text()
if "_infmax_pp1_draft_view" in src:
    print(f"draft-PP verification patch already applied: {path}")
    raise SystemExit(0)

pat = re.compile(
    r"self\.draft_model_config\.verify_with_parallel_config\(\s*"
    r"(self\.(?:draft|target)_parallel_config)\s*,?\s*\)",
    re.S,
)
matches = list(pat.finditer(src))
if len(matches) != 1:
    raise SystemExit(
        f"expected exactly one draft verify_with_parallel_config call in "
        f"{path}, found {len(matches)} — image layout changed, refusing to patch"
    )
src = pat.sub(
    lambda m: (
        "self.draft_model_config.verify_with_parallel_config("
        f"_infmax_pp1_draft_view({m.group(1)}))"
    ),
    src,
    count=1,
)
src += '''

def _infmax_pp1_draft_view(parallel_config):
    """InferenceX: V1 drafters load whole on the final pipeline stage, so the
    draft model is verified against a pp=1 view of the parallel config (the
    DSpark draft head does not implement SupportsPP)."""
    import copy

    pc = copy.deepcopy(parallel_config)
    pc.pipeline_parallel_size = 1
    return pc
'''
path.write_text(src)
print(f"Patched draft-model PP verification: {path}")
PY

# DSpark under pipeline parallelism, part 2: transport aux hidden states
# across PP stages. The K3 DSpark head consumes aux hidden states from five
# target layers (target_layer_ids [2, 23, 47, 71, 89]); the target model
# captures them per PP stage, so with TP8xPP2 the layers-2/23 captures live
# on stage 0 and never reach the last-stage drafter (whose
# combine_hidden_states expects hidden_size x 5). This build's V2 runner
# already broadcasts sampled counts across PP ranks (PPHandler), so the only
# missing piece is the aux transport — the same gap vllm-ascend PR #12507
# fixes for EAGLE3. Three coordinated edits to kimi_k3/nvidia/model.py:
#   (a) preallocate aux_hidden_<layer> intermediate-tensor buffers on
#       receiving ranks for aux layers owned by earlier stages;
#   (b) non-last ranks attach their captured aux states (and pass through
#       upstream ones) to the outgoing IntermediateTensors;
#   (c) the last rank merges received + local aux states in global layer
#       order before returning them to the drafter.
# Idempotent; hard-asserts on exact anchors and refuses to patch otherwise.
python3 - <<'PY'
import pathlib

import vllm.models.kimi_k3.nvidia.model as m

path = pathlib.Path(m.__file__)
src = path.read_text()
if "_infmax_aux_pp" in src:
    print(f"aux-PP transport patch already applied: {path}")
    raise SystemExit(0)

def replace_once(src: str, old: str, new: str, tag: str) -> str:
    n = src.count(old)
    if n != 1:
        raise SystemExit(
            f"aux-PP patch anchor {tag!r} matched {n} times in {path} — "
            "image layout changed, refusing to patch"
        )
    return src.replace(old, new)

# (a) preallocate upstream-aux receive buffers
old_a = """        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(residual_shape, dtype=dtype, device=device),
            }
        )"""
new_a = """        _infmax_aux_pp = {
            f"aux_hidden_{_l}": torch.zeros(
                (batch_size, self.config.hidden_size), dtype=dtype, device=device
            )
            for _l in sorted(getattr(self, "aux_hidden_state_layers", ()) or ())
            if _l < self.start_layer
        }
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(residual_shape, dtype=dtype, device=device),
                **_infmax_aux_pp,
            }
        )"""
src = replace_once(src, old_a, new_a, "prealloc")

# (b) non-last ranks: send local aux states, pass through upstream ones
old_b = """            if prefix_sum is not None:
                hidden_states = hidden_states + prefix_sum
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )"""
new_b = """            if prefix_sum is not None:
                hidden_states = hidden_states + prefix_sum
            _infmax_out = {"hidden_states": hidden_states, "residual": residual}
            if intermediate_tensors is not None:
                for _k, _v in intermediate_tensors.tensors.items():
                    if _k.startswith("aux_hidden_"):
                        _infmax_out.setdefault(_k, _v)
            _infmax_local = sorted(
                _l
                for _l in (getattr(self, "aux_hidden_state_layers", ()) or ())
                if self.start_layer <= _l <= self.end_layer
            )
            for _l, _t in zip(_infmax_local, aux_hidden_states):
                _infmax_out[f"aux_hidden_{_l}"] = _t
            return IntermediateTensors(_infmax_out)"""
src = replace_once(src, old_b, new_b, "send")

# (c) last rank: merge received + local aux in global layer order
old_c = """        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states"""
new_c = """        if aux_hidden_states:
            _infmax_merged = {}
            if intermediate_tensors is not None:
                for _k, _v in intermediate_tensors.tensors.items():
                    if _k.startswith("aux_hidden_"):
                        _infmax_merged[int(_k.rsplit("_", 1)[1])] = _v
            _infmax_local = sorted(
                _l
                for _l in (getattr(self, "aux_hidden_state_layers", ()) or ())
                if self.start_layer <= _l <= self.end_layer
            )
            _infmax_merged.update(zip(_infmax_local, aux_hidden_states))
            aux_hidden_states = [_infmax_merged[_k] for _k in sorted(_infmax_merged)]
            return hidden_states, aux_hidden_states
        return hidden_states"""
src = replace_once(src, old_c, new_c, "merge")

path.write_text(src)
print(f"Patched aux-hidden-state PP transport ({path})")
PY

# DSpark under pipeline parallelism, part 3: lift the runner's blanket guard.
# The V2 runner refuses aux-hidden-state spec methods (eagle3/dflash/dspark)
# under PP because the aux captures were stage-local — exactly the gap the
# aux-transport patch above closes. With transport in place (and PPHandler
# already broadcasting sampled counts across ranks), downgrade the hard
# ValueError to a warning. Idempotent; refuses to patch on layout drift.
python3 - <<'PY'
import pathlib

import vllm.v1.worker.gpu.model_runner as mr

path = pathlib.Path(mr.__file__)
src = path.read_text()
if "_infmax_pp_aux_enabled" in src:
    print(f"PP spec-method guard patch already applied: {path}")
    raise SystemExit(0)

old = """                self.use_aux_hidden_state_outputs = True
                if self.use_pp:
                    raise ValueError(
                        f"{self.speculative_config.method} with pipeline parallel "
                        "is not supported."
                    )"""
new = """                self.use_aux_hidden_state_outputs = True
                if self.use_pp:
                    # _infmax_pp_aux_enabled: aux hidden states are shipped
                    # across PP stages by the InferenceX container patch.
                    logger.warning(
                        "InferenceX: %s under pipeline parallelism enabled "
                        "via patched aux-hidden-state transport.",
                        self.speculative_config.method,
                    )"""
n = src.count(old)
if n != 1:
    raise SystemExit(
        f"expected exactly one PP spec-method guard in {path}, found {n} — "
        "image layout changed, refusing to patch"
    )
path.write_text(src.replace(old, new))
print(f"Lifted PP spec-method guard: {path}")
PY
