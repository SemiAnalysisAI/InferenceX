#!/usr/bin/env python3
"""Repair Kimi-K3 PP2 DSpark boundary-tap duplication on vLLM v0.28."""

from __future__ import annotations

from pathlib import Path
import py_compile


SITE = Path("/usr/local/lib/python3.12/dist-packages")
MARKER = "K3 v0.28 PP DSpark five-feature contract"


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if MARKER in source:
        print(f"{path}: already patched")
        return
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one compatibility anchor, found {count}"
        )
    path.write_text(source.replace(old, new, 1))
    py_compile.compile(str(path), doraise=True)
    print(f"{path}: patched and compiled")


model_candidates = (
    SITE / "vllm/models/kimi_k3/nvidia/model.py",
    SITE / "vllm/models/kimi_k3/amd/linear.py",
)
old_capture = """        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
"""
new_capture = f"""        # {MARKER}: PP0 already transports the boundary tap whose capture
        # ID equals PP1's start_layer. Only the first PP rank may seed the
        # start-layer feature; otherwise layer 48 appears twice.
        aux_hidden_states: list[torch.Tensor] = []
        if get_pp_group().is_first_rank:
            aux_hidden_states = self._maybe_add_hidden_state(
                aux_hidden_states, self.start_layer, hidden_states, residual
            )
"""

patched_model = False
for model_path in model_candidates:
    if not model_path.is_file():
        continue
    source = model_path.read_text()
    if MARKER in source:
        patched_model = True
        print(f"{model_path}: already patched")
        continue
    if old_capture in source:
        replace_once(model_path, old_capture, new_capture)
        patched_model = True

if not patched_model:
    # Newer Kimi implementations may already contain the first-rank guard.
    guarded = """            get_pp_group().is_first_rank
            and self.start_layer in self.aux_hidden_state_layers
"""
    if not any(
        path.is_file() and guarded in path.read_text() for path in model_candidates
    ):
        raise RuntimeError("no Kimi PP boundary-capture implementation found")
    print("Kimi PP boundary capture already has a first-rank guard")


speculator = (
    SITE / "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
)
old_propose = """        if aux_hidden_states:
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
"""
new_propose = f"""        if aux_hidden_states:
            # {MARKER}. Kimi-K3 DSpark is trained on exactly five ordered
            # target features: capture IDs (3, 24, 48, 72, 90).
            shapes = tuple(tuple(t.shape) for t in aux_hidden_states)
            if len(aux_hidden_states) != 5 or any(
                t.ndim != 2 or t.shape[-1] != 7168 for t in aux_hidden_states
            ):
                raise RuntimeError(
                    "Kimi-K3 DSpark target-feature contract failed: "
                    f"expected 5 x [N,7168], got {{shapes}}"
                )
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
"""
replace_once(speculator, old_propose, new_propose)
print("Kimi-K3 v0.28 PP DSpark auxiliary contract ready")
