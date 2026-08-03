#!/usr/bin/env python3
"""Route RadixArk/Kimi-K3-DSpark to vLLM's Qwen3 DSpark class instead of DeepSeek-V4.

Why this is needed
------------------
There are two different DSpark draft models for Kimi-K3, and they are not
interchangeable:

  Inferact/Kimi-K3-DSpark  architectures=["K3DSparkModel"], model_type=k3_dspark
                           4B, block-diffusion backbone, 5 dense non-causal
                           layers, MLA (q_lora_rank / kv_lora_rank /
                           qk_rope_head_dim). Card reports mean acc_len 3.85.
                           Documented for vLLM; its card's speculative-config is
                           byte-identical to the one this recipe ships.

  RadixArk/Kimi-K3-DSpark  architectures=["DSparkDraftModel"], model_type=qwen3
                           2.2B, 5 Qwen3-style GQA layers (64 q / 16 kv), Markov
                           logit-bias head + confidence head, trained with
                           SpecForge. Card reports mean acc_len 4.26 (GSM8K
                           5.42, HumanEval 5.51). Documented for SGLang only
                           (--speculative-algorithm DSPARK); no vLLM config
                           given.

vLLM's registry contains BOTH of these:

    "DSparkDraftModel": ("vllm.models.deepseek_v4", "DSparkDeepseekV4ForCausalLM")
    "Qwen3DSparkModel": ("qwen3_dspark", "Qwen3DSparkForCausalLM")

RadixArk's config.json declares ``DSparkDraftModel``, so stock vLLM would load a
**Qwen3-backbone** drafter through the **DeepSeek-V4** implementation. The class
that actually matches it is ``Qwen3DSparkModel``.

Why not --hf-overrides
----------------------
`SpeculativeConfig.compose_draft_hf_overrides` says it outright: "Dict overrides
are target-specific key patches and are **not applied to the draft**." Only
*callable* overrides propagate to the draft config, and a callable cannot be
expressed on the CLI. So the CLI route does not exist; the correct hook is
`SpeculativeConfig.hf_config_override`, which is the function that already
rewrites draft architectures for every other MTP family (DeepSeek, GLM, MiMo,
Ernie, Nemotron...). This adds one more entry in exactly that style.

Why the weights should load
---------------------------
Checked against the checkpoint's safetensors header (62 tensors) and
`qwen3_dspark.py` / `qwen3_dflash.py`:

  RadixArk ships               vLLM expects
  ---------------------------  ------------------------------------------
  fc.weight                    DFlashQwen3Model.fc              OK
  hidden_norm.weight           DFlashQwen3Model.hidden_norm     OK
  norm.weight                  DFlashQwen3Model.norm            OK
  layers.N.self_attn.{q,k,v,o}_proj, q_norm, k_norm,
  layers.N.mlp.{gate,up,down}_proj, {input,post_attention}_layernorm
                               standard Qwen3 block             OK
  markov_head.markov_w1/w2     DSparkMarkovHead.markov_w1/w2    OK
  confidence_head.*            explicitly in load_weights'
                               skip_substrs                     SKIPPED
  (no embed_tokens, no lm_head) load_weights adds both to
                               skip_substrs when absent and
                               load_dspark_model aliases the
                               target's                         OK

`draft_vocab_size` is absent from RadixArk's config, and
`Qwen3DSparkForCausalLM.__init__` already falls back to `vocab_size` for exactly
that case. `markov_rank` is present (256). `block_size` is 7, matching
num_speculative_tokens 7.

RESIDUAL RISK, stated plainly: `qwen3_dflash.py` also reads `causal`,
`mask_token_id` and `swa_window_size`, which RadixArk's config does not define.
Those are read through `config.get(...)` in places, so they may default
cleanly -- but this combination has never been run, and a missing field will
surface as an engine-init failure, which is a cheap and obvious outcome. This is
an experiment, not a supported path.

Enabled only when DSPARK_ARCH_FIX=1. Idempotent, and fails loudly.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import os
import sys

REL = "vllm/config/speculative.py"

ANCHOR = """    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
"""

INJECT = '''    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
        # LOCAL PATCH -- RadixArk/Kimi-K3-DSpark declares architectures=
        # ["DSparkDraftModel"], which this registry maps to the DeepSeek-V4
        # drafter (vllm.models.deepseek_v4.DSparkDeepseekV4ForCausalLM). That is
        # the wrong implementation: RadixArk is a Qwen3-backbone drafter (5
        # GQA layers, 64 q / 16 kv, Markov logit-bias head) and the matching
        # class is Qwen3DSparkModel -> qwen3_dspark.Qwen3DSparkForCausalLM.
        # Discriminate on model_type so a genuine DeepSeek-V4 DSpark draft
        # (model_type deepseek_v4) is untouched.
        if (
            initial_architecture == "DSparkDraftModel"
            and getattr(hf_config, "model_type", None) == "qwen3"
        ):
            hf_config.update({"architectures": ["Qwen3DSparkModel"]})
            initial_architecture = "Qwen3DSparkModel"
'''


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def find_target() -> str:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        die("vllm is not importable in this interpreter")
    root = os.path.dirname(list(spec.submodule_search_locations)[0])
    path = os.path.join(root, REL)
    if not os.path.isfile(path):
        die(f"{path} not found")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-out")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    path = find_target()
    original = open(path, encoding="utf-8").read()

    if "Qwen3DSparkModel" in original and "LOCAL PATCH -- RadixArk" in original:
        print(f"{path} already patched; nothing to do")
        return

    if original.count(ANCHOR) != 1:
        die(
            f"hf_config_override anchor matched {original.count(ANCHOR)} times, "
            "expected 1 -- the image drifted; re-derive before running."
        )

    text = original.replace(ANCHOR, INJECT)
    compile(text, path, "exec")

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile=f"a/{REL}",
            tofile=f"b/{REL}",
        )
    )
    if args.diff_out:
        with open(args.diff_out, "w", encoding="utf-8") as fh:
            fh.write(diff)
    print(diff)

    if args.check:
        print("--check: anchor OK, result compiles; not written")
        return

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Patched {path}: DSparkDraftModel+qwen3 -> Qwen3DSparkModel")


if __name__ == "__main__":
    main()
