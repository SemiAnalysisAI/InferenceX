"""Measure installed vLLM n-gram hashing with explicit prefill fixtures."""

from __future__ import annotations

import argparse
import json
import statistics
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from experiment import kernel_samples


def main():
    import torch
    import vllm
    from vllm.models.deepseek_v41.common.engram import (
        EngramLayout,
        NgramHashState,
        compute_hash_multipliers,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[384006168, 384016682],
        engram_max_ngram_size=4,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_compressed_vocab_size=99092,
        engram_pad_token_id=2,
        engram_vocab_size=16000000,
    )
    layout = EngramLayout(config)
    state = NgramHashState.__new__(NgramHashState)
    torch.nn.Module.__init__(state)
    state.use_slot_cache = False
    state.block_size = 128
    state.pad_id = 2
    state.token_map = torch.arange(99092, device="cuda", dtype=torch.int32)
    state.primes = torch.tensor(layout.primes, device="cuda")
    state.offsets = layout.offsets.cuda()
    state.multipliers = compute_hash_multipliers((1, 14), 4, 99092).cuda()
    rows = []
    report = {
        "gpu": torch.cuda.get_device_name(),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "image": args.image,
        "config": vars(config),
        "rows": rows,
        "qualification": "Native NgramHashState.forward V2 no-slot-cache path. One prefill request, identity compressed-token map and deterministic token IDs; tokenizer normalization/setup excluded. The source does not publish sufficient fixture or timing boundaries for an exact match. Both layers and all 24 hash columns per layer are computed.",
    }
    for tokens in (8192, 16384):
        ids = ((torch.arange(tokens, device="cuda") * 37 + 11) % 99092).int()
        positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
        starts = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        dead = torch.zeros(tokens, device="cuda", dtype=torch.bool)
        lookback = torch.full((1, 3), -1, device="cuda", dtype=torch.int32)
        lookback_dead = torch.zeros((1, 3), device="cuda", dtype=torch.bool)

        run = partial(
            state.forward,
            ids,
            positions,
            starts,
            dead,
            lookback,
            lookback_dead,
            None,
            None,
        )
        actual = run().cpu()
        multipliers = state.multipliers.cpu().tolist()
        primes = state.primes.cpu().tolist()
        offsets = state.offsets.cpu().tolist()
        expected = []
        # Independent scalar integer arithmetic; no GPU hash helper used here.
        for position in range(tokens):
            layers = []
            for layer in range(2):
                columns = []
                for ngram in (2, 3, 4):
                    value = 0
                    for shift in range(ngram):
                        token = (
                            ((position - shift) * 37 + 11) % 99092
                            if position >= shift
                            else 2
                        )
                        value ^= token * multipliers[layer][shift]
                    columns.extend(
                        value % primes[layer][ngram - 2][head]
                        + offsets[layer][(ngram - 2) * 8 + head]
                        for head in range(8)
                    )
                layers.append(columns)
            expected.append(layers)
        torch.testing.assert_close(
            actual, torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0
        )
        for round_id in range(3):
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for i in range(300):
                    with torch.profiler.record_function(f"target_sample_{i}"):
                        run()
                    torch.cuda.synchronize()
            path = args.output / f"tokens{tokens}_round{round_id}.trace.json"
            prof.export_chrome_trace(str(path))
            samples = kernel_samples(json.loads(path.read_text()), 300)
            rows.append(
                {
                    "tokens": tokens,
                    "round": round_id,
                    "warmups": 3,
                    "samples": 300,
                    "kernel_sum_samples_us": samples,
                    "kernel_sum_mean_us": statistics.mean(samples),
                    "exact_integer_check": True,
                    "output_shape": list(actual.shape),
                    "trace": path.name,
                }
            )
            (args.output / "results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(tokens, round_id, statistics.mean(samples), flush=True)


if __name__ == "__main__":
    main()
