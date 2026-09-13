# Engram ablation on DeepSeek-V4.1-Flash — consolidated results

What Engram is here: an n-gram memory at layers 1 and 14. Its gate is computed
and consumed inside the fused Triton kernel `_fused_engram_post_wkv_kernel`
(`hidden + gate * value` in one store), so it is invisible to hooks. Ablation is
done the module's own documented way — an all-False `token_mask` through the
real forward, applied by the shipped kernel itself. That is why every ablated
measurement below reads exactly `0.0` rather than merely small.

All runs: DeepSeek-V4.1-Flash MXFP4, B200, vLLM, `enforce_eager=True` (the
meter's host-side work invalidates a CUDA graph capture), prefix caching off.

## Headline

| measurement | baseline | ablated | effect | verified |
| --- | ---: | ---: | ---: | --- |
| gsm8k pass@1 (1319 items) | — | — | **+0.0008** (none) | ✓ 120k gate-shut calls |
| NLL, 16 domains, 2.70M scored tokens | — | — | **+0.851 bits/token** (median) | ✓ ablated max 0.0 on all 1508 chunks |
| CRUXEval-O pass@1 (800 items) | 0.6388 | 0.4925 | **−0.1463**, χ²=74.3 (~8.6σ) | ✓ 16,480 calls, ablated exactly 0.0 |
| Terminal-Bench 4.0 | — | — | **not obtained** (runs stopped) | harness validated, eval incomplete |

Engram is load-bearing for likelihood and for code-output prediction, and
invisible on grade-school math. Two reproductions of CRUXEval agreed to four
decimals.

## Where Engram acts: phase arms

Two independent splits, both saying the same thing — Engram matters at the point
of prediction, and its value depends on having been open over the context too.

**Likelihood** (positional split of one 3584-token prefill at boundary 1792;
scoring prompt logprobs has no decode phase). Median across 16 domains, delta
bits/token vs baseline:

| arm | median Δ bits/token | recovery of ablation loss |
| --- | ---: | ---: |
| ablated | 0.851 | 0% |
| prefill_only (open on prefix, shut at prediction) | 0.681 | 18% |
| decode_only (shut on prefix, open at prediction) | 0.053 | **94%** |

**CRUXEval-O** (the real engine phase, from `query_start_loc`;
`phase_mask_unavailable` false on all four arms):

| arm | pass@1 | Δ | McNemar χ² | ~σ |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.6512 | — | — | — |
| prefill_only | 0.6100 | −0.0413 | 14.4 | 3.8 |
| decode_only | 0.4850 | −0.1663 | 92.2 | 9.6 |
| ablated | 0.4888 | −0.1625 | 85.8 | 9.3 |

The labels invert because the splits differ: likelihood `decode_only` means
Engram open *on the scored tokens*; CRUXEval `decode_only` means open only while
generating, shut over the whole prompt. Read together: neither half suffices.
Open on the context but shut at prediction returns 18%; open at prediction but
shut over the context is indistinguishable from full ablation.

## The divergence, and what actually explains it

Likelihood `decode_only` recovers 94%; CRUXEval `decode_only` recovers ~2%. I
proposed this was context-shaped — CRUXEval starves the whole prompt, likelihood
only half a chunk — and tested it with a boundary sweep before asserting it.

Median `decode_only` recovery vs gate-shut prefix length (16 domains):

| prefix shut | 512 | 1792 | 3072 | 3456 |
| --- | ---: | ---: | ---: | ---: |
| decode_only recovery | 0.980 | 0.943 | 0.821 | 0.591 |

**Directionally confirmed, quantitatively insufficient.** The monotone fall is
real and large (wiki_full: 0.031 → 0.794 bits/token, 13.0σ). But even starving
3456 of 3584 tokens — more extreme than CRUXEval's prompt — still returns 59%,
where CRUXEval returns ~2%. Extrapolation predicts ~90%. **Context length is a
partial explanation at best; leading with it as the resolution was wrong.** A
task-shaped factor dominates: per-token likelihood tolerates a degraded prefix
in a way a single graded whole-answer exact match, produced through a long
self-conditioned generation, does not.

Uncontrolled confound in that sweep: raising the boundary lengthens the shut
prefix *and* shrinks the scored suffix (76,800 → 3,200 tokens), so scored tokens
also sit deeper in context. A fixed-width window slid along a longer chunk would
separate them; this run does not.

## By answer kind (CRUXEval, 800 items)

| kind | items | baseline | ablated Δ | prefill_only Δ | decode_only Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| str | 371 | 0.577 | −0.159 | −0.038 | −0.173 |
| container | 280 | 0.721 | −0.175 | −0.032 | −0.171 |
| number | 98 | 0.663 | −0.163 | −0.082 | −0.143 |
| bool/None | 49 | 0.776 | −0.102 | −0.041 | −0.102 |

I predicted the loss would concentrate on copy-shaped (string/container)
answers. It does not — numbers fall comparably. That prediction was wrong.

## Open items and caveats

- **Terminal-Bench 4.0 has no result.** The first paired run (7h35m, 66 tasks)
  scored 0.045/0.030 against a published 31.2 and was refused as a measurement:
  root cause was my own `model_info` error declaring the context window as
  LiteLLM's per-call *output* cap, which floored both arms (164/32 turns died on
  `max_tokens must be at least 1, got 0`). The fix was validated on a slice
  (zero-budget rejections 164→0, summarization truncations 225→2, 524s 29→1),
  but the v2 paired runs were stopped before finishing. The harness is ready;
  the eval is not done.
- **gsm8k vs CRUXEval numbers remain confounded.** gsm8k ran through the chat
  template with CoT; CRUXEval ran raw few-shot completion (this checkpoint copy
  ships no `chat_template`). Pattern completion is exactly where an n-gram
  memory should help most, so prompt regime is a live alternative to "task".
- **CRUXEval baselines differ across runs** (0.6388 twice, 0.6512 in the
  four-arm run). Batch shape changes at temperature 0; all comparisons are
  within-run and paired per item, so deltas hold, but the 1.2pp shift measures
  nothing.
- Two Chinese corpora invert significantly but tinily under ablation (wiki_zh
  −0.028 bits at −3.1σ, chat_zh −0.007 at −4.5σ). Recorded, unexplained.

## Provenance

| result | run |
| --- | --- |
| NLL phase arms | 34705399251 |
| CRUXEval-O paired | 34706153348, 34707533583 |
| CRUXEval phase arms | 34732450349 |
| NLL boundary sweep | 34735263999 |
| Terminal-Bench v2 (stopped) | 34729702998, 34729704839 |

Detail: `RESULTS-4grams.md`, `RESULTS-cruxeval.md`, `RESULTS-phase-arms.md`.
Data: `nll_phase_arms.json`, `nll_boundary_sweep.json`,
`cruxeval_ablation.json`, `cruxeval_phase_arms.json`.
