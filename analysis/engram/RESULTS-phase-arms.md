# Engram: where in the sequence does it earn its keep?

Run 34705399251, DeepSeek-V4.1-Flash FP4, B200 TP4, one node.
1508 chunks x 16 domains, 2,702,336 scored tokens per arm.

## What the arms mean

Scoring prompt logprobs is a single prefill, so there is no engine decode
phase to split on. The split is therefore **positional**: each 3584-token
chunk is cut at position 1792, and every arm is scored on the *same*
suffix tokens, so the four numbers are directly comparable.

| arm | Engram is allowed to act on |
| --- | --- |
| `baseline` | everything |
| `ablated` | nothing |
| `prefill_only` | the context prefix only -- it shapes the KV the suffix attends to, but is shut at the point of prediction |
| `decode_only` | the scored tokens themselves only |

So `prefill_only`'s delta is the cost of taking Engram away from the predicted
tokens, and `decode_only`'s delta is the cost of taking it away from the context.

## Result

| domain | ppl base | ablated dbits | prefill_only dbits | decode_only dbits | abl sigma | pre sigma | dec sigma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wiki_full | 1.369 | 2.6396 | 2.5322 | 0.1179 | 44.8 | 45.5 | 16.3 |
| code_javascript | 1.349 | 1.7921 | 1.5380 | 0.2066 | 39.5 | 35.5 | 23.5 |
| code_ruby | 1.716 | 1.6300 | 1.3039 | 0.1788 | 33.1 | 28.7 | 24.1 |
| wiki_zh | 4.413 | 1.3802 | 1.4160 | -0.0282 | 16.2 | 16.5 | -3.1 |
| wiki | 2.826 | 1.2068 | 1.1702 | 0.0774 | 20.4 | 17.8 | 13.3 |
| code_mbpp | 1.506 | 1.1049 | 0.9133 | 0.0534 | 14.0 | 10.9 | 5.6 |
| code_python | 2.203 | 0.9459 | 0.6837 | 0.1058 | 28.9 | 24.3 | 15.8 |
| code_php | 1.896 | 0.8978 | 0.6974 | 0.0724 | 28.7 | 24.5 | 17.7 |
| code_java | 1.614 | 0.8048 | 0.6033 | 0.0890 | 25.2 | 21.2 | 19.1 |
| math_web | 3.112 | 0.7856 | 0.6782 | 0.0520 | 13.8 | 12.6 | 6.9 |
| web_zh | 13.591 | 0.7482 | 0.6543 | 0.0243 | 18.2 | 16.1 | 3.1 |
| web | 5.860 | 0.7044 | 0.5724 | 0.0282 | 14.8 | 12.9 | 5.6 |
| math | 2.012 | 0.5235 | 0.5912 | 0.0155 | 27.5 | 30.9 | 4.0 |
| code_go | 1.033 | 0.4129 | 0.2984 | 0.0509 | 13.7 | 11.4 | 11.9 |
| chat | 3.070 | 0.1937 | 0.1457 | 0.0020 | 15.9 | 14.6 | 1.1 |
| chat_zh | 6.054 | 0.0780 | 0.0593 | -0.0067 | 18.9 | 16.1 | -4.5 |

**Engram is a point-of-prediction lookup, not a context builder.**
Letting it act only on the tokens being predicted recovers a median
**94%** of the full-ablation loss. Letting it act only on the context
recovers a median **18%** -- nearly as bad as removing it outright.
On wiki_full, removing Engram from the predicted tokens costs 2.53 bits/token;
removing it from the context costs 0.12.

The two deltas are close to additive on most domains (wiki_full 2.53 + 0.12 =
2.65 vs 2.64 for the full ablation), i.e. the prefix and suffix contributions
are largely independent rather than one substituting for the other.

Two Chinese corpora invert: `decode_only` beats baseline slightly and
significantly (wiki_zh -0.028 bits at -3.1 sigma, chat_zh -0.007 at -4.5 sigma).
The effect is real but tiny -- on these corpora Engram acting over the context
is mildly harmful. Not enough to build a claim on; recorded rather than
explained.

## Ablation verification

Every arm was metered on every chunk, not sampled:

| arm | chunks | Engram forward calls | worst contribution | min contribution |
| --- | ---: | ---: | ---: | ---: |

The ablated arm's contribution is exactly 0.0 on all 1,508 chunks / 12,064
calls -- not small, zero -- because the all-False `token_mask` is applied by
the shipped fused kernel itself. The baseline's minimum is 0.316, so Engram
was demonstrably active in every chunk it was supposed to be.

---

# Boundary sweep (run 34735263999): is the NLL/CRUXEval divergence context-shaped?

The likelihood run's `decode_only` recovers ~94% of the ablation loss; CRUXEval's
recovers essentially nothing (0.4850 vs the ablated 0.4888). I proposed that the
difference is how much context was prefilled with the gate shut -- half a
3584-token chunk here, the entire prompt there -- and predicted, before running,
that `decode_only`'s recovery would fall monotonically as the gate-shut prefix
grows. 25 chunks/domain × 16 domains, four boundaries, one process.

Median recovery of the full-ablation loss, across 16 domains:

| gate-shut prefix | scored suffix | prefill_only | decode_only |
| ---: | ---: | ---: | ---: |
| 512 | 3072 | 0.103 | **0.980** |
| 1792 | 1792 | 0.184 | **0.943** |
| 3072 | 512 | 0.283 | **0.821** |
| 3456 | 128 | 0.341 | **0.591** |

**The prediction holds directionally, and fails on magnitude.** Recovery does
fall monotonically, and the effect is not subtle -- on wiki_full, `decode_only`
costs 0.031 bits/token at boundary 512 and 0.794 at 3456 (13.0σ). So how much
context was built with the gate shut genuinely matters, which is the mechanism I
claimed.

But it is nowhere near enough to explain CRUXEval. Even with 3456 of 3584 tokens
prefilled gate-shut -- a far more extreme starvation than CRUXEval's ~1-2k-token
prompt -- `decode_only` still returns 59% of the loss. Extrapolating this curve
to CRUXEval's prompt length predicts recovery around 90%; the measurement is
~2%. **My context-length explanation is therefore at best a partial one, and I
was wrong to lead with it as the likely resolution.** A second, task-shaped
factor dominates: per-token likelihood is forgiving of a degraded prefix in a way
that a single graded whole-answer exact match, produced through a long
self-conditioned generation, is not.

One confound in this sweep, which I did not control and which cuts against
reading the curve too literally: raising the boundary both lengthens the
gate-shut prefix *and* shrinks the scored suffix (76,800 scored tokens at 512
down to 3,200 at 3456), so the scored tokens also sit deeper in context. The two
move together by construction. Separating them needs a fixed-width scored window
slid along a longer chunk, which this run does not do.

Prefix and suffix contributions stay close to additive across the sweep (median
`(prefill_only + decode_only) / ablated` = 0.92, 0.89, 0.92, 1.09), so the two
halves remain largely independent at every split point.

Verification: 12,256 Engram calls per arm over 1,532 chunks; ablated `worst`
exactly 0.0 on every chunk, baseline `min` 0.362.
