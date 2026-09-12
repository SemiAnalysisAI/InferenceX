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
