# CRUXEval-O with and without Engram, nothing executed

Runs 34706153348 and 34707533583 (identical pass@1 both times), DeepSeek-V4.1-Flash
FP4, B200 TP4, one node. 800 items, prompt style `raw-completion`.

The model is denied any execution tool and has to simulate the program.
Grading executes nothing either: both the reference and the model's answer are
parsed with `ast.literal_eval` and compared as values, so an unsimplified
expression is graded wrong rather than evaluated.

## Headline

| arm | pass@1 |
| --- | ---: |
| baseline | 0.6388 |
| Engram removed | 0.4925 |

Delta **-0.1463**. Paired over the same items:
362 both correct, 257 both wrong,
**149 only baseline**, **32 only ablated**;
McNemar chi2 74.343 (~8.62 sigma).

This is the largest accuracy effect measured so far -- gsm8k moved +0.0008.

## Split by what the answer demands

The obvious confound is copying: many CRUXEval-O answers are strings or
containers largely rearranged from characters already in the prompt, and an
n-gram memory helps reproduce those without simulating anything. Numbers and
booleans have nothing to copy.

| answer kind | items | baseline | ablated | delta | only base | only abl | McNemar sigma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| str | 371 | 0.5714 | 0.4232 | -0.1482 | 68 | 13 | 6.0 |
| container | 280 | 0.7071 | 0.5464 | -0.1607 | 51 | 6 | 5.8 |
| number | 98 | 0.6327 | 0.5102 | -0.1224 | 20 | 8 | 2.1 |
| bool/None | 49 | 0.7551 | 0.6735 | -0.0816 | 9 | 5 | 0.8 |
| other | 2 | 1.0000 | 0.5000 | -0.5000 | 1 | 0 | 0.0 |

**The copy hypothesis does not carry the result.** The drop is broad, not
concentrated on the copy-shaped items: `str` -14.8pp, `container` -16.1pp,
`number` -12.2pp. Removing Engram hurts items whose answer is a computed
integer nearly as much as items whose answer is a string.

Where the split does bite is significance, and that is worth being precise
about rather than reading the point estimates alone: `str` (6.0 sigma) and
`container` (5.8 sigma) are solid; `number` is 2.1 sigma on 98 items;
`bool/None` is 0.8 sigma on 49 items and is not a result. So the confident
claim is that Engram matters for string and container answers, and the
number effect is suggestive but individually underpowered.

## The tension with gsm8k, unresolved

gsm8k answers are integers and showed no effect at all (+0.0008 over 1319
items, 120,000 verified gate-shut calls). Here integer answers drop ~12pp.
The two differ in more than the task: gsm8k ran through the chat template
with chain of thought, while this checkpoint copy carries no `chat_template`
so CRUXEval ran as raw few-shot completion. Pattern completion is exactly the
regime where an n-gram memory should matter most, so the prompt style is a
live confound for the cross-task comparison. Stated, not resolved.

## Ablation verification

| arm | Engram forward calls | mean contribution | max contribution | empty generations |
| --- | ---: | ---: | ---: | ---: |
| baseline | 16480 | 0.30961170 | 3.49387527 | 3 |
| ablated | 16480 | 0.00000000 | 0.00000000 | 0 |

The ablated arm's contribution is exactly 0.0 across all 16,480 calls -- the
all-False `token_mask` is applied by the shipped fused kernel itself, so this
is zero rather than small. Both runs reproduced pass@1 to four decimals.

---

# Phase arms: prefill-only vs decode-only (run 34732450349)

Same 800 items, same raw-completion prompt, four arms in one process. Unlike
the likelihood run, CRUXEval generates, so `prefill_only` / `decode_only` are
the **real engine phase** derived from `query_start_loc`, not a positional
proxy. `phase_mask_unavailable` is false for all four arms, so every number
below was measured under the mask it claims.

| arm | pass@1 | Δ vs baseline | only baseline correct | only arm correct | McNemar χ² | ~σ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.6512 | — | — | — | — | — |
| prefill_only | 0.6100 | −0.0413 | 52 | 19 | 14.423 | 3.8 |
| decode_only | 0.4850 | −0.1663 | 161 | 28 | 92.190 | 9.6 |
| ablated | 0.4888 | −0.1625 | 162 | 32 | 85.778 | 9.26 |

**Engram is a decode-time lookup.** Letting it act only during prefill
recovers almost the whole baseline (−0.04 vs the full −0.16); letting it act
only during decode is statistically indistinguishable from removing it
entirely (0.4850 vs 0.4888, a 0.4pp gap on 800 items). That is the same
conclusion the likelihood run reached from the opposite direction — there
`decode_only` recovered a median 94.4% of the ablation loss — and it is worth
being explicit about why the two look inverted: the NLL arms are a
*positional* split of one prefill, where "decode_only" means Engram is open on
the scored tokens; here "decode_only" means Engram is open only while
generating, and shut over the whole two-shot prompt. Both say the same thing:
what matters is Engram acting on the tokens being predicted from, and shutting
it over the prompt costs nearly everything.

By answer kind, prefill_only is uniformly mild and decode_only tracks full
ablation closely:

| kind | items | baseline | prefill_only | decode_only | ablated |
| --- | ---: | ---: | ---: | ---: | ---: |
| str | 371 | 0.5768 | −0.0377 | −0.1725 | −0.1590 |
| container | 280 | 0.7214 | −0.0321 | −0.1714 | −0.1750 |
| number | 98 | 0.6633 | −0.0816 | −0.1429 | −0.1633 |
| bool/None | 49 | 0.7755 | −0.0408 | −0.1020 | −0.1020 |

## Ablation verification (phase run)

| arm | calls | mean contribution | max contribution | phase mask | empty gens |
| --- | ---: | ---: | ---: | --- | ---: |
| baseline | 16480 | 0.28390 | 0.95459 | ok | 3 |
| ablated | 16480 | 0.00000 | 0.00000 | ok | 0 |
| prefill_only | 16488 | 0.00362 | 0.79323 | ok | 8 |
| decode_only | 16480 | 0.33595 | 0.80077 | ok | 0 |

The contribution column is itself the check on the phase split: `prefill_only`
averages 0.0036 because the overwhelming majority of forward calls in a
generation run are decode steps, where that arm holds the gate shut;
`decode_only` averages *above* baseline because its non-zero calls are exactly
the decode steps, with the diluting prefill calls contributing zero.

## Caveat on the baseline

This run's baseline is 0.6512 against 0.6388 in the two earlier paired runs
(which agreed with each other to four decimals). Same temperature 0.0, same
items; the four-arm run batches differently, and the checkpoint is not
bitwise-deterministic across batch shapes. All comparisons above are within
this run and paired per item, so the deltas are unaffected, but do not read
the 1.2pp baseline shift as a measurement of anything.
