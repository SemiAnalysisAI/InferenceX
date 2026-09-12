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
