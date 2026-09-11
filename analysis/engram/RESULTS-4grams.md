# Engram gate activation: top suffix 4-grams

DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash`, MXFP4 routed experts),
vLLM `deepseekv41-flash-0909`, TP=8 on 2x H100 nodes (8 GPUs each).
Run [34610313633](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34610313633),
2026-09-11, 4h51m wall-clock.

## Method

The Engram gate is computed inside a fused Triton kernel and consumed in the
same store (`hidden + gate * value`), so it is not observable from a hook.
`analysis/engram/gate_probe.py` wraps `Engram.forward` and recomputes it in
torch from the module's public tensors, mirroring the kernel's arithmetic:

```
hidden_rms = rsqrt(mean(hidden^2) + eps)
key_rms    = rsqrt(mean(key^2) + eps)
dot        = sum(hidden * q * k * key) * hidden_rms * key_rms * dim^-0.5
gate       = sigmoid(sign(dot) * sqrt(max(|dot|, clamp)))
```

Per token the gate is the **maximum over the four hyper-connection copies**
(an earlier run averaged them, which suppressed the peak ~3x). Each
domain is read as 3584-token chunks, prefilled with `max_tokens=1`. Within
each (domain, layer) the top 1% of gate values are taken as strong, and every
strong position is attributed to the 2-, 3-, and 4-gram ending at it. Rows
below are ranked by mean gate with a minimum of 3 occurrences; only the
4-gram tier is shown here.

**Corpus: 434.6M tokens over 16 domains,
English and Chinese only.**

| domain | tokens | source |
| --- | --- | --- |
| `wiki` | 2.40M | Salesforce/wikitext, wikitext-2-raw-v1 |
| `chat` | 35.84M | HuggingFaceH4/ultrachat_200k (train_sft) |
| `math` | 1.18M | openai/gsm8k (main) |
| `code_mbpp` | 0.03M | google-research-datasets/mbpp (full) |
| `wiki_zh` | 35.84M | wikimedia/wikipedia 20231101.zh |
| `chat_zh` | 35.84M | BelleGroup/train_1M_CN |
| `web_zh` | 35.84M | HuggingFaceFW/fineweb-2 cmn_Hani |
| `code_python` | 35.84M | code_search_net (python) |
| `code_javascript` | 30.84M | code_search_net (javascript) |
| `code_java` | 35.84M | code_search_net (java) |
| `code_go` | 35.84M | code_search_net (go) |
| `code_php` | 35.84M | code_search_net (php) |
| `code_ruby` | 5.87M | code_search_net (ruby) |
| `web` | 35.84M | HuggingFaceFW/fineweb-edu sample-10BT |
| `wiki_full` | 35.84M | wikimedia/wikipedia 20231101.en |
| `math_web` | 35.84M | open-web-math/open-web-math |

## Caveats

1. **Rankings are the union of three per-shard top-50 lists**, each with its
   own threshold, not a strict global top-N. An n-gram strong in one shard can
   outrank one moderately strong in all three.
2. **The `chat` domains carry code.** UltraChat and Belle both contain code
   snippets, so those tables are not clean dialogue and do not compare
   directly to a DailyDialog baseline (DailyDialog is script-based and no
   longer loadable).
3. **`code_mbpp` and `math` are small** (0.03M and 1.18M tokens) and exhaust
   long before the chunk cap; their tables are thin by construction.
4. Chinese tokenizes at roughly 1-1.5 characters per token, so a Chinese
   4-gram spans about four characters -- morpheme-level, not the multi-word
   phrases seen in the English tables.
5. Rows whose text decodes mid-character (a lone replacement glyph) are
   tokenizer-boundary artifacts of non-ASCII literals, not real n-grams.

## Summary: what each layer selects for

Layer 1 fires on rare surface strings -- proper nouns, unusual identifiers,
continuations with no parametric structure. Layer 14 fires on templated
structure. `math` shows the split cleanly: layer 1 takes the incidental nouns
of a word problem, layer 14 takes the schema (`three times as many`,
`7 years older than`). MBPP repeats it: identifiers at layer 1,
`Write a function to` at layer 14.

Two specific results:

- **ISBN publisher prefixes dominate `wiki_zh` layer 14** (`-7167-`, `3-406-`,
  `0-393-`, `0-345-`), and `0-393-` independently reaches the top of English
  `web` layer 14. These are digit sequences with no parametric structure --
  obtainable only by memorization.
- **Chinese layer 14 is boilerplate**: ICP license numbers,
  `未经授权禁止`, `本文仅代表作者`. The same templated-structure behavior as
  English, in a corpus-specific vocabulary.

## Top 10 overall

Highest mean gate of any 4-gram in the run, across all 16 domains and both
layers. Every one is code or markup: a fixed idiom the tokenizer splits into
several pieces, where the next piece is fully determined by the ones before.

| # | gate | count | domain / layer | 4-gram |
| --: | --: | --: | --- | --- |
| 1 | 1.0000 | 3 | `code_ruby` / 0 | `http://id.loc` |
| 2 | 1.0000 | 3 | `code_java` / 0 | `ProcessInstanceDefiniton` |
| 3 | 0.9999 | 5 | `code_go` / 0 | ` fakaʻiniton` |
| 4 | 0.9995 | 5 | `math_web` / 0 | `@@ -1,` |
| 5 | 0.9981 | 5 | `code_go` / 0 | `\treturn func(_ context` |
| 6 | 0.9957 | 17 | `code_python` / 0 | ` \| QtCore.Q` |
| 7 | 0.9393 | 5 | `math_web` / 1 | `.com/questions/156` |
| 8 | 0.9382 | 6 | `code_go` / 0 | `.Getenv("RED` |
| 9 | 0.9317 | 19 | `code_python` / 0 | `        response = ur` |
| 10 | 0.9315 | 42 | `code_php` / 0 | `' => $in` |

The PHP leader `' => $in` is an array-literal fragment; the three
`response = ur` rows are the same `urllib` call at three indentation depths,
which the tokenizer makes into three distinct 4-grams. `:=PCGroup([` is GAP
computer-algebra syntax from a maths forum.

## Notable entries

Hand-picked from the tables above, because what the gate opens on is easier
to see in specific cases than in aggregate. These are not the strongest
rows -- they are the legible ones.

| gate | count | domain / layer | 4-gram | what it is |
| --: | --: | --- | --- | --- |
| 0.4387 | 3 | `wiki` / 1 | ` " Run Run Rudolph` | Chuck Berry, 1958. |
| 0.5060 | 3 | `wiki` / 1 | `able Kimmy Schmidt` | Unbreakable Kimmy Schmidt, caught mid-word. |
| 0.4092 | 3 | `wiki` / 1 | `ane Clown Pos` | Insane Clown Posse -- gate opens inside a word. |
| 0.5489 | 3 | `wiki` / 1 | ` , Super Mario Land` | Game Boy, 1989. |
| 0.6037 | 5 | `wiki_full` / 1 | ` Sabbath Bloody Sabbath` | Black Sabbath, 1973. |
| 0.7682 | 12 | `web` / 1 | ` Johannes Gutenberg University` | Mainz. |
| 0.7911 | 3 | `web` / 1 | ` All Rights Reserved.` | Boilerplate, fully determined. |
| 0.7598 | 3 | `web` / 1 | ` material from the Wikipedia` | Attribution boilerplate. |
| 0.8667 | 3 | `web_zh` / 1 | `免责声明】本文` | Chinese disclaimer header. |
| 0.7326 | 10 | `web_zh` / 1 | `本文僅代表作者` | 'views are the author's own', traditional script. |
| 0.6592 | 12 | `web_zh` / 1 | ` 未经授权禁止` | 'reproduction without authorisation prohibited'. |
| 0.6199 | 6 | `web_zh` / 1 | `Copyright 2010` | A year the model cannot guess, only recall. |
| 0.3174 | 54 | `web_zh` / 0 | `玄奘西游记` | Xuanzang's Journey to the West. |
| 0.3444 | 7 | `chat_zh` / 1 | `imedia.org/wikipedia` | A URL stem, mid-token. |
| 0.3330 | 9 | `code_javascript` / 0 | ` @namespace SugarNamespace` | A framework's docblock tag. |
| 0.9995 | 5 | `math_web` / 0 | `@@ -1,` | A unified-diff hunk header. |
| 0.9957 | 17 | `code_python` / 0 | ` \| QtCore.Q` | PyQt flag-OR idiom. |
| 0.9981 | 5 | `code_go` / 0 | `\treturn func(_ context` | Go middleware signature. |
| 1.0000 | 3 | `code_ruby` / 0 | `http://id.loc` | Library of Congress URI namespace. |

The pattern across all of them: a rare multi-token name whose later pieces
are unguessable from the model's weights but fully determined once the
earlier pieces are known. `ane Clown Pos` is the clearest case -- the gate
opens in the middle of a word, on a boundary that exists only because of how
the tokenizer split a band's name. The maths rows show the same mechanism on
invented props: once a GSM8K problem has said "pints of frozen", the next
token is not in doubt.

## Gate distribution

Measured over every gate value, not just the strong tail -- earlier runs
kept only the top 1%, so the observed maximum was an artifact of
selection. Per hyper-connection copy, combined across both shards.

| domain / layer | mean | q99 | q99.99 | max |
| --- | --: | --: | --: | --: |
| `code_go` / 0 | 0.0298 | 0.2125 | 0.2775 | **0.99999** |
| `code_python` / 0 | 0.0279 | 0.2075 | 0.2825 | **0.99998** |
| `code_javascript` / 0 | 0.0276 | 0.2125 | 0.2850 | **0.99998** |
| `code_java` / 0 | 0.0300 | 0.2225 | 0.2925 | **0.99998** |
| `code_php` / 0 | 0.0309 | 0.2175 | 0.2925 | **0.99998** |
| `code_ruby` / 0 | 0.0289 | 0.2075 | 0.2725 | **0.99998** |
| `math_web` / 0 | 0.0247 | 0.1925 | 0.2925 | **0.99953** |
| `chat` / 0 | 0.0229 | 0.1825 | 0.2825 | **0.99933** |
| `web` / 0 | 0.0234 | 0.1900 | 0.2925 | **0.99926** |
| `wiki_full` / 0 | 0.0252 | 0.2075 | 0.3025 | **0.99914** |
| `web_zh` / 0 | 0.0283 | 0.1925 | 0.2725 | **0.99612** |
| `math_web` / 1 | 0.0212 | 0.2050 | 0.7400 | **0.99103** |

The gate is a sigmoid, so it is bounded by 1 and approaches it only
asymptotically; the largest value measured here is **0.99999**. It is
shut almost everywhere -- mean 0.0243, 99th
percentile around 0.21 -- and opens hard on a thin tail. The copies are
markedly asymmetric at layer 1, where two of the four carry nearly all
of the signal.

## Reference-study domains

### `wiki`

2.40M tokens &middot; Salesforce/wikitext, wikitext-2-raw-v1

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 99 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.3076 | 4 | ` Tim and Chris Stam` |
| 2 | 0.3075 | 6 | ` Jermaine Dup` |
| 3 | 0.3055 | 3 | ` The Roud Folk` |
| 4 | 0.3024 | 3 | ` in Famous Fantastic Myst` |
| 5 | 0.3022 | 3 | ` =The Good Terror` |
| 6 | 0.3010 | 4 | ` in The Good Terror` |
| 7 | 0.2938 | 4 | `psilophodont` |
| 8 | 0.2937 | 7 | ` Stephen Thomas Erlew` |
| 9 | 0.2927 | 6 | ` " Mandatory Suicide` |
| 10 | 0.2910 | 3 | ` El Moutawak` |
| 11 | 0.2904 | 3 | `yanvapi Mosque` |
| 12 | 0.2903 | 3 | ` " Run Run Rudolph` |
| 13 | 0.2903 | 3 | ` a stegosaur` |
| 14 | 0.2899 | 4 | ` and Scleroderma` |
| 15 | 0.2892 | 3 | ` =Chasing Verm` |
| 16 | 0.2890 | 3 | ` Story of Patty Cake` |
| 17 | 0.2884 | 3 | `th Armored Cavalry` |
| 18 | 0.2873 | 5 | ` Tuojiangosaurus` |
| 19 | 0.2872 | 3 | ` van Giersbergen` |
| 20 | 0.2870 | 3 | `ley and Josh Wein` |
| 21 | 0.2857 | 3 | ` and Ghost of Sparta` |
| 22 | 0.2851 | 4 | ` , the Joint Typh` |
| 23 | 0.2849 | 8 | ` Yasunori Mits` |
| 24 | 0.2847 | 3 | ` " Casimir Pul` |
| 25 | 0.2842 | 3 | ` of The Holocaust Industry` |
| 26 | 0.2838 | 6 | ` Church of Christ Pant` |
| 27 | 0.2830 | 3 | ` songwriter Mariah Carey` |
| 28 | 0.2825 | 3 | ` , Lambeosaurus` |
| 29 | 0.2819 | 3 | ` Gharana Mog` |
| 30 | 0.2806 | 3 | ` James W. Carey` |
| 31 | 0.2802 | 3 | ` at Tittenhurst` |
| 32 | 0.2797 | 3 | ` and The Fame Monster` |
| 33 | 0.2797 | 3 | ` the Airborne Cemetery` |
| 34 | 0.2797 | 3 | ` Feet in the Clouds` |
| 35 | 0.2796 | 4 | `8th Scottish Rif` |
| 36 | 0.2794 | 3 | `able Kimmy Schmidt` |
| 37 | 0.2782 | 3 | ` God of War Saga` |
| 38 | 0.2780 | 4 | ` – North Rim Parkway` |
| 39 | 0.2779 | 5 | ` at Anzac Cove` |
| 40 | 0.2779 | 3 | ` the Imperial Camel Brigade` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 92 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.7801 | 3 | ` =The Good Terror` |
| 2 | 0.7096 | 3 | `" Far Away Places` |
| 3 | 0.6529 | 3 | ` on the Tracks` |
| 4 | 0.6466 | 4 | ` " Unnatural Selection` |
| 5 | 0.6318 | 3 | ` for " Man Down` |
| 6 | 0.6270 | 3 | ` " Oath Sign` |
| 7 | 0.6200 | 12 | ` the Pulse of Morning` |
| 8 | 0.6114 | 3 | `zenjammer Kids` |
| 9 | 0.6076 | 3 | ` It All Back Home` |
| 10 | 0.5896 | 3 | ` =The Coldrum` |
| 11 | 0.5867 | 3 | ` =Chasing Verm` |
| 12 | 0.5789 | 11 | ` The Sixth Extinction` |
| 13 | 0.5771 | 3 | ` The Roud Folk` |
| 14 | 0.5619 | 6 | ` Life Is Worth Living` |
| 15 | 0.5598 | 3 | ` at Tittenhurst` |
| 16 | 0.5521 | 3 | ` , Memory Almost Full` |
| 17 | 0.5505 | 4 | ` Ramnagar Fort` |
| 18 | 0.5489 | 3 | ` , Super Mario Land` |
| 19 | 0.5397 | 6 | ` the Medway Meg` |
| 20 | 0.5336 | 3 | ` Everglades Agricultural` |
| 21 | 0.5290 | 12 | ` " One Sweet Day` |
| 22 | 0.5263 | 3 | ` , " My Happiness` |
| 23 | 0.5258 | 5 | ` Casualties of Cool` |
| 24 | 0.5213 | 10 | ` " Back to Tennessee` |
| 25 | 0.5174 | 3 | ` The Rocky Mountain Horse` |
| 26 | 0.5171 | 3 | ` Let Me Be Mis` |
| 27 | 0.5078 | 5 | ` Am Not a Robot` |
| 28 | 0.5060 | 3 | `able Kimmy Schmidt` |
| 29 | 0.5018 | 4 | ` " West End Girls` |
| 30 | 0.4985 | 4 | ` Harajuku Girls` |
| 31 | 0.4948 | 3 | ` =Not Quite Hollywood` |
| 32 | 0.4935 | 3 | ` =The corn cra` |
| 33 | 0.4932 | 3 | ` 61 Revisited` |
| 34 | 0.4872 | 5 | ` Beat of My Drum` |
| 35 | 0.4869 | 3 | ` Unbreakable Kim` |
| 36 | 0.4867 | 3 | ` Citadel of Fear` |
| 37 | 0.4826 | 5 | ` and Better Updated Uno` |
| 38 | 0.4821 | 3 | ` , " True Blue` |
| 39 | 0.4771 | 4 | ` The Stolen Eagle` |
| 40 | 0.4743 | 7 | ` Asthmatic Kitty` |

### `chat`

35.84M tokens &middot; HuggingFaceH4/ultrachat_200k (train_sft)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 97 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.7461 | 4 | ` RotateTransition(D` |
| 2 | 0.7212 | 4 | `.add(layers.D` |
| 3 | 0.6877 | 3 | ` if (!filter_var` |
| 4 | 0.6699 | 4 | ` Output: The file` |
| 5 | 0.5980 | 5 | ` or artistic creation found` |
| 6 | 0.5142 | 3 | `').magnificPopup` |
| 7 | 0.4072 | 4 | `, 123 Main` |
| 8 | 0.3419 | 3 | `Authenticated, (` |
| 9 | 0.3368 | 8 | `jwt_required` |
| 10 | 0.3330 | 3 | ` Today. MediLex` |
| 11 | 0.3301 | 3 | ` of a Christian Arist` |
| 12 | 0.3288 | 3 | ` Initiative for Consumer Hortic` |
| 13 | 0.3245 | 3 | `" by Sharon Salz` |
| 14 | 0.3239 | 5 | ` together with Jana Hau` |
| 15 | 0.3237 | 4 | `. Rashmi Chaud` |
| 16 | 0.3234 | 6 | `" by Jon Krak` |
| 17 | 0.3233 | 7 | `" by Eric Ries` |
| 18 | 0.3213 | 3 | `namespace FactorialCalculator` |
| 19 | 0.3183 | 3 | `://www.allrecipes` |
| 20 | 0.3177 | 4 | `The Cosmos Unve` |
| 21 | 0.3160 | 4 | `The Mindful Minute` |
| 22 | 0.3160 | 3 | `s: repeat(auto` |
| 23 | 0.3147 | 3 | ` and Voyager Therapeutics` |
| 24 | 0.3145 | 40 | ` J. Alfred Pru` |
| 25 | 0.3137 | 6 | ` in the Cellular Jail` |
| 26 | 0.3132 | 5 | ` html\nhtml(l` |
| 27 | 0.3131 | 3 | ` by Björn Kuh` |
| 28 | 0.3126 | 4 | `. Kathleen Doheny` |
| 29 | 0.3113 | 4 | ` done by Australian Prostate` |
| 30 | 0.3105 | 4 | ` "Christmas Cake Murder` |
| 31 | 0.3101 | 3 | ` Darlene Schuster` |
| 32 | 0.3097 | 4 | ` Park by Nancy Webster` |
| 33 | 0.3091 | 3 | ` Daniel J. Siegel` |
| 34 | 0.3091 | 3 | ` by Todd Pletcher` |
| 35 | 0.3090 | 3 | ` Cooks Corner Gour` |
| 36 | 0.3089 | 4 | ` Society of Emergency Radiology` |
| 37 | 0.3086 | 4 | `ure by Michelle Humphrey` |
| 38 | 0.3083 | 3 | `The Beauty of Bones` |
| 39 | 0.3082 | 7 | ` the Angkor Archaeological` |
| 40 | 0.3079 | 3 | ` Guide to Raising Adolescents` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 97 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.8137 | 3 | `Field, SubmitField` |
| 2 | 0.7326 | 3 | ` require('formidable` |
| 3 | 0.7317 | 3 | ` 'vuex-p` |
| 4 | 0.6926 | 3 | ` “Games at Twilight` |
| 5 | 0.6600 | 7 | `k.corpus` |
| 6 | 0.6577 | 4 | `cloud import WordCloud` |
| 7 | 0.6527 | 3 | ` new PHPMail` |
| 8 | 0.6469 | 4 | ` = SentimentInt` |
| 9 | 0.6432 | 3 | `from flask_wtf` |
| 10 | 0.6429 | 3 | `Bitter Sweet Symphony` |
| 11 | 0.6160 | 3 | `.githubusercontent.com/872` |
| 12 | 0.6154 | 3 | `4. CoSchedule` |
| 13 | 0.6021 | 3 | `Swallow Me Whole` |
| 14 | 0.5985 | 3 | `\nfrom textbl` |
| 15 | 0.5944 | 5 | `api.openweathermap` |
| 16 | 0.5879 | 6 | `2.0 feed` |
| 17 | 0.5870 | 5 | ` import SentimentInt` |
| 18 | 0.5855 | 4 | ` = canvas.Can` |
| 19 | 0.5801 | 3 | `(FlaskForm` |
| 20 | 0.5767 | 3 | ` "The Big Sick` |
| 21 | 0.5757 | 3 | ` com.google.api.services` |
| 22 | 0.5634 | 3 | ` Top of the Rock` |
| 23 | 0.5616 | 4 | `.detectMultiScale` |
| 24 | 0.5612 | 3 | `. Ubersuggest` |
| 25 | 0.5489 | 4 | ` "Let Girls Learn` |
| 26 | 0.5446 | 3 | `Like Water for Chocolate` |
| 27 | 0.5429 | 3 | `obiologist Maria McN` |
| 28 | 0.5423 | 3 | ` Salinger Year` |
| 29 | 0.5373 | 5 | ` the Angels in Adoption` |
| 30 | 0.5351 | 5 | `litosphere Progressive Poem` |
| 31 | 0.5348 | 4 | ` The Art of Charm` |
| 32 | 0.5334 | 3 | ` '@react-native-fire` |
| 33 | 0.5310 | 4 | `appier with Gret` |
| 34 | 0.5298 | 3 | ` Where Are Your Keys` |
| 35 | 0.5274 | 3 | `, distinctUntilChanged` |
| 36 | 0.5229 | 4 | `The Farmer's Bride` |
| 37 | 0.5223 | 3 | ` Roselinde Torres` |
| 38 | 0.5205 | 5 | `y at the Bat` |
| 39 | 0.5200 | 4 | ` All The Pretty Horses` |
| 40 | 0.5182 | 3 | `ow Meow Tweet` |

### `math`

1.18M tokens &middot; openai/gsm8k (main)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 97 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2755 | 3 | ` The Fancy Salon` |
| 2 | 0.2677 | 4 | ` Yummy Dog Kib` |
| 3 | 0.2625 | 3 | ` 5 medium pizz` |
| 4 | 0.2608 | 4 | `, the snail kite` |
| 5 | 0.2572 | 9 | `g of packing peanuts` |
| 6 | 0.2492 | 7 | ` Shark Bite Cove` |
| 7 | 0.2437 | 3 | ` cans of cherry soda` |
| 8 | 0.2407 | 3 | ` number of red marbles` |
| 9 | 0.2391 | 3 | ` 4 small pizz` |
| 10 | 0.2378 | 3 | ` number of ripe mango` |
| 11 | 0.2378 | 3 | ` red and blue marbles` |
| 12 | 0.2372 | 6 | ` number of blue marbles` |
| 13 | 0.2368 | 8 | ` How many jellybeans` |
| 14 | 0.2366 | 3 | ` boxes of birdseed` |
| 15 | 0.2365 | 3 | ` pints of frozen yogurt` |
| 16 | 0.2364 | 4 | ` hawksbill turtles` |
| 17 | 0.2359 | 4 | ` pint of frozen yogurt` |
| 18 | 0.2355 | 3 | ` the Taco Grande` |
| 19 | 0.2353 | 3 | `: Armoured Command` |
| 20 | 0.2346 | 4 | ` boxes of Graham crackers` |
| 21 | 0.2342 | 4 | ` as many blue marbles` |
| 22 | 0.2335 | 3 | ` number of pinecones` |
| 23 | 0.2333 | 3 | ` play the alto sax` |
| 24 | 0.2333 | 3 | ` hour, Stephen travels` |
| 25 | 0.2331 | 5 | ` as many water balloons` |
| 26 | 0.2331 | 3 | ` the two smart telev` |
| 27 | 0.2325 | 3 | ` number of green turtles` |
| 28 | 0.2324 | 3 | ` 4 tiger encl` |
| 29 | 0.2323 | 3 | `2000 pinecones` |
| 30 | 0.2322 | 3 | `60>>60 marbles` |
| 31 | 0.2309 | 3 | ` car's gas mileage` |
| 32 | 0.2305 | 4 | ` bag of jellybeans` |
| 33 | 0.2303 | 3 | `12>>12 chocol` |
| 34 | 0.2291 | 3 | ` the Bobbit worm` |
| 35 | 0.2283 | 3 | `onglow Orchard` |
| 36 | 0.2281 | 3 | ` as many jellybeans` |
| 37 | 0.2279 | 3 | `. Wilsborough` |
| 38 | 0.2279 | 3 | `60>>60 oranges` |
| 39 | 0.2275 | 3 | ` marbles as red marbles` |
| 40 | 0.2268 | 3 | ` bar of steel weighs` |

#### Layer 14 (`layer_hash_index` 1)

Showing 21 of 21 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.6582 | 3 | ` three times as many` |
| 2 | 0.2785 | 9 | ` twice as old as` |
| 3 | 0.2642 | 10 | ` times as old as` |
| 4 | 0.2546 | 3 | ` times the number of` |
| 5 | 0.2489 | 4 | ` in the fourth basket` |
| 6 | 0.2433 | 3 | ` of the remaining pizza` |
| 7 | 0.2370 | 3 | ` as many water balloons` |
| 8 | 0.2338 | 3 | `The sum of the` |
| 9 | 0.2309 | 3 | ` twice as many frogs` |
| 10 | 0.2268 | 3 | ` the first three baskets` |
| 11 | 0.2252 | 3 | ` crackers for her scout` |
| 12 | 0.2245 | 3 | ` client will pay Baylor` |
| 13 | 0.2229 | 3 | ` throw, Christine throws` |
| 14 | 0.2216 | 3 | ` The Kickers scored` |
| 15 | 0.2181 | 3 | `s Woodworking LLC` |
| 16 | 0.2156 | 3 | ` of a single necklace` |
| 17 | 0.2149 | 3 | `2 hours of overtime` |
| 18 | 0.2130 | 3 | ` for her scout troop` |
| 19 | 0.2123 | 3 | ` room with tall mirrors` |
| 20 | 0.2116 | 3 | `*2=<<` |
| 21 | 0.2083 | 3 | ` is older than Bobby` |

### `code_mbpp`

0.03M tokens &middot; google-research-datasets/mbpp (full)

#### Layer 1 (`layer_hash_index` 0)

Showing 2 of 2 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2873 | 3 | ` SumOfPrimeDivisors` |
| 2 | 0.2399 | 5 | ` = list(filter(lambda` |

#### Layer 14 (`layer_hash_index` 1)

Showing 10 of 10 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.8119 | 3 | ` calculate the sum of` |
| 2 | 0.7246 | 7 | ` find the sum of` |
| 3 | 0.6604 | 4 | ` the sum of all` |
| 4 | 0.6595 | 5 | ` to check if the` |
| 5 | 0.6270 | 178 | `Write a function to` |
| 6 | 0.5925 | 4 | ` to find sum of` |
| 7 | 0.5882 | 31 | ` function to find the` |
| 8 | 0.5693 | 7 | ` to check whether the` |
| 9 | 0.5643 | 3 | ` count the number of` |
| 10 | 0.4001 | 3 | `Write a function that` |

## Chinese

### `wiki_zh`

35.84M tokens &middot; wikimedia/wikipedia 20231101.zh

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 90 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.3287 | 3 | `《太公兵法` |
| 2 | 0.3282 | 9 | ` 《中国伦理学史` |
| 3 | 0.3199 | 3 | `李卫公兵法` |
| 4 | 0.3186 | 3 | `《再说一次我爱你` |
| 5 | 0.3166 | 3 | `《七子之歌` |
| 6 | 0.3153 | 3 | `《花舞大唐` |
| 7 | 0.3144 | 14 | ` of Far Eastern Antiqu` |
| 8 | 0.3141 | 5 | `《伶人往事` |
| 9 | 0.3139 | 3 | `。\n田仲一成` |
| 10 | 0.3127 | 11 | `《Big Comic Spir` |
| 11 | 0.3101 | 3 | `\n潘震宙` |
| 12 | 0.3088 | 3 | ` A Vedic Concord` |
| 13 | 0.3085 | 4 | ` Angiosperm Phylogen` |
| 14 | 0.3073 | 3 | `发的《深化党和国家` |
| 15 | 0.3064 | 5 | `《反满抗日` |
| 16 | 0.3062 | 33 | `《伊索寓言` |
| 17 | 0.3062 | 3 | ` Fyodor Dost` |
| 18 | 0.3062 | 3 | `《中国海洋大学学报` |
| 19 | 0.3061 | 3 | `《陈云文选` |
| 20 | 0.3056 | 4 | `饮，不好侈` |
| 21 | 0.3056 | 3 | `《銀河擂台` |
| 22 | 0.3053 | 11 | `、一级解放勋章` |
| 23 | 0.3053 | 3 | `副主任陳佐洱` |
| 24 | 0.3050 | 4 | `胡锦涛文选` |
| 25 | 0.3044 | 3 | `家王秀杞` |
| 26 | 0.3044 | 3 | `村田雄二郎` |
| 27 | 0.3040 | 5 | `\n 《春光乍` |
| 28 | 0.3037 | 3 | `、白寿彝` |
| 29 | 0.3037 | 3 | `《互联网新闻信息服务` |
| 30 | 0.3030 | 249 | `台、高清翡翠` |
| 31 | 0.3029 | 3 | `《金粉世家` |
| 32 | 0.3027 | 6 | `。《伊索寓言` |
| 33 | 0.3027 | 3 | `谓遵祖宗之法` |
| 34 | 0.3025 | 4 | `《中国有色金属学报` |
| 35 | 0.3024 | 5 | ` Ed. Alan Dund` |
| 36 | 0.3020 | 5 | `《大品般若` |
| 37 | 0.3019 | 8 | `刘心武揭秘` |
| 38 | 0.3019 | 17 | `\n一级解放勋章` |
| 39 | 0.3018 | 3 | ` TextExtractingVisitor` |
| 40 | 0.3017 | 3 | `《The Greatest Hits` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 84 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.9023 | 3 | `-7432-` |
| 2 | 0.8965 | 3 | `-8274-` |
| 3 | 0.8953 | 3 | `-7126-` |
| 4 | 0.8728 | 4 | `0-321-` |
| 5 | 0.8551 | 3 | `0-534-` |
| 6 | 0.8476 | 9 | `-8053-` |
| 7 | 0.8468 | 4 | `-7167-` |
| 8 | 0.8448 | 3 | `-3-937` |
| 9 | 0.8108 | 3 | `adnezzar` |
| 10 | 0.8076 | 3 | `-4039-` |
| 11 | 0.8065 | 3 | `-4000-` |
| 12 | 0.7998 | 4 | `-4051-` |
| 13 | 0.7963 | 3 | `-7603-` |
| 14 | 0.7935 | 3 | `-8032-` |
| 15 | 0.7904 | 3 | `-8020-` |
| 16 | 0.7836 | 21 | `0-393-` |
| 17 | 0.7757 | 22 | `3-406-` |
| 18 | 0.7726 | 8 | `0-316-` |
| 19 | 0.7453 | 3 | `-7658-` |
| 20 | 0.7374 | 16 | `0-385-` |
| 21 | 0.7319 | 13 | `0-684-` |
| 22 | 0.7240 | 19 | `0-375-` |
| 23 | 0.7190 | 10 | `0-345-` |
| 24 | 0.7040 | 4 | `3-423-` |
| 25 | 0.7002 | 6 | `0-671-` |
| 26 | 0.6892 | 6 | `-8021-` |
| 27 | 0.6760 | 3 | `-8147-` |
| 28 | 0.6730 | 15 | `0-674-` |
| 29 | 0.6617 | 3 | `-7567-` |
| 30 | 0.6614 | 22 | `0-471-` |
| 31 | 0.6537 | 6 | `-8062-` |
| 32 | 0.6536 | 3 | ` Septuagint` |
| 33 | 0.6526 | 3 | ` 3-938` |
| 34 | 0.6513 | 9 | `0-312-` |
| 35 | 0.6510 | 5 | `-8018-` |
| 36 | 0.6417 | 26 | `0-201-` |
| 37 | 0.6329 | 3 | `《北京市民宣言` |
| 38 | 0.6243 | 4 | `-8014-` |
| 39 | 0.6233 | 3 | `-7868-` |
| 40 | 0.6210 | 3 | `0-571-` |

### `chat_zh`

35.84M tokens &middot; BelleGroup/train_1M_CN

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 79 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.8597 | 3 | `(random.choice(string.d` |
| 2 | 0.7814 | 4 | ` random.choice(string.d` |
| 3 | 0.7004 | 4 | `dum et males` |
| 4 | 0.3067 | 3 | ` by Paul Pairet` |
| 5 | 0.3058 | 10 | `：Ian Goodfellow` |
| 6 | 0.3052 | 3 | ` '.join(word[::-` |
| 7 | 0.3036 | 18 | `Don't Stop Belie` |
| 8 | 0.2999 | 4 | ` by Gillian Flynn` |
| 9 | 0.2984 | 37 | `def generate_password(length` |
| 10 | 0.2984 | 5 | `和Peter Norvig` |
| 11 | 0.2979 | 8 | `包括《命运交响` |
| 12 | 0.2978 | 4 | `和Donna Strick` |
| 13 | 0.2972 | 5 | `()\nreversed_words` |
| 14 | 0.2969 | 3 | `曲》、《命运交响` |
| 15 | 0.2962 | 3 | ` '.join(sorted_words` |
| 16 | 0.2959 | 3 | `如《命运交响` |
| 17 | 0.2945 | 12 | `和 Peter Norvig` |
| 18 | 0.2944 | 4 | `Dr. Grace Augustine` |
| 19 | 0.2944 | 4 | `读史使人明智` |
| 20 | 0.2941 | 6 | `：Jean-Pierre Sau` |
| 21 | 0.2934 | 7 | `\ndef capitalize_words` |
| 22 | 0.2925 | 6 | ` and Peter Norvig` |
| 23 | 0.2925 | 3 | `》、《落跑新娘` |
| 24 | 0.2923 | 4 | `《东篱乐` |
| 25 | 0.2920 | 35 | `夜泊牛渚` |
| 26 | 0.2904 | 23 | `瑜伽、阴瑜伽` |
| 27 | 0.2901 | 16 | `Rebecca Lat` |
| 28 | 0.2900 | 16 | ` = [word[::-` |
| 29 | 0.2900 | 7 | `：Rainer Weiss` |
| 30 | 0.2899 | 3 | `（Daisy Buchanan` |
| 31 | 0.2885 | 9 | `io和Aaron Cour` |
| 32 | 0.2885 | 7 | `和Akira Yosh` |
| 33 | 0.2883 | 3 | `简介：Dom Cobb` |
| 34 | 0.2878 | 3 | `女人Daisy Buchanan` |
| 35 | 0.2873 | 3 | `\ndef insertion_sort` |
| 36 | 0.2872 | 23 | `瑜伽、流瑜伽` |
| 37 | 0.2871 | 40 | `斯（Tim Robbins` |
| 38 | 0.2869 | 4 | `电影《海上钢琴` |
| 39 | 0.2868 | 14 | `的《命运交响` |
| 40 | 0.2867 | 9 | `, and Aaron Cour` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 80 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.7740 | 7 | ` email.mime.text` |
| 2 | 0.7542 | 6 | `pus import wordnet` |
| 3 | 0.7148 | 3 | `\nfrom textbl` |
| 4 | 0.6704 | 7 | `cloud import WordCloud` |
| 5 | 0.6664 | 6 | ` import SentimentInt` |
| 6 | 0.6224 | 13 | `ize import word_token` |
| 7 | 0.6028 | 25 | `k.corpus` |
| 8 | 0.5977 | 4 | `. Dirt Candy` |
| 9 | 0.5573 | 3 | `izer.fit_on_text` |
| 10 | 0.5560 | 3 | ` elit. Donec` |
| 11 | 0.5522 | 3 | `sparse_categorical` |
| 12 | 0.5511 | 11 | `When Harry Met Sally` |
| 13 | 0.5435 | 9 | ` elit. Nulla` |
| 14 | 0.5138 | 6 | `_20newsgroups` |
| 15 | 0.4905 | 4 | ` "Say You Won` |
| 16 | 0.4895 | 4 | `IMEMultipart` |
| 17 | 0.4723 | 13 | `pus import stopwords` |
| 18 | 0.4669 | 3 | `Subodh Gupta` |
| 19 | 0.4668 | 4 | ` MIMEText` |
| 20 | 0.4447 | 3 | ` Captain! My Captain` |
| 21 | 0.4412 | 10 | `NetLemmatizer` |
| 22 | 0.4402 | 4 | `.sentiment import` |
| 23 | 0.4341 | 3 | `《Hungry Hearts` |
| 24 | 0.4258 | 3 | `. Rock in Rio` |
| 25 | 0.4242 | 3 | `etrics.pairwise` |
| 26 | 0.4202 | 5 | ` mean_squared_error` |
| 27 | 0.4178 | 3 | `. Superiority Burger` |
| 28 | 0.4173 | 4 | `fidfTransformer` |
| 29 | 0.4138 | 7 | `Can't Help Falling` |
| 30 | 0.4127 | 6 | ` = SentimentInt` |
| 31 | 0.4123 | 5 | ` elit. Nullam` |
| 32 | 0.4119 | 32 | `. Aliquam` |
| 33 | 0.4108 | 12 | `Viva la Vida` |
| 34 | 0.4082 | 18 | `Viva La Vida` |
| 35 | 0.4073 | 5 | `kelti Williamson` |
| 36 | 0.4054 | 45 | `ature_extraction.text` |
| 37 | 0.4023 | 3 | ` Mr & Mrs Bund` |
| 38 | 0.4002 | 12 | `《控方证人` |
| 39 | 0.3988 | 3 | `. 《奇特的一生` |
| 40 | 0.3910 | 3 | `uca di Beppo` |

### `web_zh`

35.84M tokens &middot; HuggingFaceFW/fineweb-2 cmn_Hani

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 78 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.5897 | 12 | ` Suites by Hilton` |
| 2 | 0.3652 | 11 | `cgi.blog.ro` |
| 3 | 0.3599 | 4 | `再想想期中考` |
| 4 | 0.3573 | 10 | `&list=PL` |
| 5 | 0.3328 | 4 | `剑】：素还真` |
| 6 | 0.3267 | 78 | `Submit()">登` |
| 7 | 0.3174 | 54 | `玄奘西游记` |
| 8 | 0.3174 | 3 | `，常念恭敬` |
| 9 | 0.3142 | 8 | `（Juliana Schro` |
| 10 | 0.3092 | 6 | `记者杨淑伶` |
| 11 | 0.3076 | 3 | `uth and Bruce Rapp` |
| 12 | 0.3065 | 62 | `《虎符传奇` |
| 13 | 0.3065 | 4 | `- 《风流书生` |
| 14 | 0.3060 | 5 | `《卿本佳人` |
| 15 | 0.3058 | 6 | `�记者曹馥` |
| 16 | 0.3056 | 7 | `istic and Roman Sparta` |
| 17 | 0.3053 | 7 | `The Spaghetti Incident` |
| 18 | 0.3042 | 3 | `师张旭铠` |
| 19 | 0.3030 | 22 | `记者丁彦伶` |
| 20 | 0.3028 | 3 | `关于人民公社若干问题的` |
| 21 | 0.3025 | 5 | `互联网+信用三农` |
| 22 | 0.3018 | 24 | `《伊索寓言` |
| 23 | 0.3012 | 9 | `b/824684` |
| 24 | 0.3008 | 4 | ` Handbook of Greek Mythology` |
| 25 | 0.2987 | 13 | `／记者郑朝阳` |
| 26 | 0.2985 | 17 | `.victorymedical` |
| 27 | 0.2983 | 6 | `：)\nFrank Dau` |
| 28 | 0.2976 | 4 | `下列两种颜色中选择` |
| 29 | 0.2975 | 67 | `电影《明日之歌` |
| 30 | 0.2969 | 3 | `嗎？\nMatthew Berger` |
| 31 | 0.2967 | 3 | `、营业性歌舞` |
| 32 | 0.2966 | 3 | `／记者凌筠` |
| 33 | 0.2964 | 8 | `维（Christopher Ree` |
| 34 | 0.2959 | 3 | `又称万荣温泉` |
| 35 | 0.2949 | 6 | `CEO古永锵` |
| 36 | 0.2948 | 3 | `. Gino Rig` |
| 37 | 0.2947 | 8 | `记者姜宜菁` |
| 38 | 0.2947 | 4 | ` 中国互联网视听` |
| 39 | 0.2946 | 3 | `�记者郑朝阳` |
| 40 | 0.2945 | 3 | `记者程炳璋` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 73 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.8667 | 3 | `免责声明】本文` |
| 2 | 0.8474 | 12 | `0-2015` |
| 3 | 0.8099 | 10 | `沪ICP备150` |
| 4 | 0.7670 | 3 | ` © 2002` |
| 5 | 0.7560 | 3 | `】本文仅代表` |
| 6 | 0.7553 | 10 | ` 2007-` |
| 7 | 0.7539 | 20 | ` 2010-` |
| 8 | 0.7531 | 12 | `复制或建立镜像` |
| 9 | 0.7326 | 10 | `本文僅代表作者` |
| 10 | 0.7292 | 6 | `微信扫一扫分享` |
| 11 | 0.7157 | 87 | `京ICP备100` |
| 12 | 0.7034 | 6 | `代表作者个人观点` |
| 13 | 0.7027 | 12 | `本文仅代表作者` |
| 14 | 0.7023 | 11 | `ICP备150037` |
| 15 | 0.7021 | 3 | `0-2014` |
| 16 | 0.6950 | 5 | ` 2002-` |
| 17 | 0.6930 | 3 | `《大城小事` |
| 18 | 0.6721 | 7 | `：本文仅代表` |
| 19 | 0.6673 | 5 | ` 1996-` |
| 20 | 0.6623 | 4 | `粤ICP备090` |
| 21 | 0.6619 | 4 | ` 2009-` |
| 22 | 0.6597 | 7 | `僅代表作者本人` |
| 23 | 0.6592 | 12 | ` 未经授权禁止` |
| 24 | 0.6591 | 6 | ` 2006-` |
| 25 | 0.6580 | 3 | ` Processed in ` |
| 26 | 0.6505 | 7 | ` 2012-` |
| 27 | 0.6392 | 3 | ` © 2000` |
| 28 | 0.6380 | 3 | `沪ICP备102` |
| 29 | 0.6363 | 47 | `影片《老大不小` |
| 30 | 0.6345 | 6 | ` 2004-` |
| 31 | 0.6307 | 8 | `（Juliana Schro` |
| 32 | 0.6253 | 4 | ` 2005-` |
| 33 | 0.6214 | 8 | ` you agree to our` |
| 34 | 0.6199 | 6 | `Copyright 2010` |
| 35 | 0.6179 | 3 | `「光之穹` |
| 36 | 0.6173 | 3 | `《蜜桃成熟` |
| 37 | 0.6097 | 38 | `记者郭庚儒` |
| 38 | 0.6058 | 3 | `森林公园内的琵琶湖` |
| 39 | 0.6017 | 5 | `，未经授权禁止` |
| 40 | 0.5996 | 37 | ` © 2014` |

## Code (CodeSearchNet)

### `code_python`

35.84M tokens &middot; code_search_net (python)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 73 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.9957 | 17 | ` \| QtCore.Q` |
| 2 | 0.9317 | 19 | `        response = ur` |
| 3 | 0.9314 | 6 | `    response = ur` |
| 4 | 0.9309 | 13 | `            response = ur` |
| 5 | 0.9238 | 22 | `get_context_data(**` |
| 6 | 0.8863 | 3 | `        with open(re` |
| 7 | 0.8862 | 4 | `:\n            progress_c` |
| 8 | 0.8857 | 11 | ` end_time=end` |
| 9 | 0.8855 | 7 | ` object\n\n        Rais` |
| 10 | 0.8719 | 6 | `.loads(json.d` |
| 11 | 0.8716 | 8 | `'.format(os.get` |
| 12 | 0.8707 | 13 | `:\n                progress_c` |
| 13 | 0.8703 | 19 | ` if self.verb` |
| 14 | 0.8660 | 4 | `Application.setAttribute(Q` |
| 15 | 0.8651 | 8 | ` 'end_date':` |
| 16 | 0.8623 | 4 | `().split('T` |
| 17 | 0.8609 | 17 | ` pdb; p` |
| 18 | 0.8576 | 3 | ` dictionary\n\n    Rais` |
| 19 | 0.8540 | 6 | ` = ctypes.p` |
| 20 | 0.8515 | 4 | ` cluster\n\n        Rais` |
| 21 | 0.8513 | 9 | ` end_time = end` |
| 22 | 0.8482 | 10 | ` if self.Verb` |
| 23 | 0.8446 | 74 | ` = ctypes.P` |
| 24 | 0.8445 | 13 | ` None\n\n    Rais` |
| 25 | 0.8439 | 7 | ` logging.disable(log` |
| 26 | 0.8415 | 4 | ` objects\n\n        Rais` |
| 27 | 0.8357 | 8 | ` None\n\n        Rais` |
| 28 | 0.8342 | 4 | `_otp_se` |
| 29 | 0.8334 | 3 | ` seconds\n\n    Rais` |
| 30 | 0.8310 | 6 | ` "--quiet", action` |
| 31 | 0.8271 | 13 | `)\n    except File` |
| 32 | 0.8245 | 6 | `        ctypes.p` |
| 33 | 0.8242 | 8 | `)\n            except File` |
| 34 | 0.8219 | 36 | ` None\n        Rais` |
| 35 | 0.8216 | 149 | ` value\n\n        Rais` |
| 36 | 0.8212 | 3 | `)\n                except File` |
| 37 | 0.8208 | 27 | `)\n        except File` |
| 38 | 0.8152 | 9 | `TENSORBO` |
| 39 | 0.8138 | 4 | ` self.setAttribute(Q` |
| 40 | 0.8126 | 6 | ` found\n\n        Rais` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 86 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.9112 | 3 | `.com/questions/262` |
| 2 | 0.8675 | 3 | `.com/questions/180` |
| 3 | 0.8418 | 5 | `aginate_quer` |
| 4 | 0.8332 | 3 | ` 0x101` |
| 5 | 0.8231 | 3 | ` of the Auto Scaling` |
| 6 | 0.8205 | 3 | `.ly/2m` |
| 7 | 0.8181 | 3 | `.com/a/131` |
| 8 | 0.8103 | 3 | `3166-1` |
| 9 | 0.8043 | 3 | ` import schedula` |
| 10 | 0.7963 | 4 | ` This is the main` |
| 11 | 0.7895 | 3 | `3-09-` |
| 12 | 0.7889 | 10 | ` getopt.getopt` |
| 13 | 0.7856 | 9 | `_params = __salt` |
| 14 | 0.7820 | 79 | ` '', 'changes':` |
| 15 | 0.7755 | 3 | `.com/questions/233` |
| 16 | 0.7755 | 3 | ` a string representing the` |
| 17 | 0.7740 | 5 | ` current = __salt` |
| 18 | 0.7663 | 5 | ` This function gets called` |
| 19 | 0.7639 | 4 | `2E6B` |
| 20 | 0.7631 | 5 | ` if we have a` |
| 21 | 0.7627 | 23 | `parse.ArgumentDefaults` |
| 22 | 0.7616 | 3 | `r.GetDriverByName` |
| 23 | 0.7582 | 3 | `        # Create the` |
| 24 | 0.7569 | 3 | ` # redirect to the` |
| 25 | 0.7540 | 3 | `1 for real tokens` |
| 26 | 0.7528 | 4 | ` L{xmantissa` |
| 27 | 0.7514 | 3 | ` can be used to` |
| 28 | 0.7494 | 3 | `0-09-` |
| 29 | 0.7476 | 3 | ` ElementTree.iterparse` |
| 30 | 0.7439 | 12 | `0-04-` |
| 31 | 0.7438 | 3 | `5-07-` |
| 32 | 0.7425 | 3 | `    **Key Arguments` |
| 33 | 0.7418 | 4 | `5-06-` |
| 34 | 0.7403 | 9 | `3-01-` |
| 35 | 0.7403 | 4 | ` information on how to` |
| 36 | 0.7399 | 3 | `6-01-` |
| 37 | 0.7390 | 6 | `://uwsgi` |
| 38 | 0.7380 | 4 | `, sharex=True` |
| 39 | 0.7347 | 3 | `4-04-` |
| 40 | 0.7339 | 17 | ` >>> from dwave` |

### `code_javascript`

30.84M tokens &middot; code_search_net (javascript)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 82 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.8560 | 3 | ` = req.app.loc` |
| 2 | 0.8388 | 4 | `      if (!ne` |
| 3 | 0.8375 | 4 | `        if (!ne` |
| 4 | 0.8192 | 3 | ` data:image/*` |
| 5 | 0.8184 | 5 | ` storage.setItem(d` |
| 6 | 0.8129 | 11 | `err, rows)` |
| 7 | 0.7826 | 4 | ` process.env.GITH` |
| 8 | 0.7560 | 62 | `OpenLayers.G` |
| 9 | 0.7506 | 183 | ` OpenLayers.G` |
| 10 | 0.7497 | 4 | `        this.isColl` |
| 11 | 0.7110 | 4 | `GODlhAQ` |
| 12 | 0.6927 | 8 | ` {\n                document.exec` |
| 13 | 0.6889 | 3 | `URL('image/j` |
| 14 | 0.6869 | 5 | `    return document.d` |
| 15 | 0.6795 | 3 | `w0KGgo` |
| 16 | 0.6560 | 171 | ` @memberOf _\n` |
| 17 | 0.6037 | 3 | `*/regeneratorRuntime` |
| 18 | 0.6013 | 13 | ` (request, reply` |
| 19 | 0.5923 | 4 | `);\n      gl.bl` |
| 20 | 0.5724 | 15 | `    if (!basic` |
| 21 | 0.5578 | 9 | ` refreshToken, profile` |
| 22 | 0.5464 | 3 | `) => __awa` |
| 23 | 0.5270 | 4 | `        $scope.st` |
| 24 | 0.5215 | 3 | `\t\t\t\treturn _results` |
| 25 | 0.4710 | 13 | `      if (!basic` |
| 26 | 0.4137 | 3 | `    Object(__WEB` |
| 27 | 0.3736 | 3 | `link", "href` |
| 28 | 0.3569 | 3 | `;\n                    _scroll` |
| 29 | 0.3436 | 6 | `[0]).to` |
| 30 | 0.3378 | 3 | `                                if (_i` |
| 31 | 0.3344 | 3 | ` this.editorSerialize` |
| 32 | 0.3343 | 7 | `.options.styles.listen` |
| 33 | 0.3330 | 9 | ` @namespace SugarNamespace` |
| 34 | 0.3326 | 9 | ` here by Simon Fis` |
| 35 | 0.3309 | 3 | ` { return __awa` |
| 36 | 0.3302 | 3 | ` SECONDS_PER_MIN` |
| 37 | 0.3294 | 3 | `                            if (_i` |
| 38 | 0.3256 | 34 | `ATCH by Simon Fis` |
| 39 | 0.3249 | 8 | ` _regeneratorRuntime` |
| 40 | 0.3249 | 5 | `]._ColReorder` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 85 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.8720 | 3 | `0KGgoAAA` |
| 2 | 0.8485 | 3 | `.com/questions/428` |
| 3 | 0.8475 | 3 | `There was a problem` |
| 4 | 0.8450 | 4 | `fix ui-corner` |
| 5 | 0.8323 | 6 | `\t// Destroy the` |
| 6 | 0.8248 | 3 | ` to see if the` |
| 7 | 0.8211 | 4 | ` Make sure that the` |
| 8 | 0.8198 | 3 | `\t\t * Create the` |
| 9 | 0.8160 | 3 | `      // Attach` |
| 10 | 0.8137 | 3 | ` // Workaround for` |
| 11 | 0.8126 | 5 | ` Determines if the` |
| 12 | 0.8047 | 3 | ` Destroys the` |
| 13 | 0.8012 | 4 | `("Unrecognized` |
| 14 | 0.8009 | 14 | `-2011,` |
| 15 | 0.7964 | 3 | `1b873593` |
| 16 | 0.7921 | 3 | `            // When the` |
| 17 | 0.7911 | 3 | `.stdin.setRawMode` |
| 18 | 0.7827 | 7 | `\t// calculate the` |
| 19 | 0.7810 | 3 | `wysihtml` |
| 20 | 0.7793 | 7 | `\t\t * Get the` |
| 21 | 0.7787 | 5 | `. DocuSign` |
| 22 | 0.7754 | 3 | ` // Note: We` |
| 23 | 0.7738 | 6 | ` + 245760` |
| 24 | 0.7713 | 3 | `      if (true` |
| 25 | 0.7705 | 3 | `\t// Override` |
| 26 | 0.7699 | 3 | `) 2007` |
| 27 | 0.7695 | 11 | ` _typeof=typeof` |
| 28 | 0.7644 | 4 | ` //Read in the` |
| 29 | 0.7608 | 3 | ` _overlayScroll` |
| 30 | 0.7530 | 10 | `\t\t * Set the` |
| 31 | 0.7522 | 3 | `\t// run the` |
| 32 | 0.7505 | 4 | `ator.getGamep` |
| 33 | 0.7503 | 3 | `// make sure the` |
| 34 | 0.7448 | 4 | `('mdui` |
| 35 | 0.7433 | 3 | ` = $rdf` |
| 36 | 0.7408 | 14 | ` _objectWithoutProperties` |
| 37 | 0.7387 | 10 | `\t// issue #` |
| 38 | 0.7312 | 4 | ` we have a valid` |
| 39 | 0.7309 | 3 | ` Whether or not the` |
| 40 | 0.7286 | 4 | `There was an error` |

### `code_java`

35.84M tokens &middot; code_search_net (java)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 70 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 1.0000 | 3 | `ProcessInstanceDefiniton` |
| 2 | 0.9082 | 10 | `.class).in(S` |
| 3 | 0.8809 | 5 | `Key().getEnc` |
| 4 | 0.8795 | 46 | ` throws IOException, SA` |
| 5 | 0.8794 | 3 | `throws IOException, SA` |
| 6 | 0.8776 | 31 | ` KeyStroke(Key` |
| 7 | 0.8738 | 3 | `KeyStroke(Key` |
| 8 | 0.8629 | 8 | `, IOException, SA` |
| 9 | 0.8595 | 9 | `SystemService(Context.W` |
| 10 | 0.8595 | 6 | `, RelativeLayout.T` |
| 11 | 0.8486 | 31 | `(ByteOrder.L` |
| 12 | 0.8479 | 6 | ` throws IOException,SA` |
| 13 | 0.8434 | 5 | `.compress(Bit` |
| 14 | 0.8429 | 10 | `, CascadeType.M` |
| 15 | 0.8419 | 38 | ` defaultstate="coll` |
| 16 | 0.8375 | 29 | `().obtainStyled` |
| 17 | 0.8364 | 4 | `.TitledBorder.D` |
| 18 | 0.8276 | 6 | `polator(new Dec` |
| 19 | 0.8269 | 174 | `Constraints.fill = java` |
| 20 | 0.8268 | 9 | `        if (!ne` |
| 21 | 0.8261 | 3 | `)\n      throws Throw` |
| 22 | 0.8168 | 5 | `)\n            throws Throw` |
| 23 | 0.8045 | 3 | `Types = Maps.new` |
| 24 | 0.8005 | 4 | `Uri = Maps.new` |
| 25 | 0.7968 | 282 | `Performed(ev` |
| 26 | 0.7943 | 3 | `Locale().equals(l` |
| 27 | 0.7894 | 3 | `Type().equals(L` |
| 28 | 0.7862 | 46 | ` javax.swing.JCom` |
| 29 | 0.7853 | 3 | `ing = Maps.new` |
| 30 | 0.7776 | 3 | `Node = Maps.new` |
| 31 | 0.7705 | 3 | ` (null != my` |
| 32 | 0.7693 | 3 | `()) / (100` |
| 33 | 0.7666 | 83 | `.CONTENT_L` |
| 34 | 0.7608 | 4 | `Path().equals(l` |
| 35 | 0.7541 | 20 | ` result = Maps.new` |
| 36 | 0.7525 | 19 | `        synchronized (list` |
| 37 | 0.7494 | 17 | `Map = Maps.new` |
| 38 | 0.7468 | 27 | `.obtainStyled` |
| 39 | 0.7466 | 11 | ` .obtainStyled` |
| 40 | 0.7453 | 4 | `    synchronized (list` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 83 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.8435 | 4 | ` to see if the` |
| 2 | 0.8381 | 3 | ` // Loop through the` |
| 3 | 0.8348 | 4 | `("Unexpected exception` |
| 4 | 0.8100 | 3 | ` graphics.getFontRender` |
| 5 | 0.8075 | 3 | ` // Read in the` |
| 6 | 0.7980 | 3 | `DisplayHomeAsUp` |
| 7 | 0.7867 | 5 | `Could not find the` |
| 8 | 0.7730 | 3 | `XTRA_OUTPUT` |
| 9 | 0.7714 | 6 | ` // Instantiate the` |
| 10 | 0.7682 | 3 | ` = new Deflater` |
| 11 | 0.7669 | 7 | `ADING, false` |
| 12 | 0.7645 | 3 | ` = new JarOutputStream` |
| 13 | 0.7600 | 4 | `        setSupportAction` |
| 14 | 0.7595 | 3 | ` // Make sure the` |
| 15 | 0.7571 | 6 | `.GroupLayout(getContent` |
| 16 | 0.7489 | 4 | `/opencms` |
| 17 | 0.7451 | 13 | `CloseOperation(jav` |
| 18 | 0.7405 | 24 | ` new LineNumberReader` |
| 19 | 0.7393 | 3 | `        SSLEng` |
| 20 | 0.7382 | 3 | ` = new XMLEncoder` |
| 21 | 0.7361 | 10 | ` switch ( input.L` |
| 22 | 0.7303 | 3 | `org.postgresql` |
| 23 | 0.7301 | 3 | `Couldn't create` |
| 24 | 0.7292 | 3 | ` (AuthorizeCallback` |
| 25 | 0.7289 | 3 | ` // In case of` |
| 26 | 0.7288 | 8 | `("Failed to parse` |
| 27 | 0.7280 | 3 | `  // <editor` |
| 28 | 0.7277 | 7 | ` REVISIT:` |
| 29 | 0.7267 | 3 | `et=utf-8` |
| 30 | 0.7257 | 14 | `Generated Code">//` |
| 31 | 0.7246 | 3 | ` DocumentsContract.isDocument` |
| 32 | 0.7228 | 5 | ` Make sure that the` |
| 33 | 0.7176 | 3 | ` (WildcardType` |
| 34 | 0.7150 | 3 | `.Voldemort` |
| 35 | 0.7124 | 447 | ` (state.failed` |
| 36 | 0.7100 | 3 | ` Iterate through the` |
| 37 | 0.7099 | 5 | `Could not create the` |
| 38 | 0.7095 | 13 | `org.voltd` |
| 39 | 0.7093 | 3 | ` initialize OpenCms` |
| 40 | 0.7081 | 12 | `Layout.setVerticalGroup` |

### `code_go`

35.84M tokens &middot; code_search_net (go)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 63 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.9999 | 5 | ` fakaʻiniton` |
| 2 | 0.9981 | 5 | `\treturn func(_ context` |
| 3 | 0.9382 | 6 | `.Getenv("RED` |
| 4 | 0.9270 | 4 | `("Could not mars` |
| 5 | 0.9243 | 5 | `("could not mars` |
| 6 | 0.8884 | 36 | ` := err.(aw` |
| 7 | 0.8777 | 5 | `\t\tcontext.Back` |
| 8 | 0.8766 | 3 | `\t\t\tcontext.Back` |
| 9 | 0.8550 | 34 | `, base64.St` |
| 10 | 0.8452 | 59 | ` = base64.St` |
| 11 | 0.8448 | 5 | ` "+base64.St` |
| 12 | 0.8324 | 30 | `Prefix(strings.To` |
| 13 | 0.8155 | 7 | `:      context.Back` |
| 14 | 0.8135 | 16 | ` idx := strings.Last` |
| 15 | 0.8123 | 12 | `: base64.St` |
| 16 | 0.8119 | 14 | `", base64.St` |
| 17 | 0.8055 | 9 | `ockRecorder{m` |
| 18 | 0.8019 | 5 | `Idx := strings.Last` |
| 19 | 0.7992 | 51 | `\treturn base64.St` |
| 20 | 0.7990 | 21 | `idx := strings.Last` |
| 21 | 0.7937 | 8 | `: ", log.L` |
| 22 | 0.7856 | 3 | `\t{"$inc` |
| 23 | 0.7789 | 117 | ` context.WithCancel(context` |
| 24 | 0.7768 | 241 | ` := base64.St` |
| 25 | 0.7694 | 15 | `.Split(r.URL` |
| 26 | 0.7530 | 46 | ` := &http.S` |
| 27 | 0.7497 | 32 | `\tbase64.St` |
| 28 | 0.7457 | 8 | ` %v", job` |
| 29 | 0.7454 | 161 | `rrors = append` |
| 30 | 0.7343 | 8 | `, Err: %` |
| 31 | 0.7324 | 4 | `_events_url":` |
| 32 | 0.7281 | 264 | `MockRecorder{m` |
| 33 | 0.7242 | 24 | `}, bson.M` |
| 34 | 0.7234 | 4 | `GSIb3DQ` |
| 35 | 0.7204 | 143 | `\t}\n\n\tlog.D` |
| 36 | 0.7169 | 4 | ` += base64.St` |
| 37 | 0.7149 | 7 | `BQADgg` |
| 38 | 0.7147 | 163 | `, err: %` |
| 39 | 0.7119 | 13 | `] ", log.L` |
| 40 | 0.7004 | 135 | `\t}\n\tlog.D` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 86 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.8059 | 4 | `ama.NewAsyncProducer` |
| 2 | 0.8002 | 3 | `\t// Start up` |
| 3 | 0.7996 | 3 | `27 22:` |
| 4 | 0.7941 | 3 | `// NOTE: The` |
| 5 | 0.7908 | 3 | `glBeginQueryIndex` |
| 6 | 0.7793 | 3 | `\t// run the` |
| 7 | 0.7719 | 12 | ` screen and (pre` |
| 8 | 0.7717 | 3 | ` Sanity check the` |
| 9 | 0.7706 | 3 | `// Note: the` |
| 10 | 0.7660 | 49 | ` flag.NewFlagSet` |
| 11 | 0.7639 | 9 | `onboulle` |
| 12 | 0.7611 | 3 | `// Read in the` |
| 13 | 0.7555 | 7 | `\t// Create our` |
| 14 | 0.7547 | 4 | `.suppressDep` |
| 15 | 0.7518 | 3 | ` TODO(halseth` |
| 16 | 0.7514 | 4 | `\tMaxNumberOfMessages` |
| 17 | 0.7510 | 7 | `MULTISIG` |
| 18 | 0.7510 | 4 | ` Unmarshal the` |
| 19 | 0.7478 | 4 | `>.card:only` |
| 20 | 0.7458 | 6 | `BeginConditionalRender` |
| 21 | 0.7423 | 8 | `Once sync.Once` |
| 22 | 0.7395 | 9 | `ADCCAQoC` |
| 23 | 0.7372 | 3 | `pBeginQueryIndex` |
| 24 | 0.7253 | 12 | `(vdemeester` |
| 25 | 0.7214 | 7 | `anastasiam` |
| 26 | 0.7208 | 10 | `inkMacSystemFont` |
| 27 | 0.7150 | 3 | ` We'll start by` |
| 28 | 0.7100 | 4 | ` p.SetControlMessage` |
| 29 | 0.7094 | 7 | `// We have to` |
| 30 | 0.7087 | 7 | `\t// Setup the` |
| 31 | 0.7085 | 3 | ` If there are any` |
| 32 | 0.7067 | 9 | `\t// Start the` |
| 33 | 0.6988 | 3 | `// Set up a` |
| 34 | 0.6951 | 3 | `.normalizeProperty` |
| 35 | 0.6944 | 3 | `// Instantiate the` |
| 36 | 0.6872 | 6 | ` fsnotify.New` |
| 37 | 0.6870 | 4 | `// Initialize a new` |
| 38 | 0.6868 | 5 | `(jwt.Signing` |
| 39 | 0.6843 | 7 | `\t// Fetch the` |
| 40 | 0.6831 | 4 | `this._monthsSt` |

### `code_php`

35.84M tokens &middot; code_search_net (php)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 75 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.9315 | 42 | `' => $in` |
| 2 | 0.8752 | 9 | ` {\n                    return Pel` |
| 3 | 0.8609 | 3 | `urlComponents['sche` |
| 4 | 0.8538 | 4 | `($components['sche` |
| 5 | 0.8249 | 5 | ` $components['sche` |
| 6 | 0.8235 | 3 | `$components['sche` |
| 7 | 0.8056 | 140 | ` '0', STR` |
| 8 | 0.7960 | 28 | ` $this->Coll` |
| 9 | 0.7903 | 661 | `:\n                            return Pel` |
| 10 | 0.7735 | 114 | `:\n                        return Pel` |
| 11 | 0.7550 | 3 | `:\n                return Pel` |
| 12 | 0.7316 | 1670 | ` $this->coll` |
| 13 | 0.7251 | 62 | `            ->getRepository` |
| 14 | 0.7219 | 9 | `                    ->getRepository` |
| 15 | 0.7191 | 9 | `user_id == user` |
| 16 | 0.7165 | 21 | `                ->getRepository` |
| 17 | 0.7131 | 745 | ` $this->solution` |
| 18 | 0.7085 | 8 | ` "Authorization: Basic` |
| 19 | 0.6972 | 11 | `::create('oauth` |
| 20 | 0.6926 | 6 | `w0KGgo` |
| 21 | 0.6879 | 14 | `$table->tim` |
| 22 | 0.6839 | 16 | `GODlhAQ` |
| 23 | 0.6777 | 5 | ` if (! filter_var` |
| 24 | 0.6771 | 54 | ` if (!filter_var` |
| 25 | 0.6758 | 3 | `FTkSuQ` |
| 26 | 0.6752 | 3 | ` if(! filter_var` |
| 27 | 0.6744 | 219 | ` $table->tim` |
| 28 | 0.6694 | 8 | ` $_SERVER["REM` |
| 29 | 0.6397 | 4 | `_PAYMENT_P` |
| 30 | 0.4454 | 5 | `);\n                return Pel` |
| 31 | 0.3807 | 3 | `.299095438` |
| 32 | 0.3505 | 46 | `author Vova Feldman` |
| 33 | 0.3439 | 5 | `->getPlainForeignKey` |
| 34 | 0.3392 | 3 | `getEltDecoration` |
| 35 | 0.3368 | 3 | `->getFirstForeignKey` |
| 36 | 0.3333 | 3 | ` 'ReplacingMerge` |
| 37 | 0.3326 | 3 | `$this->_initUrls` |
| 38 | 0.3323 | 3 | `Adapter()->dropForeignKey` |
| 39 | 0.3306 | 3 | ` function getAdvancedForeignKey` |
| 40 | 0.3305 | 11 | `->getDropForeignKey` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 83 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.8775 | 6 | `0KGgoAAA` |
| 2 | 0.8578 | 10 | `There was a problem` |
| 3 | 0.8498 | 10 | `An error has occurred` |
| 4 | 0.8384 | 3 | ` \\token_get_all` |
| 5 | 0.8346 | 3 | ` @token_get_all` |
| 6 | 0.8242 | 3 | `URBVDj` |
| 7 | 0.8182 | 45 | `, $makeNew` |
| 8 | 0.8143 | 31 | ` = token_get_all` |
| 9 | 0.8132 | 3 | `        // init the` |
| 10 | 0.8096 | 3 | `("Unable to parse` |
| 11 | 0.8095 | 3 | `\tstream_set_time` |
| 12 | 0.8086 | 4 | ` // Check for the` |
| 13 | 0.8079 | 14 | ` (token_get_all` |
| 14 | 0.8019 | 3 | `\t// Bind the` |
| 15 | 0.8006 | 4 | `dBTUEAA` |
| 16 | 0.8006 | 11 | `        stream_set_time` |
| 17 | 0.8004 | 7 | `\t// Enqueue` |
| 18 | 0.7992 | 7 | ` First, we will` |
| 19 | 0.7926 | 4 | `        // Invoke` |
| 20 | 0.7920 | 3 | `audio/x-aiff` |
| 21 | 0.7904 | 3 | ` we have a valid` |
| 22 | 0.7878 | 7 | `// Make sure the` |
| 23 | 0.7856 | 3 | ` => 'checkboxes` |
| 24 | 0.7824 | 11 | `            stream_set_time` |
| 25 | 0.7821 | 4 | `        // Split the` |
| 26 | 0.7799 | 5 | ` = imagecolort` |
| 27 | 0.7790 | 5 | `sl_pkey_new` |
| 28 | 0.7702 | 3 | `        // Fetch the` |
| 29 | 0.7702 | 15 | ` [];\n        $_header` |
| 30 | 0.7665 | 9 | ` xml_parser_create` |
| 31 | 0.7651 | 3 | `         * Register the` |
| 32 | 0.7628 | 7 | `\t// Load the` |
| 33 | 0.7622 | 3 | ` 'Unrecognized` |
| 34 | 0.7611 | 14 | ` to see if the` |
| 35 | 0.7609 | 3 | ` 'Freemius` |
| 36 | 0.7593 | 3 | ` by the Mouf` |
| 37 | 0.7561 | 6 | `\t// Prepare the` |
| 38 | 0.7560 | 21 | `        // parse inputs` |
| 39 | 0.7552 | 22 | `process = proc_open` |
| 40 | 0.7524 | 4 | ` curl_multi_init` |

### `code_ruby`

5.87M tokens &middot; code_search_net (ruby)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 92 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 1.0000 | 3 | `http://id.loc` |
| 2 | 0.8499 | 17 | ` = Base64.st` |
| 3 | 0.8247 | 3 | `: Base64.st` |
| 4 | 0.7963 | 4 | `      Base64.st` |
| 5 | 0.7847 | 3 | ` += Base64.st` |
| 6 | 0.6924 | 3 | ` + Base64.st` |
| 7 | 0.6770 | 11 | ` req.send_request(options` |
| 8 | 0.3283 | 13 | ` 1 && args` |
| 9 | 0.3237 | 3 | `updated_at = DateTime` |
| 10 | 0.3079 | 10 | ` builder.use FaradayMiddleware` |
| 11 | 0.3067 | 4 | ` conn.use FaradayMiddleware` |
| 12 | 0.3011 | 3 | `::TooManyRedirect` |
| 13 | 0.2986 | 4 | `ator::RelationDecor` |
| 14 | 0.2950 | 12 | ` @window.wrefresh` |
| 15 | 0.2939 | 3 | `_inheritable_attribute` |
| 16 | 0.2931 | 3 | ` @pupu.params` |
| 17 | 0.2923 | 3 | `_encrypted_attribute` |
| 18 | 0.2916 | 3 | ` res.response.each_header` |
| 19 | 0.2912 | 3 | `print._natural_sort` |
| 20 | 0.2912 | 3 | `, but Simon Sap` |
| 21 | 0.2911 | 3 | `::ReplicationProtected` |
| 22 | 0.2897 | 3 | `)\n        instance_e` |
| 23 | 0.2893 | 4 | ` @user.errors.full` |
| 24 | 0.2890 | 3 | `socket.sync_close` |
| 25 | 0.2890 | 3 | `_effecting_attribute` |
| 26 | 0.2887 | 3 | `.attributes.each_attribute` |
| 27 | 0.2873 | 15 | ` check_critical_attribute` |
| 28 | 0.2871 | 3 | `          Gl.glUniform` |
| 29 | 0.2871 | 4 | `Db::DocumentBlueprint` |
| 30 | 0.2863 | 3 | ` @access_token.params` |
| 31 | 0.2859 | 3 | `_namespaced_attribute` |
| 32 | 0.2848 | 3 | `_opulent_buffer` |
| 33 | 0.2846 | 3 | `::BalancedHttp` |
| 34 | 0.2840 | 4 | `.ecies_dec` |
| 35 | 0.2836 | 4 | `ark.ruby.exec` |
| 36 | 0.2833 | 3 | `{TimelineSetter` |
| 37 | 0.2829 | 4 | ` = find_attribute_column` |
| 38 | 0.2826 | 4 | `client.reset_instance_attribute` |
| 39 | 0.2824 | 7 | `.class.human_attribute` |
| 40 | 0.2808 | 4 | `force_sync_attribute` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 89 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.6161 | 3 | `4-08-` |
| 2 | 0.5902 | 3 | `MULTISIG` |
| 3 | 0.5795 | 3 | `        'AmazonAuthorization` |
| 4 | 0.5665 | 3 | ` = Openwsman` |
| 5 | 0.5653 | 29 | ` = OptionParser.new` |
| 6 | 0.5325 | 3 | `abel::OBConversion` |
| 7 | 0.5271 | 3 | `        raise Synvert` |
| 8 | 0.5267 | 4 | ` OctocatalogDiff` |
| 9 | 0.5257 | 3 | `::ReplicationProtected` |
| 10 | 0.5231 | 3 | `            :SeasonSegment` |
| 11 | 0.5171 | 3 | `::S3.new` |
| 12 | 0.5126 | 3 | ` raise Octocatalog` |
| 13 | 0.4999 | 3 | `8-12-` |
| 14 | 0.4973 | 4 | ` CelluloidPub` |
| 15 | 0.4970 | 3 | `            :PerMode` |
| 16 | 0.4949 | 3 | `    rescue KJess` |
| 17 | 0.4913 | 3 | `\n      SemanticLogger` |
| 18 | 0.4824 | 16 | `1-10-` |
| 19 | 0.4807 | 3 | ` = ChunkyPN` |
| 20 | 0.4771 | 3 | `pec = OpenAssets` |
| 21 | 0.4770 | 22 | ` = MnoEnterprise` |
| 22 | 0.4715 | 4 | ` beaker-hostgener` |
| 23 | 0.4713 | 8 | `1-09-` |
| 24 | 0.4591 | 4 | `Couldn't find` |
| 25 | 0.4586 | 12 | ` raise SSRFProxy` |
| 26 | 0.4572 | 3 | `client.discovered_api` |
| 27 | 0.4558 | 3 | `= OptionParser.new` |
| 28 | 0.4555 | 3 | ` = MaRuKu` |
| 29 | 0.4526 | 5 | ` resp = Caboose` |
| 30 | 0.4495 | 3 | `QP::Channel.new` |
| 31 | 0.4470 | 4 | `)\n      Axlsx` |
| 32 | 0.4462 | 3 | `            :GameSegment` |
| 33 | 0.4446 | 4 | ` rescue CouchRest` |
| 34 | 0.4402 | 3 | `kiqUniqueJobs` |
| 35 | 0.4363 | 3 | `1-11-` |
| 36 | 0.4362 | 10 | `      MnoEnterprise` |
| 37 | 0.4349 | 3 | `        "select_fields` |
| 38 | 0.4348 | 3 | ` => 'Caboose` |
| 39 | 0.4327 | 3 | `Trebuchet MS` |
| 40 | 0.4327 | 4 | ` why? why not` |

## Depth corpora

### `web`

35.84M tokens &middot; HuggingFaceFW/fineweb-edu sample-10BT

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 97 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.7324 | 3 | `1986 © Harper` |
| 2 | 0.6577 | 3 | `.0/) applies` |
| 3 | 0.6379 | 9 | ` to jurisdictional claims in` |
| 4 | 0.6199 | 4 | ` with rights of surviv` |
| 5 | 0.5770 | 15 | `This note was uploaded` |
| 6 | 0.4924 | 5 | `.\nOther nearby markers` |
| 7 | 0.3366 | 3 | ` & vb. n` |
| 8 | 0.3354 | 3 | ` of Eugenic Femin` |
| 9 | 0.3273 | 3 | ` Dr. Susan Keg` |
| 10 | 0.3265 | 3 | ` the Flavian Amph` |
| 11 | 0.3235 | 4 | ` Illustrated Encyclopedia of Hinduism` |
| 12 | 0.3218 | 4 | `6 by Alan Crom` |
| 13 | 0.3210 | 3 | ` 'Transatlantic Slavery` |
| 14 | 0.3210 | 3 | ` John Archibald Venn` |
| 15 | 0.3194 | 6 | `://buhlplanet` |
| 16 | 0.3170 | 3 | ` Pravasi Bhar` |
| 17 | 0.3169 | 3 | ` Journal of Applied Meteor` |
| 18 | 0.3164 | 4 | ` Tunguska Meteor` |
| 19 | 0.3162 | 6 | `, by Bill Cough` |
| 20 | 0.3154 | 3 | `\n- James Loch` |
| 21 | 0.3153 | 3 | `ierry Van Bast` |
| 22 | 0.3150 | 4 | `By Lynn Byczyn` |
| 23 | 0.3147 | 5 | ` http://globalvoices` |
| 24 | 0.3130 | 4 | ` History of Negro Slavery` |
| 25 | 0.3130 | 3 | ` the Great Siberian Expl` |
| 26 | 0.3121 | 5 | ` Encyclopedia of Indian Philosoph` |
| 27 | 0.3118 | 3 | ` by Robert Wayne Atkins` |
| 28 | 0.3118 | 4 | ` Sri Narayana Guru` |
| 29 | 0.3106 | 3 | `: Abraham Dee Bartlett` |
| 30 | 0.3105 | 3 | ` Search of Hannah Crafts` |
| 31 | 0.3104 | 5 | ` the Torrey Botanical` |
| 32 | 0.3094 | 4 | ` The Feminine Myst` |
| 33 | 0.3090 | 3 | `Berthe Morisot` |
| 34 | 0.3089 | 4 | ` The Accidental Slave` |
| 35 | 0.3083 | 3 | ` Neurobiology of Preference` |
| 36 | 0.3080 | 3 | ` Allen and Charlotte Gins` |
| 37 | 0.3077 | 4 | ` Aesthetics of Everyday` |
| 38 | 0.3076 | 4 | `ATIS Telecom Glossary` |
| 39 | 0.3072 | 3 | `.; Hildebolt` |
| 40 | 0.3070 | 3 | ` "American FactFinder` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 83 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.8812 | 4 | ` this article, please` |
| 2 | 0.8771 | 3 | `1-118-` |
| 3 | 0.8710 | 3 | `-4129-` |
| 4 | 0.8685 | 3 | ` our free email newsletter` |
| 5 | 0.8661 | 3 | `1-136-` |
| 6 | 0.8605 | 3 | `1-137-` |
| 7 | 0.8573 | 21 | ` this article is licensed` |
| 8 | 0.8492 | 3 | `.\nMake sure that` |
| 9 | 0.8438 | 7 | `able Conditions of Work` |
| 10 | 0.8355 | 3 | ` arbitrary arrest and detention` |
| 11 | 0.8354 | 3 | `0-321-` |
| 12 | 0.8299 | 3 | `-8047-` |
| 13 | 0.8244 | 4 | ` Bargain Collectively` |
| 14 | 0.8221 | 3 | ` 1997-` |
| 15 | 0.8130 | 3 | ` The views and opinions` |
| 16 | 0.8053 | 3 | ` 1998-` |
| 17 | 0.8037 | 7 | ` This article is for` |
| 18 | 0.8027 | 3 | `-7607-` |
| 19 | 0.8019 | 3 | `-7524-` |
| 20 | 0.7992 | 14 | ` be logged in to` |
| 21 | 0.7938 | 3 | `-7817-` |
| 22 | 0.7937 | 14 | ` GNU Free Documentation License` |
| 23 | 0.7925 | 3 | ` Rights Reserved. \|` |
| 24 | 0.7916 | 3 | ` and opinions expressed in` |
| 25 | 0.7911 | 3 | ` All Rights Reserved.` |
| 26 | 0.7905 | 3 | `-7619-` |
| 27 | 0.7889 | 3 | `Sign up to get` |
| 28 | 0.7874 | 5 | `Unless otherwise noted,` |
| 29 | 0.7736 | 3 | `\nPost your comments` |
| 30 | 0.7712 | 12 | `0-470-` |
| 31 | 0.7682 | 12 | ` Johannes Gutenberg University` |
| 32 | 0.7662 | 12 | ` 1996-` |
| 33 | 0.7619 | 30 | ` must be logged in` |
| 34 | 0.7614 | 10 | `0-393-` |
| 35 | 0.7598 | 3 | ` material from the Wikipedia` |
| 36 | 0.7582 | 3 | ` In Search of the` |
| 37 | 0.7581 | 13 | ` to review this product` |
| 38 | 0.7578 | 6 | `-8160-` |
| 39 | 0.7513 | 8 | `This article is licensed` |
| 40 | 0.7501 | 3 | ` otherwise noted, content` |

### `wiki_full`

35.84M tokens &middot; wikimedia/wikipedia 20231101.en

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 95 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.6788 | 3 | `: BATE Boris` |
| 2 | 0.4215 | 11 | ` Firearms (Amendment` |
| 3 | 0.3500 | 4 | `, he played collegiate` |
| 4 | 0.3440 | 4 | ` de Música Latino` |
| 5 | 0.3394 | 4 | ` 'The Portable Nietzsche` |
| 6 | 0.3390 | 3 | ` Esbjörn Svens` |
| 7 | 0.3358 | 3 | ` Robert R. McCorm` |
| 8 | 0.3346 | 5 | ` directed by Stuart Gos` |
| 9 | 0.3332 | 3 | ` of a Random Trigonometric` |
| 10 | 0.3325 | 5 | ` Knud Rasmussen Glacier` |
| 11 | 0.3321 | 4 | `ónio Manuel Ribe` |
| 12 | 0.3293 | 3 | ` an Amish Menn` |
| 13 | 0.3292 | 5 | ` Henry F. Schae` |
| 14 | 0.3283 | 4 | ` Handbook of Classical Mythology` |
| 15 | 0.3275 | 3 | ` of Monopoly Capitalism` |
| 16 | 0.3244 | 3 | ` Stephen J. Cann` |
| 17 | 0.3240 | 4 | `Elisabeth Schae` |
| 18 | 0.3235 | 7 | ` the Virtuti Milit` |
| 19 | 0.3222 | 5 | `y Amish Menn` |
| 20 | 0.3219 | 4 | ` "Amish Menn` |
| 21 | 0.3214 | 3 | `. Mustafa Khat` |
| 22 | 0.3205 | 13 | ` Welcome to Cottage Hamlet` |
| 23 | 0.3205 | 3 | ` Sérgio Mend` |
| 24 | 0.3204 | 3 | ` Journal of Global Buddhism` |
| 25 | 0.3204 | 3 | ` Dictionary of Classical Mythology` |
| 26 | 0.3203 | 3 | ` conducting by Jimmy Haskell` |
| 27 | 0.3197 | 4 | ` Brezhnev Doctrine` |
| 28 | 0.3196 | 3 | ` Eric W. Weis` |
| 29 | 0.3196 | 3 | `, with Derek Frid` |
| 30 | 0.3195 | 4 | ` by Stephen Gaghan` |
| 31 | 0.3195 | 3 | ` by Takashi Mura` |
| 32 | 0.3187 | 3 | `\nDurga McB` |
| 33 | 0.3185 | 3 | ` Gift of Southern Cooking` |
| 34 | 0.3181 | 3 | ` are the Salisbury Glacier` |
| 35 | 0.3176 | 19 | ` La Haye Sainte` |
| 36 | 0.3174 | 3 | ` of Antoninus Liber` |
| 37 | 0.3172 | 4 | `) – Neil Brewer` |
| 38 | 0.3171 | 3 | ` by Yoshiki Nakamura` |
| 39 | 0.3168 | 3 | `: Paul DiFrances` |
| 40 | 0.3162 | 3 | ` Guide to Texas Gardening` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 89 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.9265 | 3 | `male 284,` |
| 2 | 0.8721 | 3 | ` single and never married` |
| 3 | 0.7957 | 3 | ` is a listing of` |
| 4 | 0.7874 | 13 | `, had an enrollment` |
| 5 | 0.7782 | 3 | ` , the construction rate` |
| 6 | 0.7571 | 3 | ` article incorporates text from` |
| 7 | 0.7505 | 12 | ` might be described thus` |
| 8 | 0.7409 | 4 | ` CIA World Factbook` |
| 9 | 0.7366 | 4 | `-Saxon origin and derives` |
| 10 | 0.7267 | 8 | ` other benzodiazepines` |
| 11 | 0.7212 | 3 | ` "Herrenvol` |
| 12 | 0.7134 | 9 | ` Association (NJSI` |
| 13 | 0.7122 | 3 | ` Sign of the Beaver` |
| 14 | 0.7033 | 3 | ` district of Lincolnshire` |
| 15 | 0.7032 | 4 | ` responsibility for local issues` |
| 16 | 0.6992 | 3 | ` Lincolnshire, England` |
| 17 | 0.6950 | 7 | ` for reduced-cost lunch` |
| 18 | 0.6932 | 10 | ` three most popular parties` |
| 19 | 0.6903 | 11 | `ically be described thus` |
| 20 | 0.6901 | 3 | ` of Florida Athletic Hall` |
| 21 | 0.6846 | 5 | `"Station to Station` |
| 22 | 0.6838 | 3 | `.\n\nPeople of Praise` |
| 23 | 0.6832 | 10 | ` women, took part` |
| 24 | 0.6786 | 15 | `-lane undivided` |
| 25 | 0.6628 | 3 | ` Yolanda Sal` |
| 26 | 0.6627 | 4 | ` Methodist Church, elected` |
| 27 | 0.6609 | 6 | `, the highway runs` |
| 28 | 0.6587 | 7 | ` produced newscasts` |
| 29 | 0.6575 | 5 | `, Queensland, Australia` |
| 30 | 0.6574 | 3 | ` "Break the Ice` |
| 31 | 0.6517 | 3 | ` Dr. Jack Gram` |
| 32 | 0.6506 | 3 | ` most athletic competition purposes` |
| 33 | 0.6484 | 3 | `ath Bloody Sabbath` |
| 34 | 0.6467 | 3 | ` of the Actor model` |
| 35 | 0.6464 | 4 | ` Birds of a Feather` |
| 36 | 0.6432 | 5 | ` Sphingidae` |
| 37 | 0.6429 | 4 | ` The Pleasure Garden` |
| 38 | 0.6427 | 4 | ` "At Your Command` |
| 39 | 0.6403 | 4 | `ory Dickory Dock` |
| 40 | 0.6385 | 3 | ` in The Mirror Crack` |

### `math_web`

35.84M tokens &middot; open-web-math/open-web-math

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 76 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.9995 | 5 | `@@ -1,` |
| 2 | 0.7648 | 14 | `:=PCGroup([` |
| 3 | 0.7573 | 5 | `\n# A tib` |
| 4 | 0.7510 | 23 | `\n\n Thread Tools Display` |
| 5 | 0.7470 | 16 | ` if {[catch` |
| 6 | 0.7386 | 17 | `mathrm{new}}(\\` |
| 7 | 0.7363 | 4 | `q^r})$` |
| 8 | 0.7209 | 5 | `.add(layers.D` |
| 9 | 0.7144 | 3 | `901960784314` |
| 10 | 0.6839 | 3 | `'ll also get unlimited` |
| 11 | 0.6782 | 3 | `> <q cite` |
| 12 | 0.6778 | 4 | `, CW_USED` |
| 13 | 0.6717 | 75 | ` to jurisdictional claims in` |
| 14 | 0.6267 | 34 | `5/rdf http` |
| 15 | 0.6238 | 10 | ` eLearning\n\nPosted` |
| 16 | 0.6153 | 6 | `) · GW(p` |
| 17 | 0.5919 | 3 | `/grantAgreement` |
| 18 | 0.5800 | 3 | `email) (contrib` |
| 19 | 0.5772 | 4 | `1Q     Median` |
| 20 | 0.5485 | 11 | `1Q   Median` |
| 21 | 0.5372 | 86 | `\nOpen Detailed Calendar` |
| 22 | 0.5241 | 3 | ` EmilValued Senior` |
| 23 | 0.4786 | 34 | ` $\\square$\n\nLemma` |
| 24 | 0.4736 | 3 | `>:1(<module` |
| 25 | 0.4412 | 25 | `## # A tib` |
| 26 | 0.4278 | 3 | ` ## # A tib` |
| 27 | 0.4251 | 8 | ` $F_p(T` |
| 28 | 0.4057 | 13 | `Non-Human User` |
| 29 | 0.3742 | 3 | `&list=PL` |
| 30 | 0.3682 | 26 | `ulate this circuit –` |
| 31 | 0.3621 | 3 | `#> # A tib` |
| 32 | 0.3601 | 4 | `\n\nExpert-verified` |
| 33 | 0.3471 | 9 | `)\n[TeX:]` |
| 34 | 0.3401 | 4 | `_Word_Problems` |
| 35 | 0.3378 | 14 | `\\usepackage{amsfonts` |
| 36 | 0.3377 | 3 | `)\n\nwith pm.Model` |
| 37 | 0.3359 | 34 | `}{l}\\require` |
| 38 | 0.3343 | 3 | `., Korsbak` |
| 39 | 0.3306 | 8 | ` Eric W. Weis` |
| 40 | 0.3297 | 13 | `\xa0Martin Sleziak` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 91 ranked 4-grams.

| # | gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.9393 | 5 | `.com/questions/156` |
| 2 | 0.9299 | 7 | `2014JA020` |
| 3 | 0.9296 | 3 | ` 2020 ` |
| 4 | 0.9284 | 3 | ` do you write the` |
| 5 | 0.9268 | 7 | `2021 at ` |
| 6 | 0.9253 | 14 | ` 2006 ` |
| 7 | 0.9251 | 4 | `4 7:` |
| 8 | 0.9201 | 21 | ` 2004 ` |
| 9 | 0.9191 | 3 | `2016JA023` |
| 10 | 0.9190 | 6 | ` 2014 ` |
| 11 | 0.9172 | 3 | ` we have discussed the` |
| 12 | 0.9136 | 3 | ` 2005 ` |
| 13 | 0.9122 | 4 | ` Where can I find` |
| 14 | 0.9081 | 3 | `6/science.120` |
| 15 | 0.9044 | 3 | ` at 05:` |
| 16 | 0.9025 | 7 | ` 2007 ` |
| 17 | 0.9015 | 3 | `4 4:` |
| 18 | 0.9002 | 4 | `-4757-` |
| 19 | 0.8957 | 3 | ` How to deal with` |
| 20 | 0.8946 | 4 | `3 5:` |
| 21 | 0.8936 | 12 | ` That is, the` |
| 22 | 0.8927 | 3 | ` the source of the` |
| 23 | 0.8851 | 3 | `### How does the` |
| 24 | 0.8847 | 7 | ` For instance, the` |
| 25 | 0.8846 | 3 | ` a circle with a` |
| 26 | 0.8842 | 3 | ` the centroid of the` |
| 27 | 0.8820 | 7 | `2015JA021` |
| 28 | 0.8817 | 6 | ` the feckin` |
| 29 | 0.8810 | 4 | ` we look at the` |
| 30 | 0.8810 | 4 | ` other hand, the` |
| 31 | 0.8770 | 4 | ` Zbl\xa0075` |
| 32 | 0.8751 | 3 | ` This page contains a` |
| 33 | 0.8746 | 4 | ` example, consider the` |
| 34 | 0.8735 | 3 | `This problem is a` |
| 35 | 0.8735 | 3 | ` Is it safe to` |
| 36 | 0.8732 | 3 | ` a total of ` |
| 37 | 0.8721 | 4 | ` This Day in Math` |
| 38 | 0.8721 | 3 | ` figure above, the` |
| 39 | 0.8713 | 13 | ` this article we will` |
| 40 | 0.8710 | 6 | ` 2008 ` |

---

Generated from `analysis/engram/scan.py` output; merged across shards 0-2.
