# Engram gate activation: top suffix 4-grams

DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash`, MXFP4 routed experts),
vLLM `deepseekv41-flash-0909`, TP=8 on 3x H100 nodes (8 GPUs each).
Run [34583988417](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34583988417),
2026-09-11, 3h47m wall-clock.

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

Per token the gate is averaged over the four hyper-connection copies. Each
domain is read as 3584-token chunks, prefilled with `max_tokens=1`. Within
each (domain, layer) the top 1% of gate values are taken as strong, and every
strong position is attributed to the 2-, 3-, and 4-gram ending at it. Rows
below are ranked by mean gate with a minimum of 3 occurrences; only the
4-gram tier is shown here.

**Corpus: 235.3M tokens over 16 domains,
English and Chinese only.**

| domain | tokens | source |
| --- | --- | --- |
| `wiki` | 2.40M | Salesforce/wikitext, wikitext-2-raw-v1 |
| `chat` | 18.82M | HuggingFaceH4/ultrachat_200k (train_sft) |
| `math` | 1.18M | openai/gsm8k (main) |
| `code_mbpp` | 0.03M | google-research-datasets/mbpp (full) |
| `wiki_zh` | 18.82M | wikimedia/wikipedia 20231101.zh |
| `chat_zh` | 18.82M | BelleGroup/train_1M_CN |
| `web_zh` | 18.82M | HuggingFaceFW/fineweb-2 cmn_Hani |
| `code_python` | 18.82M | code_search_net (python) |
| `code_javascript` | 18.82M | code_search_net (javascript) |
| `code_java` | 18.82M | code_search_net (java) |
| `code_go` | 18.82M | code_search_net (go) |
| `code_php` | 18.82M | code_search_net (php) |
| `code_ruby` | 5.87M | code_search_net (ruby) |
| `web` | 18.82M | HuggingFaceFW/fineweb-edu sample-10BT |
| `wiki_full` | 18.82M | wikimedia/wikipedia 20231101.en |
| `math_web` | 18.82M | open-web-math/open-web-math |

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

| # | mean gate | count | domain / layer | 4-gram |
| --: | --: | --: | --- | --- |
| 1 | 0.4486 | 28 | `code_php` / 0 | `' => $in` |
| 2 | 0.4313 | 7 | `code_python` / 0 | `    response = ur` |
| 3 | 0.4298 | 16 | `code_python` / 0 | `        response = ur` |
| 4 | 0.4296 | 12 | `code_python` / 0 | `            response = ur` |
| 5 | 0.4232 | 8 | `math_web` / 1 | `:=PCGroup([` |
| 6 | 0.4231 | 10 | `code_java` / 0 | `.class).in(S` |
| 7 | 0.4059 | 3 | `code_python` / 0 | ` None) or get` |
| 8 | 0.3934 | 3 | `code_python` / 0 | `        with open(re` |
| 9 | 0.3799 | 3 | `math_web` / 1 | ` The Best Or Nothing` |
| 10 | 0.3606 | 5 | `math_web` / 1 | `: 17 Dec` |

The PHP leader `' => $in` is an array-literal fragment; the three
`response = ur` rows are the same `urllib` call at three indentation depths,
which the tokenizer makes into three distinct 4-grams. `:=PCGroup([` is GAP
computer-algebra syntax from a maths forum.

## Notable entries

Hand-picked from the tables above, because what the gate opens on is easier
to see in specific cases than in aggregate. These are not the strongest
rows -- they are the legible ones.

| mean gate | count | domain / layer | 4-gram | what it is |
| --: | --: | --- | --- | --- |
| 0.0916 | 10 | `wiki` / 0 | ` Wright : Ace Attorney` | The game subtitle, memorized whole. |
| 0.1155 | 3 | `wiki` / 0 | ` " Run Run Rudolph` | Chuck Berry, 1958. |
| 0.1112 | 3 | `wiki` / 0 | ` Treehouse of Horror` | The Simpsons' Halloween episodes. |
| 0.1797 | 3 | `wiki` / 1 | `ane Clown Pos` | Insane Clown Posse, mid-token. |
| 0.1804 | 3 | `wiki_full` / 1 | ` Sabbath Bloody Sabbath` | Black Sabbath, 1973. |
| 0.1751 | 3 | `wiki_full` / 1 | ` The Spectacular Spider` | ...-Man. |
| 0.2123 | 3 | `wiki` / 1 | ` , Super Mario Land` | Game Boy, 1989. |
| 0.1234 | 3 | `wiki` / 1 | `ll Always Have Paris` | Casablanca, via a TNG episode title. |
| 0.1830 | 3 | `wiki` / 1 | ` Life Is Worth Living` | Fulton Sheen's 1950s TV show. |
| 0.1062 | 17 | `chat_zh` / 0 | `xpialidocious` | The tail of supercalifragilistic-. |
| 0.1075 | 7 | `chat_zh` / 0 | `.141592653` | Nine digits of pi after the point. |
| 0.1125 | 27 | `chat_zh` / 0 | ` a truth universally acknowledged` | Pride and Prejudice, opening line. |
| 0.1098 | 3 | `wiki_zh` / 0 | `KING OF ZIPANGU` | A 1990s NHK drama's romanized title. |
| 0.1282 | 18 | `web_zh` / 0 | `玄奘西游记` | Xuanzang's Journey to the West. |
| 0.1083 | 4 | `chat_zh` / 0 | `《荒岛余生` | Cast Away, in Chinese. |
| 0.0839 | 3 | `math` / 0 | ` pints of frozen yogurt` | A GSM8K word-problem prop. |
| 0.0965 | 9 | `math` / 0 | `g of packing peanuts` | Another one. |
| 0.0791 | 3 | `math` / 0 | ` The Fancy Salon` | An invented GSM8K business. |
| 0.0790 | 3 | `math` / 0 | ` 10 Baby Ruth` | Candy bars, being counted. |
| 0.0838 | 4 | `wiki` / 0 | ` Wisteria Lane` | Desperate Housewives. |
| 0.1778 | 7 | `wiki_full` / 1 | ` Angiosperm Phylogen` | ...y Group, the botanical classification. |
| 0.1178 | 3 | `web` / 0 | ` Elders of Zion` | From a Project Gutenberg catalogue page. |
| 0.1033 | 5 | `chat` / 0 | `The Life of Pablo` | Kanye West, 2016. |

The pattern across all of them: a rare multi-token name whose later pieces
are unguessable from the model's weights but fully determined once the
earlier pieces are known. `ane Clown Pos` is the clearest case -- the gate
opens in the middle of a word, on a boundary that exists only because of how
the tokenizer split a band's name. The maths rows show the same mechanism on
invented props: once a GSM8K problem has said "pints of frozen", the next
token is not in doubt.

## Reference-study domains

### `wiki`

2.40M tokens &middot; Salesforce/wikitext, wikitext-2-raw-v1

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 131 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.1155 | 3 | ` " Run Run Rudolph` |
| 2 | 0.1112 | 3 | ` Treehouse of Horror` |
| 3 | 0.1071 | 7 | ` Jermaine Dup` |
| 4 | 0.1055 | 6 | `ll Always Have Paris` |
| 5 | 0.1053 | 4 | ` in The Good Terror` |
| 6 | 0.1042 | 15 | ` Importance of Being Earn` |
| 7 | 0.1017 | 3 | ` the Annals of Ulster` |
| 8 | 0.1013 | 13 | ` Wolverhampton Wander` |
| 9 | 0.1009 | 3 | `th Armored Cavalry` |
| 10 | 0.0994 | 3 | ` , the Joint Typh` |
| 11 | 0.0970 | 3 | `Clocked ReMix` |
| 12 | 0.0956 | 3 | ` Hermit the Frog` |
| 13 | 0.0951 | 3 | ` and Ghost of Sparta` |
| 14 | 0.0943 | 3 | ` Feet in the Clouds` |
| 15 | 0.0931 | 3 | `able Kimmy Schmidt` |
| 16 | 0.0928 | 3 | ` . Ernest Borgnine` |
| 17 | 0.0928 | 4 | ` : Stand Alone Complex` |
| 18 | 0.0925 | 4 | ` A Rush of Blood` |
| 19 | 0.0923 | 4 | `psilophodont` |
| 20 | 0.0921 | 5 | ` I Put a Spell` |
| 21 | 0.0919 | 5 | ` H. Jon Benjamin` |
| 22 | 0.0919 | 6 | ` Church of Christ Pant` |
| 23 | 0.0919 | 3 | `-@ Marie Apost` |
| 24 | 0.0917 | 3 | `98 million household viewers` |
| 25 | 0.0917 | 3 | ` van Giersbergen` |
| 26 | 0.0916 | 10 | ` Wright : Ace Attorney` |
| 27 | 0.0916 | 4 | ` Tim and Chris Stam` |
| 28 | 0.0915 | 8 | ` Stephen Thomas Erlew` |
| 29 | 0.0914 | 6 | ` and North East Somerset` |
| 30 | 0.0910 | 3 | ` and The Fame Monster` |
| 31 | 0.0910 | 3 | ` Cross and Red Crescent` |
| 32 | 0.0910 | 4 | ` the Kashi Vish` |
| 33 | 0.0908 | 6 | `2012 Summer Paral` |
| 34 | 0.0905 | 3 | ` Gaudium et spes` |
| 35 | 0.0900 | 3 | `. Chasuble` |
| 36 | 0.0900 | 4 | ` Joint Typhoon Warning` |
| 37 | 0.0899 | 6 | ` Am Not a Robot` |
| 38 | 0.0898 | 3 | ` Operation Enduring Freedom` |
| 39 | 0.0898 | 3 | ` =Chasing Verm` |
| 40 | 0.0893 | 3 | `1st Battle Squadron` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 133 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2123 | 3 | ` , Super Mario Land` |
| 2 | 0.1909 | 3 | ` on the Tracks` |
| 3 | 0.1899 | 3 | `" Far Away Places` |
| 4 | 0.1830 | 3 | ` Life Is Worth Living` |
| 5 | 0.1797 | 3 | `ane Clown Pos` |
| 6 | 0.1751 | 12 | ` the Pulse of Morning` |
| 7 | 0.1673 | 3 | ` It All Back Home` |
| 8 | 0.1668 | 4 | ` and Better Updated Uno` |
| 9 | 0.1661 | 3 | ` at Tittenhurst` |
| 10 | 0.1660 | 4 | ` " Unnatural Selection` |
| 11 | 0.1656 | 3 | ` for " Man Down` |
| 12 | 0.1635 | 3 | ` =The corn cra` |
| 13 | 0.1629 | 3 | ` Let Me Be Mis` |
| 14 | 0.1626 | 4 | ` Ramnagar Fort` |
| 15 | 0.1624 | 3 | ` " Oath Sign` |
| 16 | 0.1586 | 3 | ` " Through the Rain` |
| 17 | 0.1586 | 7 | ` VanDerWerff` |
| 18 | 0.1569 | 7 | ` state trunkline highway` |
| 19 | 0.1561 | 3 | ` =Chasing Verm` |
| 20 | 0.1551 | 4 | ` the Medway Meg` |
| 21 | 0.1544 | 3 | ` Everglades Agricultural` |
| 22 | 0.1542 | 3 | `s national wheelchair basketball` |
| 23 | 0.1528 | 4 | ` the Devin Townsend` |
| 24 | 0.1505 | 11 | ` The Sixth Extinction` |
| 25 | 0.1504 | 3 | ` , Memory Almost Full` |
| 26 | 0.1491 | 3 | ` 61 Revisited` |
| 27 | 0.1463 | 3 | `able Kimmy Schmidt` |
| 28 | 0.1462 | 4 | ` " West End Girls` |
| 29 | 0.1447 | 3 | ` The Rocky Mountain Horse` |
| 30 | 0.1429 | 4 | `y Ullman Show` |
| 31 | 0.1426 | 3 | ` Force ( RAAF` |
| 32 | 0.1424 | 3 | ` Threepwood` |
| 33 | 0.1418 | 10 | ` " One Sweet Day` |
| 34 | 0.1409 | 3 | ` Unbreakable Kim` |
| 35 | 0.1391 | 3 | ` " Loverboy` |
| 36 | 0.1387 | 3 | ` the New Forest coven` |
| 37 | 0.1376 | 3 | ` hexafluoropl` |
| 38 | 0.1369 | 4 | ` Simpsons Guide` |
| 39 | 0.1363 | 5 | ` Casualties of Cool` |
| 40 | 0.1362 | 3 | ` Joe : Retaliation` |

### `chat`

18.82M tokens &middot; HuggingFaceH4/ultrachat_200k (train_sft)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 132 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2355 | 3 | ` if (!filter_var` |
| 2 | 0.2293 | 4 | `.add(layers.D` |
| 3 | 0.2107 | 3 | `').magnificPopup` |
| 4 | 0.2019 | 4 | ` Output: The file` |
| 5 | 0.1567 | 4 | `, 123 Main` |
| 6 | 0.1408 | 3 | `s: repeat(auto` |
| 7 | 0.1337 | 4 | `; use Ada.Text` |
| 8 | 0.1317 | 5 | ` html\nhtml(l` |
| 9 | 0.1247 | 4 | `ood.com/recipe` |
| 10 | 0.1208 | 4 | `Eye of the Tiger` |
| 11 | 0.1184 | 24 | `FADE TO BLACK` |
| 12 | 0.1163 | 3 | `plt.tight_layout` |
| 13 | 0.1150 | 3 | ` Anna Dello Russo` |
| 14 | 0.1129 | 3 | ` to UMass Lowell` |
| 15 | 0.1123 | 3 | `specialchars($_SERVER` |
| 16 | 0.1117 | 11 | `The Art of Poss` |
| 17 | 0.1112 | 3 | ` of sound and fury` |
| 18 | 0.1111 | 12 | `CMAKE_C_COMP` |
| 19 | 0.1108 | 3 | ` them. Bon App` |
| 20 | 0.1103 | 3 | ` Initiative for Consumer Hortic` |
| 21 | 0.1102 | 29 | `NESS WHEREOF` |
| 22 | 0.1096 | 4 | `orris-Pratt` |
| 23 | 0.1092 | 3 | `. Representations and Warrant` |
| 24 | 0.1090 | 6 | ` 54th Massachusetts` |
| 25 | 0.1082 | 8 | `\n- Total Carbohyd` |
| 26 | 0.1079 | 4 | ` Swachh Bharat` |
| 27 | 0.1073 | 3 | ` data from OpenWeather` |
| 28 | 0.1072 | 3 | `i House of Worship` |
| 29 | 0.1065 | 3 | `om It May Concern` |
| 30 | 0.1064 | 4 | `_date <- Sys.Date` |
| 31 | 0.1064 | 3 | ` Morgana Le Fay` |
| 32 | 0.1062 | 4 | ` or Pinot Grig` |
| 33 | 0.1059 | 5 | `/Resources/Private` |
| 34 | 0.1058 | 3 | ` the Centre for Addiction` |
| 35 | 0.1055 | 9 | `Don't Stop Belie` |
| 36 | 0.1055 | 11 | ` "$extract_dir` |
| 37 | 0.1054 | 7 | `NOW, THEREFORE` |
| 38 | 0.1054 | 7 | ` the Stolen Valor` |
| 39 | 0.1054 | 9 | ` "To His Coy` |
| 40 | 0.1053 | 4 | ` the UMass Lowell` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 138 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2980 | 3 | `Field, SubmitField` |
| 2 | 0.2646 | 3 | `from flask_wtf` |
| 3 | 0.2254 | 3 | `(FlaskForm` |
| 4 | 0.2167 | 4 | `cloud import WordCloud` |
| 5 | 0.2154 | 3 | ` textblob import` |
| 6 | 0.2131 | 3 | `.hidden_tag()` |
| 7 | 0.2119 | 3 | `api.openweathermap` |
| 8 | 0.2114 | 3 | ` email.mime.base` |
| 9 | 0.2079 | 3 | ` = canvas.Can` |
| 10 | 0.2031 | 3 | `').magnificPopup` |
| 11 | 0.1924 | 3 | ` new PHPMail` |
| 12 | 0.1918 | 3 | `xml = simplexml` |
| 13 | 0.1910 | 3 | `-geolocation-service` |
| 14 | 0.1888 | 9 | `2.0 feed` |
| 15 | 0.1851 | 4 | `cloud = WordCloud` |
| 16 | 0.1829 | 3 | `alphavantage.co` |
| 17 | 0.1817 | 3 | `")\nauth.set_access` |
| 18 | 0.1806 | 3 | ` “Games at Twilight` |
| 19 | 0.1767 | 3 | `Swallow Me Whole` |
| 20 | 0.1750 | 3 | `ators import DataRequired` |
| 21 | 0.1700 | 4 | `:http/http.dart` |
| 22 | 0.1680 | 3 | `epy.API` |
| 23 | 0.1673 | 3 | ` Table, TableStyle` |
| 24 | 0.1671 | 3 | ` @ManyToMany` |
| 25 | 0.1651 | 3 | `    if form.validate` |
| 26 | 0.1630 | 4 | `; use Ada.Text` |
| 27 | 0.1603 | 3 | ` email.mime.text` |
| 28 | 0.1585 | 5 | ` of porphyrobl` |
| 29 | 0.1569 | 3 | `SecurityConfigurerAdapter` |
| 30 | 0.1561 | 3 | ` = test_input($_` |
| 31 | 0.1559 | 3 | `s: repeat(auto` |
| 32 | 0.1540 | 5 | ` com.google.api.services` |
| 33 | 0.1530 | 4 | ` UIDatePicker` |
| 34 | 0.1527 | 3 | `-native-vector-icons` |
| 35 | 0.1523 | 3 | ` form.hidden_tag` |
| 36 | 0.1522 | 4 | `itches of East End` |
| 37 | 0.1521 | 4 | ` UISearchBar` |
| 38 | 0.1520 | 4 | ` to Bookreporter` |
| 39 | 0.1520 | 3 | `import { AngularFire` |
| 40 | 0.1503 | 3 | ` Crimes & Punishments` |

### `math`

1.18M tokens &middot; openai/gsm8k (main)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 130 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.0965 | 9 | `g of packing peanuts` |
| 2 | 0.0839 | 3 | ` pints of frozen yogurt` |
| 3 | 0.0816 | 4 | ` pint of frozen yogurt` |
| 4 | 0.0791 | 3 | ` The Fancy Salon` |
| 5 | 0.0790 | 3 | ` 10 Baby Ruth` |
| 6 | 0.0762 | 4 | `, the snail kite` |
| 7 | 0.0751 | 3 | ` 5 medium pizz` |
| 8 | 0.0739 | 4 | ` bag of jellybeans` |
| 9 | 0.0735 | 3 | ` boxes of birdseed` |
| 10 | 0.0733 | 3 | ` 47 baby carrots` |
| 11 | 0.0730 | 3 | ` of apples and oranges` |
| 12 | 0.0727 | 3 | ` the total square footage` |
| 13 | 0.0726 | 4 | ` Yummy Dog Kib` |
| 14 | 0.0722 | 7 | ` How many jellybeans` |
| 15 | 0.0714 | 4 | ` of $100 bills` |
| 16 | 0.0709 | 3 | ` ride the roller coaster` |
| 17 | 0.0708 | 3 | ` 3 snapdrag` |
| 18 | 0.0707 | 3 | ` cup of birdseed` |
| 19 | 0.0703 | 3 | ` as many jellybeans` |
| 20 | 0.0700 | 3 | `2000 pinecones` |
| 21 | 0.0699 | 4 | ` Art of the Deal` |
| 22 | 0.0698 | 3 | ` three flights of stairs` |
| 23 | 0.0696 | 5 | ` as many water balloons` |
| 24 | 0.0694 | 7 | ` number of blue marbles` |
| 25 | 0.0690 | 3 | ` number of pinecones` |
| 26 | 0.0688 | 3 | ` red and blue marbles` |
| 27 | 0.0687 | 281 | `*2=<<` |
| 28 | 0.0687 | 7 | ` Shark Bite Cove` |
| 29 | 0.0687 | 3 | `’s Woodworking` |
| 30 | 0.0683 | 3 | `12>>12 chocol` |
| 31 | 0.0683 | 3 | ` to make applesauce` |
| 32 | 0.0682 | 6 | ` 2 large pizz` |
| 33 | 0.0682 | 3 | ` Game of Thrones` |
| 34 | 0.0680 | 3 | ` number of red marbles` |
| 35 | 0.0680 | 3 | `aroni and cheese` |
| 36 | 0.0679 | 3 | `3 flights of stairs` |
| 37 | 0.0679 | 3 | ` play the alto sax` |
| 38 | 0.0678 | 4 | ` how many jelly beans` |
| 39 | 0.0676 | 3 | ` 3 poodles` |
| 40 | 0.0675 | 4 | ` to the Nile Delta` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 83 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.1669 | 3 | ` three times as many` |
| 2 | 0.1148 | 3 | `At the beginning of` |
| 3 | 0.0973 | 3 | ` a new pair of` |
| 4 | 0.0906 | 3 | `The total number of` |
| 5 | 0.0904 | 3 | `7 years older than` |
| 6 | 0.0851 | 14 | ` twice as old as` |
| 7 | 0.0835 | 3 | ` times the number of` |
| 8 | 0.0813 | 19 | ` times as old as` |
| 9 | 0.0807 | 3 | ` 3 times older` |
| 10 | 0.0794 | 15 | ` hours = $<<` |
| 11 | 0.0794 | 3 | ` of an adult ticket` |
| 12 | 0.0788 | 3 | ` is in charge of` |
| 13 | 0.0787 | 3 | ` gallons = $<<` |
| 14 | 0.0774 | 3 | ` books = $<<` |
| 15 | 0.0771 | 4 | ` tickets = $<<` |
| 16 | 0.0769 | 3 | ` shirts = $<<` |
| 17 | 0.0765 | 5 | `250 = $<<` |
| 18 | 0.0758 | 3 | ` fish in his aquarium` |
| 19 | 0.0757 | 6 | `9 = $<<` |
| 20 | 0.0757 | 9 | `17 = $<<` |
| 21 | 0.0753 | 9 | `11 = $<<` |
| 22 | 0.0752 | 16 | ` months = $<<` |
| 23 | 0.0751 | 25 | `% = $<<` |
| 24 | 0.0747 | 6 | ` pounds = $<<` |
| 25 | 0.0745 | 129 | `100 = $<<` |
| 26 | 0.0745 | 24 | `000 = $<<` |
| 27 | 0.0742 | 3 | ` of boys to girls` |
| 28 | 0.0740 | 13 | `/hour = $<<` |
| 29 | 0.0740 | 3 | `54 = $<<` |
| 30 | 0.0738 | 20 | `75 = $<<` |
| 31 | 0.0738 | 5 | `48 = $<<` |
| 32 | 0.0737 | 49 | `7 = $<<` |
| 33 | 0.0737 | 4 | `160 = $<<` |
| 34 | 0.0736 | 14 | `/week = $<<` |
| 35 | 0.0736 | 44 | `30 = $<<` |
| 36 | 0.0735 | 3 | `. How many dollars` |
| 37 | 0.0734 | 84 | `6 = $<<` |
| 38 | 0.0734 | 62 | `12 = $<<` |
| 39 | 0.0733 | 9 | `14 = $<<` |
| 40 | 0.0733 | 10 | `/day = $<<` |

### `code_mbpp`

0.03M tokens &middot; google-research-datasets/mbpp (full)

#### Layer 1 (`layer_hash_index` 0)

Showing 4 of 4 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.0869 | 3 | ` SumOfPrimeDivisors` |
| 2 | 0.0839 | 3 | ` result = list(filter` |
| 3 | 0.0802 | 3 | ` function to calucl` |
| 4 | 0.0740 | 4 | ` = list(filter(lambda` |

#### Layer 14 (`layer_hash_index` 1)

Showing 9 of 9 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2238 | 3 | ` calculate the sum of` |
| 2 | 0.1998 | 9 | ` find the sum of` |
| 3 | 0.1897 | 3 | ` the sum of all` |
| 4 | 0.1866 | 178 | `Write a function to` |
| 5 | 0.1823 | 4 | ` to check if the` |
| 6 | 0.1778 | 6 | ` to check whether the` |
| 7 | 0.1718 | 27 | ` function to find the` |
| 8 | 0.1564 | 3 | ` to find sum of` |
| 9 | 0.1515 | 3 | ` function to calculate the` |

## Chinese

### `wiki_zh`

18.82M tokens &middot; wikimedia/wikipedia 20231101.zh

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 123 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.1241 | 4 | ` Angiosperm Phylogen` |
| 2 | 0.1129 | 3 | `rawberry Fields Forever` |
| 3 | 0.1098 | 3 | `KING OF ZIPANGU` |
| 4 | 0.1097 | 3 | `:異塵餘` |
| 5 | 0.1079 | 6 | `军委、中央文革` |
| 6 | 0.1077 | 3 | `韓;朝鲜民主主义` |
| 7 | 0.1066 | 48 | `、江青反革命` |
| 8 | 0.1046 | 3 | `三子胤祉` |
| 9 | 0.1043 | 6 | `、乌孜别克` |
| 10 | 0.1037 | 3 | `、苏联最高苏维埃` |
| 11 | 0.1019 | 3 | `.141592653` |
| 12 | 0.1015 | 4 | `克里米亚�` |
| 13 | 0.1015 | 3 | ` Structure of Scientific Revol` |
| 14 | 0.1013 | 4 | ` 池田大作` |
| 15 | 0.1007 | 3 | `Unicode注音符` |
| 16 | 0.1005 | 38 | `as IV Philopat` |
| 17 | 0.1004 | 3 | `畑健二郎` |
| 18 | 0.0995 | 3 | `", The Stanford Encyclopedia` |
| 19 | 0.0987 | 10 | `《十面埋伏` |
| 20 | 0.0986 | 9 | `」《春光乍` |
| 21 | 0.0986 | 7 | `《开罗宣言` |
| 22 | 0.0985 | 7 | `经·海内经` |
| 23 | 0.0984 | 3 | `（按姓氏笔画` |
| 24 | 0.0980 | 3 | `刻拍案惊奇` |
| 25 | 0.0979 | 8 | `南（泉漳` |
| 26 | 0.0978 | 5 | `中国科学院紫金山天文` |
| 27 | 0.0977 | 6 | `class="wikitable` |
| 28 | 0.0972 | 7 | `HD 209458` |
| 29 | 0.0972 | 3 | ` at Project Gutenberg` |
| 30 | 0.0972 | 3 | `\n潘震宙` |
| 31 | 0.0971 | 5 | `九省通衢` |
| 32 | 0.0971 | 9 | ` 《中国伦理学史` |
| 33 | 0.0970 | 4 | `稿）》向全会` |
| 34 | 0.0970 | 4 | `中华人民共和国、朝鲜民主主义` |
| 35 | 0.0966 | 3 | ` of the Divine Comedy` |
| 36 | 0.0965 | 4 | `. The Monroe Doctrine` |
| 37 | 0.0965 | 73 | ` class="wikitable` |
| 38 | 0.0964 | 4 | `（Autodromo` |
| 39 | 0.0964 | 3 | `高校基础能力建设工程` |
| 40 | 0.0963 | 3 | `领导人、朝鲜民主主义` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 123 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2466 | 3 | `-7167-` |
| 2 | 0.2413 | 3 | `3-406-` |
| 3 | 0.2363 | 14 | `0-393-` |
| 4 | 0.2261 | 3 | `0-345-` |
| 5 | 0.2240 | 3 | `-8021-` |
| 6 | 0.2186 | 7 | `0-684-` |
| 7 | 0.2116 | 3 | `0-385-` |
| 8 | 0.2101 | 3 | `-8050-` |
| 9 | 0.2088 | 4 | `0-375-` |
| 10 | 0.2037 | 3 | ` The World Factbook` |
| 11 | 0.1969 | 12 | `0-201-` |
| 12 | 0.1917 | 5 | ` Septuagint` |
| 13 | 0.1914 | 6 | `0-679-` |
| 14 | 0.1905 | 3 | `0-316-` |
| 15 | 0.1794 | 3 | `戒急用忍` |
| 16 | 0.1785 | 8 | `0-471-` |
| 17 | 0.1752 | 4 | `Folsom Prison` |
| 18 | 0.1707 | 3 | `0-387-` |
| 19 | 0.1676 | 3 | `蒙兀儿史记` |
| 20 | 0.1674 | 3 | `何處不相逢` |
| 21 | 0.1665 | 11 | `0-520-` |
| 22 | 0.1626 | 5 | `《西楚霸王` |
| 23 | 0.1616 | 3 | `-7923-` |
| 24 | 0.1586 | 6 | `中部。市境` |
| 25 | 0.1584 | 4 | `olsom Prison Blues` |
| 26 | 0.1580 | 5 | `《伶人往事` |
| 27 | 0.1555 | 3 | `4.4BSD` |
| 28 | 0.1554 | 4 | `國立第四中山` |
| 29 | 0.1536 | 22 | `部。市境` |
| 30 | 0.1516 | 3 | ` Nag Hammadi Library` |
| 31 | 0.1514 | 3 | `'s Solar System Exploration` |
| 32 | 0.1508 | 3 | `《没有别的爱` |
| 33 | 0.1501 | 4 | `冬冬的假期` |
| 34 | 0.1492 | 6 | `口通商章程` |
| 35 | 0.1491 | 6 | `东部。市境` |
| 36 | 0.1477 | 3 | ` Folsom Prison` |
| 37 | 0.1468 | 4 | `#define lchild rt` |
| 38 | 0.1465 | 6 | `使三浦梧` |
| 39 | 0.1460 | 19 | `0-415-` |
| 40 | 0.1449 | 3 | `《横扫一切牛` |

### `chat_zh`

18.82M tokens &middot; BelleGroup/train_1M_CN

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 96 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.1214 | 3 | `image.png](attachment` |
| 2 | 0.1125 | 27 | ` a truth universally acknowledged` |
| 3 | 0.1083 | 4 | `《荒岛余生` |
| 4 | 0.1075 | 7 | `.141592653` |
| 5 | 0.1062 | 17 | `xpialidocious` |
| 6 | 0.1052 | 16 | `_list = list(set` |
| 7 | 0.1044 | 57 | `hemian Rhaps` |
| 8 | 0.1037 | 6 | `Don't Stop Belie` |
| 9 | 0.1035 | 4 | ` "Thinking Out Loud` |
| 10 | 0.1033 | 17 | `《机器人总动员` |
| 11 | 0.1033 | 21 | `性和不可篡改性` |
| 12 | 0.1032 | 13 | ` Bohemian Rhaps` |
| 13 | 0.1030 | 8 | `idisestablishmentarianism` |
| 14 | 0.1023 | 3 | `：Ian Goodfellow` |
| 15 | 0.1022 | 3 | `    except ZeroDivision` |
| 16 | 0.1013 | 8 | `    return list(set` |
| 17 | 0.1012 | 3 | `. 《统计学习方法` |
| 18 | 0.1008 | 4 | `让人忍俊不禁` |
| 19 | 0.1006 | 3 | ` = sorted(list(set` |
| 20 | 0.1003 | 11 | `: 123 Main` |
| 21 | 0.1003 | 15 | `《雾都孤儿` |
| 22 | 0.1001 | 6 | `01001 011` |
| 23 | 0.0998 | 16 | `唐代诗人王之涣` |
| 24 | 0.0982 | 7 | `《球状闪电` |
| 25 | 0.0982 | 4 | `To Kill a Mock` |
| 26 | 0.0979 | 3 | ` 机器人总动员` |
| 27 | 0.0979 | 3 | `Habitat for Humanity` |
| 28 | 0.0977 | 20 | ` vectorizer.fit_transform` |
| 29 | 0.0977 | 9 | `：《沉默的羔` |
| 30 | 0.0975 | 5 | `嗷嗷待哺` |
| 31 | 0.0973 | 68 | `寻梦环游记` |
| 32 | 0.0973 | 6 | `lst = list(set` |
| 33 | 0.0969 | 4 | `st = list(set` |
| 34 | 0.0969 | 11 | `和《最后的晚餐` |
| 35 | 0.0966 | 4 | ` = cosine_similar` |
| 36 | 0.0965 | 39 | `join(random.choice(char` |
| 37 | 0.0963 | 4 | `result = list(set` |
| 38 | 0.0962 | 4 | `- W3Schools` |
| 39 | 0.0958 | 8 | `. HackerRank` |
| 40 | 0.0956 | 10 | `- FreeCodeCamp` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 94 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2846 | 4 | ` email.mime.text` |
| 2 | 0.2674 | 3 | `pus import wordnet` |
| 3 | 0.2284 | 15 | `k.corpus` |
| 4 | 0.2131 | 5 | `pus import stopwords` |
| 5 | 0.2038 | 3 | `ize import word_token` |
| 6 | 0.2035 | 3 | `.zero_grad` |
| 7 | 0.2000 | 7 | `_20newsgroups` |
| 8 | 0.1904 | 25 | `ature_extraction.text` |
| 9 | 0.1882 | 4 | ` elit. Nulla` |
| 10 | 0.1829 | 7 | ` = load_iris` |
| 11 | 0.1771 | 3 | `fidfTransformer` |
| 12 | 0.1734 | 3 | `etrics.pairwise` |
| 13 | 0.1675 | 7 | `NetLemmatizer` |
| 14 | 0.1639 | 20 | ` vectorizer.fit_transform` |
| 15 | 0.1627 | 7 | ` MIMEText` |
| 16 | 0.1527 | 3 | ` keras.models import Sequential` |
| 17 | 0.1506 | 16 | `. Aliquam` |
| 18 | 0.1495 | 35 | ` sklearn.model_selection` |
| 19 | 0.1478 | 5 | `imedia.org/wikipedia` |
| 20 | 0.1476 | 3 | `. Pellentesque` |
| 21 | 0.1472 | 23 | `. Suspendisse` |
| 22 | 0.1396 | 4 | `ib.SMTP` |
| 23 | 0.1391 | 19 | ` = CountVectorizer` |
| 24 | 0.1384 | 34 | `lection import train_test` |
| 25 | 0.1375 | 3 | `《控方证人` |
| 26 | 0.1367 | 16 | `from sklearn.datasets` |
| 27 | 0.1361 | 25 | `etrics import accuracy_score` |
| 28 | 0.1358 | 4 | `ltk.pos_tag` |
| 29 | 0.1351 | 9 | `.maketrans` |
| 30 | 0.1346 | 4 | ` googletrans` |
| 31 | 0.1305 | 7 | ` a Snowy Evening` |
| 32 | 0.1303 | 3 | `from keras.layers` |
| 33 | 0.1293 | 3 | `reg = LinearRegression` |
| 34 | 0.1292 | 6 | `^&*()` |
| 35 | 0.1269 | 35 | ` import train_test_split` |
| 36 | 0.1268 | 5 | ` WordNetLemmat` |
| 37 | 0.1267 | 99 | `, consectetur adipiscing elit` |
| 38 | 0.1264 | 37 | `choices(string.ascii` |
| 39 | 0.1259 | 5 | `/wikipedia/commons` |
| 40 | 0.1252 | 11 | `.choice(string.ascii` |

### `web_zh`

18.82M tokens &middot; HuggingFaceFW/fineweb-2 cmn_Hani

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 97 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.1912 | 9 | ` Suites by Hilton` |
| 2 | 0.1282 | 18 | `玄奘西游记` |
| 3 | 0.1208 | 9 | `cgi.blog.ro` |
| 4 | 0.1206 | 3 | `：乌有之乡` |
| 5 | 0.1160 | 7 | `非美国通用会计准则` |
| 6 | 0.1082 | 14 | `www.wretch.cc` |
| 7 | 0.1072 | 3 | `顿酒店及度假` |
| 8 | 0.1066 | 18 | `和《春风沉醉` |
| 9 | 0.1064 | 3 | `，不管三七二十一` |
| 10 | 0.1062 | 3 | `本文来源：网易` |
| 11 | 0.1056 | 6 | `美輪美�` |
| 12 | 0.1056 | 6 | ` Emma, Forever Ago` |
| 13 | 0.1055 | 3 | `按照美国通用会计准则` |
| 14 | 0.1054 | 7 | `地下城与勇士` |
| 15 | 0.1048 | 3 | `【MyGoNews` |
| 16 | 0.1043 | 8 | `b/824684` |
| 17 | 0.1040 | 3 | `.tw.tranews` |
| 18 | 0.1036 | 3 | `Lord of the Rings` |
| 19 | 0.1033 | 3 | `、营业性歌舞` |
| 20 | 0.1031 | 35 | `，文责自负` |
| 21 | 0.1029 | 7 | `.com.cn/s/blog` |
| 22 | 0.1028 | 76 | `Submit()">登` |
| 23 | 0.1027 | 19 | `《我的野蛮女友` |
| 24 | 0.1023 | 7 | `://www.nownews` |
| 25 | 0.1022 | 4 | `《金粉世家` |
| 26 | 0.1021 | 3 | `化契約應記載` |
| 27 | 0.1016 | 11 | `.wretch.cc/blog` |
| 28 | 0.1012 | 33 | `惊心食人族` |
| 29 | 0.1012 | 16 | `。\n- developerWorks` |
| 30 | 0.1011 | 4 | `.researchandmarkets` |
| 31 | 0.1010 | 3 | ` 作文网 www` |
| 32 | 0.1007 | 3 | `3/PhysRev` |
| 33 | 0.1004 | 26 | ` © United States Holocaust` |
| 34 | 0.1001 | 3 | `，眼睜睜` |
| 35 | 0.0999 | 12 | `。\nWinterIsComing` |
| 36 | 0.0996 | 4 | `_CLASSES_ROOT` |
| 37 | 0.0995 | 13 | `真心话大冒险` |
| 38 | 0.0995 | 3 | `；多采多姿` |
| 39 | 0.0994 | 3 | `/wps/portal` |
| 40 | 0.0994 | 3 | `《卿本佳人` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 109 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2448 | 10 | `ICP备150037` |
| 2 | 0.2386 | 11 | ` 未经授权禁止` |
| 3 | 0.2317 | 4 | `本文仅代表作者` |
| 4 | 0.2296 | 6 | `僅代表作者本人` |
| 5 | 0.2266 | 8 | `本文僅代表作者` |
| 6 | 0.2255 | 10 | `沪ICP备150` |
| 7 | 0.2242 | 3 | `微信扫一扫分享` |
| 8 | 0.2224 | 3 | ` 2002-` |
| 9 | 0.2174 | 15 | ` 2010-` |
| 10 | 0.2079 | 5 | `0-2015` |
| 11 | 0.2077 | 39 | `京ICP备100` |
| 12 | 0.2042 | 8 | ` 2007-` |
| 13 | 0.2007 | 30 | `2-200503` |
| 14 | 0.1977 | 3 | `，请注明出处` |
| 15 | 0.1953 | 10 | ` \| 加入收藏` |
| 16 | 0.1928 | 3 | `聲明】本文` |
| 17 | 0.1906 | 3 | ` 刁卿蕙` |
| 18 | 0.1890 | 34 | `本网部分文章` |
| 19 | 0.1839 | 3 | ` 2004-` |
| 20 | 0.1833 | 3 | `-2014 ` |
| 21 | 0.1830 | 4 | `记者郭庚儒` |
| 22 | 0.1818 | 3 | ` 2005-` |
| 23 | 0.1816 | 10 | `，未经授权禁止` |
| 24 | 0.1815 | 6 | ` you agree to our` |
| 25 | 0.1793 | 3 | ` © 2002` |
| 26 | 0.1779 | 36 | `.net/blog/post/` |
| 27 | 0.1776 | 10 | `oth塔羅牌的` |
| 28 | 0.1759 | 3 | `免责声明：本文` |
| 29 | 0.1752 | 6 | `《蜜桃成熟` |
| 30 | 0.1751 | 4 | ` 2012-` |
| 31 | 0.1744 | 3 | `用微信扫一扫` |
| 32 | 0.1730 | 3 | `.com/fwlink` |
| 33 | 0.1718 | 3 | `沪ICP备102` |
| 34 | 0.1702 | 3 | ` 時季常` |
| 35 | 0.1692 | 10 | `Copyright 2010` |
| 36 | 0.1672 | 4 | `_CLASSES_ROOT` |
| 37 | 0.1671 | 19 | `复制或建立镜像` |
| 38 | 0.1669 | 3 | `「光之穹` |
| 39 | 0.1642 | 3 | `随时关注 developerWorks` |
| 40 | 0.1639 | 15 | `影片《老大不小` |

## Code (CodeSearchNet)

### `code_python`

18.82M tokens &middot; code_search_net (python)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 90 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.4313 | 7 | `    response = ur` |
| 2 | 0.4298 | 16 | `        response = ur` |
| 3 | 0.4296 | 12 | `            response = ur` |
| 4 | 0.4059 | 3 | ` None) or get` |
| 5 | 0.3934 | 3 | `        with open(re` |
| 6 | 0.3163 | 22 | `get_context_data(**` |
| 7 | 0.3058 | 4 | `:\n            progress_c` |
| 8 | 0.2833 | 7 | ` logging.disable(log` |
| 9 | 0.2808 | 13 | ` pdb; p` |
| 10 | 0.2788 | 3 | ` = ctypes.p` |
| 11 | 0.2726 | 12 | ` if self.Verb` |
| 12 | 0.2711 | 10 | ` if self.verb` |
| 13 | 0.2637 | 4 | `        ctypes.p` |
| 14 | 0.2568 | 4 | `)\n    except File` |
| 15 | 0.2554 | 13 | `)\n        except File` |
| 16 | 0.2524 | 13 | `(datetime.t` |
| 17 | 0.2461 | 4 | ` cluster\n\n        Rais` |
| 18 | 0.2452 | 36 | ` None\n        Rais` |
| 19 | 0.2405 | 14 | `.subplot2grid` |
| 20 | 0.2404 | 7 | `            if self.sc` |
| 21 | 0.2402 | 31 | `cio.ensure_f` |
| 22 | 0.2393 | 16 | `        if self.sc` |
| 23 | 0.2355 | 145 | ` value\n\n        Rais` |
| 24 | 0.2354 | 3 | ` found\n\n        Rais` |
| 25 | 0.2306 | 3 | ` values\n    Rais` |
| 26 | 0.2250 | 3 | ` n\n\n    Rais` |
| 27 | 0.2242 | 8 | `\r\n    Rais` |
| 28 | 0.2193 | 3 | `.\n        \n        Rais` |
| 29 | 0.2176 | 3 | `        service : str` |
| 30 | 0.2082 | 47 | ` and self.verb` |
| 31 | 0.2059 | 5 | `err = p.com` |
| 32 | 0.2035 | 3 | `0\n            Rais` |
| 33 | 0.2006 | 6 | `.user_id == user` |
| 34 | 0.1470 | 7 | ` for attempt in range` |
| 35 | 0.1454 | 4 | `one, re.split` |
| 36 | 0.1401 | 9 | ` in settings.INST` |
| 37 | 0.1350 | 3 | `        self.Verb` |
| 38 | 0.1284 | 3 | `src = tf.shape` |
| 39 | 0.1274 | 5 | ` if key == ord` |
| 40 | 0.1215 | 4 | `    >>> dis.dis` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 111 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.3090 | 7 | ` getopt.getopt` |
| 2 | 0.2994 | 5 | `aginate_quer` |
| 3 | 0.2961 | 3 | ` from pykechain` |
| 4 | 0.2835 | 3 | `(add_help=False` |
| 5 | 0.2812 | 16 | `parse.ArgumentDefaults` |
| 6 | 0.2761 | 3 | ` GetQueuedCompletion` |
| 7 | 0.2719 | 4 | `(allow_no_value` |
| 8 | 0.2640 | 50 | `.add_subparsers` |
| 9 | 0.2601 | 7 | `        from nefert` |
| 10 | 0.2598 | 3 | ` >>> from caspo` |
| 11 | 0.2591 | 7 | `ast.NodeVisitor` |
| 12 | 0.2555 | 6 | ` >>> from dwave` |
| 13 | 0.2537 | 4 | `, sharex=True` |
| 14 | 0.2466 | 3 | ` import schedula` |
| 15 | 0.2456 | 3 | ` >>> from ibeis` |
| 16 | 0.2447 | 7 | ` from dwave.cloud` |
| 17 | 0.2421 | 13 | ` >>> import dcor` |
| 18 | 0.2417 | 4 | ` of the Auto Scaling` |
| 19 | 0.2415 | 3 | `.com/questions/147` |
| 20 | 0.2408 | 6 | `    from nefert` |
| 21 | 0.2303 | 3 | ` This is the main` |
| 22 | 0.2294 | 4 | `    pysat` |
| 23 | 0.2294 | 3 | `: an cogent` |
| 24 | 0.2290 | 3 | `arnings(record=True` |
| 25 | 0.2288 | 16 | `        **Key Arguments` |
| 26 | 0.2286 | 3 | ` information on how to` |
| 27 | 0.2277 | 40 | `ually_exclusive_group` |
| 28 | 0.2272 | 3 | ` --pretty=format` |
| 29 | 0.2262 | 6 | `_name="parsl` |
| 30 | 0.2262 | 3 | `2E6B` |
| 31 | 0.2253 | 13 | ` = imp.find_module` |
| 32 | 0.2218 | 8 | `File(delete=False` |
| 33 | 0.2215 | 10 | ` from peltak` |
| 34 | 0.2214 | 3 | ` pycuda.g` |
| 35 | 0.2210 | 4 | `_mako_plus` |
| 36 | 0.2208 | 15 | ` from pycbc` |
| 37 | 0.2205 | 4 | ` asynchronously applied` |
| 38 | 0.2196 | 3 | `developer.wunderlist` |
| 39 | 0.2183 | 11 | `abstract_osid` |
| 40 | 0.2182 | 21 | `        from dlkit` |

### `code_javascript`

18.82M tokens &middot; code_search_net (javascript)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 94 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2961 | 3 | ` = req.app.loc` |
| 2 | 0.2558 | 3 | `        if (!ne` |
| 3 | 0.2451 | 62 | `OpenLayers.G` |
| 4 | 0.2418 | 183 | ` OpenLayers.G` |
| 5 | 0.2357 | 9 | `err, rows)` |
| 6 | 0.2334 | 7 | ` {\n                document.exec` |
| 7 | 0.2250 | 3 | `    return document.d` |
| 8 | 0.2225 | 3 | `').css('op` |
| 9 | 0.2198 | 3 | ` if (indexPath` |
| 10 | 0.2158 | 3 | `*/regeneratorRuntime` |
| 11 | 0.2122 | 157 | ` @memberOf _\n` |
| 12 | 0.2094 | 4 | `);\n      gl.bl` |
| 13 | 0.1907 | 15 | `    if (!basic` |
| 14 | 0.1853 | 4 | `        $scope.st` |
| 15 | 0.1845 | 4 | ` (request, reply` |
| 16 | 0.1819 | 3 | `) => __awa` |
| 17 | 0.1664 | 13 | `      if (!basic` |
| 18 | 0.1654 | 3 | `\t\t\t\treturn _results` |
| 19 | 0.1605 | 3 | ` refreshToken, profile` |
| 20 | 0.1468 | 8 | ` _regeneratorRuntime` |
| 21 | 0.1424 | 3 | ` SECONDS_PER_MIN` |
| 22 | 0.1335 | 3 | `;\n                    _scroll` |
| 23 | 0.1324 | 3 | `_len = Number.MAX` |
| 24 | 0.1309 | 3 | `[0]).to` |
| 25 | 0.1263 | 14 | `($.proxy(function` |
| 26 | 0.1262 | 3 | ` min = Number.MAX` |
| 27 | 0.1234 | 11 | `iterator.next()).done` |
| 28 | 0.1224 | 8 | `ION */}.call` |
| 29 | 0.1207 | 69 | `classCallCheck(this` |
| 30 | 0.1196 | 4 | ` && Object.prototype.toString` |
| 31 | 0.1193 | 12 | `name ui.router` |
| 32 | 0.1184 | 7 | ` = newPromiseCap` |
| 33 | 0.1178 | 4 | `Dist = Number.MAX` |
| 34 | 0.1174 | 5 | `Datatable.aiDisplay` |
| 35 | 0.1174 | 26 | ` instanceof AST_Symbol` |
| 36 | 0.1165 | 12 | `.tmpdir(),` |
| 37 | 0.1162 | 40 | ` * @param {...` |
| 38 | 0.1160 | 46 | ` (Object.prototype.toString` |
| 39 | 0.1159 | 34 | ` = window.getSelection` |
| 40 | 0.1152 | 3 | `            return Number.MAX` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 114 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.3158 | 3 | `.stdin.setRawMode` |
| 2 | 0.3137 | 11 | `      bindToController` |
| 3 | 0.2917 | 3 | ` context.createScriptProcessor` |
| 4 | 0.2892 | 7 | ` domainManager.registerDomain` |
| 5 | 0.2770 | 3 | `.com/questions/588` |
| 6 | 0.2675 | 12 | ` args = [], len` |
| 7 | 0.2669 | 4 | `            bindToController` |
| 8 | 0.2663 | 53 | ` * @owner Observable` |
| 9 | 0.2609 | 3 | ` = $rdf` |
| 10 | 0.2569 | 13 | `    bindToController` |
| 11 | 0.2561 | 6 | `ator.getGamep` |
| 12 | 0.2538 | 3 | `. DocuSign` |
| 13 | 0.2517 | 9 | `('@google-cloud` |
| 14 | 0.2445 | 3 | ` Keeps track of` |
| 15 | 0.2438 | 3 | ` "fancytree` |
| 16 | 0.2408 | 13 | ` OpenSeadragon` |
| 17 | 0.2402 | 3 | `.com/questions/144` |
| 18 | 0.2398 | 4 | ` = new mindmaps` |
| 19 | 0.2383 | 3 | `module:echarts` |
| 20 | 0.2376 | 3 | `.occlusionTexture` |
| 21 | 0.2365 | 3 | ` Determines if the` |
| 22 | 0.2354 | 4 | `/*jshint` |
| 23 | 0.2342 | 6 | ` __generator(this` |
| 24 | 0.2333 | 3 | ` n.enumerable \|\|\n` |
| 25 | 0.2325 | 7 | `Crafty.c` |
| 26 | 0.2321 | 7 | `wysihtml` |
| 27 | 0.2305 | 6 | ` OpenLayers.P` |
| 28 | 0.2290 | 6 | `ptxGenJS` |
| 29 | 0.2287 | 3 | `\t// calculate the` |
| 30 | 0.2273 | 38 | ` exports, __web` |
| 31 | 0.2264 | 9 | ` = new WorldWind` |
| 32 | 0.2253 | 4 | `OR.dialogCommand` |
| 33 | 0.2253 | 3 | ` user-scalable` |
| 34 | 0.2225 | 3 | `isStrictCompar` |
| 35 | 0.2223 | 10 | ` this.adjustBounds` |
| 36 | 0.2222 | 4 | `object.byteLength !=` |
| 37 | 0.2210 | 6 | `\t\t * Get the` |
| 38 | 0.2196 | 3 | ` = svgedit` |
| 39 | 0.2188 | 3 | ` "development" !==` |
| 40 | 0.2184 | 3 | `(/[xy]/` |

### `code_java`

18.82M tokens &middot; code_search_net (java)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 92 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.4231 | 10 | `.class).in(S` |
| 2 | 0.3558 | 6 | `SystemService(Context.W` |
| 3 | 0.2932 | 3 | `)\n            throws Throw` |
| 4 | 0.2861 | 25 | `().obtainStyled` |
| 5 | 0.2771 | 6 | `, RelativeLayout.T` |
| 6 | 0.2741 | 18 | `(ByteOrder.L` |
| 7 | 0.2732 | 31 | ` KeyStroke(Key` |
| 8 | 0.2718 | 8 | `, CascadeType.M` |
| 9 | 0.2701 | 34 | ` defaultstate="coll` |
| 10 | 0.2685 | 4 | `.TitledBorder.D` |
| 11 | 0.2682 | 4 | `Uri = Maps.new` |
| 12 | 0.2676 | 25 | ` throws IOException, SA` |
| 13 | 0.2676 | 23 | ` javax.swing.JCom` |
| 14 | 0.2616 | 3 | `.compress(Bit` |
| 15 | 0.2614 | 3 | `ing = Maps.new` |
| 16 | 0.2604 | 174 | `Constraints.fill = java` |
| 17 | 0.2584 | 9 | ` .obtainStyled` |
| 18 | 0.2561 | 5 | ` throws IOException,SA` |
| 19 | 0.2561 | 11 | `.obtainStyled` |
| 20 | 0.2503 | 4 | ` map = Maps.new` |
| 21 | 0.2458 | 36 | `.CONTENT_L` |
| 22 | 0.2445 | 5 | `Path().equals(l` |
| 23 | 0.2444 | 3 | `Type().equals(L` |
| 24 | 0.2433 | 19 | `.DAY_OF_Y` |
| 25 | 0.2380 | 223 | `Performed(ev` |
| 26 | 0.2311 | 4 | `.setEntity(new Url` |
| 27 | 0.2294 | 18 | `        synchronized (list` |
| 28 | 0.2238 | 3 | `    synchronized (list` |
| 29 | 0.2205 | 318 | `.event.ActionEvent ev` |
| 30 | 0.2161 | 3 | `.id().equals(l` |
| 31 | 0.2126 | 3 | ` )\n\t\t\t\t\tint alt` |
| 32 | 0.2126 | 61 | ` )\n\t\t\tint alt` |
| 33 | 0.2041 | 693 | ` new javax.swing.J` |
| 34 | 0.1711 | 61 | `.setLayout(new java` |
| 35 | 0.1491 | 33 | `_INTERNAL_S` |
| 36 | 0.1479 | 14 | `XException, P` |
| 37 | 0.1421 | 8 | ` ExecutionEnvironment.getExecution` |
| 38 | 0.1420 | 3 | `();\n        scanner.sc` |
| 39 | 0.1341 | 9 | ` SECONDS_PER_MIN` |
| 40 | 0.1311 | 3 | `Zone)super.clone` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 108 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.3044 | 3 | ` = new JarOutputStream` |
| 2 | 0.2972 | 3 | ` = new StreamTokenizer` |
| 3 | 0.2822 | 4 | `.apache.xml.serial` |
| 4 | 0.2788 | 3 | ` conn.setDoOutput` |
| 5 | 0.2724 | 9 | `.com/classgraph` |
| 6 | 0.2710 | 3 | `        setSupportAction` |
| 7 | 0.2676 | 21 | ` = new ZipOutputStream` |
| 8 | 0.2673 | 4 | `.getSystemJavaCompiler` |
| 9 | 0.2649 | 3 | `conn.setDoOutput` |
| 10 | 0.2608 | 3 | `org.javasimon` |
| 11 | 0.2597 | 7 | `CloseOperation(jav` |
| 12 | 0.2592 | 5 | ` new LineNumberReader` |
| 13 | 0.2589 | 3 | ` = pm.newQuery` |
| 14 | 0.2584 | 3 | `Systems.newFileSystem` |
| 15 | 0.2584 | 3 | `Connection.setDoOutput` |
| 16 | 0.2545 | 8 | ` env = ExecutionEnvironment` |
| 17 | 0.2531 | 3 | ` = new JarInputStream` |
| 18 | 0.2522 | 7 | ` = new HelpFormatter` |
| 19 | 0.2512 | 27 | `ridBagConstraints grid` |
| 20 | 0.2507 | 14 | `org.voltd` |
| 21 | 0.2496 | 3 | `XTRA_OUTPUT` |
| 22 | 0.2475 | 3 | ` XSSFReader` |
| 23 | 0.2468 | 9 | `.igormaz` |
| 24 | 0.2435 | 8 | `.calimero` |
| 25 | 0.2430 | 7 | ` new SAXBuilder` |
| 26 | 0.2413 | 3 | `.obtainStyled` |
| 27 | 0.2405 | 24 | `cher.appendReplacement` |
| 28 | 0.2399 | 7 | `is.getNextEntry` |
| 29 | 0.2390 | 8 | `.stanford.n` |
| 30 | 0.2374 | 4 | `PSignatureGenerator` |
| 31 | 0.2363 | 3 | ` tFactory.newTransformer` |
| 32 | 0.2363 | 3 | ` getRevokedCertificate` |
| 33 | 0.2348 | 4 | ` = new Deflater` |
| 34 | 0.2344 | 6 | ` = new ClassReader` |
| 35 | 0.2343 | 20 | `Generated Code">//` |
| 36 | 0.2340 | 3 | `Factory = CertificateFactory` |
| 37 | 0.2320 | 7 | `("Unexpected exception` |
| 38 | 0.2300 | 7 | ` = SecretKeyFactory` |
| 39 | 0.2293 | 9 | ` org.voltd` |
| 40 | 0.2286 | 11 | `.remoting.prot` |

### `code_go`

18.82M tokens &middot; code_search_net (go)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 71 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.3039 | 22 | `Prefix(strings.To` |
| 2 | 0.2978 | 3 | `\t\t\tcontext.Back` |
| 3 | 0.2950 | 3 | `\t\tcontext.Back` |
| 4 | 0.2921 | 3 | `("could not mars` |
| 5 | 0.2823 | 27 | ` := err.(aw` |
| 6 | 0.2775 | 7 | ` idx := strings.Last` |
| 7 | 0.2756 | 10 | `idx := strings.Last` |
| 8 | 0.2717 | 11 | `, base64.St` |
| 9 | 0.2711 | 29 | ` = base64.St` |
| 10 | 0.2617 | 6 | `: base64.St` |
| 11 | 0.2586 | 81 | ` context.WithCancel(context` |
| 12 | 0.2537 | 21 | `\treturn base64.St` |
| 13 | 0.2509 | 10 | `] ", log.L` |
| 14 | 0.2482 | 109 | ` := base64.St` |
| 15 | 0.2458 | 14 | `.Split(r.URL` |
| 16 | 0.2445 | 104 | `rrors = append` |
| 17 | 0.2416 | 17 | `\tbase64.St` |
| 18 | 0.2402 | 16 | `.HistogramOpt` |
| 19 | 0.2325 | 33 | ` := &http.S` |
| 20 | 0.2293 | 4 | ` += base64.St` |
| 21 | 0.2233 | 95 | `MockRecorder{m` |
| 22 | 0.2228 | 44 | `, err: %` |
| 23 | 0.2218 | 21 | `}, bson.M` |
| 24 | 0.2212 | 18 | ` fmt.Sprintf("Unable` |
| 25 | 0.2181 | 83 | `\t}\n\n\tlog.D` |
| 26 | 0.2120 | 92 | `\t}\n\tlog.D` |
| 27 | 0.2012 | 83 | ` = os.Stat` |
| 28 | 0.2012 | 5 | `)\n\n\td.SetId` |
| 29 | 0.1893 | 39 | ` := recover(); err` |
| 30 | 0.1850 | 7 | ` + base64.St` |
| 31 | 0.1825 | 3 | `Updates, payload.S` |
| 32 | 0.1803 | 32 | ` == syscall.E` |
| 33 | 0.1578 | 3 | `\treturn ErrInvalidLength` |
| 34 | 0.1576 | 34 | `\tif strings.EqualFold` |
| 35 | 0.1477 | 429 | `(logrus.Fields` |
| 36 | 0.1454 | 28 | ` credentials.AnonymousCredentials` |
| 37 | 0.1410 | 4 | ` if strings.EqualFold` |
| 38 | 0.1337 | 3 | ` !strings.EqualFold` |
| 39 | 0.1309 | 59 | `, _ := pem` |
| 40 | 0.1306 | 6 | ` fmt.Errorf("proto` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 108 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2869 | 3 | `RateLimitingQueue` |
| 2 | 0.2822 | 7 | ` fsnotify.New` |
| 3 | 0.2795 | 7 | `onboulle` |
| 4 | 0.2702 | 3 | ` := csv.NewWriter` |
| 5 | 0.2697 | 15 | ` := multipart.New` |
| 6 | 0.2685 | 4 | `.GenerateFromPassword` |
| 7 | 0.2672 | 53 | ` := x509.New` |
| 8 | 0.2552 | 6 | `MULTISIG` |
| 9 | 0.2551 | 3 | `) DeleteBucketAnal` |
| 10 | 0.2504 | 3 | `(jwt.Signing` |
| 11 | 0.2489 | 37 | ` flag.NewFlagSet` |
| 12 | 0.2468 | 4 | ` DeleteFieldLevelEnc` |
| 13 | 0.2402 | 6 | ` grpc.NewServer` |
| 14 | 0.2396 | 4 | ` authInfo runtime.Client` |
| 15 | 0.2376 | 6 | `SetTerminationProt` |
| 16 | 0.2373 | 146 | ` bufio.NewScanner` |
| 17 | 0.2371 | 3 | `-Allow-Methods` |
| 18 | 0.2361 | 4 | `ssh.ServerConfig` |
| 19 | 0.2359 | 7 | `.SystemCertPool` |
| 20 | 0.2336 | 18 | `http.Hijacker` |
| 21 | 0.2319 | 3 | `gulacsi` |
| 22 | 0.2318 | 4 | `) Collect(ch chan` |
| 23 | 0.2310 | 8 | `(vdemeester` |
| 24 | 0.2310 | 3 | `CreateConditionalForward` |
| 25 | 0.2306 | 3 | ` := swag.Read` |
| 26 | 0.2267 | 3 | `// Figure out the` |
| 27 | 0.2261 | 11 | ` *caddy.Cont` |
| 28 | 0.2255 | 7 | `ify.NewWatcher` |
| 29 | 0.2254 | 3 | ` We'll start by` |
| 30 | 0.2252 | 5 | `.com/goadesign` |
| 31 | 0.2249 | 3 | ` api.NewOpenStorage` |
| 32 | 0.2247 | 3 | ` := html.NewTokenizer` |
| 33 | 0.2244 | 3 | `) SetVoiceConn` |
| 34 | 0.2237 | 62 | `\t// map header` |
| 35 | 0.2237 | 6 | `CreateDBClusterEndpoint` |
| 36 | 0.2228 | 4 | ` := parser.ParseDir` |
| 37 | 0.2220 | 3 | ` := clientcmd.New` |
| 38 | 0.2189 | 16 | `) DecodeMsg` |
| 39 | 0.2185 | 3 | `\t// Fetch the` |
| 40 | 0.2183 | 6 | `DeleteConditionalForward` |

### `code_php`

18.82M tokens &middot; code_search_net (php)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 97 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.4486 | 28 | `' => $in` |
| 2 | 0.2852 | 6 | ` {\n                    return Pel` |
| 3 | 0.2848 | 3 | ` {\n            return Pel` |
| 4 | 0.2630 | 3 | `$components['sche` |
| 5 | 0.2602 | 3 | ` $components['sche` |
| 6 | 0.2517 | 28 | ` $this->Coll` |
| 7 | 0.2464 | 30 | `            ->getRepository` |
| 8 | 0.2456 | 6 | `                    ->getRepository` |
| 9 | 0.2427 | 3 | ` if(!filter_var` |
| 10 | 0.2427 | 9 | `                ->getRepository` |
| 11 | 0.2418 | 661 | `:\n                            return Pel` |
| 12 | 0.2412 | 60 | ` '0', STR` |
| 13 | 0.2396 | 3 | `:\n                return Pel` |
| 14 | 0.2387 | 114 | `:\n                        return Pel` |
| 15 | 0.2346 | 1620 | ` $this->coll` |
| 16 | 0.2336 | 4 | ` if (! filter_var` |
| 17 | 0.2330 | 26 | ` if (!filter_var` |
| 18 | 0.2329 | 3 | ` if(! filter_var` |
| 19 | 0.2315 | 8 | `$table->tim` |
| 20 | 0.2278 | 4 | ` "Authorization: Basic` |
| 21 | 0.2258 | 5 | `::create('oauth` |
| 22 | 0.2231 | 132 | ` $table->tim` |
| 23 | 0.2172 | 7 | `user_id == user` |
| 24 | 0.2165 | 4 | ` $_SERVER["REM` |
| 25 | 0.1500 | 4 | `);\n                return Pel` |
| 26 | 0.1402 | 4 | `SECONDS_PER_MIN` |
| 27 | 0.1398 | 14 | `_INTERNAL_S` |
| 28 | 0.1370 | 46 | `author Vova Feldman` |
| 29 | 0.1370 | 23 | ` json_encode($request` |
| 30 | 0.1255 | 10 | `ParentQuery::create` |
| 31 | 0.1225 | 4 | `this->MultiCell` |
| 32 | 0.1214 | 3 | ` add_meta_box` |
| 33 | 0.1202 | 3 | `_rate_ibfk` |
| 34 | 0.1197 | 3 | `id, PHP_URL` |
| 35 | 0.1194 | 6 | `\tadd_meta_box` |
| 36 | 0.1194 | 4 | ` 'Authorization: Basic` |
| 37 | 0.1189 | 3 | `([T_STRING` |
| 38 | 0.1186 | 3 | `rong( __METHOD` |
| 39 | 0.1158 | 3 | `e = preg_split` |
| 40 | 0.1157 | 3 | `descendant-or-self` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 100 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.3275 | 3 | ` (\\MvcCore` |
| 2 | 0.3249 | 13 | `ap_get_entries` |
| 3 | 0.3212 | 12 | ` = token_get_all` |
| 4 | 0.3152 | 8 | `sl_pkey_new` |
| 5 | 0.3076 | 11 | `process = proc_open` |
| 6 | 0.3039 | 3 | ` curl_multi_init` |
| 7 | 0.3036 | 6 | `        stream_set_time` |
| 8 | 0.3029 | 46 | `author Vova Feldman` |
| 9 | 0.2997 | 9 | ` (token_get_all` |
| 10 | 0.2931 | 17 | `        // parse inputs` |
| 11 | 0.2923 | 13 | ` [];\n        $_header` |
| 12 | 0.2923 | 44 | `, $makeNew` |
| 13 | 0.2893 | 3 | `ar->startBuff` |
| 14 | 0.2879 | 3 | `proc = proc_open` |
| 15 | 0.2731 | 20 | `_HEADER_OUT` |
| 16 | 0.2693 | 4 | `An error has occurred` |
| 17 | 0.2678 | 85 | `File->getTokens` |
| 18 | 0.2664 | 33 | ` Vova Feldman (@` |
| 19 | 0.2659 | 7 | `->isDeleted())` |
| 20 | 0.2652 | 13 | `ALC_FOUND` |
| 21 | 0.2640 | 8 | ` \\MakinaCor` |
| 22 | 0.2633 | 18 | `, $joinBehavior` |
| 23 | 0.2620 | 3 | `// First we'll` |
| 24 | 0.2587 | 5 | `ulti_add_handle` |
| 25 | 0.2561 | 8 | ` = set_error_handler` |
| 26 | 0.2556 | 4 | `('<?xml encoding` |
| 27 | 0.2502 | 18 | `context = stream_context` |
| 28 | 0.2458 | 3 | `class = eZ` |
| 29 | 0.2451 | 5 | `Model = Gdn` |
| 30 | 0.2426 | 3 | `Could not find the` |
| 31 | 0.2417 | 3 | `, like Gecko` |
| 32 | 0.2395 | 4 | ` = wp_remote` |
| 33 | 0.2391 | 3 | ` ldap_connect` |
| 34 | 0.2390 | 3 | `(\\Jivoo` |
| 35 | 0.2363 | 13 | ` imagettfbbox` |
| 36 | 0.2360 | 33 | ` \\MvcCore` |
| 37 | 0.2347 | 3 | `ock = socket_create` |
| 38 | 0.2339 | 3 | `\t// Prepare the` |
| 39 | 0.2335 | 7 | ` imagecreatefrompng` |
| 40 | 0.2332 | 5 | `Fields(FormMapper` |

### `code_ruby`

5.87M tokens &middot; code_search_net (ruby)

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 104 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.3133 | 3 | `http://id.loc` |
| 2 | 0.2716 | 17 | ` = Base64.st` |
| 3 | 0.2651 | 3 | `: Base64.st` |
| 4 | 0.2555 | 4 | `      Base64.st` |
| 5 | 0.2467 | 3 | ` += Base64.st` |
| 6 | 0.2387 | 10 | ` req.send_request(options` |
| 7 | 0.2251 | 3 | ` + Base64.st` |
| 8 | 0.1288 | 13 | ` 1 && args` |
| 9 | 0.1234 | 14 | `      return broadcast(:` |
| 10 | 0.1141 | 5 | ` = APIHelper.json` |
| 11 | 0.1128 | 6 | ` = Vector3.new` |
| 12 | 0.1093 | 3 | `('about:blank` |
| 13 | 0.1090 | 18 | `\n\n      promise.execute` |
| 14 | 0.1090 | 3 | `_node = Struct.new` |
| 15 | 0.1081 | 3 | `\n          instance_e` |
| 16 | 0.1066 | 3 | `, but Simon Sap` |
| 17 | 0.1052 | 7 | `-lang.org/issues` |
| 18 | 0.1022 | 10 | `c = Proc.new` |
| 19 | 0.1019 | 3 | `        attr_reader` |
| 20 | 0.1008 | 4 | `2005/Atom` |
| 21 | 0.1004 | 6 | `\n        flash.now` |
| 22 | 0.1003 | 8 | `.decode_www_form` |
| 23 | 0.0996 | 6 | `RakeTask.new` |
| 24 | 0.0987 | 3 | `::TooManyRedirect` |
| 25 | 0.0986 | 3 | `::MethodNotAllowed` |
| 26 | 0.0985 | 123 | `ocop:disable` |
| 27 | 0.0982 | 6 | `plan = Proc.new` |
| 28 | 0.0982 | 34 | ` ActiveSupport::Notifications` |
| 29 | 0.0974 | 4 | `\n        module_function` |
| 30 | 0.0973 | 3 | `.keys.each_with_object` |
| 31 | 0.0971 | 7 | ` c = Class.new` |
| 32 | 0.0969 | 4 | `connect = Proc.new` |
| 33 | 0.0968 | 45 | `.encode_www_form` |
| 34 | 0.0965 | 3 | ` OP_HASH160` |
| 35 | 0.0960 | 3 | `] = Proc.new` |
| 36 | 0.0958 | 5 | ` hash.each_with_object` |
| 37 | 0.0957 | 3 | `_update = Proc.new` |
| 38 | 0.0957 | 3 | ` resources.extract_options` |
| 39 | 0.0953 | 11 | `\n        instance_e` |
| 40 | 0.0948 | 3 | `raw.each_with_object` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 106 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2313 | 31 | ` = OptionParser.new` |
| 2 | 0.2016 | 3 | ` = Savon.client` |
| 3 | 0.1896 | 51 | `::XmlMarkup` |
| 4 | 0.1848 | 4 | `MULTISIG` |
| 5 | 0.1842 | 5 | `::ReplicationProtected` |
| 6 | 0.1758 | 3 | `1-09-` |
| 7 | 0.1756 | 6 | ` => 'caboose` |
| 8 | 0.1735 | 3 | `        'AmazonAuthorization` |
| 9 | 0.1731 | 4 | ` OctocatalogDiff` |
| 10 | 0.1689 | 4 | `.sf.jasper` |
| 11 | 0.1688 | 3 | ` = Solargraph` |
| 12 | 0.1685 | 4 | ` = Openwsman` |
| 13 | 0.1677 | 3 | ` Hpricot` |
| 14 | 0.1675 | 8 | `Access-Control-Allow` |
| 15 | 0.1658 | 3 | ` beaker-hostgener` |
| 16 | 0.1641 | 23 | `SSL::Cipher` |
| 17 | 0.1618 | 8 | ` scanner = StringScanner` |
| 18 | 0.1601 | 3 | `' << Axlsx` |
| 19 | 0.1593 | 4 | ` rescue CouchRest` |
| 20 | 0.1586 | 4 | `Trebuchet MS` |
| 21 | 0.1571 | 3 | `:twoCellAnchor` |
| 22 | 0.1560 | 3 | `\n      SemanticLogger` |
| 23 | 0.1550 | 3 | ` raise Octocatalog` |
| 24 | 0.1538 | 250 | ` = Net::HTTP` |
| 25 | 0.1536 | 18 | `: [[MsRest` |
| 26 | 0.1523 | 3 | `        raise Synvert` |
| 27 | 0.1522 | 7 | `        Net::HTTP` |
| 28 | 0.1521 | 10 | `SSLSocket.new` |
| 29 | 0.1517 | 3 | ` = MaRuKu` |
| 30 | 0.1516 | 3 | ` response = RSpot` |
| 31 | 0.1514 | 8 | ` raise SSRFProxy` |
| 32 | 0.1506 | 4 | ` google.visualization` |
| 33 | 0.1501 | 86 | ` args.extract_options` |
| 34 | 0.1500 | 14 | `::XML::Builder` |
| 35 | 0.1497 | 4 | ` response = TaxCloud` |
| 36 | 0.1479 | 4 | ` CelluloidPub` |
| 37 | 0.1478 | 3 | `c:varyColors` |
| 38 | 0.1478 | 5 | `_types << Axlsx` |
| 39 | 0.1476 | 3 | ` ['khipu` |
| 40 | 0.1469 | 21 | ` = MnoEnterprise` |

## Depth corpora

### `web`

18.82M tokens &middot; HuggingFaceFW/fineweb-edu sample-10BT

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 128 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.1911 | 7 | `This note was uploaded` |
| 2 | 0.1651 | 3 | `.\nOther nearby markers` |
| 3 | 0.1212 | 4 | ` Encyclopedia of World Biography` |
| 4 | 0.1178 | 3 | ` Elders of Zion` |
| 5 | 0.1172 | 3 | `DAILY BEA` |
| 6 | 0.1166 | 3 | `′ W. Marker` |
| 7 | 0.1154 | 3 | ` Digital Journal of Ophthalm` |
| 8 | 0.1120 | 3 | ` to the Vaccine Adverse` |
| 9 | 0.1113 | 19 | ` [1913 Webster` |
| 10 | 0.1108 | 3 | ` Advanced Camera for Surveys` |
| 11 | 0.1098 | 4 | ` Illustrated Encyclopedia of Hinduism` |
| 12 | 0.1089 | 3 | ` of Modern Human Origins` |
| 13 | 0.1083 | 4 | ` under the topic Early` |
| 14 | 0.1083 | 3 | ` "Access to Insight` |
| 15 | 0.1081 | 3 | ` of the Austrian Success` |
| 16 | 0.1080 | 3 | `," Bacon\'s Rebellion` |
| 17 | 0.1076 | 3 | ` the Trail of Tears` |
| 18 | 0.1072 | 3 | ` EBSCO Publishing` |
| 19 | 0.1071 | 8 | ` of Studies on Alcohol` |
| 20 | 0.1070 | 3 | ` the Academy of Nutrition` |
| 21 | 0.1068 | 5 | ` The Feminine Myst` |
| 22 | 0.1063 | 12 | `ist Mennonite Encyclopedia` |
| 23 | 0.1062 | 7 | ` Nihil Obstat` |
| 24 | 0.1061 | 6 | `, by Bill Cough` |
| 25 | 0.1045 | 5 | ` Association of Clinical Endocr` |
| 26 | 0.1045 | 5 | ` Viennese Walt` |
| 27 | 0.1044 | 3 | `\|\|New Georgia Encyclopedia` |
| 28 | 0.1035 | 3 | ` of Angkor Thom` |
| 29 | 0.1035 | 12 | `ard Sadi Carn` |
| 30 | 0.1035 | 3 | ` Bioluminescence Resonance` |
| 31 | 0.1034 | 4 | `iço de Prote` |
| 32 | 0.1032 | 3 | ` Importance of Being Earn` |
| 33 | 0.1030 | 3 | ` Journal of Applied Meteor` |
| 34 | 0.1030 | 3 | `American College of Obst` |
| 35 | 0.1028 | 3 | `. Arch Pediatr Adoles` |
| 36 | 0.1028 | 15 | `.\nThe Columbia Electronic` |
| 37 | 0.1026 | 3 | `, The Southern Poverty` |
| 38 | 0.1025 | 7 | ` Kirkcudbright` |
| 39 | 0.1024 | 26 | ` Dictionary and Cyclopedia` |
| 40 | 0.1023 | 8 | `Published on PsychCentral` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 112 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2744 | 3 | ` Johannes Gutenberg University` |
| 2 | 0.2685 | 14 | ` this article is licensed` |
| 3 | 0.2617 | 3 | ` EBSCO Publishing` |
| 4 | 0.2513 | 4 | ` to review this product` |
| 5 | 0.2436 | 3 | `0-393-` |
| 6 | 0.2421 | 4 | `Causes, incidence` |
| 7 | 0.2414 | 3 | ` This article is for` |
| 8 | 0.2358 | 6 | ` The World Factbook` |
| 9 | 0.2329 | 3 | `-7603-` |
| 10 | 0.2279 | 6 | ` must be logged in` |
| 11 | 0.2260 | 6 | ` continue reading this article` |
| 12 | 0.2244 | 3 | `0-517-` |
| 13 | 0.2217 | 10 | ` The rateable annual` |
| 14 | 0.2176 | 5 | ` 1996-` |
| 15 | 0.2162 | 3 | ` Boomerang Books` |
| 16 | 0.2151 | 3 | ` California, Santa Cruz` |
| 17 | 0.2140 | 6 | `-8032-` |
| 18 | 0.2084 | 3 | ` for information only and` |
| 19 | 0.2047 | 3 | ` OPTIMIZE TABLE` |
| 20 | 0.2025 | 3 | `ailing to Byzantium` |
| 21 | 0.2024 | 21 | ` 1995-` |
| 22 | 0.2014 | 3 | ` post a comment.` |
| 23 | 0.2003 | 17 | `This work is licensed` |
| 24 | 0.2002 | 3 | ` to post a comment` |
| 25 | 0.1990 | 5 | ` Story of an Hour` |
| 26 | 0.1969 | 9 | ` by Disqus` |
| 27 | 0.1968 | 8 | ` not intended to be` |
| 28 | 0.1967 | 7 | ` the first to review` |
| 29 | 0.1958 | 3 | `3-642-` |
| 30 | 0.1949 | 6 | ` is for information only` |
| 31 | 0.1917 | 4 | ` page was last updated` |
| 32 | 0.1915 | 21 | ` is licensed under a` |
| 33 | 0.1905 | 3 | `th Trench Mort` |
| 34 | 0.1898 | 3 | ` Tree of Life Web` |
| 35 | 0.1889 | 3 | ` by Robert Wayne Atkins` |
| 36 | 0.1879 | 3 | ` be logged in to` |
| 37 | 0.1877 | 3 | `3-540-` |
| 38 | 0.1864 | 5 | ` Low-Density Sup` |
| 39 | 0.1863 | 3 | `To Build a Fire` |
| 40 | 0.1857 | 3 | ` Australia's Online Independent` |

### `wiki_full`

18.82M tokens &middot; wikimedia/wikipedia 20231101.en

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 116 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.1480 | 19 | ` La Haye Sainte` |
| 2 | 0.1350 | 5 | `, Asaf Jah` |
| 3 | 0.1293 | 3 | ` Dilwale Dul` |
| 4 | 0.1183 | 3 | ` Croix de guerre` |
| 5 | 0.1180 | 4 | `Angiosperm Phylogen` |
| 6 | 0.1176 | 22 | ` Dream of Electric Sheep` |
| 7 | 0.1150 | 3 | `. Mustafa Khat` |
| 8 | 0.1150 | 16 | ` Angiosperm Phylogen` |
| 9 | 0.1141 | 11 | ` Abominable Snow` |
| 10 | 0.1111 | 12 | ` Elders of Zion` |
| 11 | 0.1106 | 3 | `I'm Only Sleeping` |
| 12 | 0.1104 | 3 | ` On Rotten Tomatoes` |
| 13 | 0.1099 | 10 | ` website Rotten Tomatoes` |
| 14 | 0.1095 | 13 | ` of the Austrian Success` |
| 15 | 0.1095 | 3 | ` to Regimental Combat` |
| 16 | 0.1092 | 14 | `sortable wikitable` |
| 17 | 0.1091 | 7 | `ator Rotten Tomatoes` |
| 18 | 0.1087 | 15 | ` On You Crazy Diamond` |
| 19 | 0.1081 | 3 | `'s Imago Mund` |
| 20 | 0.1081 | 4 | ` the Trail of Tears` |
| 21 | 0.1077 | 3 | `, Camille Claud` |
| 22 | 0.1074 | 3 | `, Action Against Hunger` |
| 23 | 0.1073 | 6 | `ughters of Mary Help` |
| 24 | 0.1070 | 3 | `The Murder of Roger` |
| 25 | 0.1070 | 3 | ` Six Degrees of Separation` |
| 26 | 0.1069 | 7 | ` fictional Dunder Mifflin` |
| 27 | 0.1067 | 3 | `.141592653` |
| 28 | 0.1067 | 3 | ` Missionaries of Charity` |
| 29 | 0.1066 | 3 | `, Blazing Sadd` |
| 30 | 0.1066 | 12 | ` Murder of Roger Ack` |
| 31 | 0.1065 | 3 | ` the Virtuti Milit` |
| 32 | 0.1064 | 7 | `World Series of Poker` |
| 33 | 0.1063 | 3 | ` The Tragically Hip` |
| 34 | 0.1062 | 6 | ` The Spectacular Spider` |
| 35 | 0.1057 | 4 | `-Siberian Orchestra` |
| 36 | 0.1056 | 7 | ` on Rotten Tomatoes` |
| 37 | 0.1055 | 3 | ` Baron De La Warr` |
| 38 | 0.1053 | 5 | ` of Blazing Sadd` |
| 39 | 0.1050 | 28 | ` Bout Fatal Fury` |
| 40 | 0.1047 | 3 | ` Nusrat Fate` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 131 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2807 | 3 | `, had an enrollment` |
| 2 | 0.2305 | 3 | `, Queensland, Australia` |
| 3 | 0.2272 | 3 | `'s Finest Comics` |
| 4 | 0.2261 | 3 | ` of the Actor model` |
| 5 | 0.2253 | 3 | `In general, the` |
| 6 | 0.2247 | 47 | ` The World Factbook` |
| 7 | 0.2230 | 3 | `The definition of the` |
| 8 | 0.2183 | 3 | ` of the municipal coat` |
| 9 | 0.2183 | 3 | ` article incorporates text from` |
| 10 | 0.2090 | 6 | ` than any other location` |
| 11 | 0.2077 | 3 | ` a German association football` |
| 12 | 0.2063 | 12 | ` a list of the` |
| 13 | 0.2035 | 3 | ` of the Loyal Legion` |
| 14 | 0.2017 | 3 | ` responsibility for local issues` |
| 15 | 0.2002 | 3 | ` Anabaptist` |
| 16 | 0.2001 | 55 | ` (Eastern Orthodox lit` |
| 17 | 0.1997 | 4 | `Internet censorship and surveillance` |
| 18 | 0.1996 | 4 | ` the South Fork Fishing` |
| 19 | 0.1988 | 3 | ` Spacetime Odyssey` |
| 20 | 0.1961 | 3 | ` of the Formigas` |
| 21 | 0.1956 | 4 | `-Birkenau` |
| 22 | 0.1932 | 3 | `The Acropolis of` |
| 23 | 0.1931 | 11 | `16th Street Baptist` |
| 24 | 0.1912 | 3 | `The station first signed` |
| 25 | 0.1898 | 3 | ` population) who belonged` |
| 26 | 0.1884 | 4 | `ory Dickory Dock` |
| 27 | 0.1881 | 4 | ` The Pleasure Garden` |
| 28 | 0.1870 | 9 | ` school had an enrolment` |
| 29 | 0.1864 | 3 | ` Interferometry Mission` |
| 30 | 0.1863 | 3 | ` Harland and Wolff` |
| 31 | 0.1859 | 3 | `-mandatory upper` |
| 32 | 0.1841 | 3 | `IR/SHAKEN` |
| 33 | 0.1840 | 15 | ` Sea region of Turkey` |
| 34 | 0.1837 | 6 | ` "Alice\'s Restaurant` |
| 35 | 0.1836 | 3 | ` Hymn to Aph` |
| 36 | 0.1833 | 3 | ` El Shoqafa` |
| 37 | 0.1831 | 3 | ` Rise and Fall of` |
| 38 | 0.1816 | 3 | ` Historic Site of Canada` |
| 39 | 0.1808 | 3 | `The status of the` |
| 40 | 0.1804 | 3 | ` Sabbath Bloody Sabbath` |

### `math_web`

18.82M tokens &middot; open-web-math/open-web-math

#### Layer 1 (`layer_hash_index` 0)

Showing 40 of 93 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.2524 | 15 | `mathrm{new}}(\\` |
| 2 | 0.2452 | 11 | `\n\n Thread Tools Display` |
| 3 | 0.2445 | 8 | `:=PCGroup([` |
| 4 | 0.2354 | 6 | `\n# A tib` |
| 5 | 0.2342 | 3 | `901960784314` |
| 6 | 0.2266 | 5 | `.add(layers.D` |
| 7 | 0.2167 | 3 | `'ll also get unlimited` |
| 8 | 0.2070 | 20 | `5/rdf http` |
| 9 | 0.2066 | 57 | `\nOpen Detailed Calendar` |
| 10 | 0.1942 | 42 | ` to jurisdictional claims in` |
| 11 | 0.1876 | 3 | ` EmilValued Senior` |
| 12 | 0.1863 | 10 | ` eLearning\n\nPosted` |
| 13 | 0.1806 | 4 | `1Q     Median` |
| 14 | 0.1560 | 3 | ` $F_p(T` |
| 15 | 0.1537 | 38 | ` available in Project Eucl` |
| 16 | 0.1516 | 4 | `## # A tib` |
| 17 | 0.1515 | 8 | `Non-Human User` |
| 18 | 0.1515 | 23 | ` $\\square$\n\nLemma` |
| 19 | 0.1445 | 3 | `## Hecke kernels` |
| 20 | 0.1393 | 66 | `\n\n### GMAT Club` |
| 21 | 0.1392 | 5 | ` grDevices utils` |
| 22 | 0.1357 | 19 | `}{l}\\require` |
| 23 | 0.1340 | 5 | `}\n\\ExplSyntax` |
| 24 | 0.1333 | 40 | ` HideShow timer Statistics` |
| 25 | 0.1313 | 5 | `ild\nHomework Helper` |
| 26 | 0.1277 | 7 | `\\usepackage{amsfonts` |
| 27 | 0.1269 | 15 | `)\n[TeX:]` |
| 28 | 0.1247 | 6 | `us: Early Transc` |
| 29 | 0.1243 | 4 | ` help from StudySoup` |
| 30 | 0.1240 | 13 | `itus\nHomework Helper` |
| 31 | 0.1237 | 16 | `\nSimilar\nTopics` |
| 32 | 0.1232 | 17 | `: Mike Shulman` |
| 33 | 0.1231 | 9 | ` Calculus: Early Transc` |
| 34 | 0.1226 | 3 | `\n\nGMAT Club Legend` |
| 35 | 0.1225 | 35 | ` with Beat the GMAT` |
| 36 | 0.1224 | 12 | ` - 0 dislike` |
| 37 | 0.1219 | 3 | `4\n\nKudos [?` |
| 38 | 0.1215 | 3 | `://golem.ph` |
| 39 | 0.1213 | 5 | `AuthorMike Shulman` |
| 40 | 0.1212 | 3 | `Edited by joigus` |

#### Layer 14 (`layer_hash_index` 1)

Showing 40 of 121 ranked 4-grams.

| # | mean gate | count | 4-gram |
| --: | --: | --: | --- |
| 1 | 0.4232 | 8 | `:=PCGroup([` |
| 2 | 0.3799 | 3 | ` The Best Or Nothing` |
| 3 | 0.3606 | 5 | `: 17 Dec` |
| 4 | 0.3591 | 3 | ` QA & VA Forum` |
| 5 | 0.3298 | 3 | `## Hecke characteristic` |
| 6 | 0.3258 | 4 | `\nLocation: Oklahoma` |
| 7 | 0.3256 | 3 | `### M Theory Lesson` |
| 8 | 0.3224 | 3 | `## Hecke kernels` |
| 9 | 0.3217 | 3 | `trichoplax` |
| 10 | 0.3095 | 4 | `: 13 May` |
| 11 | 0.3090 | 7 | `: Pune, India` |
| 12 | 0.3076 | 26 | `.A. Wevers` |
| 13 | 0.3062 | 3 | `### ehrenfest` |
| 14 | 0.2989 | 3 | `\nLocation: London` |
| 15 | 0.2983 | 3 | ` George Gassaway` |
| 16 | 0.2953 | 3 | `\nStatus: QA` |
| 17 | 0.2904 | 7 | ` this article we will` |
| 18 | 0.2902 | 12 | ` Permalink \| Reply` |
| 19 | 0.2900 | 7 | `\nLocation: Pune` |
| 20 | 0.2896 | 3 | `xiomOfChoice` |
| 21 | 0.2882 | 6 | `. See the answer` |
| 22 | 0.2850 | 14 | `## Reading the Comics` |
| 23 | 0.2834 | 5 | `: 02 Aug` |
| 24 | 0.2810 | 3 | `f Sum of Sq` |
| 25 | 0.2799 | 7 | `color{DarkRed` |
| 26 | 0.2775 | 3 | `: 04 Jan` |
| 27 | 0.2770 | 9 | ` my lessons more memorable` |
| 28 | 0.2758 | 9 | `alin Pithwa` |
| 29 | 0.2750 | 4 | ` Inverse of a matrix` |
| 30 | 0.2742 | 65 | ` , given: ` |
| 31 | 0.2730 | 6 | `RSpriggs` |
| 32 | 0.2725 | 3 | ` = tf.placeholder` |
| 33 | 0.2689 | 4 | `arePairedDel` |
| 34 | 0.2686 | 5 | ` the feckin` |
| 35 | 0.2680 | 3 | ` tutorial, we will` |
| 36 | 0.2666 | 3 | `: 21 Jul` |
| 37 | 0.2646 | 3 | ` area of the triangle` |
| 38 | 0.2635 | 7 | `Posted by distler` |
| 39 | 0.2611 | 3 | ` For instance, the` |
| 40 | 0.2611 | 3 | ` $F_p(T` |

---

Generated from `analysis/engram/scan.py` output; merged across shards 0-2.
