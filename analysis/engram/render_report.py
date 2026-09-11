import json

TOP = 40
ORDER = [
    ("Reference-study domains", ["wiki", "chat", "math", "code_mbpp"]),
    ("Chinese", ["wiki_zh", "chat_zh", "web_zh"]),
    ("Code (CodeSearchNet)", ["code_python", "code_javascript", "code_java",
                              "code_go", "code_php", "code_ruby"]),
    ("Depth corpora", ["web", "wiki_full", "math_web"]),
]
SOURCE = {
    "wiki": "Salesforce/wikitext, wikitext-2-raw-v1",
    "chat": "HuggingFaceH4/ultrachat_200k (train_sft)",
    "math": "openai/gsm8k (main)",
    "code_mbpp": "google-research-datasets/mbpp (full)",
    "wiki_zh": "wikimedia/wikipedia 20231101.zh",
    "chat_zh": "BelleGroup/train_1M_CN",
    "web_zh": "HuggingFaceFW/fineweb-2 cmn_Hani",
    "code_python": "code_search_net (python)",
    "code_javascript": "code_search_net (javascript)",
    "code_java": "code_search_net (java)",
    "code_go": "code_search_net (go)",
    "code_php": "code_search_net (php)",
    "code_ruby": "code_search_net (ruby)",
    "web": "HuggingFaceFW/fineweb-edu sample-10BT",
    "wiki_full": "wikimedia/wikipedia 20231101.en",
    "math_web": "open-web-math/open-web-math",
}
LAYER = {0: "Layer 1 (`layer_hash_index` 0)", 1: "Layer 14 (`layer_hash_index` 1)"}


def cell(ngram):
    """Render an n-gram as inline code that survives a markdown table."""
    body = repr(ngram)[1:-1].replace("|", "\\|")
    fence = "``" if "`" in body else "`"
    pad = " " if body.startswith("`") or body.endswith("`") else ""
    return f"{fence}{pad}{body}{pad}{fence}"


import os
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "engram_merged.json")))
tables, tokens = d["tables"], d["tokens"]
out = []
w = out.append

w("# Engram gate activation: top suffix 4-grams")
w("")
w("DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash`, MXFP4 routed experts),")
w("vLLM `deepseekv41-flash-0909`, TP=8 on 3x H100 nodes (8 GPUs each).")
w(f"Run [34583988417](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34583988417),")
w("2026-09-11, 3h47m wall-clock.")
w("")
w("## Method")
w("")
w("The Engram gate is computed inside a fused Triton kernel and consumed in the")
w("same store (`hidden + gate * value`), so it is not observable from a hook.")
w("`analysis/engram/gate_probe.py` wraps `Engram.forward` and recomputes it in")
w("torch from the module's public tensors, mirroring the kernel's arithmetic:")
w("")
w("```")
w("hidden_rms = rsqrt(mean(hidden^2) + eps)")
w("key_rms    = rsqrt(mean(key^2) + eps)")
w("dot        = sum(hidden * q * k * key) * hidden_rms * key_rms * dim^-0.5")
w("gate       = sigmoid(sign(dot) * sqrt(max(|dot|, clamp)))")
w("```")
w("")
w("Per token the gate is averaged over the four hyper-connection copies. Each")
w("domain is read as 3584-token chunks, prefilled with `max_tokens=1`. Within")
w("each (domain, layer) the top 1% of gate values are taken as strong, and every")
w("strong position is attributed to the 2-, 3-, and 4-gram ending at it. Rows")
w("below are ranked by mean gate with a minimum of 3 occurrences; only the")
w("4-gram tier is shown here.")
w("")
w(f"**Corpus: {sum(tokens.values())/1e6:.1f}M tokens over {len(tokens)} domains,")
w("English and Chinese only.**")
w("")
w("| domain | tokens | source |")
w("| --- | --- | --- |")
for _, doms in ORDER:
    for dom in doms:
        w(f"| `{dom}` | {tokens.get(dom,0)/1e6:.2f}M | {SOURCE[dom]} |")
w("")
w("## Caveats")
w("")
w("1. **Rankings are the union of three per-shard top-50 lists**, each with its")
w("   own threshold, not a strict global top-N. An n-gram strong in one shard can")
w("   outrank one moderately strong in all three.")
w("2. **The `chat` domains carry code.** UltraChat and Belle both contain code")
w("   snippets, so those tables are not clean dialogue and do not compare")
w("   directly to a DailyDialog baseline (DailyDialog is script-based and no")
w("   longer loadable).")
w("3. **`code_mbpp` and `math` are small** (0.03M and 1.18M tokens) and exhaust")
w("   long before the chunk cap; their tables are thin by construction.")
w("4. Chinese tokenizes at roughly 1-1.5 characters per token, so a Chinese")
w("   4-gram spans about four characters -- morpheme-level, not the multi-word")
w("   phrases seen in the English tables.")
w("5. Rows whose text decodes mid-character (a lone replacement glyph) are")
w("   tokenizer-boundary artifacts of non-ASCII literals, not real n-grams.")
w("")
w("## Summary: what each layer selects for")
w("")
w("Layer 1 fires on rare surface strings -- proper nouns, unusual identifiers,")
w("continuations with no parametric structure. Layer 14 fires on templated")
w("structure. `math` shows the split cleanly: layer 1 takes the incidental nouns")
w("of a word problem, layer 14 takes the schema (`three times as many`,")
w("`7 years older than`). MBPP repeats it: identifiers at layer 1,")
w("`Write a function to` at layer 14.")
w("")
w("Two specific results:")
w("")
w("- **ISBN publisher prefixes dominate `wiki_zh` layer 14** (`-7167-`, `3-406-`,")
w("  `0-393-`, `0-345-`), and `0-393-` independently reaches the top of English")
w("  `web` layer 14. These are digit sequences with no parametric structure --")
w("  obtainable only by memorization.")
w("- **Chinese layer 14 is boilerplate**: ICP license numbers,")
w("  `未经授权禁止`, `本文仅代表作者`. The same templated-structure behavior as")
w("  English, in a corpus-specific vocabulary.")
w("")

# --- Top 10 overall -------------------------------------------------------
flat = [
    (r["avg_gate"], table, r["ngram"], r["count"])
    for table, rows in tables.items()
    for r in rows
]
w("## Top 10 overall")
w("")
w("Highest mean gate of any 4-gram in the run, across all 16 domains and both")
w("layers. Every one is code or markup: a fixed idiom the tokenizer splits into")
w("several pieces, where the next piece is fully determined by the ones before.")
w("")
w("| # | mean gate | count | domain / layer | 4-gram |")
w("| --: | --: | --: | --- | --- |")
for i, (g, table, ng, c) in enumerate(
    sorted((r for r in flat if r[1].endswith("4gram")), reverse=True)[:10], 1
):
    dom, layer, _ = table.split("/")
    w(f"| {i} | {g:.4f} | {c} | `{dom}` / {layer[-1]} | {cell(ng)} |")
w("")
w("The PHP leader `' => $in` is an array-literal fragment; the three")
w("`response = ur` rows are the same `urllib` call at three indentation depths,")
w("which the tokenizer makes into three distinct 4-grams. `:=PCGroup([` is GAP")
w("computer-algebra syntax from a maths forum.")
w("")

# --- Notable -------------------------------------------------------------
NOTABLE = [
    ("wiki/engram0/4gram", " Wright : Ace Attorney", "The game subtitle, memorized whole."),
    ("wiki/engram0/4gram", ' " Run Run Rudolph', "Chuck Berry, 1958."),
    ("wiki/engram0/4gram", " Treehouse of Horror", "The Simpsons' Halloween episodes."),
    ("wiki/engram1/4gram", "ane Clown Pos", "Insane Clown Posse, mid-token."),
    ("wiki_full/engram1/4gram", " Sabbath Bloody Sabbath", "Black Sabbath, 1973."),
    ("wiki_full/engram1/4gram", " The Spectacular Spider", "...-Man."),
    ("wiki/engram1/4gram", " , Super Mario Land", "Game Boy, 1989."),
    ("wiki/engram1/4gram", "ll Always Have Paris", "Casablanca, via a TNG episode title."),
    ("wiki/engram1/4gram", " Life Is Worth Living", "Fulton Sheen's 1950s TV show."),
    ("chat_zh/engram0/4gram", "xpialidocious", "The tail of supercalifragilistic-."),
    ("chat_zh/engram0/4gram", ".141592653", "Nine digits of pi after the point."),
    ("chat_zh/engram0/4gram", " a truth universally acknowledged", "Pride and Prejudice, opening line."),
    ("wiki_zh/engram0/4gram", "KING OF ZIPANGU", "A 1990s NHK drama's romanized title."),
    ("web_zh/engram0/4gram", "玄奘西游记", "Xuanzang's Journey to the West."),
    ("chat_zh/engram0/4gram", "《荒岛余生", "Cast Away, in Chinese."),
    ("math/engram0/4gram", " pints of frozen yogurt", "A GSM8K word-problem prop."),
    ("math/engram0/4gram", "g of packing peanuts", "Another one."),
    ("math/engram0/4gram", " The Fancy Salon", "An invented GSM8K business."),
    ("math/engram0/4gram", " 10 Baby Ruth", "Candy bars, being counted."),
    ("wiki/engram0/4gram", " Wisteria Lane", "Desperate Housewives."),
    ("wiki_full/engram1/4gram", " Angiosperm Phylogen", "...y Group, the botanical classification."),
    ("web/engram0/4gram", " Elders of Zion", "From a Project Gutenberg catalogue page."),
    ("chat/engram0/4gram", "The Life of Pablo", "Kanye West, 2016."),
]
index = {
    (table, r["ngram"]): r
    for table, rows in tables.items()
    for r in rows
}
w("## Notable entries")
w("")
w("Hand-picked from the tables above, because what the gate opens on is easier")
w("to see in specific cases than in aggregate. These are not the strongest")
w("rows -- they are the legible ones.")
w("")
w("| mean gate | count | domain / layer | 4-gram | what it is |")
w("| --: | --: | --- | --- | --- |")
missing = []
for table, ngram, note in NOTABLE:
    row = index.get((table, ngram))
    if row is None:
        missing.append((table, ngram))
        continue
    dom, layer, _ = table.split("/")
    w(f"| {row['avg_gate']:.4f} | {row['count']} | `{dom}` / {layer[-1]} "
      f"| {cell(ngram)} | {note} |")
assert not missing, f"notable rows not found in the data: {missing}"
w("")
w("The pattern across all of them: a rare multi-token name whose later pieces")
w("are unguessable from the model's weights but fully determined once the")
w("earlier pieces are known. `ane Clown Pos` is the clearest case -- the gate")
w("opens in the middle of a word, on a boundary that exists only because of how")
w("the tokenizer split a band's name. The maths rows show the same mechanism on")
w("invented props: once a GSM8K problem has said \"pints of frozen\", the next")
w("token is not in doubt.")
w("")


for heading, doms in ORDER:
    w(f"## {heading}")
    w("")
    for dom in doms:
        w(f"### `{dom}`")
        w("")
        w(f"{tokens.get(dom,0)/1e6:.2f}M tokens &middot; {SOURCE[dom]}")
        w("")
        for layer in (0, 1):
            rows = tables.get(f"{dom}/engram{layer}/4gram", [])
            w(f"#### {LAYER[layer]}")
            w("")
            if not rows:
                w("_No rows survived the minimum-count filter._")
                w("")
                continue
            shown = rows[:TOP]
            w(f"Showing {len(shown)} of {len(rows)} ranked 4-grams.")
            w("")
            w("| # | mean gate | count | 4-gram |")
            w("| --: | --: | --: | --- |")
            for i, r in enumerate(shown, 1):
                w(f"| {i} | {r['avg_gate']:.4f} | {r['count']} | {cell(r['ngram'])} |")
            w("")
w("---")
w("")
w("Generated from `analysis/engram/scan.py` output; merged across shards 0-2.")
w("")

open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS-4grams.md"), "w").write("\n".join(out))
print("lines:", len(out))
