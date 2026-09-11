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

open("engram-4grams.md", "w").write("\n".join(out))
print("lines:", len(out))
