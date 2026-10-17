#!/usr/bin/env bash
# Forward cached_tokens_details through SGLang's multi-tokenizer path, which
# older releases drop, so AIPerf sees per-request cache hits.
set -euo pipefail
file=/sgl-workspace/sglang/python/sglang/srt/managers/multi_tokenizer_mixin.py
if ! sed -n '/elif isinstance(output, BatchStrOutput):/,/input_token_logprobs_val=_extract_field_by_index/p' "$file" \
    | grep -q 'cached_tokens_details=_extract_field_by_index'; then
    sed -i '/elif isinstance(output, BatchStrOutput):/,/input_token_logprobs_val=_extract_field_by_index/ {
        /cached_tokens=_extract_field_by_index(output, "cached_tokens", i),/a\
            cached_tokens_details=_extract_field_by_index(\
                output, "cached_tokens_details", i\
            ),
    }' "$file"
fi
