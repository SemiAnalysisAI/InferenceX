#!/usr/bin/env bash
# Stage the MiniMax-M3 EAGLE3 GQA draft and the tokenizer dependencies the ATOM
# image lacks, in a side directory that the server's interpreter imports.
set -euo pipefail
hf download Inferact/MiniMax-M3-EAGLE3-GQA
deps=/tmp/inferencex-atom-runtime-deps
/opt/venv/bin/python -m pip install --quiet --target "$deps" --no-deps sentencepiece tiktoken
site=$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$deps" > "$site/inferencex-atom-runtime-deps.pth"
