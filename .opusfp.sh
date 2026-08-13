#!/bin/bash
D=/usr/local/lib/python3.12/dist-packages/aiter/ops/opus
find "$D" -name '*.py' | sort | xargs md5sum 2>/dev/null | sed "s#$D/##"
echo "--- fused_moe.py ---"
md5sum /usr/local/lib/python3.12/dist-packages/aiter/fused_moe.py
