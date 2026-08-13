#!/bin/bash
# Fingerprint the whole aiter python tree so we can size the nightly-vs-baked gap.
cd /usr/local/lib/python3.12/dist-packages/aiter || exit 1
find . -name '*.py' -not -path './jit/*' | sort | xargs md5sum 2>/dev/null
