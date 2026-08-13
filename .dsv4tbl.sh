#!/bin/bash
f=/usr/local/lib/python3.12/dist-packages/aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv
echo "lines=$(wc -l < "$f")  md5=$(md5sum "$f" | cut -d' ' -f1)"
echo "--- header ---"; head -1 "$f"
echo "--- rows naming x256x256 ---"; grep -c 'x256x256' "$f"
echo "--- sample rows ---"; grep 'x256x256' "$f" | head -3
