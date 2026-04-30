#!/usr/bin/env bash

# B300 uses the same low-concurrency TRTLLM bring-up recipe as B200. The B300
# runner may rewrite MODEL to the pre-staged /data/models/dsv4-pro path before
# this script is invoked.

bash "$(dirname "$0")/dsv4_fp4_b200_trt.sh"
