#!/usr/bin/env bash
set -eo pipefail
dir=$(dirname "${BASH_SOURCE[0]}")
bash "$dir/musespark-gemm-ag.sh"
bash "$dir/musespark-fp8-latent.sh"
