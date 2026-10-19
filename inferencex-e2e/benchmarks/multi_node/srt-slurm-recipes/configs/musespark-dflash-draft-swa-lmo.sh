#!/usr/bin/env bash
set -eo pipefail
dir=$(dirname "${BASH_SOURCE[0]}")
bash "$dir/musespark-dflash-draft-swa.sh"
bash "$dir/musespark-language-model-only.sh"
