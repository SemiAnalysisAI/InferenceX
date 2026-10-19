#!/usr/bin/env bash
set -eo pipefail
pip install torchao
bash "$(dirname "${BASH_SOURCE[0]}")/musespark-dflash-draft-swa.sh"
