#!/usr/bin/env bash
# Shared tail-call used by the small model recipe wrappers in this directory.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_disaggregated.sh"
