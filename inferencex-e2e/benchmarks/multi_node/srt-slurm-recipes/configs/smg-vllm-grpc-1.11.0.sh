#!/usr/bin/env bash
# SMG 1.11.0 router plus the vLLM gRPC servicer, for vLLM workers behind SMG over gRPC.
# The recipe setup script runs in every container (router and engines).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$here/smg-1.11.0.sh"
bash "$here/smg-grpc-servicer-0.13.0.sh"
