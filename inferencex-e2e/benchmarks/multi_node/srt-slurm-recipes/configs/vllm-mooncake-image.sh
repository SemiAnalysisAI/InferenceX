#!/usr/bin/env bash
# Mooncake ships in the image, with HBM segments (mount_external_segment). Check it instead of
# installing a wheel, for the vLLM workers (setup_script) and the mooncake-master service (preamble).
set -eo pipefail
command -v mooncake_master >/dev/null
python3 -c "from mooncake.store import MooncakeDistributedStore as S; assert hasattr(S, 'mount_external_segment')"
