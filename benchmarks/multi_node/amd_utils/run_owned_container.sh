#!/usr/bin/env bash
# Keep a signal-responsive task shell while the container CLI is attached.
# A foreground CLI can defer Bash traps indefinitely without a container init.
set -eo pipefail
runtime_spec=${1:?container runtime}
container=${2:?owned main container}
router=${3-}
shift 3
[[ "$container" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$ ]]
[[ -z "$router" || "$router" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$ ]]
read -r -a runtime <<< "$runtime_spec"

finish() {
    local result=$?
    trap - EXIT
    trap '' INT TERM HUP
    printf '[container-lifecycle] %s cleanup start name=%s rc=%s\n' "$(date -u +%FT%TZ)" "$container" "$result"
    # These are cleanup bounds, not new serving or benchmark deadlines. The
    # normal launcher still verifies physical GPU release before reuse.
    timeout --kill-after=5s 15s "${runtime[@]}" stop --time 10 "$container" || true
    if [[ -n "$router" ]]; then
        timeout --kill-after=5s 10s "${runtime[@]}" rm -f "$router" || true
    fi
    printf '[container-lifecycle] %s cleanup end name=%s rc=%s\n' "$(date -u +%FT%TZ)" "$container" "$result"
    exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

"${runtime[@]}" run "$@" &
wait "$!"
