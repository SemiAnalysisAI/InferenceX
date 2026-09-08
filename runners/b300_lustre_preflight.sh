#!/usr/bin/bash

# Run from node-local /tmp, before opening anything on /data. This function is
# sent in the srun command so an unhealthy client need not read a shared script.
check_b300_lustre() {
    local imports states
    local deadline=$((SECONDS + ${LUSTRE_PREFLIGHT_TIMEOUT:-30}))

    while :; do
        if ! imports=$(timeout --kill-after=2 10 lctl get_param \
            'osc.*.import' 'mdc.*.import' 2>&1); then
            echo "Error: cannot inspect Lustre clients on $(hostname): $imports" >&2
            return 1
        fi
        states=$(awk '/^[[:space:]]*state:/ { print $2 }' <<< "$imports")
        if [[ -n "$states" ]] && ! grep -qv '^FULL$' <<< "$states"; then
            return 0
        fi
        if (( SECONDS >= deadline )); then
            echo "Error: Lustre clients are not ready on $(hostname); refusing shared-filesystem I/O." >&2
            awk '/^[[:space:]]*(name|state):/' <<< "$imports" >&2
            echo "A cluster administrator must restore Lustre connectivity before retrying." >&2
            return 1
        fi
        echo "Waiting for Lustre clients on $(hostname) to reconnect..." >&2
        sleep 5
    done
}
