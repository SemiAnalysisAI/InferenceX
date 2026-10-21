#!/usr/bin/env bash

# Usage: forward_signals COMMAND [ARG...]; runs COMMAND, passing INT, TERM and HUP on to it,
# and returns its exit status. A runner that cancels a step signals only the step's shell.
forward_signals() {
    local child status
    # Without job control, some bash versions give background jobs /dev/null as stdin.
    "$@" <&0 &
    child=$!
    # errexit applies inside traps, and COMMAND may already have exited.
    trap 'kill -s INT "$child" 2>/dev/null || :' INT
    trap 'kill -s TERM "$child" 2>/dev/null || :' TERM
    trap 'kill -s HUP "$child" 2>/dev/null || :' HUP
    # A trapped signal ends wait early; wait again until COMMAND has exited.
    while :; do
        wait "$child" && status=0 || status=$?
        kill -0 "$child" 2>/dev/null || break
    done
    trap - INT TERM HUP
    return "$status"
}
