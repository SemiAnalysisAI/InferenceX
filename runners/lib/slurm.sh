#!/usr/bin/env bash
# Shared Slurm terminal-state checks for multi-node launchers.

require_slurm_job_succeeded() {
    local job_id="$1"
    local record=""
    local state=""
    local exit_code=""

    if command -v sacct >/dev/null 2>&1; then
        for _ in {1..12}; do
            record="$(
                sacct -n -P -j "$job_id" --format=JobIDRaw,State,ExitCode \
                    2>/dev/null \
                    | awk -F '|' -v id="$job_id" '$1 == id { print $2 "|" $3; exit }'
            )"
            [[ -n "$record" ]] && break
            sleep 5
        done
    fi

    if [[ -n "$record" ]]; then
        state="${record%%|*}"
        exit_code="${record#*|}"
    elif command -v scontrol >/dev/null 2>&1; then
        record="$(scontrol show job "$job_id" --oneliner 2>/dev/null || true)"
        state="$(sed -n 's/.* JobState=\([^ ]*\).*/\1/p' <<< "$record")"
        exit_code="$(sed -n 's/.* ExitCode=\([^ ]*\).*/\1/p' <<< "$record")"
    fi

    state="${state%%+}"
    if [[ "$state" != "COMPLETED" || "$exit_code" != "0:0" ]]; then
        echo "Error: Slurm job $job_id did not succeed (state=${state:-unknown}, exit=${exit_code:-unknown})" >&2
        command -v sacct >/dev/null 2>&1 \
            && sacct -P -j "$job_id" --format=JobID,State,ExitCode,Elapsed >&2 \
            || true
        return 1
    fi

    echo "Slurm job $job_id completed successfully (exit 0:0)"
}
