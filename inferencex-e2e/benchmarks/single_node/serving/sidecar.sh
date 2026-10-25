#!/usr/bin/env bash
# Router or storage sidecar of one serving-YAML point, started by job.sh as an --overlap step:
# expand the {{...}} port placeholders, export the env lines, exec the command.
#   sidecar.sh router   ROUTER_CMD / ROUTER_ENV / ROUTER_PROFILE (dynamo | generic)
#   sidecar.sh storage  STORAGE_CMD / STORAGE_ENV
set -eo pipefail
mode="${1:-}"

expand() {
    local text="$1" urls="" port
    text="${text//\{\{router_port\}\}/${PORT:-}}"
    text="${text//\{\{router_metrics_port\}\}/${ROUTER_METRICS_PORT:-}}"
    text="${text//\{\{engine_port\}\}/${ENGINE_PORT:-}}"
    text="${text//\{\{engine_url\}\}/http://127.0.0.1:${ENGINE_PORT:-}}"
    for port in ${ENGINE_PORTS:-}; do
        urls+="${urls:+ }http://127.0.0.1:$port"
    done
    text="${text//\{\{engine_urls\}\}/$urls}"
    [[ -n "${STORAGE_MASTER_PORT:-}" ]] && text="${text//\{\{storage_master_port\}\}/$STORAGE_MASTER_PORT}"
    [[ -n "${STORAGE_META_PORT:-}" ]] && text="${text//\{\{storage_meta_port\}\}/$STORAGE_META_PORT}"
    printf '%s' "$text"
}

case "$mode" in
    router) env_lines="${ROUTER_ENV:-}"; command="${ROUTER_CMD:?}" ;;
    storage) env_lines="${STORAGE_ENV:-}"; command="${STORAGE_CMD:?}" ;;
    *) echo "usage: sidecar.sh router|storage" >&2; exit 2 ;;
esac
dynamo=""
[[ "$mode" == router && "${ROUTER_PROFILE:-}" == dynamo ]] && dynamo=1

# Env values may hold tokens: export them without echoing the values.
while IFS= read -r assignment; do
    [[ -z "$assignment" ]] && continue
    assignment="$(expand "$assignment")"
    export "$assignment"
    echo "$mode env: ${assignment%%=*}" >&2
done <<< "$env_lines"

if [[ -n "$dynamo" ]]; then
    eval "set -- $(expand "$command")"
    set -- "$@" --http-host 0.0.0.0 --http-port "$PORT"
else
    eval "set -- $(expand "$command")"
fi
mkdir -p "$RESULT_DIR"
printf '%q ' "$@" > "$RESULT_DIR/${mode}_command.txt"
printf '\n' >> "$RESULT_DIR/${mode}_command.txt"
exec "$@"
