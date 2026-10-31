#!/usr/bin/env bash
# Dispatch one serving YAML on vngcloud/InferenceX:
#   inferencex-e2e/utils/serving/dispatch.sh inferencex-e2e/configs/serving/<name>.yaml [--ref <branch>]
# Validates locally with the code CI runs, starts gn-benchmark.yml with the YAML as input,
# and prints the run URL.
set -eo pipefail

E2E="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GH_REPO="vngcloud/InferenceX"

usage() {
    echo "usage: utils/serving/dispatch.sh <serving.yaml> [--ref <branch>]" >&2
    exit 2
}
[[ $# -ge 1 ]] || usage
yaml_file="$(realpath "$1")"
shift
# The workflow input is the file content, not HEAD: warn when the run would differ from git.
if [ -n "$(git -C "$E2E" status --porcelain -- "$yaml_file")" ]; then
    echo "WARNING: $yaml_file differs from HEAD; the run dispatches the working-tree content." >&2
fi
ref=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref) [[ $# -ge 2 && -n "$2" ]] || usage; ref="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

title=$(cd "$E2E" && PYTHONPATH="$E2E" uv run -q --no-project --python 3.12 --with pydantic --with pyyaml \
    python -P -m infx.serving.to_matrix --check "$yaml_file") || exit 2

secret=$(cd "$E2E" && uv run -q --no-project --python 3.12 --with pyyaml python -P -c \
    'import sys, yaml; print((yaml.safe_load(open(sys.argv[1])).get("endpoint") or {}).get("api-key-secret") or "")' \
    "$yaml_file")
if [[ -n "$secret" ]]; then
    if names=$(gh secret list --repo "$GH_REPO" --json name --jq '.[].name' 2>/dev/null); then
        if ! grep -qx "$secret" <<<"$names"; then
            echo "ERROR endpoint.api-key-secret: $secret is not a secret of $GH_REPO; add it: gh secret set $secret --repo $GH_REPO" >&2
            exit 2
        fi
    else
        echo "WARNING cannot list secrets of $GH_REPO (needs admin); assuming $secret exists" >&2
    fi
fi

if [[ -z "$ref" ]]; then
    ref="$(git -C "$E2E" rev-parse --abbrev-ref HEAD)"
    remote_sha="$(git -C "$E2E" ls-remote origin "refs/heads/$ref" | cut -f1)"
    if [[ -z "$remote_sha" ]]; then
        echo "ERROR branch $ref is not on origin; push it first (CI runs the pushed code)" >&2
        exit 2
    fi
    if [[ "$remote_sha" != "$(git -C "$E2E" rev-parse HEAD)" ]]; then
        echo "ERROR local HEAD differs from origin/$ref; push or pull first (CI runs the pushed code)" >&2
        exit 2
    fi
fi

started=$(( $(date +%s) - 60 ))  # 60 s early: tolerate a local clock ahead of GitHub's
gh workflow run gn-benchmark.yml --repo "$GH_REPO" --ref "$ref" -f serving="$(cat "$yaml_file")" -f test-name="$title"
for _ in $(seq 1 30); do
    sleep 4
    run_id=$(gh run list --repo "$GH_REPO" -w gn-benchmark.yml -L 20 --json databaseId,displayTitle,createdAt \
        --jq "[.[] | select(.displayTitle == \"$title\" and (.createdAt | fromdateiso8601) >= $started)][0].databaseId // empty")
    if [[ -n "$run_id" ]]; then
        echo "https://github.com/$GH_REPO/actions/runs/$run_id"
        exit 0
    fi
done
echo "ERROR dispatched, but the run did not show up within 2 minutes; do not redispatch, check: gh run list --repo $GH_REPO -w gn-benchmark.yml -L 5" >&2
exit 1
