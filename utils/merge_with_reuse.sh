#!/usr/bin/env bash
# Merge a PR while reusing its already-completed full sweep on push to main.
#
# Steps performed for the given PR:
#   1. Post `/reuse-sweep-run` so the merge-to-main run authorizes reuse.
#   2. Merge origin/main into the PR branch.  Any `perf-changelog.yaml`
#      conflict is auto-resolved by accepting main's entries and re-appending
#      the PR's entry at the bottom with `XXX` -> the canonical PR URL.
#   3. Canonicalize appended links and push a fresh synchronization commit.
#      The PR run observes the reuse authorization and skips sweep setup and
#      benchmark jobs.
#   4. Wait for the PR checks, then squash-merge the PR to main (--admin).
#
# Usage: utils/merge_with_reuse.sh <pr-number>
# Env:   REPO (default SemiAnalysisAI/InferenceX)
#        CHECK_TIMEOUT_SECONDS (default 900)

set -euo pipefail

REPO="${REPO:-SemiAnalysisAI/InferenceX}"
CHANGELOG="perf-changelog.yaml"
CHECK_TIMEOUT_SECONDS="${CHECK_TIMEOUT_SECONDS:-900}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -ne 1 ] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <pr-number>" >&2
    exit 2
fi
PR="$1"

log() { printf '\033[1;36m→\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

[ -z "$(git status --porcelain)" ] || die "Working tree is not clean"

wait_for_check() {
    local sha="$1"
    local check_name="$2"
    local deadline=$((SECONDS + CHECK_TIMEOUT_SECONDS))

    log "Waiting for ${check_name} on ${sha:0:8}"
    while ((SECONDS < deadline)); do
        local checks check status conclusion details
        checks="$(gh api "repos/${REPO}/commits/${sha}/check-runs?per_page=100")"
        check="$(jq -c --arg name "$check_name" '
            [.check_runs[] | select(.name == $name)]
            | sort_by(.started_at)
            | last // {}
        ' <<<"$checks")"
        status="$(jq -r '.status // ""' <<<"$check")"
        conclusion="$(jq -r '.conclusion // ""' <<<"$check")"
        details="$(jq -r '.details_url // ""' <<<"$check")"

        if [ "$status" = "completed" ]; then
            if [ "$conclusion" = "success" ]; then
                ok "${check_name} passed${details:+ - ${details}}"
                return 0
            fi
            die "${check_name} concluded ${conclusion:-unknown}${details:+ - ${details}}"
        fi

        sleep 5
    done

    die "Timed out after ${CHECK_TIMEOUT_SECONDS}s waiting for ${check_name} on ${sha}"
}

ORIGINAL_BRANCH="$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)"
LOCAL_BRANCH=""
cleanup() {
    git checkout --quiet "$ORIGINAL_BRANCH" 2>/dev/null || true
    if [ -n "$LOCAL_BRANCH" ]; then
        git branch -D "$LOCAL_BRANCH" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# --- preflight ---------------------------------------------------------------
PR_INFO="$(
    gh pr view "$PR" --repo "$REPO" \
        --json headRefName,isCrossRepository,state,labels
)"
PR_STATE="$(jq -r '.state' <<<"$PR_INFO")"
[ "$PR_STATE" = "OPEN" ] || die "PR #${PR} is ${PR_STATE}, expected OPEN"
[ "$(jq -r '.isCrossRepository' <<<"$PR_INFO")" = "false" ] \
    || die "PR #${PR} is from a fork; the merge helper cannot update its branch"

HEAD_BRANCH="$(jq -r '.headRefName' <<<"$PR_INFO")"
SWEEP_LABELS="$(jq -c '
    [
      .labels[].name |
      select(
        . == "sweep-enabled" or
        . == "full-sweep-enabled" or
        . == "non-canary-full-sweep-enabled" or
        . == "full-sweep-fail-fast" or
        . == "full-sweep-fail-fast-no-canary"
      )
    ]
' <<<"$PR_INFO")"
SWEEP_LABEL_COUNT="$(jq 'length' <<<"$SWEEP_LABELS")"
[ "$SWEEP_LABEL_COUNT" -eq 1 ] \
    || die "PR #${PR} must have exactly one sweep label"
SELECTED_SWEEP_LABEL="$(jq -r '.[0]' <<<"$SWEEP_LABELS")"
case "$SELECTED_SWEEP_LABEL" in
    full-sweep-enabled|non-canary-full-sweep-enabled|full-sweep-fail-fast|full-sweep-fail-fast-no-canary)
        ;;
    *)
        die "PR #${PR} must use a full-sweep label for artifact reuse"
        ;;
esac
[ "$(jq '[.labels[].name | select(. == "evals-only")] | length' <<<"$PR_INFO")" -eq 0 ] \
    || die "PR #${PR} uses evals-only, which is not eligible for artifact reuse"

# Fail early unless a successful run with reusable artifacts exists on a
# current PR commit. This excludes reuse-gate-only success runs.
PR_SHAS="$(gh api "repos/${REPO}/pulls/${PR}/commits" --paginate --jq '.[].sha')"
ELIGIBLE_RUN=""
while IFS=$'\t' read -r run_id run_sha; do
    grep -qxF "$run_sha" <<<"$PR_SHAS" || continue
    artifact_names="$(
        gh api "repos/${REPO}/actions/runs/${run_id}/artifacts?per_page=100" \
            --paginate \
            --jq '.artifacts[] | select(.expired == false) | .name'
    )"
    if grep -Eq '^(results_bmk|eval_results_all|bmk_agentic_)' \
        <<<"$artifact_names"; then
        ELIGIBLE_RUN="$run_id"
        break
    fi
done < <(
    gh api \
        "repos/${REPO}/actions/workflows/run-sweep.yml/runs?event=pull_request&branch=${HEAD_BRANCH}&status=completed&per_page=100" \
        --paginate \
        --jq '.workflow_runs[] | select(.conclusion == "success") | [.id, .head_sha] | @tsv'
)
if [ -z "$ELIGIBLE_RUN" ]; then
    die "PR #${PR} has no successful reusable run-sweep.yml run on a current commit"
fi

# --- step 1: comment ---------------------------------------------------------
log "Posting /reuse-sweep-run on PR #${PR}"
gh pr comment "$PR" --repo "$REPO" --body "/reuse-sweep-run" >/dev/null
ok "Comment posted"

# --- step 2: merge main into PR branch --------------------------------------
LOCAL_BRANCH="pr-${PR}-reuse-$$"
log "Fetching PR branch ${HEAD_BRANCH}"
git fetch origin "pull/${PR}/head:${LOCAL_BRANCH}" --quiet
git checkout --quiet "$LOCAL_BRANCH"
git fetch origin main --quiet

PRE_MERGE="$(git rev-parse HEAD)"
log "Merging origin/main"
set +e
git merge origin/main --no-ff --no-edit
merge_status=$?
set -e

if [ "$merge_status" -ne 0 ]; then
    unresolved="$(git diff --name-only --diff-filter=U)"
    if [ "$unresolved" != "$CHANGELOG" ]; then
        git merge --abort
        die "Unexpected conflict(s) in: ${unresolved} — only ${CHANGELOG} is auto-resolved"
    fi
    log "Resolving ${CHANGELOG} conflict"
    if ! python3 "$SCRIPT_DIR/prepare_perf_changelog_merge.py" \
        resolve-conflict \
        --changelog-file "$CHANGELOG" \
        --pr-number "$PR" \
        --repo "$REPO"; then
        git merge --abort
        die "Could not safely resolve ${CHANGELOG}"
    fi
    git add "$CHANGELOG"
    git commit --no-edit
fi

HEAD_AFTER_MERGE="$(git rev-parse HEAD)"
python3 "$SCRIPT_DIR/prepare_perf_changelog_merge.py" \
    canonicalize \
    --changelog-file "$CHANGELOG" \
    --base-ref origin/main \
    --pr-number "$PR" \
    --repo "$REPO"

if ! git diff --quiet -- "$CHANGELOG"; then
    git add "$CHANGELOG"
    if [ "$HEAD_AFTER_MERGE" != "$PRE_MERGE" ]; then
        git commit --amend --no-edit
    else
        git commit -m "fix: canonicalize PR #${PR} changelog link [skip-sweep]"
    fi
fi

# Always create a synchronize event when the branch was already prepared.
# This guarantees the reuse gate sees the authorization on the current SHA.
if [ "$PRE_MERGE" = "$(git rev-parse HEAD)" ]; then
    git commit --allow-empty \
        -m "chore: refresh PR #${PR} for sweep reuse [skip-sweep]"
fi

# --- step 3: push prepared commit --------------------------------------------
POST_MERGE="$(git rev-parse HEAD)"
log "Pushing prepared commit ${POST_MERGE:0:8}"
git push origin "${LOCAL_BRANCH}:${HEAD_BRANCH}"
ok "Push complete; reuse authorization will be evaluated on the new head"

# --- step 4: squash-merge to main -------------------------------------------
CURRENT_HEAD="$(gh pr view "$PR" --repo "$REPO" --json headRefOid --jq '.headRefOid')"
[ "$CURRENT_HEAD" = "$POST_MERGE" ] \
    || die "PR head changed to ${CURRENT_HEAD:0:8}; expected ${POST_MERGE:0:8}"

# `gh pr checks --watch` fails if GitHub has not registered any checks yet.
wait_for_check "$POST_MERGE" "check-changelog"

log "Waiting for all PR checks"
gh pr checks "$PR" --repo "$REPO" --watch --fail-fast
ok "All PR checks passed"

CURRENT_HEAD="$(gh pr view "$PR" --repo "$REPO" --json headRefOid --jq '.headRefOid')"
[ "$CURRENT_HEAD" = "$POST_MERGE" ] \
    || die "PR head changed to ${CURRENT_HEAD:0:8}; expected ${POST_MERGE:0:8}"

log "Squash-merging PR #${PR} into main"
gh pr merge "$PR" --repo "$REPO" --squash --admin >/dev/null

MERGE_SHA="$(gh pr view "$PR" --repo "$REPO" --json mergeCommit --jq '.mergeCommit.oid')"
ok "PR #${PR} merged as ${MERGE_SHA:0:8} — the push-to-main run will reuse the prior successful sweep."
