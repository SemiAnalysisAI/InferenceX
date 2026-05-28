#!/usr/bin/env bash
# Merge a PR while reusing its already-completed full sweep on push to main.
#
# Steps performed for the given PR:
#   1. Post `/reuse-sweep-run` so the merge-to-main run authorizes reuse.
#   2. Merge origin/main into the PR branch.  Any `perf-changelog.yaml`
#      conflict is auto-resolved by accepting main's entries and re-appending
#      the PR's entry at the bottom with `XXX` -> the PR number.
#   3. Push the merge commit and cancel the sweep it triggers (the prior
#      successful sweep is what the merge run will reuse).
#   4. Squash-merge the PR to main (--admin).
#
# Usage: utils/merge_with_reuse.sh <pr-number>
# Env:   REPO (default SemiAnalysisAI/InferenceX)

set -euo pipefail

REPO="${REPO:-SemiAnalysisAI/InferenceX}"
CHANGELOG="perf-changelog.yaml"

if [ $# -ne 1 ] || ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <pr-number>" >&2
    exit 2
fi
PR="$1"

log() { printf '\033[1;36m→\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

ORIGINAL_BRANCH="$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)"
cleanup() { git checkout --quiet "$ORIGINAL_BRANCH" 2>/dev/null || true; }
trap cleanup EXIT

# --- preflight ---------------------------------------------------------------
PR_INFO="$(gh pr view "$PR" --repo "$REPO" --json headRefName,state,labels)"
PR_STATE="$(jq -r '.state' <<<"$PR_INFO")"
[ "$PR_STATE" = "OPEN" ] || die "PR #${PR} is ${PR_STATE}, expected OPEN"

HEAD_BRANCH="$(jq -r '.headRefName' <<<"$PR_INFO")"
HAS_FULL_SWEEP="$(jq -r '
    [.labels[].name] as $names
    | if (($names | index("full-sweep-enabled")) != null)
         or (($names | index("non-canary-full-sweep-enabled")) != null)
      then "1" else "" end
' <<<"$PR_INFO")"
[ -n "$HAS_FULL_SWEEP" ] || die "PR #${PR} is missing 'full-sweep-enabled' or 'non-canary-full-sweep-enabled' label"

# Warn early if no successful run exists on any current PR commit.
PR_SHAS="$(gh api "repos/${REPO}/pulls/${PR}/commits" --paginate --jq '.[].sha')"
SUCCESS_RUNS="$(gh api "repos/${REPO}/actions/workflows/run-sweep.yml/runs?event=pull_request&branch=${HEAD_BRANCH}&status=completed&per_page=100" \
    --jq '.workflow_runs[] | select(.conclusion=="success") | .head_sha' || true)"
if ! grep -qFxf <(echo "$SUCCESS_RUNS") <(echo "$PR_SHAS"); then
    die "PR #${PR} has no successful run-sweep.yml run on any of its current commits"
fi

# --- step 1: comment ---------------------------------------------------------
log "Posting /reuse-sweep-run on PR #${PR}"
gh pr comment "$PR" --repo "$REPO" --body "/reuse-sweep-run" >/dev/null
ok "Comment posted"

# --- step 2: merge main into PR branch --------------------------------------
LOCAL_BRANCH="pr-${PR}-reuse"
log "Fetching PR branch ${HEAD_BRANCH}"
git fetch origin "pull/${PR}/head:${LOCAL_BRANCH}" --force --quiet
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
    python3 - "$CHANGELOG" "$PR" "$REPO" <<'PY'
import re
import subprocess
import sys

import yaml

path, pr, repo = sys.argv[1], sys.argv[2], sys.argv[3]
pr_link_full = f"https://github.com/{repo}/pull/{pr}"


def read_stage(stage: int) -> str:
    return subprocess.check_output(["git", "show", f":{stage}:{path}"]).decode()


def split_entries(text: str) -> tuple[str, list[str]]:
    """Split a perf-changelog.yaml into (preamble, [entry_text, ...])."""
    parts = re.split(r"\n(?=- config-keys:)", text)
    if not parts:
        return "", []
    if parts[0].startswith("- config-keys:"):
        return "", [p.rstrip("\n") for p in parts]
    return parts[0], [p.rstrip("\n") for p in parts[1:]]


def entry_signature(entry: dict) -> tuple:
    """Identity used to detect duplicates across sides.

    Same config-keys + same description = same logical entry, even if pr-link
    differs (which is exactly the case when a placeholder XXX entry on the PR
    side collides with a real entry that landed on main from another PR)."""
    keys = tuple(sorted(entry.get("config-keys") or []))
    desc = tuple(entry.get("description") or [])
    return (keys, desc)


# Stage 2 = HEAD (PR before merge); Stage 3 = MERGE_HEAD (origin/main).
pr_text = read_stage(2)
main_text = read_stage(3)

pr_preamble, pr_blocks = split_entries(pr_text)
main_preamble, main_blocks = split_entries(main_text)

pr_data = yaml.safe_load(pr_text) or []
main_data = yaml.safe_load(main_text) or []

if len(pr_data) != len(pr_blocks) or len(main_data) != len(main_blocks):
    sys.exit(
        f"Entry/text-block count mismatch — file uses an unsupported shape. "
        f"pr_data={len(pr_data)} pr_blocks={len(pr_blocks)} "
        f"main_data={len(main_data)} main_blocks={len(main_blocks)}"
    )

main_sigs = {entry_signature(e) for e in main_data}

contribs: list[str] = []
for entry, block in zip(pr_data, pr_blocks):
    link = str(entry.get("pr-link") or "")
    if "XXX" not in link and not link.endswith(f"/pull/{pr}"):
        continue
    if entry_signature(entry) in main_sigs:
        continue  # Same logical entry already on main (e.g. XXX placeholder vs real pr-link).
    # Force the pr-link line to the canonical full URL, regardless of whether
    # the PR's entry used bare `XXX` or `/pull/XXX`.
    new_block, n = re.subn(
        r"^(\s*pr-link:).*$",
        lambda m: f"{m.group(1)} {pr_link_full}",
        block,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        sys.exit(f"Could not locate pr-link line in entry: {entry}")
    contribs.append(new_block)

if not contribs:
    sys.exit(
        f"No PR contributions found in {path} "
        f"(expected entry tagged with XXX or /pull/{pr} with new content)"
    )

sections: list[str] = []
if main_preamble.strip():
    sections.append(main_preamble.rstrip("\n"))
sections.extend(main_blocks)
sections.extend(contribs)

result = "\n\n".join(s for s in sections if s) + "\n"
open(path, "w").write(result)
PY
    python3 -c "
import yaml
entries = yaml.safe_load(open('$CHANGELOG'))
last = entries[-1]
assert last['pr-link'].endswith('/$PR'), f'last entry not for PR #$PR: {last}'
print(f'  Last entry: {last[\"config-keys\"]} -> #$PR')
"
    [ -z "$(tail -c 1 "$CHANGELOG")" ] || die "${CHANGELOG} missing trailing newline"
    git add "$CHANGELOG"
    git commit --no-edit -m "Merge branch 'main' into ${HEAD_BRANCH}"
fi

POST_MERGE="$(git rev-parse HEAD)"

# --- step 3: push and cancel triggered sweep --------------------------------
if [ "$PRE_MERGE" = "$POST_MERGE" ]; then
    log "PR already up to date with main; skipping push + cancel"
else
    log "Pushing merge commit ${POST_MERGE:0:8}"
    git push origin "${LOCAL_BRANCH}:${HEAD_BRANCH}"

    log "Waiting for triggered sweep run to register"
    RUN_ID=""
    for _ in 1 2 3 4 5; do
        sleep 5
        RUN_ID="$(gh run list --repo "$REPO" --branch "$HEAD_BRANCH" \
            --workflow run-sweep.yml --limit 5 \
            --json databaseId,headSha,status \
            --jq ".[] | select(.headSha==\"${POST_MERGE}\" and (.status==\"queued\" or .status==\"in_progress\")) | .databaseId" | head -1)"
        [ -n "$RUN_ID" ] && break
    done
    if [ -n "$RUN_ID" ]; then
        log "Cancelling sweep run ${RUN_ID}"
        gh run cancel "$RUN_ID" --repo "$REPO" >/dev/null
        ok "Sweep cancelled"
    else
        echo "  No queued/in-progress sweep found after 25s — proceeding."
    fi
fi

# --- step 4: squash-merge to main -------------------------------------------
log "Squash-merging PR #${PR} into main"
gh pr merge "$PR" --repo "$REPO" --squash --admin >/dev/null

MERGE_SHA="$(gh pr view "$PR" --repo "$REPO" --json mergeCommit --jq '.mergeCommit.oid')"
ok "PR #${PR} merged as ${MERGE_SHA:0:8} — the push-to-main run will reuse the prior successful sweep."
