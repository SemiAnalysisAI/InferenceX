#!/bin/bash

set -e

if [ $# -lt 3 ]; then
  echo "Usage: $0 <START_INDEX> <END_INDEX> <BASE_DIR> [SESSION_NAME]"
  echo "Example: $0 0 9 /path/to/base-dir my-session"
  exit 1
fi

START="$1"
END="$2"
BASE_DIR="$3"
SESSION="${4:-github-actions}"

# Kill existing session if it exists
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session with the first runner
PADDED_START=$(printf "%02d" "$START")
tmux new-session -d -s "$SESSION" -n "runners"
tmux send-keys -t "$SESSION" "cd ${BASE_DIR}/gharunner${PADDED_START}/actions-runner && ./run.sh" Enter

# Create additional panes for the rest
for i in $(seq $((START + 1)) "$END"); do
  PADDED=$(printf "%02d" "$i")
  tmux split-window -t "$SESSION"
  tmux send-keys -t "$SESSION" "cd ${BASE_DIR}/gharunner${PADDED}/actions-runner && ./run.sh" Enter
  tmux select-layout -t "$SESSION" tiled
done
