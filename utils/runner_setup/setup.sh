#!/bin/bash

set -e

if [ $# -lt 7 ]; then
  echo "Usage: $0 <TOKEN> <RUNNER_URL> <START_INDEX> <END_INDEX> <BASE_DIR> <BASE_RUNNER_NAME> <ADDITIONAL_RUNNER_TAGS> [REPO_URL]"
  echo "Example: $0 AOPHAHI... https://github.com/actions/runner/releases/download/v2.331.0/actions-runner-linux-x64-2.331.0.tar.gz 0 9 . my-runner slurm https://github.com/MyOrg/MyRepo"
  exit 1
fi

TOKEN="$1"
RUNNER_URL="$2"
START="$3"
END="$4"
BASE_DIR="$5"
BASE_RUNNER_NAME="$6"
ADDITIONAL_RUNNER_TAGS="$7"
REPO_URL="${8:-https://github.com/SemiAnalysisAI/InferenceX}"
RUNNER_TAR=$(basename "$RUNNER_URL")

# Download the tarball once
if [ ! -f "${BASE_DIR}/${RUNNER_TAR}" ]; then
  echo "Downloading runner tarball..."
  curl -o "${BASE_DIR}/${RUNNER_TAR}" -L "$RUNNER_URL"
fi

for i in $(seq "$START" "$END"); do
  (
    PADDED=$(printf "%02d" "$i")
    RUNNER_NAME="${BASE_RUNNER_NAME}_${PADDED}"
    DIR="${BASE_DIR}/gharunner${PADDED}"

    echo "[${RUNNER_NAME}] Setting up..."

    mkdir -p "${DIR}/actions-runner"
    tar xzf "${BASE_DIR}/${RUNNER_TAR}" -C "${DIR}/actions-runner"

    cd "${DIR}/actions-runner"
    ./config.sh --unattended \
      --url "$REPO_URL" \
      --token "$TOKEN" \
      --name "$RUNNER_NAME" \
      --labels "${ADDITIONAL_RUNNER_TAGS},${RUNNER_NAME}" \
      --replace

    echo "[${RUNNER_NAME}] Done."
  ) &
done

wait
echo "All runners configured!"
