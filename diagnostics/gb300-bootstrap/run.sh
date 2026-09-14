#!/usr/bin/env bash
set -euo pipefail
SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EVIDENCE_DIR=${1:?evidence output required}
mkdir -p "$EVIDENCE_DIR"
EVIDENCE_DIR=$(cd -- "$EVIDENCE_DIR" && pwd)
(cd "$SOURCE_DIR" && sha256sum -c SHA256SUMS) > "$EVIDENCE_DIR/source-checks.log"
cp "$SOURCE_DIR/source-manifest.json" "$SOURCE_DIR/SHA256SUMS" "$EVIDENCE_DIR/"
TASK_DIR=$(mktemp -d "${RUNNER_TEMP:?}/gb300-bootstrap.XXXXXX")
trap 'rm -rf -- "$TASK_DIR"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
mkdir -p "$TASK_DIR/work/configs" "$TASK_DIR/trace-bin"
cp "$SOURCE_DIR/srtslurm.yaml.example" "$TASK_DIR/work/srtslurm.yaml"
export BOOTSTRAP_CURL BOOTSTRAP_WGET
BOOTSTRAP_CURL=$(command -v curl)
BOOTSTRAP_WGET=$(command -v wget)
cat > "$TASK_DIR/trace-bin/curl" <<'CURL'
#!/usr/bin/env bash
exec "$BOOTSTRAP_CURL" --write-out '%{stderr}\nbootstrap_http_status=%{http_code} bytes=%{size_download} duration_s=%{time_total}\n' "$@"
CURL
cat > "$TASK_DIR/trace-bin/wget" <<'WGET'
#!/usr/bin/env bash
exec "$BOOTSTRAP_WGET" "$@" --server-response --no-quiet
WGET
chmod +x "$TASK_DIR/trace-bin/curl" "$TASK_DIR/trace-bin/wget"
{
  date -u +%FT%TZ
  uname -m
  id
  uv --version
  file "$(command -v uv)"
  sha256sum "$(command -v uv)"
  printf 'run=%s attempt=%s runner=%s task_dir=%s\n' "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT" "$RUNNER_NAME" "$TASK_DIR"
} > "$EVIDENCE_DIR/environment.txt"
export PATH="$TASK_DIR/trace-bin:$PATH"
check_binaries() {
  local binary
  for binary in configs/nats-server configs/etcd configs/etcdctl bin/uv; do
    test -x "$binary" && file "$binary" | grep -q aarch64 || return 1
  done
}
for arm in baseline candidate; do
  cp "$SOURCE_DIR/$arm.Makefile" "$TASK_DIR/work/Makefile"
  start=$(date -u +%FT%TZ)
  set +e
  (cd "$TASK_DIR/work" && timeout --signal=TERM --kill-after=5s 160s make setup ARCH=aarch64 SHELL='bash -x') > "$EVIDENCE_DIR/$arm.log" 2>&1
  rc=$?
  (cd "$TASK_DIR/work" && check_binaries)
  binary_rc=$?
  set -e
  printf 'started=%s\ncompleted=%s\nmake_exit=%s\nbinary_check_exit=%s\n' "$start" "$(date -u +%FT%TZ)" "$rc" "$binary_rc" > "$EVIDENCE_DIR/$arm-status.txt"
  (cd "$TASK_DIR/work"; for binary in configs/nats-server configs/etcd configs/etcdctl bin/uv; do
    if test -f "$binary"; then file "$binary"; sha256sum "$binary"; fi
  done) > "$EVIDENCE_DIR/$arm-binaries.txt"
  if [[ "$arm" == candidate ]]; then
    [[ "$rc" == 0 && "$binary_rc" == 0 ]] || exit 1
  fi
 done
