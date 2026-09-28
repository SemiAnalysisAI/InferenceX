#!/usr/bin/env bash
# Exact-source backports: reject drift before changing any installed source.
set -eo pipefail
source_dir=${1:?source directory}
patch_file=${2:?patch file}
before=${3:?original SHA256 manifest}
after=${4:?candidate SHA256 manifest}
test -f "$patch_file" && test -f "$before" && test -f "$after"
exec 9> "$source_dir/.inferencex-patch.lock"
flock 9
if (cd "$source_dir" && sha256sum --status -c "$after"); then
    printf 'AIPerf checked patch already applied: %s\n' "$patch_file"
    exit 0
fi
if ! (cd "$source_dir" && sha256sum --status -c "$before"); then
    echo 'ERROR: AIPerf source differs from the validated backport; refusing to patch' >&2
    exit 1
fi
patch --batch --fuzz=0 --forward --dry-run -p1 -d "$source_dir" < "$patch_file"
patch --batch --fuzz=0 --forward -p1 -d "$source_dir" < "$patch_file"
(cd "$source_dir" && sha256sum -c "$after")
