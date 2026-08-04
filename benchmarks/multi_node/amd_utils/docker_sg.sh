#!/usr/bin/env bash
# Run `docker` under the 'docker' group.
#
# Why: Slurm launches job steps without the user's 'docker' supplementary group
# in the active credential set. The user IS a docker-group member (getent group
# docker lists them) but the group is missing from `id -G` inside the step, so
# the group-owned socket (/var/run/docker.sock, 0660 root:docker) is unreachable
# via a plain `docker` call. `sg docker -c` re-activates the group for this one
# command — no sudo, no password, no persistent host change.
#
# Used by job.slurm's DOCKER_CMD detection as the fallback when plain `docker`
# fails but `sg docker -c 'docker ps'` succeeds.
#
# argv is passed across the `sg` shell hop by NUL-delimited base64 (NOT string
# re-quoting): naive `printf %q` mangles the big multiline `docker run ... bash
# -lc '<script>'` argument (trailing newline became a literal 'n', spawning a
# stray `$n`). base64 round-trips arbitrary bytes (newlines, quotes) exactly,
# and only the base64 blob (safe chars) is interpolated into the sg command.
b64=$(printf '%s\0' "$@" | base64 | tr -d '\n')
exec sg docker -c "bash -c 'mapfile -d \"\" -t __A < <(printf %s \"$b64\" | base64 -d); exec docker \"\${__A[@]}\"'"
