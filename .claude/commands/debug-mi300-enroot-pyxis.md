---
description: Debug enroot/pyxis container-start failures on the MI300X Vultr cluster (chi-mi300x-*) — userns sysctl drift survey via the slurm controller, approved fix, and sweep reruns
argument-hint: [failed-run-or-job-url ...]
---

Debug `pyxis: couldn't start container` failures on the MI300X runners
(`mi300x-amds_*`, nodes `chi-mi300x-*`). The canonical signature in the job log:

```
error: pyxis:     enroot-nsenter: failed to create user namespace: Permission denied
error: pyxis: couldn't start container
error: spank: required plugin spank_pyxis.so: task_init() failed with rc=-1
srun: error: chi-mi300x-0XX: task 0: Exited with exit code 1
```

Known root cause (July 2026): **provisioning drift** — nodes run Ubuntu 24.04,
and freshly (re)provisioned nodes carry the distro default
`kernel.apparmor_restrict_unprivileged_userns=1`, which blocks pyxis's
`enroot-nsenter` from creating user namespaces. Nodes with the flag at `0`
work; jobs pass or fail by node lottery. Plain `unshare -U` still works even
on broken nodes (Ubuntu ships an AppArmor profile for `unshare`), so don't let
that mislead you — test the actual enroot path or check the sysctl directly.

## Access

`ssh amd-vultr-mi300` lands as **root on the slurm controller**
(`slurm-sa-mi300-controller-01...`). Compute nodes do NOT accept direct root
SSH — reach them with `srun`:

```bash
ssh amd-vultr-mi300 'srun -w chi-mi300x-043 -N1 --immediate=30 bash -c "<cmd>"'
```

## Procedure

1. **Confirm the signature** from the failing GitHub job log(s) in `$ARGUMENTS`
   (or via `gh run view --log-failed`), and note the failing node names from
   the `srun: error: chi-mi300x-0XX` lines.

2. **Read-only survey** of every `idle`/`mixed`/`alloc` node
   (`sinfo -N` for the list):

   ```bash
   ssh amd-vultr-mi300 'for n in $(sinfo -N -h -o "%N" | sort -u); do
     v=$(srun -w $n -N1 --immediate=20 sysctl -n kernel.apparmor_restrict_unprivileged_userns 2>&1 | tail -1)
     echo "$n: $v"
   done'
   ```

   Expect a split: failing nodes at `1`, working nodes at `0`. If ALL nodes are
   at `0` and failures persist, this is a different bug — check the enroot
   AppArmor profile coverage of `/usr/local/bin/enroot-nsenter`, enroot
   versions, and pyxis plugin state instead.

3. **Fix only with explicit user approval** (this disables a kernel security
   mitigation — AskUserQuestion first, always). On each drifted node, set the
   flag to the cluster's working baseline and persist it:

   ```bash
   ssh amd-vultr-mi300 'for n in <drifted nodes>; do
     srun -w $n -N1 --immediate=30 bash -c \
       "sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 && \
        echo kernel.apparmor_restrict_unprivileged_userns=0 > /etc/sysctl.d/99-enroot-userns.conf"
   done'
   ```

   Verify each node reads back `0` and the sysctl.d file exists (survives
   reboot).

4. **Rerun the affected sweeps**: `gh run rerun <id> --failed` for each failed
   run; full `gh run rerun` if a cancelled run refuses partial rerun.

5. **Flag the durable fix**: the sysctl belongs in the node provisioning image
   — any node that gets (re)provisioned without it will regress (watch nodes
   listed as down with "provisioning incomplete" in `sinfo`; they'll likely
   come up drifted). Ping the cluster owners rather than treating step 3 as
   permanent.
