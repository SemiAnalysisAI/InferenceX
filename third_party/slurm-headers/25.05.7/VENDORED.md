# Slurm 25.05.7 Public Headers

Vendored public C headers from SchedMD Slurm, for building PySlurm on clusters
that have the Slurm runtime (`libslurm.so`, `sinfo`, etc.) but not the
development headers (`slurm-dev` / `slurm-smd-dev` package).

- **Version**: 25.05.7
- **Source**: `https://github.com/SchedMD/slurm` tag `slurm-25-05-7-1`
- **License**: GPLv2 (see the notice at the top of each header)

`slurm_version.h` is generated (not in the source tree); the copy here was
synthesised with the correct `SLURM_VERSION_NUMBER` for 25.05.7 (`0x190507`).

## How to re-vendor for a different Slurm minor

```bash
TAG="slurm-25-05-7-1"        # adjust
VER="25.05.7"
DEST="third_party/slurm-headers/$VER/slurm"
mkdir -p "$DEST"
for h in slurm.h slurmdb.h slurm_errno.h slurm_version.h spank.h pmi.h; do
  gh api "repos/SchedMD/slurm/contents/slurm/$h?ref=$TAG" --jq '.content' \
    | base64 -d > "$DEST/$h" 2>/dev/null || true
done
# Generate slurm_version.h if the tree only has slurm_version.h.in
```
