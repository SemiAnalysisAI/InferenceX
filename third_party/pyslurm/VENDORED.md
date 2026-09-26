# PySlurm 25.11.2 — Vendored Source

- **Version**: 25.11.2
- **Origin**: PyPI via `sfw.semianalysis.com` package proxy
- **Tarball SHA-256**: `6e0e4bc639ea3177591b102b77c1dd8df55a9e411ac42ef339a4adb3908cca34`
- **License**: GPLv2 (see `COPYING.txt`)
- **Upstream**: <https://github.com/PySlurm/pyslurm>

## What was dropped

`docs/`, `examples/`, `tests/`, `pyslurm.egg-info/` were excluded to keep the
vendor tree small. They are not needed at build time.

## How to re-vendor

```bash
# 1. Download from the package proxy
pip download --no-deps --no-binary :all: \
    --index-url https://sfw.semianalysis.com/simple/ \
    pyslurm==25.11.2 -d /tmp/pyslurm_dl

# 2. Verify SHA-256
sha256sum /tmp/pyslurm_dl/pyslurm-25.11.2.tar.gz
# expect: 6e0e4bc639ea3177591b102b77c1dd8df55a9e411ac42ef339a4adb3908cca34

# 3. Extract, drop unneeded dirs, copy to third_party/pyslurm/
tar xzf /tmp/pyslurm_dl/pyslurm-25.11.2.tar.gz -C /tmp/pyslurm_dl
rsync -a --exclude='docs/' --exclude='examples/' --exclude='tests/' \
    --exclude='pyslurm.egg-info/' \
    /tmp/pyslurm_dl/pyslurm-25.11.2/ third_party/pyslurm/
```
