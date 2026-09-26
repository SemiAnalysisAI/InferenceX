# <img src="logo.png" alt="PySlurm Logo">

pyslurm is the Python client library for the [Slurm Workload Manager](https://slurm.schedmd.com)

## Requirements

* [Slurm](https://slurm.schedmd.com) 25.11.x — shared library and header files
* [Python](https://www.python.org) >= 3.6

## Versioning

PySlurm uses the format `X.Y.Z` where `X.Y` tracks the Slurm major release
and `Z` is PySlurm's own patch increment, reset to `0` on each new Slurm
major release.

| PySlurm | Slurm |
|---|---|
| 25.11.x | 25.11.x |
| 24.05.x | 24.05.x |
| 23.11.x | 23.11.x |

## Installation

PySlurm requires the Slurm development headers and shared library at build
time.

### Slurm header and library paths

If you have `slurm-devel` installed in the default paths, skip this section.

By default, PySlurm looks in:

* `/usr/include` — for Slurm header files (`slurm/slurm.h`)
* `/usr/lib64` — for the Slurm shared library (`libslurmfull.so`)

If your Slurm installation is not in the default paths, set these before
installing:

```shell
export SLURM_INCLUDE_DIR=/path/to/slurm/include
export SLURM_LIB_DIR=/path/to/slurm/lib
```

### From PyPI

```shell
pip install pyslurm
```

### From source

```shell
git clone https://github.com/PySlurm/pyslurm.git && cd pyslurm
scripts/build.sh
```

Use `-j` to build with multiple cores (or set `PYSLURM_BUILD_JOBS`):

```shell
scripts/build.sh -j4
```

## Contributors

pyslurm is made by [contributors like
you](https://github.com/PySlurm/pyslurm/graphs/contributors).

## Support

Feel free to ask questions in the [GitHub
Discussions](https://github.com/orgs/PySlurm/discussions)

Found a bug or you are missing a feature? Feel free to [open an Issue!](https://github.com/PySlurm/pyslurm/issues/new)
