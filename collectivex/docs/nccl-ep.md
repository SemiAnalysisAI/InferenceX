# NCCL-EP nightly version

`runtime/common.sh` pins `nccl-extensions[cu13]==0.2.0.dev20261005` alongside
`nccl4py[cu13]==0.5.0`. Installation uses `--pre --extra-index-url https://pypi.nvidia.com`;
the package specification keys the backend cache. Each result records the installed
wheel version verbatim in `implementation.library_version`, including the nightly date.
The value comes from distribution metadata, not the requested pin or native library version.
