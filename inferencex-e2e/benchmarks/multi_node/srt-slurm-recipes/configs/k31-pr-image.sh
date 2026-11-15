#!/usr/bin/env bash
set -eo pipefail
export PATH=/opt/runtime-venv/bin:$PATH
/opt/runtime-venv/bin/python /opt/k31-provenance/verify-image.py
