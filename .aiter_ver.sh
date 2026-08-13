#!/bin/bash
D=/usr/local/lib/python3.12/dist-packages
for f in "$D"/aiter*.dist-info/METADATA "$D"/aiter/__init__.py; do
  [ -e "$f" ] && echo "== $f" && grep -m3 -iE '^version:|__version__|commit' "$f"
done
ls -d "$D"/aiter*.dist-info 2>/dev/null
