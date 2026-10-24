"""Private startup hooks; installed package source and native args are unchanged."""
import os
if os.environ.get('SA_LOAD_PROBE_DIR'):
    from probe import install
    install()
