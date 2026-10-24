"""Private diagnostic import hooks; installed engine files stay unchanged."""
import os
if os.environ.get('SA_LOAD_PROBE_DIR'):
    if os.environ.get('SA_LOAD_PROBE_MODE')=='boundary_only':
        from boundary_probe import install
    else:
        from probe import install
    install()
