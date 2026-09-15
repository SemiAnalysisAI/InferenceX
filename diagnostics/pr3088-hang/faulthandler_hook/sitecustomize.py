"""Injected only into this diagnostic benchmark's Python subprocess tree."""
import os
if os.environ.get('POWERX_STACK_DIR'):
    from registration import install
    install()
