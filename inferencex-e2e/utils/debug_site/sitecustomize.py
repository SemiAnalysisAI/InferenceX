# HACK (manual e2e only, do not merge): with DEBUG_STACK_DUMP_S set, every Python process prints all its threads'
# stacks to stderr every DEBUG_STACK_DUMP_S seconds (+ pid % 60, so processes do not interleave). faulthandler's
# watchdog is a C thread, so it fires even when the main thread holds the GIL in a stuck call.
import faulthandler
import os

if os.environ.get("DEBUG_STACK_DUMP_S"):
    faulthandler.dump_traceback_later(int(os.environ["DEBUG_STACK_DUMP_S"]) + os.getpid() % 60, repeat=True)
try:  # the image's own sitecustomize, which this one shadows
    exec(open("/etc/python3.12/sitecustomize.py").read())
except OSError:
    pass
