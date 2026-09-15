"""One own-child CPU check before model startup; uses the actual registered path."""
import json
import os
from pathlib import Path
import subprocess
import sys
from observe import capture_python, proc

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
readers = {}
child = subprocess.Popen([sys.executable, '-u', '-c', 'import time; print("READY", flush=True); time.sleep(20)'], stdout=subprocess.PIPE, text=True)
try:
    if child.stdout.readline().strip() != 'READY':
        raise RuntimeError('injected child did not complete Python startup')
    result = capture_python(child.pid, os.getpid(), proc(os.getpid())['start_ticks'],
                            Path(os.environ['POWERX_STACK_DIR']), os.environ['POWERX_DIAG_JOB'],
                            output / 'registered-child-stack.txt', 256 * 1024, readers)
    if not result['bytes_saved']:
        raise RuntimeError('registered child produced no Python stack')
    (output / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
finally:
    child.terminate()
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=2)
    for fd in readers.values():
        os.close(fd)
