#!/usr/bin/env python3
import json, os, pathlib, re, shutil, subprocess, sys, time
out = pathlib.Path(sys.argv[1])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
receipt = {"probe_pid": child.pid, "probe_owner": os.getuid(), "tools": [], "scope": "own disposable CPU child only"}
try:
    for name in ("py-spy", "gdb"):
        tool = shutil.which(name)
        if not tool:
            receipt["tools"].append({"name": name, "available": False})
            continue
        cmd = ([tool, "dump", "--pid", str(child.pid)] if name == "py-spy" else
               [tool, "-q", "-n", "--batch", "-p", str(child.pid), "-ex", "set pagination off", "-ex", "thread apply all bt", "-ex", "detach"])
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5, text=True)
            text, rc = result.stdout, result.returncode
        except subprocess.TimeoutExpired as exc:
            text, rc = str(exc), 124
        out.joinpath(name + "-self-child.log").write_text(text[:1024 * 1024])
        receipt["tools"].append({"name": name, "available": True, "path": tool, "command": cmd, "returncode": rc, "successful_self_child_attach": rc == 0 and bool(re.search(r"(?m)^(Thread |#0\s)", text)) and "Operation not permitted" not in text})
finally:
    child.terminate()
    try: child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        child.kill(); child.wait()
    receipt["probe_child_reaped"] = child.returncode is not None
    out.joinpath("attach-capability.json").write_text(json.dumps(receipt, indent=2) + "\n")
