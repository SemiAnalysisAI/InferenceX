"""Fixed-shape load for synthetic profiling: a prefill phase, then a decode phase."""

import json
import os
import random
import sys
import threading
import time
import urllib.request

PHASE_MAX_S = float(os.environ.get("INFX_SYNTH_PHASE_MAX_S") or 900)
TOKEN_RANGE = (1000, 30000)


def post(url, body, timeout=1800):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def model_name(server):
    with urllib.request.urlopen(f"{server}/v1/models", timeout=60) as response:
        return json.loads(response.read())["data"][0]["id"]


def first_window_done(windows_log):
    """The window client has closed window 0 (its clock note follows the profiler stop)."""
    try:
        with open(windows_log) as f:
            return any(json.loads(line).get("event") == "clocks" and json.loads(line).get("window") == 0
                       for line in f if line.strip())
    except (OSError, ValueError):
        return False


def run_phase(name, server, model, conc, isl, osl, done, stats, phase_log):
    with open(phase_log, "a", buffering=1) as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - synthetic_load - NOTICE - "
                f"Phase {name} ({name}) started | conc={conc} isl={isl} osl={osl}\n")
    deadline = time.time() + PHASE_MAX_S
    rng = random.Random(name)
    lock = threading.Lock()
    wave = threading.Barrier(conc)

    def lane():
        while not done() and time.time() < deadline:
            try:
                wave.wait(timeout=max(1.0, deadline - time.time()))
            except threading.BrokenBarrierError:
                return
            with lock:
                prompt = [rng.randrange(*TOKEN_RANGE) for _ in range(isl)]
            body = {"model": model, "prompt": prompt, "max_tokens": osl, "min_tokens": osl,
                    "ignore_eos": True, "temperature": 0.0}
            try:
                post(f"{server}/v1/completions", body)
                key = "completed"
            except Exception as error:
                key = "errors"
                stats.setdefault("last_error", repr(error)[:300])
            with lock:
                stats[key] = stats.get(key, 0) + 1

    started = time.time()
    for _ in range(conc):
        threading.Thread(target=lane, daemon=True).start()
    while not done() and time.time() < deadline:
        time.sleep(0.5)
    stats["seconds"] = round(time.time() - started, 1)
    stats["stopped_by"] = "deadline" if time.time() >= deadline else "window client"


def main():
    server, phase_log, windows_log, result_json = sys.argv[1:5]
    conc = int(os.environ["CONC"])
    isl = int(os.environ.get("INFX_SYNTH_ISL") or 8192)
    osl = int(os.environ.get("INFX_SYNTH_OSL") or 4096)
    stop_file = os.environ["INFX_SYNTH_STOP_FILE"]
    model = model_name(server)
    result = {"mode": "synthetic", "model": model, "conc": conc, "isl": isl, "osl": osl,
              "prefill": {}, "decode": {}}
    run_phase("warmup", server, model, conc, isl, 1, lambda: first_window_done(windows_log),
              result["prefill"], phase_log)
    run_phase("profiling", server, model, conc, isl, osl, lambda: os.path.exists(stop_file),
              result["decode"], phase_log)
    with open(result_json, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result))
    return 0 if result["prefill"].get("completed") else 1


if __name__ == "__main__":
    sys.exit(main())
