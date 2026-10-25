import argparse
import json
import signal
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--moriio-workers", required=True)
    known, args = parser.parse_known_args()
    with open(known.moriio_workers) as stream:
        specs = json.load(stream)
    port = args[args.index("--port") + 1]
    base = f"http://127.0.0.1:{port}"
    child = subprocess.Popen(["/tmp/smg-router-venv/bin/smg", "launch", *args])

    def forward(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        for _ in range(240):
            if child.poll() is not None:
                raise RuntimeError(
                    f"SMG exited before registration: {child.returncode}"
                )
            try:
                urllib.request.urlopen(base + "/workers", timeout=3).close()
                break
            except (OSError, urllib.error.URLError):
                time.sleep(0.5)
        else:
            raise TimeoutError("SMG registration endpoint did not start")
        for spec in specs:
            req = urllib.request.Request(
                base + "/workers",
                json.dumps(spec).encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                print(
                    "SMG registered",
                    spec["worker_type"],
                    spec["url"],
                    response.status,
                    response.read().decode(),
                    flush=True,
                )
        raise SystemExit(child.wait())
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
