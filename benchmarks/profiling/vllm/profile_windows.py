"""Open torch profiler windows on every vLLM server during an AgentX replay.

Usage: profile_windows.py WINDOWS_JSON LOG_PATH SERVER_URL...

WINDOWS_JSON lists [delay_seconds, iterations] pairs, delays measured from
this script's start (the replay start). Each window POSTs /start_profile to
every server; vLLM stops on its own after its configured max_iterations, and
a /stop_profile after a grace period closes a window that did not fill (for
example when the servers went idle).
"""

import json
import sys
import time
import urllib.request


def post(url: str) -> str:
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return f"{response.status}"
    except Exception as error:  # a failed window must not fail the benchmark
        return f"error: {error}"


def main() -> None:
    windows = json.loads(sys.argv[1])
    log_path = sys.argv[2]
    servers = [url.rstrip("/") for url in sys.argv[3:]]
    start = time.monotonic()
    with open(log_path, "a", buffering=1) as log:
        def note(**record):
            record["t_unix"] = time.time()
            log.write(json.dumps(record) + "\n")

        note(event="servers", servers=servers, windows=windows)
        for index, (delay, iterations) in enumerate(windows):
            time.sleep(max(0.0, start + float(delay) - time.monotonic()))
            for server in servers:
                note(event="start", window=index, server=server, iterations=iterations,
                     status=post(f"{server}/start_profile"))
            # Generous grace: a step is well under a second at every agentic point.
            time.sleep(max(120.0, 4.0 * float(iterations)))
            for server in servers:
                note(event="stop", window=index, server=server,
                     status=post(f"{server}/stop_profile"))


if __name__ == "__main__":
    main()
