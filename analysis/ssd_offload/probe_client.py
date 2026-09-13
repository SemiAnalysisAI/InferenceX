#!/usr/bin/env python3
"""Drive the SSD-offload probe and print a verdict.

The measurement is deliberately crude because the question is binary: does KV
land on disk and come back. Three phases --

  cold    first send of a long prompt, nothing cached anywhere
  flush   distinct long prompts, enough of them to evict the cold one out of
          GPU KV and out of LMCache's 1 GB CPU tier
  warm    the cold prompt again; a hit here can only have come off disk

Disk bytes are sampled around each phase so a "fast" warm response that wrote
and read nothing can be recognised as prefix caching rather than offload.
"""
import json
import os
import random
import subprocess
import time
import urllib.request

PORT = os.environ["PORT"]
MODEL = os.environ["MODEL"]
DISK_DIR = os.environ["DISK_DIR"]
RESULT_DIR = os.environ["RESULT_DIR"]
URL = f"http://127.0.0.1:{PORT}/v1/completions"

# ~24k tokens per prompt: long enough that offload is worth doing and that a
# handful of them exceed the GPU KV pool at max-num-seqs 8.
WORDS_PER_PROMPT = int(os.environ.get("SSD_PROBE_WORDS", "18000"))
N_FLUSH = int(os.environ.get("SSD_PROBE_FLUSH", "12"))

VOCAB = [
    "cache", "tensor", "kernel", "latency", "throughput", "offload", "block",
    "prefix", "attention", "router", "scheduler", "segment", "bandwidth",
    "device", "stream", "capture", "quantize", "expert", "gate", "token",
]


def make_prompt(seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(VOCAB) for _ in range(WORDS_PER_PROMPT))


def send(prompt: str) -> float:
    """Return wall time for a 1-token completion: prefill cost, essentially."""
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": 1,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        resp.read()
    return time.perf_counter() - t0


def disk_bytes() -> int:
    out = subprocess.run(["du", "-sb", DISK_DIR], capture_output=True, text=True)
    try:
        return int(out.stdout.split()[0])
    except (ValueError, IndexError):
        return -1


def main() -> None:
    subject = make_prompt(0)

    b_start = disk_bytes()
    t_cold = send(subject)
    b_after_cold = disk_bytes()
    print(f"cold: {t_cold:.2f}s  disk {b_start} -> {b_after_cold}", flush=True)

    flush_times = []
    for i in range(1, N_FLUSH + 1):
        flush_times.append(send(make_prompt(i)))
        print(f"flush {i}/{N_FLUSH}: {flush_times[-1]:.2f}s "
              f"disk {disk_bytes()}", flush=True)
    b_after_flush = disk_bytes()

    t_warm = send(subject)
    b_end = disk_bytes()
    print(f"warm: {t_warm:.2f}s  disk {b_after_flush} -> {b_end}", flush=True)

    # A second cold prompt bounds what "no reuse at all" costs right now, so
    # the warm number is compared against a contemporary baseline rather than
    # against the very first request (which also paid warmup costs).
    t_cold2 = send(make_prompt(999))
    print(f"cold2: {t_cold2:.2f}s", flush=True)

    wrote = b_after_cold > b_start if b_start >= 0 else None
    speedup = (t_cold2 / t_warm) if t_warm > 0 else 0.0
    verdict = {
        "ok": bool(wrote) and speedup >= 1.5,
        "disk_dir": DISK_DIR,
        "disk_device": os.environ.get("DISK_SRC"),
        "disk_fstype": os.environ.get("DISK_FSTYPE"),
        "dd_write": os.environ.get("DD_W"),
        "dd_read": os.environ.get("DD_R"),
        "bytes_start": b_start,
        "bytes_after_cold": b_after_cold,
        "bytes_after_flush": b_after_flush,
        "bytes_end": b_end,
        "kv_written_to_disk": wrote,
        "t_cold_s": round(t_cold, 3),
        "t_warm_s": round(t_warm, 3),
        "t_cold_contemporary_s": round(t_cold2, 3),
        "warm_speedup_vs_cold": round(speedup, 3),
        "flush_prompts": N_FLUSH,
        "words_per_prompt": WORDS_PER_PROMPT,
    }
    path = os.path.join(RESULT_DIR, "ssd_offload_probe.json")
    with open(path, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print("SSD OFFLOAD VERDICT " + json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
