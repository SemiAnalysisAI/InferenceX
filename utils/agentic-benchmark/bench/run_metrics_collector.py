#!/usr/bin/env python3
"""
Standalone metrics collector for vLLM server.

Polls the vLLM /metrics endpoint and generates server-side plots.
Designed to run alongside any benchmark client (aiperf, custom, etc.).

Usage:
    # Start collecting, run your benchmark, then Ctrl+C or kill to stop:
    python -m bench.run_metrics_collector \
        --url http://localhost:8888 \
        --output-prefix results/metrics \
        --duration 600

    # Or run in background and signal when done:
    python -m bench.run_metrics_collector \
        --url http://localhost:8888 \
        --output-prefix results/metrics \
        --pid-file /tmp/metrics_collector.pid
"""

import argparse
import asyncio
import os
import signal
import sys

from bench.metrics_collector import MetricsCollector


async def run(args):
    collector = MetricsCollector(
        base_url=args.url,
        poll_interval=args.poll_interval,
    )

    collector.start()
    print(f"Metrics collector started (polling {args.url}/metrics every {args.poll_interval}s)")

    if args.pid_file:
        with open(args.pid_file, "w") as f:
            f.write(str(os.getpid()))
        print(f"PID written to {args.pid_file}")

    # Set up graceful shutdown
    stop_event = asyncio.Event()

    def handle_signal(*_):
        print("\nStopping metrics collector...")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    # Wait for duration or signal
    if args.duration:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=args.duration)
        except asyncio.TimeoutError:
            print(f"Duration limit reached ({args.duration}s)")
    else:
        await stop_event.wait()

    await collector.stop()

    # Generate outputs
    if len(collector.snapshots) < 2:
        print("Not enough data points collected")
        sys.exit(1)

    print(f"Collected {len(collector.snapshots)} snapshots")

    # Generate plots (without client metrics — server-only)
    collector.generate_plots(output_prefix=args.output_prefix)

    # Export CSV
    collector.export_csv(output_prefix=args.output_prefix)

    # Clean up PID file
    if args.pid_file and os.path.exists(args.pid_file):
        os.remove(args.pid_file)

    print("Done")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone vLLM metrics collector"
    )
    parser.add_argument(
        "--url", "-u",
        default="http://localhost:8888",
        help="vLLM server base URL (default: http://localhost:8888)",
    )
    parser.add_argument(
        "--output-prefix", "-o",
        default="metrics",
        help="Output file prefix (default: metrics)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=None,
        help="Max collection duration in seconds (default: unlimited, stop with signal)",
    )
    parser.add_argument(
        "--pid-file",
        default=None,
        help="Write PID to this file for external signaling",
    )
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
