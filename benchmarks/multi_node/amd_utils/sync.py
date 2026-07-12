#!/usr/bin/env python3
"""
Multi-node synchronization utilities for disaggregated inference.

Subcommands:
    barrier  - Wait until all specified nodes have opened their ports (TCP barrier)
               Optionally wait for HTTP health endpoints to return 200
    wait     - Block until a remote port closes (shutdown coordination)
"""

import socket
import time
import threading
import argparse
import sys
import urllib.request
import urllib.error


def is_port_open(ip, port, timeout=2):
    """Check if a given IP and port are accessible."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((ip, port)) == 0


def check_health(ip, port, path="/health", timeout=2):
    """Return True if http://ip:port/path returns HTTP 200."""
    try:
        url = f"http://{ip}:{port}{path}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return getattr(resp, "status", 200) == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


# =============================================================================
# barrier subcommand
# =============================================================================

def cmd_barrier(args):
    """Wait until all nodes have opened the specified ports."""
    NODE_IPS = [ip.strip() for ip in args.node_ips.split(",") if ip.strip()]
    NODE_PORTS = [int(p.strip()) for p in args.node_ports.split(",") if p.strip()]

    if not NODE_IPS:
        print("Error: NODE_IPS argument is empty or not set.")
        sys.exit(1)

    if len(NODE_PORTS) == 1:
        NODE_PORTS *= len(NODE_IPS)
    elif len(NODE_PORTS) != len(NODE_IPS):
        print("Error: Number of ports must match number of node IPs or only one port should be given for all.")
        sys.exit(1)

    server_socket = None

    def open_port():
        nonlocal server_socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((args.local_ip, args.local_port))
        server_socket.listen(5)
        print(f"Port {args.local_port} is now open on {args.local_ip}.")
        while True:
            conn, addr = server_socket.accept()
            conn.close()

    def close_port():
        nonlocal server_socket
        if server_socket:
            server_socket.close()
            print(f"Port {args.local_port} has been closed on {args.local_ip}.")

    if args.enable_port:
        threading.Thread(target=open_port, daemon=True).start()

    # Wait for all ports (TCP check)
    if args.wait_for_all_ports:
        start_time = time.time()
        timeout = args.timeout
        seen_open = set()

        while True:
            status = {(ip, port): is_port_open(ip, port)
                      for ip, port in zip(NODE_IPS, NODE_PORTS)}

            # Fail fast: a port that was open and is now closed means that
            # server died/was killed; don't wait out the full timeout.
            regressed = [t for t in seen_open if not status[t]]
            if regressed:
                print("ERROR: the following ports were open then went down (server died):", flush=True)
                for ip, port in regressed:
                    print(f"  - {ip}:{port}", flush=True)
                sys.exit(1)
            seen_open.update(t for t, ok in status.items() if ok)

            if all(status.values()):
                break

            if timeout > 0:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    not_open = [t for t, ok in status.items() if not ok]
                    print(f"ERROR: Timeout after {timeout} seconds waiting for ports to open.", flush=True)
                    print("The following nodes/ports are still not responding:", flush=True)
                    for ip, port in not_open:
                        print(f"  - {ip}:{port}", flush=True)
                    sys.exit(1)
                remaining = timeout - (time.time() - start_time)
                print(f"Waiting for nodes.{NODE_PORTS},{NODE_IPS} . . ({remaining:.0f}s remaining)", flush=True)
            else:
                print(f"Waiting for nodes.{NODE_PORTS},{NODE_IPS} . .", flush=True)
            time.sleep(5)

    # Wait for all health endpoints (HTTP check)
    if args.wait_for_all_health:
        health_path = args.health_endpoint
        start_time = time.time()
        timeout = args.timeout
        seen_ready = set()

        while True:
            status = {
                (ip, port): check_health(ip, port, health_path)
                for ip, port in zip(NODE_IPS, NODE_PORTS)
            }

            # Fail fast: an endpoint that was healthy and is now down means the
            # server died/was killed; don't wait out the full timeout.
            regressed = [t for t in seen_ready if not status[t]]
            if regressed:
                print(f"ERROR: the following ({health_path}) were healthy then went down (server died):", flush=True)
                for ip, port in regressed:
                    print(f"  - http://{ip}:{port}{health_path}", flush=True)
                sys.exit(1)
            seen_ready.update(t for t, ok in status.items() if ok)

            if all(status.values()):
                break

            if timeout > 0:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    not_ready = [t for t, ok in status.items() if not ok]
                    print(f"ERROR: Timeout after {timeout} seconds waiting for health endpoints.", flush=True)
                    print(f"The following (http://ip:port{health_path}) are still not responding:", flush=True)
                    for ip, port in not_ready:
                        print(f"  - http://{ip}:{port}{health_path}", flush=True)
                    sys.exit(1)
                remaining = timeout - (time.time() - start_time)
                print(
                    f"Waiting for health on {list(zip(NODE_IPS, NODE_PORTS))} ({health_path}) .. ({remaining:.0f}s remaining)",
                    flush=True,
                )
            else:
                print(f"Waiting for health on {list(zip(NODE_IPS, NODE_PORTS))} ({health_path}) ..", flush=True)
            time.sleep(30)

    if args.enable_port:
        # Keep the port open long enough for slow nodes to pass their barrier.
        # The previous 30s was too short when setup times vary by minutes.
        grace = max(60, args.timeout // 2) if args.timeout > 0 else 300
        time.sleep(grace)
        close_port()


# =============================================================================
# wait subcommand
# =============================================================================

def cmd_wait(args):
    """Wait while a remote port remains open, exit when it closes."""
    print(
        f"Waiting while port {args.remote_port} on {args.remote_ip} is open...",
        flush=True,
    )
    while is_port_open(args.remote_ip, args.remote_port):
        time.sleep(5)
    print(f"Port {args.remote_port} on {args.remote_ip} is now closed.", flush=True)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-node synchronization utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # barrier subcommand
    bp = subparsers.add_parser("barrier", help="Wait for all nodes to open specified ports.")
    bp.add_argument("--local-ip", required=False, help="Local IP address to bind the server.")
    bp.add_argument("--local-port", type=int, required=False, help="Port number to bind the server.")
    bp.add_argument("--enable-port", action="store_true", help="Enable opening and closing of local port.")
    bp.add_argument("--node-ips", required=True, help="Comma-separated list of node IPs.")
    bp.add_argument("--node-ports", required=True, help="Comma-separated list of ports to check.")
    bp.add_argument("--timeout", type=int, default=600,
                    help="Timeout in seconds (default: 600). Set to 0 for no timeout.")
    bp.add_argument("--wait-for-all-ports", action="store_true",
                    help="Wait until all node ports are open (TCP).")
    bp.add_argument("--wait-for-all-health", action="store_true",
                    help="Wait until http://ip:port/health returns 200 for all nodes.")
    bp.add_argument("--health-endpoint", default="/health",
                    help="Path for health check (default: /health).")
    bp.set_defaults(func=cmd_barrier)

    # wait subcommand
    wp = subparsers.add_parser("wait", help="Wait while a remote port remains open.")
    wp.add_argument("--remote-ip", required=True, help="Remote server IP address.")
    wp.add_argument("--remote-port", type=int, required=True, help="Remote port number.")
    wp.set_defaults(func=cmd_wait)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
