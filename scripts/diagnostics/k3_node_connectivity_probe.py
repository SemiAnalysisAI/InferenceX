#!/usr/bin/env python3
"""Temporary two-rank TCP connectivity probe for the selected MI300X nodes."""

from __future__ import annotations

import os
import socket
import threading
import time


TARGETS = (
    ("management", "107.191.51.90", 29601),
    ("private", "10.162.224.147", 29602),
)


def serve(label: str, address: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((address, port))
        server.listen(1)
        server.settimeout(20)
        connection, peer = server.accept()
        with connection:
            payload = connection.recv(16)
        print(
            f"K3_CONNECTIVITY server label={label} address={address} "
            f"peer={peer[0]} payload={payload.decode()}",
            flush=True,
        )


rank = int(os.environ["SLURM_PROCID"])
if rank == 0:
    threads = [
        threading.Thread(target=serve, args=target, daemon=False)
        for target in TARGETS
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
else:
    time.sleep(2)
    for label, address, port in TARGETS:
        with socket.create_connection((address, port), timeout=10) as connection:
            connection.sendall(label.encode())
        print(
            f"K3_CONNECTIVITY client label={label} address={address} status=ok",
            flush=True,
        )
