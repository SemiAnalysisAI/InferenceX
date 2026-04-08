#!/usr/bin/env python3
"""
Copy a container image from GHCR to Docker Hub using the registry API.
Streams each blob directly without writing to disk.

Usage:
  python3 push-to-dockerhub.py

Requires env vars:
  GH_PAT          - GitHub PAT with packages:read scope
  DOCKER_HUB_PAT  - Docker Hub PAT for the target account

Hardcoded:
  SRC: ghcr.io/jordannanos/sgl-mi325x-mori:v0.5.9-bnxt-good
  DST: docker.io/semianalysiswork/sgl-bnxt-cdna3:v0.5.9-bnxt
"""

import hashlib
import json
import os
import sys
import urllib.request
import urllib.error

GH_PAT = os.environ["GH_PAT"]
DOCKER_HUB_PAT = os.environ["DOCKER_HUB_PAT"]

SRC_REPO = "jordannanos/sgl-mi325x-mori"
SRC_TAG = "v0.5.9-bnxt-good"
DST_NAMESPACE = "semianalysiswork"
DST_REPO = "sgl-bnxt-cdna3"
DST_TAG = "v0.5.9-bnxt"

GHCR_REG = "https://ghcr.io"
DH_REG = "https://registry-1.docker.io"
DH_AUTH = "https://auth.docker.io"

CHUNK = 32 * 1024 * 1024  # 32 MiB upload chunks


def ghcr_token(scope: str) -> str:
    import base64
    creds = base64.b64encode(f"jordannanos:{GH_PAT}".encode()).decode()
    req = urllib.request.Request(
        f"{GHCR_REG}/token?scope={scope}&service=ghcr.io",
        headers={"Authorization": f"Basic {creds}"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)["token"]


def dh_token(scope: str) -> str:
    import base64
    creds = base64.b64encode(f"clustermax:{DOCKER_HUB_PAT}".encode()).decode()
    req = urllib.request.Request(
        f"{DH_AUTH}/token?service=registry.docker.io&scope={scope}",
        headers={"Authorization": f"Basic {creds}"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)["access_token"]


def blob_exists(registry: str, token: str, repo: str, digest: str) -> bool:
    req = urllib.request.Request(
        f"{registry}/v2/{repo}/blobs/{digest}",
        method="HEAD",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req):
            return True
    except urllib.error.HTTPError as e:
        if e.code in (404, 405):
            return False
        raise


def upload_blob(
    src_registry: str,
    src_token: str,
    src_repo: str,
    dst_registry: str,
    dst_token: str,
    dst_repo: str,
    digest: str,
    size: int,
) -> None:
    # Initiate upload on Docker Hub
    init_req = urllib.request.Request(
        f"{dst_registry}/v2/{dst_repo}/blobs/uploads/",
        method="POST",
        headers={"Authorization": f"Bearer {dst_token}"},
    )
    try:
        with urllib.request.urlopen(init_req) as r:
            upload_url = r.headers.get("Location")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to initiate upload: {e.code} {e.read()[:200]}")

    # Make upload_url absolute if relative
    if upload_url.startswith("/"):
        upload_url = dst_registry + upload_url

    # Stream blob from GHCR
    src_req = urllib.request.Request(
        f"{src_registry}/v2/{src_repo}/blobs/{digest}",
        headers={"Authorization": f"Bearer {src_token}"},
    )
    with urllib.request.urlopen(src_req) as src_resp:
        data = src_resp.read()

    # Monolithic PUT upload (append digest query param)
    sep = "&" if "?" in upload_url else "?"
    put_url = f"{upload_url}{sep}digest={digest}"
    put_req = urllib.request.Request(
        put_url,
        method="PUT",
        data=data,
        headers={
            "Authorization": f"Bearer {dst_token}",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(data)),
        },
    )
    try:
        with urllib.request.urlopen(put_req) as r:
            status = r.status
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Blob upload failed: {e.code} {e.read()[:200]}"
        )
    if status not in (201, 202):
        raise RuntimeError(f"Unexpected status {status} on blob upload")


def main():
    print(f"Source: {GHCR_REG}/{SRC_REPO}:{SRC_TAG}")
    print(f"Dest:   {DH_REG}/{DST_NAMESPACE}/{DST_REPO}:{DST_TAG}")
    print()

    # Auth tokens
    print("Getting auth tokens...")
    src_scope = f"repository:{SRC_REPO}:pull"
    dst_scope = f"repository:{DST_NAMESPACE}/{DST_REPO}:push,pull"
    pull_token = ghcr_token(src_scope)
    push_token = dh_token(dst_scope)

    # Fetch manifest
    print(f"Fetching manifest from {SRC_REPO}:{SRC_TAG}...")
    manifest_req = urllib.request.Request(
        f"{GHCR_REG}/v2/{SRC_REPO}/manifests/{SRC_TAG}",
        headers={
            "Authorization": f"Bearer {pull_token}",
            "Accept": "application/vnd.docker.distribution.manifest.v2+json",
        },
    )
    with urllib.request.urlopen(manifest_req) as r:
        manifest_bytes = r.read()
    manifest = json.loads(manifest_bytes)

    blobs = [manifest["config"]] + manifest["layers"]
    total = len(blobs)
    print(f"Blobs to process: {total}")
    print()

    dst_full_repo = f"{DST_NAMESPACE}/{DST_REPO}"
    uploaded = skipped = failed = 0

    for i, blob in enumerate(blobs, 1):
        digest = blob["digest"]
        size_mb = blob["size"] / 1024 / 1024
        print(f"[{i:2d}/{total}] {digest[:19]}... ({size_mb:.1f} MiB)", end=" ", flush=True)

        # Refresh tokens every 30 blobs (they expire)
        if i % 30 == 1 and i > 1:
            pull_token = ghcr_token(src_scope)
            push_token = dh_token(dst_scope)

        # Check if already uploaded
        if blob_exists(DH_REG, push_token, dst_full_repo, digest):
            print("already exists, skipping")
            skipped += 1
            continue

        try:
            upload_blob(
                GHCR_REG, pull_token, SRC_REPO,
                DH_REG, push_token, dst_full_repo,
                digest, blob["size"],
            )
            print("uploaded")
            uploaded += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print()
    print(f"Blobs: {uploaded} uploaded, {skipped} skipped, {failed} failed")

    if failed > 0:
        print("ERROR: Some blobs failed to upload")
        sys.exit(1)

    # Push manifest
    print(f"\nPushing manifest to {dst_full_repo}:{DST_TAG}...")
    # Refresh push token
    push_token = dh_token(dst_scope)
    put_req = urllib.request.Request(
        f"{DH_REG}/v2/{dst_full_repo}/manifests/{DST_TAG}",
        method="PUT",
        data=manifest_bytes,
        headers={
            "Authorization": f"Bearer {push_token}",
            "Content-Type": "application/vnd.docker.distribution.manifest.v2+json",
        },
    )
    try:
        with urllib.request.urlopen(put_req) as r:
            status = r.status
            location = r.headers.get("Location", "")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Manifest push failed: {e.code} {e.read()[:400]}")

    print(f"Done. Status: {status}")
    print(f"Image: docker.io/{dst_full_repo}:{DST_TAG}")


if __name__ == "__main__":
    main()
