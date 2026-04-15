#!/usr/bin/env bash
# setup_local_nvme.sh — Format and mount local NVMe drives for model caching.
#
# Detects unformatted/unmounted NVMe drives and sets up a mount point for
# caching model weights locally. Designed to be run once per node (idempotent).
#
# Usage (run on each compute node, requires root):
#   sudo bash utils/setup_local_nvme.sh [mount_point]
#
# Default mount point: /local-nvme
#
# This script:
#   1. Finds the first available NVMe drive that is not the boot device
#   2. Formats it with ext4 if not already formatted
#   3. Mounts it at the specified mount point
#   4. Adds an fstab entry for persistence across reboots
#
# For RAID-0 across multiple NVMe drives (maximum throughput), use:
#   sudo bash utils/setup_local_nvme.sh --raid [mount_point]

set -euo pipefail

USE_RAID=false
MOUNT_POINT="/local-nvme"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raid) USE_RAID=true; shift ;;
        *) MOUNT_POINT="$1"; shift ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo)" >&2
    exit 1
fi

echo "[nvme-setup] Mount point: $MOUNT_POINT"

# Already mounted?
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "[nvme-setup] $MOUNT_POINT is already mounted:"
    df -h "$MOUNT_POINT"
    exit 0
fi

# Find NVMe drives that are not part of the root filesystem
ROOT_DEV=$(findmnt -n -o SOURCE / | sed 's/[0-9]*$//' | sed 's/p$//')
NVME_DRIVES=()
for dev in /dev/nvme*n1; do
    [[ -b "$dev" ]] || continue
    # Skip if this drive is part of root
    if [[ "$dev" == "$ROOT_DEV"* ]]; then
        echo "[nvme-setup] Skipping $dev (root device)"
        continue
    fi
    # Skip if already mounted
    if mount | grep -q "^$dev "; then
        echo "[nvme-setup] Skipping $dev (already mounted)"
        continue
    fi
    # Skip if part of an md array
    if grep -q "$(basename "$dev")" /proc/mdstat 2>/dev/null; then
        echo "[nvme-setup] Skipping $dev (part of md array)"
        continue
    fi
    NVME_DRIVES+=("$dev")
done

if [[ ${#NVME_DRIVES[@]} -eq 0 ]]; then
    echo "[nvme-setup] No available NVMe drives found."
    exit 1
fi

echo "[nvme-setup] Found ${#NVME_DRIVES[@]} available NVMe drives: ${NVME_DRIVES[*]}"

if [[ "$USE_RAID" == true ]] && [[ ${#NVME_DRIVES[@]} -gt 1 ]]; then
    # RAID-0 for maximum throughput
    MD_DEV="/dev/md10"
    echo "[nvme-setup] Creating RAID-0 array across ${#NVME_DRIVES[@]} drives..."

    if [[ -b "$MD_DEV" ]]; then
        echo "[nvme-setup] $MD_DEV already exists, using it"
    else
        mdadm --create "$MD_DEV" --level=0 --raid-devices=${#NVME_DRIVES[@]} "${NVME_DRIVES[@]}" --run
    fi

    TARGET_DEV="$MD_DEV"
else
    # Single drive (use the first available)
    TARGET_DEV="${NVME_DRIVES[0]}"
    echo "[nvme-setup] Using single drive: $TARGET_DEV"
fi

# Format if needed
if ! blkid "$TARGET_DEV" | grep -q 'TYPE="ext4"'; then
    echo "[nvme-setup] Formatting $TARGET_DEV with ext4..."
    mkfs.ext4 -F -L local-nvme "$TARGET_DEV"
else
    echo "[nvme-setup] $TARGET_DEV already has ext4 filesystem"
fi

# Mount
mkdir -p "$MOUNT_POINT"
mount -o noatime,discard "$TARGET_DEV" "$MOUNT_POINT"

# Set permissions so non-root users can write
chmod 1777 "$MOUNT_POINT"

# Add fstab entry if not present
if ! grep -q "$MOUNT_POINT" /etc/fstab; then
    UUID=$(blkid -s UUID -o value "$TARGET_DEV")
    echo "UUID=$UUID $MOUNT_POINT ext4 noatime,discard,nofail 0 2" >> /etc/fstab
    echo "[nvme-setup] Added fstab entry"
fi

echo "[nvme-setup] Done:"
df -h "$MOUNT_POINT"
