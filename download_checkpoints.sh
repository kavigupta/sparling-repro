#!/usr/bin/env bash
set -euo pipefail

# Download pre-trained model checkpoints from Hugging Face.
#
# This is an alternative to training models yourself and running
# select_checkpoints.py.  The checkpoints are extracted into
# selected_checkpoints/.
#
# Usage:
#   ./download_checkpoints.sh                   # download from Hugging Face
#   ./download_checkpoints.sh /path/to/archives # use local archive files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="${1:-}"

HF_REPO="kavigupta/sparling-repro-data"
HF_URL="https://huggingface.co/datasets/${HF_REPO}/resolve/main"

ARCHIVES=(
    selected_checkpoints_pae_early.tar.gz
    selected_checkpoints_pae_late.tar.gz
    selected_checkpoints_other.tar.gz
)

cd "$SCRIPT_DIR"

for archive in "${ARCHIVES[@]}"; do
    if [[ -n "$ARCHIVE_DIR" ]]; then
        tarball="${ARCHIVE_DIR}/${archive}"
    else
        tarball="${SCRIPT_DIR}/${archive}"
        if [[ ! -f "$tarball" ]]; then
            echo "Downloading ${archive} ..."
            curl -L -o "$tarball" "${HF_URL}/${archive}"
        fi
    fi

    if [[ ! -f "$tarball" ]]; then
        echo "WARNING: ${tarball} not found, skipping."
        continue
    fi

    echo "Extracting ${archive} ..."
    tar -xzf "$tarball"

    # Clean up downloaded file (keep if using local archive dir)
    if [[ -z "$ARCHIVE_DIR" ]]; then
        rm -f "$tarball"
    fi
done

echo "Checkpoints downloaded to selected_checkpoints/."
