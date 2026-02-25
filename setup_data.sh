#!/usr/bin/env bash
set -euo pipefail

# Download and extract data archives, and clone AudioMNIST.
#
# Usage:
#   ./setup_data.sh                       # download from Hugging Face
#   ./setup_data.sh /path/to/archives     # use local archive files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="${1:-}"

HF_REPO="kavigupta/sparling-repro-data"
HF_URL="https://huggingface.co/datasets/${HF_REPO}/resolve/main"

ARCHIVES=(
    selected_checkpoints_pae_early.tar.gz
    selected_checkpoints_pae_late.tar.gz
    selected_checkpoints_other.tar.gz
    spliceai_data.tar.gz
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

if [[ -d "AudioMNIST" ]]; then
    echo "AudioMNIST/ already exists, skipping."
else
    echo "Cloning AudioMNIST ..."
    git clone https://github.com/soerenab/AudioMNIST.git AudioMNIST
fi

mkdir -p cache

echo "Data setup complete."
