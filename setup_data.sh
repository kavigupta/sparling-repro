#!/usr/bin/env bash
set -euo pipefail

# Download required data (spliceai_data, AudioMNIST) and set up cache dir.
#
# Usage:
#   ./setup_data.sh                       # download from Hugging Face
#   ./setup_data.sh /path/to/archives     # use local archive files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_DIR="${1:-}"

HF_REPO="kavigupta/sparling-repro-data"
HF_URL="https://huggingface.co/datasets/${HF_REPO}/resolve/main"

cd "$SCRIPT_DIR"

download_and_extract() {
    local archive="$1"
    local tarball

    if [[ -n "$ARCHIVE_DIR" ]]; then
        tarball="${ARCHIVE_DIR}/${archive}"
    else
        tarball="${SCRIPT_DIR}/${archive}"
        if [[ ! -f "$tarball" ]]; then
            echo "Downloading ${archive} ..."
            curl -L --retry 5 --retry-delay 10 -C - -o "$tarball" "${HF_URL}/${archive}"
        fi
    fi

    if [[ ! -f "$tarball" ]]; then
        echo "WARNING: ${tarball} not found, skipping."
        return 1
    fi

    echo "Extracting ${archive} ..."
    tar -xzf "$tarball"

    # Clean up downloaded file (keep if using local archive dir)
    if [[ -z "$ARCHIVE_DIR" ]]; then
        rm -f "$tarball"
    fi
}

# spliceai_data
if [[ -d "spliceai_data" ]]; then
    echo "spliceai_data/ already exists, skipping."
else
    download_and_extract spliceai_data.tar.gz
fi

# AudioMNIST
if [[ -d "AudioMNIST" ]]; then
    echo "AudioMNIST/ already exists, skipping."
else
    echo "Cloning AudioMNIST ..."
    git clone https://github.com/soerenab/AudioMNIST.git AudioMNIST
fi

mkdir -p cache

echo "Data setup complete."
