#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="latex-domain-tests"

# Data directories on the host (default: sibling/child paths relative to repo)
DATA_DIR="${DATA_DIR:-$(dirname "$REPO_DIR")}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${REPO_DIR}/selected_checkpoints}"
SPLICEAI_DIR="${SPLICEAI_DIR:-${REPO_DIR}/spliceai_data}"
AUDIOMNIST_DIR="${AUDIOMNIST_DIR:-${REPO_DIR}/AudioMNIST}"

# Build the image if it doesn't exist or if --build is passed
if [[ "${1:-}" == "--build" ]]; then
    shift
    docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$REPO_DIR"
elif ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Image '$IMAGE_NAME' not found, building..."
    docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$REPO_DIR"
fi

# Assemble volume mounts (only mount directories that exist on the host)
VOLUMES=()
if [[ -d "$CHECKPOINTS_DIR" ]]; then
    VOLUMES+=(-v "${CHECKPOINTS_DIR}:/app/selected_checkpoints:ro")
fi
if [[ -d "$SPLICEAI_DIR" ]]; then
    VOLUMES+=(-v "${SPLICEAI_DIR}:/app/spliceai_data:ro")
fi
if [[ -d "$AUDIOMNIST_DIR" ]]; then
    VOLUMES+=(-v "${AUDIOMNIST_DIR}:/app/AudioMNIST:ro")
fi

# Enable GPU passthrough if nvidia-container-toolkit is available
GPU_FLAGS=()
if docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    GPU_FLAGS=(--gpus all)
fi

# Run tests — extra args are forwarded to pytest
exec docker run --rm \
    "${GPU_FLAGS[@]}" \
    "${VOLUMES[@]}" \
    "$IMAGE_NAME" \
    python -m pytest tests/ -v "$@"
