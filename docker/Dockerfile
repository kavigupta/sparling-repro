FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages: Python 3.9 (matching host), texlive-full (matching host),
# PDF tools, audio library
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common gnupg curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3.9-distutils \
    texlive-full \
    poppler-utils \
    libsndfile1 \
    fontconfig \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.9 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

WORKDIR /app

# Install PyTorch first from CUDA 12.6 index (matching host torch==2.7.1+cu126)
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Install remaining Python deps (torch already satisfied, won't be reinstalled)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repo source (large data excluded via .dockerignore)
COPY . .

# Create mount points and writable dirs
RUN mkdir -p /app/selected_checkpoints /app/AudioMNIST /app/cache

ENV PYTHONPATH=/app

CMD ["python", "-m", "pytest", "tests/", "-v"]
