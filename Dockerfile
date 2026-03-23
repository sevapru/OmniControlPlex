# syntax=docker/dockerfile:1
# Build context: repo root (set in docker-compose: context: .)
# Reason: the uv.lock lives at the workspace root, not inside this sub-repo.
# The root pyproject.toml declares a uv workspace covering all three model repos.
FROM nvcr.io/nvidia/pytorch:25.08-py3

# System deps needed for:
#   ffmpeg         — MP4 video generation
#   libgl1-mesa-glx, libsm6, libxext6, libxrender-dev — headless OpenGL for trimesh/pyrender
#   libglib2.0-0   — GLib (OpenCV runtime dep)
#   git, curl      — source installs (CLIP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv — copy the static binary from the official image (ARM64-safe)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ── Workspace setup ──────────────────────────────────────────────────────────
# The uv workspace root lives here. uv traverses upward from any member dir
# to find pyproject.toml + uv.lock, so /workspace must contain them.
WORKDIR /workspace
COPY pyproject.toml uv.lock ./

# Copy this workspace member's pyproject.toml for dep resolution
# (separate COPY so layer cache only busts when THIS repo's deps change)
COPY OmniControlPlex/pyproject.toml /workspace/OmniControlPlex/

# ── Dependency install ────────────────────────────────────────────────────────
# --frozen:              use the committed lock file, fail if it's stale
# --no-install-project:  don't install omnicontrolplex itself (it's source code, not a package)
# --no-group ngc-provided: skip torch/torchvision/torchaudio (already in NGC base)
# --system:              install into the system Python (NGC's Python, which has GPU torch)
WORKDIR /workspace/OmniControlPlex
RUN uv sync --frozen --no-install-project --no-group ngc-provided --system

# ── Application code ─────────────────────────────────────────────────────────
COPY OmniControlPlex/ /workspace/OmniControlPlex/

# Download spaCy English model for text processing
RUN uv run --system python -m spacy download en_core_web_sm

# ── Runtime env ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/workspace/OmniControlPlex
ENV GRADIO_SERVER_PORT=4567
ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 4567

CMD ["uv", "run", "--system", "python", "app.py"]
