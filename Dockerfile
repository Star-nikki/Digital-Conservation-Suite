# ─────────────────────────────────────────────────────────────────────────────
# Digital Conservation Suite — Multi-stage Dockerfile
# Optimized for Home & Campus Networks (Fixes Debian Trixie/Bookworm errors)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

# Fix: Change mirrors to bypass network blocks and update library names for new Debian versions
RUN sed -i 's|deb.debian.org|ftp.us.debian.org|g' /etc/apt/sources.list.d/debian.sources || true && \
    echo 'Acquire::Retries "5"; Acquire::http::Timeout "60";' > /etc/apt/apt.conf.d/99custom && \
    apt-get update || true && \
    apt-get install -y --fix-missing --no-install-recommends \
        gcc \
        g++ \
        libgl1 \
        libglib2.0-0t64 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create an isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel

# Copy and install dependencies (Will use your C:\Users\Nikki Rani\Downloads\pip_cache if volume is mounted)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download AI weights so they are "baked" into the image
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
RUN python -c "from torchvision import models; models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)"


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="Digital Conservation Suite"

# Fix: Ensure runtime also uses the correct library names and mirrors
RUN sed -i 's|deb.debian.org|ftp.us.debian.org|g' /etc/apt/sources.list.d/debian.sources || true && \
    apt-get update || true && \
    apt-get install -y --fix-missing --no-install-recommends \
        libgl1 \
        libglib2.0-0t64 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv
COPY --from=builder /opt/venv /opt/venv

# Copy cached model weights
COPY --from=builder /root/.config/Ultralytics /root/.config/Ultralytics
COPY --from=builder /root/.cache/torch /root/.cache/torch

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app
COPY algorithms.py .
COPY app.py .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]