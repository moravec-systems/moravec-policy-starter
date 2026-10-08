# syntax=docker/dockerfile:1
# Build from the repository root for a Linux x86_64 NVIDIA GPU host.
# openpi targets Python 3.11 (its .python-version); its lock has no 3.12 wheel for mujoco.
FROM python:3.11.14-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.9.21 /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH" \
    OPENPI_DATA_HOME=/cache/openpi HF_HOME=/cache/huggingface \
    XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential linux-libc-dev git ffmpeg libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Physical Intelligence's openpi, pinned, installed from its own lockfile.
ARG OPENPI_REVISION=215abfb217dbac7d5f1273282331b9b1866c0479
RUN git clone https://github.com/Physical-Intelligence/openpi /opt/openpi \
    && git -C /opt/openpi checkout ${OPENPI_REVISION} \
    && GIT_LFS_SKIP_SMUDGE=1 uv sync --python /usr/local/bin/python --project /opt/openpi --frozen --no-dev
COPY . /app
# Install the starter without re-resolving the model environment.
RUN uv pip install --python /opt/venv/bin/python --no-deps msgpack-numpy==0.4.8 -e /app \
    && python -c "from moravec_policy.policy.Pi_05.model import Model"
EXPOSE 6000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m \
    CMD python -c "from websockets.sync.client import connect; connect('ws://127.0.0.1:6000', open_timeout=3).close()"
CMD ["python", "setup_policy_server.py", "--config_path", "policy/Pi_05/deploy.yml", "--host", "0.0.0.0"]
