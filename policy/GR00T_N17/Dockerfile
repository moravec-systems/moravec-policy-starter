# syntax=docker/dockerfile:1
# Build from the repository root for a Linux x86_64 NVIDIA GPU host.
FROM python:3.10.19-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/cache/huggingface NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libglib2.0-0 libgl1 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY policy/GR00T_N17/requirements.lock /tmp/requirements.lock
RUN python -m pip install --upgrade pip==25.3 \
    && python -m pip install -r /tmp/requirements.lock
# NVIDIA's GR00T source, pinned. Dependencies are pinned above; training extras are not needed.
ARG GR00T_REVISION=23ace64f17aa5015259b8609d371eb61a357c776
RUN python -m pip install --no-deps "gr00t @ git+https://github.com/NVIDIA/Isaac-GR00T@${GR00T_REVISION}"
COPY . /app
RUN python -m pip install --no-deps -e /app \
    && python -c "from moravec_policy.policy.GR00T_N17.model import Model"
EXPOSE 6000
# WebSocket opens only after model initialization. This does not test inference.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m \
    CMD python -c "from websockets.sync.client import connect; connect('ws://127.0.0.1:6000', open_timeout=3).close()"
CMD ["python", "setup_policy_server.py", "--config_path", "policy/GR00T_N17/deploy.yml", "--host", "0.0.0.0"]
