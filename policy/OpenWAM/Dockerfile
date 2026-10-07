# syntax=docker/dockerfile:1
# Build from the repository root on Linux x86_64.
FROM python:3.12.15-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/cache/huggingface NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libglib2.0-0 libgl1 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY policy/OpenWAM/requirements.lock /tmp/requirements.lock
RUN python -m pip install --upgrade pip==25.3 \
    && python -m pip install -r /tmp/requirements.lock
# OpenWAM, pinned. Editable so the adapter can read its configs/deploy.yaml.
ARG OPENWAM_REVISION=48bd67b89d489b14d03b8d92bc66e65d306df32e
RUN git clone https://github.com/OpenWAM-Official/OpenWAM /opt/OpenWAM \
    && git -C /opt/OpenWAM checkout ${OPENWAM_REVISION} \
    && python -m pip install --no-deps -e /opt/OpenWAM
COPY . /app
RUN python -m pip install --no-deps -e /app \
    && mkdir -p /env_cfg/robot \
    && cp utils/robot/_robot_info.json /env_cfg/robot/_robot_info.json \
    && printf 'config:\n  robot: arx_x5\n' > /env_cfg/arx_x5.yml \
    && python -c "from moravec_policy.policy.OpenWAM.model import Model; from openwam.deploy.server import build_server_from_config"
EXPOSE 6000
# The WebSocket opens after checkpoint initialization; this does not test inference.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m \
    CMD python -c "from websockets.sync.client import connect; connect('ws://127.0.0.1:6000', open_timeout=3).close()"
CMD ["python", "setup_policy_server.py", "--config_path", "policy/OpenWAM/deploy.yml", "--overrides", "host=0.0.0.0", "ckpt_dir=/weights/openwam-sim", "allow_dummy_policy=false"]
