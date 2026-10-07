# syntax=docker/dockerfile:1
# Build from repository root for Linux x86_64 NVIDIA GPU hosts.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/cache/huggingface NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
COPY policy/MolmoAct2/requirements.lock /tmp/requirements.lock
RUN python -m pip install --upgrade pip==25.3 \
    && python -m pip install --require-hashes -r /tmp/requirements.lock
COPY . /app
RUN python -m pip install --no-deps --no-build-isolation -e /app \
    && python -c "from moravec_policy.policy.MolmoAct2.model import Model"
EXPOSE 6000
# Listening means initialization completed, not successful inference or evaluation.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m \
    CMD python -c "from websockets.sync.client import connect; connect('ws://127.0.0.1:6000', open_timeout=3).close()"
CMD ["python", "setup_policy_server.py", "--config_path", "policy/MolmoAct2/deploy.yml", "--host", "0.0.0.0"]
