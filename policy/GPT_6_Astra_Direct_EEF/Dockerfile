# syntax=docker/dockerfile:1
# CPU-only LLM agent policy server. Pass OPENAI_API_KEY at runtime; never bake it in.
FROM python:3.12.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY policy/GPT_6_Astra_Direct_EEF/requirements.lock /tmp/requirements.lock
RUN python -m pip install --upgrade pip==25.3 \
    && python -m pip install -r /tmp/requirements.lock
COPY . /app
RUN python -m pip install --no-deps -e /app \
    && python -c "from moravec_policy.policy.GPT_6_Astra_Direct_EEF.model import Model"
EXPOSE 6000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "from websockets.sync.client import connect; connect('ws://127.0.0.1:6000', open_timeout=3).close()"
CMD ["python", "setup_policy_server.py", "--config_path", "policy/GPT_6_Astra_Direct_EEF/deploy.yml", "--host", "0.0.0.0"]
