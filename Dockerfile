# CPU contract demo only. GPU models use their separate upstream CUDA environments.
FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
RUN pip install --no-cache-dir --require-hashes -r requirements.lock \
    && pip install --no-cache-dir --no-deps . \
    && useradd --uid 10001 --create-home policy
COPY examples/mock ./examples/mock
USER policy
EXPOSE 8000
CMD ["moravec-policy", "serve", "examples/mock/policy.json", "--host", "0.0.0.0"]
