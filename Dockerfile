# Override BASE_IMAGE with a digest-pinned, model-specific CUDA image for learned models.
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}
WORKDIR /app
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
RUN pip install --no-cache-dir --require-hashes -r requirements.lock \
    && pip install --no-cache-dir --no-deps . \
    && useradd --uid 10001 --create-home policy
COPY policies ./policies
COPY robots ./robots
COPY licenses ./licenses
COPY THIRD_PARTY_NOTICES.md ./
USER policy
EXPOSE 8000
CMD ["moravec-policy", "serve", "policies/contract-fixture/policy.json", "--host", "0.0.0.0"]
