ARG BASE_IMAGE=moravec-policy-base:local
FROM ${BASE_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY policies/GR00T_N17/libero/runtime-requirements.txt /tmp/runtime-requirements.txt
RUN pip install --no-cache-dir -r /tmp/runtime-requirements.txt
RUN git init /opt/gr00t-source \
    && cd /opt/gr00t-source \
    && git remote add origin https://github.com/XPolicyLab/XPolicyLab.git \
    && git sparse-checkout init --cone \
    && git sparse-checkout set policy/GR00T_N17/gr00t_n17 \
    && git fetch --depth 1 --filter=blob:none origin bb9a0b5f5136a74503b679af830bfd0a3a837d5c \
    && git checkout --detach FETCH_HEAD \
    && pip install --no-deps ./policy/GR00T_N17/gr00t_n17
COPY policies/GR00T_N17/libero /app/policies/GR00T_N17/libero
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
USER 10001
CMD ["moravec-policy", "serve", "policies/GR00T_N17/libero/policy.json", "--host", "0.0.0.0"]
