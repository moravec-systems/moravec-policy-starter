# syntax=docker/dockerfile:1
FROM python:3.12.12-slim-bookworm
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -e . \
    && mkdir -p /env_cfg/robot \
    && cp utils/robot/_robot_info.json /env_cfg/robot/_robot_info.json \
    && printf 'config:\n  robot: yam\n' > /env_cfg/yam.yml
EXPOSE 6000
CMD ["python", "setup_policy_server.py", "--config_path", "policy/demo_policy/deploy.yml", "--overrides", "host=0.0.0.0", "port=6000", "env_cfg_type=yam", "action_type=joint"]
