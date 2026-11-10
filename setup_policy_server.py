import argparse
import ast
import asyncio
import importlib

import yaml

from client_server.model_server import PolicyServer, PolicyServerConfig


def main(deploy_cfg):
    """Load the configured policy and serve it over WebSocket."""
    protocol = deploy_cfg.get("protocol", "ws")
    if protocol != "ws":
        raise ValueError(f"Unsupported protocol {protocol!r}; only ws is supported")
    policy_name = deploy_cfg["policy_name"]
    module = importlib.import_module(f"moravec_policy.policy.{policy_name}.model")
    model = module.Model(deploy_cfg)
    server = PolicyServer(
        model,
        PolicyServerConfig(
            host=deploy_cfg.get("host", "127.0.0.1"),
            port=int(deploy_cfg["port"]),
            ws_ping_interval_s=deploy_cfg.get("ws_ping_interval_s"),
            ws_ping_timeout_s=deploy_cfg.get("ws_ping_timeout_s"),
        ),
    )
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("\nShutting down websocket policy server...")


def parse_args_and_config():
    """Parse CLI args and YAML config, merge overrides"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "--config-path", dest="config_path", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--protocol", choices=("ws",), help="Policy server protocol")
    parser.add_argument("--host", help="Policy server bind host")
    parser.add_argument("--port", type=int, help="Policy server bind port")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER, help="Override config values")
    args = parser.parse_args()

    # Load base config
    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Parse overrides: --key value pairs

    def _parse_val(s: str):
        # safer than eval; supports numbers/bool/None/list/dict when properly quoted
        try:
            return ast.literal_eval(s)
        except Exception:
            return s
        
    if args.overrides:
        tokens = args.overrides

        # Case A: key=value key=value ...
        if all(("=" in t and not t.startswith("-")) for t in tokens):
            for t in tokens:
                k, v = t.split("=", 1)
                cfg[k] = _parse_val(v)
        else:
            # Case B: --key value --key value ...
            if len(tokens) % 2 != 0:
                raise ValueError(f"--overrides expects key value pairs, got: {tokens}")

            it = iter(tokens)
            for key in it:
                val = next(it)
                cfg[key.lstrip("-")] = _parse_val(val)

    if args.protocol is not None:
        cfg["protocol"] = args.protocol
    else:
        cfg.setdefault("protocol", "ws")
    if args.host is not None:
        cfg["host"] = args.host
    if args.port is not None:
        cfg["port"] = args.port

    def _require_non_empty(key: str):
        if key not in cfg:
            raise ValueError(f"{key} must be specified in config or overrides")
        val = cfg[key]
        if val is None:
            raise ValueError(f"{key} must be non-empty")
        if isinstance(val, str) and not val.strip():
            raise ValueError(f"{key} must be non-empty")

    _require_non_empty("host")
    _require_non_empty("port")
    return cfg

if __name__ == "__main__":
    deploy_cfg = parse_args_and_config()
    main(deploy_cfg)
