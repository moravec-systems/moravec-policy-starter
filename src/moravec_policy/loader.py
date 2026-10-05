"""Load trusted policy code shipped in the submitted container, never from wire requests."""

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

from .lifecycle import Policy


def load_entrypoint(entrypoint, manifest_path):
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator or not module_name or not attribute.isidentifier():
        raise ValueError("entrypoint must be module:callable or relative.py:callable")
    if module_name.endswith(".py"):
        path = (Path(manifest_path).resolve().parent / module_name).resolve()
        name = "moravec_submission_" + hashlib.sha256(str(path).encode()).hexdigest()
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ValueError("cannot load policy module")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(name, None)
                raise
        module = sys.modules[name]
    else:
        module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError("policy entrypoint must be callable")
    return factory


def load_policy(manifest, path):
    policy = load_entrypoint(manifest.entrypoint, path)(manifest, Path(path))
    if not isinstance(policy, Policy):
        raise TypeError("default server policies must implement the Policy lifecycle")
    return policy
