from pathlib import Path


def resolve_policy_path(policy_dir, value) -> Path:
    """Resolve a deploy.yml path: absolute paths as given, relative ones from the policy's folder."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (Path(policy_dir) / path).resolve()
