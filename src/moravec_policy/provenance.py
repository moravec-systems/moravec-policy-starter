"""Artifact hashes exclude neither weights nor normalization/config files."""

import hashlib
from pathlib import Path


def tree_hash(root: Path) -> str:
    if not root.is_dir():
        raise ValueError("checkpoint must be a local directory")
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError("checkpoint directory is empty")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        file_hash = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_hash.update(chunk)
        digest.update(file_hash.digest())
    return digest.hexdigest()
