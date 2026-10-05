"""Fetch a pinned model implementation without installing its dependencies.

Run with Python 3.12+: python scripts/fetch_policy.py GR00T_N17
"""
import argparse
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

REVISION = '45ae74c1a427206f62d55f08f48f00295d2dc1a3'
URL = f'https://codeload.github.com/XPolicyLab/XPolicyLab/tar.gz/{REVISION}'
SOURCES = {
    'GR00T_N17': 'policy/GR00T_N17/gr00t_n17',
    'Pi_05': 'policy/Pi_05/openpi',
    'OpenWAM': 'policy/OpenWAM/OpenWAM',
    'GPT_6_Astra_Direct_EEF': 'policy/GPT_6_Astra_Direct_EEF/inspect',
}


def adapt_imports(root: Path):
    """Retarget integration imports while retaining upstream licenses/notices."""
    for directory, folders, files in os.walk(root):
        folders[:] = [name for name in folders
                      if not name.startswith('.') and name not in {'venv', '__pycache__'}]
        for name in files:
            if not name.endswith('.py'):
                continue
            path = Path(directory) / name
            original = path.read_text(encoding='utf-8')
            updated = original.replace('from XPolicyLab.', 'from moravec_policy.').replace(
                'import XPolicyLab.', 'import moravec_policy.')
            if updated != original:
                path.write_text('# Moravec integration: package imports adapted.\n' + updated,
                                encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('policy', choices=SOURCES)
    parser.add_argument('--adapt-existing', action='store_true',
                        help='Update package imports in an existing runtime without replacing it')
    parser.add_argument('--destination', type=Path, help='Optional external destination for verification')
    args = parser.parse_args()
    source = SOURCES[args.policy]
    destination = args.destination or Path(__file__).resolve().parents[1] / source
    if args.adapt_existing:
        if not destination.is_dir():
            parser.error(f'{destination} is not an existing runtime directory')
        adapt_imports(destination)
        print(f'Updated integration imports in {destination}; existing source retained.')
        return
    if destination.exists():
        parser.error(f'{destination} already exists; refusing to overwrite local code')
    if not hasattr(tarfile, 'data_filter'):
        parser.error('Use Python 3.12 or newer for safe archive extraction')
    cache = Path.home() / '.cache/moravec-policy-starter'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f'{REVISION}.tar.gz'
    if not archive.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            download = Path(temporary) / 'source.tar.gz'
            print(f'Downloading pinned source {REVISION}', flush=True)
            with urllib.request.urlopen(URL, timeout=120) as response, download.open('wb') as output:
                shutil.copyfileobj(response, output)
            # Confirm archive integrity before caching it.
            with tarfile.open(download, 'r:gz') as bundle:
                bundle.getmembers()
            download.replace(archive)
    prefix = f'XPolicyLab-{REVISION}/{source}/'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        extracted = Path(temporary)
        with tarfile.open(archive, 'r:gz') as bundle:
            members = [member for member in bundle.getmembers() if member.name.startswith(prefix)]
            if not members:
                raise RuntimeError(f'Missing source directory in pinned archive: {source}')
            bundle.extractall(extracted, members=members, filter='data')
        root = extracted / f'XPolicyLab-{REVISION}' / source
        adapt_imports(root)
        root.rename(destination)
    print(f'Fetched {args.policy} at {REVISION} into {destination}')
    print('Source only. Follow the policy README to install its runtime; weights are separate.')


if __name__ == '__main__':
    main()
