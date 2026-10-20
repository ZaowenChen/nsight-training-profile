#!/usr/bin/env python3
"""Install the bundled agent skill without replacing an existing installation."""
import argparse
import json
import os
from pathlib import Path
import shutil

NAME = 'nsight-training-profile'
SOURCE = Path(__file__).resolve().parents[1] / 'skills' / NAME


def install_skill(skills_dir):
    destination = Path(skills_dir).expanduser().resolve() / NAME
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Existing installation retained: {destination}. Back it up first or select a different --skills-dir.')
    if not (SOURCE / 'SKILL.md').is_file():
        raise FileNotFoundError(f'Skill source missing: {SOURCE}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'local-install.json'))
    return destination


def main():
    codex_root = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skills-dir', type=Path, default=codex_root / 'skills')
    args = parser.parse_args()
    try:
        target = install_skill(args.skills_dir)
    except OSError as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps({'installed': str(target), 'skill': NAME}))


if __name__ == '__main__':
    main()
