"""Bound artifact basenames without losing recipe or point identity."""
from __future__ import annotations

import argparse
import hashlib
import os


def compact(value: str, budget: int) -> str:
    if not value or any(char in value for char in '/\\\r\n\x00'):
        raise ValueError('Expected a nonempty filename component')
    encoded = value.encode('utf-8')
    if len(encoded) <= budget:
        return value
    suffix = '-' + hashlib.sha256(encoded).hexdigest()[:20]
    if budget < len(suffix):
        raise ValueError('Result suffix leaves no filename identity budget')
    return encoded[:budget - len(suffix)].decode('utf-8', errors='ignore') + suffix


def result_stem(base: str, fingerprint: str = '') -> str:
    # Leave room for SRT point suffixes, power_validation_, .json and atomic .tmp.
    # The hash includes the entire fingerprint; full metadata stays in the JSON.
    value = base + (f'_recipe-{fingerprint}' if fingerprint else '')
    return compact(value, 120)


def point_filename(stem: str, config: str, conc: str, gpus: str,
                   ctx: str = '', gen: str = '') -> str:
    numbers = [conc, gpus, *([ctx, gen] if ctx or gen else [])]
    if any(not number.isascii() or not number.isdecimal() for number in numbers):
        raise ValueError('Expected numeric point identity')
    tail = f'_conc{conc}_gpus_{gpus}' + (f'_ctx_{ctx}_gen_{gen}' if ctx else '')
    # Keep the numeric tail readable by existing workflow GPU-count extraction.
    budget = 255 - len(('power_validation_' + stem + '_' + tail + '.json.tmp').encode())
    return f'{stem}_{compact(config, budget)}{tail}.json'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point', nargs=6, metavar=('STEM', 'CONFIG', 'CONC', 'GPUS', 'CTX', 'GEN'))
    args = parser.parse_args()
    print(point_filename(*args.point) if args.point else result_stem(
        os.environ['RESULT_FILENAME_BASE'], os.environ.get('RECIPE_FINGERPRINT', '')))


if __name__ == '__main__':
    main()
