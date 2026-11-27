#!/usr/bin/env python3
"""Score the pdftotext/mupdf extractions of the synthetic cases."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
CASES_DIR = HERE / 'cases'
OUT_DIR = HERE / 'out_cases'
VARIANTS = ('line', 'token', 'run')


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    return ' '.join(text.split())


def check_one(check: dict, extracted: str) -> bool:
    kind = check['kind']
    if kind == 'contains':
        return check['value'] in extracted
    if kind == 'not_contains':
        return check['value'] not in extracted
    if kind == 'sequence':
        pos = -1
        for value in check['values']:
            found = extracted.find(value, pos + 1)
            if found == -1:
                return False
            pos = found
        return True
    raise ValueError(f'unknown check {kind}')


def spaces_report(extracted: str) -> dict:
    doubled = extracted.count('  ')
    return {'doubled_spaces': doubled}


def main() -> None:
    manifest = json.loads((CASES_DIR / 'manifest.json').read_text(encoding='utf-8'))
    results = {'tools': {}, 'detail': []}
    tools = ('pdftotext', 'mupdf')
    for variant in VARIANTS:
        stats = {tool: {'cases': 0, 'passed': 0, 'order_passed': 0}
                 for tool in tools}
        for case in manifest:
            name = case['name']
            entry = {'case': name, 'variant': variant, 'checks': {}}
            for tool in tools:
                txt = OUT_DIR / f'{name}__{variant}.{tool}.txt'
                if not txt.exists():
                    entry['checks'][tool] = [{'kind': 'missing', 'ok': False}]
                    continue
                extracted = normalize(txt.read_text(encoding='utf-8'))
                entry.setdefault('extracted', {})[tool] = extracted
                entry['spaces'] = spaces_report(extracted)
                checks = [
                    {'kind': c['kind'], 'value': c.get('value', c.get('values')),
                     'ok': check_one(c, extracted), 'class': 'direction'}
                    for c in case['checks']
                ]
                order = case.get('order_check')
                if order:
                    checks.append({'kind': 'sequence', 'class': 'order',
                                   'value': order['values'],
                                   'ok': check_one(order, extracted)})
                entry['checks'][tool] = checks
                stats[tool]['cases'] += 1
                stats[tool]['passed'] += int(all(c['ok'] for c in checks))
                stats[tool]['order_passed'] += int(
                    all(c['ok'] for c in checks if c['class'] == 'order')
                )
            results['detail'].append(entry)
        for tool in tools:
            cases = stats[tool]['cases'] or 1
            stats[tool]['fraction'] = round(stats[tool]['passed'] / cases, 3)
            stats[tool]['order_fraction'] = round(stats[tool]['order_passed'] / cases, 3)
        results['tools'][variant] = stats
    # fallback rate per variant
    for variant in VARIANTS:
        aligned = fallback = 0
        for case in manifest:
            stats_path = OUT_DIR / f"{case['name']}__{variant}.stats.json"
            if stats_path.exists():
                data = json.loads(stats_path.read_text())
                aligned += data.get('aligned', 0)
                fallback += data.get('fallback', 0)
        results['tools'][variant]['alignment_fallback'] = fallback
        results['tools'][variant]['alignment_ok'] = aligned
    out = HERE / 'out_cases' / 'results.json'
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    for variant in VARIANTS:
        print(variant, json.dumps(results['tools'][variant], ensure_ascii=False))


if __name__ == '__main__':
    main()
