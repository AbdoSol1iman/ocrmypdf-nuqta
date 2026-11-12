#!/usr/bin/env python3
"""Run the three audited book pages (p21/p22/p27) through the stub engine.

Line boxes and line texts come from the earlier spike run
(experiments/spike_out/page_N.json, recorded on CPU with the real model),
so no model is needed here.  Each page is run through OCRmyPDF 17.13 in
each hOCR variant and the extraction is scored on the digit runs it
contains.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT_DIR = HERE / 'out_real'
VARIANTS = ('line', 'token', 'run')
PAGES = (21, 22, 27)
PAGE_IMAGE = '/home/dev911/ocr-bench/data/book/page_{}.png'
SPIKE = HERE / 'spike_out'
ARABIC_DIGITS = re.compile(r'[٠-٩]{2,}|\d{2,}')


def env_for(variant: str, page: int, stats_path: Path) -> dict:
    env = dict(os.environ)
    env['NUQTA_STUB_VARIANT'] = variant
    env['NUQTA_STUB_LINES_JSON'] = str(SPIKE / f'page_{page}.json')
    env['NUQTA_STUB_STATS'] = str(stats_path)
    env['PATH'] = '/tmp/opencode/tessroot/usr/bin:' + env['PATH']
    env['LD_LIBRARY_PATH'] = '/tmp/opencode/tessroot/usr/lib64'
    env['PYTHONPATH'] = str(HERE.parent / 'src') + ':/tmp/opencode/nuqta-ocr/src'
    return env


def digit_runs(text: str) -> list[str]:
    return [m for m in ARABIC_DIGITS.findall(unicodedata.normalize('NFKC', text))]


def main() -> int:
    if not SPIKE.exists():
        print('spike_out recordings not found; cannot run real pages', file=sys.stderr)
        return 2
    OUT_DIR.mkdir(exist_ok=True)
    failures = []
    summary = []
    for page in PAGES:
        data = json.loads((SPIKE / f'page_{page}.json').read_text(encoding='utf-8'))
        all_text = '\n'.join(item['text'] for item in data['lines'])
        expected_digits = sorted(set(digit_runs(all_text)))
        for variant in VARIANTS:
            stats_path = OUT_DIR / f'p{page}__{variant}.stats.json'
            pdf = OUT_DIR / f'p{page}__{variant}.pdf'
            cmd = [
                sys.executable, '-m', 'ocrmypdf',
                '--plugin', str(HERE / 'stub_engine.py'),
                '--image-dpi', '200', '-l', 'ara',
                '--output-type', 'pdf', '--jobs', '1', '--force-ocr',
                PAGE_IMAGE.format(page), str(pdf),
            ]
            run = subprocess.run(cmd, capture_output=True, text=True,
                                 env=env_for(variant, page, stats_path))
            if run.returncode != 0:
                failures.append((page, variant, 'ocrmypdf', run.stderr[-300:]))
                continue
            entry = {'page': page, 'variant': variant, 'expected_digit_runs': expected_digits}
            for tool, extractor in (
                ('pdftotext', ['pdftotext', '-enc', 'UTF-8']),
                ('mupdf', None),
            ):
                txt = OUT_DIR / f'p{page}__{variant}.{tool}.txt'
                if extractor:
                    pop = subprocess.run(extractor + [str(pdf), str(txt)],
                                         capture_output=True, text=True)
                    if pop.returncode != 0:
                        failures.append((page, variant, tool, pop.stderr[-300:]))
                        continue
                else:
                    import pymupdf

                    doc = pymupdf.open(pdf)
                    txt.write_text('\n'.join(p.get_text() for p in doc), encoding='utf-8')
                    doc.close()
                extracted = unicodedata.normalize('NFKC', txt.read_text(encoding='utf-8'))
                found = digit_runs(extracted)
                reversed_runs = sorted(
                    run for run in expected_digits
                    if run not in found and run[::-1] in found
                )
                missing_runs = sorted(run for run in expected_digits if run not in found)
                entry[tool] = {
                    'expected': expected_digits,
                    'reversed': reversed_runs,
                    'missing': missing_runs,
                    'reversed_fraction': round(len(reversed_runs) / max(1, len(expected_digits)), 3),
                }
            summary.append(entry)
    (OUT_DIR / 'results.json').write_text(
        json.dumps({'detail': summary, 'failures': [f[2] for f in failures]},
                   ensure_ascii=False, indent=1),
        encoding='utf-8',
    )
    for f in failures:
        print('FAIL', f)
    for entry in summary:
        pd = entry['pdftotext']
        mu = entry['mupdf']
        print(f"p{entry['page']} [{entry['variant']:5}] "
              f"pdftotext reversed {pd['reversed_fraction']:.0%} "
              f"({pd['reversed'] or '-'}) mupdf reversed {mu['reversed_fraction']:.0%} "
              f"({mu['reversed'] or '-'})")
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
