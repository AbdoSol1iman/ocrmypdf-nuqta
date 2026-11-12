#!/usr/bin/env python3
"""Run the synthetic cases through OCRmyPDF 17.13 for each hOCR variant."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CASES_DIR = HERE / 'cases'
OUT_DIR = HERE / 'out_cases'
VARIANTS = ('line', 'token', 'run')


def env_for(variant: str, case_name: str, stats_path: Path) -> dict:
    env = dict(os.environ)
    env['NUQTA_STUB_VARIANT'] = variant
    env['NUQTA_STUB_TEXT'] = str(CASES_DIR / f'{case_name}.txt')
    env['NUQTA_STUB_STATS'] = str(stats_path)
    # local tesseract satisfies OCRmyPDF's binary check (no engine use)
    env['PATH'] = '/tmp/opencode/tessroot/usr/bin:' + env['PATH']
    env['LD_LIBRARY_PATH'] = '/tmp/opencode/tessroot/usr/lib64'
    env['PYTHONPATH'] = str(HERE.parent / 'src') + ':/tmp/opencode/nuqta-ocr/src'
    return env


def write_case_texts() -> None:
    manifest = json.loads((CASES_DIR / 'manifest.json').read_text(encoding='utf-8'))
    for case in manifest:
        (CASES_DIR / f"{case['name']}.txt").write_text(case['text'], encoding='utf-8')


def main() -> int:
    if shutil.which('pdftotext') is None:
        print('pdftotext not found', file=sys.stderr)
        return 2
    write_case_texts()
    manifest = json.loads((CASES_DIR / 'manifest.json').read_text(encoding='utf-8'))
    OUT_DIR.mkdir(exist_ok=True)
    failures = []
    for variant in VARIANTS:
        for case in manifest:
            name = case['name']
            png = CASES_DIR / f'{name}.png'
            stats_path = OUT_DIR / f'{name}__{variant}.stats.json'
            pdf = OUT_DIR / f'{name}__{variant}.pdf'
            cmd = [
                sys.executable, '-m', 'ocrmypdf',
                '--plugin', str(HERE / 'stub_engine.py'),
                '--image-dpi', '200', '-l', 'ara',
                '--output-type', 'pdf', '--jobs', '1',
                '--force-ocr', str(png), str(pdf),
            ]
            run = subprocess.run(cmd, capture_output=True, text=True,
                                 env=env_for(variant, name, stats_path))
            if run.returncode != 0:
                failures.append((name, variant, 'ocrmypdf failed', run.stderr[-400:]))
                continue
            txt = OUT_DIR / f'{name}__{variant}.pdftotext.txt'
            pop = subprocess.run(['pdftotext', '-enc', 'UTF-8', str(pdf), str(txt)],
                                 capture_output=True, text=True)
            if pop.returncode != 0:
                failures.append((name, variant, 'pdftotext failed', pop.stderr[-400:]))
            mtxt = OUT_DIR / f'{name}__{variant}.mupdf.txt'
            import pymupdf

            doc = pymupdf.open(pdf)
            mtxt.write_text('\n'.join(page.get_text() for page in doc), encoding='utf-8')
            doc.close()
    if failures:
        for name, variant, what, detail in failures:
            print(f'FAIL {name} [{variant}] {what}: {detail}')
        return 1
    print('all runs completed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
