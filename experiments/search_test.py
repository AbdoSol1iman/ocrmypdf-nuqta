#!/usr/bin/env python3
"""Search behaviour: unvocalized queries against as-written and bare layers.

Builds p22 (the heavily vocalized audited page) twice through the stub
engine with the word-level hOCR, once with tashkeel (as-written) and once
without (bare), then searches both PDFs with poppler pdftotext and
MuPDF's page.search_for.  No model is involved: line texts come from the
earlier spike recordings.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
OUT_DIR = HERE / 'out_search'
PAGE = 22
PAGE_IMAGE = f'/home/dev911/ocr-bench/data/book/page_{PAGE}.png'
SPIKE = HERE / 'spike_out' / f'page_{PAGE}.json'

TASHKEEL_RANGES = (
    (0x064B, 0x065F), (0x06D6, 0x06ED),
)


def strip_tashkeel(text: str) -> str:
    return ''.join(
        char for char in text
        if not any(lo <= ord(char) <= hi for lo, hi in TASHKEEL_RANGES)
        and ord(char) not in (0x0670, 0x0610, 0x0611, 0x0612, 0x0613, 0x0614, 0x0615)
    )


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    return strip_tashkeel(' '.join(text.split()))


def build_pdf(variant: str, layer: str, out: Path) -> None:
    env = dict(os.environ)
    env['NUQTA_STUB_VARIANT'] = 'run'
    env['NUQTA_STUB_LINES_JSON'] = str(SPIKE)
    if layer == 'bare':
        env['NUQTA_STUB_BARE'] = '1'
    env['PATH'] = '/tmp/opencode/tessroot/usr/bin:' + env['PATH']
    env['LD_LIBRARY_PATH'] = '/tmp/opencode/tessroot/usr/lib64'
    env['PYTHONPATH'] = str(HERE.parent / 'src') + ':/tmp/opencode/nuqta-ocr/src'
    cmd = [
        sys.executable, '-m', 'ocrmypdf',
        '--plugin', str(HERE / 'stub_engine.py'),
        '--image-dpi', '200', '-l', 'ara', '--output-type', 'pdf',
        '--jobs', '1', '--force-ocr', PAGE_IMAGE, str(out),
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if run.returncode != 0:
        raise RuntimeError(f'ocrmypdf failed for {layer}: {run.stderr[-500:]}')


def pdftotext_search(pdf: Path, query: str) -> tuple[tuple[bool, bool], str]:
    txt = pdf.with_suffix('.pdftotext.txt')
    subprocess.run(['pdftotext', '-enc', 'UTF-8', str(pdf), str(txt)],
                   capture_output=True, text=True, check=True)
    extracted = unicodedata.normalize('NFKC', txt.read_text(encoding='utf-8'))
    raw = query in extracted
    normalized = strip_tashkeel(query) in strip_tashkeel(extracted)
    return (raw, normalized), extracted


def mupdf_search(pdf: Path, query: str) -> tuple[tuple[bool, int], int]:
    import pymupdf

    doc = pymupdf.open(pdf)
    raw = 0
    for page in doc:
        raw += len(page.search_for(query))
    doc.close()
    return (raw > 0, raw), raw


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    data = json.loads(SPIKE.read_text(encoding='utf-8'))
    lines = [item['text'] for item in data['lines']]
    # queries: unvocalized forms of some lines that carry tashkeel
    queries = []
    for text in lines:
        if any(0x064B <= ord(c) <= 0x065F for c in text) and len(text) > 25:
            query = strip_tashkeel(text).strip()
            if len(query) > 15:
                queries.append(query[:40])
        if len(queries) >= 5:
            break
    results = {'tools': {}, 'queries': []}
    layers = {}
    for layer in ('as-written', 'bare'):
        pdf = OUT_DIR / f'p{PAGE}__{layer}.pdf'
        build_pdf('run', layer, pdf)
        layers[layer] = pdf
    for query in queries:
        entry = {'query': query}
        for layer, pdf in layers.items():
            (raw, norm), extracted = pdftotext_search(pdf, query)
            (mraw, mcount), mcount_raw = mupdf_search(pdf, query)
            entry[layer] = {
                'pdftotext_raw_match': raw,
                'pdftotext_normalized_match': norm,
                'mupdf_search_for_hits_raw': mcount,
            }
        results['queries'].append(entry)
    # full-layer fidelity: does the bare layer actually drop harakat, does
    # the as-written layer keep them?
    for layer, pdf in layers.items():
        txt = pdf.with_suffix('.pdftotext.txt')
        subprocess.run(['pdftotext', '-enc', 'UTF-8', str(pdf), str(txt)],
                       capture_output=True, text=True, check=True)
        extracted = unicodedata.normalize('NFKC', txt.read_text(encoding='utf-8'))
        marks = sum(1 for c in extracted if 0x064B <= ord(c) <= 0x065F)
        results['tools'][layer] = {
            'harakat_chars_in_extracted_layer': marks,
            'extracted_chars': len(extracted.strip()),
        }
    (OUT_DIR / 'results.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
