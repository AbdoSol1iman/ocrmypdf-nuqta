#!/usr/bin/env python3
"""End-to-end OCRmyPDF + Nuqta benchmark for a CUDA machine (no GPU here).

What it does (10+ page PDF expected):

1. Installs nothing; assumes the package is importable and CUDA is visible.
2. Runs ``ocrmypdf`` with the Nuqta plugin once on the whole document
   (``--nuqta-device cuda``), timing the run.
3. Reports total and per-page seconds (from OCRmyPDF's log output), peak
   VRAM (background sampler via torch counters and nvidia-smi), versions.
4. Extracts each page's text layer and scores it: reversed Arabic-Indic
   digit runs, digit runs lost, and search queries run against the layer.
5. Writes everything to a JSON report next to the output PDF.

Usage:
    python scripts/gpu_e2e.py input.pdf output.pdf [--device cuda]
        [--dpi 200] [--pages 1-12] [--query "النبي" ...]

Compile-checked only (py_compile) on the CPU-only machine that produced the
0.1.0 alpha work; treat every number it prints as unverified until a real
T4/RTX run reproduces them.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path

PLUGIN = 'ocrmypdf_nuqta'          # entry point; or pass --plugin path
ARABIC_DIGITS = re.compile(r'[٠-٩]+|\d+')
DEFAULT_QUERIES = ['النبي', 'قال', 'الله']


def collect_versions() -> dict:
    import importlib.metadata as md

    versions = {'python': sys.version.split()[0]}
    for dist in ('ocrmypdf', 'ocrmypdf-nuqta', 'nuqta-ocr', 'torch',
                 'transformers', 'accelerate', 'pymupdf', 'vllm'):
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            versions[dist] = None
    try:
        import torch

        versions['torch.cuda.available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            versions['torch.cuda.device'] = torch.cuda.get_device_name(0)
            versions['torch.cuda'] = torch.version.cuda
    except ImportError:
        versions['torch.cuda.available'] = 'torch missing'
    tesseract = shutil.which('tesseract')
    if tesseract:
        out = subprocess.run([tesseract, '--version'], capture_output=True, text=True)
        match = re.search(r'tesseract\s+(\S+)', out.stdout + out.stderr)
        versions['tesseract'] = match.group(1) if match else 'present'
    else:
        versions['tesseract'] = None
    pdftotext = shutil.which('pdftotext')
    if pdftotext:
        out = subprocess.run(['pdftotext', '-v'], capture_output=True, text=True)
        versions['pdftotext'] = (out.stderr or out.stdout).strip().splitlines()[0]
    return versions


class VramSampler(threading.Thread):
    """Samples peak allocated VRAM and nvidia-smi usage in the background."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop = threading.Event()
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.peak_smi = 0

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                import torch

                if torch.cuda.is_available():
                    self.peak_allocated = max(
                        self.peak_allocated, torch.cuda.max_memory_allocated()
                    )
                    self.peak_reserved = max(
                        self.peak_reserved, torch.cuda.max_memory_reserved()
                    )
            except ImportError:
                pass
            smi = shutil.which('nvidia-smi')
            if smi:
                out = subprocess.run(
                    [smi, '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True,
                )
                for line in out.stdout.splitlines():
                    try:
                        self.peak_smi = max(self.peak_smi, int(line.strip()))
                    except ValueError:
                        pass
            self.stop.wait(self.interval)


def digit_audit(text: str, expected_digits: list[str]) -> dict:
    normalized = unicodedata.normalize('NFKC', text)
    found = {*ARABIC_DIGITS.findall(normalized)}
    reversed_runs = sorted(
        d for d in expected_digits if d not in found and d[::-1] in found
    )
    missing = sorted(d for d in expected_digits if d not in found)
    return {
        'expected': sorted(set(expected_digits)),
        'reversed': reversed_runs,
        'missing': missing,
        'reversed_fraction': round(len(reversed_runs) / max(1, len(set(expected_digits))), 3),
    }


def search_audit(pdf: Path, queries: list[str]) -> tuple[dict, str]:
    results = {}
    import pymupdf

    doc = pymupdf.open(pdf)
    layer = '\n'.join(page.get_text() for page in doc)
    for query in queries:
        hits = sum(len(page.search_for(query)) for page in doc)
        results[query] = {'mupdf_hits': hits, 'in_mupdf_layer': query in layer}
    doc.close()
    pdftotext = shutil.which('pdftotext')
    if pdftotext:
        txt = pdf.with_suffix('.pdftotext.txt')
        subprocess.run(['pdftotext', '-enc', 'UTF-8', str(pdf), str(txt)],
                       capture_output=True, text=True, check=True)
        extracted = unicodedata.normalize('NFKC', txt.read_text(encoding='utf-8'))
        for query in queries:
            results[query]['pdftotext_raw'] = query in extracted
    return results, layer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('input_pdf')
    parser.add_argument('output_pdf')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--pages', default=None,
                       help='OCRmyPDF page selection, e.g. 1-12')
    parser.add_argument('--query', action='append', default=[],
                        help='extra search query; defaults used otherwise')
    parser.add_argument('--dtype', default='auto')
    args = parser.parse_args()
    if args.pages and not re.fullmatch(r'\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*', args.pages):
        parser.error('--pages must look like "1-12" or "1,3,5-8"')

    input_pdf, output_pdf = Path(args.input_pdf), Path(args.output_pdf)
    if not input_pdf.exists():
        parser.error(f'input not found: {input_pdf}')

    import pypdfium2 as pdfium

    source = pdfium.PdfDocument(str(input_pdf))
    page_texts = []
    for page in source:
        page_texts.append(page.get_textpage().get_text_bounded())
    source.close()
    expected_digits = sorted(set(d for text in page_texts
                                 for d in ARABIC_DIGITS.findall(text)))

    vram = VramSampler()
    vram.start()
    cmd = [
        sys.executable, '-m', 'ocrmypdf',
        '--plugin', PLUGIN,
        '--image-dpi', str(args.dpi),
        '-l', 'ara',
        '--output-type', 'pdf',
        '--jobs', '1',
        '--force-ocr',
        '--nuqta-device', args.device,
        '--nuqta-dtype', args.dtype,
        '--nuqta-text-layer', 'as-written',
    ]
    if args.pages:
        cmd += ['--pages', args.pages]
    cmd += [str(input_pdf), str(output_pdf)]

    started = time.perf_counter()
    run = subprocess.run(cmd, capture_output=True, text=True)
    total_seconds = time.perf_counter() - started
    vram.stop.set()
    vram.join(timeout=2)
    if run.returncode != 0:
        report = {
            'ok': False,
            'command': cmd,
            'stderr_tail': run.stderr[-4000:],
            'total_seconds': round(total_seconds, 2),
            'versions': collect_versions(),
        }
        out = output_pdf.with_suffix('.e2e.json')
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                       encoding='utf-8')
        print(f'ocrmypdf failed; report at {out}\n{run.stderr[-1500:]}')
        return 1

    per_page = {}
    for line in run.stderr.splitlines():
        page_match = re.match(r'\s*page (\d+)\b', line)
        time_match = re.search(r'([\d.]+)s', line)
        if page_match and time_match:
            per_page[int(page_match.group(1))] = float(time_match.group(1))

    searches, layer = search_audit(output_pdf, args.query or DEFAULT_QUERIES)
    per_page_digits = {}
    import pypdfium2 as _pdfium

    doc = _pdfium.PdfDocument(str(output_pdf))
    for index, page in enumerate(doc, 1):
        text = page.get_textpage().get_text_bounded()
        per_page_digits[index] = digit_audit(text, expected_digits)
    doc.close()

    report = {
        'ok': True,
        'input': str(input_pdf),
        'output': str(output_pdf),
        'command': cmd,
        'total_seconds': round(total_seconds, 2),
        'per_page_seconds_stderr': per_page,
        'pages': len(page_texts),
        'vram': {
            'peak_allocated_bytes': vram.peak_allocated,
            'peak_reserved_bytes': vram.peak_reserved,
            'peak_nvidia_smi_mib': vram.peak_smi,
        },
        'versions': collect_versions(),
        'digit_audit_whole_document': digit_audit(layer, expected_digits),
        'digit_audit_per_page': per_page_digits,
        'search': searches,
        'notes': [
            'per_page_seconds_stderr depends on OCRmyPDF log lines; treat as approximate',
            'digit audit compares against the SOURCE PDF text, not ground truth',
        ],
    }
    out = output_pdf.with_suffix('.e2e.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    print(json.dumps({k: report[k] for k in (
        'total_seconds', 'pages', 'vram', 'digit_audit_whole_document')},
        ensure_ascii=False, indent=1))
    print(f'full report: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
