"""OCRmyPDF engine plugin stub for measurement runs (no model, no download).

Reads the line texts and the hOCR variant to emit from environment variables:

- ``NUQTA_STUB_VARIANT``: ``line`` (one ocrx_word per line, the pre-0.1
  shape), ``token`` (one word per token, DP-aligned) or ``run`` (one word per
  script run).
- ``NUQTA_STUB_TEXT``: file with the line text for single-line synthetic
  pages; the page is segmented with nuqta.layout and the text is attached to
  the first detected line box.
- ``NUQTA_STUB_LINES_JSON``: JSON ``{"lines": [{"text": ..., "box": [...]}, ...]}``
  for real pages whose line boxes were recorded earlier (spike_out).
- ``NUQTA_STUB_BARE``: set to strip tashkeel from the recorded line texts
  (the ``bare`` text-layer variant).
- ``NUQTA_STUB_STATS``: path to write the word-alignment stats JSON to.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence


def _strip_tashkeel(text: str) -> str:
    """Remove harakat/annotation marks; used for the 'bare' text layer."""
    return ''.join(
        char for char in text
        if not (0x064B <= ord(char) <= 0x065F or 0x06D6 <= ord(char) <= 0x06ED
                or ord(char) in (0x0670, 0x0610, 0x0611, 0x0612, 0x0613, 0x0614, 0x0615))
    )


class StubEngine(OcrEngine):
    @staticmethod
    def version() -> str:
        return 'stub-nuqta-words'

    @staticmethod
    def creator_tag(options) -> str:
        return 'Nuqta word-level prototype'

    def __str__(self) -> str:
        return 'stub-nuqta-words'

    @staticmethod
    def languages(options) -> set[str]:
        return {'ara', 'ar', 'eng'}

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options) -> float:
        return 0.0

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options) -> None:
        from nuqta import (
            Line,
            LinePage,
            LineResult,
            segment_page,
            to_hocr as line_to_hocr,
            validate_hocr,
        )
        from ocrmypdf_nuqta import words as words_mod
        from PIL import Image

        variant = os.environ.get('NUQTA_STUB_VARIANT', 'run')
        with Image.open(input_file) as image:
            dpi_pair = image.info.get('dpi')
            dpi = round(float(dpi_pair[0])) if dpi_pair else 200
            width, height = image.size

        lines_json = os.environ.get('NUQTA_STUB_LINES_JSON')
        if lines_json:
            data = json.loads(Path(lines_json).read_text(encoding='utf-8'))
            if os.environ.get('NUQTA_STUB_BARE'):
                data = {
                    **data,
                    'lines': [
                        {**item, 'text': _strip_tashkeel(item['text'])}
                        for item in data['lines']
                    ],
                }
            lines = [
                Line(item['text'], tuple(int(v) for v in item.get('box', item.get('box_original'))))
                for item in data['lines']
                if item.get('text', '').strip()
            ]
        else:
            text = ''
            text_file = os.environ.get('NUQTA_STUB_TEXT')
            if text_file:
                text = ' '.join(Path(text_file).read_text(encoding='utf-8').split())
            segmented = segment_page(Image.open(input_file))
            boxes = segmented.boxes or [(20, 10, width - 20, height - 10)]
            lines = [Line(text, boxes[0])] if text else []

        result = LineResult(
            pages=[LinePage(
                text='\n'.join(line.text for line in lines),
                lines=lines, width=width, height=height, dpi=dpi,
            )],
            mode='as-written',
        )
        document = None
        stats: dict = {}
        if variant in ('token', 'run') and lines:
            try:
                document = words_mod.to_word_hocr(
                    result, [Image.open(input_file)], lang='ara', dpi=dpi,
                    split_runs=(variant == 'run'), stats=stats,
                )
            except Exception:
                sys.stderr.write(f'stub: word hOCR failed: {sys.exc_info()[1]}\n')
                document = None
        if document is None:
            document = line_to_hocr(result, lang='ara', dpi=dpi, title=Path(input_file).name)
        validate_hocr(document)
        if stats and os.environ.get('NUQTA_STUB_STATS'):
            Path(os.environ['NUQTA_STUB_STATS']).write_text(json.dumps(stats))
        Path(output_hocr).write_text(document, encoding='utf-8')
        Path(output_text).write_text(result.pages[0].text + '\n', encoding='utf-8')

    @staticmethod
    def generate_pdf(input_file: Path, output_pdf: Path, output_text: Path, options) -> None:
        raise NotImplementedError('stub supports the hOCR path only')


@hookimpl
def get_ocr_engine(options):
    return StubEngine()
