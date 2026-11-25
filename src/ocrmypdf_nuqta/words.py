"""Word-level bounding box splitting and hOCR export for Nuqta OCR.

Nuqta's :func:`nuqta.read_lines` returns one box and one text string per
detected line.  OCRmyPDF's hOCR parser and fpdf2 PDF renderer expect
``ocrx_word`` children inside each ``ocr_line``; placing a whole line in a
single ``ocrx_word`` causes OCRmyPDF's RTL renderer to reverse embedded digits
and Latin runs character-by-character.

This module lives outside core inference and uses only the Python standard
library (no NumPy or OpenCV).  It splits each line box proportionally by
non-combining character width, keeping logical reading order in the hOCR
element list while placing RTL tokens right-to-left and embedded LTR/numeric
runs left-to-right at their visual x-coordinates.
"""
from __future__ import annotations

import html
import unicodedata
from typing import Protocol

Box = tuple[int, int, int, int]

# Inter-word gap as a fraction of one average character unit.
_DEFAULT_GAP_RATIO = 0.45
# Extra clearance on the right of the line-final RTL word: OCRmyPDF's fpdf2
# renderer appends a trailing space to the last word and reverses RTL words,
# which shifts that word's glyphs rightward by ~1 space width.
_DEFAULT_LAST_RTL_EXTRA = 0.35


class _LineLike(Protocol):
    text: str
    bbox: Box


class _PageLike(Protocol):
    width: int
    height: int
    dpi: int
    lines: list[_LineLike]


class _ResultLike(Protocol):
    pages: list[_PageLike]


def _char_weight(ch: str) -> int:
    """Return display-width weight for one character (0 for combining marks)."""
    expanded = unicodedata.normalize('NFKC', ch)
    return sum(
        1
        for c in expanded
        if unicodedata.category(c) != 'Mn' and not unicodedata.combining(c)
    )


def _token_weight(token: str) -> float:
    return max(1.0, float(sum(_char_weight(ch) for ch in token)))


def _classify_token(token: str) -> str:
    """Classify a whitespace-delimited token as 'RTL', 'LTR', or 'NEUTRAL'."""
    has_rtl = False
    has_ltr = False
    for ch in token:
        bidi = unicodedata.bidirectional(ch)
        if bidi in ('R', 'AL'):
            has_rtl = True
        elif bidi in ('L', 'EN', 'AN'):
            has_ltr = True
    if has_rtl and not has_ltr:
        return 'RTL'
    if has_ltr and not has_rtl:
        return 'LTR'
    if has_rtl and has_ltr:
        return 'RTL'
    return 'NEUTRAL'


def _visual_order_indices(tokens: list[str]) -> tuple[list[int], list[str]]:
    """Return token indices in visual left-to-right order for an RTL line.

    Tokens with RTL characters flow right-to-left; contiguous LTR/numeric
    runs (and neutral separators between them) stay left-to-right inside
    their RTL slot so Poppler/PDF viewers do not reverse multi-word Latin or
    number-range runs.
    """
    kinds = [_classify_token(token) for token in tokens]
    for i, kind in enumerate(kinds):
        if kind == 'NEUTRAL':
            prev_kind = kinds[i - 1] if i > 0 else 'RTL'
            next_kind = kinds[i + 1] if i + 1 < len(kinds) else 'RTL'
            kinds[i] = 'LTR' if (prev_kind == 'LTR' and next_kind == 'LTR') else 'RTL'

    groups: list[list[int]] = []
    i = 0
    while i < len(tokens):
        if kinds[i] == 'LTR':
            j = i
            while j < len(tokens) and kinds[j] == 'LTR':
                j += 1
            groups.append(list(range(i, j)))
            i = j
        else:
            groups.append([i])
            i += 1

    visual_indices = [idx for group in reversed(groups) for idx in group]
    return visual_indices, kinds


def split_line_words(
    text: str,
    bbox: Box,
    *,
    gap_ratio: float = _DEFAULT_GAP_RATIO,
    last_rtl_extra: float = _DEFAULT_LAST_RTL_EXTRA,
) -> list[tuple[str, Box]]:
    """Split a line box proportionally into ``(word_text, word_bbox)`` pairs.

    Pairs are returned in **logical reading order** (matching ``text.split()``),
    while each word's bounding box reflects its visual horizontal position on
    an RTL line (RTL words placed right-to-left; embedded LTR/digit runs
    placed left-to-right within their slot).

    All returned boxes have integer coordinates clamped inside *bbox* with
    ``wx0 < wx1`` and ``wy0 < wy1``.
    """
    x0, y0, x1, y1 = (int(v) for v in bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f'invalid line bbox {bbox!r}')

    tokens = text.split()
    if not tokens:
        return []
    if len(tokens) == 1:
        return [(tokens[0], (x0, y0, x1, y1))]

    visual_indices, kinds = _visual_order_indices(tokens)
    weights = [_token_weight(tokens[idx]) for idx in visual_indices]
    gaps = [gap_ratio] * (len(visual_indices) - 1)

    if last_rtl_extra > 0 and kinds[-1] == 'RTL':
        last_pos = visual_indices.index(len(tokens) - 1)
        if last_pos < len(gaps):
            gaps[last_pos] += last_rtl_extra

    total_units = sum(weights) + sum(gaps)
    unit_px = (x1 - x0) / total_units

    boxes: list[Box | None] = [None] * len(tokens)
    cursor = float(x0)
    for pos, (token_idx, weight) in enumerate(zip(visual_indices, weights)):
        span = weight * unit_px
        wx0 = round(cursor)
        wx1 = round(cursor + span)
        wx0 = max(x0, min(wx0, x1 - 1))
        wx1 = min(x1, max(wx1, wx0 + 1))
        boxes[token_idx] = (wx0, y0, wx1, y1)
        if pos < len(gaps):
            cursor += span + gaps[pos] * unit_px

    return [(token, box) for token, box in zip(tokens, boxes) if box is not None]


def _format_bbox(box: Box) -> str:
    x0, y0, x1, y1 = box
    return f'bbox {x0} {y0} {x1} {y1}'


def build_hocr(
    result: _ResultLike,
    *,
    title: str = 'Nuqta OCR',
    lang: str = 'ara',
    dpi: int | None = None,
) -> str:
    """Render a line-level Nuqta result as word-level hOCR for OCRmyPDF."""
    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE html>',
        (
            f'<html xmlns="http://www.w3.org/1999/xhtml" '
            f'xml:lang="{escape_attr(lang)}" lang="{escape_attr(lang)}" dir="rtl">'
        ),
        '<head>',
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />',
        '<meta name="ocr-system" content="nuqta" />',
        '<meta name="ocr-capabilities" content="ocr_page ocr_line ocrx_word" />',
        f'<title>{html.escape(title)}</title>',
        '</head>',
        '<body>',
    ]

    for page_number, page in enumerate(result.pages):
        width, height = int(page.width), int(page.height)
        if width <= 0 or height <= 0:
            raise ValueError(
                'hOCR needs positive page dimensions; use nuqta.read_lines'
            )
        page_dpi = int(dpi) if dpi is not None else int(getattr(page, 'dpi', 0) or 0)
        scan_res = f'; scan_res {page_dpi} {page_dpi}' if page_dpi > 0 else ''
        out.append(
            f'<div class="ocr_page" id="page_{page_number + 1}" '
            f'title="bbox 0 0 {width} {height}; ppageno {page_number}{scan_res}" '
            f'lang="{escape_attr(lang)}" dir="rtl">'
        )

        visible = [line for line in page.lines if line.text.strip()]
        for line_number, line in enumerate(visible, 1):
            words = split_line_words(line.text, line.bbox)
            if not words:
                continue
            line_id = f'line_{page_number + 1}_{line_number}'
            out.append(f'<p class="ocr_par" lang="{escape_attr(lang)}" dir="rtl">')
            out.append(
                f'<span class="ocr_line" id="{line_id}" '
                f'title="{_format_bbox(line.bbox)}" lang="{escape_attr(lang)}" dir="rtl">'
            )
            for word_number, (word_text, word_box) in enumerate(words, 1):
                word_id = f'word_{page_number + 1}_{line_number}_{word_number}'
                out.append(
                    f'<span class="ocrx_word" id="{word_id}" '
                    f'title="{_format_bbox(word_box)}">{html.escape(word_text)}</span>'
                )
            out.append('</span>')
            out.append('</p>')

        out.append('</div>')

    out.extend(['</body>', '</html>', ''])
    return '\n'.join(out)


def escape_attr(value: str) -> str:
    return html.escape(value, quote=True)
