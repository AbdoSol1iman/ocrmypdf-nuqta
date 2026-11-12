"""Word-level boxes for line-level Nuqta readings.

Nuqta (since PR #3) returns one text and one box per line.  OCRmyPDF's fpdf2
renderer decides per ``ocrx_word`` whether to pre-reverse the text for bidi
extraction (``ocrmypdf.fpdf_renderer.renderer._is_rtl_text``), so a whole line
kept as a single word garbles Arabic-Indic digits and Latin runs inside an
Arabic line.  The fix is word-level boxes: one ``ocrx_word`` per token, so the
renderer's RTL decision is made per script run.

This module turns a :class:`nuqta.LineResult` plus the page image into word
hOCR.  It is deliberately standalone (only NumPy/OpenCV/PIL and nuqta types)
and named so it could move upstream into ``nuqta.layout`` later:

1. Binarize the page once (Otsu), then for each line box split the crop at
   vertical-projection gaps into *ink segments* with pixel x-ranges.
2. Split the line text into *tokens* (whitespace) and each token into *script
   runs* (Arabic, Arabic-Indic digits, Latin, ASCII digits, other).
3. Align tokens to segments with dynamic programming over expected glyph
   widths (tashkeel weight 0): pairings, a token spanning several segments
   (merge), and several tokens sharing one segment (split) are all allowed,
   instead of requiring an exact 1:1 count match.
4. If a line's aligned boxes come out touching (dense vocalized print), the
   line is re-segmented at tighter thresholds and the variant with the fewest
   touching word pairs wins; the DP can merge the extra fragments.
5. If segmentation yields nothing usable, fall back to proportional
   distribution across the line box and record the fallback.

Base direction is detected per line from the first strong character; in an
RTL line, visual (left-to-right) segment order is reversed to match the
logical order of the tokens Nuqta returned.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from html import escape
from xml.etree import ElementTree as ET

Box = tuple[int, int, int, int]

# Relative advance widths, calibrated by eye against the Arabic font OCRmyPDF
# embeds; only ratios matter because they are rescaled per line.
_TASHKEEL_WEIGHT = 0.0          # marks carry no advance
_TATWEEL_WEIGHT = 0.30
_ARABIC_LETTER_WEIGHT = 0.60    # connected script; narrower than isolated glyphs
_DIGIT_WEIGHT = 0.70            # Arabic-Indic digits in Arabic fonts
_LATIN_WEIGHT = 0.50
_OTHER_WEIGHT = 0.35


def _is_tashkeel(char: str) -> bool:
    code = ord(char)
    return (
        0x064B <= code <= 0x065F      # harakat, shadda, sukun, ...
        or code == 0x0670             # superscript alef
        or 0x06D6 <= code <= 0x06ED   # Quranic annotation marks
        or code in (0x0610, 0x0611, 0x0612, 0x0613, 0x0614, 0x0615)
    )


def char_weight(char: str) -> float:
    """Relative advance of one character; tashkeel weigh nothing."""
    code = ord(char)
    if _is_tashkeel(char):
        return _TASHKEEL_WEIGHT
    if code == 0x0640:                # tatweel
        return _TATWEEL_WEIGHT
    category = unicodedata.category(char)
    if category.startswith('M'):      # any other combining mark
        return _TASHKEEL_WEIGHT
    if 0x0660 <= code <= 0x0669 or 0x06F0 <= code <= 0x06F9:
        return _DIGIT_WEIGHT
    if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0xFB50 <= code <= 0xFEFF:
        return _ARABIC_LETTER_WEIGHT
    if char.isascii() and (char.isalpha() or char.isdigit()):
        return _LATIN_WEIGHT
    return _OTHER_WEIGHT


def expected_width(text: str) -> float:
    return sum(char_weight(char) for char in text)


def tokens_of(text: str) -> list[str]:
    """Whitespace-separated tokens of a line, preserving inner content."""
    return text.split()


_ARABIC_RANGES = (
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
)
_DIGIT_RANGES = ((0x0660, 0x0669), (0x06F0, 0x06F9))


def _in_ranges(code: int, ranges) -> bool:
    return any(lo <= code <= hi for lo, hi in ranges)


def _run_class(char: str) -> str:
    code = ord(char)
    if _in_ranges(code, _DIGIT_RANGES):
        return 'arabic_digit'
    if _in_ranges(code, _ARABIC_RANGES):
        return 'arabic'
    if char.isascii() and char.isalpha():
        return 'latin'
    if char.isascii() and char.isdigit():
        return 'ascii_digit'
    if unicodedata.category(char).startswith('M'):
        return 'mark'                 # combines with the previous run
    return 'other'                    # punctuation and neutral characters


def script_runs(token: str) -> list[tuple[str, str]]:
    """Split a token into (class, run) pairs; neutrals join the previous run."""
    runs: list[tuple[str, str]] = []
    for char in token:
        cls = _run_class(char)
        if cls in ('other', 'mark') and runs:
            runs[-1] = (runs[-1][0], runs[-1][1] + char)
            continue
        if runs and runs[-1][0] == cls:
            runs[-1] = (cls, runs[-1][1] + char)
        else:
            runs.append((cls, char))
    return runs


def base_direction(text: str) -> str:
    """First strong character decides: 'rtl' (default) or 'ltr'.

    Arabic-Indic digits (bidi AN) are deliberately not strong: on their own
    they stay with the Arabic paragraph direction.
    """
    for char in text:
        bidi = unicodedata.bidirectional(char)
        if bidi in ('R', 'AL'):
            return 'rtl'
        if bidi == 'L':
            return 'ltr'
    return 'rtl'


@dataclass(frozen=True)
class Segment:
    """One ink run found in a line crop: an x-range on the original image."""

    x0: int
    x1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0


@dataclass
class WordBox:
    """One ``ocrx_word`` to emit: text and its box on the page image."""

    text: str
    bbox: Box
    classes: tuple[str, ...] = ()


@dataclass
class WordLine:
    """Word-level reading of one line."""

    text: str
    bbox: Box
    words: list[WordBox]
    aligned: bool                       # False when proportional fallback ran
    segments: int                       # ink segments found (0 on fallback)


@dataclass
class WordPage:
    lines: list[WordLine]
    width: int
    height: int
    dpi: int = 200


@dataclass
class WordResult:
    pages: list[WordPage]
    mode: str = 'as-written'
    alignment_stats: dict = field(default_factory=dict)


def line_segments(page_binary, bbox: Box, *, min_gap_ratio: float = 0.12, min_segment_px: int = 3):
    """Split one line's ink into x-ranges at vertical-projection gaps.

    ``min_gap_ratio`` is relative to the line height: inter-word spaces in
    Arabic print are roughly 0.15-0.5 of the line height, while gaps between
    letters (and under diacritics) are smaller.
    """
    import cv2
    import numpy as np

    x0, y0, x1, y1 = bbox
    crop = page_binary[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    columns = crop.sum(axis=0, dtype=np.int64)
    ink = columns > 0
    height = y1 - y0
    min_gap = max(2, int(height * min_gap_ratio))

    segments: list[Segment] = []
    start = None
    last_ink = None
    for index, value in enumerate(ink):
        if value:
            if start is None:
                start = index
            last_ink = index
        elif start is not None and index - last_ink >= min_gap:
            segments.append(Segment(x0 + start, x0 + last_ink + 1))
            start = None
    if start is not None:
        segments.append(Segment(x0 + start, x0 + last_ink + 1))
    # Drop specks narrower than a letter, unless they are all we have.
    usable = [seg for seg in segments if seg.width >= min_segment_px]
    return usable if usable else segments


def _mismatch(expected: float, width: float) -> float:
    """Scale-free cost of a width mismatch, in [0, 1)."""
    total = expected + width
    return 0.0 if total <= 0 else abs(expected - width) / total


def align_tokens(
    token_texts: list[str],
    segments: list[Segment],
    *,
    direction: str = 'rtl',
) -> tuple[list[Box | None], float]:
    """Assign a box to every token with dynamic programming.

    Returns per-token boxes in *visual* order of the segments given, plus the
    total DP cost.  ``None`` marks a token the DP could not place (only when
    there are no segments at all).
    """
    if not segments:
        return [None] * len(token_texts), 0.0
    if not token_texts:
        return [], 0.0
    if len(token_texts) == 1:
        return [(min(s.x0 for s in segments), 0, max(s.x1 for s in segments), 0)], _mismatch(
            expected_width(token_texts[0]), sum(s.width for s in segments)
        )

    n, m = len(token_texts), len(segments)
    widths = [seg.width for seg in segments]
    expected = [expected_width(token) for token in token_texts]
    total_expected = sum(expected)
    # Rescale glyph weights so they sum to the observed ink width: only the
    # *relative* widths are trustworthy.
    observed = sum(widths)
    scale = observed / total_expected if total_expected > 0 else 1.0
    exp = [value * scale for value in expected]

    # dp[i][j]: best cost placing tokens[i:] against segments[j:].
    NEG = float('inf')
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    choice: list[list[tuple]] = [[() for _ in range(m + 1)] for _ in range(n + 1)]
    dp[n][m] = 0.0
    # Trailing segments nobody claimed: moderate penalty, they are probably
    # noise but a skip is never free.
    for j in range(m - 1, -1, -1):
        dp[n][j] = dp[n][j + 1] + 0.15 * widths[j] / max(1, observed)
        choice[n][j] = ('skip', j + 1)

    # Non-pairing operations carry a small fixed cost: tokens and ink segments
    # correspond one-to-one most of the time, so a merge/split must be clearly
    # better than a pair to win.
    OP_COST = {'pair': 0.0, 'merge': 0.05, 'split': 0.05, 'skip': 0.0}

    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best, move = NEG, ()
            # 1:1 pairing
            cost = dp[i + 1][j + 1] + _mismatch(exp[i], widths[j]) + OP_COST['pair']
            if cost < best:
                best, move = cost, ('pair', j + 1)
            # one token spanning several segments (merged)
            acc_w = 0.0
            for k in range(j, min(j + 4, m)):
                acc_w += widths[k]
                cost = dp[i + 1][k + 1] + _mismatch(exp[i], acc_w) + OP_COST['merge']
                if cost < best:
                    best, move = cost, ('merge', k + 1)
            # several tokens sharing one segment (split)
            acc_e = 0.0
            for k in range(i, min(i + 4, n)):
                acc_e += exp[k]
                cost = dp[k + 1][j + 1] + _mismatch(acc_e, widths[j]) + OP_COST['split']
                if cost < best:
                    best, move = cost, ('split', k + 1)
            # unclaimed segment (noise)
            cost = dp[i][j + 1] + 0.15 * widths[j] / max(1, observed)
            if cost < best:
                best, move = cost, ('skip', j + 1)
            dp[i][j] = best
            choice[i][j] = move

    # Walk the choices and materialise boxes.
    boxes: list[Box | None] = []
    i = j = 0
    while i < n and j < m:
        kind, step = choice[i][j]
        if kind == 'pair':
            boxes.append(_xrange_box(segments[j], segments[j]))
            i, j = i + 1, j + 1
        elif kind == 'merge':
            boxes.append(_xrange_box(segments[j], segments[step - 1]))
            i, j = i + 1, step
        elif kind == 'split':
            group = token_texts[i:step]
            boxes.extend(_distribute(group, segments[j], direction=direction))
            i, j = step, j + 1
        else:  # skip segment
            j = step
    while i < n:
        boxes.append(None)
        i += 1
    return boxes, dp[0][0]


def _xrange_box(first: Segment, last: Segment) -> Box:
    # Segments may arrive in logical (RTL) order, i.e. descending x; normalize.
    return (min(first.x0, last.x0), 0, max(first.x1, last.x1), 0)


def _distribute(
    group: list[str],
    segment: Segment,
    *,
    direction: str = 'rtl',
) -> list[Box]:
    """Share one segment's x-range among tokens proportionally to width.

    The first logical token is the rightmost box in an RTL line.
    """
    weights = [max(expected_width(token), 0.2) for token in group]
    total = sum(weights)
    boxes: list[Box] = [None] * len(group)  # type: ignore[list-item]
    if direction == 'rtl':
        cursor = float(segment.x1)
        for index, (token, weight) in enumerate(zip(group, weights)):
            span = segment.width * weight / total if total > 0 else segment.width / len(group)
            boxes[index] = (int(round(cursor - span)), 0, int(round(cursor)), 0)
            cursor -= span
    else:
        cursor = float(segment.x0)
        for index, (token, weight) in enumerate(zip(group, weights)):
            span = segment.width * weight / total if total > 0 else segment.width / len(group)
            boxes[index] = (int(round(cursor)), 0, int(round(cursor + span)), 0)
            cursor += span
    return boxes


def word_boxes_for_line(
    line_text: str,
    line_bbox: Box,
    segments: list[Segment],
) -> tuple[list[WordBox], bool]:
    """Word boxes for one line: aligned if possible, proportional otherwise.

    Returns (words, aligned).  The caller fills in the y extents from the line
    box; x ranges come from the segments or the fallback.
    """
    token_texts = tokens_of(line_text)
    if not token_texts:
        return [], True
    direction = base_direction(line_text)
    logical_segments = list(segments)
    if direction == 'rtl':
        logical_segments = list(reversed(logical_segments))
    x_boxes, _cost = align_tokens(token_texts, logical_segments, direction=direction)

    if any(box is None for box in x_boxes) or not logical_segments:
        # Proportional fallback across the whole line box.
        x_boxes = _distribute(
            token_texts,
            Segment(line_bbox[0], line_bbox[2]),
            direction=direction,
        )
        return (
            [WordBox(token, _to_box(xbox, line_bbox)) for token, xbox in zip(token_texts, x_boxes)],
            False,
        )
    return (
        [WordBox(token, _to_box(xbox, line_bbox)) for token, xbox in zip(token_texts, x_boxes)],
        True,
    )


def _to_box(xbox: Box, line_bbox: Box) -> Box:
    return (xbox[0], line_bbox[1], xbox[2], line_bbox[3])


def run_boxes(word: WordBox, direction: str) -> list[WordBox]:
    """Subdivide one token box into per-script-run boxes (approach B)."""
    runs = script_runs(word.text)
    if len(runs) <= 1:
        return [word]
    weights = [max(expected_width(text), 0.2) for _cls, text in runs]
    span = word.bbox[2] - word.bbox[0]
    total = sum(weights)
    boxes: list[WordBox] = []
    cursor = float(word.bbox[0])
    order = range(len(runs)) if direction == 'ltr' else range(len(runs) - 1, -1, -1)
    placed: dict[int, Box] = {}
    for index in order:
        width = span * weights[index] / total if total > 0 else span / len(runs)
        left, right = cursor, cursor + width
        placed[index] = ((int(round(left)), word.bbox[1], int(round(right)), word.bbox[3]))
        cursor = right
    for index in range(len(runs)):
        cls, text = runs[index]
        boxes.append(WordBox(text, placed[index], classes=(cls,)))
    return boxes


def adjacent_word_pairs(words: list[WordBox]) -> int:
    """Count consecutive token boxes that touch (no whitespace between).

    Extractors turn the gap between rendered boxes into a space, so adjacent
    boxes make words run together in the text layer.  Measured on the token
    boxes, before script runs are split off.
    """
    count = 0
    for first, second in zip(words, words[1:]):
        gap = max(0, max(first.bbox[0], second.bbox[0])
                  - min(first.bbox[2], second.bbox[2]))
        if gap <= 2:
            count += 1
    return count


def refined_word_boxes(
    line_text: str,
    line_bbox: Box,
    page_binary,
) -> tuple[list[WordBox], bool, int]:
    """Token boxes for one line, retrying tighter thresholds when they touch.

    The default threshold splits most lines correctly, but dense vocalized
    print sometimes leaves whole words in one ink island, and the DP then
    hands adjacent slivers to different tokens: the words fuse in the text
    layer.  Since the token count is known, tighter thresholds are tried
    only for lines that come out with touching boxes; among those the
    variant with the fewest touching pairs wins (DP alignment cost breaks
    ties).  Good lines are never re-segmented.
    """
    token_texts = tokens_of(' '.join(line_text.split()))
    if not token_texts:
        return [], True, 0
    segments = line_segments(page_binary, line_bbox)
    words, aligned = word_boxes_for_line(line_text, line_bbox, segments)
    if not aligned:
        return words, aligned, len(segments)
    touching = adjacent_word_pairs(words)
    if touching == 0:
        return words, aligned, len(segments)
    best = (touching, words, len(segments))
    best_cost = float('inf')
    direction = base_direction(line_text)
    logical = list(reversed(segments)) if direction == 'rtl' else segments
    _boxes, best_cost = align_tokens(token_texts, logical, direction=direction)
    for ratio in (0.08, 0.05, 0.03):
        tighter = line_segments(page_binary, line_bbox, min_gap_ratio=ratio)
        if len(tighter) > 4 * len(token_texts):
            break               # speck noise, stop tightening
        candidate_words, candidate_aligned = word_boxes_for_line(
            line_text, line_bbox, tighter
        )
        if not candidate_aligned:
            continue
        candidate_touching = adjacent_word_pairs(candidate_words)
        candidate_logical = list(reversed(tighter)) if direction == 'rtl' else tighter
        _boxes, candidate_cost = align_tokens(
            token_texts, candidate_logical, direction=direction
        )
        if candidate_touching < best[0] or (
            candidate_touching == best[0] and candidate_cost < best_cost
        ):
            best = (candidate_touching, candidate_words, len(tighter))
            best_cost = candidate_cost
            if candidate_touching == 0:
                break
    return best[1], True, best[2]


def build_word_lines(
    page_lines,      # iterable of nuqta Line(text, bbox)
    page_binary,     # numpy uint8, 255 = ink, same size as the page image
    *,
    split_runs: bool = True,
) -> tuple[list[WordLine], dict]:
    """Word-level reading of one page, from line boxes + binarized page."""
    stats = {'lines': 0, 'aligned': 0, 'fallback': 0}
    word_lines: list[WordLine] = []
    for line in page_lines:
        text = ' '.join(line.text.split())
        if not text:
            continue
        stats['lines'] += 1
        words, aligned, segments_count = refined_word_boxes(text, line.bbox, page_binary)
        stats['aligned' if aligned else 'fallback'] += 1
        if split_runs:
            direction = base_direction(text)
            final: list[WordBox] = []
            for word in words:
                final.extend(run_boxes(word, direction))
            words = final
        word_lines.append(
            WordLine(text=text, bbox=line.bbox, words=words, aligned=aligned,
                     segments=segments_count)
        )
    return word_lines, stats


def to_word_hocr(
    result,
    page_images: list,
    *,
    title: str = 'Nuqta OCR',
    lang: str = 'ara',
    dpi: int | None = None,
    split_runs: bool = True,
    stats: dict | None = None,
) -> str:
    """Render a word-level hOCR document.

    ``result`` is a :class:`nuqta.LineResult`; ``page_images`` holds one PIL
    image per page, in the same order, used only for the vertical ink
    segmentation.  Raises ``ValueError`` if page sizes are missing.  When
    ``stats`` is a dict, alignment counters are written into it.
    """
    import numpy as np

    if len(page_images) != len(result.pages):
        raise ValueError('one page image per result page is required')
    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE html>',
        f'<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{escape(lang)}" '
        f'lang="{escape(lang)}" dir="rtl">',
        '<head>',
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />',
        '<meta name="ocr-system" content="nuqta word-level prototype" />',
        '<meta name="ocr-capabilities" content="ocr_page ocr_line ocrx_word" />',
        f'<title>{escape(title)}</title>',
        '</head>',
        '<body>',
    ]
    stats_total = {'lines': 0, 'aligned': 0, 'fallback': 0}
    for page_number, (page, image) in enumerate(zip(result.pages, page_images)):
        width, height = int(page.width), int(page.height)
        if width <= 0 or height <= 0:
            raise ValueError(
                'word hOCR needs the page size; use nuqta.read_lines, whose '
                'pages record their width and height'
            )
        page_dpi = int(dpi) if dpi is not None else int(page.dpi or 0)
        scan_res = f'; scan_res {page_dpi} {page_dpi}' if page_dpi > 0 else ''
        out.append(
            f'<div class="ocr_page" id="page_{page_number + 1}" '
            f'title="bbox 0 0 {width} {height}; ppageno {page_number}{scan_res}" '
            f'lang="{escape(lang)}" dir="rtl">'
        )
        import cv2

        gray = np.asarray(image.convert('L'), dtype=np.uint8)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        word_lines, page_stats = build_word_lines(page.lines, binary, split_runs=split_runs)
        for key in stats_total:
            stats_total[key] += page_stats[key]
        for line_number, wline in enumerate(word_lines, 1):
            x0, y0, x1, y1 = wline.bbox
            line_id = f'line_{page_number + 1}_{line_number}'
            out.append(f'<p class="ocr_par" lang="{escape(lang)}" dir="rtl">')
            out.append(
                f'<span class="ocr_line" id="{line_id}" '
                f'title="bbox {x0} {y0} {x1} {y1}" lang="{escape(lang)}" dir="rtl">'
            )
            for word_number, word in enumerate(wline.words, 1):
                wx0, wy0, wx1, wy1 = word.bbox
                out.append(
                    f'<span class="ocrx_word" id="word_{page_number + 1}_{line_number}_{word_number}" '
                    f'title="bbox {max(0, wx0)} {wy0} {min(width, wx1)} {wy1}">'
                    f'{escape(word.text)}</span>'
                )
            out.append('</span>')
            out.append('</p>')
        out.append('</div>')
    out += ['</body>', '</html>', '']
    if stats is not None:
        stats.update(stats_total)
    return '\n'.join(out)
