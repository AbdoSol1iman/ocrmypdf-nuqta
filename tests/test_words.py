"""Word-level box tests: synthetic images and fake line data, no model."""
from __future__ import annotations

import numpy as np
from xml.etree import ElementTree as ET

import pytest
from nuqta import Line, LinePage, LineResult, validate_hocr
from ocrmypdf_nuqta.words import (
    adjacent_word_pairs,
    align_tokens,
    base_direction,
    build_word_lines,
    char_weight,
    line_segments,
    refined_word_boxes,
    script_runs,
    to_word_hocr,
)
from PIL import Image, ImageDraw

XHTML = '{http://www.w3.org/1999/xhtml}'


def blob_image(specs, width=600, height=200):
    """specs: list of (x, y, width) solid ink islands standing in for words."""
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    for x, y, w in specs:
        draw.rectangle((x, y, x + w, y + 24), fill='black')
    return image


def binary_of(image):
    import cv2

    gray = np.asarray(image.convert('L'), dtype=np.uint8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def test_char_weights_ignore_tashkeel():
    assert char_weight('َ') == 0.0        # fatha
    assert char_weight('ّ') == 0.0        # shadda
    assert char_weight('ب') > 0.0
    assert char_weight('٢') > 0.0
    assert char_weight('x') > 0.0


def test_script_runs_split_digits_and_latin():
    runs = script_runs('كلمة٢٤word')
    assert [cls for cls, _text in runs] == ['arabic', 'arabic_digit', 'latin']
    assert runs[1][1] == '٢٤'


def test_base_direction():
    assert base_direction('الكلمة ٢٤') == 'rtl'
    assert base_direction('2024 PDF') == 'ltr'
    assert base_direction(' ﴿٢٤﴾ ') == 'rtl'


def test_align_tokens_pairs_and_splits():
    segments = [(100, 160), (200, 240), (300, 360)]  # visual left to right
    segments = [type('S', (), {'x0': a, 'x1': b, 'width': b - a})() for a, b in segments]
    # RTL: first token is the rightmost segment
    boxes, cost = align_tokens(['كلمة', '٢٤', 'Python'], list(reversed(segments)), direction='rtl')
    assert all(box is not None for box in boxes)
    assert boxes[0][0] == 300
    assert boxes[0][2] == 360
    assert boxes[1][0] == 200
    assert boxes[1][2] == 240
    assert boxes[2][0] == 100
    assert boxes[2][2] == 160


def test_align_tokens_distributes_when_counts_differ():
    segments = [type('S', (), {'x0': 100, 'x1': 200, 'width': 100})()]
    boxes, _cost = align_tokens(['واحد', 'اثنان'], segments, direction='rtl')
    assert all(box is not None for box in boxes)
    # RTL: first token takes the right part of the segment
    assert boxes[0][2] <= 200
    assert boxes[0][0] > boxes[1][0]


def test_line_segments_splits_at_gaps():
    image = blob_image([(80, 40, 60), (200, 40, 40), (300, 40, 60)])
    segments = line_segments(binary_of(image), (70, 30, 380, 74))
    assert len(segments) == 3
    assert segments[0].x0 < segments[1].x0 < segments[2].x0


def test_blank_line_falls_back_proportionally():
    image = Image.new('RGB', (600, 200), 'white')
    line = Line('كلمة أخرى', (100, 40, 400, 74))
    word_lines, stats = build_word_lines([line], binary_of(image))
    assert stats['fallback'] == 1
    words = word_lines[0].words
    assert [word.text for word in words] == ['كلمة', 'أخرى']
    # RTL: first token is the right box
    assert words[0].bbox[0] > words[1].bbox[0]
    assert word_lines[0].aligned is False


def test_run_splitting_gives_the_digit_its_own_box():
    image = blob_image([(80, 40, 120)])
    line = Line('كلمة٢٤', (70, 30, 210, 74))
    word_lines, _stats = build_word_lines([line], binary_of(image), split_runs=True)
    texts = [word.text for word in word_lines[0].words]
    assert '٢٤' in texts
    assert 'كلمة' in texts
    digit_box = next(word.bbox for word in word_lines[0].words if word.text == '٢٤')
    base_box = next(word.bbox for word in word_lines[0].words if word.text == 'كلمة')
    # RTL: the Arabic base is the right part of the token box
    assert base_box[0] > digit_box[0]


def test_pure_arabic_line_is_one_word_per_token():
    image = blob_image([(80, 40, 60), (200, 40, 60)])
    line = Line('سطر عربي', (70, 30, 280, 74))
    word_lines, stats = build_word_lines([line], binary_of(image), split_runs=False)
    assert stats['aligned'] == 1
    assert [word.text for word in word_lines[0].words] == ['سطر', 'عربي']
    assert word_lines[0].words[0].bbox[0] == 200   # rightmost ink first


def test_single_token_line_covers_the_line():
    image = blob_image([(100, 40, 200)])
    line = Line('كلمة', (90, 30, 310, 74))
    word_lines, _stats = build_word_lines([line], binary_of(image), split_runs=False)
    word = word_lines[0].words[0]
    # full ink island (right edge exclusive)
    assert word.bbox[0] == 100
    assert word.bbox[2] == 301


def test_adjacent_pairs_counted_on_token_boxes():
    words = [
        ('واحد', (100, 0, 150, 10)),
        ('اثنان', (152, 0, 220, 10)),      # touching the first
        ('ثلاثة', (260, 0, 320, 10)),      # separated
    ]
    assert adjacent_word_pairs(
        [type('W', (), {'text': t, 'bbox': b})() for t, b in words]
    ) == 1


def test_refine_splits_dense_line_when_tokens_touch():
    # Two words whose gap (4px) falls below the default threshold
    # (line height 54 -> min gap 6px) but above the tight ones (4px):
    # the refine step must find the split.
    image = blob_image([(100, 40, 60), (164, 40, 60)])
    line = Line('واحد اثنان', (90, 30, 240, 84))
    words, aligned, _segments = refined_word_boxes(line.text, line.bbox, binary_of(image))
    assert aligned
    assert [word.text for word in words] == ['واحد', 'اثنان']
    gap = words[0].bbox[0] - words[1].bbox[2]
    assert gap > 2


def test_word_hocr_is_valid_and_inside_the_page():
    image = blob_image([(80, 40, 60), (200, 40, 40), (300, 40, 60)])
    page = LinePage(
        text='الكلمة ٢٤ Python',
        lines=[Line('الكلمة ٢٤ Python', (70, 30, 380, 74))],
        width=600, height=200, dpi=200,
    )
    result = LineResult(pages=[page], mode='as-written')
    stats: dict = {}
    document = to_word_hocr(result, [image], stats=stats)
    validate_hocr(document)
    root = ET.fromstring(document)
    words = root.findall(f'.//{XHTML}span[@class="ocrx_word"]')
    assert len(words) >= 3
    for word in words:
        x0, y0, x1, y1 = (int(v) for v in word.get('title').split()[1:5])
        assert 0 <= x0 < x1 <= 600
        assert 0 <= y0 < y1 <= 200
    assert stats['lines'] == 1
    assert stats['aligned'] == 1


def test_to_word_hocr_checks_page_count():
    page = LinePage(text='', lines=[], width=10, height=10, dpi=200)
    result = LineResult(pages=[page], mode='as-written')
    with pytest.raises(ValueError, match='one page image'):
        to_word_hocr(result, [])
