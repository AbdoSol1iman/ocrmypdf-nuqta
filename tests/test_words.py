from __future__ import annotations

import sys
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import pytest
from nuqta import Line, LinePage, LineResult, validate_hocr
from ocrmypdf.hocrtransform import HocrParser
from PIL import Image, ImageDraw

from ocrmypdf_nuqta.words import (
    adjacent_word_pairs,
    align_tokens,
    base_direction,
    build_hocr,
    build_word_lines,
    char_weight,
    line_segments,
    refined_word_boxes,
    script_runs,
    split_line_words,
    to_word_hocr,
)

XHTML = '{http://www.w3.org/1999/xhtml}'


@dataclass
class _FakeLine:
    text: str
    bbox: tuple[int, int, int, int]


@dataclass
class _FakePage:
    text: str
    lines: list[_FakeLine]
    width: int
    height: int
    dpi: int = 200


@dataclass
class _FakeResult:
    pages: list[_FakePage]
    mode: str = 'as-written'


def test_split_pure_arabic_rtl_order_and_tashkeel_weighting():
    line_box = (80, 100, 920, 140)
    words = split_line_words('مِنْ كِتَابٍ مُفِيدٍ', line_box)
    assert [w for w, _ in words] == ['مِنْ', 'كِتَابٍ', 'مُفِيدٍ']

    # Logical token 0 is rightmost; boxes are disjoint and within line_box.
    boxes = [b for _, b in words]
    assert boxes[0][0] > boxes[1][2]
    assert boxes[1][0] > boxes[2][2]
    for wx0, wy0, wx1, wy1 in boxes:
        assert line_box[0] <= wx0 < wx1 <= line_box[2]
        assert wy0 == line_box[1] and wy1 == line_box[3]

    # Diacritics must not make a 2-letter word wider than a 4-letter word.
    short_heavy_marks = split_line_words('مِنْ الكتاب', line_box)
    w_short = short_heavy_marks[0][1][2] - short_heavy_marks[0][1][0]
    w_long = short_heavy_marks[1][1][2] - short_heavy_marks[1][1][0]
    assert w_long > w_short


def test_split_mixed_arabic_digits_and_latin_visual_slots():
    line_box = (80, 100, 960, 140)

    # Embedded multi-word Latin run keeps left-to-right internal order in its RTL slot.
    mixed_latin = dict(split_line_words('مرحبا Hello World بكم', line_box))
    assert mixed_latin['مرحبا'][0] > mixed_latin['World'][2]
    assert mixed_latin['Hello'][2] < mixed_latin['World'][0]
    assert mixed_latin['Hello'][0] > mixed_latin['بكم'][2]

    # Embedded Arabic-Indic numeric range keeps left-to-right internal order.
    mixed_digits = dict(split_line_words('آية ٢٢ - ٢٦ فأمر', line_box))
    assert mixed_digits['آية'][0] > mixed_digits['٢٦'][2]
    assert mixed_digits['٢٢'][2] < mixed_digits['-'][0]
    assert mixed_digits['-'][2] < mixed_digits['٢٦'][0]
    assert mixed_digits['٢٢'][0] > mixed_digits['فأمر'][2]

    # Pure LTR line stays left-to-right.
    pure_ltr = split_line_words('Python 3.12 test', line_box)
    assert pure_ltr[0][1][2] < pure_ltr[1][1][0] < pure_ltr[1][1][2] < pure_ltr[2][1][0]


def test_split_edge_cases():
    assert split_line_words('   \t ', (0, 0, 100, 20)) == []
    assert split_line_words('كلمة', (10, 20, 110, 50)) == [('كلمة', (10, 20, 110, 50))]

    narrow = split_line_words('أ ب ج', (10, 20, 12, 40))
    assert len(narrow) == 3
    for _, (wx0, wy0, wx1, wy1) in narrow:
        assert 10 <= wx0 < wx1 <= 12
        assert (wy0, wy1) == (20, 40)

    with pytest.raises(ValueError):
        split_line_words('كلمة', (50, 20, 50, 40))


def test_build_hocr_validates_and_parses(tmp_path):
    result = _FakeResult(
        pages=[
            _FakePage(
                text='أمراض القلوب ٢٤\nمرحبا Hello World بكم',
                lines=[
                    _FakeLine('أمراض القلوب ٢٤', (80, 100, 900, 140)),
                    _FakeLine('مرحبا Hello World بكم', (80, 200, 900, 240)),
                ],
                width=1000,
                height=1200,
                dpi=200,
            )
        ]
    )
    document = build_hocr(result, title='sample.png', lang='ara', dpi=300)
    validate_hocr(document)

    root = ET.fromstring(document)
    lines = root.findall(f'.//{XHTML}span[@class="ocr_line"]')
    assert len(lines) == 2
    assert [w.text for w in lines[0].findall(f'.//{XHTML}span[@class="ocrx_word"]')] == [
        'أمراض',
        'القلوب',
        '٢٤',
    ]
    assert [w.text for w in lines[1].findall(f'.//{XHTML}span[@class="ocrx_word"]')] == [
        'مرحبا',
        'Hello',
        'World',
        'بكم',
    ]

    hocr_path = tmp_path / 'sample.hocr'
    hocr_path.write_text(document, encoding='utf-8')
    parsed = HocrParser(hocr_path).parse()
    assert parsed.dpi == 300.0
    assert len(parsed.paragraphs) == 2


def test_words_module_does_not_require_numpy_or_cv2(monkeypatch):
    monkeypatch.setitem(sys.modules, 'numpy', None)
    monkeypatch.setitem(sys.modules, 'cv2', None)
    words = split_line_words('كتاب العلم المفيد', (40, 80, 760, 120))
    assert [w for w, _ in words] == ['كتاب', 'العلم', 'المفيد']


def test_multiword_ligature_expansion_and_discovery():
    import unicodedata

    multiword_chars = [
        chr(cp)
        for cp in range(0x110000)
        if any(c.isspace() for c in unicodedata.normalize('NFKC', chr(cp)).strip())
    ]
    assert '\ufdfa' in multiword_chars  # ﷺ
    assert '\ufdfb' in multiword_chars  # ﷻ

    line_box = (80, 100, 920, 140)
    words = split_line_words('قال النبي ﷺ فإنه', line_box)
    assert [w for w, _ in words] == ['قال', 'النبي', 'صلى', 'الله', 'عليه', 'وسلم', 'فإنه']

    boxes = [b for _, b in words]
    for i in range(len(boxes) - 1):
        assert boxes[i][0] > boxes[i + 1][2]
    for wx0, wy0, wx1, wy1 in boxes:
        assert line_box[0] <= wx0 < wx1 <= line_box[2]
        assert wy0 == line_box[1] and wy1 == line_box[3]


def test_non_multiword_lines_unchanged():
    line_box = (80, 100, 920, 140)
    assert split_line_words('بِسْمِ ﷲ الرَّحْمَٰنِ ﻻ إِلَٰهَ', line_box) == [
        ('بِسْمِ', (795, 100, 920, 140)),
        ('ﷲ', (609, 100, 776, 140)),
        ('الرَّحْمَٰنِ', (341, 100, 591, 140)),
        ('ﻻ', (238, 100, 322, 140)),
        ('إِلَٰهَ', (80, 100, 205, 140)),
    ]


def test_multiword_ligature_mixed_with_digits_and_latin():
    line_box = (80, 100, 960, 140)
    words = split_line_words('النبي ﷺ عام 2024 test ٢٤ خير', line_box)
    assert [w for w, _ in words] == [
        'النبي',
        'صلى',
        'الله',
        'عليه',
        'وسلم',
        'عام',
        '2024',
        'test',
        '٢٤',
        'خير',
    ]
    by_word = dict(words)
    assert by_word['النبي'][0] > by_word['صلى'][2]
    assert by_word['وسلم'][0] > by_word['عام'][2]
    assert by_word['عام'][0] > by_word['٢٤'][2]
    assert by_word['2024'][2] < by_word['test'][0]
    assert by_word['test'][2] < by_word['٢٤'][0]
    assert by_word['2024'][0] > by_word['خير'][2]


# --- Ink-projection / DP word alignment tests (from main) ---



def blob_image(specs, width=600, height=200):
    """specs: list of (x, y, width) solid ink islands standing in for words."""
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    for x, y, w in specs:
        draw.rectangle((x, y, x + w, y + 24), fill='black')
    return image


def binary_of(image):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

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
    boxes, _cost = align_tokens(['كلمة', '٢٤', 'Python'], list(reversed(segments)), direction='rtl')
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
