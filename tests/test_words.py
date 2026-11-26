from __future__ import annotations

import sys
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import pytest
from nuqta import validate_hocr
from ocrmypdf.hocrtransform import HocrParser

from ocrmypdf_nuqta.words import build_hocr, split_line_words

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
