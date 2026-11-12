#!/usr/bin/env python3
"""Draw the synthetic case pages (real shaped Arabic via Noto Naskh + raqm)."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
CASES_DIR = HERE / 'cases'
FONT_PATH = '/usr/share/fonts/google-noto-vf/NotoNaskhArabic[wght].ttf'

# Each case is one line of text on an otherwise empty page, plus the checks
# that pdftotext output must pass for the extraction to count as correct.
# 'order_check' tests the logical token order; the other checks test digit
# direction and run integrity.
CASES = [
    {
        'name': 'digits_isolated',
        'text': 'رقم الصفحة ٢٤',
        'checks': [
            {'kind': 'contains', 'value': '٢٤'},
            {'kind': 'not_contains', 'value': '٤٢'},
        ],
        'order_check': {'kind': 'sequence', 'values': ['رقم', 'الصفحة', '٢٤']},
    },
    {
        'name': 'range_dash',
        'text': '[سورة المطفئين آية: ٢٢ - ٢٦]',
        'checks': [
            {'kind': 'sequence', 'values': ['٢٢', '٢٦']},
            {'kind': 'not_contains', 'value': '٦٢'},
            {'kind': 'not_contains', 'value': '٢٢ - ٢٦'[::-1]},
        ],
        'order_check': {'kind': 'sequence', 'values': ['٢٢', '٢٦']},
    },
    {
        'name': 'list_comma',
        'text': 'راجع الآيات ٧٥، ٧٦',
        'checks': [
            {'kind': 'sequence', 'values': ['٧٥', '٧٦']},
            {'kind': 'not_contains', 'value': '٥٧'},
        ],
        'order_check': {'kind': 'sequence', 'values': ['٧٥', '٧٦']},
    },
    {
        'name': 'latin_inside',
        'text': 'نظام Python للبرمجة',
        'checks': [
            {'kind': 'contains', 'value': 'Python'},
            {'kind': 'not_contains', 'value': 'nohtyP'},
        ],
        'order_check': {'kind': 'sequence', 'values': ['نظام', 'Python', 'للبرمجة']},
    },
    {
        'name': 'mixed_all',
        'text': 'استخدم ٢٤ Word ثم ٥٦ Excel',
        'checks': [
            {'kind': 'contains', 'value': '٢٤'},
            {'kind': 'contains', 'value': '٥٦'},
            {'kind': 'contains', 'value': 'Word'},
            {'kind': 'contains', 'value': 'Excel'},
            {'kind': 'not_contains', 'value': '٤٢'},
            {'kind': 'not_contains', 'value': 'droW'},
        ],
        'order_check': {'kind': 'sequence', 'values': ['استخدم', 'Word', 'Excel']},
    },
    {
        'name': 'verse_marks',
        'text': '﴿٢٤﴾ وقال تعالى',
        'checks': [
            {'kind': 'contains', 'value': '٢٤'},
            {'kind': 'not_contains', 'value': '٤٢'},
            {'kind': 'contains', 'value': '﴿'},
        ],
        'order_check': {'kind': 'sequence', 'values': ['٢٤', 'وقال', 'تعالى']},
    },
    {
        'name': 'footnote_mark',
        'text': 'الحديث(١) هنا',
        'checks': [
            {'kind': 'contains', 'value': '١'},
            {'kind': 'not_contains', 'value': '(١)'[::-1]},
        ],
        'order_check': {'kind': 'sequence', 'values': ['الحديث', '١', 'هنا']},
    },
    {
        'name': 'pure_arabic',
        'text': 'سطر عربي خالص من الحروف المتصلة',
        'checks': [
            {'kind': 'contains', 'value': 'سطر عربي'},
            {'kind': 'contains', 'value': 'الحروف'},
        ],
        'order_check': {'kind': 'sequence',
                        'values': ['سطر', 'عربي', 'خالص', 'الحروف', 'المتصلة']},
    },
]


def draw_case(text: str, path: Path) -> None:
    font = ImageFont.truetype(FONT_PATH, 36, layout_engine=ImageFont.Layout.RAQM)
    width, height = 1100, 140
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    # RTL page: right-aligned line, decent margins
    bbox = draw.textbbox((0, 0), text, font=font, direction='rtl')
    text_width = bbox[2] - bbox[0]
    x = width - 60 - text_width
    if x < 60:
        x = 60
    draw.text((x, 40), text, font=font, fill='black', direction='rtl')
    image.save(path, dpi=(200, 200))


def main() -> None:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case in CASES:
        png = CASES_DIR / f"{case['name']}.png"
        draw_case(case['text'], png)
        manifest.append(case)
    (CASES_DIR / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8'
    )
    print(f'wrote {len(manifest)} case pages to {CASES_DIR}')


if __name__ == '__main__':
    main()
