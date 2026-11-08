from __future__ import annotations

from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
from PIL import Image, ImageDraw

import ocrmypdf_nuqta as plugin
from nuqta import Page, Result
from ocrmypdf_nuqta import NuqtaOcrEngine, NuqtaOptions

XHTML = '{http://www.w3.org/1999/xhtml}'


def text_image() -> Image.Image:
    image = Image.new('RGB', (600, 800), 'white')
    draw = ImageDraw.Draw(image)
    for y in (100, 200, 300):
        for x in range(80, 520, 20):
            draw.rectangle((x, y, x + 8, y + 30), fill='black')
    return image


class FakeNuqta:
    def __init__(self):
        self.calls = []

    def read_many(self, srcs, mode='as-written', max_new_tokens=1024):
        self.calls.append((len(srcs), mode, max_new_tokens))
        return [Result([Page(f'سطر {index + 1}')], mode) for index in range(len(srcs))]


@pytest.fixture(autouse=True)
def clear_model_cache():
    plugin._MODELS.clear()
    yield
    plugin._MODELS.clear()


def options(**overrides):
    base = {'ocr_engine': 'auto', 'jobs': 0, 'nuqta': NuqtaOptions()}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_engine_selection():
    assert isinstance(plugin.get_ocr_engine(options()), NuqtaOcrEngine)
    assert plugin.get_ocr_engine(options(ocr_engine='tesseract')) is None
    assert plugin.get_ocr_engine(options(ocr_engine='none')) is None


def test_check_options_forces_one_worker():
    opts = options(jobs=4)
    plugin.check_options(opts)
    assert opts.jobs == 1
    auto = options(jobs=None)
    plugin.check_options(auto)
    assert auto.jobs == 1
    tess = options(jobs=4, ocr_engine='tesseract')
    plugin.check_options(tess)
    assert tess.jobs == 4


def test_languages_are_permissive_for_the_default_eng():
    assert {'ara', 'eng'} <= NuqtaOcrEngine.languages(options())


def test_generate_hocr_writes_a_valid_text_layer(tmp_path, monkeypatch):
    image_path = tmp_path / 'page.png'
    text_image().save(image_path, dpi=(300, 300))
    fake = FakeNuqta()
    monkeypatch.setattr(plugin, '_load_ocr', lambda opts: fake)
    hocr_path, text_path = tmp_path / 'page.hocr', tmp_path / 'page.txt'

    NuqtaOcrEngine.generate_hocr(
        input_file=image_path, output_hocr=hocr_path, output_text=text_path, options=options()
    )

    document = hocr_path.read_text(encoding='utf-8')
    root = ET.fromstring(document)
    page = root.find(f'.//{XHTML}div[@class="ocr_page"]')
    assert 'bbox 0 0 600 800' in page.get('title')
    assert 'scan_res 300 300' in page.get('title')
    lines = root.findall(f'.//{XHTML}span[@class="ocr_line"]')
    assert len(lines) == 3
    assert all(line.get('lang') == 'ara' and line.get('dir') == 'rtl' for line in lines)
    assert text_path.read_text(encoding='utf-8') == 'سطر 1\nسطر 2\nسطر 3\n'
    assert fake.calls == [(3, 'as-written', 1024)]


def test_generate_hocr_reuses_one_model(tmp_path, monkeypatch):
    image_path = tmp_path / 'page.png'
    text_image().save(image_path, dpi=(300, 300))
    loads = []
    fake = FakeNuqta()

    def load(opts):
        loads.append(opts)
        return fake

    monkeypatch.setattr(plugin, '_load_ocr', load)
    for number in range(2):
        NuqtaOcrEngine.generate_hocr(
            input_file=image_path,
            output_hocr=tmp_path / f'{number}.hocr',
            output_text=tmp_path / f'{number}.txt',
            options=options(),
        )
    assert len(loads) == 1
    assert len(fake.calls) == 2


def test_sandwich_renderer_is_rejected(tmp_path):
    with pytest.raises(NotImplementedError, match='fpdf2'):
        NuqtaOcrEngine.generate_pdf(
            input_file=tmp_path / 'page.png',
            output_pdf=tmp_path / 'page.pdf',
            output_text=tmp_path / 'page.txt',
            options=options(),
        )
